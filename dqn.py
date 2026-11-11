"""Deep Q-Network (replay buffer + target network) for CartPole-v1 and ALE/MsPacman-v5.

    python dqn.py --env CartPole-v1 --seed 0 --episodes 1000 --out results/cartpole_s0
    python dqn.py --env ALE/MsPacman-v5 --seed 0 --episodes 5000 --hours 14 --device cuda --out results/mspacman
    python dqn.py --out results/<run> --eval-only

Training appends one row per finished episode to train.csv and saves model.pt (final and best-snapshot weights) at
every snapshot check. Afterwards 500 greedy evaluation episodes are played and eval.csv, metrics.json,
q_values.png, rewards.png and eval_hist.png are written to --out. --eval-only rebuilds all of these from model.pt
and train.csv, so an interrupted run still yields every output.
"""
import argparse
import copy
import csv
import json
import os
import time
from collections import deque

import ale_py
import gymnasium as gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

gym.register_envs(ale_py)
torch.set_num_threads(1)  # small networks: extra CPU threads only add overhead and contention between runs

# Per-game settings; train(**overrides) replaces any of them. Steps are environment (agent) steps.
ENVS = {
    "CartPole-v1": dict(gamma=0.95, pixels=False, env_kwargs={}, lr=1e-3, batch=64, buffer=50_000, warmup=1_000,
                        train_every=1, target_sync=500, eps_start=1.0, eps_end=0.05, eps_decay=10_000,
                        select_every=20, select_episodes=10),
    "ALE/MsPacman-v5": dict(gamma=0.99, pixels=True, env_kwargs={}, lr=1e-4, batch=32, buffer=100_000,
                            warmup=20_000, train_every=4, target_sync=10_000, eps_start=1.0, eps_end=0.1,
                            eps_decay=500_000, select_every=100, select_episodes=10),
}
# Reset seeds: training episode k of seed s uses s * 10**6 + k, which stays below both ranges.
EVAL_SEED, SELECT_SEED = 10**9, 2 * 10**9
COLUMNS = ["episode", "score", "steps", "epsilon", "q_mean", "q_max", "q0", "loss", "seconds"]

mspacman_color = 210 + 164 + 74


def preprocess_observation(obs):
    # Copied from the assignment. `sum` promotes to uint64, so `img // 3 - 128` wraps around, but the int8 cast keeps
    # the low 8 bits and the result is exact; `sum` also returns a new array, so `obs` is never modified.
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80, 1)


def q_net(cfg, obs_shape, n_actions):
    if not cfg["pixels"]:
        return nn.Sequential(nn.Linear(obs_shape[0], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                             nn.Linear(128, n_actions))
    return nn.Sequential(  # 4x88x80 -> 32x21x19 -> 64x9x8 -> 64x7x6 = 2688
        nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        nn.Flatten(), nn.Linear(2688, 512), nn.ReLU(), nn.Linear(512, n_actions))


def tensor(x, device):
    """numpy -> float tensor on `device`. int8 frames travel as int8 (4x less traffic) and are scaled to [-1, 1)."""
    t = torch.as_tensor(x).to(device)
    return t.float() / 128 if t.dtype == torch.int8 else t.float()


class Buffer:
    """Uniform replay memory in preallocated arrays; when full, the oldest transition is overwritten."""

    def __init__(self, capacity, shape, dtype):
        self.s, self.s2 = np.zeros((capacity, *shape), dtype), np.zeros((capacity, *shape), dtype)
        self.a, self.r, self.d = np.zeros(capacity, np.int64), np.zeros(capacity, np.float32), np.zeros(capacity, np.float32)
        self.capacity, self.size, self.i = capacity, 0, 0

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n, rng, device):
        idx = rng.integers(self.size, size=n)  # filled slots only
        return (tensor(self.s[idx], device), torch.as_tensor(self.a[idx], device=device), tensor(self.r[idx], device),
                tensor(self.s2[idx], device), tensor(self.d[idx], device))


def epsilon(step, cfg):
    return max(cfg["eps_end"], cfg["eps_start"] - (cfg["eps_start"] - cfg["eps_end"]) * step / cfg["eps_decay"])


def td_targets(target, r, s2, d, gamma):
    """y = r + gamma * (1 - d) * max_a' Q_target(s', a'). d = 1 only for a real termination, never a truncation."""
    with torch.no_grad():
        return r + gamma * (1 - d) * target(s2).max(1).values


def learn(net, target, opt, batch, gamma):
    """One Huber-loss gradient step on a batch (s, a, r, s', d). A non-finite loss is returned without a step."""
    s, a, r, s2, d = batch
    q = net(s).gather(1, a[:, None]).squeeze(1)
    loss = nn.functional.smooth_l1_loss(q, td_targets(target, r, s2, d, gamma))
    value = loss.item()
    if np.isfinite(value):
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10)
        opt.step()
    return value


def run_episode(env, net, cfg, seed, rng, eps=0.0, agent=None, on_step=None):
    """Play one episode with epsilon-greedy actions from `net` and return its statistics.

    With `agent` (dict: target, opt, buffer, step) it is a training episode: epsilon follows the schedule, every
    transition is stored with its reward clipped to its sign, the network learns and the target network is synced.
    `on_step(obs, reward, info)` sees every raw frame, raw reward and info dict (used by validation).
    """
    device = next(net.parameters()).device
    obs, _ = env.reset(seed=seed)
    if cfg["pixels"]:
        frames = deque([preprocess_observation(obs)[..., 0]] * 4, maxlen=4)  # no frame of the previous episode
        state = np.stack(frames)
    else:
        state = obs
    score, g0, q0, qmax, losses, t, done = 0.0, 0.0, 0.0, [], [], 0, False
    while not done:
        with torch.no_grad():
            q = net(tensor(state[None], device))[0].cpu().numpy()
        if not np.isfinite(q).all():
            raise FloatingPointError(f"non-finite Q-value at step {t} of the episode")
        if agent:
            eps = epsilon(agent["step"], cfg)
        a = int(rng.integers(len(q))) if rng.random() < eps else int(q.argmax())
        if t == 0:
            q0 = float(q[a])
        obs, r, terminated, truncated, info = env.step(a)
        if on_step:
            on_step(obs, r, info)
        if cfg["pixels"]:
            frames.append(preprocess_observation(obs)[..., 0])
            next_state = np.stack(frames)
        else:
            next_state = obs
        score += r
        g0 += cfg["gamma"] ** t * np.sign(r)
        qmax.append(float(q.max()))
        if agent:
            buf = agent["buffer"]
            buf.add(state, a, np.sign(r), next_state, float(terminated))  # truncation keeps d = 0: bootstrap
            agent["step"] += 1
            if buf.size >= cfg["warmup"] and agent["step"] % cfg["train_every"] == 0:
                losses.append(learn(net, agent["target"], agent["opt"], buf.sample(cfg["batch"], rng, device),
                                    cfg["gamma"]))
                if not np.isfinite(losses[-1]):
                    raise FloatingPointError(f"non-finite loss {losses[-1]} at step {t} of the episode")
            if agent["step"] % cfg["target_sync"] == 0:
                agent["target"].load_state_dict(net.state_dict())
        state = next_state
        t += 1
        done = terminated or truncated
    return dict(score=float(score), steps=t, q_mean=float(np.mean(qmax)), q_max=max(qmax), q0=q0, g0=float(g0),
                loss=float(np.mean(losses)) if losses else float("nan"))


def make_agent(cfg, net, shape):
    """Training state for run_episode: a target copy of `net` that the optimizer never sees, Adam, an empty buffer."""
    return dict(target=copy.deepcopy(net).requires_grad_(False), opt=torch.optim.Adam(net.parameters(), lr=cfg["lr"]),
                step=0, buffer=Buffer(cfg["buffer"], shape, np.int8 if cfg["pixels"] else np.float32))


def train(env_id, seed, out, episodes, hours=None, device="cpu", **overrides):
    """Train for `episodes` episodes or until `hours` have passed (the running episode is finished), then report."""
    cfg = dict(copy.deepcopy(ENVS[env_id]), env=env_id)
    cfg.update(overrides)
    assert cfg["warmup"] >= cfg["batch"], "the buffer must hold a batch before the first update"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(env_id, **cfg["env_kwargs"])
    shape = (4, 88, 80) if cfg["pixels"] else env.observation_space.shape
    net = q_net(cfg, shape, env.action_space.n).to(device)
    agent = make_agent(cfg, net, shape)
    meta = dict(cfg, seed=seed, episodes=episodes, hours=hours, device=device,
                device_name=torch.cuda.get_device_name(device) if device.startswith("cuda") else "cpu",
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    best, best_score, best_episode = None, -np.inf, 0
    start = time.time()
    with open(f"{out}/train.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(COLUMNS)
        for k in range(1, episodes + 1):
            try:
                st = run_episode(env, net, cfg, seed * 10**6 + k, rng, agent=agent)
            except FloatingPointError as e:
                raise RuntimeError(f"training diverged at episode {k}, global step {agent['step']}: {e}") from None
            log.writerow([k, st["score"], st["steps"], epsilon(agent["step"], cfg), st["q_mean"], st["q_max"], st["q0"],
                          st["loss"], round(time.time() - start, 1)])
            f.flush()
            last = k == episodes or (hours is not None and time.time() - start > hours * 3600)
            if k % cfg["select_every"] == 0 or last:
                score = np.mean([run_episode(env, net, cfg, SELECT_SEED + i, rng)["score"]
                                 for i in range(cfg["select_episodes"])])
                if score >= best_score:  # ties go to the later snapshot, which has trained on more data
                    best, best_score, best_episode = copy.deepcopy(net.state_dict()), score, k
                meta.update(episodes_done=k, stop="episodes" if k == episodes else "time" if last else None,
                            best_score=float(best_score), best_episode=best_episode, global_steps=agent["step"])
                torch.save(dict(final=net.state_dict(), best=best, meta=meta), f"{out}/model.pt.tmp")
                os.replace(f"{out}/model.pt.tmp", f"{out}/model.pt")  # never leaves a half-written model.pt
            if last:
                break
    report(out)


def moving_average(x, w=100):
    # np.convolve(..., "valid") swaps its arguments when x is shorter than w, so short inputs are handled here.
    return np.convolve(x, np.ones(w) / w, "valid") if len(x) >= w else np.array([])


def plot_hist(scores, path, title):
    scores = np.asarray(scores, float)
    bins = [scores[0] - 0.5, scores[0] + 0.5] if np.all(scores == scores[0]) else 30  # one visible bar if all equal
    plt.figure(figsize=(7, 4))
    plt.hist(scores, bins=bins, edgecolor="black")
    std = scores.std(ddof=1) if len(scores) > 1 else 0.0
    plt.title(f"{title}\nmean {scores.mean():.1f}, std {std:.1f}, n = {len(scores)}")
    plt.xlabel("episode reward")
    plt.ylabel("number of episodes")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def stats(x):
    x = np.asarray(x, float)
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return dict(mean=float(x.mean()), std=std, se=std / np.sqrt(len(x)), min=float(x.min()), max=float(x.max()),
                n=len(x))


def report(out, n=500):
    """Evaluate the best snapshot (and the final weights if they differ) greedily on n episodes; write plots."""
    ckpt = torch.load(f"{out}/model.pt", map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    device = meta["device"] if torch.cuda.is_available() else "cpu"
    rows = list(csv.DictReader(open(f"{out}/train.csv")))
    log = {c: np.array([float(r[c]) for r in rows]) for c in COLUMNS}
    env = gym.make(meta["env"], **meta["env_kwargs"])
    shape = (4, 88, 80) if meta["pixels"] else env.observation_space.shape
    net = q_net(meta, shape, env.action_space.n).to(device)
    name = meta["env"].split("/")[-1]
    metrics = dict(meta=meta, train=dict(episodes=len(rows), first100=stats(log["score"][:100]),
                                         last100=stats(log["score"][-100:])))
    same = all(torch.equal(ckpt["best"][key], ckpt["final"][key]) for key in ckpt["best"])
    for which in ["best"] if same else ["best", "final"]:
        net.load_state_dict(ckpt[which])
        rng = np.random.default_rng(0)  # greedy policy: only the unused epsilon draws come from here
        ev = [run_episode(env, net, meta, EVAL_SEED + i, rng) for i in range(n)]
        suffix = "" if which == "best" else "_final"
        with open(f"{out}/eval{suffix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "score", "steps", "q0", "g0"])
            w.writerows([EVAL_SEED + i, e["score"], e["steps"], e["q0"], e["g0"]] for i, e in enumerate(ev))
        scores = [e["score"] for e in ev]
        metrics[f"eval_{which}"] = dict(stats(scores), q0=stats([e["q0"] for e in ev]), g0=stats([e["g0"] for e in ev]))
        if env.spec.max_episode_steps:
            metrics[f"eval_{which}"]["share_at_time_limit"] = float(np.mean([e["steps"] == env.spec.max_episode_steps
                                                                             for e in ev]))
        label = "best snapshot" if which == "best" else "final weights"
        plot_hist(scores, f"{out}/eval_hist{suffix}.png", f"{name}: rewards of {n} greedy episodes ({label}, "
                                                          f"episode {meta['best_episode'] if which == 'best' else len(rows)})")
    metrics["same_best_final"] = same
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)

    ep = log["episode"]
    plt.figure(figsize=(8, 4))
    plt.plot(ep, log["q_mean"], lw=0.8, label="episode mean of max_a Q(s, a)")
    plt.plot(ep, log["q_max"], lw=0.8, alpha=0.7, label="episode max of max_a Q(s, a)")
    plt.title(f"{name}: maximum Q-value per training episode")
    plt.xlabel("training episode")
    plt.ylabel("Q-value" + (" (clipped-reward units)" if meta["pixels"] else ""))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out}/q_values.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(ep, log["score"], lw=0.6, alpha=0.6, label="episode reward")
    ma = moving_average(log["score"])
    if len(ma):
        plt.plot(ep[99:], ma, lw=2, label="mean of the last 100 episodes")
    plt.title(f"{name}: training episode reward")
    plt.xlabel("training episode")
    plt.ylabel("episode reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out}/rewards.png", dpi=120)
    plt.close()
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=list(ENVS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int)
    p.add_argument("--hours", type=float, help="wall-clock training limit; the running episode is finished")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    p.add_argument("--eval-only", action="store_true", help="only evaluate and plot from <out>/model.pt and train.csv")
    args = p.parse_args()
    if args.eval_only:
        report(args.out)
    else:
        if args.env is None or args.episodes is None:
            p.error("--env and --episodes are required for training")
        train(args.env, args.seed, args.out, args.episodes, args.hours, args.device)


if __name__ == "__main__":
    main()
