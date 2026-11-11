"""Validation experiments for dqn.py. Every check prints PASS/FAIL with its observed value and the expectation that
was written down before it first ran (README, "Validation"). Results go to results/validation_<group>.json and the
exit code is 1 if any check fails.

    python validate_dqn.py checks      # edge cases and algorithm checks (about 25 s, CPU)
    python validate_dqn.py cartpole    # CartPole experiments; trains its control runs once into results/val_*
    python validate_dqn.py mspacman    # MsPacman experiments on results/mspacman (one GPU)
"""
import copy
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dqn import (ENVS, EVAL_SEED, SELECT_SEED, Buffer, epsilon, learn, make_agent, moving_average, plot_hist,
                 preprocess_observation, q_net, run_episode, stats, td_targets, tensor, train)

VAL_SEED = 3 * 10**9  # reset seeds of validation-only episodes, disjoint from training, selection and evaluation
OUTPUTS = ["train.csv", "model.pt", "eval.csv", "metrics.json", "q_values.png", "rewards.png", "eval_hist.png"]
ROWS = []


def check(name, what, expected, observed, ok):
    ROWS.append(dict(name=name, what=what, expected=expected, observed=observed, ok=bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {observed}   [expected: {expected}]")


def setup(env_id, seed=0, **overrides):
    """cfg, env, CPU network and agent for one validation experiment (the same pieces train() uses)."""
    cfg = dict(ENVS[env_id], env=env_id, **overrides)
    env = gym.make(env_id, **cfg["env_kwargs"])
    shape = (4, 88, 80) if cfg["pixels"] else env.observation_space.shape
    torch.manual_seed(seed)
    net = q_net(cfg, shape, env.action_space.n)
    return cfg, env, net, make_agent(cfg, net, shape)


def play(env, net, cfg, agent, seed, rng):
    """One agent-mode episode that also records each raw frame (and a copy taken before preprocessing),
    raw reward and lives. Returns (stats, first buffer index of the episode, records)."""
    rec, start = [], agent["buffer"].size
    st = run_episode(env, net, cfg, seed, rng, agent=agent,
                     on_step=lambda o, r, i: rec.append((o, o.copy(), float(r), i.get("lives"))))
    return st, start, rec


def trace(env_id, seed, **overrides):
    """First 10 steps of one episode: what the agent saw, valued, did, received, stored and would learn towards."""
    cfg, env, net, agent = setup(env_id, warmup=10**9, buffer=30_000, **overrides)
    st, _, rec = play(env, net, cfg, agent, seed, np.random.default_rng(0))
    b = agent["buffer"]
    s, s2 = tensor(b.s[:10], "cpu"), tensor(b.s2[:10], "cpu")  # the same input scaling as training
    with torch.no_grad():
        q = net(s).numpy()
        y = td_targets(agent["target"], torch.as_tensor(b.r[:10]), s2, torch.as_tensor(b.d[:10]), cfg["gamma"]).numpy()
    ids = {}  # each distinct preprocessed frame gets a number, so the stack shift is visible
    lines = [f"{env_id} seed {seed} {overrides}: episode length {st['steps']}, score {st['score']}"]
    for t in range(10):
        state = ("frames " + str([ids.setdefault(f.tobytes(), len(ids)) for f in b.s[t]]) if cfg["pixels"]
                 else "s " + np.array2string(b.s[t], precision=3))
        lives = f" lives {rec[t][3]}" if cfg["pixels"] else ""
        lines.append(f"t={t} {state} Q {np.array2string(q[t], precision=3)} eps {epsilon(t, cfg):.4f} "
                     f"a {b.a[t]} r_raw {rec[t][2]:g} r {b.r[t]:g} d {b.d[t]:g}{lives} y {y[t]:.4f}")
    return lines


def group_checks():
    # --- preprocessing, frame stacks, lives and rewards on two random MsPacman episodes (no learning)
    cfg, env, net, agent = setup("ALE/MsPacman-v5", warmup=10**9, buffer=30_000, eps_start=1.0, eps_end=1.0)
    rng = np.random.default_rng(0)
    st1, i1, rec1 = play(env, net, cfg, agent, VAL_SEED, rng)
    _, i2, _ = play(env, net, cfg, agent, VAL_SEED + 1, rng)
    exact = unchanged = foreign = visible = compact = 0
    for obs, before, _, _ in rec1:
        p = preprocess_observation(obs)
        crop = obs[1:176:2, ::2].astype(np.int64)
        ref = crop.sum(2)
        ref[ref == 448] = 0
        exact += np.array_equal(p[..., 0], ref // 3 - 128) and p.shape == (88, 80, 1) and p.dtype == np.int8
        unchanged += np.array_equal(obs, before)
        mine = np.all(crop == (210, 164, 74), axis=2)
        foreign += int(((crop.sum(2) == 448) & ~mine).sum())
        if mine.any():
            visible += 1
            r, c = np.nonzero(mine)
            compact += np.ptp(r) < 12 and np.ptp(c) < 12
    n = len(rec1)
    check("preprocess_exact", "assignment preprocessing equals the same formula in signed int64 (shape 88x80x1, int8)",
          f"{n}/{n} frames", f"{exact}/{n} frames", exact == n)
    check("preprocess_raw_unchanged", "the raw frame is not modified by preprocessing", f"{n}/{n} frames",
          f"{unchanged}/{n} frames", unchanged == n)
    check("mspacman_color_unique", "pixels with RGB sum 448 that are not Ms. Pac-Man's (210,164,74)", "0",
          str(foreign), foreign == 0)
    check("mspacman_visible", "share of frames in which Ms. Pac-Man is visible", ">= 0.90", f"{visible / n:.3f}",
          visible / n >= 0.90)
    check("mspacman_compact", "share of visible frames whose (210,164,74) pixels fit in 12x12 (one sprite)",
          ">= 0.99", f"{compact / max(visible, 1):.3f}", compact / max(visible, 1) >= 0.99)

    b = agent["buffer"]
    first = preprocess_observation(env.reset(seed=VAL_SEED + 1)[0])[..., 0]
    ok = all(np.array_equal(f, first) for f in b.s[i2]) and not np.array_equal(b.s[i2][0], b.s2[i2 - 1][3])
    check("frame_stack_reset", "first state of episode 2 = 4 copies of its own reset frame, none from episode 1",
          "True", str(ok), ok)

    lives = [l for *_, l in rec1]
    drops = [t for t in range(1, n) if lives[t] < lives[t - 1]]
    d = b.d[i1:i2]
    ok = (sorted(set(lives), reverse=True) == [3, 2, 1, 0] and np.flatnonzero(d).tolist() == [n - 1]
          and all(d[t] == 0 for t in drops if t != n - 1))
    check("life_loss_not_terminal", "lives fall 3->0 in one episode; only the game-over transition has d=1",
          f"lives [3,2,1,0], d=1 only at step {n - 1}",
          f"lives {sorted(set(lives), reverse=True)}, drops at {drops}, d=1 at {np.flatnonzero(d).tolist()}", ok)
    raw = np.array([r for _, _, r, _ in rec1])
    ok = np.array_equal(b.r[i1:i2], np.sign(raw)) and set(b.r[i1:i2].tolist()) <= {0.0, 1.0} and st1["score"] == raw.sum()
    check("reward_clip_vs_score", "stored reward = sign(raw reward); score = sum of raw rewards",
          "True", f"raw rewards {sorted(set(raw.tolist()))}, stored {sorted(set(b.r[i1:i2].tolist()))}, "
                  f"score {st1['score']} vs raw sum {raw.sum()}", ok)

    # --- replay buffer and epsilon schedule
    b = Buffer(5, (1,), np.float32)
    for i in range(1, 4):
        b.add([i], i, i, [i], 0)
    seen = set(b.sample(1000, np.random.default_rng(0), "cpu")[0][:, 0].tolist())
    check("buffer_partial", "capacity 5 holding 3: 1000 samples come only from filled slots", "{1, 2, 3}",
          str(sorted(seen)), seen == {1, 2, 3})
    for i in range(4, 9):
        b.add([i], i, i, [i], 0)
    held = sorted(b.s[:, 0].tolist())
    check("buffer_wrap", "capacity 5 after 8 adds holds the 5 newest", "[4, 5, 6, 7, 8], size 5",
          f"{held}, size {b.size}", held == [4, 5, 6, 7, 8] and b.size == 5)
    for env_id, c in ENVS.items():
        got = [epsilon(s, c) for s in (0, c["eps_decay"] // 2, c["eps_decay"], 10 * c["eps_decay"])]
        want = [c["eps_start"], (c["eps_start"] + c["eps_end"]) / 2, c["eps_end"], c["eps_end"]]
        check(f"epsilon_bounds[{env_id}]", "epsilon at steps 0, decay/2, decay, 10*decay", str(want),
              str([round(g, 12) for g in got]), np.allclose(got, want, atol=1e-12, rtol=0))

    # --- target network: frozen within a sync window, equal to the online network right after a sync
    cfg, env, net, agent = setup("CartPole-v1", target_sync=100, warmup=32, batch=32)
    vec = nn.utils.parameters_to_vector
    snaps = []
    rng = np.random.default_rng(0)
    while agent["step"] < 300:
        run_episode(env, net, cfg, VAL_SEED + len(snaps), rng, agent=agent,
                    on_step=lambda o, r, i: snaps.append((agent["step"], vec(agent["target"].parameters()).clone(),
                                                          vec(net.parameters()).clone())))
    # on_step runs before the step's update and sync, so records with equal step // 100 share one sync window
    frozen = all(torch.equal(t, t2) for (s, t, _), (s2, t2, _) in zip(snaps, snaps[1:]) if s // 100 == s2 // 100)
    synced = [s for s, t, o in snaps if s % 100 == 0 and s > 0]
    equal_after_sync = all(torch.equal(t, o) for s, t, o in snaps if s % 100 == 0 and s > 0)
    differ = all(not torch.equal(t, o) for s, t, o in snaps if s % 100 != 0 and s > cfg["warmup"])
    check("target_sync", "target constant within each 100-step window, = online right after a sync, != online otherwise",
          "True, True, True", f"{frozen}, {equal_after_sync} (sync steps {synced}), {differ}",
          frozen and equal_after_sync and differ and synced == list(range(100, snaps[-1][0] + 1, 100)) and len(synced) >= 2)

    # --- truncation bootstraps, termination does not (CartPole limits: |x| <= 2.4, |theta| <= 0.2095)
    inside = lambda o: abs(o[0]) <= 2.4 and abs(o[2]) <= 0.2095  # noqa: E731
    cfg, env, net, agent = setup("CartPole-v1", warmup=10**9, eps_start=1.0, eps_end=1.0,
                                 env_kwargs={"max_episode_steps": 20})
    rng = np.random.default_rng(0)
    for i in range(50):
        st, _, rec = play(env, net, cfg, agent, VAL_SEED + i, rng)
        if st["steps"] == 20 and inside(rec[-1][0]):
            break
    last_d = agent["buffer"].d[agent["buffer"].size - 1]
    check("truncation_bootstrap", "episode cut by a 20-step limit with the pole still up: last stored d",
          "0 (bootstrap)", f"{last_d:g} (episode seed offset {i}, {st['steps']} steps)", st["steps"] == 20 and last_d == 0)
    cfg, env, net, agent = setup("CartPole-v1", warmup=10**9, eps_start=1.0, eps_end=1.0)
    st, start, rec = play(env, net, cfg, agent, VAL_SEED, np.random.default_rng(0))
    d = agent["buffer"].d[start:agent["buffer"].size]
    ok = not inside(rec[-1][0]) and np.flatnonzero(d).tolist() == [len(d) - 1]
    check("termination_terminal", "pole falls (last state outside the limits): only the last transition has d=1",
          f"d=1 only at step {len(d) - 1}", f"d=1 at {np.flatnonzero(d).tolist()}, last state {np.round(rec[-1][0], 3)}", ok)

    # --- TD update on a 5-state chain with a closed-form Q*: right moves toward the goal (state 4, reward 1, terminal)
    gamma, eye = 0.9, torch.eye(5)
    tr = [(s, 1, float(s == 3), s + 1, float(s == 3)) for s in range(4)] + [(s, 0, 0.0, max(s - 1, 0), 0.0) for s in range(4)]
    s, a, r, s2, d = (torch.as_tensor(x) for x in zip(*tr))
    batch = (eye[s], a, r.float(), eye[s2], d.float())
    net = nn.Linear(5, 2, bias=False)
    nn.init.zeros_(net.weight)
    target = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.SGD(net.parameters(), lr=1.0)
    for i in range(5000):
        learn(net, target, opt, batch, gamma)
        if (i + 1) % 50 == 0:
            target.load_state_dict(net.state_dict())
    q = net(eye[:4]).detach().numpy()
    v = gamma ** (3 - np.arange(4))
    q_star = np.stack([gamma * v[np.maximum(np.arange(4) - 1, 0)], v], 1)
    err = np.abs(q - q_star).max()
    check("chain_q_star", "learn() on all 8 transitions converges to the closed-form Q* (gamma 0.9)",
          f"max error <= 1e-3, Q*(s,right) {v.round(4).tolist()}", f"max error {err:.2e}, Q(s,right) "
          f"{q[:, 1].round(4).tolist()}, Q(s,left) {q[:, 0].round(4).tolist()}", err <= 1e-3)
    check("chain_greedy", "greedy action in every state", "right (1) in all 4", str(q.argmax(1).tolist()),
          (q.argmax(1) == 1).all())
    check("chain_terminal", "Q(3, right): the goal transition is terminal, so no bootstrapped term", "1.0 +- 1e-3",
          f"{q[3, 1]:.6f}", abs(q[3, 1] - 1) <= 1e-3)
    tnet = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        tnet.weight.copy_(torch.tensor([[1.0, -1.0], [5.0, 0.5], [2.0, 0.25]]))  # Q(s'1) = [1,5,2], Q(s'2) = [-1,.5,.25]
    y = td_targets(tnet, torch.tensor([1.0, 2.0]), torch.eye(2), torch.tensor([0.0, 1.0]), 0.9)
    check("td_targets_hand", "targets for a non-terminal and a terminal transition", "[5.5, 2.0]",
          str(y.numpy().round(6).tolist()), torch.allclose(y, torch.tensor([5.5, 2.0]), atol=1e-6))

    # --- first-10-step traces
    cart = trace("CartPole-v1", VAL_SEED)
    pac = [trace("ALE/MsPacman-v5", VAL_SEED, env_kwargs={"repeat_action_probability": 0.0}) for _ in range(2)]
    with open("results/trace_cartpole.txt", "w") as f:
        f.write("\n".join(cart) + "\n")
    with open("results/trace_mspacman.txt", "w") as f:
        f.write("\n".join(pac[0]) + "\n")
    print("\n".join(cart + pac[0]))
    check("trace_mspacman_deterministic", "two MsPacman traces without sticky actions, same seed", "identical",
          "identical" if pac[0] == pac[1] else "different", pac[0] == pac[1])

    # --- whole-run behaviour (temporary run folders, CPU)
    tmp = tempfile.mkdtemp()
    try:
        train("CartPole-v1", 0, f"{tmp}/div", 200, lr=1e30)  # learning starts near episode 45
        msg = "no error"
    except RuntimeError as e:
        msg = str(e)
    rows = list(csv.DictReader(open(f"{tmp}/div/train.csv")))
    saved = torch.load(f"{tmp}/div/model.pt", weights_only=False)["meta"]["episodes_done"]
    ok = "episode" in msg and "step" in msg and [int(r["episode"]) for r in rows] == list(range(1, len(rows) + 1))
    check("divergence_guard", "lr = 1e30 must stop training with an error naming episode and step; saved files load",
          "RuntimeError at the first learning episode, model.pt and train.csv readable",
          f"{msg!r}; train.csv episodes 1..{len(rows)}, model.pt from episode {saved}", ok)

    train("CartPole-v1", 0, f"{tmp}/time", 10**6, hours=2 / 3600)
    meta = torch.load(f"{tmp}/time/model.pt", weights_only=False)["meta"]
    rows = list(csv.DictReader(open(f"{tmp}/time/train.csv")))
    secs = [float(r["seconds"]) for r in rows]
    files = all(os.path.exists(f"{tmp}/time/{f}") for f in OUTPUTS)
    # A row's time is taken just before the limit test and rounded to 0.1 s, so only the earlier rows are bounded.
    ok = meta["stop"] == "time" and meta["episodes_done"] == len(rows) < 10**6 and max(secs[:-1]) <= 2 and files
    check("wall_clock_stop", "a 2-second limit ends training after the first episode that finishes past 2 s",
          "stop=time, episodes_done = rows, every row but the last at <= 2.0 s, all outputs written",
          f"stop={meta['stop']}, episodes_done={meta['episodes_done']}, rows={len(rows)}, last two times "
          f"{secs[-2:]}, outputs {files}", ok)

    out = f"{tmp}/kill"
    cmd = [sys.executable, "dqn.py", "--env", "CartPole-v1", "--episodes", str(10**6), "--out", out]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None and not (os.path.exists(f"{out}/model.pt") and sum(1 for _ in open(f"{out}/train.csv")) > 30):
        time.sleep(0.2)  # stops waiting if the subprocess dies on its own; the check below then fails
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    episodes = [int(r["episode"]) for r in csv.DictReader(open(f"{out}/train.csv"))]
    code = subprocess.run([sys.executable, "dqn.py", "--out", out, "--eval-only"], capture_output=True).returncode
    files = all(os.path.exists(f"{out}/{f}") for f in OUTPUTS)
    k = json.load(open(f"{out}/metrics.json"))["train"]["episodes"] if files else None
    ok = episodes == list(range(1, len(episodes) + 1)) and code == 0 and files and k == len(episodes)
    check("interrupted_run", "SIGKILL during training, then --eval-only",
          "episodes 1..k once each, eval-only exit 0, all outputs, metrics episodes = k",
          f"episodes 1..{len(episodes)} in order: {episodes == list(range(1, len(episodes) + 1))}, exit {code}, "
          f"outputs {files}, metrics episodes {k}", ok)

    logs, weights = [], []
    for i in range(2):
        train("CartPole-v1", 0, f"{tmp}/det{i}", 40, warmup=200)
        logs.append([{c: r[c] for c in r if c != "seconds"} for r in csv.DictReader(open(f"{tmp}/det{i}/train.csv"))])
        weights.append(torch.load(f"{tmp}/det{i}/model.pt", weights_only=False)["final"])
    same_w = all(torch.equal(weights[0][key], weights[1][key]) for key in weights[0])
    learned = any(r["loss"] != "nan" for r in logs[0])
    check("determinism_cpu", "two 40-episode CartPole runs with the same seed (learning from step 200)",
          "identical logs and final weights", f"logs identical {logs[0] == logs[1]}, weights identical {same_w}, "
          f"learning happened {learned}", logs[0] == logs[1] and same_w and learned)
    shutil.rmtree(tmp)

    ranges = [(s * 10**6 + 1, s * 10**6 + 10**6) for s in range(3)] + [
        (EVAL_SEED, EVAL_SEED + 499), (SELECT_SEED, SELECT_SEED + 9), (VAL_SEED, VAL_SEED + 10**4)]
    ranges.sort()
    ok = all(a[1] < b[0] for a, b in zip(ranges, ranges[1:])) and ranges[-1][1] < 2**32
    check("seed_disjoint", "reset-seed ranges of training (seeds 0-2), evaluation, selection and validation",
          "pairwise disjoint, all < 2^32", f"{ranges}", ok)

    ma = moving_average(np.arange(1, 151))
    check("moving_average_alignment", "trailing 100-episode mean of 1..150, and of a 30-episode run",
          "first 50.5, length 51; short run empty", f"first {ma[0]}, length {len(ma)}; short run length "
          f"{len(moving_average(np.arange(30)))}", ma[0] == 50.5 and len(ma) == 51 and len(moving_average(np.arange(30))) == 0)
    plot_hist([500.0] * 500, "results/check_hist_single_value.png", "CartPole-v1: histogram check, 500 equal scores")
    sd = stats([500.0] * 500)["std"]
    check("histogram_single_value", "500 equal evaluation scores", "plot written, std 0",
          f"plot written {os.path.exists('results/check_hist_single_value.png')}, std {sd}",
          os.path.exists("results/check_hist_single_value.png") and sd == 0)


def ensure_run(out, env_id, seed, episodes, **overrides):
    """Train a validation run once; later calls reuse it (metrics.json is written last, so it marks a finished run)."""
    if not os.path.exists(f"{out}/metrics.json"):
        train(env_id, seed, out, episodes, **overrides)
    return json.load(open(f"{out}/metrics.json"))


def column(run, name):
    return np.array([float(r[name]) for r in csv.DictReader(open(f"{run}/train.csv"))])


def group_cartpole():
    env_id, main_dir = "CartPole-v1", "results/cartpole_s0"
    main = json.load(open(f"{main_dir}/metrics.json"))
    meta, ev = main["meta"], main["eval_best"]
    base = {k: meta[k] for k in ENVS[env_id]}  # every control run uses the main run's settings, changing one thing
    episodes = meta["episodes"]
    check("solved_seed0", "500-episode greedy evaluation of the seed-0 best snapshot", "mean >= 475",
          f"mean {ev['mean']:.1f} +- {ev['std']:.1f} (std), {ev['share_at_time_limit']:.1%} reach 500, "
          f"snapshot of episode {meta['best_episode']}", ev["mean"] >= 475)

    runs = {0.5: "results/val_cartpole_gamma0.5", 0.9: "results/val_cartpole_gamma0.9", 0.95: main_dir}
    level = {}
    for g, run in runs.items():
        if run != main_dir:
            ensure_run(run, env_id, 0, episodes, **dict(base, gamma=g))
        level[g] = column(run, "q_max")[-50:].mean()
        ratio = level[g] * (1 - g)
        check(f"gamma_scale[{g}]", f"mean of the last 50 episode max-Q values vs 1/(1-gamma) = {1 / (1 - g):g}",
              "ratio in [0.9, 1.15]", f"{level[g]:.3f} (ratio {ratio:.3f})", 0.9 <= ratio <= 1.15)
    check("gamma_monotone", "Q-value scale increases with gamma", "Q(0.5) < Q(0.9) < Q(0.95)",
          str({g: round(v, 2) for g, v in level.items()}), level[0.5] < level[0.9] < level[0.95])

    env = gym.make(env_id)
    rng = np.random.default_rng(1)
    torch.manual_seed(0)
    init = q_net(base, (4,), 2)  # train() seeds torch the same way right before building its network
    rand = stats([run_episode(env, init, base, EVAL_SEED + i, rng, eps=1.0)["score"] for i in range(500)])
    bound = rand["mean"] + 3 * rand["se"]
    lr0 = ensure_run("results/val_cartpole_lr0", env_id, 0, episodes, **dict(base, lr=0.0))
    final = torch.load("results/val_cartpole_lr0/model.pt", weights_only=False)["final"]
    same = all(torch.equal(final[k], v) for k, v in init.state_dict().items())
    check("lr0_weights_unchanged", "learning rate 0: final weights vs the seeded initial weights", "identical",
          "identical" if same else "different", same)
    check("lr0_no_learning", "learning rate 0: greedy evaluation vs the uniform-random policy",
          f"<= random + 3 SE = {bound:.1f}", f"{lr0['eval_best']['mean']:.1f} (random {rand['mean']:.1f} +- "
          f"{rand['se']:.1f} SE)", lr0["eval_best"]["mean"] <= bound)
    untrained = stats([run_episode(env, init, base, EVAL_SEED + i, rng)["score"] for i in range(500)])
    z = z_score(ev, lr0["eval_best"])
    check("lr0_vs_untrained_posthoc", "[post-hoc, after lr0_no_learning failed] lr 0 evaluation vs the untrained "
          "network's own greedy evaluation, and the trained network vs lr 0", "equal; trained - lr0 > 3 SE",
          f"lr0 {lr0['eval_best']['mean']:.3f} vs untrained {untrained['mean']:.3f}; trained {ev['mean']:.1f} (z {z:.0f})",
          lr0["eval_best"]["mean"] == untrained["mean"] and z > 3)
    eps1 = ensure_run("results/val_cartpole_eps1", env_id, 0, episodes, **dict(base, eps_start=1.0, eps_end=1.0))
    check("eps1_off_policy", "trained only on uniformly random actions, evaluated greedily",
          f"> random + 3 SE = {bound:.1f}", f"{eps1['eval_best']['mean']:.1f} +- {eps1['eval_best']['se']:.1f} SE",
          eps1["eval_best"]["mean"] > bound)

    net = q_net(base, (4,), 2)
    net.load_state_dict(torch.load(f"{main_dir}/model.pt", weights_only=False)["best"])
    th, thd = np.linspace(-0.2, 0.2, 61), np.linspace(-1.5, 1.5, 61)
    T, TD = np.meshgrid(th, thd)  # rows: angular velocity, columns: angle
    grid = np.stack([np.zeros_like(T), np.zeros_like(T), T, TD], -1).reshape(-1, 4)
    with torch.no_grad():
        q = net(torch.as_tensor(grid, dtype=torch.float32)).numpy().reshape(61, 61, 2)
    act, gap = q.argmax(-1), q[..., 1] - q[..., 0]
    right = (act[(T > 0) & (TD > 0)] == 1).mean()
    left = (act[(T < 0) & (TD < 0)] == 0).mean()
    check("physics_quadrants", "greedy action on the (theta, theta_dot) grid, cart at rest in the centre",
          ">= 95% push right when falling right, >= 95% push left when falling left",
          f"{right:.1%} right, {left:.1%} left", right >= 0.95 and left >= 0.95)
    # Where the action cannot matter: next angle theta + 0.02 * theta_dot (Euler step) is past the limit whatever
    # the action, or the network sees no real difference between the actions (|gap| < 0.5 = 2.5% of 1/(1-gamma)).
    matters = (np.abs(T + 0.02 * TD) <= 0.2095) & (np.abs(gap) >= 0.5)
    right_m = (gap[(T > 0) & (TD > 0) & matters] > 0).mean()
    left_m = (gap[(T < 0) & (TD < 0) & matters] < 0).mean()
    check("physics_where_it_matters_posthoc", "[post-hoc, after physics_quadrants failed] the same quadrants, only "
          "cells not doomed next step and with |Q gap| >= 0.5", ">= 95% right in (+,+), >= 95% left in (-,-)",
          f"{right_m:.1%} right ({((T > 0) & (TD > 0) & matters).sum()} cells), {left_m:.1%} left "
          f"({((T < 0) & (TD < 0) & matters).sum()} cells)", right_m >= 0.95 and left_m >= 0.95)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ext = [th[0], th[-1], thd[0], thd[-1]]
    im = ax[0].imshow(act, origin="lower", extent=ext, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    ax[0].set_title("greedy action (blue: push left, red: push right)")
    m = np.abs(gap).max()
    im = ax[1].imshow(gap, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-m, vmax=m)
    ax[1].set_title("Q(s, right) - Q(s, left)")
    fig.colorbar(im, ax=ax[1], label="Q-value gap")
    for a in ax:
        a.axhline(0, color="k", lw=0.5)
        a.axvline(0, color="k", lw=0.5)
        for sign in (1, -1):  # beyond these lines the pole falls at the next step whatever the action
            a.plot(th, (sign * 0.2095 - th) / 0.02, "k--", lw=1, label="doomed beyond" if sign == 1 else None)
        a.set_ylim(thd[0], thd[-1])
        a.set_xlabel("pole angle θ (rad, > 0 leans right)")
        a.set_ylabel("pole angular velocity θ̇ (rad/s)")
    ax[0].legend(loc="lower right")
    fig.suptitle(f"CartPole-v1: learned policy over pole angle and angular velocity (x = ẋ = 0, seed 0 best snapshot)\n"
                 f"push right in {right:.1%} of (+,+) and push left in {left:.1%} of (−,−); where the action matters "
                 f"(not doomed, |gap| ≥ 0.5): {right_m:.1%} and {left_m:.1%}")
    fig.tight_layout()
    fig.savefig("results/cartpole_policy_map.png", dpi=120)
    plt.close(fig)

    causes = {}
    for which in ("best", "final"):  # how do greedy episodes end? (limits: |x| <= 2.4, |theta| <= 0.2095)
        net.load_state_dict(torch.load(f"{main_dir}/model.pt", weights_only=False)[which])
        ends = []
        for i in range(100):
            last = []
            run_episode(env, net, base, EVAL_SEED + i, rng, on_step=lambda o, r, info: last.append(o))
            o = last[-1]
            ends.append("cart off track" if abs(o[0]) > 2.4 else "pole fell" if abs(o[2]) > 0.2095 else "time limit")
        causes[which] = {c: ends.count(c) for c in ("time limit", "cart off track", "pole fell")}
    check("end_causes_posthoc", "[diagnostic, post-hoc] how 100 greedy evaluation episodes end, best snapshot and "
          "final weights", "reported", str(causes), True)
    ratio = ev["q0"]["mean"] / ev["g0"]["mean"]
    check("q0_vs_g0", "predicted Q(s0, a0) vs realized discounted return of the 500 evaluation episodes",
          "|ratio - 1| <= 0.10", f"Q {ev['q0']['mean']:.3f} vs G {ev['g0']['mean']:.3f} (ratio {ratio:.3f})",
          abs(ratio - 1) <= 0.10)


def z_score(a, b):
    """Distance between two means in standard errors of their difference (a, b are stats() dicts)."""
    return (a["mean"] - b["mean"]) / np.sqrt(a["se"] ** 2 + b["se"] ** 2)


def group_mspacman():
    env_id, run = "ALE/MsPacman-v5", "results/mspacman"
    main = json.load(open(f"{run}/metrics.json"))
    meta, ev = main["meta"], main["eval_best"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make(env_id)
    net = q_net(meta, (4, 88, 80), env.action_space.n).to(device)
    net.load_state_dict(torch.load(f"{run}/model.pt", weights_only=False)["best"])

    def visits(eps, n):
        """Play n evaluation-seed episodes; return their scores and, per episode, the set of 88x80 cells that
        Ms. Pac-Man's centre (centroid of her (210, 164, 74) pixels) visited."""
        scores, cells = [], []
        for i in range(n):
            seen = set()

            def where(obs, r, info):
                rows, cols = np.nonzero(np.all(obs[1:176:2, ::2] == (210, 164, 74), axis=2))
                if len(rows):  # invisible frames are skipped
                    seen.add((int(rows.mean().round()), int(cols.mean().round())))
            scores.append(run_episode(env, net, meta, EVAL_SEED + i, np.random.default_rng(i), eps, on_step=where)["score"])
            cells.append(seen)
        return scores, cells

    r_scores, r_cells = visits(1.0, 500)
    rand = stats(r_scores)
    check("random_reference", "uniform-random policy on the 500 evaluation seeds", "about 220 to 260 (reported)",
          f"{rand['mean']:.1f} +- {rand['se']:.1f} SE (std {rand['std']:.1f})", True)
    log = column(run, "score")
    first, last = stats(log[:100]), stats(log[-100:])
    for name, x, ref, label in [("learning_last100", last, rand, "random"), ("learning_eval", ev, rand, "random"),
                                ("learning_trend", last, first, "first 100 training episodes")]:
        z = z_score(x, ref)
        check(name, f"{'last 100 training episodes' if x is last else '500-episode greedy evaluation'} vs {label}",
              "z > 3", f"{x['mean']:.1f} +- {x['se']:.1f} vs {ref['mean']:.1f} +- {ref['se']:.1f} (z {z:.1f})", z > 3)

    ratio = ev["q0"]["mean"] / ev["g0"]["mean"]
    check("q0_vs_g0", "predicted Q(s0, a0) vs realized discounted clipped return, 500 evaluation episodes",
          "ratio in [0.5, 2]", f"Q {ev['q0']['mean']:.2f} vs G {ev['g0']['mean']:.2f} (ratio {ratio:.2f})",
          0.5 <= ratio <= 2)
    # [post-hoc, after q0_vs_g0 failed] Q(s_t, a_t) vs the realized discounted clipped return G_t at every step
    cfg = dict(meta, warmup=10**9, buffer=50_000, eps_start=0.0, eps_end=0.0)  # greedy, no learning
    agent = make_agent(cfg, net, (4, 88, 80))
    b, qs, gs, firsts = agent["buffer"], [], [], []
    for i in range(20):
        start = b.size
        run_episode(env, net, cfg, EVAL_SEED + i, np.random.default_rng(i), agent=agent)
        r, n = b.r[start:b.size], b.size - start
        g, G = 0.0, np.zeros(n)
        for t in range(n - 1, -1, -1):  # the last transition is the terminal one
            g = G[t] = r[t] + meta["gamma"] * g
        with torch.no_grad():
            q = torch.cat([net(tensor(b.s[k:min(k + 512, b.size)], device)) for k in range(start, b.size, 512)])
        qs.append(q.cpu().numpy()[np.arange(n), b.a[start:b.size]])
        gs.append(G)
        firsts.append(int(np.argmax(r > 0)))
    split = {name: (np.concatenate([q[sel(f)] for q, f in zip(qs, firsts)]).mean(),
                    np.concatenate([g[sel(f)] for g, f in zip(gs, firsts)]).mean())
             for name, sel in [("all", lambda f: slice(None)), ("opening", lambda f: slice(0, f)),
                               ("play", lambda f: slice(f, None))]}
    ratio = split["all"][0] / split["all"][1]
    check("q_vs_return_posthoc", "[post-hoc, after q0_vs_g0 failed] Q(s_t, a_t) vs realized return G_t over every step "
          "of 20 greedy episodes, also split at the first reward (opening with no possible reward / play)",
          "all-steps ratio in [0.5, 2]", f"first reward at t = {sorted(set(firsts))}; " + ", ".join(
              f"{k}: Q {q:.2f} vs G {g:.2f} ({q / g:.2f})" for k, (q, g) in split.items()), 0.5 <= ratio <= 2)
    T = min(len(q) for q in qs)
    plt.figure(figsize=(8, 4))
    plt.plot(np.mean([q[:T] for q in qs], 0), label="predicted Q(s_t, a_t)")
    plt.plot(np.mean([g[:T] for g in gs], 0), label="realized discounted return G_t (clipped rewards)")
    plt.axvline(min(firsts), color="k", ls="--", lw=1, label=f"first reward possible (t = {min(firsts)})")
    plt.title("MsPacman-v5: predicted vs realized value along an episode (mean of 20 greedy episodes)")
    plt.xlabel("step t of the episode")
    plt.ylabel("value (clipped-reward units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/mspacman_q_vs_return.png", dpi=120)
    plt.close()

    q = np.concatenate([column(run, "q_mean"), column(run, "q_max")])
    check("q_bounds", "every logged Q-value statistic of training", "within [-1, 100] (clipped rewards in {0, 1})",
          f"[{q.min():.2f}, {q.max():.2f}]", q.min() >= -1 and q.max() <= 100)

    _, t_cells = visits(0.0, 100)
    rc, tc = stats([len(c) for c in r_cells[:100]]), stats([len(c) for c in t_cells])
    z = z_score(tc, rc)
    check("coverage", "distinct maze cells visited per episode, trained (greedy) vs random, seeds 0-99", "z > 3",
          f"{tc['mean']:.0f} +- {tc['se']:.0f} vs {rc['mean']:.0f} +- {rc['se']:.0f} (z {z:.1f})", z > 3)
    maze = preprocess_observation(env.reset(seed=EVAL_SEED)[0])[..., 0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.6), layout="constrained")
    for a, cells, c, label in [(ax[0], r_cells[:100], rc, "uniform-random policy"), (ax[1], t_cells, tc, "trained DQN, greedy")]:
        heat = np.zeros((88, 80))
        for seen in cells:
            for cell in seen:
                heat[cell] += 1 / len(cells)
        a.imshow(maze, cmap="gray", alpha=0.35)
        im = a.imshow(np.ma.masked_equal(heat, 0), cmap="viridis", vmin=0, vmax=1)
        a.set_title(f"{label}: {c['mean']:.0f} ± {c['se']:.0f} cells per episode")
        a.set_xlabel("column (downsampled pixels)")
        a.set_ylabel("row (downsampled pixels)")
    fig.colorbar(im, ax=list(ax), label="share of episodes in which Ms. Pac-Man's centre visited the cell")
    fig.suptitle("MsPacman-v5: where Ms. Pac-Man goes, 100 evaluation episodes (seeds identical for both agents)")
    fig.savefig("results/mspacman_position_heatmap.png", dpi=120)
    plt.close(fig)


GROUPS = {"checks": group_checks, "cartpole": group_cartpole, "mspacman": group_mspacman}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in GROUPS:
        sys.exit(f"usage: python validate_dqn.py {{{'|'.join(GROUPS)}}}")
    group = sys.argv[1]
    GROUPS[group]()
    with open(f"results/validation_{group}.json", "w") as f:
        json.dump(ROWS, f, indent=1)
    failed = [r["name"] for r in ROWS if not r["ok"]]
    print(f"\n{group}: {len(ROWS) - len(failed)}/{len(ROWS)} PASS" + (f", FAIL: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
