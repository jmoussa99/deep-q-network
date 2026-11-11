# deep-q-network

Deep Q-Network (online network, replay buffer, target network) for `CartPole-v1` (γ = 0.95) and `ALE/MsPacman-v5`
(γ = 0.99), with validation experiments for edge cases and correctness. MLDS Assignment 2.

| file | content |
|---|---|
| `dqn.py` | the DQN: settings per game (`ENVS`), preprocessing, Q-network, replay buffer, training, evaluation, plots, CLI |
| `validate_dqn.py` | validation groups `checks`, `cartpole`, `mspacman` (each prints PASS/FAIL and writes `results/validation_<group>.json`) |
| `results/` | one folder per run (`train.csv`, `eval.csv`, `metrics.json`, plots), validation JSON files, traces |
| `report/report.md` | report draft |

## Setup

Tested on deepdish (Python 3.13, NVIDIA Blackwell GPUs, so PyTorch must be a CUDA ≥ 12.8 build):

```bash
git clone https://github.com/jmoussa99/deep-q-network && cd deep-q-network
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # pinned; pulls torch 2.11.0+cu128 and the Atari ROMs (ale-py)
```

Single-GPU rule: always expose exactly one card to a GPU run, for example
`export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. CartPole and the `checks` group run on the CPU.

## Training and evaluation

```bash
for s in 0 1 2; do .venv/bin/python dqn.py --env CartPole-v1 --seed $s --episodes 1000 --out results/cartpole_s$s & done; wait
.venv/bin/python dqn.py --env ALE/MsPacman-v5 --seed 0 --episodes 5000 --hours 17 --device cuda --out results/mspacman
.venv/bin/python dqn.py --out results/mspacman --eval-only     # rebuild evaluation and plots from saved files
```

A run trains for `--episodes` episodes, or until `--hours` have passed (the running episode is finished). It appends
one row per episode to `train.csv` and saves `model.pt` (final weights and the best snapshot) at every snapshot
check. It then plays 500 greedy episodes with the best snapshot (and with the final weights, if they differ) and
writes `eval.csv`, `metrics.json`, `q_values.png` (maximum Q-value per episode), `rewards.png` (episode reward and
trailing 100-episode mean) and `eval_hist.png`. Settings per game are in `ENVS` at the top of `dqn.py`.

## Validation

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python validate_dqn.py checks    # about 25 s on the CPU
CUDA_VISIBLE_DEVICES= .venv/bin/python validate_dqn.py cartpole  # needs results/cartpole_s0; trains 4 control runs once (about 5 min), then seconds
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv/bin/python validate_dqn.py mspacman  # needs results/mspacman
```

Every check compares an observed value with an expectation that was written down, with its source, before the check
first ran. A check passes only if the expectation is met. The group exits with status 1 if any check fails.

### `checks`: edge cases and algorithm correctness

| check | what it does | how it validates | expected (source) |
|---|---|---|---|
| `preprocess_exact` | preprocesses every frame of a random MsPacman episode | compares with the assignment formula recomputed in signed int64. The given code sums uint8 into uint64, so `// 3 - 128` wraps, and only the int8 cast makes it exact | all frames equal, shape 88×80×1, int8 (assignment formula) |
| `preprocess_raw_unchanged` | keeps a copy of each raw frame before preprocessing | byte comparison after preprocessing | all frames unchanged (`sum` returns a new array) |
| `mspacman_color_unique` | finds pixels whose RGB sum is 448, the value the contrast step erases | counts those whose RGB is not exactly Ms. Pac-Man's (210, 164, 74) | 0, so no other object is erased (assignment comment) |
| `mspacman_visible`, `mspacman_compact` | finds Ms. Pac-Man's color in each frame | share of frames where she is visible, and whether her pixels fit in one 12×12 box | ≥ 90% visible, ≥ 99% compact (Atari sprite size) |
| `frame_stack_reset` | plays two episodes in a row | the first stored state of episode 2 must be 4 copies of its own reset frame | true (no frame leaks between episodes) |
| `life_loss_not_terminal` | records `info["lives"]` and the stored done flags of a full episode | lives fall 3 → 0 inside one episode, and only the game-over transition is terminal | d = 1 only at the last step (ALE: game over after the last life) |
| `reward_clip_vs_score` | records raw rewards next to the stored ones | stored = sign(raw), score = Σ raw | true (learning uses clipped rewards, every plot uses raw scores) |
| `buffer_partial`, `buffer_wrap` | a replay buffer of capacity 5 with 3 and then 8 transitions | samples only come from filled slots; after wrapping it holds the 5 newest | {1, 2, 3}; [4..8] (uniform sampling, FIFO overwrite) |
| `epsilon_bounds` | evaluates the ε schedule at 0, decay/2, decay, 10·decay | linear decay, then constant | start, midpoint, end, end (design) |
| `target_sync` | watches the target and online networks at every step of a CartPole run with sync period 100 | the target is constant inside each window, equal to the online net right after a sync, different otherwise | all three hold (hard target copy every C steps) |
| `truncation_bootstrap` | CartPole with a 20-step time limit, an episode cut while the pole is still up | the last stored transition | d = 0, so the target bootstraps (a time limit is not a failure) |
| `termination_terminal` | a random CartPole episode where the pole falls | the last state lies outside the limits (\|x\| > 2.4 or \|θ\| > 0.2095) | d = 1 only at the last step (Gymnasium termination rule) |
| `chain_q_star` / `chain_greedy` / `chain_terminal` | runs the training update `learn()` on all 8 transitions of a 5-state chain (goal reward 1, terminal, γ = 0.9) with a tabular network | compares with the closed-form optimum V*(s) = γ^(3−s), Q*(s, left) = γ·V*(max(s−1, 0)) | max error ≤ 1e-3, greedy = right everywhere, Q(3, right) = 1 (closed form) |
| `td_targets_hand` | a terminal and a non-terminal transition with a hand-set target network | compares with targets computed by hand | [5.5, 2.0] (hand calculation) |
| `trace_mspacman_deterministic` | the first 10 steps of a MsPacman episode, twice, without sticky actions | the two traces must be identical | identical (ALE is deterministic without sticky actions) |
| `divergence_guard` | trains CartPole with lr = 1e30, which overflows the network at the first update | the run must stop with an error naming the episode and step, and the files saved earlier must still load | `RuntimeError` in the first learning episode (design: never train on NaN/inf) |
| `wall_clock_stop` | trains CartPole with a 2-second limit and a budget of 10^6 episodes | the stop reason, the row count, and the logged times | `stop = time`, the earlier rows at ≤ 2.0 s, every output written |
| `interrupted_run` | starts `dqn.py` in a subprocess, kills it with SIGKILL after its first save, then runs `--eval-only` | the log must list episodes 1..k once each, and eval-only must rebuild everything from the saved files | true, exit 0 (per-row flush, atomic `model.pt`) |
| `determinism_cpu` | trains the same 40-episode CartPole run twice, with learning from step 200 | compares the logs and final weights | identical (seeded, CPU) |
| `seed_disjoint` | lists the reset-seed ranges of training, evaluation, snapshot selection and validation | pairwise overlap and the 2^32 seed limit | disjoint, so evaluation episodes are never seen in training |
| `moving_average_alignment` | trailing 100-episode mean of 1..150 and of a 30-episode run | first value and length; `np.convolve(..., "valid")` would silently swap its arguments for short input | 50.5 and 51; empty for 30 episodes |
| `histogram_single_value` | histogram of 500 equal scores (a perfectly solved CartPole) | the plot must render one visible bar, std 0 | `results/check_hist_single_value.png` |

The first-10-step traces are saved to `results/trace_cartpole.txt` and `results/trace_mspacman.txt`. Each row shows
the state (for MsPacman, the ids of the 4 stacked frames, which shift by one each step), the Q-values, ε, the action,
the raw and stored reward, d, and the TD target.

### `cartpole`: is the agent right for the right reasons?

The control runs (`results/val_cartpole_*`) use the main run's settings and change exactly one thing. Seed 0 is the
reported CartPole agent; seeds 1 and 2 show the spread.

| check | what it does | how it validates | expected (source) |
|---|---|---|---|
| `solved_seed0` | 500 greedy episodes of the seed-0 best snapshot | mean score | ≥ 475 (Gymnasium's threshold for CartPole-v1) |
| `gamma_scale[γ]`, `gamma_monotone` | trains with γ = 0.5 and 0.9 next to the main γ = 0.95 | the mean over the last 50 episodes of the episode-max Q, divided by 1/(1−γ) | ratio in [0.9, 1.15] and increasing in γ. A policy that survives many horizons earns Σγ^t = 1/(1−γ) = 2, 10, 20; the max in the target may overestimate a little |
| `lr0_weights_unchanged` | trains with learning rate 0 | final vs initial weights | identical (Adam with lr 0) |
| `lr0_no_learning` | the lr = 0 network, evaluated greedily | vs the uniform-random policy on the same seeds | ≤ random + 3 SE. **Known FAIL: wrong expectation** (see below) |
| `lr0_vs_untrained_posthoc` | the untrained seed-0 network, evaluated greedily | lr = 0 result vs the untrained network, and trained vs lr = 0 | equal; trained > lr0 by > 3 SE (post-hoc) |
| `eps1_off_policy` | trains on uniformly random actions only (ε = 1 throughout), then evaluates greedily | vs the random policy | > random + 3 SE (Q-learning learns the greedy policy off-policy) |
| `physics_quadrants` | greedy action on a 61×61 grid of pole angle θ × angular velocity θ̇ (cart at rest, centered) | share of "push right" when the pole leans and falls right, and of "push left" in the mirrored case | ≥ 95% each (CartPole equations: pushing right gives negative angular acceleration). **Known FAIL: wrong expectation** (see below) |
| `physics_where_it_matters_posthoc` | the same grid, only cells where the pole is not doomed next step and the network's Q gap is ≥ 0.5 | same shares | ≥ 95% each (post-hoc) |
| `end_causes_posthoc` | plays 100 greedy evaluation episodes with the best snapshot and with the final weights | records whether each episode ended at the time limit, by the cart leaving the track, or by the pole falling | reported (diagnostic, added after the failure analysis) |
| `q0_vs_g0` | Q(s₀, a₀) predicted at the start of each evaluation episode | vs the realized discounted return | within 10% (definition of Q) |

The group exits with status 1 because of two FAILs that are kept on purpose: the expectations were written before
the runs and turned out to be wrong.

- `lr0_no_learning`: a fixed random network is a deterministic controller. The seed-0 initialization alone balances
  for about 91 steps, while random actions last about 23.
- `physics_quadrants`: near upright, and where the pole falls at the next step whatever the action, both actions
  are worth almost the same, so the argmax is a tie.

The `*_posthoc` rows test the corrected hypotheses, but they were defined after seeing the data. The plot
`results/cartpole_policy_map.png` is the non-standard CartPole figure: the greedy action and the Q-value gap over
(θ, θ̇), with the doomed region marked.

### `mspacman`: does the agent learn, and do its values make sense?

Runs on the trained `results/mspacman` (best snapshot), on one GPU, in about 5 minutes. z is the difference of two
means divided by the standard error of that difference.

| check | what it does | how it validates | expected (source) |
|---|---|---|---|
| `random_reference` | plays the uniform-random policy on the 500 evaluation seeds | reported, the baseline for the tests below | about 220 to 260 (planning probe, first training episodes) |
| `learning_last100` | last 100 training episodes vs random | z | > 3 (the assignment grades learning and improvement) |
| `learning_eval` | 500 greedy evaluation episodes vs random | z | > 3 |
| `learning_trend` | last 100 vs first 100 training episodes (ε ≈ 0.9 early, so close to random) | z | > 3 |
| `q0_vs_g0` | Q(s₀, a₀) predicted at the start of each evaluation episode vs the realized discounted return of the clipped rewards (γ = 0.99) | ratio of the means | in [0.5, 2]; above 1 if the max overestimates |
| `q_vs_return_posthoc` | Q(s_t, a_t) vs the realized return G_t at every step of 20 greedy episodes, also split at the first reward | ratio of the means (all steps, opening, play) | all-steps ratio in [0.5, 2] (post-hoc, see below) |
| `q_bounds` | every logged per-episode mean and max of max_a Q | range | within [−1, 100]: clipped rewards are 0 or 1, so 0 ≤ Q ≤ 1/(1−0.99) |
| `coverage` | tracks Ms. Pac-Man's position (centroid of her color) in 100 episodes of the random and of the trained agent, on the same seeds | distinct maze cells visited per episode, z | > 3: a good agent clears pellets all over the maze, while a random walk stays near the start |

The group exits with status 1 because of one FAIL that is kept on purpose. `q0_vs_g0` fails (ratio 0.28): every
MsPacman episode opens with 67 steps in which no reward is possible and actions have no effect, and the network
does not represent this countdown (Q ≈ 3.8 while the realized return rises from 13 to 27). Over all steps the ratio
is 1.07, and during play it is 1.21 (overestimation, as expected). `results/mspacman_q_vs_return.png` plots Q_t and
G_t along the episode. The post-hoc row was defined after seeing the failure.

The non-standard MsPacman figure, `results/mspacman_position_heatmap.png`, shows for each maze cell the share of
episodes in which Ms. Pac-Man passed through it, for the random and the trained agent.
