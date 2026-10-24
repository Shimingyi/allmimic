# Differences from the original implementation

The code starts from **dhmbb2/supermimic_vibe** (https://github.com/dhmbb2/supermimic_vibe), commit
`072f0910dadc9d95c720e625b546120e9ce917c9`, an unofficial re-implementation of SuperMimic (arXiv 2610.28480):

| here | original |
|---|---|
| `allmimic/` (package) | `code/supermimic/` |
| `scripts/` (`prepare_motions.py`, `filter_bones.py`, `push_test.py`, `step_test.py`, `bench_rollout.py`) | `code/scripts/` |
| `requirements.txt` | `requirements.txt` |

The package is renamed `supermimic` → `allmimic`, and the tree is formatted with `ruff format` (`ruff.toml`; to diff
against the original, format an original checkout with the same file first).

Every change below is a switch whose default keeps the original behavior (bit-identical where noted). The numbers
(#N) are referenced from code comments and tests. `python -m allmimic.train --preset recommended` turns on #11, #13,
#16, #19, #20, #21, #23 and #24, the configuration described in the README; the others are optional.

## Training and model

| # | change | default = original? | check |
|---|---|---|---|
| 1 | `--goal-pos ref`: the K future reference root xy positions are expressed in the first future frame's heading frame, with the absolute reference root height as z, instead of relative to the robot root, so the goal needs no global robot position. The height stays absolute because a pure displacement loses the target height (crouching, lying, getting up). | yes (`robot`) | `tests/test_goal_modes.py` |
| 2 | `--goal-rot ref`: future reference root orientations relative to the first future frame's heading instead of the robot root; with `--goal-pos ref` the goal carries no robot state. | yes (`robot`) | `tests/test_goal_modes.py` |
| 3 | `--goal-pos-noise M`: uniform ±M m noise on the goal root positions, policy input only (collection and imagination), never critic or evaluation. | yes (0) | `tests/test_goal_modes.py` |
| 4 | Privileged critic goal: with a reference-frame goal (#1 / #2), the critic also gets the robot-relative goal root pose (`Task.goal_priv`), because the tracking rewards depend on it while the actor's goal does not. | yes | `tests/test_goal_modes.py` |
| 5 | Diverged-world guard: non-finite simulator states are replaced by a finite placeholder and hard-terminated (`info['diverged']`), stored with episode id −1 so no WM window or imagination start uses them, and a non-finite gradient norm zeroes that step's gradients (WM, policy, critic). No host syncs. Not wired into `ppo_wm.py` / `velocity.py`. | yes (bit-identical when nothing diverges) | `tests/test_nan_guard.py` |
| 6 | Compact motion library: `MotionLibrary` also reads `compact_v1` files (root position float32, the other 68 channels float16; 148 B/frame instead of ~300 B) and replaces the per-frame index tables by `clip_of()` / `bin_of()`. Builder: `tools/build_allmimic_library.py`. | yes (original format bit-identical to the original loader) | `tests/test_motion_lib.py` |
| 7 | Resume fix: warm-up runs while the replay buffer is short, not only at iteration 0 (the buffer is not checkpointed, so resuming crashed). | yes (identical on fresh runs) | — |
| 8 | `evaluate()` also returns `per_clip_steps`. | yes | — |
| 9 | Reset-sampling bin update: episodes that end in the same 1 s bin in one step are averaged and all counted; indexed writes kept only the last one. | bit-identical without such collisions | `tests/test_flags.py` |
| 10 | Range tensors built once instead of three host→device copies per step; `Normalizer.update` writes in place (needed for CUDA graphs). | yes (bit-identical) | `tests/test_flags.py` |
| 11 | Residual actions (as in SuperTrack): `--action res --res-scale α`, setpoint `q_ref(t+1) + α·action_scale·a`. The simulator, the WM and the replay windows always see the absolute setpoint, so the learned dynamics do not depend on the reference; the action-rate reward and the observation history keep the policy output. | yes (`abs`) | `tests/test_flags.py` |
| 12 | `--future-offsets 1,2,3,4,6,10,15,25,40,60`: a sparse 1.2 s goal window with the same frame count instead of 10 consecutive frames. | yes (1..10) | `tests/test_flags.py` |
| 13 | `--latent N`: width of the policy's tanh bottleneck between motion encoder and decoder (original 32). | yes (32) | `tests/test_flags.py` |
| 14 | `--heading twist`: heading frames from the swing-twist yaw, singular only when fully inverted, instead of the pelvis x-axis projection, which is singular when lying, crawling or in a handstand. The reward's yaw alignment is unchanged. | yes (`xaxis`) | `tests/test_flags.py` |
| 15 | `allmimic/adapt.py`: continue training a checkpoint on new clips only within a wall-clock budget (compilation outside the clock), evaluating periodically. | new module | — |
| 16 | The policy update is split into `imagination_inputs()` / `rollout()` / `img_step()`. `--compile-bptt MODE` compiles one imagined step (called H times); `--wm-compile-mode MODE` sets the compile mode of the WM window loss. `reduce-overhead` (CUDA graphs) matches eager gradients in our checks; `max-autotune-no-cudagraphs` produced wrong policy gradients, do not use it. Check new hardware with the GPU test. | yes (eager, bit-identical to the original loop) | `tests/gpu/test_bptt_compile.py` |
| 17 | `--bptt-checkpoint k`: recompute every k-th imagined step in the backward pass instead of storing its activations (bit-identical gradients, less memory, more time). | yes (off) | `tests/gpu/test_bptt_checkpoint.py` |
| 18 | `--rot-repr tripoint`: root orientation inputs as three marker points instead of 6D (inputs only). | yes (`6d`) | `tests/test_flags.py` |
| 19 | `--softmax-cap k`: caps each 1 s bin's share of the return-based reset softmax at k × the uniform share, so hard-example mining cannot concentrate on a few unlearnable bins. `--clip-list file`: train only on the listed clips (e.g. the feasible list of `tools/filter_infeasible.py`). | yes (off) | `tests/test_flags.py` |
| 20 | `--res-legs abs`: the 12 hip / knee / ankle joints use absolute actions while the upper body stays residual (leg targets tied to the reference fight root balance); `--res-scale-legs` gives the legs their own residual scale. `--wm-norm-eps`: std floor of the WM input normalizer. | yes (`res`, 0, 1e-3) | `tests/test_flags.py` |
| 21 | `--wm-joint-weights {tracked,axis_sq}`: per-joint weights on the q / qd dims of the WM state-error loss, from D. Holden's "Joint Error Propagation" (2024). `tracked` sums, over the tracked reward bodies in each joint's subtree, the squared lever arm + 1 (`allmimic/joint_weights.py`, generated by `tools/joint_error_weights.py`), clipped to [1/c, c] by `--wm-joint-weight-clip c` (4) and normalized to mean 1. The logged WM errors stay unweighted. | yes (uniform) | `tests/test_flags.py` |
| 22 | Robustness: evaluators refuse checkpoints whose cfg has keys this code does not know (`train.unknown_cfg_keys`); `adapt` clears the training clip whitelist; evaluation envs with fewer than 256 worlds get a contact pool sized for 256 (a 1-world env dropped contacts). Known issue: the per-world constraint pool (`njmax = 250`) can overflow in crawling / falling clips. | yes (except small evaluation envs keep all contacts) | `tests/test_flags.py` |
| 23 | `--collect-det-frac f`: the first f of the collection envs act without exploration noise. Collection otherwise always adds noise while imagination is deterministic, so the WM never sees the states a deterministic (deployed) policy reaches and the policy can exploit WM errors there. `--img-action-noise x` is the alternative: N(0, x) on the imagined actions. | yes (0 / 0) | `tests/test_wm_grad_fidelity.py`, `tools/wm_grad_fidelity.py` |
| 24 | `--pi-clip c`: the policy gradient-norm clip (original 1.0). The policy loss is divided by the horizon H = 32, so its gradient norm is far below 1.0 and that clip never acts; c = 1/32 is the 1.0 clip without the 1/H. `--critic-heads n`: an ensemble of value heads, the policy bootstraps on their mean. | yes (1.0 / 1) | `tests/test_flags.py` |
| 25 | `--wm-rot-out tripoint`: the WM predicts the rotation change as tri-point markers (Gram-Schmidt decode) instead of axis-angle. | yes (`axis`) | `tests/test_flags.py` |
| 26 | WM-consistency options: `--clean-prev-action`, `--w-residual`, `--retire-g`, `--bptt-detach`, and an imagination-gap monitor `--gap-every` (logging only). | yes (off) | `tests/test_flags.py` |
| 27 | `--optimizer muon`: Muon on the hidden weight matrices of WM and policy, Adam elsewhere (torch ≥ 2.9). | yes (`adam`) | `tests/test_flags.py` |
| 28 | `--bin-prior file.npz`: cluster-balanced reset sampling; `tools/bin_clusters.py` clusters the 1 s bins by reference kinematics and writes per-bin weights that multiply the return-based probabilities (rare motions such as flips are a tiny share of the bins). | yes (off) | `tests/test_flags.py` |
| 29 | `--value-clip true`: the critic values used as lambda-return targets and as the policy's bootstrap are clamped to ±max_step_reward / (1 − γ) (±10 with the default reward and γ = 0.99). Long runs without it diverged after 5 to 7 h: the critic overestimated a few states, the policy followed the bootstrap there and the critic loss exploded. Normal values are 6.5 to 7, so the clamp only acts on those outliers. `--preset recommended` also sets `--bin-prior-alpha 0.3` (#28) and `--value-clip true`. | yes (off) | `tests/test_flags.py` |

## Other differences

- `--preset NAME` in `allmimic.train` (`train.PRESETS`): a named set of flags. It only replaces argparse defaults, so
  explicit flags override it; it is not stored in checkpoints (the saved cfg holds the resolved values).
- `scripts/push_test.py` builds its evaluation env from each checkpoint's task cfg (it used the default one, which
  would drive a residual-action policy with absolute setpoints).
- New tools: library builder with compact storage and 10 s segments (`tools/build_allmimic_library.py`,
  `tools/make_segments.py`), BONES-SEED held-out splits (`tools/make_bones_splits.py`), infeasible-clip filter
  (`tools/filter_infeasible.py`), ground-penetration correction (`tools/ground_correct.py`), bin clustering
  (`tools/bin_clusters.py`), test-set evaluation and failure report (`tools/eval_allmimic.py`,
  `tools/failure_report.py`), WM joint weights (`tools/joint_error_weights.py`), WM gradient-fidelity probe
  (`tools/wm_grad_fidelity.py`).
- CPU tests in `tests/` (pytest or scripts), GPU checks in `tests/gpu/`.

## Credits for borrowed ideas

- SuperTrack (Fussell et al., 2021): residual actions (#11).
- BeyondMimic: reward terms and training-noise conventions.
- D. Holden, "Joint Error Propagation" (theorangeduck.com, 2024): WM joint weights (#21).
- Other open SuperMimic re-implementations: `supermimic_vibed` (the gradient-fidelity probe in
  `tools/wm_grad_fidelity.py`; the policy clip without the 1/H factor, #24) and `rethinking-supermimic` (critic
  ensemble, #24; Muon, #27).
