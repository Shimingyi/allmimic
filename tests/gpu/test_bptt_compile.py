"""GPU check of TrainCfg.compile_bptt (CHANGES #16): Trainer.rollout with a compiled img_step vs the eager reference.

  PYTHONPATH=. python tests/gpu/test_bptt_compile.py --motions <lib.npz> [--modes default reduce-overhead]
  # the recommended preset (task / network settings and its compile modes):
  PYTHONPATH=. python tests/gpu/test_bptt_compile.py --motions <lib.npz> --preset recommended --modes reduce-overhead \
      --traj ref:- bptt:reduce-overhead wm:reduce-overhead

1. the refactored eager rollout (img_step) equals the frozen pre-refactor loop bit for bit (noise on, same seed)
2. compiled img_step vs eager in fp32 (AMP off), observation noise off (eager and inductor draw random numbers
   differently): J relative error < 2e-3, gradient cosine > 0.995, norm ratio +-2%. Suspected: in bf16 the 32-step
   gradient is chaotic (bf16 per-call checks gave cos 0.63 for `default` on one run, 1.0 on another), so the eager
   path is also run with the start state moved by 1e-6 and that cosine is printed as the bf16 noise floor.
3. update_policy() time per call (rollout + backward + Adam + critic), noise back on
4. WM window loss (TrainCfg.wm_compile_mode) vs the default compile on the same windows: loss and WM gradients
   (same tolerances), then update_wm() time per call (10 window batches)
5. trajectory gate (the one that counts): from one snapshot and seed, --traj-updates full updates (update_wm +
   update_policy, noise off, fresh Adam state) per candidate; the parameter change (final - snapshot) of WM and
   policy vs the reference path's change: cosine > 0.99 and norm ratio within 5% (raw parameters are useless: 5
   updates move them ~1e-4; with the snapshot's Adam momentum every candidate passed). 'ref:-' = noise floor. A single-call gradient check passed reduce-overhead
   img_step once and failed it (cos -0.09) on a rerun, so per-call checks alone are not enough.
"""

import argparse
import sys
import time

import torch

from allmimic import networks
from allmimic.networks import apply_delta_c, run_mlp
from allmimic.task import flat_to_state
from allmimic.train import PRESETS, Trainer, TrainCfg


def legacy_rollout(tr, s0, xi, obs_hist, a_prev, bias, ref, ref_kin, valid, rem_all):
    """frozen pre-#16 Trainer.update_policy loop (J only)"""
    cfg, task, wm = tr.cfg, tr.task, tr.wm
    B, H, K = cfg.pi_batch, cfg.horizon, tr.tcfg.future
    ref_at = lambda t: {k: v[:, t] for k, v in ref.items()}
    ref_future = lambda t: {k: v[:, t + 1 : t + 1 + K] for k, v in ref.items()}
    ref_kin_at = lambda t: tuple(x[:, t] for x in ref_kin)
    s = flat_to_state(s0)
    hist = obs_hist + task.sample_obs_noise((B, tr.tcfg.history))
    an = task.analytics_fast(s, ref_at(0), ref_kin_at(0), ref_future(0), a_prev, a_prev, bias)
    c = torch.ones(B, device=tr.device)
    J = torch.zeros(B, device=tr.device)
    std, mean = wm.out_norm.std, wm.out_norm.mean
    for t in range(H):
        goal, feats = an['goal'], an['feats']
        a = tr.policy(hist.flatten(1), task.noisy_goal(goal))
        a_sp = task.setpoint_action(a, ref['q'][:, t + 1])
        dlt = run_mlp(wm.net, wm.in_norm(torch.cat([feats, a_sp, xi], dim=-1))) * std + mean
        s1 = apply_delta_c(s, dlt)
        an = task.analytics_fast(s1, ref_at(t + 1), ref_kin_at(t + 1), ref_future(t + 1), a, a_prev, bias)
        r, d = an['r'], an['soft']
        J = J + (cfg.gamma**t) * c * r * valid[:, t + 1]
        c = c * (1 - d)
        o1 = an['obs'] + task.sample_obs_noise((B,))
        hist = torch.cat([hist[:, 1:], o1.unsqueeze(1)], dim=1)
        s, a_prev = s1, a
    xH = tr.critic_input(an['feats'], an['goal'], xi, rem_all[:, H], an.get('goal_priv'))
    if cfg.use_critic:
        vH = tr.critic(xH) * valid[:, H]
        J = J + (cfg.gamma**H) * c * vH
    return (J,)


def loss_and_grads(tr, fn, args):
    """rollout fn -> (J (B,), flat policy gradient of the BPTT loss)"""
    for p in tr.wm.net.parameters():
        p.requires_grad_(False)
    tr.opt_pi.zero_grad(set_to_none=True)
    J = fn(*args)[0]
    (-J.mean() / tr.cfg.horizon).backward()
    params = [p for g in tr.opt_pi.param_groups for p in g['params']]
    g = torch.cat([p.grad.flatten() for p in params])
    for p in tr.wm.net.parameters():
        p.requires_grad_(True)
    tr.opt_pi.zero_grad(set_to_none=True)
    tr.critic.zero_grad(set_to_none=True)
    return J.detach(), g


def time_updates(tr, n=20):
    for _ in range(3):
        tr.update_policy()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        tr.update_policy()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--motions', required=True)
    ap.add_argument('--out', default='/tmp/test_bptt_compile')
    ap.add_argument('--pi-batch', type=int, default=2048)
    ap.add_argument('--modes', nargs='+', default=['default'])
    ap.add_argument('--wm-modes', nargs='*', default=['reduce-overhead'])
    ap.add_argument('--traj', nargs='*', default=['ref:-', 'bptt:default', 'wm:reduce-overhead'],
                    help='candidates for the trajectory gate, kind:mode with kind bptt or wm')  # fmt: skip
    ap.add_argument('--traj-updates', type=int, default=5)
    ap.add_argument('--preset', default='', choices=['', *PRESETS],
                    help='task / network settings of a train.py preset (its compile modes are ignored: the candidates '
                         'are compiled here)')  # fmt: skip
    args = ap.parse_args(argv)
    pre = {k: v for k, v in PRESETS.get(args.preset, {}).items() if k not in ('compile_bptt', 'wm_compile_mode')}
    cfg = TrainCfg(**pre, motions=args.motions, out=args.out, pi_batch=args.pi_batch, eval_every=0)
    tr = Trainer(cfg)
    while tr.buf.size < cfg.warmup_steps:
        tr.collect()
    for _ in range(20):  # non-trivial normalizers / weights
        tr.collect()
        tr.update_wm()
        tr.update_policy()
    t_eager = time_updates(tr)
    print(f'eager update_policy {t_eager:.1f} ms', flush=True)

    torch.manual_seed(1)
    inp = tr.imagination_inputs()
    torch.manual_seed(2)
    Jl, gl = loss_and_grads(tr, lambda *a: legacy_rollout(tr, *a), inp)
    torch.manual_seed(2)
    Je, ge = loss_and_grads(tr, tr.rollout, inp)
    ok = torch.equal(Jl, Je) and torch.equal(gl, ge)
    print(f'refactor bit-identical to the frozen loop (noise on): {ok}', flush=True)

    noise = tr.task.obs_noise.clone()
    eager_step = tr.img_step

    def cmp(a, b):
        """(J, g) pairs -> (J rel err, grad cos, grad norm ratio)"""
        rel = ((b[0] - a[0]).abs().mean() / a[0].abs().mean()).item()
        return rel, torch.nn.functional.cosine_similarity(a[1], b[1], dim=0).item(), (b[1].norm() / a[1].norm()).item()

    # noise floor: the 32-step BPTT gradient is chaotic (steep soft terminations), so eager bf16 itself moves this much
    # when the start state is perturbed by 1e-6 relative
    tr.task.obs_noise.zero_()
    tr._img_step = eager_step
    base = loss_and_grads(tr, tr.rollout, inp)
    inp_p = (inp[0] * (1 + 1e-6 * torch.randn_like(inp[0])),) + tuple(inp[1:])
    print('bf16 eager, start state perturbed 1e-6: J rel err %.1e  grad cos %.5f  norm ratio %.4f'
          % cmp(base, loss_and_grads(tr, tr.rollout, inp_p)), flush=True)  # fmt: skip
    tr.task.obs_noise.copy_(noise)
    for mode in args.modes:
        compiled = torch.compile(tr.img_step, dynamic=False, mode=mode)
        # correctness in fp32 (AMP off): no bf16 rounding for the chaos to amplify
        networks.AMP['enabled'] = False
        tr.task.obs_noise.zero_()
        tr._img_step = eager_step
        J0, g0 = loss_and_grads(tr, tr.rollout, inp)
        tr._img_step = compiled
        t0 = time.time()
        loss_and_grads(tr, tr.rollout, inp)
        compile_s = time.time() - t0
        loss_and_grads(tr, tr.rollout, inp)
        J1, g1 = loss_and_grads(tr, tr.rollout, inp)  # third call: past compile and cudagraph warm-up / recording
        tr.task.obs_noise.copy_(noise)
        networks.AMP['enabled'] = cfg.amp
        rel = ((J1 - J0).abs().mean() / J0.abs().mean()).item()
        cos = torch.nn.functional.cosine_similarity(g0, g1, dim=0).item()
        ratio = (g1.norm() / g0.norm()).item()
        good = rel < 2e-3 and cos > 0.995 and abs(ratio - 1) < 0.02
        ok &= good
        t = time_updates(tr)
        tr._img_step = eager_step
        print(f'{mode:28s} fp32: J rel err {rel:.1e}  grad cos {cos:.5f}  norm ratio {ratio:.4f}  '
              f'{"OK" if good else "FAIL"} | compile {compile_s:.0f}s | bf16 update_policy {t:.1f} ms '
              f'({t_eager / t:.2f}x)', flush=True)  # fmt: skip

    def time_wm(n=10):
        for _ in range(2):
            tr.update_wm()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            tr.update_wm()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    def wm_loss_grads(fn, w):
        tr.opt_wm.zero_grad(set_to_none=True)
        loss = fn(*w)[0]
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in tr.wm.net.parameters()])
        tr.opt_wm.zero_grad(set_to_none=True)
        return loss.detach(), g

    t_wm = time_wm()
    print(f'default-compiled update_wm {t_wm:.1f} ms', flush=True)
    s0, A, S1, xi, mask = tr.buf.sample_windows(cfg.wm_batch, cfg.k)
    w = (s0, A, S1, xi, mask.float())
    default_loss = tr._wm_loss
    L0, g0 = wm_loss_grads(default_loss, w)
    for mode in args.wm_modes:
        fn = torch.compile(tr.wm_window_loss, dynamic=False, mode=mode)
        t0 = time.time()
        wm_loss_grads(fn, w)
        compile_s = time.time() - t0
        L1, g1 = wm_loss_grads(fn, w)
        rel = ((L1 - L0).abs() / L0.abs()).item()
        cos = torch.nn.functional.cosine_similarity(g0, g1, dim=0).item()
        ratio = (g1.norm() / g0.norm()).item()
        good = rel < 2e-3 and cos > 0.995 and abs(ratio - 1) < 0.02
        ok &= good
        tr._wm_loss = fn
        t = time_wm()
        tr._wm_loss = default_loss
        print(f'wm {mode:25s} loss rel err {rel:.1e}  grad cos {cos:.5f}  norm ratio {ratio:.4f}  '
              f'{"OK" if good else "FAIL"} | compile {compile_s:.0f}s | update_wm {t:.1f} ms ({t_wm / t:.2f}x)',
              flush=True)  # fmt: skip

    # 5. trajectory gate
    import copy

    mods = (tr.wm, tr.policy, tr.critic, tr.opt_wm, tr.opt_pi, tr.opt_v)
    snap = [copy.deepcopy(m.state_dict()) for m in mods]

    def trajectory(img_step, wm_loss):
        for m, sd in zip(mods, snap):
            m.load_state_dict(copy.deepcopy(sd))
        for o in (tr.opt_wm, tr.opt_pi, tr.opt_v):  # fresh Adam: loaded momentum swamped 5 steps of new gradients
            o.state.clear()
        tr._img_step, tr._wm_loss = img_step, wm_loss
        tr.task.obs_noise.zero_()
        torch.manual_seed(3)
        for _ in range(args.traj_updates):
            tr.update_wm()
            tr.update_policy()
        tr.task.obs_noise.copy_(noise)
        tr._img_step, tr._wm_loss = eager_step, default_loss
        return torch.cat([p.detach().flatten() for p in list(tr.wm.net.parameters()) + list(tr.policy.parameters())])

    for m, sd in zip(mods, snap):
        m.load_state_dict(copy.deepcopy(sd))
    p0 = torch.cat([p.detach().flatten() for p in list(tr.wm.net.parameters()) + list(tr.policy.parameters())])
    d_ref = trajectory(eager_step, default_loss) - p0
    for cand in args.traj:
        kind, mode = cand.split(':')
        step = torch.compile(tr.img_step, dynamic=False, mode=mode) if kind == 'bptt' else eager_step
        wml = torch.compile(tr.wm_window_loss, dynamic=False, mode=mode) if kind == 'wm' else default_loss
        trajectory(step, wml)  # compile + warm-up (cudagraph trees record on later calls)
        d = trajectory(step, wml) - p0
        cos = torch.nn.functional.cosine_similarity(d, d_ref, dim=0).item()
        ratio = (d.norm() / d_ref.norm()).item()
        good = cos > 0.99 and abs(ratio - 1) < 0.05
        ok &= good
        print(f'trajectory {cand:24s} {args.traj_updates} updates: update cos {cos:.5f}  norm ratio {ratio:.4f}  '
              f'{"OK" if good else "FAIL"}', flush=True)  # fmt: skip
    print('bptt compile checks', 'passed' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
