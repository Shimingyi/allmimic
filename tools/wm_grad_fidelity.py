"""World-model gradient fidelity: WM vs simulator directional derivatives of the H-step return w.r.t. the actions.

CHANGES #23 (why the WM can be exploited where it has no data).

  PYTHONPATH=. python tools/wm_grad_fidelity.py --lib data/library_eval/lafan1_seg10.npz \
      runs/bones/ckpt_0032000.pt runs/bones/ckpt_0048000.pt     # GPU

Port of the gradient-fidelity probe of the supermimic_vibed reimplementation (experiments/ablation/fidelity.py;
Pearson 0.97 / 0.87 / 0.09 at H 1 / 8 / 32 there). Per checkpoint and bank mode (--banks):
  bank   --states start frames (random, room for prefix + H before the clip end), --prefix steps of the checkpoint's own
         policy in the deployment env (eval mode, fp32), deterministic (det, as deployed) or with N(0, cfg.action_noise)
         on the executed action (noisy, as Trainer.collect); a state that hard-terminated in the prefix is dropped.
         H more steps in the same mode give the nominal action sequence a (H, 29) (policy outputs).
  WM     J(a) = sum_t gamma^t c_t r_t through the frozen WM with a held fixed (open loop: no policy, no critic), step by
         step as Trainer.img_step / rollout (setpoint_action, analytics_fast, c = soft survival); dJ/da by autograd.
  sim    central differences (J(a + eps u) - J(a - eps u)) / 2 eps along --dirs random unit u in (H, 29) space (+ one
         along the WM gradient), from the restored bank state, same reward. The sim never resets inside a rollout;
         termination only weights the return: soft = the WM's sigmoid survival, hard = the 0/1 task termination.
  resolved  |return change| > 5x the repeat noise and the estimate within 25% of an adjacent epsilon's.
Restore: every input of mjwarp.step besides ctrl (qpos, qvel, act, qacc_warmstart, time, applied forces) is copied back;
replay_err = live trajectory vs restored replay, repeat_err = two restored replays (the GPU nondeterminism floor).
wm_err1 / wm_errH: teacher-forced one-step / open-loop H-step WM error along the replayed nominal trajectory, in the
units of the training log (mean (state_error / out_norm.std)^2), over steps before a hard termination.
"""

import argparse
import json
import os

import torch

from allmimic import networks
from allmimic.evaluate import load_supermimic_policy, make_eval_env
from allmimic.motion_lib import MotionLibrary
from allmimic.networks import WorldModel, apply_delta, apply_delta_tri, run_mlp, state_error
from allmimic.task import NJ, XI_DIM, flat_to_state, state_to_flat, task_cfg

NOISE_MULT, ABS_FLOOR, REL_TOL = 5.0, 1e-7, 0.25  # resolved: signal > max(5 x repeat noise, 1e-7), adjacent eps +-25%

# Math (CPU checks: tests/test_wm_grad_fidelity.py).


def disc_return(r, d, gamma):
    """(T,B) rewards, (T,B) termination in [0, 1] -> (B,) sum_t gamma^t prod_{j<t} (1 - d_j) r_t (Trainer.rollout's J)"""
    alive = torch.cumprod(torch.cat([torch.ones_like(d[:1]), 1 - d[:-1]]), 0)
    return ((gamma ** torch.arange(len(r), dtype=r.dtype, device=r.device))[:, None] * alive * r).sum(0)


def unit_directions(n, k, h, gen, device='cpu'):
    """(n, k, h, NJ) Gaussian directions, unit L2 norm over each (h, NJ) action sequence"""
    u = torch.randn(n, k, h, NJ, generator=gen, dtype=torch.float64).float().to(device)
    return u / u.flatten(2).norm(dim=-1)[..., None, None]


def resolved(deriv, delta, noise):
    """(E,P) FD derivatives and return changes (epsilons increasing), (P,) repeat noise -> (E,P) bool: signal above the
    noise and within REL_TOL of an adjacent epsilon's estimate (supermimic_vibed fidelity_math.reference_quality)"""
    ok = torch.isfinite(deriv) & (delta.abs() > (NOISE_MULT * noise).clamp_min(ABS_FLOOR))
    scale = torch.maximum(deriv[:-1].abs(), deriv[1:].abs()).clamp_min(1e-12)
    pair = ok[:-1] & ok[1:] & ((deriv[:-1] - deriv[1:]).abs() <= REL_TOL * scale)
    out = torch.zeros_like(ok)
    out[:-1] |= pair
    out[1:] |= pair
    return out


def agreement(fd, pred, keep):
    """FD (reference) and WM (predicted) derivatives -> n, resolved share, sign agreement and Pearson over kept pairs"""
    keep = keep & torch.isfinite(fd) & torch.isfinite(pred)
    a, b = fd[keep].double(), pred[keep].double()
    out = {'n': keep.numel(), 'resolved': keep.double().mean().item() if keep.numel() else 0.0}
    out['sign'] = (torch.sign(a) == torch.sign(b)).double().mean().item() if len(a) else None
    out['pearson'] = torch.corrcoef(torch.stack([a, b]))[0, 1].item() if len(a) >= 3 and a.std() * b.std() > 0 else None
    return out


def fd_jobs(u, eps):
    """(n, D, h, 29) directions -> perturbations (J, h, 29) and bank state (J,) of the sim rollouts: base, repeat, then
    (eps, +/-, direction) blocks of the n states"""
    n, _, h, nj = u.shape
    pert = torch.stack([sg * e * u for e in eps for sg in (1, -1)]).transpose(1, 2).reshape(-1, h, nj)
    pert = torch.cat([u.new_zeros(2 * n, h, nj), pert])
    return pert, torch.arange(n, device=u.device).repeat(len(pert) // n)


def fd_estimates(J, n, eps):
    """(J,) returns of the fd_jobs rollouts -> repeat noise (n,), return changes and central differences (E, D, n)"""
    pm = J[2 * n :].view(len(eps), 2, -1, n)
    delta = pm[:, 0] - pm[:, 1]
    return (J[n : 2 * n] - J[:n]).abs(), delta, delta / (2 * J.new_tensor(eps).view(-1, 1, 1))


def step_reward(an, objective, dt):
    """'full' task reward, or 'state' without the action-rate term (its action gradient is analytic on both sides)"""
    if objective == 'full':
        return an['r']
    return sum(v for k, v in an['terms'].items() if k != 'action_rate') * dt


# World model.


def wm_step(wm, s, feats, a_sp, xi):
    """the WM update of Trainer.img_step (eager apply_delta)"""
    d = run_mlp(wm.net, wm.in_norm(torch.cat([feats, a_sp, xi], dim=-1))) * wm.out_norm.std + wm.out_norm.mean
    return (apply_delta if wm.rot_out == 'axis' else apply_delta_tri)(s, d)


def wm_rollout(wm, task, s, acts, a_prev, xi, ref, ref_kin, off, gamma, objective='full'):
    """open-loop Trainer.rollout without critic: (B,H,29) fixed policy outputs -> (J (B,) soft-survival discounted
    return, predicted states [s_1 .. s_H]); ref (B,T,..) / ref_kin windows start at the bank frame (ref_window)"""

    def at(t):
        return (
            {k: v[:, t] for k, v in ref.items()},
            tuple(x[:, t] for x in ref_kin),
            {k: v[:, t + off] for k, v in ref.items()},
        )

    bias = xi[:, 1 : 1 + NJ]
    an = task.analytics_fast(s, *at(0), a_prev, a_prev, bias)
    rs, ds, preds = [], [], []
    for t in range(acts.shape[1]):
        a = acts[:, t]
        ref1, kin1, fut1 = at(t + 1)
        s = wm_step(wm, s, an['feats'], task.setpoint_action(a, ref1['q']), xi)
        an = task.analytics_fast(s, ref1, kin1, fut1, a, a_prev, bias)
        rs.append(step_reward(an, objective, task.cfg.dt))
        ds.append(an['soft'])
        preds.append(s)
        a_prev = a
    return disc_return(torch.stack(rs), torch.stack(ds), gamma), preds


def wm_err(wm, pred, true):
    """WM state error in the training log's units: mean over the 70 dims of (state_error / out_norm.std)^2"""
    return (state_error(pred, true) / wm.err_std).pow(2).mean(-1)


def ref_window(env, frame0, h):
    """reference frames frame0 + 0 .. h + last goal offset, clamped to the clip, and their body kinematics (as
    Trainer.imagination_inputs)"""
    steps = torch.arange(0, h + int(env.future_off[-1]) + 1, device=frame0.device)
    ref = env.lib.frame(env.lib.clamp_to_clip(frame0[:, None] + steps, frame0))
    return ref, env.kin.forward(ref['pos'], ref['rot'], ref['q'], ref['v'], ref['w'], ref['qd'])


# Simulator.


def phys_fields(env):
    """mjwarp Data fields mjwarp.step reads besides ctrl (written before every step) and the model (eval: nominal xi)"""
    return ['qpos', 'qvel', 'qacc_warmstart', 'time', 'qfrc_applied', 'xfrc_applied'] + ['act'] * (env.mj_model.na > 0)


def restore(env, snap, b):
    """world w <- bank state b[w]: plain copies, no reset_data (it zeroes the solver warmstart)"""
    for k, v in snap.items():
        getattr(env.sim.data, k)[:] = v[b]


def sim_step(env, a, frame, a_prev):
    """TrackingSimEnv.step without bookkeeping, noise or resets (a fallen robot keeps going) -> state, analytics, finite"""
    ref = env.lib.frame(frame + 1)
    a_sp = env.task.setpoint_action(a, ref['q'])
    env.sim.data.ctrl[:] = (env.action_scale * a_sp + env.q0 - env.enc_bias)[:, env.act_to_joint]
    for _ in range(env.decimation):
        env.sim.step()
    s, finite = env._sanitize(env.read_state())
    ref_kin = env.kin_c(ref['pos'], ref['rot'], ref['q'], ref['v'], ref['w'], ref['qd'])
    an = env.task.analytics_fast(s, ref, ref_kin, env.ref_future(frame + 1), a, a_prev, env.enc_bias)
    return s, an, finite


@torch.no_grad()
def collect_bank(env, pol, n, prefix, h, noise, seed):
    """n start states after `prefix` policy steps, then h more steps in the same mode -> bank (worlds >= n run copies of
    the first n starts and are ignored)"""
    lib, dev, m = env.lib, env.device, env.num_envs
    g = torch.Generator().manual_seed(seed)
    ok = (lib.lengths > prefix + h + 2).nonzero().squeeze(-1).cpu()
    assert len(ok), f'no clip longer than prefix + H + 2 = {prefix + h + 2} frames'
    clips = ok[torch.randperm(len(ok), generator=g)[torch.arange(n) % len(ok)]].to(dev)
    f0 = lib.starts[clips] + (torch.rand(n, generator=g).to(dev) * (lib.lengths[clips] - prefix - h - 2)).long()
    ids = torch.arange(m, device=dev)
    env.reset_envs(ids, frames=f0[ids % n])
    env._finish_resets(ids, force=True)
    hist, goal, frame, a_prev = env.obs_hist, env.goal_buf, env.frame.clone(), env.prev_action.clone()
    fell = torch.zeros(m, dtype=torch.bool, device=dev)
    acts, live = [], []
    for t in range(prefix + h):
        if t == prefix:
            snap = {k: getattr(env.sim.data, k)[:n].clone() for k in phys_fields(env)}
            valid = ~fell[:n]
            live.append(state_to_flat(env.read_state())[:n])
            bank = {'phys': snap, 's0': live[0], 'frame': frame[:n], 'a_prev': a_prev[:n], 'xi': env.xi[:n].clone()}
        a = pol(hist.flatten(1), goal)
        if noise:
            a = a + noise * torch.randn_like(a)
        s, an, finite = sim_step(env, a, frame, a_prev)
        fell |= an['hard'] | ~finite
        hist = torch.cat([hist[:, 1:], an['obs'].unsqueeze(1)], dim=1)  # eval env: no observation noise
        goal, frame, a_prev = an['goal'], frame + 1, a
        if t >= prefix:
            acts.append(a[:n])
            live.append(state_to_flat(s)[:n])
    return {**bank, 'acts': torch.stack(acts, 1), 'live': torch.stack(live), 'valid': valid, 'fell': fell[:n] & valid}


@torch.no_grad()
def sim_jobs(env, bank, b, acts, gamma, objective):
    """open-loop rollouts of acts (J,h,29) from bank states b (J,), env.num_envs at a time -> soft / hard returns (J,)
    (NaN if the world diverged) and the first chunk's states (h+1, M, 71), hard flags (h, M), restore error"""
    m, nj = env.num_envs, len(b)
    out = {'soft': [], 'hard': []}
    for c0 in range(0, nj, m):
        idx = torch.arange(c0, c0 + m, device=b.device).clamp(max=nj - 1)  # pad the last chunk with the last job
        bi = b[idx]
        restore(env, bank['phys'], bi)
        s = env.read_state()
        frame, a_prev, ok = bank['frame'][bi], bank['a_prev'][bi], torch.ones(m, dtype=torch.bool, device=b.device)
        xs, rs, ds, dh = [state_to_flat(s)], [], [], []
        for t in range(acts.shape[1]):
            a = acts[idx, t]
            s, an, finite = sim_step(env, a, frame, a_prev)
            ok &= finite
            rs.append(step_reward(an, objective, env.tcfg.dt))
            ds.append(an['soft'])
            dh.append((an['hard'] | ~finite).float())
            frame, a_prev = frame + 1, a
            if c0 == 0:
                xs.append(state_to_flat(s))
        r, nan = torch.stack(rs), torch.tensor(float('nan'), device=b.device)
        out['soft'].append(torch.where(ok, disc_return(r, torch.stack(ds), gamma), nan)[: nj - c0])
        out['hard'].append(torch.where(ok, disc_return(r, torch.stack(dh), gamma), nan)[: nj - c0])
        if c0 == 0:
            first = {'states': torch.stack(xs), 'hard_t': torch.stack(dh)}
            first['restore_err'] = (xs[0] - bank['s0'][bi]).abs().max().item()
    return torch.cat(out['soft']), torch.cat(out['hard']), first


def probe(env, wm, bank, h, args, gamma):
    """one horizon: WM autograd vs sim FD directional derivatives -> metrics dict"""
    task, dev, n, eps = env.task, env.device, len(bank['frame']), args.eps
    E, K, v = len(eps), args.dirs, bank['valid']
    ref, ref_kin = ref_window(env, bank['frame'], h)
    a = bank['acts'][:, :h].clone().requires_grad_(True)
    off = env.future_off
    J_wm, preds = wm_rollout(
        wm, task, flat_to_state(bank['s0']), a, bank['a_prev'], bank['xi'], ref, ref_kin, off, gamma, args.objective
    )
    (grad,) = torch.autograd.grad(J_wm.sum(), a)
    gn = grad.flatten(1).norm(dim=-1)
    u = unit_directions(n, K, h, torch.Generator().manual_seed(args.seed + h), dev)
    u = torch.cat([u, (grad / gn.clamp_min(1e-12)[:, None, None])[:, None]], 1)  # (n, K+1, h, 29)
    pred = (grad[:, None] * u).sum((2, 3)).T  # (K+1, n)
    pert, b = fd_jobs(u, eps)
    J_soft, J_hard, first = sim_jobs(env, bank, b, bank['acts'][b, :h] + pert, gamma, args.objective)
    with torch.no_grad():
        S = first['states'][:, :n]  # base replay (h+1, n, 71)
        hard = first['hard_t'][:, :n]
        alive = torch.cumprod(torch.cat([torch.ones_like(hard[:1]), 1 - hard[:-1]]), 0)  # s_t not terminated
        st, st1 = flat_to_state(S[:-1].flatten(0, 1)), flat_to_state(S[1:].flatten(0, 1))  # time-major (h * n)
        a_sp = task.setpoint_action(
            a.detach().transpose(0, 1).flatten(0, 1), ref['q'][:, 1 : h + 1].transpose(0, 1).flatten(0, 1)
        )
        e1 = wm_err(wm, wm_step(wm, st, task.state_features(st), a_sp, bank['xi'].repeat(h, 1)), st1).view(h, n)
        w1 = alive * v
        eH = wm_err(wm, {k: x.detach() for k, x in preds[-1].items()}, flat_to_state(S[-1]))
        keepH = v & (alive[-1] > 0)
        rec = {
            'wm_err1': ((e1 * w1).sum() / w1.sum().clamp_min(1)).item(),
            'wm_errH': eH[keepH].mean().item() if keepH.any() else None,
            'restore_err': first['restore_err'],
            'replay_err_max': (S - bank['live'][: h + 1]).abs().amax((0, 2))[v].max().item(),
            'replay_err_median': (S - bank['live'][: h + 1]).abs().amax((0, 2))[v].median().item(),
            'repeat_err_max': (first['states'][:, n : 2 * n] - S).abs().amax((0, 2))[v].max().item(),
            'grad_norm_median': gn[v].median().item(),
            'wm_return_mean': J_wm.detach()[v].mean().item(),
            'sim_return_soft_mean': J_soft[:n][v].mean().item(),
            'sim_return_hard_mean': J_hard[:n][v].mean().item(),
        }
        for name, Jx in (('soft', J_soft), ('hard', J_hard)):
            noise, delta, fd = fd_estimates(Jx, n, eps)
            res = resolved(fd.flatten(1), delta.flatten(1), noise.repeat(K + 1)).view(E, K + 1, n)
            rows, pv = [], pred[:, v]
            for i, e in enumerate(eps):
                fi, ri = fd[i][:, v], res[i][:, v]  # (K+1, valid states); x[i, :K, v] would put the mask dim first
                row = {'eps': e, **agreement(fi[:K], pv[:K], ri[:K])}
                row['pearson_all'] = agreement(fi[:K], pv[:K], torch.ones_like(ri[:K]))['pearson']
                g_fd, g_ok = fi[K], ri[K]  # along the WM gradient: does the true return go up?
                row['grad_dir_up'] = (g_fd[g_ok] > 0).double().mean().item() if g_ok.any() else None
                row['grad_dir_ratio'] = (g_fd[g_ok] / pv[K][g_ok]).median().item() if g_ok.any() else None
                rows.append(row)
            rec[name] = {'repeat_noise_median': noise[v].median().item(), 'rows': rows}
    return rec


def report(rec):
    def f(x, w=6, p=3):
        return f'{x:{w}.{p}f}' if x is not None else ' ' * (w - 1) + '-'

    print(f'{rec["ckpt"]}  bank {rec["bank"]}  objective {rec["objective"]}: {rec["valid"]}/{rec["states"]} states '
          f'valid, {rec["fell_in_window"]:.2f} fall inside the window', flush=True)  # fmt: skip
    print('     eps |  soft: res   sign  pearson  r_all  g-up  g-ratio |  hard: res   sign  pearson  r_all')
    for h, x in rec['horizons'].items():
        print(f'  H {h}: wm_err1 {x["wm_err1"]:.4f}  wm_err{h} {f(x["wm_errH"], 1, 4)}  J wm {x["wm_return_mean"]:.3f} '
              f'sim {x["sim_return_soft_mean"]:.3f}  |g| {x["grad_norm_median"]:.3g}  restore {x["restore_err"]:.0e}  '
              f'replay {x["replay_err_max"]:.0e}  repeat {x["repeat_err_max"]:.0e}')  # fmt: skip
        for s, d in zip(x['soft']['rows'], x['hard']['rows']):
            print(f'  {s["eps"]:6.3f} |  {f(s["resolved"])} {f(s["sign"])} {f(s["pearson"], 8)} {f(s["pearson_all"])} '
                  f'{f(s["grad_dir_up"])} {f(s["grad_dir_ratio"], 8, 2)} |  {f(d["resolved"])} {f(d["sign"])} '
                  f'{f(d["pearson"], 8)} {f(d["pearson_all"])}')  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('ckpts', nargs='+', help='checkpoints of runs with one task cfg (the env is shared)')
    ap.add_argument('--lib', required=True)
    ap.add_argument('--banks', nargs='+', default=['det', 'noisy'], choices=['det', 'noisy'])
    ap.add_argument('--states', type=int, default=64)
    ap.add_argument(
        '--prefix', type=int, default=50, help='policy steps before the probe (a collapsed policy falls after ~1.5 s)'
    )
    ap.add_argument('--horizons', type=int, nargs='+', default=[1, 8, 32])
    ap.add_argument('--dirs', type=int, default=8, help='random unit directions per state (+1 along the WM gradient)')
    ap.add_argument('--eps', type=float, nargs='+', default=[0.003, 0.01, 0.03, 0.1])
    ap.add_argument('--objective', default='full', choices=['full', 'state'])
    ap.add_argument('--worlds', type=int, default=4096, help='simulator worlds per batch of FD rollouts')
    ap.add_argument('--seed', type=int, default=0, help='start frames and directions (same across checkpoints)')
    ap.add_argument('--out', default='', help='directory for the JSON files (default: next to each checkpoint)')
    ap.add_argument('--device', default='cuda', help='cpu runs mujoco_warp on the warp CPU backend (smoke tests)')
    args = ap.parse_args(argv)
    if args.eps != sorted(set(args.eps)) or len(args.eps) < 2:
        raise SystemExit('--eps needs >= 2 increasing values')
    dev = args.device
    networks.AMP['enabled'] = False  # fp32 MLPs (deployment); TF32 stays off (torch default) for clean differences
    lib = MotionLibrary(args.lib, dev)
    n = args.states
    jobs = n * (2 + 2 * len(args.eps) * (args.dirs + 1))
    chunks = -(-jobs // max(args.worlds, 2 * n))
    m = max(2 * n, -(-jobs // chunks))  # balanced chunks; base and repeat rollouts land in the first one
    env = tcfg0 = None
    for path in args.ckpts:
        pol, ck = load_supermimic_policy(path, dev)
        tcfg = task_cfg(ck['cfg'], train=False)
        if env is None:
            env, tcfg0 = make_eval_env(lib, m, dev, tcfg), tcfg
            env.reset()
        if tcfg != tcfg0:
            raise SystemExit(f'{path}: task cfg differs from the first checkpoint; probe it in a separate call')
        feat_dim = env.task.state_features(env.state).shape[-1]
        wm = WorldModel(
            env.task,
            feat_dim,
            XI_DIM,
            norm_eps=ck['cfg'].get('wm_norm_eps', 1e-3),
            rot_out=ck['cfg'].get('wm_rot_out', 'axis'),
        ).to(dev)
        wm.load_state_dict(ck['wm'])
        wm.requires_grad_(False)
        gamma = ck['cfg'].get('gamma', 0.99)
        for mode in args.banks:
            noise = ck['cfg'].get('action_noise', 0.1) if mode == 'noisy' else 0.0
            bank = collect_bank(env, pol, n, args.prefix, max(args.horizons), noise, args.seed)
            v = bank['valid']
            if v.sum() < 3:
                raise SystemExit(f'{path} {mode}: {int(v.sum())} of {n} states survived the prefix; lower --prefix')
            rec = {'ckpt': path, 'lib': args.lib, 'bank': mode, 'action_noise': noise, 'objective': args.objective,
                   'states': n, 'valid': int(v.sum()), 'fell_in_window': bank['fell'][v].float().mean().item(),
                   'prefix': args.prefix, 'dirs': args.dirs, 'eps': args.eps, 'gamma': gamma, 'worlds': m,
                   'seed': args.seed, 'termination': 'no resets in the sim; soft = sigmoid survival (WM objective), '
                   'hard = 0/1 task termination; the WM side always uses soft', 'horizons': {}}  # fmt: skip
            for h in args.horizons:
                rec['horizons'][h] = probe(env, wm, bank, h, args, gamma)
            report(rec)
            stem = os.path.splitext(os.path.basename(path))[0]
            dst = args.out or os.path.dirname(path)
            os.makedirs(dst or '.', exist_ok=True)
            dst = os.path.join(dst, f'{stem}_wm_grad_fidelity_{mode}_{args.objective}.json')
            with open(dst, 'w') as fh:
                json.dump(rec, fh, indent=1)
            print(f'  -> {dst}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
