"""CPU checks of tools/wm_grad_fidelity.py: discounted return, directional derivative of a linear 'WM' against its
analytic value and central differences, FD job layout, resolution screen and agreement metrics, and the open-loop WM
objective against Trainer.rollout (real G1 kinematics and analytics, small random WM, residual actions at scale 2).

  PYTHONPATH=. python tests/test_wm_grad_fidelity.py     # needs mjlab (CPU MjModel of the G1)
"""

import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from allmimic import train
from allmimic.networks import WorldModel, apply_delta
from allmimic.rotations import axis_angle_to_mat
from allmimic.task import NJ, OBS_DIM, XI_DIM, Task, flat_to_state, state_to_flat, task_cfg
from tools.wm_grad_fidelity import (
    agreement,
    disc_return,
    fd_estimates,
    fd_jobs,
    resolved,
    unit_directions,
    wm_rollout,
)

F64 = {'dtype': torch.float64}


def check_disc_return():
    g = torch.Generator().manual_seed(0)
    r, d = torch.rand(9, 5, generator=g, **F64), torch.rand(9, 5, generator=g, **F64)
    J, c = torch.zeros(5, **F64), torch.ones(5, **F64)
    for t in range(9):  # Trainer.rollout's accumulation
        J = J + 0.97**t * c * r[t]
        c = c * (1 - d[t])
    assert torch.allclose(disc_return(r, d, 0.97), J, rtol=1e-12)
    u = unit_directions(4, 3, 6, torch.Generator().manual_seed(5))
    assert u.shape == (4, 3, 6, NJ) and torch.allclose(u.flatten(2).norm(dim=-1), torch.ones(4, 3))
    assert torch.equal(u, unit_directions(4, 3, 6, torch.Generator().manual_seed(5)))
    print('  discounted return (= Trainer.rollout accumulation), unit directions ok')


def linear_wm(A, Bm, w, v, s0, gamma, soft=None):
    """'WM' s' = A s + Bm a, reward w.s' + v.a, termination 0 or sigmoid(soft.s' - 1) -> J(actions (B,H,NJ)) (B,)"""

    def J(acts):
        s, rs, ds = s0, [], []
        for t in range(acts.shape[1]):
            s = s @ A.T + acts[:, t] @ Bm.T
            rs.append(s @ w + acts[:, t] @ v)
            ds.append(torch.zeros_like(rs[-1]) if soft is None else torch.sigmoid(s @ soft - 1))
        return disc_return(torch.stack(rs), torch.stack(ds), gamma)

    return J


def directional(J, acts, u):
    """autograd directional derivatives (B, K) of J at acts along u (B, K, H, NJ)"""
    a = acts.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(J(a).sum(), a)
    return (grad[:, None] * u).sum((2, 3))


def central(J, acts, u, e):
    return torch.stack([(J(acts + e * u[:, k]) - J(acts - e * u[:, k])) / (2 * e) for k in range(u.shape[1])], 1)


def check_linear_wm():
    g = torch.Generator().manual_seed(1)
    B, H, D, K, eps = 6, 8, 5, 4, [1e-3, 1e-2, 1e-1, 1.0]
    A = torch.eye(D, **F64) + 0.1 * torch.randn(D, D, generator=g, **F64)
    Bm, w, v = 0.1 * torch.randn(D, NJ, generator=g, **F64), torch.randn(D, generator=g, **F64), torch.randn(NJ, **F64)
    s0, acts = torch.randn(B, D, generator=g, **F64), 0.3 * torch.randn(B, H, NJ, generator=g, **F64)
    u = unit_directions(B, K, H, g).double()
    # linear: autograd == analytic (the response to u from a zero start) == central differences at any epsilon
    J, J0 = linear_wm(A, Bm, w, v, s0, 0.99), linear_wm(A, Bm, w, v, torch.zeros_like(s0), 0.99)
    pred = directional(J, acts, u)
    assert torch.allclose(pred, torch.stack([J0(u[:, k]) for k in range(K)], 1), rtol=1e-10, atol=1e-12)
    fd = torch.stack([central(J, acts, u, e) for e in eps])  # (E, B, K)
    assert torch.allclose(fd, pred.expand_as(fd), rtol=1e-8, atol=1e-10)
    delta = 2 * fd * torch.tensor(eps, **F64).view(-1, 1, 1)
    res = resolved(fd.flatten(1), delta.flatten(1), torch.zeros(B * K, **F64))
    m = agreement(fd[0].flatten(), pred.flatten(), res[0])
    assert res.all() and m['resolved'] == 1 and m['sign'] == 1 and abs(m['pearson'] - 1) < 1e-12
    # smooth termination: central-difference error O(eps^2), ~100x smaller per 10x smaller epsilon
    Js = linear_wm(A, Bm, w, v, s0, 0.99, soft=0.5 * torch.randn(D, generator=g, **F64))
    pred = directional(Js, acts, u)
    err = [(central(Js, acts, u, e) - pred).abs().max().item() for e in (1e-2, 1e-3)]
    assert err[0] > 1e-9 and err[1] < 0.02 * err[0], err
    print(f'  linear WM ok (exact FD at every eps; with termination FD error {err[0]:.1e} -> {err[1]:.1e})')


def check_fd_layout():
    """fd_jobs / fd_estimates on a 'simulator' whose return is linear in the actions, c_b . a, per bank state b"""
    g = torch.Generator().manual_seed(2)
    n, D, h, eps = 3, 4, 5, [0.01, 0.1, 1.0]
    c, a0 = torch.randn(n, h, NJ, generator=g, **F64), torch.randn(n, h, NJ, generator=g, **F64)
    u = unit_directions(n, D, h, g).double()
    pert, b = fd_jobs(u, eps)
    assert len(pert) == n * (2 + 2 * len(eps) * D) and torch.equal(b[:n], torch.arange(n))
    Jx = (c[b] * (a0[b] + pert)).sum((1, 2))
    Jx[n : 2 * n] += torch.tensor([0.0, 1e-3, -2e-3], **F64)  # repeat noise
    noise, delta, fd = fd_estimates(Jx, n, eps)
    want = (c[:, None] * u).sum((2, 3)).T  # (D, n)
    assert torch.allclose(noise, torch.tensor([0.0, 1e-3, 2e-3], **F64), atol=1e-12)
    assert torch.allclose(fd, want.expand_as(fd), rtol=1e-9, atol=1e-11)
    assert torch.allclose(delta, 2 * want * torch.tensor(eps, **F64).view(-1, 1, 1), rtol=1e-9, atol=1e-11)
    print('  FD job layout ok')


def check_resolved():
    nan = float('nan')
    cases = [  # derivative at the 4 epsilons, repeat noise, expected
        ([1, 1, 1, 1], 0.0, [1, 1, 1, 1]),
        ([1, 1, 1, 5], 0.0, [1, 1, 1, 0]),  # largest eps off
        ([1, 3, 2, 2.1], 0.0, [0, 0, 1, 1]),  # only the two largest agree
        ([1, 1.2, 2, 3], 0.0, [1, 1, 0, 0]),  # 1 vs 1.2 within 25 %, 1.2 vs 2 and 2 vs 3 not
        ([nan, 1, 1, 1], 0.0, [0, 1, 1, 1]),
        ([1e-6] * 4, 0.0, [0, 0, 0, 0]),  # return change below 1e-7 except at the largest eps: no pair
        ([1, 1, 1, 1], 0.05, [0, 0, 0, 0]),  # every change below 5x the repeat noise
    ]
    eps = torch.tensor([0.003, 0.01, 0.03, 0.1], **F64)
    deriv = torch.tensor([c[0] for c in cases], **F64).T  # (E, P)
    res = resolved(deriv, 2 * eps[:, None] * deriv, torch.tensor([c[1] for c in cases], **F64))
    assert torch.equal(res, torch.tensor([c[2] for c in cases], dtype=torch.bool).T), res.T.int()
    print('  resolved screen ok')


def check_agreement():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(50, generator=g, **F64)
    y = 0.5 * x + 0.5 * torch.randn(50, generator=g, **F64)
    keep = torch.ones(50, dtype=torch.bool)
    m = agreement(x, y, keep)
    assert abs(m['pearson'] - np.corrcoef(x.numpy(), y.numpy())[0, 1]) < 1e-12 and m['n'] == 50
    assert m['sign'] == ((x > 0) == (y > 0)).double().mean().item() and m['resolved'] == 1
    m = agreement(x, -2 * x, keep)
    assert m['sign'] == 0 and abs(m['pearson'] + 1) < 1e-12
    part = keep.clone()
    part[10:] = False
    m = agreement(x, y, part)
    assert m['resolved'] == 0.2 and abs(m['pearson'] - np.corrcoef(x[:10].numpy(), y[:10].numpy())[0, 1]) < 1e-12
    xn = x.clone()
    xn[0] = float('nan')
    assert agreement(xn, y, keep)['resolved'] == 49 / 50
    assert agreement(x[:2], y[:2], keep[:2])['pearson'] is None
    m = agreement(x, y, torch.zeros(50, dtype=torch.bool))
    assert m['sign'] is None and m['pearson'] is None and m['resolved'] == 0
    print('  agreement metrics ok')


def g1_task(tcfg):
    """Task on the real G1 kinematic tree (CPU MjModel; q0 = 0 and action scale 0.4 stand in for the robot cfg)"""
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    from allmimic.kinematics import FastTreeKinematics

    cfg = SceneCfg(terrain=TerrainEntityCfg(terrain_type='plane'), num_envs=1, entities={'robot': get_g1_robot_cfg()})
    m = Scene(cfg, device='cpu').compile()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).split('/')[-1] for b in range(m.nbody)]
    root = names.index('pelvis')
    hinge = [m.body_jntadr[root + 1 + i] for i in range(NJ)]
    lim = torch.tensor(m.jnt_range[hinge], dtype=torch.float32)
    task = Task(FastTreeKinematics(m, 'cpu', root_body=root), torch.zeros(NJ), lim, torch.full((NJ,), 0.4), tcfg, 'cpu')
    task.set_joint_names([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).split('/')[-1] for j in hinge])
    return task


def rand_states(g, *shape):
    return {
        'pos': torch.tensor([0.0, 0.0, 0.78]) + 0.05 * torch.randn(*shape, 3, generator=g),
        'rot': axis_angle_to_mat(0.1 * torch.randn(*shape, 3, generator=g)),
        'q': 0.2 * torch.randn(*shape, NJ, generator=g),
        'v': 0.3 * torch.randn(*shape, 3, generator=g),
        'w': 0.3 * torch.randn(*shape, 3, generator=g),
        'qd': 0.5 * torch.randn(*shape, NJ, generator=g),
    }


def check_wm_objective():
    torch.manual_seed(0)
    train.apply_delta_c = apply_delta  # eager on CPU (img_step uses the compiled copy of the same function)
    task = g1_task(task_cfg({'action': 'res', 'res_scale': 2.0}, train=False))
    g = torch.Generator().manual_seed(4)
    B, H, off = 6, 5, torch.tensor(task.cfg.offsets)
    s_flat = state_to_flat(rand_states(g, B))
    s0 = flat_to_state(s_flat)  # as the trainer unpacks it (the flat round trip moves rot by up to ~1e-4)
    wm = WorldModel(task, task.state_features(s0).shape[-1], XI_DIM, hidden=(64, 64))
    with torch.no_grad():
        wm.net[-1].weight.mul_(10)  # state changes ~0.1 per step instead of ~0.01
    wm.requires_grad_(False)
    ref = rand_states(g, B, H + int(off[-1]) + 1)
    ref_kin = task.kin.forward(ref['pos'], ref['rot'], ref['q'], ref['v'], ref['w'], ref['qd'])
    xi = torch.cat([torch.ones(B, 1), 0.01 * torch.randn(B, NJ + 3, generator=g)], -1)
    a_prev, acts = 0.3 * torch.randn(B, NJ, generator=g), 0.3 * torch.randn(B, H, NJ, generator=g)

    def J(a, objective='full'):
        return wm_rollout(wm, task, s0, a, a_prev, xi, ref, ref_kin, off, 0.99, objective)[0]

    # Trainer.rollout without critic, with a policy that replays the same actions: same return and action gradient
    a1, a2 = acts.clone().requires_grad_(True), acts.clone().requires_grad_(True)
    (g1,) = torch.autograd.grad(J(a1).sum(), a1)
    steps = iter(range(H))
    cfg = SimpleNamespace(pi_batch=B, horizon=H, gamma=0.99, bptt_checkpoint=0, use_critic=False, critic_heads=1,
                          img_action_noise=0.0, clean_prev_action=False, bptt_detach=0)  # fmt: skip
    st = SimpleNamespace(cfg=cfg, task=task, tcfg=task.cfg, wm=wm, device='cpu', priv_dim=0, _img_step_ckpt=None,
                         env=SimpleNamespace(future_off=off), policy=lambda o, gl: a2[:, next(steps)])  # fmt: skip
    st._img_step = lambda *a: train.Trainer.img_step(st, *a)
    st.critic_input = lambda *a: train.Trainer.critic_input(st, *a)
    ones = torch.ones(B, H + 1)
    J2 = train.Trainer.rollout(st, s_flat, xi, torch.zeros(B, task.cfg.history, OBS_DIM), a_prev,
                               xi[:, 1 : 1 + NJ], ref, ref_kin, ones, ones)[0]  # fmt: skip
    (g2,) = torch.autograd.grad(J2.sum(), a2)
    assert torch.allclose(J(acts), J2, rtol=1e-5, atol=1e-6), (J(acts), J2)
    assert torch.allclose(g1, g2, rtol=1e-4, atol=1e-7) and g1.abs().max() > 0
    # autograd vs central differences (fp32) of the objective without the action-rate term: all through the dynamics
    u = unit_directions(B, 4, H, g)
    pred = directional(lambda a: J(a, 'state'), acts, u)
    with torch.no_grad():
        fd = central(lambda a: J(a, 'state'), acts, u, 1e-2)
    rel = ((fd - pred).abs().max() / pred.abs().max()).item()
    assert rel < 0.03 and pred.abs().max() > 1e-5, (rel, pred.abs().max())
    print(f'  WM objective ok (= Trainer.rollout J and dJ/da; autograd vs FD max rel. error {rel:.1e})')


def main():
    check_disc_return()
    check_linear_wm()
    check_fd_layout()
    check_resolved()
    check_agreement()
    check_wm_objective()
    print('wm_grad_fidelity checks passed')
    return 0


def test_wm_grad_fidelity():  # pytest entry point
    main()


if __name__ == '__main__':
    sys.exit(main())
