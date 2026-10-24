"""CPU checks of the training switches (CHANGES.md numbers): bin-update fix (#9), in-place normalizer (#10), residual
actions (#11), goal offsets (#12), latent (#13), twist heading (#14), tri-point orientation inputs (#18),
reset-sampling cap and clip whitelist (#19),
per-joint-group residual and WM normalizer floor (#20), WM joint error-propagation weights (#21), critic ensemble
(#24), tri-point WM rotation output (#25), WM-consistency switches (#26), Muon option (#27), cluster-balanced bin
prior (#28).

PYTHONPATH=. python tests/test_flags.py
Defaults must stay bit-identical to upstream (frozen formulas copied below); new paths must match their definitions.
"""

import math
import sys

import torch

from allmimic import rotations
from allmimic.networks import Normalizer, Policy
from allmimic.replay import SDIM, ReplayBuffer
from allmimic.sim_env import update_bins
from allmimic.task import NJ, Task, TaskCfg, task_cfg


def upstream_bins(bin_G, bin_visits, b, ratio, a):
    """frozen upstream update (sim_env.py before #9)"""
    G_old = bin_G[b]
    new = torch.where(bin_visits[b] > 0, (1 - a) * G_old + a * ratio, ratio)
    bin_G[b] = new.clamp(0, 1)
    bin_visits[b] += 1


def upstream_norm_update(nm, x):
    """frozen upstream Normalizer.update (single process), returns (mean, var, count)"""
    mean, var, count = nm.mean.clone(), nm.var.clone(), nm.count.clone()
    n, m, v = x.shape[0], x.mean(0), x.var(0, unbiased=False)
    tot = count + n
    w = n / tot
    delta = m - mean
    mean += (w * delta).float()
    var = (var * (1 - w) + v * w + delta**2 * (w * (1 - w))).float()
    return mean, var, tot


def check_task_cfg():
    d = task_cfg({})
    assert d == TaskCfg(), 'empty run cfg must give the upstream TaskCfg'
    assert d.offsets == tuple(range(1, 11))
    t = task_cfg({'goal_pos_noise': 0.25, 'future_offsets': '1,2,3,4,6,10,15,25,40,60', 'action': 'res'})
    assert t.goal_pos_noise == 0.25 and t.future == 10 and t.offsets[-1] == 60 and t.action == 'res'
    assert task_cfg({'goal_pos_noise': 0.25}, train=False).goal_pos_noise == 0.0
    for bad in ('2,1', '0,1', '1,1'):
        try:
            task_cfg({'future_offsets': bad})
        except SystemExit:
            continue
        raise AssertionError(f'offsets {bad} accepted')
    print('  task_cfg ok')


def check_setpoint():
    t = object.__new__(Task)
    t.q0, t.action_scale = torch.randn(NJ), torch.rand(NJ) + 0.1
    a, qr = torch.randn(64, NJ), torch.randn(64, NJ)
    names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
             'left_ankle_pitch_joint', 'left_ankle_roll_joint'] * 2 + ['waist_yaw_joint'] + ['left_shoulder_pitch_joint'] * 16  # fmt: skip
    legs = torch.tensor([any(k in n for k in ('hip', 'knee', 'ankle')) for n in names])
    t.cfg = TaskCfg()
    t.set_joint_names(names)
    assert t.setpoint_action(a, qr) is a
    t.cfg = TaskCfg(action='res', res_scale=0.5)
    t.set_joint_names(names)
    q_sp = t.q0 + t.action_scale * t.setpoint_action(a, qr)
    assert torch.allclose(q_sp, qr + 0.5 * t.action_scale * a, atol=1e-5)
    # upstream residual formula, bit for bit
    assert torch.equal(t.setpoint_action(a, qr), 0.5 * a + (qr - t.q0) / t.action_scale)
    t.cfg = TaskCfg(action='res', res_legs='abs')
    t.set_joint_names(names)
    q_sp = t.q0 + t.action_scale * t.setpoint_action(a, qr)
    assert torch.allclose(q_sp[:, legs], (t.q0 + t.action_scale * a)[:, legs], atol=1e-5), 'legs absolute'
    assert torch.allclose(q_sp[:, ~legs], (qr + t.action_scale * a)[:, ~legs], atol=1e-5), 'upper body residual'
    t.cfg = TaskCfg(action='res', res_scale_legs=2.0)
    t.set_joint_names(names)
    q_sp = t.q0 + t.action_scale * t.setpoint_action(a, qr)
    assert torch.allclose(q_sp[:, legs], (qr + 2.0 * t.action_scale * a)[:, legs], atol=1e-5)
    assert torch.allclose(q_sp[:, ~legs], (qr + t.action_scale * a)[:, ~legs], atol=1e-5)
    from allmimic.networks import WorldModel

    wm = object.__new__(WorldModel)
    torch.nn.Module.__init__(wm)
    assert Normalizer(4, eps=0.05).eps == 0.05 and 'norm_eps' in WorldModel.__init__.__code__.co_varnames
    print('  setpoint_action ok (abs / res / legs abs / legs scale; upstream residual bit-identical)')


def check_bins():
    g = torch.Generator().manual_seed(0)
    nb = 1000
    G0 = torch.rand(nb, generator=g)
    V0 = (torch.rand(nb, generator=g) < 0.5).float() * 3
    b = torch.randperm(nb, generator=g)[:200]  # no duplicates: bit-identical
    r = torch.rand(200, generator=g)
    G1, V1, G2, V2 = G0.clone(), V0.clone(), G0.clone(), V0.clone()
    upstream_bins(G1, V1, b, r, 0.1)
    update_bins(G2, V2, b, r, 0.1)
    assert torch.equal(G1, G2) and torch.equal(V1, V2)
    # duplicates: every episode counted, the bin moves toward the mean of its episodes
    b = torch.tensor([3, 3, 3, 7, 9, 9])
    r = torch.tensor([0.0, 0.3, 0.6, 0.5, 0.2, 0.4])
    G, V = torch.full((nb,), 0.8), torch.zeros(nb)
    V[9] = 1
    update_bins(G, V, b, r, 0.1)
    assert V[3] == 3 and V[7] == 1 and V[9] == 3
    assert torch.isclose(G[3], torch.tensor(0.3)) and torch.isclose(G[7], torch.tensor(0.5))
    assert torch.isclose(G[9], torch.tensor(0.9 * 0.8 + 0.1 * 0.3))
    print('  update_bins ok (bit-identical without duplicates)')


def check_normalizer():
    nm = Normalizer(16)
    ptr = nm.var.data_ptr()
    for k in range(5):
        x = torch.randn(100 + 10 * k, 16) * (k + 1) + k
        mean, var, count = upstream_norm_update(nm, x)
        nm.update(x)
        assert torch.equal(nm.mean, mean) and torch.equal(nm.var, var) and torch.equal(nm.count, count)
    assert nm.var.data_ptr() == ptr, 'var must be updated in place'
    print('  Normalizer in place ok (bit-identical)')


def check_replay():
    class StubTask:
        def obs_clean(self, st, a_prev, bias):
            return a_prev  # expose the actions the history is built from

    n, L, k = 8, 50, 4
    for res in (False, True):
        buf = ReplayBuffer(n, L, 1 + NJ + 3, 'cpu', policy_actions=res)
        for t in range(30):
            a_sp, a_pi = torch.full((n, NJ), float(t)), torch.full((n, NJ), 1000.0 + t)
            buf.add(torch.randn(n, SDIM), a_sp, torch.randn(n, SDIM), torch.zeros(n, 1 + NJ + 3),
                    torch.full((n,), t), torch.zeros(n, dtype=torch.long), a_pi=a_pi)  # fmt: skip
        _, A, _, _, _ = buf.sample_windows(64, k)
        assert (A < 1000).all(), 'world-model windows must use the setpoint actions'
        hist = buf.sample_starts(64, 3, StubTask())[3]
        nz = hist[hist != 0]
        assert ((nz >= 1000) if res else (nz < 1000)).all(), 'obs history must use the policy outputs'
    print('  replay policy actions ok')


def rot_xyz(roll, pitch, yaw):
    """(B,) angles -> (B,3,3) Rz(yaw) Ry(pitch) Rx(roll)"""
    from allmimic.rotations import quat_from_euler_xyz, quat_to_mat

    return quat_to_mat(quat_from_euler_xyz(roll, pitch, yaw))


def check_heading():
    B = 512
    yaw = (torch.rand(B) * 2 - 1) * 3.1
    tilt = (torch.rand(B, 2) * 2 - 1) * 1.4  # roll, pitch up to 80 deg
    R = rot_xyz(tilt[:, 0], tilt[:, 1], yaw)
    Rz = rot_xyz(torch.zeros(B), torch.zeros(B), yaw)
    a = (torch.rand(B) * 2 - 1) * 3.1
    Ra = rot_xyz(torch.zeros(B), torch.zeros(B), a)
    try:
        for mode in ('xaxis', 'twist'):
            rotations.HEADING['mode'] = mode
            H = rotations.heading_mat
            assert torch.allclose(H(Rz), Rz, atol=1e-5), f'{mode}: pure yaw must be its own heading'
            assert torch.allclose(H(Ra @ R), Ra @ H(R), atol=1e-4), f'{mode}: heading must be yaw-equivariant'
        # pitch sweep through 90 deg (lying face down) at fixed yaw: twist stays put, the x-axis heading flips 180 deg
        p = torch.linspace(1.2, 1.9, 71)
        R = rot_xyz(torch.zeros(71), p, torch.full((71,), 0.7))
        rotations.HEADING['mode'] = 'twist'
        jt = (rotations.heading_mat(R)[1:] - rotations.heading_mat(R)[:-1]).abs().amax()
        rotations.HEADING['mode'] = 'xaxis'
        jx = (rotations.heading_mat(R)[1:] - rotations.heading_mat(R)[:-1]).abs().amax()
        assert jt < 1e-4 and jx > 1.0, (jt, jx)
    finally:
        rotations.HEADING['mode'] = 'xaxis'
    assert task_cfg({'heading': 'twist'}).heading == 'twist' and TaskCfg().heading == 'xaxis'
    print('  heading ok (twist: equivariant, no flip through 90 deg pitch)')


def check_tripoint():
    from allmimic.rotations import TRI_REF, rot6d, tripoint, tripoint_decode

    B, K = 256, 10
    q = torch.nn.functional.normalize(torch.randn(B, 4), dim=-1)
    from allmimic.rotations import quat_to_mat

    R, o = quat_to_mat(q), torch.randn(B, 3)
    x = tripoint(R, o)
    assert x.shape == (B, 9)
    m = x.unflatten(-1, (3, 3))
    assert torch.allclose(m.mean(-2), o, atol=1e-6), 'marker mean must be the origin'
    assert torch.allclose(m[:, 0] - o, R @ TRI_REF[0], atol=1e-6), 'marker 0 = origin + R @ (e0 - mean)'
    R1, o1 = tripoint_decode(x)
    assert torch.allclose(R1, R, atol=1e-5) and torch.allclose(o1, o, atol=1e-6), 'Kabsch decode must invert'
    R2, _ = tripoint_decode(x + 1e-3 * torch.randn_like(x))  # noisy markers still decode to a proper rotation
    assert torch.allclose(R2 @ R2.transpose(-1, -2), torch.eye(3).expand(B, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(R2), torch.ones(B), atol=1e-5)
    # goal / privileged goal: same size as 6D, pose = tripoint(R_rel, dp); noise translates all markers
    t = object.__new__(Task)
    s_ = {'pos': torch.randn(B, 3), 'rot': quat_to_mat(torch.nn.functional.normalize(torch.randn(B, 4), dim=-1))}
    ref = {'pos': torch.randn(B, K, 3), 'rot': quat_to_mat(torch.nn.functional.normalize(torch.randn(B, K, 4), dim=-1)),
           'q': torch.randn(B, K, NJ), 'qd': torch.randn(B, K, NJ), 'v': torch.randn(B, K, 3), 'w': torch.randn(B, K, 3)}  # fmt: skip
    t.cfg = TaskCfg()
    g6 = t.goal(s_, ref).unflatten(-1, (K, -1))
    t.cfg = TaskCfg(rot_repr='tripoint', goal_pos_noise=0.25)
    gt = t.goal(s_, ref).unflatten(-1, (K, -1))
    assert gt.shape == g6.shape and torch.equal(gt[..., 9:], g6[..., 9:])
    Rr, dp = tripoint_decode(gt[..., :9])
    assert torch.allclose(dp, g6[..., :3], atol=1e-5) and torch.allclose(rot6d(Rr), g6[..., 3:9], atol=1e-4)
    gn = t.noisy_goal(gt.flatten(-2)).unflatten(-1, (K, -1))
    c, cn = gt[..., :9].unflatten(-1, (3, 3)), gn[..., :9].unflatten(-1, (3, 3))
    assert torch.allclose(c - c.mean(-2, keepdim=True), cn - cn.mean(-2, keepdim=True), atol=1e-5)
    assert (cn.mean(-2) - c.mean(-2)).abs().max() <= 0.25 + 1e-6 and (cn - c).abs().max() > 0.01
    assert t.goal_priv(s_, ref).shape == (B, K * 9)
    # state features: [height + 6D] (7) -> markers around (0, 0, height) (9)
    nb = 30
    kin = (torch.randn(B, nb, 3), None, torch.randn(B, nb, 3), None)
    t.q0 = torch.zeros(NJ)
    s_.update(v=torch.randn(B, 3), w=torch.randn(B, 3), q=torch.randn(B, NJ), qd=torch.randn(B, NJ))
    t.cfg = TaskCfg()
    f6 = t.state_features(s_, kin)
    t.cfg = TaskCfg(rot_repr='tripoint')
    ft = t.state_features(s_, kin)
    assert ft.shape[-1] == f6.shape[-1] + 2 and torch.equal(ft[..., 9:], f6[..., 7:])
    Rh, oh = tripoint_decode(ft[..., :9])
    assert torch.allclose(oh[..., 2], f6[..., 0], atol=1e-5) and torch.allclose(rot6d(Rh), f6[..., 1:7], atol=1e-4)
    assert task_cfg({'rot_repr': 'tripoint'}).rot_repr == 'tripoint' and TaskCfg().rot_repr == '6d'
    print('  tripoint ok (Kabsch inverse, goal / priv / features sizes, translation-only goal noise)')


def check_cap_and_keep():
    import os
    import tempfile

    import numpy as np

    from allmimic.motion_lib import MotionLibrary
    from allmimic.sim_env import cap_probs

    g = torch.Generator().manual_seed(0)
    nb = 100_000
    G = torch.rand(nb, generator=g) * 0.6 + 0.3
    G[:300] = torch.rand(300, generator=g) * 0.1  # a few very hard bins, as on BONES (#19)
    soft = torch.softmax(-25 * G, 0)
    cap = 50 / nb
    p = cap_probs(soft, cap)
    assert abs(p.sum().item() - 1) < 1e-4 and p.max().item() <= cap * (1 + 1e-4), (p.sum(), p.max() / cap)
    assert soft[:300].sum() > 0.5 and p[:300].sum() < 0.2, (soft[:300].sum(), p[:300].sum())
    order = torch.argsort(G)[:2000]
    assert (p[order[:-1]] >= p[order[1:]] - 1e-12).float().mean() > 0.99, 'harder bins must keep at least as much mass'
    flat = torch.full((nb,), 1 / nb)
    assert torch.equal(cap_probs(flat, cap), flat), 'no entry above the cap: unchanged'
    # whitelist: only the named clips, bins rebuilt for them
    lengths = np.array([120, 75, 200, 51, 90])
    N = lengths.sum()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'lib.npz')
        np.savez(path, qpos=np.random.randn(N, 36).astype(np.float32), qvel=np.random.randn(N, 35).astype(np.float32),
                 starts=np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths=lengths,
                 names=np.array(['a', 'b', 'c', 'd', 'e']), fps=np.array(50.0))  # fmt: skip
        full = MotionLibrary(path, 'cpu')
        sub = MotionLibrary(path, 'cpu', keep={'b', 'd', 'zz'})
        assert sub.names == ['b', 'd'] and sub.num_frames == 75 + 51 and sub.num_bins == 2 + 2
        assert torch.equal(sub.q[:75], full.q[120:195]) and torch.equal(sub.q[75:], full.q[395:446])
    print('  softmax cap ok (sum 1, max <= cap, hard bins 0.5+ -> < 0.2, order kept) / clip whitelist ok')


def check_wm_joint_weights():
    from types import SimpleNamespace

    from allmimic.joint_weights import JOINT_WEIGHTS
    from allmimic.train import Trainer

    names = list(JOINT_WEIGHTS['tracked'])[::-1]  # any order: weights follow the env's joint names
    stub = SimpleNamespace(cfg=SimpleNamespace(wm_joint_weights='', wm_joint_weight_clip=4.0),
                           env=SimpleNamespace(joint_names=names), device='cpu')  # fmt: skip
    assert Trainer._wm_error_weights(stub) is None
    for key in ('tracked', 'axis_sq'):
        stub.cfg.wm_joint_weights = key
        w = Trainer._wm_error_weights(stub)
        assert w.shape == (70,) and torch.equal(w[:12], torch.ones(12)) and torch.equal(w[12:41], w[41:])
        j = w[12:41]
        assert abs(j.mean().item() - 1) < 1e-5 and j.max() / j.min() <= 16 + 1e-4
        raw = torch.tensor([JOINT_WEIGHTS[key][n] for n in names]).clamp(0.25, 4)
        assert torch.allclose(j, raw / raw.mean())
    print('  WM joint weights ok (70 dims, root 1, clipped, mean 1, env order)')


def _legacy_update_critic(self, rec, xH, valid):
    """frozen pre-#24 Trainer.update_critic (one head, one clip over all critic parameters)"""
    from allmimic.train import zero_if_nonfinite

    cfg = self.cfg
    H = cfg.horizon
    X = torch.stack(rec['x'] + [xH], dim=1)
    self.critic.norm.update(X)
    with torch.no_grad():
        V = self.critic(X) * valid
        R = torch.zeros_like(V)
        R[:, H] = V[:, H]
        for t in range(H - 1, -1, -1):
            r, d, m = rec['r'][t], rec['d'][t], rec['m'][t]
            R[:, t] = r + cfg.gamma * (1 - d) * m * ((1 - cfg.lam) * V[:, t + 1] + cfg.lam * R[:, t + 1])
        w = torch.stack(rec['c'], dim=1) * valid[:, :H]
    Xf = X[:, :H].reshape(-1, X.shape[-1])
    Rf = R[:, :H].reshape(-1)
    wf = w.reshape(-1)
    n = Xf.shape[0]
    mb = n // cfg.critic_minibatches
    tot = torch.zeros(())
    for _ in range(cfg.critic_epochs):
        perm = torch.randperm(n)
        for i in range(cfg.critic_minibatches):
            j = perm[i * mb : (i + 1) * mb]
            err = (self.critic(Xf[j]) - Rf[j]) ** 2
            loss = (err * wf[j]).sum() / wf[j].sum().clamp_min(1)
            self.opt_v.zero_grad(set_to_none=True)
            loss.backward()
            zero_if_nonfinite(
                self.critic.net.parameters(), torch.nn.utils.clip_grad_norm_(self.critic.net.parameters(), 1.0)
            )
            self.opt_v.step()
            tot = tot + loss.detach()
    return {'v_loss': tot.item() / (cfg.critic_epochs * cfg.critic_minibatches), 'v_mean': V[:, 0].mean().item()}


def check_critic_heads():
    import copy
    from types import SimpleNamespace

    from allmimic.networks import Critic
    from allmimic.train import Trainer

    B, H, D = 32, 4, 12
    g = torch.Generator().manual_seed(3)
    rec = {'x': [torch.randn(B, D, generator=g) for _ in range(H)], 'r': [torch.rand(B, generator=g) for _ in range(H)],
           'd': [(torch.rand(B, generator=g) < 0.1).float() for _ in range(H)], 'm': [torch.ones(B) for _ in range(H)],
           'c': [torch.ones(B) for _ in range(H)]}  # fmt: skip
    xH, valid = torch.randn(B, D, generator=g), (torch.rand(B, H + 1, generator=g) > 0.05).float()

    def stub(heads):
        torch.manual_seed(0)
        c = Critic(D, hidden=(16, 8), heads=heads)
        cfg = SimpleNamespace(
            horizon=H, gamma=0.99, lam=0.95, critic_epochs=2, critic_minibatches=4, critic_heads=heads
        )
        return SimpleNamespace(cfg=cfg, critic=c, opt_v=torch.optim.Adam(c.net.parameters(), lr=1e-3), device='cpu',
                               sync_grads=lambda params: None)  # fmt: skip

    a, b = stub(1), stub(1)
    torch.manual_seed(5)
    ra = Trainer.update_critic(a, copy.deepcopy(rec), xH, valid)
    torch.manual_seed(5)
    rb = _legacy_update_critic(b, copy.deepcopy(rec), xH, valid)
    assert ra == rb and all(torch.equal(p, q) for p, q in zip(a.critic.parameters(), b.critic.parameters()))
    e = stub(2)
    torch.manual_seed(5)
    re = Trainer.update_critic(e, copy.deepcopy(rec), xH, valid)
    v = e.critic(torch.randn(7, D))
    assert v.shape == (7, 2) and math.isfinite(re['v_loss']) and not torch.equal(v[:, 0], v[:, 1])
    print('  critic heads ok (1 head bit-identical to the frozen update, 2 heads: (..., 2), trains)')


def check_wm_tripoint_out():
    from allmimic.networks import STATE_DELTA_DIM, TRI_DELTA_DIM, WorldModel, apply_delta_tri, state_delta_tri
    from allmimic.rotations import quat_to_mat, tripoint, tripoint_decode_gs

    g = torch.Generator().manual_seed(4)
    q = lambda *s: quat_to_mat(torch.nn.functional.normalize(torch.randn(*s, 4, generator=g), dim=-1))
    R = q(256)
    assert torch.allclose(tripoint_decode_gs(tripoint(R)), R, atol=1e-5), 'GS decode must invert exact markers'
    Rn = tripoint_decode_gs(tripoint(R) + 0.05 * torch.randn(256, 9, generator=g))
    assert torch.allclose(Rn @ Rn.transpose(-1, -2), torch.eye(3).expand(256, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(Rn), torch.ones(256), atol=1e-5)
    x = (tripoint(q(4)) + 0.05 * torch.randn(4, 9, generator=g)).double().requires_grad_()
    assert torch.autograd.gradcheck(tripoint_decode_gs, (x,)), 'GS decode gradient'
    B = 64
    s0 = {'pos': torch.randn(B, 3, generator=g), 'rot': q(B), 'v': torch.randn(B, 3, generator=g), 'w': torch.randn(B, 3, generator=g),
          'q': torch.randn(B, NJ, generator=g), 'qd': torch.randn(B, NJ, generator=g)}  # fmt: skip
    s1 = {k: v + 0.1 * torch.randn(v.shape, generator=g) if k != 'rot' else v @ q(B) for k, v in s0.items()}
    d = state_delta_tri(s0, s1)
    assert d.shape == (B, TRI_DELTA_DIM)
    r = apply_delta_tri(s0, d)
    assert all(torch.allclose(r[k], s1[k], atol=1e-4) for k in s1), 'apply_delta_tri must invert state_delta_tri'
    t = object.__new__(Task)
    wa = WorldModel(t, 20, 4, hidden=(8,))
    wt = WorldModel(t, 20, 4, hidden=(8,), rot_out='tripoint')
    assert wa.err_norm is None and wa.err_std is not None and wa.out_norm.mean.shape == (STATE_DELTA_DIM,)
    assert wt.out_norm.mean.shape == (TRI_DELTA_DIM,) and wt.err_std.shape == (STATE_DELTA_DIM,)
    assert wt.net[-1].out_features == TRI_DELTA_DIM and wa.net[-1].out_features == STATE_DELTA_DIM
    print('  WM tri-point rotation output ok (GS decode exact / orthonormal / gradcheck, delta round trip, sizes)')


def check_v3_consistency():
    from types import SimpleNamespace

    from allmimic.sim_env import EnvCfg, TrackingSimEnv

    # bin retirement: bins below retire_g after enough visits keep retire_w of their softmax weight
    nb = 1000
    G = torch.rand(nb)
    G[:100] = 0.01
    visits = torch.full((nb,), 30.0)
    visits[:50] = 5.0  # too few visits: not retired yet
    probs = {}
    for rg in (0.0, 0.05):
        stub = SimpleNamespace(cfg=EnvCfg(retire_g=rg), lib=SimpleNamespace(num_bins=nb), bin_G=G, bin_visits=visits,
                               bin_past_w=torch.zeros(nb), bin_prior=None)  # fmt: skip
        probs[rg] = TrackingSimEnv._start_probs(stub)
        assert abs(probs[rg].sum().item() - 1) < 1e-5
    c = EnvCfg()
    soft0 = torch.softmax(-c.softmax_temp * G, 0)
    w = torch.where((G < 0.05) & (visits >= 20), 0.1, 1.0)
    soft1 = soft0 * w / (soft0 * w).sum()
    assert torch.allclose(probs[0.05], c.p_uniform / nb + (c.p_softmax + c.p_past) * soft1, atol=1e-7)
    assert probs[0.05][50:100].sum() < probs[0.0][50:100].sum() and probs[0.05][:50].sum() > probs[0.0][:50].sum()
    # residual penalty in Task.reward on the real G1 tree: residual joints only (legs absolute excluded), absent
    # unless set, and the rest of the reward unchanged
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_wm_grad_fidelity import g1_task, rand_states

    g = torch.Generator().manual_seed(9)
    s, ref = rand_states(g, 8), rand_states(g, 8)
    a, a_prev = torch.randn(8, NJ, generator=g), torch.randn(8, NJ, generator=g)
    out = {}
    for key, kw in (('off', {}), ('on', {'w_residual': -0.02}), ('abs', {'w_residual': -0.02, 'action': 'abs'})):
        t = g1_task(TaskCfg(**{'action': 'res', 'res_legs': 'abs', **kw}))
        kin = t.kin.forward(s['pos'], s['rot'], s['q'], s['v'], s['w'], s['qd'])
        rkin = t.kin.forward(ref['pos'], ref['rot'], ref['q'], ref['v'], ref['w'], ref['qd'])
        out[key] = t.reward(s, kin, ref, rkin, a, a_prev)
    legs = t.res_ref == 0
    assert legs.sum() == 12 and 'residual' not in out['off'][1] and 'residual' not in out['abs'][1]
    assert torch.allclose(out['on'][1]['residual'], -0.02 * (a[:, ~legs] ** 2).sum(-1))
    assert torch.allclose(out['on'][0] - out['off'][0], out['on'][1]['residual'] * t.cfg.dt, atol=1e-6)
    assert task_cfg({'w_residual': -0.02}).w_residual == -0.02 and task_cfg({}).w_residual == 0.0
    print('  #26 ok (bin retirement probabilities, residual penalty on residual joints only, task_cfg plumbing)')


def check_muon():
    if not hasattr(torch.optim, 'Muon'):
        print('  Muon: torch.optim.Muon not available here, skipped')
        return
    from allmimic.networks import mlp
    from allmimic.train import MuonAdam

    torch.manual_seed(0)
    net = mlp(12, (32, 32, 16), 5)
    opt = MuonAdam([net], 1e-3)
    lin = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    muon_ids = {id(p) for p in opt.opts[0].param_groups[0]['params']}
    assert muon_ids == {id(lin[1].weight), id(lin[2].weight)}, 'hidden matrices only'
    assert len(opt.param_groups) == 2 and sum(len(g['params']) for g in opt.param_groups) == len(list(net.parameters()))
    x = torch.randn(64, 12)
    before = [p.detach().clone() for p in net.parameters()]
    for g in opt.param_groups:
        g['lr'] = 2e-3  # the trainer's LR schedule writes through param_groups
    net(x).pow(2).mean().backward()
    opt.step()
    opt.zero_grad()
    assert all(not torch.equal(a, b) for a, b in zip(before, net.parameters())), 'every parameter updated'
    sd = opt.state_dict()
    opt2 = MuonAdam([net], 1e-3)
    opt2.load_state_dict(sd)
    assert opt2.opts[1].param_groups[0]['lr'] == 2e-3
    print('  Muon ok (hidden matrices -> Muon, rest -> Adam, LR schedule via param_groups, state round trip)')


def check_bin_prior():
    import tempfile
    from types import SimpleNamespace

    import numpy as np

    from allmimic.sim_env import EnvCfg, TrackingSimEnv
    from allmimic.train import Trainer

    # npz over 4 clips (bins 2, 1, 3, 2); the library keeps clips 0, 2, 3 (a whitelist dropped clip 1)
    lengths = np.array([60, 30, 150, 51])
    w = np.array([1.0, 2.0, 5.0, 0.5, 0.5, 4.0, 1.0, 3.0], np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = f'{tmp}/bins.npz'
        np.savez(path, names=np.array(['a', 'b', 'c', 'd']), lengths=lengths, w_alpha_0_5=w, **{'w_alpha_0.5': w})
        lib = SimpleNamespace(names=['a', 'c', 'd'], lengths=torch.tensor([60, 150, 51]), bin_len=50, num_bins=7)
        stub = SimpleNamespace(cfg=SimpleNamespace(bin_prior=path, bin_prior_alpha=0.5), lib=lib, device='cpu')
        got = Trainer._bin_prior(stub)
    keep = torch.tensor([1.0, 2.0, 0.5, 0.5, 4.0, 1.0, 3.0])
    assert torch.allclose(got, keep / keep.mean()), got
    # sampler: the prior multiplies the return-based probabilities
    nb = 7
    env = SimpleNamespace(cfg=EnvCfg(), lib=SimpleNamespace(num_bins=nb), bin_G=torch.rand(nb), bin_visits=torch.ones(nb),
                          bin_past_w=torch.zeros(nb), bin_prior=None)  # fmt: skip
    base = TrackingSimEnv._start_probs(env)
    env.bin_past_w, env.bin_prior = torch.zeros(nb), got
    p = TrackingSimEnv._start_probs(env)
    assert torch.allclose(p, base * got / (base * got).sum(), atol=1e-7) and abs(p.sum().item() - 1) < 1e-6
    print('  bin prior ok (npz -> library bin order with dropped clips, mean 1; sampler multiplies it in)')


def check_latent():
    p = Policy(930, 730, latent=128)
    assert p.encoder[-1].out_features == 128 and p.decoder[0].in_features == 128 + 930
    assert Policy(930, 730).encoder[-1].out_features == 32
    print('  latent width ok')


def main(argv=None):
    torch.manual_seed(0)
    check_task_cfg()
    check_setpoint()
    check_bins()
    check_normalizer()
    check_replay()
    check_latent()
    check_heading()
    check_tripoint()
    check_cap_and_keep()
    check_wm_joint_weights()
    check_critic_heads()
    check_wm_tripoint_out()
    check_v3_consistency()
    check_muon()
    check_bin_prior()
    print('flag checks passed')
    return 0


def test_flags():  # pytest entry point
    main()


if __name__ == '__main__':
    sys.exit(main())
