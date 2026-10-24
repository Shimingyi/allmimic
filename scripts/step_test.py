"""Velocity step test (App. G): from standing, hold a fixed command for 5 s, 4 trials each.
Relative steady-state error = || mean velocity error over the last 2 s || / || command ||,
root velocity in the heading frame; per command, the median over trials that did not fall.
Commands: 15 x 15 grids on the (vx, vy), (vx, wz), (vy, wz) planes, standing left out (672).

usage: step_test.py motions.npz clip_prefixes out.json ckpt_rl.pt ckpt_distill.pt
"""

import json
import sys

import numpy as np
import torch

from allmimic import networks
from allmimic.motion_lib import MotionLibrary
from allmimic.networks import Policy, mlp
from allmimic.rotations import heading_mat, mtv
from allmimic.sim_env import EnvCfg
from allmimic.task import GOAL_FRAME_DIM, OBS_DIM, TaskCfg
from allmimic.velocity import CMD_HI, CMD_LO, VelocityEnv, cmd_norm

TRIALS = 4


def grid():
    vx = np.linspace(CMD_LO[0], CMD_HI[0], 15)
    vy = np.linspace(CMD_LO[1], CMD_HI[1], 15)
    wz = np.linspace(CMD_LO[2], CMD_HI[2], 15)
    cmds, plane = [], []
    for a in vx:
        for b in vy:
            cmds.append((a, b, 0.0))
            plane.append('vx_vy')
    for a in vx:
        for c in wz:
            cmds.append((a, 0.0, c))
            plane.append('vx_wz')
    for b in vy:
        for c in wz:
            cmds.append((0.0, b, c))
            plane.append('vy_wz')
    keep = [i for i, c in enumerate(cmds) if np.abs(c).max() > 1e-9]
    return np.array(cmds)[keep], [plane[i] for i in keep]


def standing_frames(lib, k):
    """k frames where the reference stands still with at least 6 s of clip left."""
    v = torch.linalg.norm(lib.root_lin_vel[:, :2], dim=-1)
    w = lib.root_ang_vel[:, 2].abs()
    ok = (
        (v < 0.05)
        & (w < 0.1)
        & (lib.root_pos[:, 2] > 0.7)
        & (lib.remaining(torch.arange(lib.num_frames, device=v.device)) > 300)
    )
    idx = ok.nonzero().squeeze(-1)
    # Spread the picks over different clips.
    picks, seen = [], set()
    for i in idx.tolist():
        c = int(lib.clip_of_frame[i])
        if c not in seen:
            picks.append(i)
            seen.add(c)
        if len(picks) == k:
            break
    while len(picks) < k:
        picks.append(int(idx[len(picks) * 997 % len(idx)]))
    return picks


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    tc = TaskCfg()
    pol = Policy(OBS_DIM * tc.history, GOAL_FRAME_DIM * tc.future).to(dev)
    pol.load_state_dict(ck['policy'])
    enc = mlp(3 + OBS_DIM * tc.history, (256, 256), 32).to(dev)
    enc.load_state_dict(ck['vel_enc'])
    return pol.eval(), enc.eval(), ck


@torch.no_grad()
def run(path, lib, env, cmds, starts, dev):
    pol, enc, ck = load(path, dev)
    n = env.num_envs
    C = torch.tensor(np.repeat(cmds, TRIALS, axis=0), dtype=torch.float32, device=dev)
    env.fixed_cmd = C
    env.is_vel[:] = True
    frames = torch.tensor(np.tile(starts, len(cmds)), device=dev)
    env.reset()
    ids = torch.arange(n, device=dev)
    env.reset_envs(ids, frames=frames)
    env._finish_resets(ids, force=True)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    vel = []
    for t in range(250):
        obs, goal, cn = env.policy_inputs()
        z = torch.tanh(enc(torch.cat([cn, pol.obs_norm(obs)], -1)))
        a = pol.decoder(torch.cat([z, pol.obs_norm(obs)], -1))
        s, info = env.step(a)
        fell |= info['terminated']
        v_h = mtv(heading_mat(s['rot']), s['v'])
        vel.append(torch.cat([v_h[:, :2], s['w'][:, 2:3]], -1))
    V = torch.stack(vel, 1)[:, -100:]  # last 2 s
    err = (V.mean(1) - C).norm(dim=-1) / C.norm(dim=-1)
    err = err.view(len(cmds), TRIALS).cpu().numpy()
    fl = fell.view(len(cmds), TRIALS).cpu().numpy()
    per = []
    for i in range(len(cmds)):
        good = err[i][~fl[i]]
        per.append(float(np.median(good)) if len(good) else None)
    return per, fl.all(1).tolist(), ck.get('it')


def main():
    motions, prefixes, out, ck_rl, ck_ds = sys.argv[1:6]
    dev = 'cuda'
    networks.AMP['enabled'] = False
    torch.backends.cuda.matmul.allow_tf32 = True
    lib = MotionLibrary(motions, dev, prefixes=prefixes)
    cmds, plane = grid()
    starts = standing_frames(lib, TRIALS)
    n = len(cmds) * TRIALS
    cfg = EnvCfg(num_envs=n, episode_length_s=1e9, obs_noise=False, reset_noise=False, domain_rand=True)
    env = VelocityEnv(lib, cfg, TaskCfg(), dev, mode='rl')
    env.eval_mode = True
    res = {'commands': cmds.tolist(), 'plane': plane, 'start_frames': starts}
    for name, path in (('rl', ck_rl), ('distill', ck_ds)):
        per, fell_all, it = run(path, lib, env, cmds, starts, dev)
        res[name] = {'ckpt': path, 'iteration': it, 'rel_error': per, 'fell_all_trials': fell_all}
    big = 10.0
    e_rl = np.array([x if x is not None else big for x in res['rl']['rel_error']])
    e_ds = np.array([x if x is not None else big for x in res['distill']['rel_error']])
    res['summary'] = {
        'n_commands': int(len(cmds)),
        'rl_better': int((e_rl < e_ds).sum()),
        'distill_better': int((e_ds < e_rl).sum()),
        'rl_median_rel_error': float(np.median([x for x in res['rl']['rel_error'] if x is not None])),
        'distill_median_rel_error': float(np.median([x for x in res['distill']['rel_error'] if x is not None])),
        'rl_fell_all': int(sum(res['rl']['fell_all_trials'])),
        'distill_fell_all': int(sum(res['distill']['fell_all_trials'])),
        'paper': {
            'rl_better': 650,
            'distill_better': 15,
            'rl_median': 0.16,
            'distill_median': 0.49,
            'distill_fell_all': 7,
            'n_commands': 672,
        },
    }
    print(json.dumps(res['summary'], indent=1))
    json.dump(res, open(out, 'w'))


if __name__ == '__main__':
    main()
