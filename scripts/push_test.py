"""Sec. 5.3 push test: hold a static T-pose reference, apply a horizontal velocity push to the
root, and measure the settle time (root speed back below 0.1 m/s for 0.5 s) and falls.

usage: push_test.py out.json ckpt_a.pt [ckpt_b.pt ...]
"""

import json
import math
import os
import sys
import tempfile

import numpy as np
import torch

from allmimic.evaluate import load_supermimic_policy, make_eval_env
from allmimic.motion_lib import MotionLibrary
from allmimic.task import task_cfg

PUSHES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
DIRS = 8  # push directions per magnitude


def tpose_library(path, seconds=12.0, fps=50):
    n = int(seconds * fps)
    q = np.zeros((n, 29), dtype=np.float32)
    # Knees slightly bent like the default stance, arms straight out.
    q[:, [0, 6]] = -0.1
    q[:, [3, 9]] = 0.2
    q[:, [4, 10]] = -0.1
    q[:, 16] = 1.4  # left shoulder roll
    q[:, 23] = -1.4  # right shoulder roll
    pos = np.tile(np.array([[0.0, 0.0, 0.785]], dtype=np.float32), (n, 1))
    quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1))
    qpos = np.concatenate([pos, quat, q], 1)
    qvel = np.zeros((n, 35), dtype=np.float32)
    np.savez(
        path,
        qpos=qpos,
        qvel=qvel,
        starts=np.array([0]),
        lengths=np.array([n]),
        names=np.array(['tpose']),
        fps=np.array(float(fps)),
    )


def main():
    out = sys.argv[1]
    ckpts = sys.argv[2:]
    dev = 'cuda'
    tmp = os.path.join(tempfile.mkdtemp(), 'tpose.npz')
    tpose_library(tmp)
    lib = MotionLibrary(tmp, dev)
    n = len(PUSHES) * DIRS
    push_t = 100  # 2 s after start
    results = {}
    for c in ckpts:
        pol, ck = load_supermimic_policy(c, dev)
        # the env's task encoding (goal / action / heading / offsets) must be the checkpoint's: a residual-action
        # policy in a default (absolute-action) env would be scored with wrong setpoints
        env = make_eval_env(lib, n, dev, task_cfg(ck.get('cfg', {}), train=False))
        env.reset()
        ids = torch.arange(n, device=dev)
        # Start a little into the clip so the 10 future frames are all T-pose.
        env.reset_envs(ids, frames=torch.zeros(n, dtype=torch.long, device=dev))
        env._finish_resets(ids, force=True)
        mags = torch.tensor(PUSHES, device=dev).repeat_interleave(DIRS)
        ang = torch.arange(DIRS, device=dev).repeat(len(PUSHES)).float() * (2 * math.pi / DIRS)
        fell = torch.zeros(n, dtype=torch.bool, device=dev)
        speed_hist = []
        for t in range(500):
            if t == push_t:
                qvel = env.sim.data.qvel[:].clone()
                va = env.root_vadr
                qvel[:, va] += mags * torch.cos(ang)
                qvel[:, va + 1] += mags * torch.sin(ang)
                env.sim.data.qvel[:] = qvel
                env.state = env.read_state()
            obs, goal = env.policy_inputs()
            with torch.no_grad():
                a = pol(obs, goal)
            s, info = env.step(a)
            fell |= info['terminated']
            speed_hist.append(torch.linalg.norm(s['v'][:, :2], dim=-1))
        sp = torch.stack(speed_hist, 1)[:, push_t:]  # (n, T)
        settle = []
        win = 25
        for i in range(n):
            if fell[i]:
                settle.append(None)
                continue
            ok = (sp[i] < 0.1).float()
            run = torch.nn.functional.conv1d(ok[None, None], torch.ones(1, 1, win, device=dev))[0, 0]
            idx = (run >= win).nonzero()
            settle.append(float(idx[0]) * 0.02 if len(idx) else float('inf'))
        per_mag = {}
        for k, m in enumerate(PUSHES):
            rows = settle[k * DIRS : (k + 1) * DIRS]
            falls = sum(r is None for r in rows)
            finite = [r for r in rows if r is not None and math.isfinite(r)]
            per_mag[str(m)] = {
                'falls': falls,
                'never_settled': sum(r is not None and not math.isfinite(r) for r in rows),
                'settle_s_median': float(np.median(finite)) if finite else None,
                'settle_s_max': float(max(finite)) if finite else None,
            }
        results[c] = per_mag
        print(c, json.dumps(per_mag), flush=True)
    json.dump(results, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
