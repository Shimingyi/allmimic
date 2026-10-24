"""Fig. 4a: control steps per second, learned world model vs MuJoCo Warp.

One step = one 50 Hz control step (MuJoCo Warp: 4 x 5 ms substeps). The world-model
step includes the heading-frame features (forward kinematics), the 3x2048 MLP and the
analytic integration. Measured with fresh networks (speed does not depend on weights).
"""

import argparse
import json
import time

import torch

from allmimic import networks
from allmimic.motion_lib import MotionLibrary
from allmimic.networks import WorldModel
from allmimic.sim_env import EnvCfg, TrackingSimEnv
from allmimic.task import XI_DIM, TaskCfg


def rate(fn, n_envs, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return n_envs * iters / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='512,1024,2048,4096,8192,16384,32768,65536')
    ap.add_argument('--out', default='runs/bench_rollout.json')
    ap.add_argument('--no-dr', action='store_true', help='skip per-world DR field expansion (slow at 32k+ worlds)')
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    lib = MotionLibrary('data/processed/lafan1_50hz.npz', dev)
    sizes = [int(x) for x in args.sizes.split(',')]
    res = {'gpu': torch.cuda.get_device_name(), 'sizes': sizes, 'mjwarp': [], 'wm_tf32': [], 'wm_bf16': []}
    for n in sizes:
        env = TrackingSimEnv(
            lib, EnvCfg(num_envs=n, compile=False, nconmax=35, njmax=250, domain_rand=not args.no_dr), TaskCfg(), dev
        )
        env.reset()
        env.task.compile_sizes = {n}

        def sim_step():
            for _ in range(4):
                env.sim.step()

        r_sim = rate(sim_step, n, 20)
        wm = WorldModel(env.task, env.task.features(env.state).shape[-1], XI_DIM).to(dev)
        s = env.state
        a = torch.zeros(n, 29, device=dev)
        xi = env.xi
        out = {}
        for mode in ('wm_tf32', 'wm_bf16'):
            networks.AMP['enabled'] = mode == 'wm_bf16'
            with torch.no_grad():
                out[mode] = rate(lambda: wm.step(s, a, xi), n, 20)
        res['mjwarp'].append(r_sim)
        res['wm_tf32'].append(out['wm_tf32'])
        res['wm_bf16'].append(out['wm_bf16'])
        print(
            f'N={n}: mjwarp {r_sim:.3e}  wm tf32 {out["wm_tf32"]:.3e} ({out["wm_tf32"] / r_sim:.1f}x)  '
            f'wm bf16 {out["wm_bf16"]:.3e} ({out["wm_bf16"] / r_sim:.1f}x)',
            flush=True,
        )
        del env, wm
        torch.cuda.empty_cache()
        import os

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
