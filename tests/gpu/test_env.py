"""GPU sanity checks: FK vs MuJoCo body poses/velocities, env stepping, timing.

PYTHONPATH=. python tests/gpu/test_env.py [num_envs]     # reads data/processed/lafan1_50hz.npz
"""

import sys
import time

import torch

from allmimic.motion_lib import MotionLibrary
from allmimic.rotations import quat_to_mat
from allmimic.sim_env import EnvCfg, TrackingSimEnv
from allmimic.task import TaskCfg


def main():
    dev = 'cuda'
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    lib = MotionLibrary('data/processed/lafan1_50hz.npz', dev)
    print('clips', lib.num_clips, 'frames', lib.num_frames, 'bins', lib.num_bins, 'hours', lib.hours)
    t0 = time.time()
    env = TrackingSimEnv(lib, EnvCfg(num_envs=n), TaskCfg(), dev)
    print('env build', time.time() - t0)
    print('joint names', env.joint_names[:3], '...', 'q0', env.q0[:6])
    print('action scale', env.action_scale[:6])
    env.reset()
    # Step a few times with the reference-tracking "open loop" zero action.
    for _ in range(5):
        s, info = env.step(torch.zeros(n, 29, device=dev))
    # Compare FK of the current sim state with MuJoCo's own kinematics.
    env.sim.forward()
    s = env.read_state()
    P, R, V, W = env.task.body_kin(s)
    rob = env.robot
    bp = rob.data.body_link_pos_w
    bq = rob.data.body_link_quat_w
    bv = rob.data.body_link_lin_vel_w
    bw = rob.data.body_link_ang_vel_w
    names = [nm.split('/')[-1] for nm in rob.body_names]
    perm = [names.index(nm) for nm in env.kin.body_names]
    bp, bq, bv, bw = bp[:, perm], bq[:, perm], bv[:, perm], bw[:, perm]
    origins = env.scene.env_origins[:, None, :]
    print('env origin norm', env.scene.env_origins.norm(dim=-1).max().item())
    print('pos err', (P - bp).abs().max().item(), 'pos err (minus origin)', (P - (bp - origins)).abs().max().item())
    print('rot err', (R - quat_to_mat(bq)).abs().max().item())
    print('linvel err', (V - bv).abs().max().item(), 'angvel err', (W - bw).abs().max().item())
    print('reward', info['reward'][:8])
    for k, v in info['terms'].items():
        print(f'  {k}: {v.mean().item():.4f}')
    # Timing.
    torch.cuda.synchronize()
    t0 = time.time()
    steps = 50
    for _ in range(steps):
        obs, goal = env.policy_inputs()
        s, info = env.step(0.1 * torch.randn(n, 29, device=dev))
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f'{steps} steps x {n} envs: {dt:.3f}s -> {steps * n / dt:.0f} env-steps/s; dones {info["done"].sum().item()}')
    print('obs', obs.shape, 'goal', goal.shape, 'xi', env.xi.shape)


if __name__ == '__main__':
    main()
