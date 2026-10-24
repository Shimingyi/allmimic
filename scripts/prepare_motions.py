"""Convert retargeted G1 CSV clips into one motion-library .npz at 50 Hz.

LAFAN1 (lvhaidong/LAFAN1_Retargeting_Dataset, g1/*.csv): 30 fps, columns
root xyz (m), root quat (x, y, z, w), 29 joint angles (rad).

Output arrays (frames of all clips concatenated):
  qpos  (N, 36): root pos, root quat (w, x, y, z), joint pos
  qvel  (N, 35): root lin vel (world), root ang vel (world), joint vel
  starts, lengths (C,), names (C,)
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from allmimic.rotations import mat_to_axis_angle, quat_conj, quat_mul, quat_slerp, quat_to_mat


def resample(root_pos, root_quat, dof, fps_in, fps_out):
    n = root_pos.shape[0]
    duration = (n - 1) / fps_in
    times = torch.arange(0, duration, 1.0 / fps_out, dtype=torch.float64)
    phase = times / duration
    i0 = torch.floor(phase * (n - 1)).long().clamp(max=n - 1)
    i1 = torch.minimum(i0 + 1, torch.tensor(n - 1))
    blend = (phase * (n - 1) - i0).to(torch.float64)
    lerp = lambda a: a[i0] * (1 - blend[:, None]) + a[i1] * blend[:, None]
    return lerp(root_pos), quat_slerp(root_quat[i0], root_quat[i1], blend), lerp(dof)


def velocities(root_pos, root_quat, dof, dt):
    lin = torch.gradient(root_pos, spacing=dt, dim=0)[0]
    dof_vel = torch.gradient(dof, spacing=dt, dim=0)[0]
    if root_quat.shape[0] >= 3:
        q_rel = quat_mul(root_quat[2:], quat_conj(root_quat[:-2]))
        w = mat_to_axis_angle(quat_to_mat(q_rel)) / (2 * dt)
        ang = torch.cat([w[:1], w, w[-1:]], dim=0)
    else:
        ang = torch.zeros_like(lin)
    return lin, ang, dof_vel


def load_bones_csv(path):
    """BONES-SEED G1 CSV: header row; Frame, root_translate{X,Y,Z} (cm),
    root_rotate{X,Y,Z} (deg, scipy 'xyz' as in SONIC's converter), 29 *_dof (deg)."""
    from scipy.spatial.transform import Rotation

    with open(path) as f:
        header = f.readline().strip().split(',')
    a = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    col = {n: i for i, n in enumerate(header)}
    pos = a[:, [col['root_translateX'], col['root_translateY'], col['root_translateZ']]] / 100.0
    eul = a[:, [col['root_rotateX'], col['root_rotateY'], col['root_rotateZ']]]
    quat = Rotation.from_euler('xyz', eul, degrees=True).as_quat()[:, [3, 0, 1, 2]]
    dof_cols = [i for i, n in enumerate(header) if n.endswith('_dof')]
    assert len(dof_cols) == 29, path
    dof = np.deg2rad(a[:, dof_cols])
    return torch.tensor(pos), torch.tensor(quat), torch.tensor(dof)


def load_lafan_csv(path):
    a = np.loadtxt(path, delimiter=',')
    pos = torch.tensor(a[:, :3])
    quat = torch.tensor(a[:, [6, 3, 4, 5]])  # xyzw -> wxyz
    dof = torch.tensor(a[:, 7:])
    return pos, quat, dof


def convert_one(job):
    f, fmt, fps_in, fps_out = job
    try:
        pos, quat, dof = (load_bones_csv if fmt == 'bones' else load_lafan_csv)(f)
    except Exception as e:  # noqa: BLE001
        print('skip', f, e)
        return None
    if pos.shape[0] < 3:
        return None
    quat = quat / quat.norm(dim=-1, keepdim=True)
    # Keep quaternion sign continuous before interpolation.
    sign = torch.ones(quat.shape[0], dtype=quat.dtype)
    dots = (quat[1:] * quat[:-1]).sum(-1)
    flips = torch.cumprod(torch.where(dots < 0, -1.0, 1.0).to(quat.dtype), 0)
    sign[1:] = flips
    quat = quat * sign[:, None]
    pos, quat, dof = resample(pos, quat, dof, fps_in, fps_out)
    lin, ang, dof_vel = velocities(pos, quat, dof, 1.0 / fps_out)
    return (
        torch.cat([pos, quat, dof], dim=-1).float().numpy(),
        torch.cat([lin, ang, dof_vel], dim=-1).float().numpy(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='directory with CSV clips (searched recursively)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--fps-in', type=float, default=30.0)
    ap.add_argument('--fps-out', type=float, default=50.0)
    ap.add_argument('--format', default='lafan', choices=['lafan', 'bones'])
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, '**', '*.csv'), recursive=True))
    print(f'{len(files)} clips')
    job = [(f, args.format, args.fps_in, args.fps_out) for f in files]
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context('spawn').Pool(args.workers) as pool:
            results = pool.map(convert_one, job, chunksize=64)
    else:
        results = [convert_one(j) for j in job]
    qpos_all, qvel_all, names, lengths = [], [], [], []
    for f, r in zip(files, results):
        if r is None:
            continue
        qpos_all.append(r[0])
        qvel_all.append(r[1])
        names.append(os.path.relpath(f, args.input))
        lengths.append(r[0].shape[0])
    lengths = np.array(lengths, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    np.savez(
        args.output,
        qpos=np.concatenate(qpos_all),
        qvel=np.concatenate(qvel_all),
        starts=starts,
        lengths=lengths,
        names=np.array(names),
        fps=np.array(args.fps_out),
    )
    hours = lengths.sum() / args.fps_out / 3600
    print(f'saved {args.output}: {len(names)} clips, {lengths.sum()} frames, {hours:.3f} h')


if __name__ == '__main__':
    main()
