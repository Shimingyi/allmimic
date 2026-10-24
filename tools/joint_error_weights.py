"""Per-joint error-propagation weights for the G1 (Holden, "Joint Error Propagation", theorangeduck.com 2024).

  python tools/joint_error_weights.py [--out joint_weights.csv]     # CPU, needs mujoco (+ mjlab for the G1 MJCF)

A small error dq at hinge j moves every point x of its subtree by dq * |a_j x (x - p_j)| (a_j: hinge axis, p_j: joint
anchor, rest pose = qpos0). Weights per joint, all normalized to mean 1 over the 29 actuated joints (loss scale kept):
  mesh_lin    article's mesh weighting: sum over subtree visual-mesh vertices of area * |x - p_j|
  axis_lin    same with the hinge lever |a_j x (x - p_j)| (G1 joints are 1-DoF)
  axis_sq     sum of area * |a_j x (x - p_j)|^2: matches a squared-error loss (displacement^2 per dq^2)
  tracked_sq  sum over the 14 tracked reward bodies in the subtree of |a_j x (p_k - p_j)|^2 + 1 per body (orientation
              term): what the BeyondMimic reward (allmimic.task.reward) actually sees
Use: weights on the q / qd dims of the WM state error (allmimic.networks.state_error; TrainCfg.wm_joint_weights,
CHANGES #21), which the upstream WM loss weights uniformly after per-dim std normalization; the policy's BPTT loss
already goes through FK. allmimic/joint_weights.py holds the tracked_sq column as 'tracked' and axis_sq as 'axis_sq'.
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_infeasible import g1_xml  # noqa: E402

TRACKED = ('pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link',
           'right_knee_link', 'right_ankle_roll_link', 'torso_link', 'left_shoulder_roll_link', 'left_elbow_link',
           'left_wrist_yaw_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link')  # fmt: skip


def subtree(m, body):
    """body id -> set of body ids in its subtree (itself included)"""
    out = {body}
    for b in range(body + 1, m.nbody):
        if m.body_parentid[b] in out:
            out.add(b)
    return out


def mesh_points(m, d):
    """visual-mesh vertices in world frame with per-vertex area (a third of each adjacent face) -> (pts, area, body)"""
    pts, area, body = [], [], []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        V = m.mesh_vert[v0 : v0 + nv].astype(np.float64)
        F = m.mesh_face[f0 : f0 + nf]
        tri = V[F]
        fa = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        va = np.zeros(nv)
        np.add.at(va, F.ravel(), np.repeat(fa / 3, 3))
        R = d.geom_xmat[g].reshape(3, 3)
        pts.append(V @ R.T + d.geom_xpos[g])
        area.append(va)
        body.append(np.full(nv, m.geom_bodyid[g]))
    return np.concatenate(pts), np.concatenate(area), np.concatenate(body)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, default=Path('joint_weights.csv'))
    args = ap.parse_args(argv)
    m = mujoco.MjModel.from_xml_path(g1_xml())
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    pts, area, pbody = mesh_points(m, d)
    tracked = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in TRACKED]
    rows = []
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        b = m.jnt_bodyid[j]
        sub = subtree(m, b)
        p, a = d.xanchor[j], d.xaxis[j]
        sel = np.isin(pbody, list(sub))
        r = pts[sel] - p
        lever = np.linalg.norm(np.cross(a, r), axis=1)
        trk = [k for k in tracked if k in sub]
        tl = np.linalg.norm(np.cross(a, d.xpos[trk] - p), axis=1) if trk else np.zeros(0)
        rows.append({
            'joint': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
            'mesh_lin': (area[sel] * np.linalg.norm(r, axis=1)).sum(), 'axis_lin': (area[sel] * lever).sum(),
            'axis_sq': (area[sel] * lever**2).sum(), 'tracked_sq': (tl**2).sum() + len(trk),
            'subtree_area': area[sel].sum(),
        })  # fmt: skip
    for k in ('mesh_lin', 'axis_lin', 'axis_sq', 'tracked_sq'):
        v = np.array([r[k] for r in rows]) + 1e-3
        v = v / v.mean()
        for r, x in zip(rows, v):
            r[k] = x
    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = ['joint', 'mesh_lin', 'axis_lin', 'axis_sq', 'tracked_sq', 'subtree_area']
    with open(args.out, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join([r['joint']] + [f'{r[k]:.4f}' for k in keys[1:]]) + '\n')
    print(f'{"joint":28s} {"mesh_lin":>8s} {"axis_lin":>8s} {"axis_sq":>8s} {"tracked":>8s}')
    for r in rows:
        print(f'{r["joint"]:28s} {r["mesh_lin"]:8.3f} {r["axis_lin"]:8.3f} {r["axis_sq"]:8.3f} {r["tracked_sq"]:8.3f}')
    for k in ('mesh_lin', 'axis_lin', 'axis_sq', 'tracked_sq'):
        v = np.array([r[k] for r in rows])
        print(f'{k:10s} max/min {v.max() / v.min():8.1f}   max {v.max():.2f}   min {v.min():.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
