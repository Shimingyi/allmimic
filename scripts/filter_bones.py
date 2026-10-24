"""Appendix E: filter the raw BONES-SEED G1 library to clips feasible on flat ground.

1. Name filter: drop clips whose name denotes terrain/object contact (SONIC's keyword list
   minus its acrobatics terms, plus terms it misses; the paper keeps flips and side rolls).
2. Feasibility filter from forward kinematics, z_min(t) = lowest body-link height:
   - elevated support: z_min > 0.20 m for >= 0.6 s
   - ground penetration: z_min < -0.12 m, or < -0.08 m for > 80% of the clip
   - floating feet: >= 0.5 s with both feet (exact capsule-to-plane distance) > 0.02 m while
     the pelvis is upright, at standing height and the body nearly stationary.
Writes the filtered library and a JSON report with per-stage counts.
"""

import argparse
import json

import mujoco
import numpy as np
import torch

from allmimic.kinematics import FastTreeKinematics
from allmimic.rotations import quat_to_mat

NAME_TERMS = [
    # SONIC's data-processing list without its acrobatics terms.
    'bed',
    'bike',
    'chair',
    'climb',
    'com_up_50cm',
    'sitting',
    'step_on',
    'seat',
    'table',
    '_sit_',
    'sit_',
    'ladder',
    'crutch',
    '_ride_',
    'scooter',
    'stepdown',
    '50cm_box_',
    'on_box',
    'fall_from',
    'on_1m',
    'form_box',
    'off_1m',
    '230m',
    'jump_over_obstacle_',
    'lift_crate_come_up_',
    'stair',
    'box_jump',
    'box_dips',
    'walking_on_edge',
    'push_obstacle',
    # Terms named in the paper / missed by SONIC's list.
    'box',
    'sit',
    'jump_off_50cm',
    'look_over_fence',
    'come_up_50cm',
    'fence',
    'bench',
    'sofa',
    'couch',
    'stool',
    'car_',
    'vehicle',
    'wall_',
    'railing',
    'ledge',
    'platform',
]


def name_filter(names):
    keep = []
    for i, n in enumerate(names):
        low = n.lower()
        if not any(t in low for t in NAME_TERMS):
            keep.append(i)
    return keep


def runs_at_least(mask, n):
    """True if mask (T,) has a run of >= n consecutive True values."""
    if n <= 1:
        return bool(mask.any())
    m = mask.astype(np.int32)
    c = np.convolve(m, np.ones(n, dtype=np.int32), mode='valid')
    return bool((c >= n).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--report', required=True)
    args = ap.parse_args()
    dev = 'cuda'
    d = np.load(args.input)
    names = [str(x) for x in d['names']]
    starts, lengths = d['starts'], d['lengths']
    fps = float(d['fps'])
    keep1 = name_filter(names)
    print(f'raw {len(names)}; after name filter {len(keep1)}')

    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.entity import Entity

    m = Entity(get_g1_robot_cfg()).spec.compile()
    kin = FastTreeKinematics(m, dev, root_body=1)
    foot_caps = []
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        if nm.endswith('_collision') and 'foot' in nm and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            side = 0 if nm.startswith('left') else 1
            body = kin.body_names.index(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]))
            # Capsule: center/frame in body, half-length along local z, radius.
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            axis = R.reshape(3, 3)[:, 2] * m.geom_size[g, 1]
            foot_caps.append((side, body, m.geom_pos[g] + axis, m.geom_pos[g] - axis, m.geom_size[g, 0]))
    assert len(foot_caps) == 14, len(foot_caps)

    qpos = d['qpos']
    keep2, reasons = [], {'elevated': 0, 'penetration': 0, 'floating_feet': 0}
    for k, c in enumerate(keep1):
        s, L = starts[c], lengths[c]
        qp = torch.tensor(qpos[s : s + L], device=dev)
        pos, R = qp[:, :3], quat_to_mat(qp[:, 3:7])
        with torch.no_grad():
            P, Rb = kin.forward(pos, R, qp[:, 7:])
        zmin = P[..., 2].min(-1).values.cpu().numpy()
        feet = torch.full((L, 2), 1e9, device=dev)
        for side, b, p1, p2, rad in foot_caps:
            e = torch.stack(
                [torch.tensor(p1, device=dev, dtype=torch.float32), torch.tensor(p2, device=dev, dtype=torch.float32)]
            )
            w = P[:, b, None, :] + (Rb[:, b, None] @ e[None, :, :, None]).squeeze(-1)
            dist = w[..., 2].min(-1).values - rad
            feet[:, side] = torch.minimum(feet[:, side], dist)
        feet = feet.cpu().numpy()
        upright = (R[:, 2, 2] > 0.9).cpu().numpy()
        standing = (pos[:, 2] > 0.65).cpu().numpy()
        vel = np.linalg.norm(np.gradient(qpos[s : s + L, :3], 1 / fps, axis=0), axis=-1)
        qd = np.abs(np.gradient(qpos[s : s + L, 7:], 1 / fps, axis=0)).mean(-1)
        stationary = (vel < 0.2) & (qd < 0.3)
        bad = None
        if runs_at_least(zmin > 0.20, int(round(0.6 * fps))):
            bad = 'elevated'
        elif zmin.min() < -0.12 or (zmin < -0.08).mean() > 0.8:
            bad = 'penetration'
        elif runs_at_least((feet > 0.02).all(-1) & upright & standing & stationary, int(round(0.5 * fps))):
            bad = 'floating_feet'
        if bad:
            reasons[bad] += 1
        else:
            keep2.append(c)
        if k % 5000 == 0:
            print(f'  {k}/{len(keep1)} kept {len(keep2)} {reasons}', flush=True)
    print(f'after feasibility filter {len(keep2)} {reasons}')
    sel = np.concatenate([np.arange(starts[c], starts[c] + lengths[c]) for c in keep2])
    new_len = lengths[keep2]
    np.savez(
        args.output,
        qpos=d['qpos'][sel],
        qvel=d['qvel'][sel],
        starts=np.concatenate([[0], np.cumsum(new_len)[:-1]]),
        lengths=new_len,
        names=np.array([names[c] for c in keep2]),
        fps=d['fps'],
    )
    json.dump(
        {
            'raw': len(names),
            'after_name_filter': len(keep1),
            'after_feasibility_filter': len(keep2),
            'feasibility_reasons': reasons,
            'hours': float(new_len.sum() / fps / 3600),
            'paper': {'raw': 142220, 'after_name_filter': 129667, 'after_feasibility_filter': 129514, 'hours': 264.56},
        },
        open(args.report, 'w'),
        indent=1,
    )


if __name__ == '__main__':
    main()
