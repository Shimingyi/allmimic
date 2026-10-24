"""CPU checks of the goal encodings (TaskCfg.goal_pos / goal_rot / goal_pos_noise).

PYTHONPATH=. python tests/test_goal_modes.py
"""

import torch

from allmimic.rotations import heading_mat, mtm, mtv, rot6d
from allmimic.task import GOAL_FRAME_DIM, NJ, Task, TaskCfg


def rand_rot(*shape):
    q = torch.nn.functional.normalize(torch.randn(*shape, 4, dtype=torch.float64), dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape(*shape, 3, 3)


def yaw_rot(B, a):
    c, s = torch.cos(a), torch.sin(a)
    R = torch.zeros(B, 3, 3, dtype=torch.float64)
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1
    return R


def make_task(**kw):
    t = object.__new__(Task)  # goal() only needs cfg
    t.cfg = TaskCfg(**kw)
    return t


def main():
    torch.manual_seed(0)
    B, K = 64, 10
    s = {'pos': torch.randn(B, 3, dtype=torch.float64), 'rot': rand_rot(B)}
    ref = {
        'pos': torch.randn(B, K, 3, dtype=torch.float64),
        'rot': rand_rot(B, K),
        'q': torch.randn(B, K, NJ, dtype=torch.float64),
        'qd': torch.randn(B, K, NJ, dtype=torch.float64),
        'v': torch.randn(B, K, 3, dtype=torch.float64),
        'w': torch.randn(B, K, 3, dtype=torch.float64),
    }

    # default == upstream formula
    g = make_task().goal(s, ref).unflatten(-1, (K, GOAL_FRAME_DIM))
    R = s['rot'].unsqueeze(-3)
    assert torch.allclose(g[..., :3], mtv(R, ref['pos'] - s['pos'].unsqueeze(-2)))
    assert torch.allclose(g[..., 3:9], rot6d(mtm(R, ref['rot'])))

    # goal_pos=ref: z is the absolute reference height; xy invariant to a rigid yaw+translation of the robot
    t = make_task(goal_pos='ref')
    g1 = t.goal(s, ref).unflatten(-1, (K, GOAL_FRAME_DIM))
    assert torch.allclose(g1[..., 2], ref['pos'][..., 2])
    assert torch.allclose(g1[:, 0, :2], torch.zeros(B, 2, dtype=torch.float64))
    s2 = {'pos': s['pos'] + torch.randn(B, 3, dtype=torch.float64), 'rot': s['rot']}
    assert torch.allclose(t.goal(s2, ref), t.goal(s, ref)), 'goal_pos=ref must not depend on the robot position'

    # goal_rot=ref (+ goal_pos=ref): no dependence on the robot state at all
    t = make_task(goal_pos='ref', goal_rot='ref')
    s3 = {'pos': torch.randn(B, 3, dtype=torch.float64), 'rot': rand_rot(B)}
    assert torch.allclose(t.goal(s3, ref), t.goal(s, ref)), 'goal_pos=ref, goal_rot=ref must ignore the robot state'
    H0 = heading_mat(ref['rot'][:, 0]).unsqueeze(-3)
    assert torch.allclose(t.goal(s, ref).unflatten(-1, (K, GOAL_FRAME_DIM))[..., 3:9], rot6d(mtm(H0, ref['rot'])))
    # a world yaw rotation of the reference leaves the ref-frame goal unchanged
    Y = yaw_rot(B, torch.rand(B, dtype=torch.float64) * 6.28).unsqueeze(1)
    ref_y = dict(ref, pos=(Y @ ref['pos'].unsqueeze(-1)).squeeze(-1), rot=Y @ ref['rot'])
    gy = t.goal(s, ref_y).unflatten(-1, (K, GOAL_FRAME_DIM))
    g0 = t.goal(s, ref).unflatten(-1, (K, GOAL_FRAME_DIM))
    assert torch.allclose(gy[..., :9], g0[..., :9], atol=1e-9), 'ref-frame goal must be invariant to world yaw'

    # noise: only the 3 position dims of each frame change, bounded by the half-width
    t = make_task(goal_pos_noise=0.25)
    g = t.goal(s, ref)
    gn = t.noisy_goal(g)
    d = (gn - g).unflatten(-1, (K, GOAL_FRAME_DIM))
    assert d[..., 3:].abs().max() == 0 and 0 < d[..., :3].abs().max() <= 0.25
    assert torch.equal(make_task().noisy_goal(g), g)
    # privileged critic goal (CHANGES #4): equals the robot-relative pose part of the upstream goal, ref modes only
    assert not make_task().priv_goal and make_task(goal_pos='ref').priv_goal and make_task(goal_rot='ref').priv_goal
    gp = make_task(goal_pos='ref', goal_rot='ref').goal_priv(s, ref).unflatten(-1, (K, 9))
    gr = make_task().goal(s, ref).unflatten(-1, (K, GOAL_FRAME_DIM))
    assert torch.allclose(gp, gr[..., :9])
    print('goal mode checks passed')


def test_goal_modes():  # pytest entry point
    main()


if __name__ == '__main__':
    main()
