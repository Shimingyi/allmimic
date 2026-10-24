"""CPU checks of the diverged-world guard (CHANGES #5).

PYTHONPATH=. CUDA_VISIBLE_DEVICES= python tests/test_nan_guard.py
"""

import torch

from allmimic.replay import SDIM, ReplayBuffer
from allmimic.sim_env import TrackingSimEnv
from allmimic.task import NJ
from allmimic.train import zero_if_nonfinite


class StubTask:
    def obs_clean(self, st, a_prev, bias):
        return torch.zeros(a_prev.shape[:-1] + (7,))


def test_sanitize():
    env = object.__new__(TrackingSimEnv)
    N = 6
    s = {
        'pos': torch.randn(N, 3), 'rot': torch.eye(3).repeat(N, 1, 1), 'q': torch.randn(N, NJ),
        'v': torch.randn(N, 3), 'w': torch.randn(N, 3), 'qd': torch.randn(N, NJ),
    }  # fmt: skip
    s['qd'][1, 4] = float('nan')
    s['rot'][3, 0, 0] = float('inf')
    clean, finite = env._sanitize(s)
    assert finite.tolist() == [True, False, True, False, True, True]
    assert all(torch.isfinite(v).all() for v in clean.values())
    assert torch.equal(clean['rot'][3], torch.eye(3)) and torch.equal(clean['qd'][1], torch.zeros(NJ))
    for k in s:  # healthy worlds untouched
        assert torch.equal(clean[k][finite], s[k][finite])


def test_zero_if_nonfinite():
    p = torch.nn.Parameter(torch.ones(4))
    p.grad = torch.tensor([1.0, float('nan'), 2.0, 3.0])
    zero_if_nonfinite([p], torch.tensor(float('nan')))
    assert torch.equal(p.grad, torch.zeros(4))
    p.grad = torch.tensor([1.0, 2.0, 3.0, 4.0])
    zero_if_nonfinite([p], torch.tensor(5.0))
    assert torch.equal(p.grad, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_replay_masks_diverged_slot():
    torch.manual_seed(0)
    n, L, k = 3, 20, 4
    buf = ReplayBuffer(n, L, 1 + NJ + 3, 'cpu')
    for t in range(L):
        ep = torch.zeros(n, dtype=torch.long)
        if t == 10:
            ep[1] = -1  # env 1 diverged at step 10
        buf.add(torch.randn(n, SDIM), torch.randn(n, NJ), torch.randn(n, SDIM), torch.zeros(n, 1 + NJ + 3),
                torch.full((n,), t), ep)  # fmt: skip
    bad = buf._age_index(torch.tensor(L - 1 - 10))  # ring index of step 10
    for _ in range(50):
        _, _, _, _, mask = buf.sample_windows(256, k)
    # recompute which sampled windows touch the bad slot of env 1 via the episode ids: any mixed window is invalid
    env = torch.randint(0, n, (4096,))
    age0 = torch.randint(k - 1, buf.size, (4096,))
    idx = buf._age_index(age0[:, None] - torch.arange(k))
    ep = buf.ep[idx, env[:, None].expand(-1, k)]
    mask = (ep == ep[:, :1]).all(-1) & (ep[:, 0] >= 0)
    touches = ((idx == bad) & (env[:, None] == 1)).any(-1)
    assert touches.any() and not mask[touches].any(), 'windows through a diverged slot must be masked'
    out = buf.sample_starts(4096, 3, StubTask(), with_ok=True)
    assert len(out) == 6 and out[5].dtype == torch.bool and (~out[5]).any() and out[5].any()


def main():
    test_sanitize()
    test_zero_if_nonfinite()
    test_replay_masks_diverged_slot()
    print('nan guard checks passed')


if __name__ == '__main__':
    main()
