"""GPU check of TrainCfg.bptt_checkpoint (CHANGES #17): activation recomputation per imagined step.

  PYTHONPATH=. python tests/gpu/test_bptt_checkpoint.py --motions <lib.npz> [--pi-batch 4096]

1. same inputs and seed, noise on: J and the policy gradient with recomputation equal the stored-activation path
   bit for bit (checkpoint replays the RNG state; the step is deterministic)
2. peak memory of update_policy() and its time per call, both paths
"""

import argparse
import sys
import time

import torch

from allmimic.train import Trainer, TrainCfg


def run(tr, inp, seed):
    """-> (J, flat policy gradient) of one BPTT loss from fixed inputs"""
    for p in tr.wm.net.parameters():
        p.requires_grad_(False)
    tr.opt_pi.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    J = tr.rollout(*inp)[0]
    (-J.mean() / tr.cfg.horizon).backward()
    g = torch.cat([p.grad.flatten() for g_ in tr.opt_pi.param_groups for p in g_['params']])
    for p in tr.wm.net.parameters():
        p.requires_grad_(True)
    tr.opt_pi.zero_grad(set_to_none=True)
    tr.critic.zero_grad(set_to_none=True)
    return J.detach(), g


def peak_and_time(tr, n=10):
    for _ in range(2):
        tr.update_policy()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n):
        tr.update_policy()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9, (time.perf_counter() - t0) / n * 1e3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--motions', required=True)
    ap.add_argument('--out', default='/tmp/test_bptt_checkpoint')
    ap.add_argument('--pi-batch', type=int, default=4096)
    args = ap.parse_args(argv)
    tr = Trainer(TrainCfg(motions=args.motions, out=args.out, pi_batch=args.pi_batch, eval_every=0))
    while tr.buf.size < tr.cfg.warmup_steps:
        tr.collect()
    for _ in range(10):
        tr.collect()
        tr.update_wm()
        tr.update_policy()
    stored = tr._img_step
    recompute = lambda *a: torch.utils.checkpoint.checkpoint(stored, *a, use_reentrant=False)
    torch.manual_seed(1)
    inp = tr.imagination_inputs()
    J0, g0 = run(tr, inp, seed=2)
    tr._img_step = recompute
    J1, g1 = run(tr, inp, seed=2)
    same = torch.equal(J0, J1) and torch.equal(g0, g1)
    cos = torch.nn.functional.cosine_similarity(g0, g1, dim=0).item()
    print(f'recomputation bit-identical: {same} (grad cos {cos:.6f})', flush=True)
    tr._img_step = stored
    m0, t0 = peak_and_time(tr)
    tr._img_step = recompute
    m1, t1 = peak_and_time(tr)
    print(f'pi_batch {args.pi_batch}: update_policy peak {m0:.2f} -> {m1:.2f} GB, {t0:.1f} -> {t1:.1f} ms '
          f'({t1 / t0:.2f}x time)', flush=True)  # fmt: skip
    print('bptt checkpoint checks', 'passed' if same else 'FAILED')
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main())
