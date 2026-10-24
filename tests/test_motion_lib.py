"""Equivalence checks for CHANGES #6 (compact motion library, no per-frame index tables). CPU, needs data.

  PYTHONPATH=. python tests/test_motion_lib.py --fp32 lafan1_fp32.npz --compact lafan1_compact.npz \
      --upstream data/processed/lafan1_50hz.npz
  fp32 / compact: built by tools/build_allmimic_library.py from the same CSVs; upstream: built by prepare_motions.py
"""

import argparse
import sys

import numpy as np
import torch

from allmimic.motion_lib import MotionLibrary
from legacy_motion_lib import MotionLibrary as LegacyLibrary  # tests/ is on sys.path (script dir / conftest)


def same(a, b, what):
    assert a.shape == b.shape and torch.equal(a, b), f'{what} differs'


def check_against_legacy(path, **kw):
    new, old = MotionLibrary(path, 'cpu', **kw), LegacyLibrary(path, 'cpu', **kw)
    for k in ('root_pos', 'root_quat', 'q', 'root_lin_vel', 'root_ang_vel', 'qd', 'starts', 'lengths', 'ends',
              'bin_clip', 'bin_start', 'bin_end'):  # fmt: skip
        same(getattr(new, k), getattr(old, k), k)
    same(new.clip_of_frame, old.clip_of_frame, 'clip_of_frame')
    same(new.bin_of_frame, old.bin_of_frame, 'bin_of_frame')
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, new.num_frames, (4096, 11), generator=g)
    ref = torch.randint(0, new.num_frames, (4096,), generator=g)
    for k, v in new.frame(idx).items():
        same(v, old.frame(idx)[k], f'frame[{k}]')
    same(new.clamp_to_clip(idx, ref), old.clamp_to_clip(idx, ref), 'clamp_to_clip')
    same(new.remaining(idx), old.remaining(idx), 'remaining')
    same(new.bin_of(idx), old.bin_of_frame[idx], 'bin_of')
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fp32', required=True)
    ap.add_argument('--compact', required=True)
    ap.add_argument('--upstream', required=True)
    args = ap.parse_args(argv)

    # 1. the build tool's fp32 library has the same arrays as prepare_motions.py (names are stems vs paths)
    a, b = np.load(args.fp32), np.load(args.upstream)
    assert np.array_equal(a['qpos'], b['qpos']) and np.array_equal(a['qvel'], b['qvel']), 'arrays differ from upstream'
    assert [str(n) for n in a['names']] == [str(n).split('/')[-1][: -len('.csv')] for n in b['names']]

    # 2. new library == frozen upstream library, bit for bit (whole library and a prefix selection)
    lib32 = check_against_legacy(args.fp32)
    check_against_legacy(args.fp32, prefixes='walk,dance')

    # 3. compact storage: root position exact, the rest within float16 resolution
    libc = MotionLibrary(args.compact, 'cpu')
    assert libc.compact and not lib32.compact
    idx = torch.arange(0, lib32.num_frames, 7)
    f32, fc = lib32.frame(idx), libc.frame(idx)
    same(fc['pos'], f32['pos'], 'compact root_pos')
    tol = {'quat': 1e-3, 'rot': 2e-3, 'q': 2e-3, 'v': 5e-3, 'w': 1e-2, 'qd': 3e-2}
    for k, t in tol.items():
        err = ((fc[k] - f32[k]).abs() / f32[k].abs().clamp_min(1.0)).max().item()
        assert err < t, f'compact {k}: rel/abs err {err:.2e} >= {t}'
        print(f'  compact {k:5s} max err {err:.1e}')
    for k in ('starts', 'lengths', 'bin_start'):
        same(getattr(libc, k), getattr(lib32, k), f'compact {k}')

    # 4. memory per frame on the device
    def per_frame(lib):
        tensors = (lib.root_pos, lib.root_quat, lib.q, lib.root_lin_vel, lib.root_ang_vel, lib.qd)
        storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}  # views share one
        nbytes = sum(storages.values())
        return nbytes / lib.num_frames

    print(f'  bytes/frame: fp32 {per_frame(lib32):.0f}, compact {per_frame(libc):.0f} (+16 before #6 for index tables)')
    print('motion lib checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
