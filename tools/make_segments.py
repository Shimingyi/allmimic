"""Cut a motion library into fixed-length segments, so datasets with very different clip lengths compare fairly.

  python tools/make_segments.py --lib data/library_eval/lafan1.npz --seconds 10
  #   -> data/library_eval/lafan1_seg10.npz + data/library_eval/lafan1_seg10_manifest.csv

Full-clip playback penalizes long clips (LAFAN1 averages 220 s, BONES-SEED 7 s): one failure anywhere ends the clip.
Segment mode cuts every clip into non-overlapping windows of --seconds (+1 frame; a clip shorter than that stays whole,
a tail shorter than --min-seconds is dropped) and stores each window as its own clip <clip>__s<k>, so the evaluator
starts every window from the reference state. Same rule and code as tools/build_allmimic_library.py --segments
(cut_segments). Works on both library storages (compact_v1, upstream fp32) and keeps the input's; 50 Hz only.
The manifest (name, frames, fps, duration_s, source, start_frame) feeds tools/filter_infeasible.py flag --segments.
2026-10-24: release version (the development version cut per-clip npz files of another format).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_allmimic_library import FPS_OUT, cut_segments, write_manifest  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--lib', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--min-seconds', type=float, default=2.0)
    ap.add_argument('--out', type=Path, help='default: <lib stem>_seg<seconds>.npz next to the input')
    args = ap.parse_args(argv)

    out = args.out or args.lib.with_name(f'{args.lib.stem}_seg{args.seconds:g}.npz')
    d = np.load(args.lib, allow_pickle=False)
    if float(d['fps']) != FPS_OUT:
        raise SystemExit(f'{args.lib}: {float(d["fps"]):g} Hz, expected {FPS_OUT:g} Hz')
    compact = 'storage' in d.files and str(d['storage']) == 'compact_v1'
    a, b = (d['root_pos'], d['rest']) if compact else (d['qpos'], d['qvel'])  # (N,3) + (N,68) or (N,36) + (N,35)
    names = [str(n) for n in d['names']]
    got = {n: (a[s : s + n_], b[s : s + n_]) for n, s, n_ in zip(names, d['starts'], d['lengths'])}
    keep, seg, rows = cut_segments(names, got, args.seconds, args.min_seconds)
    lengths = np.array([seg[n][0].shape[0] for n in keep], dtype=np.int64)
    common = {'starts': np.concatenate([[0], np.cumsum(lengths)[:-1]]), 'lengths': lengths, 'names': np.array(keep),
              'fps': np.array(FPS_OUT)}  # fmt: skip
    A, B = np.concatenate([seg[n][0] for n in keep]), np.concatenate([seg[n][1] for n in keep])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp.npz')
    if compact:
        np.savez(tmp, root_pos=A, rest=B, storage=np.array('compact_v1'), **common)
    else:
        np.savez(tmp, qpos=A, qvel=B, **common)
    os.replace(tmp, out)
    write_manifest(out.with_name(out.stem + '_manifest.csv'), rows)
    hours = sum(r['duration_s'] for r in rows) / 3600
    print(f'{args.lib}: {len(names)} clips -> {len(keep)} segments of <= {args.seconds:g} s ({hours:.2f} h) in {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
