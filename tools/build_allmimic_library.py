"""Build an allmimic motion library (.npz) for a clip list, straight from G1 CSVs or a tar of them.

Each clip goes through allmimic's own scripts/prepare_motions.convert_one (dhmbb2's resampling and finite
differences), so the library matches what upstream would produce; only the container differs:
  --storage compact   compact_v1: root_pos float32 (N,3) + rest float16 (N,68) -- see allmimic/motion_lib.py
  --storage fp32      upstream: qpos (N,36) / qvel (N,35) float32
Clips appear in clip-list order, named by CSV stem (the names in data/splits/*). Run from the repository root:

  PYTHONPATH=. python tools/build_allmimic_library.py --csv-dir data/raw/bones_seed/g1 --format bones --fps-in 120 \
      --clip-list data/splits/bones_seed/train.txt --out data/library/bones_train.npz
  PYTHONPATH=. python tools/build_allmimic_library.py --csv-dir data/raw/lafan1_g1/g1 --format lafan --fps-in 30 \
      --clip-list data/lafan1_clips.txt --out data/library_eval/lafan1.npz
  --tar g1.tar.gz instead of --csv-dir reads the CSVs straight from an archive; add --segments 10 for the 10 s
  segment version of a test set (+ <out>_manifest.csv for tools/filter_infeasible.py), --fps-map for per-clip input
  rates (AMASS)
"""

import argparse
import csv
import os
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SHM = '/dev/shm' if os.path.isdir('/dev/shm') else None
FPS_OUT = 50.0
MANIFEST_COLS = ('name', 'frames', 'fps', 'duration_s', 'source', 'start_frame')


def convert(job):
    """(stem, csv bytes or path, fmt, fps_in) -> (stem, qpos (T,36) f32, qvel (T,35) f32) or (stem, None, None)"""
    import torch

    from scripts.prepare_motions import convert_one  # allmimic root on sys.path

    torch.set_num_threads(1)  # one thread per worker: N workers x all-core torch pools oversubscribe the host

    stem, src, fmt, fps_in = job
    if isinstance(src, bytes):
        with tempfile.NamedTemporaryFile(suffix='.csv', dir=SHM) as f:
            f.write(src)
            f.flush()
            r = convert_one((f.name, fmt, fps_in, FPS_OUT))
    else:
        r = convert_one((src, fmt, fps_in, FPS_OUT))
    return (stem, None, None) if r is None else (stem, r[0], r[1])


def cut_segments(names, got, seconds, min_seconds=2.0):
    """clips -> non-overlapping windows of `seconds` (+1 frame), named <clip>__s<k>; a clip shorter than one
    window stays whole, a tail shorter than min_seconds is dropped (tools/make_segments.py uses the same function).
    -> segment names, {name: (qpos, qvel)}, manifest rows {name, frames, fps, duration_s, source, start_frame}"""
    win, min_len = int(round(seconds * FPS_OUT)), int(round(min_seconds * FPS_OUT))
    out_names, out, rows = [], {}, []
    for n in names:
        qp, qv = got[n]
        starts = list(range(0, qp.shape[0], win))
        for k, s in enumerate(starts):
            e = min(s + win + 1, qp.shape[0])  # +1: a window of `win` steps needs win + 1 frames
            if e - s < min_len and len(starts) > 1:
                continue
            name = f'{n}__s{k:03d}'
            out_names.append(name)
            out[name] = (qp[s:e], qv[s:e])
            rows.append({'name': name, 'frames': e - s, 'fps': FPS_OUT, 'duration_s': (e - s - 1) / FPS_OUT,
                         'source': n, 'start_frame': s})  # fmt: skip
    return out_names, out, rows


def write_manifest(path, rows):
    """segment rows -> csv (read by tools/filter_infeasible.py flag --segments), atomic"""
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def sources(args, wanted):
    """yield (stem, csv bytes or path) for the wanted stems"""
    if args.csv_dir:
        for p in sorted(Path(args.csv_dir).rglob('*.csv')):
            if p.stem in wanted:
                yield p.stem, str(p)
        return
    with tarfile.open(args.tar, mode='r|gz') as tar:
        for m in tar:
            stem = Path(m.name).stem
            if m.isfile() and m.name.endswith('.csv') and stem in wanted:
                yield stem, tar.extractfile(m).read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--tar', type=Path)
    src.add_argument('--csv-dir', type=Path)
    ap.add_argument('--format', required=True, choices=['bones', 'lafan'])
    ap.add_argument('--fps-in', type=float, required=True)
    ap.add_argument('--clip-list', type=Path, required=True, help='one CSV stem per line')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fps-map', type=Path, help='csv with name,fps columns overriding --fps-in per clip (AMASS)')
    ap.add_argument('--segments', type=float, default=0, help='cut clips into windows of this many seconds (0 = off)')
    ap.add_argument('--storage', default='compact', choices=['compact', 'fp32'])
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.clip_list.read_text().splitlines() if n.strip()]
    wanted = set(names)
    fps_of = {r['name']: float(r['fps']) for r in csv.DictReader(open(args.fps_map))} if args.fps_map else {}
    got, t0 = {}, time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        pending = []
        for stem, s in sources(args, wanted):
            pending.append(ex.submit(convert, (stem, s, args.format, fps_of.get(stem, args.fps_in))))
            if len(pending) >= 4 * args.workers:  # bound memory: drain the oldest half
                for fut in pending[: 2 * args.workers]:
                    stem_, qp, qv = fut.result()
                    got[stem_] = (qp, qv)
                pending = pending[2 * args.workers :]
                if len(got) % 5000 < 2 * args.workers:
                    print(f'  {len(got)}/{len(names)} clips, {time.time() - t0:.0f}s', flush=True)
        for fut in pending:
            stem_, qp, qv = fut.result()
            got[stem_] = (qp, qv)

    missing = [n for n in names if n not in got]
    failed = [n for n in names if n in got and got[n][0] is None]
    if missing:
        raise SystemExit(f'{len(missing)} clips not found, e.g. {missing[:3]}')
    keep = [n for n in names if got[n][0] is not None]
    if args.segments:
        keep, got, rows = cut_segments(keep, got, args.segments)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(args.out.with_name(args.out.stem + '_manifest.csv'), rows)
    lengths = np.array([got[n][0].shape[0] for n in keep], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    qpos = np.concatenate([got[n][0] for n in keep])
    qvel = np.concatenate([got[n][1] for n in keep])
    common = dict(starts=starts, lengths=lengths, names=np.array(keep), fps=np.array(FPS_OUT))
    tmp = args.out.with_suffix('.tmp.npz')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.storage == 'compact':
        rest = np.concatenate([qpos[:, 3:7], qpos[:, 7:], qvel], axis=1).astype(np.float16)  # quat, q, v, w, qd
        np.savez(tmp, root_pos=qpos[:, :3].astype(np.float32), rest=rest, storage=np.array('compact_v1'), **common)
    else:
        np.savez(tmp, qpos=qpos, qvel=qvel, **common)
    os.replace(tmp, args.out)
    gb = os.path.getsize(args.out) / 1e9
    print(f'saved {args.out}: {len(keep)} clips ({len(failed)} failed to convert), {qpos.shape[0]} frames '
          f'({qpos.shape[0] / FPS_OUT / 3600:.2f} h), {gb:.2f} GB, {args.storage}, {time.time() - t0:.0f}s')  # fmt: skip
    if failed:
        print('failed:', failed[:10])
    return 0


if __name__ == '__main__':
    sys.exit(main())
