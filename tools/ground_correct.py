"""Lift retargeted reference frames out of the floor: per-frame whole-body z shift so the lowest collision geom of
the G1 does not penetrate z = 0. Joint angles, orientations and root xy stay unchanged.

  # CPU only (OMP_NUM_THREADS=1 recommended; the 240 h BONES train split takes ~1 min with 16 workers)
  python tools/ground_correct.py run --lib data/library_eval/lafan1_seg10.npz --workers 16
  #   -> lafan1_seg10_grounded.npz next to the input, + lafan1_seg10_grounded_clips.csv / _summary.json
  python tools/ground_correct.py run --lib data/library/bones_train.npz --workers 16 \
      --meta data/raw/bones_seed/seed_metadata_v004.csv --report-dir data/ground_report
  # verification: fresh FK on the corrected file, untouched clips bit-identical, velocities, MotionLibrary load
  PYTHONPATH=. python tools/ground_correct.py check --lib X.npz --workers 16   # reads X_grounded.npz

Method:
  lowest(t)  min over the collision capsule-end spheres of (z - radius): tools/filter_infeasible.Body, i.e. the same
             MJCF and the same geom set as mjlab FULL_COLLISION; float16 quaternions renormalized as there.
  raw(t)     max(0, -lowest(t)), m.
  lift(t)    centred moving average (+-HALF) of the running max (+-HALF) of raw. Every averaged max covers t, so
             lift >= raw at every frame and the corrected lowest point is >= 0 by construction; the final clamp
             lowest + lift >= -FLOOR_TOL is only a guard.
  root z     += lift (float64 sum, stored float32).
  root v_z   += d lift / dt by np.gradient at 50 Hz: the same differences as allmimic scripts/prepare_motions.velocities
             (torch.gradient, edge_order 1: central inside, one-sided at the ends), so the stored world-frame velocity
             stays the derivative of the stored root position. Only frames where d lift / dt != 0 are rewritten.
Frames are never lowered: AMASS-G1 clips that float stay as they are. Clips without a penetrating frame are
copied bit for bit. Each clip of the library is corrected on its own (a 10 s segment does not see its neighbours).
Library: compact_v1 (root_pos f32 (N,3), rest f16 (N,68) = [quat wxyz 4, q 29, v 3, w 3, qd 29]), 50 Hz, Z-up, SI.
2026-10-24: first version.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_infeasible import FPS, GROUPS, Body, bones_meta  # noqa: E402

HALF = 0.10  # s, half window of the running max and of the moving average
FLOOR_TOL = 0.005  # m, corrected lowest point >= -FLOOR_TOL (guard; the smoothing alone gives >= 0)
PEN = (0.005, 0.01, 0.03)  # m, penetration depths reported
AIR = 0.03  # m, lowest point above this = airborne frame (handstand start, flight); reported, never touched
CONTACT = 0.02  # m, a body group this close to the floor (or below) before the lift is in contact ...
LOST = 0.03  # m, ... and loses it when the lift leaves it higher than this
STAND = 0.65  # m, pelvis height above which a frame counts as standing (offset estimate, start_diagnostics)
VZ = 35  # column of the root linear velocity z in `rest`
GN = list(GROUPS)  # body group names, column order of the per-group lowest points


# ---- lowest point per frame


_W = {}


def _init(xml):
    _W['body'] = Body(xml)  # rp / rest are inherited from the parent (fork): no copy of a 6.5 GB library


def _work(job):
    """[(clip id, start, end)] -> [(id, (T,) lowest f64, (T,G) per-group lowest f32)], m"""
    body, out = _W['body'], []
    for i, s, e in job:
        qq = np.asarray(_W['rest'][s:e, :33], np.float64)
        quat = qq[:, :4] / np.linalg.norm(qq[:, :4], axis=1, keepdims=True)  # f16 storage: renormalize
        ends, _, _ = body.fk(np.asarray(_W['rp'][s:e], np.float64), quat, qq[:, 4:])
        zl = ends[..., 2] - body.rad  # (T,2C) lowest point of each capsule-end sphere
        zg = np.stack([zl[:, body.group == k].min(1) for k in range(len(GN))], 1)
        out.append((i, zg.min(1), zg.astype(np.float32)))
    return out


def lowest_points(rp, rest, starts, lengths, workers, xml=None, chunk_frames=100_000):
    """library arrays -> (N,) lowest collision point f64, (N,G) per-group lowest f32, m above z = 0"""
    order = np.argsort(-lengths)  # long clips first: better load balance
    chunks, cur, acc = [], [], 0
    for i in order:
        cur.append((int(i), int(starts[i]), int(starts[i] + lengths[i])))
        acc += int(lengths[i])
        if acc >= chunk_frames:
            chunks.append(cur)
            cur, acc = [], 0
    chunks += [cur] if cur else []
    low = np.zeros(len(rp))
    zg = np.zeros((len(rp), len(GN)), np.float32)
    _W['rp'], _W['rest'] = rp, rest
    done, t0 = 0, time.time()
    with mp.get_context('fork').Pool(workers, initializer=_init, initargs=(xml,)) as pool:
        for res in pool.imap_unordered(_work, chunks):
            for i, lo, g in res:
                s = starts[i]
                low[s : s + lengths[i]], zg[s : s + lengths[i]] = lo, g
            done += len(res)
            print(f'\r  FK {done}/{len(starts)} clips, {time.time() - t0:.0f}s', end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return low, zg


# ---- lift


def lift_of(lowest, half=HALF):
    """(T,) lowest collision point (m) -> (T,) lift >= max(0, -lowest) (m) f64; all zeros when nothing penetrates"""
    raw = np.maximum(-np.asarray(lowest, np.float64), 0.0)
    if not raw.any():
        return np.zeros_like(raw)
    h = int(round(half * FPS))
    win = lambda x: sliding_window_view(np.pad(x, h, mode='edge'), 2 * h + 1)  # (T,2h+1), ends padded
    lift = win(win(raw).max(1)).mean(1)  # running max, then moving average: >= raw (see module docstring)
    return np.maximum(lift, raw - FLOOR_TOL)


def apply_lift(rp, rest, starts, lengths, lift):
    """in place: root z += lift, root v_z += d lift / dt, clip by clip -> (C,) bool touched, (N,) d lift / dt (m/s)"""
    touched = np.zeros(len(starts), bool)
    dl_all = np.zeros(len(rp))
    for c, (s, n) in enumerate(zip(starts, lengths)):
        lf = lift[s : s + n]
        if not lf.any():
            continue
        touched[c] = True
        idx = s + np.flatnonzero(lf)
        rp[idx, 2] = (rp[idx, 2].astype(np.float64) + lift[idx]).astype(np.float32)
        dl = np.gradient(lf, 1 / FPS) if n > 1 else np.zeros(n)  # = prepare_motions.velocities' torch.gradient
        dl_all[s : s + n] = dl
        idx = s + np.flatnonzero(dl)
        rest[idx, VZ] = (rest[idx, VZ].astype(np.float64) + dl_all[idx]).astype(np.float16)
    return touched, dl_all


# ---- statistics


def stats(low, lift, zg, pel_z):
    """frames of one group: lowest (T,), lift (T,), per-group lowest (T,G), pelvis z (T,) -> metrics dict (m, m/s)"""
    n = len(low)
    if n == 0:
        return {'frames': 0}
    after, raw = low + lift, np.maximum(-low, 0.0)
    lifted = lift > 0
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else 0.0
    out = {'frames': n, 'lifted_frac': float(lifted.mean())}
    for p in PEN:
        out[f'pen{p * 100:g}_before'] = float((low < -p).mean())
        out[f'pen{p * 100:g}_after'] = float((after < -p).mean())
    out.update(
        min_before=float(low.min()), min_after=float(after.min()), air3_frac=float((low > AIR).mean()),
        lift_median=q(lift, 50), lift_median_lifted=q(lift[lifted], 50), lift_p95=q(lift, 95),
        lift_max=float(lift.max()),
        extra_median=q((lift - raw)[lifted], 50), extra_p90=q((lift - raw)[lifted], 90),  # clearance added by smoothing
        after_median_lifted=q(after[lifted], 50), after_p90_lifted=q(after[lifted], 90),
    )  # fmt: skip
    st = pel_z > STAND
    out['stand_frac'] = float(st.mean())
    out['stand_lowest_median'], out['stand_lift_median'] = q(low[st], 50), q(lift[st], 50)
    lost = lifted[:, None] & (zg < CONTACT) & (zg + lift[:, None] > LOST)  # (T,G) contacts a uniform lift takes off
    out['lost_any_frac'] = float(lost.any(1).mean())  # the floor (another part was deeper): floating / support change
    out.update({f'lost_{g}_frac': float(lost[:, k].mean()) for k, g in enumerate(GN)})
    pen = low < -0.01
    if pen.any():
        deep = np.bincount(np.argmin(zg[pen], 1), minlength=len(GN))
        out['deepest_group_pen1'] = {g: int(deep[k]) for k, g in enumerate(GN) if deep[k]}
    return out


def grouped(keys, starts, lengths, low, lift, zg, pel_z, min_frames=0):
    """(C,) group label per clip -> {label: stats} over the frames of its clips (frames sorted once)"""
    labels, counts = np.unique(np.array(keys), return_counts=True)
    lid = {k: i for i, k in enumerate(labels)}
    frame_lab = np.repeat(np.array([lid[k] for k in keys]), lengths)
    order = np.argsort(frame_lab, kind='stable')
    bounds = np.searchsorted(frame_lab[order], np.arange(len(labels) + 1))
    out = {}
    for i, k in enumerate(labels):
        sel = order[bounds[i] : bounds[i + 1]]
        if len(sel) >= min_frames:
            out[str(k)] = {'clips': int(counts[i]), **stats(low[sel], lift[sel], zg[sel], pel_z[sel])}
    return out


def content_of(dataset, name):
    """dataset, clip name -> category label (tools/failure_report.category: BONES content, LAFAN1 motion, ...)"""
    if 'cat' not in _W:
        from failure_report import category

        _W['cat'] = category
    return _W['cat'](dataset, name)


# ---- run command


def load(path):
    """compact_v1 npz -> {key: array} (all keys, file order)"""
    d = np.load(path, allow_pickle=False)
    assert str(d['storage']) == 'compact_v1' and float(d['fps']) == FPS, path
    return {k: d[k] for k in d.files}


def out_path(lib):
    return Path(lib).with_name(Path(lib).stem + '_grounded.npz')


def cmd_run(args):
    out = Path(args.out) if args.out else out_path(args.lib)
    rep = Path(args.report_dir) if args.report_dir else out.parent
    if out.exists() and not args.force:
        print('exists, skip:', out)
        return 0
    t0 = time.time()
    dataset = args.dataset or Path(args.lib).stem
    d = load(args.lib)
    rp, rest, starts, lengths = d['root_pos'], d['rest'], d['starts'], d['lengths']
    names = [str(n) for n in d['names']]
    print(f'{dataset}: {len(names)} clips, {len(rp)} frames, loaded in {time.time() - t0:.0f}s')
    low, zg = lowest_points(rp, rest, starts, lengths, args.workers, args.xml)
    lift = np.zeros(len(rp))
    for s, n in zip(starts, lengths):
        lift[s : s + n] = lift_of(low[s : s + n])
    pel_z = rp[:, 2].astype(np.float64)  # before the lift (standing = pose, not correction)
    touched, dl = apply_lift(rp, rest, starts, lengths, lift)
    assert np.isfinite(rp).all() and np.isfinite(rest).all()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp.npz')
    np.savez(tmp, **d)  # same keys and order; root_pos / rest were corrected in place
    os.replace(tmp, out)
    print(f'saved {out}: {touched.sum()} / {len(names)} clips lifted, {os.path.getsize(out) / 1e9:.2f} GB')

    # per-clip CSV
    meta = bones_meta(args.meta) if args.meta else None
    content = [content_of(dataset, n) for n in names]
    mcat = [meta.get(n.split('__s')[0], ('', 'unknown'))[1] for n in names] if meta else None
    rep.mkdir(parents=True, exist_ok=True)
    cols = ['name', 'category', 'frames', 'lifted', 'pen1', 'pen3', 'air3', 'min_lowest_cm', 'min_after_cm',
            'max_lift_cm', 'median_lift_cm', 'deepest_group', 'lost_any', 'lost_foot', 'lost_hand', 'lost_knee',
            'lost_seat', 'lost_torso', 'max_dlift_ms', 'max_d2lift_ms2']  # fmt: skip
    if meta:
        cols.insert(2, 'meta_category')
    tmp = rep / f'{out.stem}_clips.csv.tmp'
    with open(tmp, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for c, (s, n) in enumerate(zip(starts, lengths)):
            lo, lf, g, v = low[s : s + n], lift[s : s + n], zg[s : s + n], dl[s : s + n]
            lifted = lf > 0
            lost = lifted[:, None] & (g < CONTACT) & (g + lf[:, None] > LOST)  # (T,G)
            d2 = np.abs(np.diff(v)).max() * FPS if n > 1 else 0.0
            row = [names[c], content[c], n, int(lifted.sum()), int((lo < -0.01).sum()), int((lo < -0.03).sum()),
                   int((lo > AIR).sum()), f'{lo.min() * 100:.2f}', f'{(lo + lf).min() * 100:.2f}',
                   f'{lf.max() * 100:.2f}', f'{np.median(lf) * 100:.2f}', GN[int(np.argmin(g[np.argmin(lo)]))],
                   int(lost.any(1).sum()),
                   *(int(lost[:, GN.index(k)].sum()) for k in ('foot', 'hand', 'knee', 'seat', 'torso')),
                   f'{np.abs(v).max():.4f}', f'{d2:.3f}']  # fmt: skip
            if meta:
                row.insert(2, mcat[c])
            w.writerow(row)
    os.replace(tmp, rep / f'{out.stem}_clips.csv')

    # summary
    mx = np.array([lift[s : s + n].max() for s, n in zip(starts, lengths)])  # (C,) max lift per clip
    lf = lift > 0
    summ = {
        'dataset': dataset, 'lib': str(args.lib), 'out': str(out), 'out_bytes': os.path.getsize(out),
        'params': {'half_s': HALF, 'floor_tol_m': FLOOR_TOL, 'air_m': AIR, 'contact_m': CONTACT, 'lost_m': LOST,
                   'stand_m': STAND},
        'clips': len(names), 'clips_lifted': int(touched.sum()),
        'clips_lifted_by_max_cm': {f'{a}-{b}': int(((mx > a / 100) & (mx <= b / 100)).sum())
                                   for a, b in ((0, 0.5), (0.5, 1), (1, 3), (3, 6), (6, 10), (10, 100))},
        'dlift_abs_max': float(np.abs(dl).max()),
        'dlift_abs_p99_lifted': float(np.percentile(np.abs(dl[lf]), 99)) if lf.any() else 0.0,
        'all': stats(low, lift, zg, pel_z),
        'by_category': grouped(content, starts, lengths, low, lift, zg, pel_z),
        'seconds': time.time() - t0,
    }  # fmt: skip
    if meta:
        summ['by_meta_category'] = grouped(mcat, starts, lengths, low, lift, zg, pel_z)
    tmp = rep / f'{out.stem}_summary.json.tmp'
    tmp.write_text(json.dumps(summ, indent=1))
    os.replace(tmp, rep / f'{out.stem}_summary.json')
    a = summ['all']
    print(
        f'{dataset}: frames < -1 cm {a["pen1_before"]:.2%} -> {a["pen1_after"]:.2%}, < -3 cm {a["pen3_before"]:.2%}'
        f' -> {a["pen3_after"]:.2%}; lifted {a["lifted_frac"]:.1%} of frames, median lift '
        f'{a["lift_median"] * 100:.2f} cm, max {a["lift_max"] * 100:.1f} cm; airborne > 3 cm {a["air3_frac"]:.2%}; '
        f'{time.time() - t0:.0f}s'
    )
    return 0


# ---- check command


def cmd_check(args):
    t0 = time.time()
    dataset = args.dataset or Path(args.lib).stem
    gpath = Path(args.grounded) if args.grounded else out_path(args.lib)
    a, b = load(args.lib), load(gpath)
    res = {'dataset': dataset, 'grounded': str(gpath)}
    assert list(a) == list(b), (list(a), list(b))
    for k in ('starts', 'lengths', 'names', 'fps', 'storage'):
        assert np.array_equal(a[k], b[k]), k
    rp0, rp1, r0, r1 = a['root_pos'], b['root_pos'], a['rest'], b['rest']
    starts, lengths = a['starts'], a['lengths']
    names = [str(n) for n in a['names']]
    other = np.delete(np.arange(r0.shape[1]), VZ)
    # 1. only root z and v_z change; root z never goes down
    res['xy_identical'] = bool(np.array_equal(rp0[:, :2].view(np.uint32), rp1[:, :2].view(np.uint32)))
    res['rest_other_identical'] = bool(np.array_equal(r0[:, other].view(np.uint16), r1[:, other].view(np.uint16)))
    dz = rp1[:, 2].astype(np.float64) - rp0[:, 2]
    res['never_lowered'] = bool((dz >= 0).all())
    res['finite'] = bool(np.isfinite(rp1).all() and np.isfinite(r1).all())
    # 2. bit-identical clips = clips without lift (from the run's per-clip CSV), and nothing else changed
    changed = np.array([not (np.array_equal(rp0[s : s + n].view(np.uint32), rp1[s : s + n].view(np.uint32))
                             and np.array_equal(r0[s : s + n].view(np.uint16), r1[s : s + n].view(np.uint16)))
                        for s, n in zip(starts, lengths)])  # fmt: skip
    csv_path = Path(args.report_dir or gpath.parent) / f'{gpath.stem}_clips.csv'
    if csv_path.exists():
        lifted = {r['name']: int(r['lifted']) > 0 for r in csv.DictReader(open(csv_path))}
        want = np.array([lifted[n] for n in names])
        res['untouched_clips'] = int((~want).sum())
        res['untouched_bit_identical'] = bool(not changed[~want].any())
        res['lifted_clips_changed'] = int(changed[want].sum())
    res['changed_clips'] = int(changed.sum())
    # 3. velocities: stored v_z vs np.gradient of the stored root z, interior frames of each clip
    inner = np.ones(len(rp0), bool)
    inner[starts] = False
    inner[starts + lengths - 1] = False
    vz_fd = lambda rp: np.concatenate([np.gradient(rp[s : s + n, 2].astype(np.float64), 1 / FPS) if n > 1
                                       else np.zeros(n) for s, n in zip(starts, lengths)])  # fmt: skip
    for tag, rp, r in (('before', rp0, r0), ('after', rp1, r1)):
        err = np.abs(r[:, VZ].astype(np.float64) - vz_fd(rp))[inner]
        res[f'vz_fd_err_{tag}'] = {'median': float(np.median(err)), 'p99': float(np.percentile(err, 99)),
                                   'max': float(err.max())}  # fmt: skip
    dv = r1[:, VZ].astype(np.float64) - r0[:, VZ]
    res['vz_change_abs_max'] = float(np.abs(dv).max())
    step = np.abs(np.diff(dv))
    step[starts[1:] - 1] = 0  # no step across clip boundaries
    res['vz_change_step_abs_max'] = float(step.max())
    # 4. fresh FK on the corrected library
    low, zg = lowest_points(rp1, r1, starts, lengths, args.workers, args.xml)
    res['after'] = {f'pen{p * 100:g}': float((low < -p).mean()) for p in PEN}
    res['after'].update(min=float(low.min()), frames=len(low))
    keys = [content_of(dataset, n) for n in names]
    by = grouped(keys, starts, lengths, low, np.zeros(len(low)), zg, rp1[:, 2].astype(np.float64))
    pen = lambda v: {f'pen{p * 100:g}': v[f'pen{p * 100:g}_before'] for p in PEN}
    # zero lift: the '_before' fields of stats() are the corrected library's values
    res['after_by_category'] = {k: {'frames': v['frames'], 'min': v['min_before'], **pen(v)} for k, v in by.items()}
    # 5. allmimic loader, CPU
    try:
        import torch

        from allmimic.motion_lib import MotionLibrary
    except ImportError as e:
        res['motion_lib'] = f'skipped: {e}'
    else:
        lib = MotionLibrary(str(gpath), 'cpu')
        fields = ('root_pos', 'root_quat', 'q', 'root_lin_vel', 'root_ang_vel', 'qd')
        fin = {f: bool(torch.isfinite(getattr(lib, f).float()).all()) for f in fields}
        fr = lib.frame(torch.randint(0, lib.num_frames, (100_000,)))
        fin['frame()'] = all(bool(torch.isfinite(v).all()) for v in fr.values())
        res['motion_lib'] = {'clips': lib.num_clips, 'frames': lib.num_frames, 'hours': lib.hours, 'finite': fin}
    res['seconds'] = time.time() - t0
    rep = Path(args.report_dir or gpath.parent)
    tmp = rep / f'{gpath.stem}_check.json.tmp'
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, rep / f'{gpath.stem}_check.json')
    print(json.dumps({k: v for k, v in res.items() if k != 'after_by_category'}, indent=1))
    return 0


# ---- CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'check'):
        p = sub.add_parser(name)
        p.add_argument('--lib', required=True, help='compact_v1 library (the original)')
        p.add_argument('--workers', type=int, default=8)
        p.add_argument('--xml', default=None)
        p.add_argument('--dataset', default=None, help='category rules (failure_report.category); default: file stem')
        p.add_argument('--report-dir', default=None, help='per-clip CSV / summary / check JSON (default: next to out)')
    p = sub.choices['run']
    p.add_argument('--out', default=None, help='default: <lib stem>_grounded.npz next to --lib')
    p.add_argument('--meta', default=None, help='BONES seed_metadata_v004.csv: adds metadata categories')
    p.add_argument('--force', action='store_true')
    sub.choices['check'].add_argument('--grounded', default=None, help='default: <lib stem>_grounded.npz')
    args = ap.parse_args(argv)
    return {'run': cmd_run, 'check': cmd_check}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
