"""Cluster the 1 s reset bins of a motion library by kinematics and derive cluster-balanced sampling weights.

  # CPU only (OMP_NUM_THREADS=1 recommended)
  python tools/bin_clusters.py --lib data/library/bones_train.npz --workers 16 \
      [--groups hard_groups.csv] --out data/bin_clusters/bones_train
  # then: python -m allmimic.train ... --bin-prior data/bin_clusters/bones_train.npz --bin-prior-alpha 0.3

Why: in full BONES training the hard motions are a sliver of the data (flips 0.04% of bins, 0.22% of episode starts
under the return-based sampler). Bins, not clips: a flip clip is mostly standing, and only its take-off second is
rare.

Per bin (allmimic's 1 s bins: 50 frames, the last one partial), from the reference itself:
  pelvis height (mean, min), uprightness R22 (mean, min; < 0 = inverted), root angular speed (max), vertical root
  velocity (max, min), joint speed RMS (mean), and contact shares from per-frame FK (tools/filter_infeasible.Body via
  ground_correct.lowest_points): airborne (no group within 3 cm of the floor), hands, knees, seat / torso / head.
Standardized by median / IQR, k-means (k-means++ on a subsample, Lloyd on all bins). Sampling weight of a bin in
cluster c: share_c / n_c with share_c ~ n_c^alpha, scaled to mean 1 over bins (alpha = 1: the library as is, 0: every
cluster equally often, 0.5: square-root share). Writes <out>.npz (names, lengths, per-bin cluster,
features, weights per alpha) and <out>_report.json (expected shares of the hard groups per alpha, rarest clusters).
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_infeasible import FPS  # noqa: E402
from ground_correct import GN, load, lowest_points  # noqa: E402

BIN = FPS  # frames per bin (allmimic MotionLibrary, bin_seconds = 1)
NEAR = 0.03  # m, a body group this close to the floor touches it
FEATS = ('z_mean', 'z_min', 'up_mean', 'up_min', 'w_max', 'vz_max', 'vz_min', 'qd_rms', 'air', 'hand', 'knee', 'body')


def frame_features(rp, rest, low_group):
    """library arrays + (N,G) per-group lowest points -> (N, 9) per-frame quantities"""
    quat = rest[:, :4].astype(np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    x, y = quat[:, 1], quat[:, 2]
    up = 1 - 2 * (x * x + y * y)  # R[2,2]: z of the body z axis
    wv = np.linalg.norm(rest[:, 36:39].astype(np.float32), axis=1)
    vz = rest[:, 35].astype(np.float32)
    qd = np.sqrt((rest[:, 39:68].astype(np.float32) ** 2).mean(1))
    g = {k: low_group[:, i] for i, k in enumerate(GN)}
    near = low_group < NEAR
    air = (~near.any(1)).astype(np.float32)
    body = (np.minimum(np.minimum(g['seat'], g['torso']), g['head']) < NEAR).astype(np.float32)
    return np.stack([rp[:, 2], up, wv, vz, qd, air, (g['hand'] < NEAR).astype(np.float32),
                     (g['knee'] < NEAR).astype(np.float32), body], 1)  # fmt: skip


def bin_features(f, starts, lengths):
    """(N, 9) per-frame quantities -> (B, 12) per-bin features in allmimic's bin order"""
    out = []
    for s, n in zip(starts, lengths):
        for b0 in range(0, n, BIN):
            x = f[s + b0 : s + min(b0 + BIN, n)]
            out.append([x[:, 0].mean(), x[:, 0].min(), x[:, 1].mean(), x[:, 1].min(), x[:, 2].max(), x[:, 3].max(),
                        x[:, 3].min(), x[:, 4].mean(), x[:, 5].mean(), x[:, 6].mean(), x[:, 7].mean(), x[:, 8].mean()])  # fmt: skip
    return np.asarray(out, np.float32)


def kmeans(x, k, iters=30, sample=200_000, seed=0):
    """(B, D) -> (B,) labels, (k, D) centres; k-means++ on a subsample, then Lloyd iterations on all points"""
    rng = np.random.default_rng(seed)
    sub = x[rng.choice(len(x), min(sample, len(x)), replace=False)]
    c = [sub[rng.integers(len(sub))]]
    d2 = ((sub - c[0]) ** 2).sum(1)
    for _ in range(k - 1):
        c.append(sub[rng.choice(len(sub), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((sub - c[-1]) ** 2).sum(1))
    c = np.stack(c)
    for _ in range(iters):
        lab = np.concatenate(
            [(((x[i : i + 65536, None] - c[None]) ** 2).sum(-1)).argmin(1) for i in range(0, len(x), 65536)]
        )
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = x[m].mean(0)
    return lab, c


def weights(lab, k, alpha):
    """(B,) cluster labels -> (B,) bin weights, share_c ~ n_c^alpha, mean 1 over bins"""
    n = np.bincount(lab, minlength=k).astype(np.float64)
    share = np.where(n > 0, n**alpha, 0)
    share /= share.sum()
    w = (share / np.maximum(n, 1))[lab]
    return (w / w.mean()).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--lib', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--groups', help='csv name,group: report the expected share of these groups')
    ap.add_argument('--k', type=int, default=256)
    ap.add_argument('--alphas', type=float, nargs='+', default=[1.0, 0.7, 0.5, 0.3, 0.0])
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args(argv)
    d = load(args.lib)
    rp, rest = d['root_pos'], d['rest']
    starts, lengths = d['starts'].astype(np.int64), d['lengths'].astype(np.int64)
    names = [str(n) for n in d['names']]
    low, low_group = lowest_points(rp, rest, starts, lengths, args.workers)
    f = frame_features(rp, rest, low_group.astype(np.float32))
    feat = bin_features(f, starts, lengths)
    med, iqr = np.median(feat, 0), np.subtract(*np.percentile(feat, [75, 25], 0))
    lab, cen = kmeans((feat - med) / np.where(iqr > 1e-6, iqr, 1.0), args.k)
    nb = (lengths + BIN - 1) // BIN
    clip_of_bin = np.repeat(np.arange(len(names)), nb)
    fam = lambda n: re.sub(r'(_\d{3})?__A\d+(_M)?$', '', n)  # noqa: E731
    w = {a: weights(lab, args.k, a) for a in args.alphas}
    rep = {'bins': int(len(lab)), 'k': args.k, 'cluster_sizes': np.sort(np.bincount(lab, minlength=args.k)).tolist()}
    if args.groups:
        grp = {r['name']: r['group'] for r in csv.DictReader(open(args.groups))}
        gb = np.array([grp.get(names[c], '') for c in clip_of_bin])
        rep['expected_share_pct'] = {
            g: {str(a): round(float(w[a][gb == g].sum() / len(lab) * 100), 4) for a in args.alphas}
            for g in sorted(set(grp.values()))
        }
    sizes = np.bincount(lab, minlength=args.k)
    rare = []
    for j in np.argsort(sizes)[:20]:
        m = np.flatnonzero(lab == j)
        fams = {}
        for b in m:
            fams[fam(names[clip_of_bin[b]])] = fams.get(fam(names[clip_of_bin[b]]), 0) + 1
        rare.append({'cluster': int(j), 'bins': int(sizes[j]), 'centre': dict(zip(FEATS, (cen[j] * np.where(iqr > 1e-6, iqr, 1.0) + med).round(3).tolist())),
                     'families': sorted(fams.items(), key=lambda x: -x[1])[:6]})  # fmt: skip
    rep['rarest_clusters'] = rare
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out.with_suffix('.npz'), names=np.array(names), lengths=lengths, cluster=lab.astype(np.int16),
             features=feat, **{f'w_alpha_{a:g}': w[a] for a in args.alphas})  # fmt: skip
    Path(f'{args.out}_report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != 'cluster_sizes'}, indent=1)[:4000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
