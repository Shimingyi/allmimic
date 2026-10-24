"""Flag reference motions a G1 cannot perform on a flat, empty floor (sitting on an invisible chair, leaning on a wall).

  # 1. per-frame support features of a compact library (CPU only, OMP_NUM_THREADS=1 recommended; the 240 h BONES
  #    train split takes ~3 min with 16 workers)
  python tools/filter_infeasible.py features --lib data/library/bones_train.npz --out data/feats/bones_train.npz \
      --workers 16
  # 2. flags -> data/infeasible/<dataset>_flagged.csv, _feasible.txt, _summary.json (+ <dataset>_seg10_* from a
  #    tools/make_segments.py manifest; --meta adds BONES content / metadata support labels)
  python tools/filter_infeasible.py flag --feats data/feats/bones_train.npz --dataset bones_train \
      --meta data/raw/bones_seed/seed_metadata_v004.csv
  python tools/filter_infeasible.py flag --feats data/feats/lafan1.npz --dataset lafan1 \
      --segments data/library_eval/lafan1_seg10_manifest.csv
  # 3. success / clipped return of per-clip eval JSONs, all vs feasible subset
  python tools/filter_infeasible.py rescore runs/bones/eval/ckpt_0120000

Method: forward kinematics of the reference with the G1 MJCF (mjlab's file,
33.34 kg) and mj_kinematics; COM from body masses. Each collision capsule end / sphere of the sim model (all of them
collide with the floor under mjlab's FULL_COLLISION) is a candidate contact, in contact when its lowest point is
within EPS of the floor. The floor is z = 0, raised per clip to the 25th percentile of the per-frame lowest point
when that is above 0 (some AMASS-G1 clips float), capped at 0.2 m. Hull distances use the support function over NDIR
directions (error < 0.25 %). Per quasi-static frame (COM speed < 0.2 m/s, |a| < 1.5 m/s^2):
  com_outside  COM and ZMP (COM - height * a_xy / (g + a_z)) more than 8 cm outside the hull of every point within
               15 cm of the floor (lenient: a hovering hand, knee or seat still counts as support)
  sit          pelvis 0.25-0.70 m, seat >= 12 cm and knees / hands / elbows >= 6 cm above the floor, COM outside the
               foot-contact hull and >= 5 cm behind its centre, pelvis >= 10 cm outside it and >= 15 cm behind;
               or the same posture with nothing within 5 cm and the COM > 4 cm outside the 15 cm hull (hover)
  float        nothing within 15 cm of the floor
A clip is flagged when such frames (gaps <= 0.1 s closed) last >= 1 s; a 10 s segment when it overlaps such a run
of its source clip for >= 0.5 s. Thresholds: RULE (sensitivity grid in <dataset>_summary.json).
Library: compact_v1 (root_pos f32, rest f16 = [quat wxyz 4, q 29, ...]), 50 Hz, Z-up, SI units.
Features file: feats (F,len(FEATS)) float16 per frame + starts / lengths / names / ground (m) per clip.
2026-10-24: first version (calibrated on hand-picked MotionDecode sitting examples and BONES metadata).
"""

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FPS = 50  # Hz
G = 9.81  # m/s^2
EPS = (0.03, 0.05, 0.08, 0.15)  # m, contact height tolerances (0.15: anything near the floor may bear weight)
NDIR = 48  # support-function directions for the hull distance
NONE = 9.0  # m, signed distance stored when there is no contact
GROUND_Q = 25  # percentile of the per-frame lowest point used as the clip's floor (only if above 0)
GROUND_MAX = 0.20  # m
SMOOTH = 5  # frames (0.1 s) moving average before differentiating the COM
GROUPS = {
    'foot': r'(left|right)_(foot\d|linkage_brace)_collision',  # brace: ankle-height capsule, part of the stance
    'knee': r'(left|right)_shin_collision',
    'hand': r'(left|right)_(wrist|hand)_collision',
    'arm': r'(left|right)_(shoulder_yaw|elbow_yaw)_collision',
    'seat': r'pelvis_collision|(left|right)_(hip|thigh)_collision',
    'torso': r'torso_collision',
    'head': r'head_collision',
}
# Flag rule (calibrated 2026-10-24 on BONES metadata labels, hand-picked MotionDecode sitting examples and the
# must-keep classes). All thresholds are physical quantities.
RULE = dict(
    v_static=0.20,  # m/s, COM horizontal speed below which a frame is quasi-static
    a_static=1.5,  # m/s^2, |a_z| and |a_xy| bound for quasi-static (excludes flight / impacts)
    com_margin=0.08,  # m, COM and ZMP outside the 15 cm contact hull by more than this -> unsupported
    #   (one-foot balance and light wall leaning stay < 0.08: retargeted COM sits 2-10 cm off a single foot)
    sit_z=(0.25, 0.70),  # m, pelvis height band of the sitting signature
    sit_clear=0.06,  # m, knee / hand / elbow must all be this far above the floor (else real support)
    sit_clear_seat=0.12,  # m, and the seat (pelvis / hips / thighs): positives sit 0.17-0.30 m up, get-ups and
    #   very low squats have it within 0.06-0.10 m of the floor
    sit_pel_out=0.10,  # m, pelvis outside the foot-contact hull by more than this
    sit_pel_back=0.15,  # m, pelvis behind the foot-contact centre (pelvis heading) by more than this
    sit_com_back=0.05,  # m, COM behind the foot-contact centre by more than this, and outside the foot hull
    sit_feet15=-0.02,  # m, COM not inside the hull of the foot points within 15 cm by more than this (side lunge
    #   with a hovering second foot)
    sit_hover=0.04,  # m, seated with no contact within 5 cm (AMASS feet hovering): COM this far outside the
    #   hull of every point within 15 cm
    gap=0.10,  # s, gaps up to this long inside a run are closed
    t_min=1.0,  # s, a run must last this long to flag a clip (0.5 s flags one-foot balance in BONES / LAFAN1)
    seg_overlap=0.5,  # s, a segment is flagged when it overlaps a qualifying run of its source clip this long
)
# relaxed rule for the 'borderline' validation sample: clips it flags but RULE does not
RELAXED = dict(
    t_min=0.5, com_margin=0.05, sit_pel_out=0.05, sit_pel_back=0.10, sit_com_back=0.0, sit_feet15=-0.05, sit_hover=0.0
)
REASONS = ('sit', 'com_outside', 'float')
FEATS = (
    ['z_' + g for g in GROUPS]  # m, lowest point of the group above the floor
    + [f'sd_com_{int(e * 100)}' for e in EPS]  # m, COM xy to the contact hull, + outside, NONE if no contact
    + ['sd_zmp_8', 'sd_zmp_15']  # m, ZMP to the hull (eps 8 / 15 cm)
    + ['sd_feet', 'pel_feet']  # m, COM / pelvis to the hull of the foot contacts (eps 5 cm)
    + ['sd_feet15']  # m, COM to the hull of the foot points within 15 cm (a hovering second foot still counts)
    + ['com_back', 'pel_back']  # m, foot-contact centre minus COM / pelvis along the pelvis heading (+ = behind feet)
    + ['com_z', 'pel_z', 'v_xy', 'v_z', 'a_xy', 'a_z']  # m, m/s, m/s^2 (COM, smoothed)
)


# ---- model


def g1_xml():
    """path of the G1 MJCF: mjlab's when installed, else a copy under data/render (pass --xml otherwise)"""
    try:
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

        return str(G1_XML)
    except ImportError:
        return str(ROOT / 'data' / 'render' / 'unitree_g1' / 'xmls' / 'g1.xml')


class Body:
    """G1 kinematics: collision primitives as contact candidates + body masses for the COM"""

    def __init__(self, xml=None):
        import mujoco

        self.mujoco = mujoco
        m = self.m = mujoco.MjModel.from_xml_path(xml or g1_xml())
        self.d = mujoco.MjData(m)
        cg = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in cg]
        cap = m.geom_type[cg] == mujoco.mjtGeom.mjGEOM_CAPSULE
        assert all(cap | (m.geom_type[cg] == mujoco.mjtGeom.mjGEOM_SPHERE)), 'only capsules and spheres expected'
        self.cg = np.array(cg)
        self.half = np.where(cap, m.geom_size[cg, 1], 0.0)  # (C,) capsule half length, 0 for spheres
        group = np.array([next(i for i, p in enumerate(GROUPS.values()) if re.fullmatch(p, n)) for n in names])
        self.rad = np.tile(m.geom_size[cg, 0], 2)  # (2C,) one sphere per capsule end
        self.group = np.tile(group, 2)  # (2C,)
        self.mass = m.body_mass.copy()
        self.mass_total = self.mass.sum()
        a = np.arange(NDIR) * 2 * np.pi / NDIR
        self.dirs = np.stack([np.cos(a), np.sin(a)], 1).astype(np.float32)  # (K,2)

    def fk(self, root_pos, quat, q):
        """(T,3), (T,4) wxyz, (T,29) -> (T,2C,3) capsule-end centres, (T,3) COM, (T,3) pelvis x axis"""
        mj, m, d = self.mujoco, self.m, self.d
        T = len(root_pos)
        Q = np.concatenate([root_pos, quat, q], 1).astype(np.float64)
        xp = np.empty((T, m.ngeom, 3))
        xm = np.empty((T, m.ngeom, 9))
        xi = np.empty((T, m.nbody, 3))
        qpos, gx, gm, bx = d.qpos, d.geom_xpos, d.geom_xmat, d.xipos  # cached views: 3 us/frame instead of 60
        for t in range(T):
            qpos[:] = Q[t]
            mj.mj_kinematics(m, d)
            xp[t] = gx
            xm[t] = gm
            xi[t] = bx
        c, axis = xp[:, self.cg], xm[:, self.cg][:, :, 2::3]  # capsule axis = local z column
        h = self.half[None, :, None] * axis
        ends = np.concatenate([c + h, c - h], 1)
        com = (self.mass[None, :, None] * xi).sum(1) / self.mass_total
        w, x, y, z = quat.T.astype(np.float64)
        fwd = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], 1)
        return ends, com, fwd

    def support(self, proj, mask):
        """(T,P,K) point projections on the directions, (T,P) bool -> (T,K) support function of the masked points"""
        return np.where(mask[..., None], proj, -np.inf).max(1)

    def sd(self, h, p):
        """(T,K) support function, (T,2) query -> (T,) signed distance to the hull (+ outside, NONE if empty)"""
        return np.minimum((p.astype(np.float32) @ self.dirs.T - h).max(1), NONE)


def smooth(x, n=SMOOTH):
    """(T,...) -> (T,...) centred moving average, edges padded with the end values"""
    if len(x) < 2 or n <= 1:
        return x
    k = np.ones(n) / n
    pad = np.concatenate([np.repeat(x[:1], n // 2, 0), x, np.repeat(x[-1:], n - 1 - n // 2, 0)], 0)
    return np.apply_along_axis(lambda s: np.convolve(s, k, mode='valid'), 0, pad)


def clip_features(body, root_pos, quat, q):
    """one clip (T,3), (T,4) wxyz, (T,29) -> (T,len(FEATS)) float32, floor height (m)"""
    ends, com, fwd = body.fk(root_pos, quat, q)
    zl = ends[..., 2] - body.rad  # (T,P) lowest point of each capsule-end sphere
    ground = float(np.clip(np.percentile(zl.min(1), GROUND_Q), 0.0, GROUND_MAX))
    zl = zl - ground
    xy = ends[..., :2]
    proj = xy.astype(np.float32) @ body.dirs.T  # (T,P,K), shared by every hull below
    T = len(com)
    out = {}
    for k, g in enumerate(GROUPS):
        out['z_' + g] = zl[:, body.group == k].min(1)
    hull = {e: body.support(proj, zl < e) for e in EPS}
    for e in EPS:
        out[f'sd_com_{int(e * 100)}'] = body.sd(hull[e], com[:, :2])
    # COM kinematics: smoothed positions, central differences (np.gradient) at 50 Hz
    cs = smooth(com)
    v = np.gradient(cs, 1 / FPS, axis=0) if T > 1 else np.zeros_like(cs)
    a = smooth(np.gradient(v, 1 / FPS, axis=0)) if T > 1 else np.zeros_like(cs)
    hz = com[:, 2] - ground
    zmp = com[:, :2] - hz[:, None] * a[:, :2] / np.maximum(G + a[:, 2], 1.0)[:, None]
    out['sd_zmp_8'], out['sd_zmp_15'] = body.sd(hull[0.08], zmp), body.sd(hull[0.15], zmp)
    feet = (zl < 0.05) & (body.group == 0)[None]
    hf = body.support(proj, feet)
    out['sd_feet'], out['pel_feet'] = body.sd(hf, com[:, :2]), body.sd(hf, root_pos[:, :2])
    out['sd_feet15'] = body.sd(body.support(proj, (zl < 0.15) & (body.group == 0)[None]), com[:, :2])
    n = feet.sum(1)
    centre = (xy * feet[..., None]).sum(1) / np.maximum(n, 1)[:, None]
    f2 = fwd[:, :2] / np.maximum(np.linalg.norm(fwd[:, :2], axis=1), 1e-6)[:, None]
    out['com_back'] = np.where(n > 0, ((centre - com[:, :2]) * f2).sum(1), 0.0)
    out['pel_back'] = np.where(n > 0, ((centre - root_pos[:, :2]) * f2).sum(1), 0.0)
    out['com_z'], out['pel_z'] = hz, root_pos[:, 2] - ground
    out['v_xy'], out['v_z'] = np.linalg.norm(v[:, :2], axis=1), v[:, 2]
    out['a_xy'], out['a_z'] = np.linalg.norm(a[:, :2], axis=1), a[:, 2]
    return np.stack([out[f] for f in FEATS], 1).astype(np.float32), ground


# ---- features command


def load_library(path):
    """compact_v1 npz -> root_pos (N,3) f32, quat (N,4), q (N,29), starts, lengths, names"""
    d = np.load(path, allow_pickle=False)
    assert str(d['storage']) == 'compact_v1' and float(d['fps']) == FPS, path
    rest = d['rest']
    return d['root_pos'], rest[:, :4], rest[:, 4:33], d['starts'], d['lengths'], [str(n) for n in d['names']]


_W = {}


def _init(cache, xml):
    # memory-mapped arrays: 16 workers share the page cache instead of 16 copies of a 6.5 GB library
    _W['rp'] = np.load(cache / 'root_pos.npy', mmap_mode='r')
    _W['qq'] = np.load(cache / 'quat_q.npy', mmap_mode='r')
    _W['body'] = Body(xml)


def _work(job):
    """[(clip id, start, end)] -> list of (id, (T,F) float16, ground)"""
    res = []
    for i, s, e in job:
        qq = np.asarray(_W['qq'][s:e], np.float64)
        quat = qq[:, :4] / np.linalg.norm(qq[:, :4], axis=1, keepdims=True)  # f16 storage: renormalize
        f, g = clip_features(_W['body'], np.asarray(_W['rp'][s:e], np.float64), quat, qq[:, 4:33])
        res.append((i, f.astype(np.float16), g))
    return res


def cmd_features(args):
    if Path(args.out).exists() and not args.force:
        print('exists, skip:', args.out)
        return 0
    rp, qt, q, starts, lengths, names = load_library(args.lib)
    cache = Path(str(args.out) + '.cache')
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / 'root_pos.npy', rp)
    np.save(cache / 'quat_q.npy', np.concatenate([qt, q], 1))
    del rp, qt, q
    n = len(names)
    order = np.argsort(-lengths)  # long clips first: better load balance
    chunks, cur, acc = [], [], 0
    for i in order:
        cur.append((int(i), int(starts[i]), int(starts[i] + lengths[i])))
        acc += int(lengths[i])
        if acc >= args.chunk_frames:
            chunks.append(cur)
            cur, acc = [], 0
    chunks += [cur] if cur else []
    feats = np.zeros((int(lengths.sum()), len(FEATS)), np.float16)
    ground = np.zeros(n, np.float32)
    off = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    done = 0
    with mp.Pool(args.workers, initializer=_init, initargs=(cache, args.xml)) as pool:
        for res in pool.imap_unordered(_work, chunks):
            for i, f, g in res:
                feats[off[i] : off[i] + lengths[i]] = f
                ground[i] = g
            done += len(res)
            print(f'\r{done}/{n} clips', end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(args.out) + '.tmp.npz'
    np.savez(tmp, feats=feats, feat_names=np.array(FEATS), starts=off, lengths=lengths, names=np.array(names),
             ground=ground, eps=np.array(EPS), lib=str(args.lib))  # fmt: skip
    os.replace(tmp, args.out)
    for f in cache.iterdir():
        f.unlink()
    cache.rmdir()
    print(f'{n} clips, {len(feats)} frames -> {args.out}')
    return 0


# ---- flag command


def frame_masks(F, col, r=RULE):
    """(T,F) features, {name: column} -> {reason: (T,) bool}, frames that violate quasi-static support"""
    f = lambda k: F[:, col[k]]
    st = (f('v_xy') < r['v_static']) & (np.abs(f('a_z')) < r['a_static']) & (f('a_xy') < r['a_static'])
    out = st & (f('sd_com_15') > r['com_margin']) & (f('sd_zmp_15') > r['com_margin']) & (f('sd_com_15') < NONE)
    flt = st & (f('sd_com_15') >= NONE)
    clear = (np.minimum.reduce([f('z_knee'), f('z_hand'), f('z_arm')]) > r['sit_clear']) & (
        f('z_seat') > r['sit_clear_seat']
    )
    sit = (st & (f('pel_z') > r['sit_z'][0]) & (f('pel_z') < r['sit_z'][1]) & clear
           & (f('sd_feet') > 0) & (f('sd_feet') < NONE) & (f('pel_feet') > r['sit_pel_out'])
           & (f('pel_back') > r['sit_pel_back']) & (f('com_back') > r['sit_com_back'])
           & (f('sd_feet15') > r['sit_feet15']))  # fmt: skip
    hover = (st & (f('pel_z') > r['sit_z'][0]) & (f('pel_z') < r['sit_z'][1]) & clear
             & (f('sd_com_5') >= NONE) & (f('sd_com_15') > r['sit_hover']) & (f('sd_com_15') < NONE))  # fmt: skip
    return {'sit': sit | hover, 'com_outside': out, 'float': flt}


def runs(m):
    """(T,) bool -> (R,2) [start, end) frame ranges of the True runs"""
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], 1)


def close_gaps(m, gap):
    """(T,) bool -> copy with False gaps of <= gap frames between True runs filled"""
    m = m.copy()
    r = runs(m)
    for (_, e0), (s1, _) in itertools.pairwise(r):
        if s1 - e0 <= gap:
            m[e0:s1] = True
    return m


def clip_runs(F, col, r=RULE):
    """one clip -> (Q,2) qualifying bad runs (>= t_min), {reason: (T,) closed mask}"""
    M = {k: close_gaps(v, round(r['gap'] * FPS)) for k, v in frame_masks(F, col, r).items()}
    bad = close_gaps(M['sit'] | M['com_outside'] | M['float'], round(r['gap'] * FPS))
    rr = runs(bad)
    return rr[rr[:, 1] - rr[:, 0] >= r['t_min'] * FPS], M


def describe(F, col, rr, M, lo=0, hi=None):
    """qualifying runs clipped to [lo, hi) -> reason + metrics of the longest run, or None"""
    hi = len(F) if hi is None else hi
    rr = np.clip(rr, lo, hi)
    rr = rr[rr[:, 1] > rr[:, 0]]
    if not len(rr):
        return None
    bad = sum(e - s for s, e in rr)
    s, e = rr[np.argmax(rr[:, 1] - rr[:, 0])]
    frames = {k: int(sum(M[k][a:b].sum() for a, b in rr)) for k in REASONS}
    med = lambda k: float(np.median(F[s:e, col[k]]))
    return {
        'reason': max(REASONS, key=lambda k: frames[k]),
        'run_s': (e - s) / FPS, 'bad_s': bad / FPS, 'start_s': (s - lo) / FPS,
        **{f'{k}_s': frames[k] / FPS for k in REASONS},
        'pel_z': med('pel_z'), 'com_out_15': med('sd_com_15'), 'com_out_feet': med('sd_feet'),
        'pel_back': med('pel_back'), 'clear_seat': med('z_seat'), 'clear_hand': med('z_hand'),
    }  # fmt: skip


def bones_meta(path):
    """seed_metadata_v004.csv -> {clip name: (content, category, body position, props, metadata support label)}

    Labels (regexes over body position / props / the seven descriptions): seat_prop (chair, sunbed, box seat),
    wall_lean, platform (hands resting on a raised surface), floor_ok (kneel / all fours / crawl / crouch / squat /
    lying / sitting on the floor / plank / handstand), other."""
    out = {}
    desc_cols = [f'content_natural_desc_{i}' for i in range(1, 5)]
    desc_cols += ['content_technical_description', 'content_short_description', 'content_short_description_2']
    seat_d = (r'\bsit\w* (?:down )?on (?:a|the) (?:chair|box|sunbed|bench|stool|sofa|crate|platform)'
              r'|sitting on a chair|seated person|in a sitting position')  # fmt: skip
    wall_d = (
        r'lean\w* (?:on|against|back on) (?:a|the) wall|support of a hand|supported by their hand|supporting themself'
    )
    floor_p = r'kneel|all fours|crawl|crouch|croach|squat|lying|sitting on floor|plank|handstand|sitting on heels|lunge'
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            pos, props = row['content_body_position'].lower(), row['content_props'].lower()
            desc = ' | '.join(row[c] for c in desc_cols).lower()
            if (
                re.search(r'\b(?:chair|sunbed|stool|bench|sofa|bed)\b', props)
                or re.search('sitting on chair|sitting on box|on a box', pos)
                or re.search(seat_d, desc)
            ):
                lab = 'seat_prop'
            elif 'leaning on wall' in pos or (
                re.search(r'\bwall\b', props) and ('lean' in pos or re.search(wall_d, desc))
            ):
                lab = 'wall_lean'
            elif re.search(r'resting their hands back on a platform|hands on the (?:table|platform|counter)', desc):
                lab = 'platform'
            elif re.search(floor_p, pos) or re.search(
                r'crawl|kneel|hands and knees|rolling|lying', row['content_type_of_movement']
            ):
                lab = 'floor_ok'
            else:
                lab = 'other'
            out[row['filename']] = (
                row['content_name'],
                row['category'],
                row['content_body_position'],
                row['content_props'],
                lab,
            )
    return out


def category(dataset, name, meta=None):
    """dataset, clip name -> category label (failure_report.category; BONES: metadata category when available)"""
    if meta is not None and name.split('__s')[0] in meta:
        return meta[name.split('__s')[0]][1]
    if 'cat' not in _W:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from failure_report import category as cat

        _W['cat'] = cat
    return _W['cat'](dataset, name)


def read_manifest(path):
    """make_segments manifest.csv -> [(segment name, source clip, start frame, frames)]"""
    with open(path, newline='') as fh:
        return [(r['name'], r['source'], int(r['start_frame']), int(r['frames'])) for r in csv.DictReader(fh)]


def write_outputs(out_dir, dataset, rows, names_all, meta, extra_cols=()):
    """flagged rows + all names -> <dataset>_flagged.csv, <dataset>_feasible.txt (atomic)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    flagged = {r['name'] for r in rows}
    cols = [
        'name',
        'reason',
        'run_s',
        'bad_s',
        'start_s',
        'sit_s',
        'com_outside_s',
        'float_s',
        'dur_s',
        'pel_z',
        'com_out_15',
        'com_out_feet',
        'pel_back',
        'clear_seat',
        'clear_hand',
        'ground',
        *extra_cols,
        'category',
    ]
    if meta is not None:
        cols += ['content', 'body_position', 'props', 'meta_label']
    tmp = out_dir / f'{dataset}_flagged.csv.tmp'
    with open(tmp, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([f'{r[c]:.3f}' if isinstance(r.get(c), float) else r.get(c, '') for c in cols])
    os.replace(tmp, out_dir / f'{dataset}_flagged.csv')
    tmp = out_dir / f'{dataset}_feasible.txt.tmp'
    tmp.write_text(''.join(n + '\n' for n in names_all if n not in flagged))
    os.replace(tmp, out_dir / f'{dataset}_feasible.txt')


def summarize(dataset, all_rows, rule):
    """all windows [{name, dur_s, flagged, reason, bad_s, category, meta_label?}] -> summary dict"""
    n, h = len(all_rows), sum(r['dur_s'] for r in all_rows) / 3600
    fl = [r for r in all_rows if r['flagged']]
    hours = lambda rs: sum(r['dur_s'] for r in rs) / 3600
    out = {'dataset': dataset, 'clips': n, 'hours': h, 'flagged': len(fl), 'flagged_hours': hours(fl),
           'bad_hours': sum(r['bad_s'] for r in fl) / 3600, 'rule': rule, 'by_reason': {}}  # fmt: skip
    for k in REASONS:
        rs = [r for r in fl if r['reason'] == k]
        out['by_reason'][k] = {'clips': len(rs), 'hours': hours(rs)}
    cats = {}
    for r in all_rows:
        c = cats.setdefault(
            r['category'], {'clips': 0, 'hours': 0.0, 'flagged': 0, 'flagged_hours': 0.0, 'bad_hours': 0.0}
        )
        c['clips'] += 1
        c['hours'] += r['dur_s'] / 3600
        if r['flagged']:
            c['flagged'] += 1
            c['flagged_hours'] += r['dur_s'] / 3600
            c['bad_hours'] += r['bad_s'] / 3600
    out['by_category'] = dict(
        sorted(((k, v) for k, v in cats.items() if v['flagged']), key=lambda kv: -kv[1]['flagged_hours'])
    )
    if 'meta_label' in all_rows[0]:
        conf = {}
        for r in all_rows:
            c = conf.setdefault(r['meta_label'], {'clips': 0, 'flagged': 0})
            c['clips'] += 1
            c['flagged'] += int(r['flagged'])
        out['meta_confusion'] = conf
    return out


def cmd_flag(args):
    rule = dict(RULE)
    rule.update(json.loads(args.rule) if args.rule else {})
    d = np.load(args.feats)
    F16, starts, lengths = d['feats'], d['starts'], d['lengths']
    names, ground = [str(n) for n in d['names']], d['ground']
    col = {str(k): i for i, k in enumerate(d['feat_names'])}
    meta = bones_meta(args.meta) if args.meta else None
    out_dir = Path(args.out_dir)
    # full clips
    per_clip, rows, all_rows = {}, [], []
    for i, n in enumerate(names):
        F = F16[starts[i] : starts[i] + lengths[i]].astype(np.float32)
        rr, M = clip_runs(F, col, rule)
        per_clip[n] = (F, rr, M)
        res = describe(F, col, rr, M)
        rec = {
            'name': n,
            'dur_s': (lengths[i] - 1) / FPS,
            'ground': float(ground[i]),
            'category': category(args.dataset, n, meta),
        }
        if meta is not None:
            m = meta.get(n, ('', '', '', '', 'unknown'))
            rec.update(content=m[0], body_position=m[2], props=m[3], meta_label=m[4])
        if res:
            rows.append({**rec, **res})
        all_rows.append(
            {
                **rec,
                'flagged': res is not None,
                'reason': res['reason'] if res else '',
                'bad_s': res['bad_s'] if res else 0.0,
            }
        )
    write_outputs(out_dir, args.dataset, rows, names, meta)
    summ = {'full': summarize(args.dataset, all_rows, rule)}
    # sensitivity of the clip count to the two main thresholds
    grid = {}
    for t_min in (0.5, 1.0, 2.0):
        for m in (0.06, 0.08, 0.10, 0.12):
            r2 = {**rule, 't_min': t_min, 'com_margin': m}
            k = 0
            h = 0.0
            for i, n in enumerate(names):
                if len(clip_runs(per_clip[n][0], col, r2)[0]):
                    k += 1
                    h += (lengths[i] - 1) / FPS / 3600
            grid[f't_min={t_min},com_margin={m}'] = {'clips': k, 'hours': h}
    summ['full']['sensitivity'] = grid
    # validation samples: random flagged clips, and random 'borderline' clips (flagged only by the relaxed rule)
    rng = np.random.default_rng(0)
    r3 = {**rule, **RELAXED}
    border = []
    for rec in all_rows:
        if not rec['flagged']:
            F, _, _ = per_clip[rec['name']]
            rr, M = clip_runs(F, col, r3)
            res = describe(F, col, rr, M)
            if res:
                border.append(
                    {k: v for k, v in {**rec, **res}.items() if k not in ('flagged', 'body_position', 'props')}
                )
    pick = lambda xs: [xs[i] for i in sorted(rng.choice(len(xs), min(15, len(xs)), replace=False))] if xs else []
    summ['full']['flagged_sample'] = pick(
        [{k: v for k, v in x.items() if k not in ('body_position', 'props')} for x in rows]
    )
    summ['full']['borderline_count'] = len(border)
    summ['full']['borderline_sample'] = pick(border)
    print(
        f'{args.dataset}: {len(rows)} / {len(names)} clips flagged, {summ["full"]["flagged_hours"]:.2f} / '
        f'{summ["full"]["hours"]:.2f} h; by reason {[(k, v["clips"]) for k, v in summ["full"]["by_reason"].items()]}'
    )
    # 10 s segments: flagged when they overlap a qualifying run of the source clip for >= seg_overlap
    if args.segments:
        seg_rows, seg_all, seg_names = [], [], []
        index = {n: i for i, n in enumerate(names)}
        for sname, src, s0, nf in read_manifest(args.segments):
            F, rr, M = per_clip[src]
            res = describe(F, col, rr, M, s0, s0 + nf)
            if res and res['bad_s'] < rule['seg_overlap']:
                res = None
            rec = {'name': sname, 'dur_s': (nf - 1) / FPS, 'ground': float(ground[index[src]]), 'source': src,
                   'seg_start_s': s0 / FPS, 'category': category(args.dataset, src, meta)}  # fmt: skip
            if meta is not None:
                m = meta.get(src, ('', '', '', '', 'unknown'))
                rec.update(content=m[0], body_position=m[2], props=m[3], meta_label=m[4])
            if res:
                seg_rows.append({**rec, **res})
            seg_all.append(
                {
                    **rec,
                    'flagged': res is not None,
                    'reason': res['reason'] if res else '',
                    'bad_s': res['bad_s'] if res else 0.0,
                }
            )
            seg_names.append(sname)
        ds = f'{args.dataset}_seg10'
        write_outputs(out_dir, ds, seg_rows, seg_names, meta, extra_cols=('source', 'seg_start_s'))
        summ['seg10'] = summarize(ds, seg_all, rule)
        print(f"{ds}: {len(seg_rows)} / {len(seg_names)} segments flagged, {summ['seg10']['flagged_hours']:.2f} / "
              f"{summ['seg10']['hours']:.2f} h")  # fmt: skip
    tmp = out_dir / f'{args.dataset}_summary.json.tmp'
    tmp.write_text(json.dumps(summ, indent=1))
    os.replace(tmp, out_dir / f'{args.dataset}_summary.json')
    return 0


# ---- rescore command


def score(rows):
    """per-clip eval rows -> {clips, failed, success_rate, mean_ret_clip01}"""
    n = len(rows)
    return {
        'clips': n,
        'failed': sum(r['failed'] for r in rows),
        'success_rate': sum(not r['failed'] for r in rows) / max(n, 1),
        'mean_ret_clip01': sum(min(max(r['ret_paper'], 0.0), 1.0) for r in rows) / max(n, 1),
    }


def cmd_rescore(args):
    out = {}
    print(f'{"test set":28s} {"clips":>6s} {"succ":>7s} {"ret01":>7s} | {"feasible":>8s} {"succ":>7s} {"ret01":>7s} | '
          f'{"flagged":>7s} {"fail":>5s}')  # fmt: skip
    for p in sorted(Path(args.eval_dir).glob('*.json')):
        if p.stem == 'summary':
            continue
        keep_path = Path(args.filter_dir) / f'{p.stem}_feasible.txt'
        if not keep_path.exists():
            continue
        keep = set(keep_path.read_text().split())
        rows = json.loads(p.read_text())['clips']
        a, f = score(rows), score([r for r in rows if r['name'] in keep])
        bad = [r for r in rows if r['name'] not in keep]
        out[p.stem] = {'all': a, 'feasible': f, 'flagged': score(bad)}
        print(
            f'{p.stem:28s} {a["clips"]:6d} {a["success_rate"]:7.4f} {a["mean_ret_clip01"]:7.4f} | {f["clips"]:8d} '
            f'{f["success_rate"]:7.4f} {f["mean_ret_clip01"]:7.4f} | {len(bad):7d} {sum(r["failed"] for r in bad):5d}'
        )
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


# ---- CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('features', help='per-frame support features of a compact library')
    p.add_argument('--lib', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--chunk-frames', type=int, default=200_000)
    p.add_argument('--xml', default=None)
    p.add_argument('--force', action='store_true')
    p = sub.add_parser('flag', help='clip / segment flags from a features file')
    p.add_argument('--feats', required=True)
    p.add_argument('--dataset', required=True, help='output stem, e.g. motiondecode_s2000')
    p.add_argument('--segments', default=None, help='make_segments manifest.csv -> also <dataset>_seg10 outputs')
    p.add_argument('--meta', default=None, help='BONES seed_metadata_v004.csv: adds content / metadata labels')
    p.add_argument('--out-dir', default=str(ROOT / 'data' / 'infeasible'))
    p.add_argument('--rule', default='', help='JSON overrides of RULE, e.g. {"t_min": 0.5}')
    p = sub.add_parser('rescore', help='success / clipped return of per-clip eval JSONs on the feasible subset')
    p.add_argument('eval_dir')
    p.add_argument('--filter-dir', default=str(ROOT / 'data' / 'infeasible'))
    p.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    return {'features': cmd_features, 'flag': cmd_flag, 'rescore': cmd_rescore}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
