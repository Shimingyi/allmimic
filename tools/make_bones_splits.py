"""Deterministic held-out splits of BONES-SEED for generalization tests.

Only clips kept by the Appendix-E filters are split. A clip and its left-right mirror
(`<name>` / `<name>_M`) always land in the same split, so no mirrored copy of a test clip is
trained on.

  test_content : every clip of ~4% of base contents (content_name without `_M`), stratified by
                 category -> motion types never seen in training.
  test_actor   : every remaining clip of ~3% of actors -> seen motion types, unseen performers.
  test_random  : ~2% of the remaining base moves (orig + mirror) -> IID control.
  train        : everything else.

  python tools/make_bones_splits.py --metadata data/raw/bones_seed/metadata/seed_metadata_v004.csv \
      --kept data/processed/bones_filtered.npz --out-dir data/splits/bones_seed

The kept clips come from the Appendix-E filter: --kept takes a library .npz (its clip names; scripts/filter_bones.py
output) or a text file with one clip name per line; --manifest takes a csv with name,keep columns instead.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

import numpy as np


def strip_mirror(name: str) -> str:
    return name[:-2] if name.endswith('_M') else name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--manifest', type=Path, help='csv with name,keep (filter output)')
    src.add_argument('--kept', type=Path, help='library .npz or text file with the kept clip names')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--content-frac', type=float, default=0.04)
    ap.add_argument('--actor-frac', type=float, default=0.03)
    ap.add_argument('--random-frac', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.manifest:
        kept = {
            r['name'] for r in csv.DictReader(open(args.manifest)) if r['keep'].strip().lower() in ('1', 'true', 'yes')
        }
    elif args.kept.suffix == '.npz':
        kept = {Path(str(n)).stem for n in np.load(args.kept)['names']}  # prepare_motions names: relative CSV paths
    else:
        kept = {n.strip() for n in args.kept.read_text().splitlines() if n.strip()}
    meta = {r['filename']: r for r in csv.DictReader(open(args.metadata)) if r['filename'] in kept}
    missing = kept - meta.keys()
    if missing:
        raise SystemExit(f'{len(missing)} kept clips have no metadata row, e.g. {sorted(missing)[:3]}')
    rng = random.Random(args.seed)

    # test_content: stratified by category over base contents
    contents_by_cat: dict[str, set[str]] = collections.defaultdict(set)
    for r in meta.values():
        contents_by_cat[r['category']].add(strip_mirror(r['content_name']))
    held_contents: set[str] = set()
    for cat in sorted(contents_by_cat):
        pool = sorted(contents_by_cat[cat])
        n = round(args.content_frac * len(pool))
        if n == 0 and len(pool) >= 5:
            n = 1
        held_contents.update(rng.sample(pool, n))

    split: dict[str, str] = {}
    for name, r in meta.items():
        if strip_mirror(r['content_name']) in held_contents:
            split[name] = 'test_content'

    # test_actor
    actors = sorted({r['take_actor'] for n, r in meta.items() if n not in split})
    held_actors = set(rng.sample(actors, round(args.actor_frac * len(actors))))
    for name, r in meta.items():
        if name not in split and r['take_actor'] in held_actors:
            split[name] = 'test_actor'

    # test_random over base moves (keeps orig + mirror together)
    bases = sorted({strip_mirror(n) for n in meta if n not in split})
    held_bases = set(rng.sample(bases, round(args.random_frac * len(bases))))
    for name in meta:
        if name not in split:
            split[name] = 'test_random' if strip_mirror(name) in held_bases else 'train'

    # every mirror pair must share one split
    by_base: dict[str, set[str]] = collections.defaultdict(set)
    for name, s in split.items():
        by_base[strip_mirror(name)].add(s)
    leaks = [b for b, s in by_base.items() if len(s) > 1]
    assert not leaks, f'mirror pairs split across sets: {leaks[:5]}'

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for s in ('train', 'test_content', 'test_actor', 'test_random'):
        names = sorted(n for n, v in split.items() if v == s)
        (args.out_dir / f'{s}.txt').write_text('\n'.join(names) + '\n')
        frames = sum(int(meta[n]['move_duration_frames']) for n in names)
        summary[s] = {'clips': len(names), 'hours@120fps': round(frames / 120 / 3600, 2)}
    summary['held_contents'] = len(held_contents)
    summary['held_actors'] = sorted(held_actors)
    summary['seed'] = args.seed
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'held_actors'}, indent=1))


if __name__ == '__main__':
    main()
