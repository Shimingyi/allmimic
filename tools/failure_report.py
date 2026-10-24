"""Failure attribution for one evaluated checkpoint : which motion categories fail, per test set.

  python tools/failure_report.py runs/bones/eval/ckpt_0120000 \
      --testsets lafan1_seg10 bones_test_content_seg10 [--top 12]

Reads the per-clip <testset>.json written by tools/eval_allmimic.py. Categories
come from clip names (no metadata needed): LAFAN1 action word (fallAndGetUp, dance, ...), MotionDecode top-level
category, AMASS source dataset, BONES content name without take / actor suffixes. Returns are clipped to [0, 1].
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def category(testset, name):
    """test set name, clip name -> category label"""
    base = name.split('__s')[0]  # drop the 10 s segment suffix
    if testset.startswith('lafan1'):
        return re.match(r'[A-Za-z]+', base).group(0)
    if testset.startswith('motiondecode'):
        toks = base.split('_')
        words = []
        for t in toks[2:]:  # "1_10_Emotional_Expressions_..._1_10_10_..." -> Emotional_Expressions_...
            if t.isdigit():
                break
            words.append(t)
        return '_'.join(words)
    if testset.startswith('amass'):
        return base.split('_')[0]
    if testset.startswith('bones'):
        content = base.split('__')[0]  # Loop_Forward_Jog_001__A017_M -> Loop_Forward_Jog_001
        return re.sub(r'(_\d+)+$', '', content)
    return base


def clip01(x):
    return min(max(x, 0.0), 1.0)


def report(path, testset, top):
    rows = json.loads(path.read_text())['clips']
    groups = defaultdict(list)
    for r in rows:
        groups[category(testset, r['name'])].append(r)
    n, nf = len(rows), sum(r['failed'] for r in rows)
    print(f'\n## {testset}: {n} clips, {nf} failed ({nf / max(n, 1):.1%}), clip01 return '
          f'{sum(clip01(r["ret_paper"]) for r in rows) / max(n, 1):.4f}, {len(groups)} categories')  # fmt: skip
    print(f'{"category":44s} {"clips":>6s} {"failed":>6s} {"rate":>6s} {"share":>6s} {"ret01":>6s}')
    ranked = sorted(groups.items(), key=lambda kv: -sum(r['failed'] for r in kv[1]))
    for cat, rs in ranked[:top]:
        f = sum(r['failed'] for r in rs)
        if f == 0:
            break
        ret = sum(clip01(r['ret_paper']) for r in rs) / len(rs)
        print(f'{cat[:44]:44s} {len(rs):6d} {f:6d} {f / len(rs):6.1%} {f / max(nf, 1):6.1%} {ret:6.3f}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('eval_dir', type=Path)
    ap.add_argument('--testsets', nargs='+', required=True)
    ap.add_argument('--top', type=int, default=12)
    args = ap.parse_args(argv)
    for ts in args.testsets:
        p = args.eval_dir / f'{ts}.json'
        if not p.exists():
            print(f'\n## {ts}: missing ({p})')
            continue
        report(p, ts, args.top)
    return 0


if __name__ == '__main__':
    sys.exit(main())
