"""Evaluate an allmimic checkpoint on test-set libraries (paper App. F protocol, allmimic's simulator and evaluate()).

  PYTHONPATH=. python tools/eval_allmimic.py --ckpt runs/bones/ckpt_0120000.pt \
      --testsets lafan1 lafan1_seg10 bones_test_content_seg10 --out-dir runs/bones/eval/ckpt_0120000

Test-set libraries: <lib-dir>/<testset>.npz (default data/library_eval; built by tools/build_allmimic_library.py,
--segments 10 for the _seg10 versions). The goal encoding is rebuilt from the checkpoint's cfg; the policy runs in
fp32 (the in-run evaluation of allmimic.train uses bf16). Output: <testset>.json {summary, clips} per test set and
a merged summary.json (read by tools/failure_report.py and tools/filter_infeasible.py rescore).
Returns are the paper normalization: reward sum / (0.1 * (frames - 1)).
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

MAX_STEP_REWARD = 0.1


def summarize(rows):
    n = len(rows)
    max_tot = sum(MAX_STEP_REWARD * max(r['frames'] - 1, 1) for r in rows)
    return {
        'clips': n,
        'failed': sum(r['failed'] for r in rows),
        'success_rate': sum(not r['failed'] for r in rows) / max(n, 1),
        'mean_ret_paper': sum(r['ret_paper'] for r in rows) / max(n, 1),
        # per-clip return clipped to [0, 1]: one diverged clip can score -1500 and swamp the mean
        'mean_ret_clip01': sum(min(max(r['ret_paper'], 0.0), 1.0) for r in rows) / max(n, 1),
        'frame_weighted_ret_paper': sum(r['ret_sum'] for r in rows) / max(max_tot, 1e-9),
        'hours': sum(r['frames'] for r in rows) / 50 / 3600,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--testsets', nargs='+', required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--lib-dir', type=Path, default=Path('data/library_eval'))
    ap.add_argument('--num-envs', type=int, default=4096)
    args = ap.parse_args(argv)

    from allmimic import networks
    from allmimic.evaluate import evaluate, load_supermimic_policy, make_eval_env
    from allmimic.motion_lib import MotionLibrary
    from allmimic.task import task_cfg
    from allmimic.train import unknown_cfg_keys

    networks.AMP['enabled'] = False
    torch.backends.cuda.matmul.allow_tf32 = True
    device = 'cuda'
    pol, ck = load_supermimic_policy(args.ckpt, device)
    bad = unknown_cfg_keys(ck.get('cfg', {}))
    if bad:  # checkpoint from newer code than this copy
        raise SystemExit(f'{args.ckpt}: cfg keys unknown here {bad}; the checkpoint comes from newer code')
    tcfg = task_cfg(ck.get('cfg', {}), train=False)  # goal / action encoding of the run, without training noise
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / 'summary.json'
    table = json.loads(summary_path.read_text())['testsets'] if summary_path.exists() else {}
    for ts in args.testsets:
        path = args.lib_dir / f'{ts}.npz'
        if not path.exists():
            print(f'{ts:22s} SKIPPED: {path} not built yet', flush=True)
            continue
        t0 = time.time()
        lib = MotionLibrary(str(path), device)
        env = make_eval_env(lib, min(lib.num_clips, args.num_envs), device, tcfg)
        with torch.no_grad():
            res = evaluate(lambda o, g: pol(o, g), lib, env)
        frames = lib.lengths.tolist()
        rows = []
        for i, name in enumerate(lib.names):
            ret_paper = res['per_clip_norm_return'][i]
            rows.append({
                'name': name, 'frames': frames[i], 'steps': res['per_clip_steps'][i], 'failed': res['per_clip_failed'][i],
                'ret_paper': ret_paper, 'ret_sum': ret_paper * MAX_STEP_REWARD * max(frames[i] - 1, 1),
            })  # fmt: skip
        s = summarize(rows)
        s['eval_seconds'] = round(time.time() - t0, 1)
        table[ts] = s
        (args.out_dir / f'{ts}.json').write_text(json.dumps({'summary': s, 'clips': rows}, indent=1))
        summary_path.write_text(json.dumps({'ckpt': str(args.ckpt), 'testsets': table}, indent=1))
        print(f"{ts:22s} clips {s['clips']:5d} | success {s['success_rate']:.4f} ({s['failed']} failed) | "
              f"ret_paper {s['mean_ret_paper']:.4f} (clip01 {s['mean_ret_clip01']:.4f}, fw {s['frame_weighted_ret_paper']:.4f}) | {s['eval_seconds']:.0f}s",
              flush=True)  # fmt: skip
        del env, lib
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    sys.exit(main())
