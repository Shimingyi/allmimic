"""Seconds-level adaptation to new motions (CHANGES #15, experimental): start from a trained checkpoint, keep training
on the new clip(s) only, and record success / return against wall-clock seconds.

  PYTHONPATH=. python -m allmimic.adapt --ckpt runs/bones/ckpt_0120000.pt --motions new_clips.npz \
      --out runs/adapt/clip_x --seconds 60 --eval-every-s 5 [--wm-update 0] [--num-envs 1024]

The clock covers everything that touches the new clips: simulator collection (the first warm-up steps included) and
WM / policy / critic updates. Process setup (simulator, torch.compile, eval env) and the periodic evaluations are
outside it: a demo keeps a warm process, and compiled graphs do not depend on the library, so a new clip needs no
recompilation (warm_compile runs throwaway updates, then restores the checkpoint). t = 0 is the zero-shot score.

Per eval point adapt.jsonl gets {clips, seconds, iterations, updates, success, norm_return, per_clip_*}.
--sweep list.txt adapts to each line's clip(s) in turn in one warm process, restoring the checkpoint state before
each (lines already in adapt.jsonl are skipped, so a rerun resumes). Model, task
encoding (goal / action / heading / offsets) and network sizes come from the checkpoint cfg; reset-sampling bins
start fresh on the new library; the learning rate is constant (--lr, default the run's base lr).
"""

import argparse
import copy
import dataclasses
import json
import os
import sys
import time

import torch

from .train import Trainer, TrainCfg, unknown_cfg_keys


def adapt_cfg(ck_cfg, args):
    """checkpoint cfg dict + CLI overrides -> TrainCfg for the adaptation process"""
    names = {f.name for f in dataclasses.fields(TrainCfg)}
    cfg = TrainCfg(**{k: v for k, v in ck_cfg.items() if k in names})
    cfg.motions, cfg.out, cfg.resume, cfg.clip_prefixes = args.motions, args.out, '', args.clips
    cfg.clip_list = ''  # the run's training whitelist (--clip-list) has none of the new clips (kept: empty library)
    cfg.bptt_checkpoint = 0  # recomputation only saves memory; adapt seconds stay comparable across bases
    cfg.num_envs, cfg.pi_batch, cfg.wm_batch = args.num_envs, args.pi_batch, args.wm_batch
    cfg.warmup_steps, cfg.buffer_steps, cfg.utd = args.warmup_steps, args.buffer_steps, args.utd
    cfg.lr = args.lr or cfg.lr
    cfg.lr_final_frac = 1.0  # constant lr
    cfg.eval_every, cfg.ckpt_every, cfg.wall_hours = 0, 10**9, 1e9
    cfg.seed = args.seed
    return cfg


class Adapter(Trainer):
    def load_nets(self, ck, optimizers=True):
        """networks (+ optimizer moments) from a training checkpoint; replay, bins and counters start fresh"""
        self.wm.load_state_dict(ck['wm'])
        self.policy.load_state_dict(ck['policy'])
        self.critic.load_state_dict(ck['critic'])
        if optimizers:
            self.opt_wm.load_state_dict(ck['opt_wm'])
            self.opt_pi.load_state_dict(ck['opt_pi'])
            self.opt_v.load_state_dict(ck['opt_v'])
        for opt in (self.opt_wm, self.opt_pi, self.opt_v):
            for g in opt.param_groups:
                g['lr'] = self.cfg.lr

    def evaluate_now(self):
        from .evaluate import evaluate, make_eval_env

        if not hasattr(self, 'eval_env'):
            self.eval_env = make_eval_env(self.lib, self.lib.num_clips, self.device, self.tcfg)
        self.policy.eval()
        from . import networks

        amp, networks.AMP['enabled'] = networks.AMP['enabled'], False  # fp32, like tools/eval_allmimic.py
        res = evaluate(lambda o, g: self.policy(o, g), self.lib, self.eval_env)
        networks.AMP['enabled'] = amp
        self.policy.train()
        return res

    def warm_compile(self, ck):
        """one collect + WM / policy update compiles every graph (minutes), then the checkpoint state is restored:
        weights, normalizers and Adam moments from ck, an empty replay buffer, fresh bins, a reset env"""
        while self.buf.size < self.cfg.warmup_steps:
            self.collect()
        self.update_wm()
        self.update_policy()
        self.restore(ck)

    def restore(self, ck):
        """checkpoint weights / normalizers / Adam moments, empty replay buffer, fresh bins, reset env and counters"""
        self.load_nets(ck, optimizers=self.load_opt)
        self.buf = self._make_buffer()
        self.env.set_library(self.lib)
        self.env.reset()
        self.it = self.updates = 0
        self.update_credit = 0.0

    def switch_clips(self, names, ck):
        """sweep step: library of the next clip(s) from the same npz, then restore(ck); no recompilation"""
        from .motion_lib import MotionLibrary

        self.lib = MotionLibrary(self.cfg.motions, self.device, prefixes=names)
        if hasattr(self, 'eval_env'):
            self.eval_env.set_library(self.lib)
        self.restore(ck)

    def log_eval(self, log, clock, tag):
        res = self.evaluate_now()
        rec = {
            'clips': tag, 'seconds': round(clock, 3), 'iterations': self.it, 'updates': self.updates,
            'success': 1 - res['failed'] / max(res['num_clips'], 1), 'norm_return': res['norm_return'],
            'per_clip_norm_return': res['per_clip_norm_return'], 'per_clip_failed': res['per_clip_failed'],
        }  # fmt: skip
        log.write(json.dumps(rec) + '\n')
        log.flush()
        print(f't {clock:6.1f}s it {self.it:5d} upd {self.updates:4d} success {rec["success"]:.3f} '
              f'return {rec["norm_return"]:.4f}', flush=True)  # fmt: skip

    def adapt(self, seconds, eval_every_s, wm_update=True, tag=''):
        """the clock counts collection on the new clips (warm-up included) and updates; evaluations pause it"""
        cfg = self.cfg
        log = open(os.path.join(cfg.out, 'adapt.jsonl'), 'a')
        clock, next_eval = 0.0, eval_every_s
        self.log_eval(log, clock, tag)  # zero-shot
        while clock < seconds:
            torch.cuda.synchronize()
            t0 = time.time()
            self.collect()
            if self.buf.size >= cfg.warmup_steps:
                self.update_credit += cfg.utd
                while self.update_credit >= 1.0:
                    self.update_credit -= 1.0
                    if wm_update:
                        self.update_wm()
                    self.update_policy()
                    self.updates += 1
            torch.cuda.synchronize()
            clock += time.time() - t0
            self.it += 1
            if clock >= next_eval or clock >= seconds:
                self.log_eval(log, clock, tag)
                next_eval += eval_every_s
        log.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--motions', required=True, help='library (.npz) with the new clip(s) only')
    ap.add_argument('--out', required=True)
    ap.add_argument('--clips', default='', help='comma-separated clip names (name prefixes) to keep from --motions')
    ap.add_argument('--sweep', default='', help='file: one --clips value per line, adapted in turn in one process')
    ap.add_argument('--seconds', type=float, default=60.0)
    ap.add_argument('--eval-every-s', type=float, default=5.0)
    ap.add_argument('--wm-update', type=int, default=1, help='1 = keep fitting the world model on the new data')
    ap.add_argument('--fresh-optimizers', action='store_true', help='reset Adam moments instead of loading them')
    ap.add_argument('--scratch', action='store_true',
                    help='control: same budget from a random init (architecture / task encoding of --ckpt)')  # fmt: skip
    ap.add_argument('--num-envs', type=int, default=1024)
    ap.add_argument('--pi-batch', type=int, default=1024)
    ap.add_argument('--wm-batch', type=int, default=1024)
    ap.add_argument('--warmup-steps', type=int, default=32)
    ap.add_argument('--buffer-steps', type=int, default=200)
    ap.add_argument('--utd', type=float, default=1.0, help='updates per collected step (training uses 0.4)')
    ap.add_argument('--lr', type=float, default=0.0, help='constant lr (0 = the run base lr)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    items = [args.clips]
    if args.sweep:
        items = [line.strip() for line in open(args.sweep) if line.strip() and not line.startswith('#')]
        args.clips = items[0]
    ck = torch.load(args.ckpt, map_location='cuda')
    if unknown_cfg_keys(ck['cfg']):  # checkpoint from newer code than this copy
        raise SystemExit(f'{args.ckpt}: cfg keys unknown here {unknown_cfg_keys(ck["cfg"])}; sync the code')
    cfg = adapt_cfg(ck['cfg'], args)
    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, 'adapt_args.json'), 'w') as f:
        json.dump({'args': vars(args), 'cfg': dataclasses.asdict(cfg), 'ckpt_it': ck.get('it')}, f, indent=1)
    ad = Adapter(cfg)
    ad.load_opt = not args.fresh_optimizers
    if args.scratch:  # restore() then always returns to this random init (weights, normalizers, Adam state)
        keys = ('wm', 'policy', 'critic', 'opt_wm', 'opt_pi', 'opt_v')
        mods = (ad.wm, ad.policy, ad.critic, ad.opt_wm, ad.opt_pi, ad.opt_v)
        ck = {'cfg': ck['cfg'], **{k: copy.deepcopy(m.state_dict()) for k, m in zip(keys, mods)}}
    t0 = time.time()
    ad.warm_compile(ck)
    print(f'setup + compile {time.time() - t0:.0f}s (outside the clock)', flush=True)
    log = os.path.join(cfg.out, 'adapt.jsonl')
    done = {json.loads(line)['clips'] for line in open(log)} if os.path.exists(log) else set()
    for i, names in enumerate(items):
        if names in done:
            continue
        if i:
            ad.switch_clips(names, ck)
        print(f'[{i + 1}/{len(items)}] {names[:100]}', flush=True)
        ad.adapt(args.seconds, args.eval_every_s, wm_update=bool(args.wm_update), tag=names)
    return 0


if __name__ == '__main__':
    sys.exit(main())
