"""SuperMimic training loop (Algorithm A.1).

Each iteration collects one simulator step from every environment. World-model and
policy updates run at `utd` updates per iteration (0.4 in the paper), so two updates
happen every five iterations. One update is `wm_batches` world-model gradient steps on
2048 k-step replay windows, then one BPTT policy step on 4096 imagined rollouts of
H steps, then critic fitting on the same rollouts.

  PYTHONPATH=. python -m allmimic.train --preset recommended --motions data/processed/lafan1_50hz.npz --out runs/lafan1
  PYTHONPATH=. torchrun --standalone --nproc_per_node=4 -m allmimic.train --preset recommended ...   # lr x sqrt(world)

Every TrainCfg field is a flag (--clip-list, --pi-batch, ...); --preset picks a named set of them (PRESETS).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time

import numpy as np
import torch
import torch.utils.checkpoint

from .motion_lib import MotionLibrary
from . import networks
from .networks import Critic, Policy, WorldModel, apply_delta_c, apply_delta_tri_c, run_mlp, state_delta_c
from .networks import state_delta_tri_c, state_error, state_error_c
from .replay import ReplayBuffer
from .sim_env import EnvCfg, TrackingSimEnv
from .task import GOAL_FRAME_DIM, NJ, OBS_DIM, XI_DIM, flat_to_state, state_to_flat, task_cfg


@dataclasses.dataclass
class TrainCfg:
    motions: str = 'data/processed/lafan1_50hz.npz'
    clip_prefixes: str = ''  # e.g. "walk,run,sprint" for the Fig. 4b subset
    clip_list: str = ''  # file with one clip name per line to keep (feasible whitelist); empty = all clips
    softmax_cap: float = 0.0  # EnvCfg.softmax_cap: per-bin cap of the reset-sampling softmax, x uniform share (#19)
    out: str = 'runs/debug'
    seed: int = 0
    num_envs: int = 4096
    buffer_steps: int = 1000
    iterations: int = 320_000
    utd: float = 0.4
    warmup_steps: int = 64
    # World model.
    k: int = 8
    wm_batch: int = 2048
    wm_batches: int = 10
    # Policy / critic.
    horizon: int = 32
    pi_batch: int = 4096
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 2e-4
    lr_final_frac: float = 0.03
    lr_decay_iters: int = 250_000
    critic_epochs: int = 5
    critic_minibatches: int = 4
    action_noise: float = 0.1
    # #23 (deterministic deployment collapse): the WM only sees noisy-action data while imagination is deterministic
    collect_det_frac: float = 0.0  # share of envs that collect without action noise, as deployed
    img_action_noise: float = 0.0  # N(0, x) on imagined actions, like the collected ones (reparameterized)
    # #24: the policy loss is divided by H, so its gradient norm is ~0.02 and the 1.0 clip never fires after ~5% of
    # training; 1 / H (= 1/32) is equivalent under Adam to "no 1/H in the loss, clip 1.0"
    pi_clip: float = 1.0
    critic_heads: int = 1  # > 1: critic ensemble, per-head lambda-returns, policy bootstraps on the mean
    wm_rot_out: str = 'axis'  # 'tripoint': the WM predicts the rotation change as tri-point markers (#25)
    # #26 (experimental): query the WM only where it was fit, and measure when that fails
    clean_prev_action: bool = False  # the obs' last action and the action-rate term use the pre-noise action
    w_residual: float = 0.0  # TaskCfg.w_residual: reward x sum of squared residual outputs (e.g. -0.02)
    retire_g: float = 0.0  # EnvCfg.retire_g: down-weight bins whose return stays below this (unlearnable tail)
    bptt_detach: int = 0  # > 0: detach the imagined state chain every k steps (truncated BPTT; J / critic unchanged)
    gap_every: int = 0  # > 0: imagination-gap monitor every k iterations (deterministic policy, sim vs WM)
    gap_envs: int = 256
    # cluster-balanced reset sampling (tools/bin_clusters.py): weights per 1 s bin, x the return-based probabilities
    bin_prior: str = ''  # npz written by tools/bin_clusters.py
    bin_prior_alpha: float = 0.5  # which weight set: cluster share ~ size^alpha
    # 'muon': Muon on the WM / policy hidden weight matrices, Adam on the rest (#27)
    optimizer: str = 'adam'
    use_critic: bool = True
    # Bookkeeping.
    log_every: int = 100
    ckpt_every: int = 4000
    eval_every: int = 4000
    eval_clips: int = 0  # 0 = all clips
    eval_envs: int = 4096
    wall_hours: float = 1e9
    tf32: bool = True
    amp: bool = True
    compile_wm: bool = True
    wm_compile_mode: str = (
        ''  # torch.compile mode of the WM window loss: '' = default (upstream), 'reduce-overhead' (#16)
    )
    lr_sqrt_world: bool = True
    norm_update_every: int = 5
    resume: str = ''
    goal_pos: str = 'robot'  # TaskCfg.goal_pos: "robot" (upstream) or "ref" (deployable)
    goal_rot: str = 'robot'  # TaskCfg.goal_rot: "robot" (upstream) or "ref" (goal without robot state)
    goal_pos_noise: float = 0.0  # TaskCfg.goal_pos_noise: training noise (m) on goal root positions
    action: str = 'abs'  # TaskCfg.action: "abs" setpoints around q0 (upstream) or "res" residual on the reference (#11)
    res_scale: float = 1.0  # TaskCfg.res_scale: residual scale in units of action_scale
    future_offsets: str = ''  # TaskCfg.future_offsets, e.g. "1,2,3,4,6,10,15,25,40,60" (#12); empty = 1..10
    latent: int = 32  # policy motion latent width (tanh bottleneck; #13)
    # torch.compile mode for one imagination step (Trainer.img_step: policy, WM, analytics) as one region, called H
    # times: '' = eager reference (upstream), 'default', 'max-autotune-no-cudagraphs', 'reduce-overhead' (#16).
    compile_bptt: str = ''
    # recompute every k-th imagined step in backward instead of storing its activations (#17; 0 = off). k = 1 at pi_batch
    # 4096: update_policy peak 14.5 -> 7.1 GB, 1.68x time; k = 2 saves about half of that for about half the time, so
    # pi_batch 4096 (dhmbb2 default) fits on a 24 GB card next to the 6.5 GB BONES library
    bptt_checkpoint: int = 0
    heading: str = 'xaxis'  # TaskCfg.heading: "xaxis" (upstream) or "twist" (no singularity when lying / crawling; #14)
    res_legs: str = 'res'  # TaskCfg.res_legs: "res" or "abs" (legs absolute, upper body residual; #20)
    res_scale_legs: float = 0.0  # TaskCfg.res_scale_legs: legs' own residual scale (0 = res_scale)
    # WM loss weights on the q / qd error dims from joint error propagation (allmimic.joint_weights; '' = uniform, upstream),
    # clipped to [1 / clip, clip] and renormalized to mean 1 (#21: truncate lever weights)
    wm_joint_weights: str = ''
    wm_joint_weight_clip: float = 4.0
    wm_norm_eps: float = 1e-3  # WM input normalizer std floor (upstream 1e-3; #20 suggests ~0.05)
    rot_repr: str = '6d'  # TaskCfg.rot_repr: "6d" (upstream) or "tripoint" (tri-point markers; inputs only; #18)


class _Timer:
    def __init__(self, on):
        self.on = on
        self.t = {}
        if on:
            torch.cuda.synchronize()
        self.last = time.time()

    def __call__(self, name):
        if not self.on:
            return
        torch.cuda.synchronize()
        now = time.time()
        self.t[name] = self.t.get(name, 0.0) + now - self.last
        self.last = now


def zero_if_nonfinite(params, gnorm):
    """Last-resort guard (#5): a non-finite gradient norm zeroes this step's gradients, without a host sync."""
    ok = torch.isfinite(gnorm)
    for p in params:
        if p.grad is not None:
            p.grad.copy_(torch.where(ok, p.grad, torch.zeros_like(p.grad)))


class MuonAdam:
    """Muon on the hidden weight matrices of the given MLPs, Adam (the trainer's previous settings) on their input /
    output layers and biases, behind the optimizer interface the trainer uses (param_groups for the LR schedule,
    zero_grad, step, state_dict). Muon's LR matches Adam's update RMS (adjust_lr_fn='match_rms_adamw'), so the same
    cfg.lr applies; no weight decay, as with Adam here."""

    def __init__(self, mlps, lr, betas=(0.9, 0.999)):
        muon, adam = [], []
        for net in mlps:
            lin = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
            for i, m in enumerate(lin):
                (muon if 0 < i < len(lin) - 1 else adam).append(m.weight)
                adam.append(m.bias)
        self.opts = [
            torch.optim.Muon(muon, lr=lr, weight_decay=0.0, adjust_lr_fn='match_rms_adamw'),
            torch.optim.Adam(adam, lr=lr, betas=betas, fused=True),
        ]

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd):
            o.load_state_dict(s)


def unknown_cfg_keys(run_cfg):
    """checkpoint cfg dict -> keys this TrainCfg does not define: the checkpoint comes from newer code and rebuilding
    it here would silently drop those switches (#22: an older evaluator without rot_repr / res_legs would score such
    checkpoints with wrong inputs and no error, since the goal / action shapes do not change)"""
    return sorted(set(run_cfg) - {f.name for f in dataclasses.fields(TrainCfg)})


def lr_at(cfg: TrainCfg, it: int) -> float:
    if it >= cfg.lr_decay_iters:
        return cfg.lr * cfg.lr_final_frac
    c = 0.5 * (1 + math.cos(math.pi * it / cfg.lr_decay_iters))
    return cfg.lr * (cfg.lr_final_frac + (1 - cfg.lr_final_frac) * c)


class Trainer:
    def __init__(self, cfg: TrainCfg, device='cuda'):
        self.cfg = cfg
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
        if self.world > 1:
            import datetime

            import torch.distributed as dist

            local = int(os.environ.get('LOCAL_RANK', '0'))
            torch.cuda.set_device(local)
            device = f'cuda:{local}'
            dist.init_process_group('nccl', timeout=datetime.timedelta(hours=2))
            networks.DIST['world'] = self.world
            if cfg.lr_sqrt_world:
                # Paper: 2 GPUs use sqrt(2) x 2e-4.
                cfg.lr = cfg.lr * math.sqrt(self.world)
        self.device = device
        torch.manual_seed(cfg.seed)
        if cfg.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        networks.AMP['enabled'] = cfg.amp
        keep = {n.strip() for n in open(cfg.clip_list).read().splitlines() if n.strip()} if cfg.clip_list else None
        self.lib = MotionLibrary(cfg.motions, device, prefixes=cfg.clip_prefixes, keep=keep)
        self.tcfg = task_cfg(dataclasses.asdict(cfg))
        self.env = self._make_env()
        if cfg.bin_prior:
            self.env.bin_prior = self._bin_prior()
        self.task = self.env.task
        self.task.compile_sizes |= {cfg.wm_batch, cfg.pi_batch}
        s = self.env.reset()
        self.feat_dim = self.task.state_features(s).shape[-1]
        self.obs_dim = OBS_DIM * self.tcfg.history
        self.goal_dim = GOAL_FRAME_DIM * self.tcfg.future
        self.wm = WorldModel(self.task, self.feat_dim, XI_DIM, norm_eps=cfg.wm_norm_eps, rot_out=cfg.wm_rot_out).to(
            device
        )
        self.policy = Policy(self.obs_dim, self.goal_dim, latent=cfg.latent).to(device)
        # reference-frame goals: the critic also gets the robot-relative goal pose (task.goal_priv, #4)
        self.priv_dim = 9 * self.tcfg.future if self.task.priv_goal else 0
        self.critic_in = self.feat_dim + self.goal_dim + XI_DIM + 1 + self.priv_dim
        self.critic = Critic(self.critic_in, heads=cfg.critic_heads).to(device)
        if cfg.optimizer == 'muon':
            self.opt_wm = MuonAdam([self.wm.net], cfg.lr)
            self.opt_pi = MuonAdam([self.policy.encoder, self.policy.decoder], cfg.lr, betas=(0.9, 0.95))
        else:
            self.opt_wm = torch.optim.Adam(self.wm.net.parameters(), lr=cfg.lr, fused=True)
            self.opt_pi = torch.optim.Adam(
                list(self.policy.encoder.parameters()) + list(self.policy.decoder.parameters()),
                lr=cfg.lr,
                betas=(0.9, 0.95),
                fused=True,
            )
        self.opt_v = torch.optim.Adam(self.critic.net.parameters(), lr=cfg.lr, fused=True)
        if self.world > 1:
            import torch.distributed as dist

            for mod in (self.wm, self.policy, self.critic):
                for t in list(mod.parameters()) + list(mod.buffers()):
                    dist.broadcast(t.data, 0)
            torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.buf = self._make_buffer()
        self.wm_err_w = self._wm_error_weights()
        self.gap_env, self.gap_stats = None, {}  # #26 imagination-gap monitor (TrainCfg.gap_every)
        self._wm_loss = self.wm_window_loss
        if cfg.compile_wm:
            self._wm_loss = torch.compile(self.wm_window_loss, dynamic=False, mode=cfg.wm_compile_mode or None)
        self._img_step = self.img_step
        if cfg.compile_bptt:  # whole-rollout compile needed >10 GB host RAM and >15 min; one step compiles once
            self._img_step = torch.compile(self.img_step, dynamic=False, mode=cfg.compile_bptt)
        self._img_step_ckpt = None
        if cfg.bptt_checkpoint:  # the recomputation replays the RNG state: gradients bit-identical
            step = self._img_step
            self._img_step_ckpt = lambda *a: torch.utils.checkpoint.checkpoint(step, *a, use_reentrant=False)
        self.it = 0
        self.updates = 0
        self.env_steps = 0
        self.train_time = 0.0
        self.update_credit = 0.0
        self.ep_returns = []
        self.ep_norm_returns = []
        self.ep_terms = []
        self.ep_lens = []
        self.ep_step_rewards = []
        self.profile = bool(int(os.environ.get('SMIMIC_PROFILE', '0')))
        self.prof = {}
        os.makedirs(cfg.out, exist_ok=True)
        self.log_f = open(os.path.join(cfg.out, 'log.jsonl' if self.rank == 0 else f'log_rank{self.rank}.jsonl'), 'a')
        if cfg.resume:
            self.load(cfg.resume)

    def _bin_prior(self):
        """TrainCfg.bin_prior npz (tools/bin_clusters.py: names, lengths, per-bin weights per alpha) -> (num_bins,)
        weights in this library's bin order, renormalized to mean 1 over its bins (the whitelist may drop clips)"""
        d = np.load(self.cfg.bin_prior)
        w_all = d[f'w_alpha_{self.cfg.bin_prior_alpha:g}']
        nb_all = (d['lengths'] + self.lib.bin_len - 1) // self.lib.bin_len
        off = np.concatenate([[0], np.cumsum(nb_all)])
        at = {str(n): i for i, n in enumerate(d['names'])}
        nb = ((self.lib.lengths + self.lib.bin_len - 1) // self.lib.bin_len).tolist()
        parts = []
        for name, n in zip(self.lib.names, nb):
            i = at[name]
            assert nb_all[i] == n, f'{name}: {nb_all[i]} bins in {self.cfg.bin_prior}, {n} here'
            parts.append(w_all[off[i] : off[i + 1]])
        w = torch.tensor(np.concatenate(parts), dtype=torch.float32, device=self.device)
        assert len(w) == self.lib.num_bins
        return w / w.mean()

    def _make_env(self):
        return TrackingSimEnv(
            self.lib,
            EnvCfg(num_envs=self.cfg.num_envs, softmax_cap=self.cfg.softmax_cap, retire_g=self.cfg.retire_g),
            self.tcfg,
            self.device,
            gamma=self.cfg.gamma,
        )

    def _make_buffer(self):
        res = self.tcfg.action == 'res'
        return ReplayBuffer(self.cfg.num_envs, self.cfg.buffer_steps, XI_DIM, self.device, policy_actions=res)

    def _extra_state(self):
        return {}

    def _load_extra(self, ck):
        pass

    def sync_grads(self, params):
        """Average gradients across data-parallel ranks (no-op on one GPU)."""
        if self.world == 1:
            return
        import torch.distributed as dist

        grads = [p.grad for p in params if p.grad is not None]
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat)
        flat /= self.world
        i = 0
        for g in grads:
            n = g.numel()
            g.copy_(flat[i : i + n].view_as(g))
            i += n

    # Checkpoints.

    def save(self, path):
        torch.save(
            {
                'cfg': dataclasses.asdict(self.cfg),
                'it': self.it,
                'updates': self.updates,
                'env_steps': self.env_steps,
                'train_time': self.train_time,
                'wm': self.wm.state_dict(),
                'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'opt_wm': self.opt_wm.state_dict(),
                'opt_pi': self.opt_pi.state_dict(),
                'opt_v': self.opt_v.state_dict(),
                'bin_G': self.env.bin_G,
                'bin_visits': self.env.bin_visits,
                'bin_past_w': self.env.bin_past_w,
                **self._extra_state(),
            },
            path + '.tmp',
        )
        os.replace(path + '.tmp', path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.wm.load_state_dict(ck['wm'])
        self.policy.load_state_dict(ck['policy'])
        self.critic.load_state_dict(ck['critic'])
        self.opt_wm.load_state_dict(ck['opt_wm'])
        self.opt_pi.load_state_dict(ck['opt_pi'])
        self.opt_v.load_state_dict(ck['opt_v'])
        self.env.bin_G[:] = ck['bin_G']
        self.env.bin_visits[:] = ck['bin_visits']
        self.env.bin_past_w[:] = ck['bin_past_w']
        self.it, self.updates = ck['it'], ck['updates']
        self._load_extra(ck)
        self.env_steps, self.train_time = ck['env_steps'], ck['train_time']
        print(f'resumed from {path} at iteration {self.it}')

    # Data collection.

    @torch.no_grad()
    def collect(self, random=False):
        env = self.env
        obs, goal = env.policy_inputs()
        self.policy.obs_norm.update(obs)
        self.policy.goal_norm.update(goal)
        a = self.policy(obs, self.task.noisy_goal(goal))
        noise = self.cfg.action_noise * torch.randn_like(a)
        if self.cfg.collect_det_frac:  # the first envs act deterministically, as at deployment
            noise[: int(round(self.cfg.collect_det_frac * env.num_envs))] = 0
        a_clean, a = a, a + noise
        s_flat = state_to_flat(env.state)
        xi, frame, ep = env.xi.clone(), env.frame.clone(), env.ep_id.clone()
        s_next, info = env.step(a, obs_action=a_clean if self.cfg.clean_prev_action else None)
        if self.cfg.clean_prev_action:
            a = a_clean  # replay a_pi (obs histories) holds the clean action; action_sp keeps the executed one
        # diverged worlds get episode id -1 so no WM window or imagination start uses the transition (#5)
        ep = torch.where(info['diverged'], torch.full_like(ep, -1), ep)
        self.buf.add(s_flat, info['action_sp'], state_to_flat(s_next), xi, frame, ep, a_pi=a)
        self.env_steps += env.num_envs
        if 'done_ep_return' in info:
            self.ep_returns.append(info['done_ep_return'].mean().item())
            L = info['done_ep_len'].float()
            self.ep_norm_returns.append(info['done_norm_return'].mean().item())
            self.ep_step_rewards.append((info['done_ep_return'] / (L * self.tcfg.max_step_reward)).mean().item())
            self.ep_terms.append(info['done_terminated'].float().mean().item())
            self.ep_lens.append(L.mean().item())

    # World model.

    def update_wm(self):
        cfg, wm, task = self.cfg, self.wm, self.task
        stats = {'wm_loss': 0.0, 'wm_err1': 0.0, 'wm_errk': 0.0}
        for bi in range(cfg.wm_batches):
            s0, A, S1, xi, mask = self.buf.sample_windows(cfg.wm_batch, cfg.k)
            s = flat_to_state(s0)
            m = mask.float()
            if bi == 0 and self.updates % cfg.norm_update_every == 0:
                # Normalizer statistics from the true transitions of valid windows.
                with torch.no_grad():
                    true_prev = s
                    xs, ds, es = [], [], []
                    delta = state_delta_c if wm.err_norm is None else state_delta_tri_c
                    for j in range(cfg.k):
                        t1 = flat_to_state(S1[:, j])
                        xs.append(wm.inputs(true_prev, A[:, j], xi)[mask])
                        ds.append(delta(true_prev, t1)[mask])
                        if wm.err_norm is not None:
                            es.append(state_delta_c(true_prev, t1)[mask])
                        true_prev = t1
                    wm.in_norm.update(torch.cat(xs))
                    wm.out_norm.update(torch.cat(ds))
                    if es:
                        wm.err_norm.update(torch.cat(es))
            loss, err1, errk = self._wm_loss(s0, A, S1, xi, m)
            self.opt_wm.zero_grad(set_to_none=True)
            loss.backward()
            self.sync_grads(list(wm.net.parameters()))
            zero_if_nonfinite(wm.net.parameters(), torch.nn.utils.clip_grad_norm_(wm.net.parameters(), 1.0))
            self.opt_wm.step()
            stats['wm_loss'] += loss.detach() / cfg.wm_batches
            stats['wm_err1'] += err1 / cfg.wm_batches
            stats['wm_errk'] += errk / cfg.wm_batches
        stats = {kk: float(vv) for kk, vv in stats.items()}
        return stats

    def _wm_error_weights(self):
        """TrainCfg.wm_joint_weights -> (70,) weights on the WM state-error dims (root 12 = 1, q 29 and qd 29 = joint
        weights in the env's joint order, clipped and mean-1), or None for the upstream uniform loss"""
        if not self.cfg.wm_joint_weights:
            return None
        from .joint_weights import JOINT_WEIGHTS

        table, c = JOINT_WEIGHTS[self.cfg.wm_joint_weights], self.cfg.wm_joint_weight_clip
        w = torch.tensor([table[n] for n in self.env.joint_names], device=self.device).clamp(1 / c, c)
        w = w / w.mean()
        return torch.cat([torch.ones(12, device=self.device), w, w])

    def wm_window_loss(self, s0, A, S1, xi, m):
        """k-step autoregressive loss on logged windows (Eq. 1, state term)."""
        wm = self.wm
        std = wm.err_std
        s = flat_to_state(s0)
        loss = 0.0
        errs = []
        for j in range(self.cfg.k):
            s = wm.step(s, A[:, j], xi)
            e = (state_error(s, flat_to_state(S1[:, j])) / std).pow(2)
            errs.append(e.mean(-1).detach())  # logged unweighted, comparable across --wm-joint-weights runs
            loss = loss + (e if self.wm_err_w is None else e * self.wm_err_w).mean(-1)
        msum = m.sum().clamp_min(1)
        loss = (loss * m).sum() / msum
        return loss, (errs[0] * m).sum() / msum, (errs[-1] * m).sum() / msum

    # Policy and critic.

    def critic_input(self, feats, goal, xi, rem, priv=None):
        parts = [feats, goal, xi, rem[:, None]]
        if self.priv_dim:
            parts.append(priv)
        return torch.cat(parts, dim=-1)

    def imagination_inputs(self, starts=None):
        """replay starts (or explicit `starts` = s0, xi, frame, obs_hist, a_prev, start_ok) + reference window -> args
        of rollout(): s0, xi, obs_hist, a_prev, bias, ref, ref_kin, valid (B,H+1), rem_all (B,H+1)"""
        cfg, task, lib = self.cfg, self.task, self.lib
        B, H = cfg.pi_batch, cfg.horizon
        dev = self.device
        if starts is None:
            starts = self.buf.sample_starts(B, self.tcfg.history, task, with_ok=True)
        s0, xi, frame, obs_hist, a_prev, start_ok = starts
        bias = xi[:, 1 : 1 + NJ]
        offs = self.tcfg.offsets
        with torch.no_grad():
            steps = torch.arange(0, H + offs[-1] + 1, device=dev)
            idx_raw = frame[:, None] + steps
            idx = lib.clamp_to_clip(idx_raw, frame)
            valid = (idx_raw == idx)[:, : H + 1].float()  # frame t still inside the clip
            valid = valid * start_ok[:, None].float()  # start slot is a real transition (#5)
            ref = lib.frame(idx)
            ref_kin = self.env.kin_c(ref['pos'], ref['rot'], ref['q'], ref['v'], ref['w'], ref['qd'])
            rem_all = lib.remaining(idx[:, : H + 1]).float().clamp(max=100) / 100.0
        return s0, xi, obs_hist, a_prev, bias, ref, ref_kin, valid, rem_all

    def update_policy(self):
        cfg, wm, H = self.cfg, self.wm, self.cfg.horizon
        args = self.imagination_inputs()
        valid = args[7]
        for p in wm.net.parameters():
            p.requires_grad_(False)
        J, c, rec, rew_terms, xH = self.rollout(*args)
        loss_pi = -J.mean() / H
        self.opt_pi.zero_grad(set_to_none=True)
        loss_pi.backward()
        params = [p for g in self.opt_pi.param_groups for p in g['params']]
        self.sync_grads(params)
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.pi_clip)
        zero_if_nonfinite(params, gnorm)
        torch.nn.utils.clip_grad_value_(params, 0.1)
        self.opt_pi.step()
        for p in wm.net.parameters():
            p.requires_grad_(True)
        self.critic.zero_grad(set_to_none=True)
        stats = {
            'pi_J': J.mean().item(),
            'pi_loss': loss_pi.item(),
            'pi_gnorm': gnorm.item(),
            'img_survival': c.mean().item(),
        }
        stats.update({f'img_r_{k}': v.item() for k, v in rew_terms.items()})
        if cfg.use_critic:
            stats.update(self.update_critic(rec, xH.detach(), valid))
        return stats

    def rollout(self, s0, xi, obs_hist, a_prev, bias, ref, ref_kin, valid, rem_all):
        """H imagined steps from replay starts through the frozen WM -> (J (B,) discounted return incl. the critic
        bootstrap, c (B,) survival, rec lists for the critic, mean reward terms, xH (B,D) last critic input).
        Each step is img_step (compiled as one region with TrainCfg.compile_bptt, #16); eager = reference."""
        cfg, task = self.cfg, self.task
        B, H, K = s0.shape[0], cfg.horizon, self.tcfg.future
        dev = self.device

        def ref_at(t):
            return {k: v[:, t] for k, v in ref.items()}

        off = self.env.future_off

        def ref_future(t):
            if self.tcfg.future_offsets:
                return {k: v[:, t + off] for k, v in ref.items()}
            return {k: v[:, t + 1 : t + 1 + K] for k, v in ref.items()}  # upstream slice (contiguous offsets)

        def ref_kin_at(t):
            return tuple(x[:, t] for x in ref_kin)

        s = flat_to_state(s0)
        hist = obs_hist + task.sample_obs_noise((B, self.tcfg.history))
        an = task.analytics_fast(s, ref_at(0), ref_kin_at(0), ref_future(0), a_prev, a_prev, bias)
        c = torch.ones(B, device=dev)
        J = torch.zeros(B, device=dev)
        rec = {'x': [], 'r': [], 'd': [], 'm': [], 'c': []}
        rew_terms = {}
        for t in range(H):
            goal, feats = an['goal'], an['feats']
            rec['x'].append(self.critic_input(feats, goal, xi, rem_all[:, t], an.get('goal_priv')).detach())
            ck = self._img_step_ckpt is not None and t % cfg.bptt_checkpoint == 0
            s1, hist, a, an = (self._img_step_ckpt if ck else self._img_step)(
                s, hist, goal, feats, a_prev, xi, bias, ref_at(t + 1), ref_kin_at(t + 1), ref_future(t + 1)
            )
            r, d = an['r'], an['soft']
            m = valid[:, t + 1]
            J = J + (cfg.gamma**t) * c * r * m
            rec['r'].append((r * m).detach())
            rec['d'].append(d.detach())
            rec['m'].append(m)
            rec['c'].append(c.detach())
            for kk, vv in an['terms'].items():
                rew_terms[kk] = rew_terms.get(kk, 0.0) + vv.detach().mean() / H
            c = c * (1 - d)
            s, a_prev = s1, a
            if cfg.bptt_detach and (t + 1) % cfg.bptt_detach == 0:  # truncated BPTT through the state chain
                s, hist = {k: v.detach() for k, v in s.items()}, hist.detach()
        xH = self.critic_input(an['feats'], an['goal'], xi, rem_all[:, H], an.get('goal_priv'))
        if cfg.use_critic:
            vH = self.critic(xH)
            vH = (vH if cfg.critic_heads == 1 else vH.mean(-1)) * valid[:, H]
            J = J + (cfg.gamma**H) * c * vH
        return J, c, rec, rew_terms, xH

    def img_step(self, s, hist, goal, feats, a_prev, xi, bias, ref1, ref_kin1, fut1):
        """one imagined step: policy -> absolute setpoint -> frozen WM -> analytic reward / termination / obs of the
        next state -> (s1, noisy obs history (B,hist,93), action (B,29), analytics of s1 without body kinematics)"""
        task, wm = self.task, self.wm
        a = self.policy(hist.flatten(1), task.noisy_goal(goal))
        a_exec = a + self.cfg.img_action_noise * torch.randn_like(a) if self.cfg.img_action_noise else a
        if not self.cfg.clean_prev_action:
            a = a_exec  # #23: the noisy action also drives the action-rate term and the obs history, as collected
        a_sp = task.setpoint_action(a_exec, ref1['q'])  # the WM consumes absolute setpoints
        dlt = run_mlp(wm.net, wm.in_norm(torch.cat([feats, a_sp, xi], dim=-1))) * wm.out_norm.std + wm.out_norm.mean
        s1 = (apply_delta_c if wm.rot_out == 'axis' else apply_delta_tri_c)(s, dlt)
        an = task.analytics_fast(s1, ref1, ref_kin1, fut1, a, a_prev, bias)
        o1 = an['obs'] + task.sample_obs_noise((a.shape[0],))
        hist1 = torch.cat([hist[:, 1:], o1.unsqueeze(1)], dim=1)
        return s1, hist1, a, {k: v for k, v in an.items() if k != 'kin'}

    @torch.no_grad()
    def imagination_gap(self):
        """#26 imagination-gap monitor: the deterministic policy, from the same states, H steps in a small simulator
        (deployment settings: no domain randomization, reset / observation / action noise) and through the WM
        (imagination with its noises off) -> logged returns and survival. Model exploitation (#23 / #24) shows as the
        WM return rising above the simulator's before any evaluation sees it."""
        cfg, task = self.cfg, self.task
        if self.gap_env is None:
            n = cfg.gap_envs
            ecfg = EnvCfg(num_envs=n, episode_length_s=1e9, obs_noise=False, reset_noise=False, domain_rand=True)
            self.gap_env = TrackingSimEnv(self.lib, ecfg, self.tcfg, self.device, gamma=cfg.gamma)
            self.gap_env.eval_mode = True
            self.gap_env.reset()
            for _ in range(50):  # leave the reference resets: states of the deterministic policy
                self.gap_env.step(self.policy(*self.gap_env.policy_inputs()))
        env, H, n = self.gap_env, cfg.horizon, cfg.gap_envs
        starts = (state_to_flat(env.state), env.xi, env.frame.clone(), env.obs_hist.clone(), env.prev_action.clone(),
                  torch.ones(n, dtype=torch.bool, device=self.device))  # fmt: skip
        saved = (self._img_step, self._img_step_ckpt, cfg.img_action_noise, task.obs_noise.clone())
        self._img_step, self._img_step_ckpt, cfg.img_action_noise = self.img_step, None, 0.0
        task.obs_noise.zero_()
        try:
            _, c_wm, rec, _, _ = self.rollout(*self.imagination_inputs(starts))
        finally:
            self._img_step, self._img_step_ckpt, cfg.img_action_noise = saved[:3]
            task.obs_noise.copy_(saved[3])
        j_wm = sum((cfg.gamma**t) * rec['c'][t] * rec['r'][t] for t in range(H))
        j_sim = torch.zeros(n, device=self.device)
        alive = torch.ones(n, device=self.device)
        fell = torch.zeros(n, dtype=torch.bool, device=self.device)
        for t in range(H):
            _, info = env.step(self.policy(*env.policy_inputs()))
            j_sim += (cfg.gamma**t) * alive * info['reward']
            fell |= info['terminated'] & (alive > 0)
            alive = alive * (~info['done']).float()
        return {
            'gap_J': (j_wm - j_sim).mean().item(),
            'gap_J_wm': j_wm.mean().item(),
            'gap_J_sim': j_sim.mean().item(),
            'gap_surv_wm': c_wm.mean().item(),
            'gap_surv_sim': 1 - fell.float().mean().item(),
        }

    def update_critic(self, rec, xH, valid):
        cfg = self.cfg
        H = cfg.horizon
        X = torch.stack(rec['x'] + [xH], dim=1)  # (B, H+1, D)
        self.critic.norm.update(X)
        ens = cfg.critic_heads > 1
        ex = (lambda t: t.unsqueeze(-1)) if ens else (lambda t: t)  # per-head returns on a trailing axis
        with torch.no_grad():
            V = self.critic(X) * ex(valid)
            R = torch.zeros_like(V)
            R[:, H] = V[:, H]
            for t in range(H - 1, -1, -1):
                r, d, m = ex(rec['r'][t]), ex(rec['d'][t]), ex(rec['m'][t])
                R[:, t] = r + cfg.gamma * (1 - d) * m * ((1 - cfg.lam) * V[:, t + 1] + cfg.lam * R[:, t + 1])
            w = torch.stack(rec['c'], dim=1) * valid[:, :H]
        Xf = X[:, :H].reshape(-1, X.shape[-1])
        Rf = R[:, :H].reshape((-1, cfg.critic_heads) if ens else (-1,))
        groups = list(self.critic.net) if ens else [self.critic.net]  # each head clipped on its own
        wf = w.reshape(-1)
        n = Xf.shape[0]
        mb = n // cfg.critic_minibatches
        tot = torch.zeros((), device=self.device)  # summed on the GPU: no host sync per minibatch (logging only)
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(cfg.critic_minibatches):
                j = perm[i * mb : (i + 1) * mb]
                err = (self.critic(Xf[j]) - Rf[j]) ** 2
                err = err.mean(-1) if ens else err
                loss = (err * wf[j]).sum() / wf[j].sum().clamp_min(1)
                self.opt_v.zero_grad(set_to_none=True)
                loss.backward()
                self.sync_grads(list(self.critic.net.parameters()))
                for g in groups:
                    zero_if_nonfinite(g.parameters(), torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0))
                self.opt_v.step()
                tot = tot + loss.detach()
        return {'v_loss': tot.item() / (cfg.critic_epochs * cfg.critic_minibatches), 'v_mean': V[:, 0].mean().item()}

    # Main loop.

    def run(self):
        cfg = self.cfg
        # warm up whenever the replay buffer is short (a fresh run, and a resumed one: the buffer is not checkpointed,
        # so `it == 0` alone crashed resumes in sample_windows; #7). Same 64 collects on a fresh run.
        while self.buf.size < cfg.warmup_steps:
            self.collect()
        t_last = time.time()
        agg = {}
        n_agg = 0
        while self.it < cfg.iterations and not self._should_stop():
            t0 = time.time()
            lr = lr_at(cfg, self.it)
            for opt in (self.opt_wm, self.opt_pi, self.opt_v):
                for g in opt.param_groups:
                    g['lr'] = lr
            tp = _Timer(self.profile)
            self.collect()
            tp('collect')
            self.update_credit += cfg.utd
            while self.update_credit >= 1.0:
                self.update_credit -= 1.0
                st = self.update_wm()
                tp('wm')
                st.update(self.update_policy())
                tp('policy+critic')
                self.updates += 1
                for kk, vv in st.items():
                    agg[kk] = agg.get(kk, 0.0) + vv
                n_agg += 1
            torch.cuda.synchronize()
            self.train_time += time.time() - t0
            self.it += 1
            if cfg.gap_every and self.it % cfg.gap_every == 0 and self.rank == 0:
                self.gap_stats = self.imagination_gap()  # outside train_time, like evaluation
            if self.profile:
                for kk, vv in tp.t.items():
                    self.prof[kk] = self.prof.get(kk, 0.0) + vv
                if self.it % cfg.log_every == 0:
                    print('profile (s per iter):', {kk: round(vv / cfg.log_every, 4) for kk, vv in self.prof.items()})
                    self.prof = {}
            if self.it % cfg.log_every == 0 and self.rank != 0:
                agg, n_agg = {}, 0
                self.ep_returns, self.ep_norm_returns, self.ep_terms, self.ep_lens, self.ep_step_rewards = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
            if self.it % cfg.log_every == 0 and self.rank == 0:
                rec = {kk: vv / max(n_agg, 1) for kk, vv in agg.items()}
                rec.setdefault('imagined_steps', self.updates * cfg.pi_batch * cfg.horizon)
                rec.update(self.gap_stats)
                self.gap_stats = {}
                rec.update(
                    it=self.it,
                    updates=self.updates,
                    env_steps=self.env_steps * self.world,
                    world=self.world,
                    hours=self.train_time / 3600,
                    lr=lr,
                    iter_s=(time.time() - t_last) / cfg.log_every,
                    ep_return=sum(self.ep_returns) / max(len(self.ep_returns), 1),
                    ep_norm_return=sum(self.ep_norm_returns) / max(len(self.ep_norm_returns), 1),
                    ep_term_frac=sum(self.ep_terms) / max(len(self.ep_terms), 1),
                    ep_len=sum(self.ep_lens) / max(len(self.ep_lens), 1),
                    ep_step_reward=sum(self.ep_step_rewards) / max(len(self.ep_step_rewards), 1),
                    bin_G_mean=self.env.bin_G[self.env.bin_visits > 0].mean().item()
                    if (self.env.bin_visits > 0).any()
                    else 0.0,
                    bins_visited=(self.env.bin_visits > 0).float().mean().item(),
                    mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                )
                self.log_f.write(json.dumps(rec) + '\n')
                self.log_f.flush()
                print(
                    f'it {self.it} upd {self.updates} h {rec["hours"]:.3f} iter {rec["iter_s"]:.3f}s '
                    f'ret {rec["ep_norm_return"]:.3f} len {rec["ep_len"]:.1f} term {rec["ep_term_frac"]:.3f} '
                    f'wm1 {rec.get("wm_err1", 0):.4f} wm8 {rec.get("wm_errk", 0):.4f} J {rec.get("pi_J", 0):.3f} '
                    f'surv {rec.get("img_survival", 0):.3f} v {rec.get("v_loss", 0):.4f} g {rec.get("pi_gnorm", 0):.3f}',
                    flush=True,
                )
                agg, n_agg = {}, 0
                self.ep_returns, self.ep_norm_returns, self.ep_terms, self.ep_lens, self.ep_step_rewards = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                t_last = time.time()
            if self.it % cfg.ckpt_every == 0 and self.rank == 0:
                self.save(os.path.join(cfg.out, f'ckpt_{self.it:07d}.pt'))
                self.save(os.path.join(cfg.out, 'last.pt'))
            if cfg.eval_every and self.it % cfg.eval_every == 0:
                if self.rank == 0:
                    self.run_eval()
                self._barrier()
        if self.rank == 0:
            self.save(os.path.join(cfg.out, 'last.pt'))
            if cfg.eval_every and self.it % cfg.eval_every != 0:
                self.run_eval()
            open(os.path.join(cfg.out, 'DONE'), 'w').close()
        self._barrier()

    def _barrier(self):
        if self.world > 1:
            import torch.distributed as dist

            dist.barrier()

    def _should_stop(self):
        stop = self.train_time / 3600 >= self.cfg.wall_hours
        if self.world > 1:
            import torch.distributed as dist

            t = torch.tensor([1.0 if stop else 0.0], device=self.device)
            dist.broadcast(t, 0)
            stop = bool(t.item() > 0)
        return stop

    def run_eval(self):
        """Benchmark protocol (not counted in train_time)."""
        from .evaluate import evaluate, make_eval_env

        cfg = self.cfg
        if not hasattr(self, 'eval_env'):
            g = torch.Generator(device='cpu').manual_seed(1234)
            if cfg.eval_clips and cfg.eval_clips < self.lib.num_clips:
                self.eval_clip_ids = torch.randperm(self.lib.num_clips, generator=g)[: cfg.eval_clips].to(self.device)
            else:
                self.eval_clip_ids = torch.arange(self.lib.num_clips, device=self.device)
            n = min(cfg.eval_envs, len(self.eval_clip_ids))
            self.eval_env = make_eval_env(self.lib, n, self.device, self.tcfg)
        t0 = time.time()
        self.policy.eval()
        res = evaluate(lambda o, g: self.policy(o, g), self.lib, self.eval_env, self.eval_clip_ids)
        self.policy.train()
        rec = {
            'it': self.it,
            'updates': self.updates,
            'hours': self.train_time / 3600,
            'env_steps': self.env_steps,
            **{k: v for k, v in res.items() if not k.startswith('per_clip')},
        }
        with open(os.path.join(cfg.out, 'eval.jsonl'), 'a') as f:
            f.write(
                json.dumps(
                    {
                        **rec,
                        'per_clip_norm_return': res['per_clip_norm_return'],
                        'per_clip_failed': res['per_clip_failed'],
                    }
                )
                + '\n'
            )
        print(
            f'EVAL it {self.it} h {rec["hours"]:.3f} norm_return {rec["norm_return"]:.4f} failed {rec["failed"]}/{rec["num_clips"]} ({time.time() - t0:.0f}s)',
            flush=True,
        )


# ---- presets
# Named flag sets for `--preset`. A preset only replaces argparse defaults, so explicit flags still override it, and
# without --preset every default is the TrainCfg (upstream) value. Not stored in the checkpoint: the cfg saved there
# holds the resolved values. The infeasible-clip whitelist (--clip-list) depends on the data and is never in a preset.
PRESETS = {
    # the README configuration: upper-body residual + absolute legs (#11, #20), latent 128 (#13), 25% noise-free
    # collection (#23), capped reset sampler (#19), a policy clip that fires (#24), WM joint error-propagation weights
    # (#21; clip at the default 4.0), CUDA graphs for imagination and WM loss (#16)
    'recommended': {'action': 'res', 'res_scale': 1.0, 'res_legs': 'abs', 'latent': 128, 'collect_det_frac': 0.25,
                    'softmax_cap': 50.0, 'pi_clip': 0.03125, 'wm_joint_weights': 'tracked', 'wm_joint_weight_clip': 4.0,
                    'compile_bptt': 'reduce-overhead', 'wm_compile_mode': 'reduce-overhead'},
}  # fmt: skip


def parse_args(argv=None) -> TrainCfg:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='', choices=['', *PRESETS], help='named flag set; explicit flags override it')
    for f in dataclasses.fields(TrainCfg):
        t = f.type if not isinstance(f.type, str) else eval(f.type)
        if t is bool:
            ap.add_argument(
                f'--{f.name.replace("_", "-")}', type=lambda x: x.lower() in ('1', 'true', 'yes'), default=f.default
            )
        else:
            ap.add_argument(f'--{f.name.replace("_", "-")}', type=t, default=f.default)
    preset = ap.parse_known_args(argv)[0].preset
    if preset:
        ap.set_defaults(**PRESETS[preset])
    args = vars(ap.parse_args(argv))
    del args['preset']
    return TrainCfg(**args)


if __name__ == '__main__':
    cfg = parse_args()
    print(cfg)
    Trainer(cfg).run()
