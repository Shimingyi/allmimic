"""PPO-Sim baseline: PPO in the simulator with SuperMimic's motions, observations,
reward, termination, domain randomization, actor/critic sizes and evaluation, and
BeyondMimic's PPO hyperparameters (8192 envs per GPU, 24 steps per env per iteration).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time

import torch
import torch.nn as nn

from . import networks
from .motion_lib import MotionLibrary
from .networks import Critic, Policy
from .sim_env import EnvCfg, TrackingSimEnv
from .task import GOAL_FRAME_DIM, NJ, OBS_DIM, XI_DIM, TaskCfg


@dataclasses.dataclass
class PPOCfg:
    motions: str = 'data/processed/lafan1_50hz.npz'
    clip_prefixes: str = ''
    out: str = 'runs/ppo_debug'
    seed: int = 0
    num_envs: int = 8192
    steps_per_env: int = 24
    iterations: int = 30_000
    lr: float = 1e-3
    desired_kl: float = 0.01
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.005
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    max_grad_norm: float = 1.0
    init_std: float = 1.0
    log_every: int = 10
    ckpt_every: int = 250
    eval_every_hours: float = 0.25
    eval_clips: int = 0
    eval_envs: int = 4096
    wall_hours: float = 18.0
    amp: bool = True
    resume: str = ''


class PPOActor(nn.Module):
    def __init__(self, obs_dim, goal_dim, init_std):
        super().__init__()
        self.mean = Policy(obs_dim, goal_dim)
        self.std = nn.Parameter(torch.full((NJ,), init_std))

    def dist(self, obs, goal):
        mu = self.mean(obs, goal)
        return torch.distributions.Normal(mu, self.std.clamp_min(1e-3).expand_as(mu))


def critic_input(task, env):
    s = env.state
    feats = task.features_fast(s)
    rem = env.lib.remaining(env.frame).float().clamp(max=100) / 100.0
    return torch.cat([feats, env.goal_buf, env.xi, rem[:, None]], dim=-1)


class PPOTrainer:
    def __init__(self, cfg: PPOCfg, device='cuda'):
        self.cfg, self.device = cfg, device
        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        networks.AMP['enabled'] = cfg.amp
        self.lib = MotionLibrary(cfg.motions, device, prefixes=cfg.clip_prefixes)
        self.tcfg = TaskCfg()
        self.env = TrackingSimEnv(self.lib, EnvCfg(num_envs=cfg.num_envs), self.tcfg, device, gamma=cfg.gamma)
        self.task = self.env.task
        self.env.reset()
        feat_dim = self.task.features_fast(self.env.state).shape[-1]
        self.obs_dim = OBS_DIM * self.tcfg.history
        self.goal_dim = GOAL_FRAME_DIM * self.tcfg.future
        self.actor = PPOActor(self.obs_dim, self.goal_dim, cfg.init_std).to(device)
        self.critic = Critic(feat_dim + self.goal_dim + XI_DIM + 1).to(device)
        params = (
            list(self.actor.mean.encoder.parameters()) + list(self.actor.mean.decoder.parameters()) + [self.actor.std]
        )
        params += list(self.critic.net.parameters())
        self.params = params
        self.opt = torch.optim.Adam(params, lr=cfg.lr, fused=True)
        self.lr = cfg.lr
        self.next_eval_h = cfg.eval_every_hours
        self.it = 0
        self.env_steps = 0
        self.train_time = 0.0
        os.makedirs(cfg.out, exist_ok=True)
        self.log_f = open(os.path.join(cfg.out, 'log.jsonl'), 'a')
        if cfg.resume:
            ck = torch.load(cfg.resume, map_location=device)
            self.actor.load_state_dict(ck['actor'])
            self.critic.load_state_dict(ck['critic'])
            self.opt.load_state_dict(ck['opt'])
            self.it, self.env_steps, self.train_time, self.lr = ck['it'], ck['env_steps'], ck['train_time'], ck['lr']
            self.env.bin_G[:] = ck['bin_G']
            self.env.bin_visits[:] = ck['bin_visits']
            self.env.bin_past_w[:] = ck['bin_past_w']

    def save(self, path):
        torch.save(
            {
                'cfg': dataclasses.asdict(self.cfg),
                'it': self.it,
                'env_steps': self.env_steps,
                'train_time': self.train_time,
                'lr': self.lr,
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'opt': self.opt.state_dict(),
                'bin_G': self.env.bin_G,
                'bin_visits': self.env.bin_visits,
                'bin_past_w': self.env.bin_past_w,
            },
            path + '.tmp',
        )
        os.replace(path + '.tmp', path)

    @torch.no_grad()
    def rollout(self):
        cfg, env = self.cfg, self.env
        T, N = cfg.steps_per_env, env.num_envs
        buf = {k: [] for k in ('obs', 'goal', 'cin', 'a', 'logp', 'mu', 'v', 'r', 'done', 'boot')}
        stats = {'ret': [], 'len': [], 'term': []}
        for _ in range(T):
            obs, goal = env.policy_inputs()
            self.actor.mean.obs_norm.update(obs)
            self.actor.mean.goal_norm.update(goal)
            cin = critic_input(self.task, env)
            self.critic.norm.update(cin)
            d = self.actor.dist(obs, goal)
            a = d.sample()
            logp = d.log_prob(a).sum(-1)
            v = self.critic(cin)
            s_next, info = env.step(a)
            r = info['reward'].clone()
            # Bootstrap time-outs (not clip ends: the motion is over, no future reward).
            timeout = info['truncated'] & (env.lib.remaining(info['frame'] + 1) > 0)
            boot = torch.zeros_like(r)
            if timeout.any():
                # Value of the pre-reset successor; its goal/obs use the next frames.
                ids = timeout.nonzero().squeeze(-1)
                boot[ids] = self._value_of(s_next, info['frame'] + 1, ids)
            for k, x in (
                ('obs', obs),
                ('goal', goal),
                ('cin', cin),
                ('a', a),
                ('logp', logp),
                ('mu', d.mean),
                ('v', v),
                ('r', r),
            ):
                buf[k].append(x)
            buf['done'].append(info['done'].float())
            buf['boot'].append(boot)
            if 'done_ep_return' in info:
                stats['ret'].append(info['done_norm_return'].mean().item())
                stats['term'].append(info['done_terminated'].float().mean().item())
        last_v = self.critic(critic_input(self.task, env))
        out = {k: torch.stack(v) for k, v in buf.items()}
        self.env_steps += T * N
        return out, last_v, stats

    def _value_of(self, s_next, frames, ids):
        env, task = self.env, self.task
        s = {k: v[ids] for k, v in s_next.items()}
        feats = task.state_features(s)
        goal = task.goal(s, env.ref_future(frames[ids]))
        rem = env.lib.remaining(frames[ids]).float().clamp(max=100) / 100.0
        x = torch.cat([feats, goal, env.xi[ids], rem[:, None]], dim=-1)
        return self.critic(x)

    def update(self, data, last_v):
        cfg = self.cfg
        T = cfg.steps_per_env
        r = data['r'] + cfg.gamma * data['boot']
        adv = torch.zeros_like(r)
        gae = torch.zeros_like(last_v)
        for t in range(T - 1, -1, -1):
            nv = last_v if t == T - 1 else data['v'][t + 1]
            nonterm = 1.0 - data['done'][t]
            delta = r[t] + cfg.gamma * nv * nonterm - data['v'][t]
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[t] = gae
        ret = adv + data['v']
        flat = {k: v.reshape(T * v.shape[1], *v.shape[2:]) for k, v in data.items()}
        adv_f, ret_f = adv.reshape(-1), ret.reshape(-1)
        n = adv_f.shape[0]
        mb = n // cfg.minibatches
        st = {'pi_loss': 0.0, 'v_loss': 0.0, 'kl': 0.0, 'entropy': 0.0}
        cnt = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(cfg.minibatches):
                j = perm[i * mb : (i + 1) * mb]
                d = self.actor.dist(flat['obs'][j], flat['goal'][j])
                with torch.no_grad():
                    # rsl_rl adaptive schedule: exact KL(old || new) of the diagonal Gaussians.
                    sig_new = d.stddev
                    sig_old = self.std_old.expand_as(sig_new)
                    kl = (
                        (
                            torch.log(sig_new / sig_old)
                            + (sig_old**2 + (flat['mu'][j] - d.mean) ** 2) / (2 * sig_new**2)
                            - 0.5
                        )
                        .sum(-1)
                        .mean()
                        .item()
                    )
                if kl > 2 * cfg.desired_kl:
                    self.lr = max(1e-5, self.lr / 1.5)
                elif 0 < kl < cfg.desired_kl / 2:
                    self.lr = min(1e-2, self.lr * 1.5)
                for g in self.opt.param_groups:
                    g['lr'] = self.lr
                logp = d.log_prob(flat['a'][j]).sum(-1)
                A = adv_f[j]
                A = (A - A.mean()) / (A.std() + 1e-8)
                ratio = torch.exp(logp - flat['logp'][j])
                pi_loss = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
                v = self.critic(flat['cin'][j])
                v_old = flat['v'][j]
                v_clip = v_old + (v - v_old).clamp(-cfg.clip, cfg.clip)
                v_loss = torch.max((v - ret_f[j]) ** 2, (v_clip - ret_f[j]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.opt.step()
                kl_true = kl
                st['pi_loss'] += pi_loss.item()
                st['v_loss'] += v_loss.item()
                st['kl'] += kl_true
                st['entropy'] += ent.item()
                cnt += 1
        return {k: v / cnt for k, v in st.items()}

    def run(self):
        cfg = self.cfg
        while self.it < cfg.iterations and self.train_time / 3600 < cfg.wall_hours:
            t0 = time.time()
            data, last_v, stats = self.rollout()
            self.std_old = self.actor.std.detach().clamp_min(1e-3).clone()
            st = self.update(data, last_v)
            torch.cuda.synchronize()
            self.train_time += time.time() - t0
            self.it += 1
            if self.it % cfg.log_every == 0:
                rec = dict(
                    it=self.it,
                    hours=self.train_time / 3600,
                    env_steps=self.env_steps,
                    lr=self.lr,
                    iter_s=time.time() - t0,
                    ep_norm_return=sum(stats['ret']) / max(len(stats['ret']), 1),
                    ep_term_frac=sum(stats['term']) / max(len(stats['term']), 1),
                    std=self.actor.std.mean().item(),
                    **st,
                )
                self.log_f.write(json.dumps(rec) + '\n')
                self.log_f.flush()
                print(
                    f'it {self.it} h {rec["hours"]:.3f} iter {rec["iter_s"]:.2f}s ret {rec["ep_norm_return"]:.3f} '
                    f'term {rec["ep_term_frac"]:.3f} kl {st["kl"]:.4f} lr {self.lr:.2e} std {rec["std"]:.3f} v {st["v_loss"]:.4f}',
                    flush=True,
                )
            if self.it % cfg.ckpt_every == 0:
                self.save(os.path.join(cfg.out, f'ckpt_{self.it:07d}.pt'))
                self.save(os.path.join(cfg.out, 'last.pt'))
            if self.train_time / 3600 >= self.next_eval_h:
                self.run_eval()
                self.next_eval_h += cfg.eval_every_hours
        self.save(os.path.join(cfg.out, 'last.pt'))
        self.run_eval()
        open(os.path.join(cfg.out, 'DONE'), 'w').close()

    def run_eval(self):
        from .evaluate import evaluate, make_eval_env

        cfg = self.cfg
        if not hasattr(self, 'eval_env'):
            g = torch.Generator(device='cpu').manual_seed(1234)
            if cfg.eval_clips and cfg.eval_clips < self.lib.num_clips:
                self.eval_clip_ids = torch.randperm(self.lib.num_clips, generator=g)[: cfg.eval_clips].to(self.device)
            else:
                self.eval_clip_ids = torch.arange(self.lib.num_clips, device=self.device)
            self.eval_env = make_eval_env(self.lib, min(cfg.eval_envs, len(self.eval_clip_ids)), self.device)
        t0 = time.time()
        res = evaluate(lambda o, g: self.actor.mean(o, g), self.lib, self.eval_env, self.eval_clip_ids)
        rec = {
            'it': self.it,
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


def load_ppo_policy(path, device):
    ck = torch.load(path, map_location=device)
    tc = TaskCfg()
    actor = PPOActor(OBS_DIM * tc.history, GOAL_FRAME_DIM * tc.future, 1.0).to(device)
    actor.load_state_dict(ck['actor'])
    actor.eval()
    meta = {'it': ck['it'], 'hours': ck['train_time'] / 3600, 'env_steps': ck['env_steps']}
    return (lambda o, g: actor.mean(o, g)), meta


def parse_args() -> PPOCfg:
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(PPOCfg):
        t = f.type if not isinstance(f.type, str) else eval(f.type)
        if t is bool:
            ap.add_argument(
                f'--{f.name.replace("_", "-")}', type=lambda x: x.lower() in ('1', 'true', 'yes'), default=f.default
            )
        else:
            ap.add_argument(f'--{f.name.replace("_", "-")}', type=t, default=f.default)
    return PPOCfg(**vars(ap.parse_args()))


if __name__ == '__main__':
    cfg = parse_args()
    print(cfg)
    PPOTrainer(cfg).run()
