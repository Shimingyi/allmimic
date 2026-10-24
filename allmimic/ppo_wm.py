"""PPO-WM (Fig. 4b): PPO on world-model rollouts instead of BPTT.

Same loop as SuperMimic (simulator collection, replay, world-model fit, UTD); only the
policy update changes: a Gaussian policy is rolled out for `wm_rollout_steps` steps in
`ppo_rollouts` parallel imagined trajectories (no gradient through the model), then
updated with BeyondMimic's PPO hyperparameters. Progress is measured in imagined steps.
"""

from __future__ import annotations

import dataclasses

import torch

from .ppo import PPOActor
from .task import NJ
from .train import TrainCfg, Trainer, parse_args as _unused  # noqa: F401
from .task import flat_to_state


@dataclasses.dataclass
class PPOWMCfg(TrainCfg):
    ppo_rollouts: int = 16384
    ppo_steps: int = 32
    ppo_lr: float = 1e-3
    ppo_desired_kl: float = 0.01
    ppo_clip: float = 0.2
    ppo_entropy: float = 0.005
    ppo_epochs: int = 5
    ppo_minibatches: int = 4
    ppo_init_std: float = 1.0


class PPOWMTrainer(Trainer):
    def __init__(self, cfg: PPOWMCfg, device='cuda'):
        super().__init__(cfg, device)
        self.actor = PPOActor(self.obs_dim, self.goal_dim, cfg.ppo_init_std).to(device)
        # The policy used for collection is the actor's mean network plus its noise.
        self.policy = self.actor.mean
        params = (
            list(self.actor.mean.encoder.parameters()) + list(self.actor.mean.decoder.parameters()) + [self.actor.std]
        )
        self.ppo_params = params + list(self.critic.net.parameters())
        self.opt_ppo = torch.optim.Adam(self.ppo_params, lr=cfg.ppo_lr, fused=True)
        self.ppo_lr = cfg.ppo_lr
        self.imagined_steps = 0
        self.task.compile_sizes |= {cfg.ppo_rollouts}

    @torch.no_grad()
    def collect(self, random=False):
        env = self.env
        obs, goal = env.policy_inputs()
        self.actor.mean.obs_norm.update(obs)
        self.actor.mean.goal_norm.update(goal)
        a = self.actor.dist(obs, goal).sample()
        from .task import state_to_flat

        s_flat = state_to_flat(env.state)
        xi, frame, ep = env.xi.clone(), env.frame.clone(), env.ep_id.clone()
        s_next, info = env.step(a)
        self.buf.add(s_flat, a, state_to_flat(s_next), xi, frame, ep)
        self.env_steps += env.num_envs
        if 'done_ep_return' in info:
            self.ep_norm_returns.append(info['done_norm_return'].mean().item())
            self.ep_terms.append(info['done_terminated'].float().mean().item())
            self.ep_lens.append(info['done_ep_len'].float().mean().item())

    @torch.no_grad()
    def imagine(self):
        cfg, task, lib, wm = self.cfg, self.task, self.lib, self.wm
        B, T, K = cfg.ppo_rollouts, cfg.ppo_steps, self.tcfg.future
        dev = self.device
        s0, xi, frame, obs_hist, a_prev = self.buf.sample_starts(B, self.tcfg.history, task)
        bias = xi[:, 1 : 1 + NJ]
        idx_raw = frame[:, None] + torch.arange(0, T + K + 1, device=dev)
        idx = lib.clamp_to_clip(idx_raw, frame)
        valid = (idx_raw == idx)[:, : T + 1].float()
        ref = lib.frame(idx)
        ref_kin = self.env.kin_c(ref['pos'], ref['rot'], ref['q'], ref['v'], ref['w'], ref['qd'])
        rem = lib.remaining(idx[:, : T + 1]).float().clamp(max=100) / 100.0
        at = lambda t: {k: v[:, t] for k, v in ref.items()}
        fut = lambda t: {k: v[:, t + 1 : t + 1 + K] for k, v in ref.items()}
        kat = lambda t: tuple(x[:, t] for x in ref_kin)
        s = flat_to_state(s0)
        hist = obs_hist + task.sample_obs_noise((B, self.tcfg.history))
        an = task.analytics_fast(s, at(0), kat(0), fut(0), a_prev, a_prev, bias)
        data = {k: [] for k in ('obs', 'goal', 'cin', 'a', 'logp', 'mu', 'v', 'r', 'cont')}
        std, mean = wm.out_norm.std, wm.out_norm.mean
        from .networks import apply_delta_c, run_mlp

        alive = torch.ones(B, device=dev)
        for t in range(T):
            obs_f = hist.flatten(1)
            cin = self.critic_input(an['feats'], an['goal'], xi, rem[:, t])
            d = self.actor.dist(obs_f, an['goal'])
            a = d.sample()
            dlt = run_mlp(wm.net, wm.in_norm(torch.cat([an['feats'], a, xi], dim=-1))) * std + mean
            s1 = apply_delta_c(s, dlt)
            an = task.analytics_fast(s1, at(t + 1), kat(t + 1), fut(t + 1), a, a_prev, bias)
            m = valid[:, t + 1]
            data['obs'].append(obs_f)
            data['goal'].append(cin[:, self.feat_dim : self.feat_dim + self.goal_dim])
            data['cin'].append(cin)
            data['a'].append(a)
            data['logp'].append(d.log_prob(a).sum(-1))
            data['mu'].append(d.mean)
            data['v'].append(self.critic(cin))
            data['r'].append(an['r'] * m * alive)
            # Expected continuation (soft termination) and clip end.
            cont = (1 - an['soft']) * m
            data['cont'].append(cont)
            alive = alive * (cont > 0.5).float()
            o1 = an['obs'] + task.sample_obs_noise((B,))
            hist = torch.cat([hist[:, 1:], o1.unsqueeze(1)], dim=1)
            s, a_prev = s1, a
        cin = self.critic_input(an['feats'], an['goal'], xi, rem[:, T])
        last_v = self.critic(cin) * valid[:, T]
        self.imagined_steps += B * T
        return {k: torch.stack(v) for k, v in data.items()}, last_v, alive

    def update_policy(self):
        cfg = self.cfg
        data, last_v, alive = self.imagine()
        T = cfg.ppo_steps
        adv = torch.zeros_like(data['r'])
        gae = torch.zeros_like(last_v)
        for t in range(T - 1, -1, -1):
            nv = last_v if t == T - 1 else data['v'][t + 1]
            c = data['cont'][t]
            delta = data['r'][t] + cfg.gamma * c * nv - data['v'][t]
            gae = delta + cfg.gamma * cfg.lam * c * gae
            adv[t] = gae
        ret = adv + data['v']
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in data.items()}
        adv_f, ret_f = adv.reshape(-1), ret.reshape(-1)
        n = adv_f.shape[0]
        mb = n // cfg.ppo_minibatches
        std_old = self.actor.std.detach().clamp_min(1e-3)
        self.critic.norm.update(flat['cin'])
        st = {'ppo_kl': 0.0, 'ppo_v_loss': 0.0}
        cnt = 0
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(cfg.ppo_minibatches):
                j = perm[i * mb : (i + 1) * mb]
                d = self.actor.dist(flat['obs'][j], flat['goal'][j])
                with torch.no_grad():
                    sn = d.stddev
                    kl = (
                        (torch.log(sn / std_old) + (std_old**2 + (flat['mu'][j] - d.mean) ** 2) / (2 * sn**2) - 0.5)
                        .sum(-1)
                        .mean()
                        .item()
                    )
                if kl > 2 * cfg.ppo_desired_kl:
                    self.ppo_lr = max(1e-5, self.ppo_lr / 1.5)
                elif 0 < kl < cfg.ppo_desired_kl / 2:
                    self.ppo_lr = min(1e-2, self.ppo_lr * 1.5)
                for g in self.opt_ppo.param_groups:
                    g['lr'] = self.ppo_lr
                logp = d.log_prob(flat['a'][j]).sum(-1)
                A = adv_f[j]
                A = (A - A.mean()) / (A.std() + 1e-8)
                ratio = torch.exp(logp - flat['logp'][j])
                pi_loss = -torch.min(ratio * A, ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * A).mean()
                v = self.critic(flat['cin'][j])
                v_old = flat['v'][j]
                v_clip = v_old + (v - v_old).clamp(-cfg.ppo_clip, cfg.ppo_clip)
                v_loss = torch.max((v - ret_f[j]) ** 2, (v_clip - ret_f[j]) ** 2).mean()
                loss = pi_loss + v_loss - cfg.ppo_entropy * d.entropy().sum(-1).mean()
                self.opt_ppo.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ppo_params, 1.0)
                self.opt_ppo.step()
                st['ppo_kl'] += kl
                st['ppo_v_loss'] += v_loss.item()
                cnt += 1
        st = {k: v / cnt for k, v in st.items()}
        st.update(
            imagined_steps=self.imagined_steps,
            ppo_lr=self.ppo_lr,
            ppo_std=self.actor.std.mean().item(),
            img_survival=alive.mean().item(),
        )
        return st


def main():
    import argparse

    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(PPOWMCfg):
        t = f.type if not isinstance(f.type, str) else eval(f.type)
        if t is bool:
            ap.add_argument(
                f'--{f.name.replace("_", "-")}', type=lambda x: x.lower() in ('1', 'true', 'yes'), default=f.default
            )
        else:
            ap.add_argument(f'--{f.name.replace("_", "-")}', type=t, default=f.default)
    cfg = PPOWMCfg(**vars(ap.parse_args()))
    print(cfg)
    PPOWMTrainer(cfg).run()


if __name__ == '__main__':
    main()
