"""Velocity-command interface trained jointly with motion tracking (Sec. 5.5, App. B and G).

Half of the simulator environments and half of every imagined batch use the velocity
interface. A velocity encoder MLP(256, 256) maps the normalized command (vx, vy, wz) and
the 10-frame observation history to the shared 32-d latent; the shared decoder turns it
into actions. On velocity rollouts the return gradient reaches the velocity encoder only
(stop-gradient on the decoder's weights).

  mode "rl":      velocity-tracking reward (Table A.6), fall-only termination, commands from
                  the reference half the time and held uniform commands otherwise, plus a
                  0.1 latent-alignment loss toward the motion latent on reference commands.
  mode "distill": motion-tracking reward and termination, commands read from the reference.
"""

from __future__ import annotations

import dataclasses
import math
import os

import torch
import torch.nn as nn
from torch.func import functional_call

from .networks import Critic, apply_delta_c, mlp, run_mlp
from .replay import ReplayBuffer
from .rotations import heading_mat, mtv
from .sim_env import EnvCfg, TrackingSimEnv
from .task import NJ, XI_DIM, flat_to_state, state_to_flat
from .train import TrainCfg, Trainer

CMD_LO = (-1.5, -1.32, -3.35)
CMD_HI = (2.0, 1.32, 3.35)
CMD_SCALE = (2.0, 1.32, 3.35)
CMD_PERIOD = 250  # control steps between command resamples


def reference_commands(lib, window: int = 25) -> torch.Tensor:
    """Per frame: centered `window`-frame mean of (vx, vy) in the reference heading frame and
    the world yaw rate, within each clip."""
    from .rotations import quat_to_mat

    R = quat_to_mat(lib.root_quat)
    v_h = mtv(heading_mat(R), lib.root_lin_vel)
    raw = torch.cat([v_h[:, :2], lib.root_ang_vel[:, 2:3]], dim=-1)
    csum = torch.cat([torch.zeros(1, 3, device=raw.device), raw.cumsum(0)], 0)
    f = torch.arange(lib.num_frames, device=raw.device)
    clip = lib.clip_of_frame
    lo = torch.maximum(f - window // 2, lib.starts[clip])
    hi = torch.minimum(f + window // 2, lib.ends[clip] - 1)
    return (csum[hi + 1] - csum[lo]) / (hi - lo + 1).float()[:, None]


def velocity_reward(task, s, kin, cmd, action, prev_action, torso_idx):
    """Table A.6; every term times dt."""
    dt = task.cfg.dt
    H = heading_mat(s['rot'])
    v_h = mtv(H, s['v'])
    r_lin = 2.0 * torch.exp(-((v_h[:, :2] - cmd[:, :2]) ** 2).sum(-1) / 0.5**2)
    r_yaw = 2.0 * torch.exp(-((s['w'][:, 2] - cmd[:, 2]) ** 2) / 0.5)
    smooth = -0.3 * ((action - prev_action) ** 2).sum(-1)
    jl = -2.0 * (torch.relu(s['q'] - task.qmax) + torch.relu(task.qmin - s['q'])).sum(-1)
    _, Rb, _, Wb = kin
    z_torso = mtv(H, Rb[:, torso_idx, :, 2])
    lean = -4.0 * z_torso[:, 1] ** 2  # sin^2 of the sideways lean (forward lean free)
    w_torso = mtv(H, Wb[:, torso_idx])
    rate = -0.05 * (w_torso[:, 0] ** 2 + w_torso[:, 1] ** 2)
    return dt * (r_lin + r_yaw + smooth + jl + lean + rate)


def fall_termination(s):
    z = s['pos'][:, 2]
    tilt = 1 - s['rot'][:, 2, 2]  # 1 + u_z, u_z = z of gravity direction in the root frame
    hard = (z < 0.35) | (tilt > 0.7)
    soft = 1 - (1 - torch.sigmoid((0.35 - z) / 0.02)) * (1 - torch.sigmoid((tilt - 0.7) / 0.04))
    return hard, soft


def cmd_norm(cmd):
    return cmd / torch.tensor(CMD_SCALE, device=cmd.device)


class VelocityEnv(TrackingSimEnv):
    """Second half of the envs follow velocity commands; rewards/terminations per interface."""

    def __init__(self, lib, cfg: EnvCfg, task_cfg, device, gamma=0.99, mode='rl'):
        super().__init__(lib, cfg, task_cfg, device, gamma)
        n = self.num_envs
        self.mode = mode
        self.is_vel = torch.arange(n, device=device) >= n // 2
        self.lib_cmd = reference_commands(lib)
        self.cmd = torch.zeros(n, 3, device=device)
        self.cmd_ref = torch.ones(n, dtype=torch.bool, device=device)
        self.cmd_timer = torch.zeros(n, dtype=torch.long, device=device)
        self.torso_idx = self.kin.body_names.index('torso_link')
        self.fixed_cmd = None  # step test: (n, 3) commands held for every env

    def _resample_cmd(self, ids):
        k = len(ids)
        if k == 0:
            return
        if self.fixed_cmd is not None:
            self.cmd_ref[ids] = False
            self.cmd[ids] = self.fixed_cmd[ids]
        elif self.mode == 'distill':
            self.cmd_ref[ids] = True
        else:
            self.cmd_ref[ids] = torch.rand(k, device=self.device) < 0.5
            lo = torch.tensor(CMD_LO, device=self.device)
            hi = torch.tensor(CMD_HI, device=self.device)
            c = torch.rand(k, 3, device=self.device) * (hi - lo) + lo
            c[torch.rand(k, device=self.device) < 0.1] = 0.0
            self.cmd[ids] = c
        self.cmd_timer[ids] = 0
        self._refresh_ref_cmd(ids)

    def _refresh_ref_cmd(self, ids):
        r = ids[self.cmd_ref[ids]]
        if len(r):
            self.cmd[r] = self.lib_cmd[self.frame[r]]

    def reset_envs(self, env_ids, frames=None):
        super().reset_envs(env_ids, frames)
        self._resample_cmd(env_ids)

    def policy_inputs(self):
        obs, goal = super().policy_inputs()
        return obs, goal, cmd_norm(self.cmd)

    def step(self, action):
        """Same as TrackingSimEnv.step, with the velocity interface's reward and termination
        for velocity envs in RL mode."""
        q_sp = self.action_scale * action + self.q0
        self.sim.data.ctrl[:] = (q_sp - self.enc_bias)[:, self.act_to_joint]
        for _ in range(self.decimation):
            self.sim.step()
        s_next = self.read_state()
        frames_next = self.frame + 1
        ref = self.lib.frame(frames_next)
        ref_kin = self.kin_c(ref['pos'], ref['rot'], ref['q'], ref['v'], ref['w'], ref['qd'])
        out = self.task.analytics_fast(
            s_next, ref, ref_kin, self.ref_future(frames_next), action, self.prev_action, self.enc_bias
        )
        r, hard = out['r'], out['hard']
        if self.mode == 'rl':
            rv = velocity_reward(self.task, s_next, out['kin'], self.cmd, action, self.prev_action, self.torso_idx)
            hv, _ = fall_termination(s_next)
            r = torch.where(self.is_vel, rv, r)
            hard = torch.where(self.is_vel, hv, hard)
        self.ep_len += 1
        clip_end = self.lib.remaining(frames_next) <= 0
        timeout = self.ep_len >= self.max_ep_len
        truncated = (clip_end | timeout) & ~hard
        done = hard | truncated
        disc = self.gamma ** (self.ep_len - 1).float()
        self.ep_disc_ret += disc * r
        self.ep_ret += r
        info = {
            'reward': r,
            'terminated': hard,
            'truncated': truncated,
            'frame': self.frame.clone(),
            'cmd': self.cmd.clone(),
            'cmd_ref': self.cmd_ref.clone(),
        }
        self.frame = frames_next
        self.prev_action = action.clone()
        o = out['obs'] + self._noise(self.num_envs)
        self.obs_hist = torch.cat([self.obs_hist[:, 1:], o.unsqueeze(1)], dim=1)
        self.goal_buf = out['goal']
        # Commands for the next step.
        self.cmd_timer += 1
        self._refresh_ref_cmd(torch.arange(self.num_envs, device=self.device))
        rs = ((self.cmd_timer >= CMD_PERIOD) & ~done).nonzero().squeeze(-1)
        self._resample_cmd(rs)
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids):
            mot = done_ids[~self.is_vel[done_ids]]
            if len(mot):
                ratio = self.ep_disc_ret[mot] / self._max_disc_return(mot)
                b = self.ep_bin[mot]
                a = self.cfg.bin_ema
                new = torch.where(self.bin_visits[b] > 0, (1 - a) * self.bin_G[b] + a * ratio, ratio)
                self.bin_G[b] = new.clamp(0, 1)
                self.bin_visits[b] += 1
            start = self.frame[done_ids] - self.ep_len[done_ids]
            potential = torch.minimum(self.lib.remaining(start), torch.full_like(start, self.max_ep_len)).float()
            info['done_ids'] = done_ids
            info['done_norm_return'] = self.ep_ret[done_ids] / (self.tcfg.max_step_reward * potential.clamp_min(1))
            info['done_ep_return'] = self.ep_ret[done_ids].clone()
            info['done_ep_len'] = self.ep_len[done_ids].clone()
            info['done_terminated'] = hard[done_ids].clone()
            info['done_is_vel'] = self.is_vel[done_ids].clone()
            self.reset_envs(done_ids)
        self._finish_resets(done_ids)
        info['done'] = done
        return s_next, info


class VelReplay(ReplayBuffer):
    def __init__(self, num_envs, capacity, xi_dim, device):
        super().__init__(num_envs, capacity, xi_dim, device)
        self.cmd = torch.zeros(capacity, num_envs, 3, device=device)
        self.cmd_ref = torch.zeros(capacity, num_envs, dtype=torch.bool, device=device)

    def add_cmd(self, cmd, cmd_ref):
        p = (self.ptr - 1) % self.L  # call right after add()
        self.cmd[p], self.cmd_ref[p] = cmd, cmd_ref

    def sample_starts_from(self, batch, history, task, env_lo, env_hi):
        """sample_starts restricted to envs [env_lo, env_hi), also returning the command."""
        env = torch.randint(env_lo, env_hi, (batch,), device=self.device)
        return self._sample_starts_env(env, history, task)

    def _sample_starts_env(self, env, history, task):
        batch = env.shape[0]
        age = torch.randint(0, self.size, (batch,), device=self.device)
        t = self._age_index(age)
        s0, xi, frame, ep0 = self.s[t, env], self.xi[t, env], self.frame[t, env], self.ep[t, env]
        back = torch.arange(history - 1, -1, -1, device=self.device)
        ages = (age[:, None] + back).clamp(max=self.size - 1)
        idx = self._age_index(ages)
        e = env[:, None].expand(-1, history)
        same = (self.ep[idx, e] == ep0[:, None]) & ((age[:, None] + back) < self.size)
        first = same.float().argmax(-1)
        idx = torch.where(same, idx, idx.gather(1, first[:, None]).expand(-1, history))
        s_h = self.s[idx, e]
        prev_idx = (idx - 1) % self.L
        prev_ok = (self.ep[prev_idx, e] == ep0[:, None]) & (idx != self._age_index(torch.full_like(idx, self.size - 1)))
        a_prev = torch.where(prev_ok[..., None], self.a[prev_idx, e], torch.zeros_like(self.a[prev_idx, e]))
        st = flat_to_state(s_h)
        bias = xi[:, None, 1 : 1 + NJ]
        obs_hist = task.obs_clean(st, a_prev, bias.expand(-1, history, -1))
        return s0, xi, frame, obs_hist, a_prev[:, -1], self.cmd[t, env], self.cmd_ref[t, env]


@dataclasses.dataclass
class VelCfg(TrainCfg):
    vel_mode: str = 'rl'
    align_weight: float = 0.1


class VelocityTrainer(Trainer):
    """Trainer with a second goal interface. Replaces the env, buffer and policy update."""

    def __init__(self, cfg: VelCfg, device='cuda'):
        self._vel_mode = cfg.vel_mode
        resume, cfg.resume = cfg.resume, ''
        super().__init__(cfg, device)
        obs_dim = self.obs_dim
        self.vel_enc = mlp(3 + obs_dim, (256, 256), 32).to(device)
        self.opt_vel = torch.optim.Adam(self.vel_enc.parameters(), lr=cfg.lr, fused=True)
        self.critic_vel = Critic(self.critic_in + 3).to(device)
        self.opt_vcrit = torch.optim.Adam(self.critic_vel.net.parameters(), lr=cfg.lr, fused=True)
        self.vel_returns, self.vel_falls = [], []
        if resume:
            cfg.resume = resume
            self.load(resume)

    def _make_env(self):
        return VelocityEnv(
            self.lib,
            EnvCfg(num_envs=self.cfg.num_envs),
            self.tcfg,
            self.device,
            gamma=self.cfg.gamma,
            mode=self._vel_mode,
        )

    def _make_buffer(self):
        return VelReplay(self.cfg.num_envs, self.cfg.buffer_steps, XI_DIM, self.device)

    # Velocity latent (shared decoder applied by the caller).
    def vel_latent(self, obs, cmdn):
        pol = self.policy
        return torch.tanh(run_mlp(self.vel_enc, torch.cat([cmdn, pol.obs_norm(obs)], dim=-1)))

    def decode(self, z, obs, frozen=False):
        pol = self.policy
        x = torch.cat([z, pol.obs_norm(obs)], dim=-1)
        if frozen:
            params = {k: v.detach() for k, v in pol.decoder.named_parameters()}
            from .networks import amp_ctx

            with amp_ctx():
                y = functional_call(pol.decoder, params, (x,))
            return y.float()
        return run_mlp(pol.decoder, x)

    @torch.no_grad()
    def collect(self, random=False):
        env = self.env
        obs, goal, cmdn = env.policy_inputs()
        self.policy.obs_norm.update(obs)
        self.policy.goal_norm.update(goal[~env.is_vel])
        a_mot = self.policy(obs, goal)
        a_vel = self.decode(self.vel_latent(obs, cmdn), obs)
        a = torch.where(env.is_vel[:, None], a_vel, a_mot)
        a = a + self.cfg.action_noise * torch.randn_like(a)
        s_flat = state_to_flat(env.state)
        xi, frame, ep = env.xi.clone(), env.frame.clone(), env.ep_id.clone()
        cmd, cref = env.cmd.clone(), env.cmd_ref.clone()
        s_next, info = env.step(a)
        self.buf.add(s_flat, a, state_to_flat(s_next), xi, frame, ep)
        self.buf.add_cmd(cmd, cref)
        self.env_steps += env.num_envs
        if 'done_ids' in info:
            mot = ~info['done_is_vel']
            if mot.any():
                self.ep_norm_returns.append(info['done_norm_return'][mot].mean().item())
                self.ep_terms.append(info['done_terminated'][mot].float().mean().item())
                self.ep_lens.append(info['done_ep_len'][mot].float().mean().item())
            if (~mot).any():
                L = info['done_ep_len'][~mot].float()
                self.vel_returns.append((info['done_ep_return'][~mot] / L).mean().item())
                self.vel_falls.append(info['done_terminated'][~mot].float().mean().item())

    def update_policy(self):
        cfg = self.cfg
        n = cfg.num_envs
        half = cfg.pi_batch // 2
        st = self._rollout(half, 0, n // 2, velocity=False)
        sv = self._rollout(half, n // 2, n, velocity=True)
        st.update({f'vel_{k}': v for k, v in sv.items()})
        if self.vel_returns:
            st['vel_ep_step_return'] = sum(self.vel_returns) / len(self.vel_returns)
            st['vel_ep_fall_frac'] = sum(self.vel_falls) / len(self.vel_falls)
            self.vel_returns, self.vel_falls = [], []
        return st

    def _rollout(self, B, env_lo, env_hi, velocity):
        cfg, task, lib, wm, env = self.cfg, self.task, self.lib, self.wm, self.env
        H, K = cfg.horizon, self.tcfg.future
        dev = self.device
        s0, xi, frame, obs_hist, a_prev, cmd0, cref = self.buf.sample_starts_from(
            B, self.tcfg.history, task, env_lo, env_hi
        )
        bias = xi[:, 1 : 1 + NJ]
        rl = velocity and env.mode == 'rl'
        with torch.no_grad():
            steps = torch.arange(0, H + K + 1, device=dev)
            idx_raw = frame[:, None] + steps
            idx = lib.clamp_to_clip(idx_raw, frame)
            valid = (idx_raw == idx)[:, : H + 1].float()
            ref = lib.frame(idx)
            ref_kin = env.kin_c(ref['pos'], ref['rot'], ref['q'], ref['v'], ref['w'], ref['qd'])
            rem_all = lib.remaining(idx[:, : H + 1]).float().clamp(max=100) / 100.0
            ref_cmd = env.lib_cmd[idx[:, : H + 1]]
            cmds = torch.where(cref[:, None, None], ref_cmd, cmd0[:, None, :])  # (B, H+1, 3)
            if rl:
                # Held commands do not depend on the reference: the clip end ends nothing.
                valid = torch.where(cref[:, None], valid, torch.ones_like(valid))

        at = lambda t: {k: v[:, t] for k, v in ref.items()}
        fut = lambda t: {k: v[:, t + 1 : t + 1 + K] for k, v in ref.items()}
        kat = lambda t: tuple(x[:, t] for x in ref_kin)
        critic = self.critic_vel if velocity else self.critic
        for p in wm.net.parameters():
            p.requires_grad_(False)
        s = flat_to_state(s0)
        hist = obs_hist + task.sample_obs_noise((B, self.tcfg.history))
        an = task.analytics_fast(s, at(0), kat(0), fut(0), a_prev, a_prev, bias)
        c = torch.ones(B, device=dev)
        J = torch.zeros(B, device=dev)
        align = torch.zeros((), device=dev)
        rec = {'x': [], 'r': [], 'd': [], 'm': [], 'c': []}
        std, mean = wm.out_norm.std, wm.out_norm.mean

        def cin(feats, goal, t):
            x = self.critic_input(feats, goal, xi, rem_all[:, t])
            return torch.cat([x, cmd_norm(cmds[:, t])], dim=-1) if velocity else x

        for t in range(H):
            goal, feats = an['goal'], an['feats']
            rec['x'].append(cin(feats, goal, t).detach())
            obs = hist.flatten(1)
            if velocity:
                z = self.vel_latent(obs, cmd_norm(cmds[:, t]))
                a = self.decode(z, obs, frozen=True)
                if rl and cfg.align_weight > 0 and cref.any():
                    with torch.no_grad():
                        zm = self.policy.encode(goal)
                    align = (
                        align + (((z - zm) ** 2).mean(-1) * cref.float()).sum() / cref.float().sum().clamp_min(1) / H
                    )
            else:
                a = self.policy(obs, goal)
            dlt = run_mlp(wm.net, wm.in_norm(torch.cat([feats, a, xi], dim=-1))) * std + mean
            s1 = apply_delta_c(s, dlt)
            an = task.analytics_fast(s1, at(t + 1), kat(t + 1), fut(t + 1), a, a_prev, bias)
            if rl:
                r = velocity_reward(task, s1, an['kin'], cmds[:, t], a, a_prev, env.torso_idx)
                _, d = fall_termination(s1)
            else:
                r, d = an['r'], an['soft']
            m = valid[:, t + 1]
            J = J + (cfg.gamma**t) * c * r * m
            rec['r'].append((r * m).detach())
            rec['d'].append(d.detach())
            rec['m'].append(m)
            rec['c'].append(c.detach())
            c = c * (1 - d)
            hist = torch.cat([hist[:, 1:], (an['obs'] + task.sample_obs_noise((B,))).unsqueeze(1)], dim=1)
            s, a_prev = s1, a
        xH = cin(an['feats'], an['goal'], H)
        J = J + (cfg.gamma**H) * c * critic(xH) * valid[:, H]
        loss = -J.mean() / H + cfg.align_weight * align
        opt = self.opt_vel if velocity else self.opt_pi
        opt.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for g in opt.param_groups for p in g['params']]
        self.sync_grads(params)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        torch.nn.utils.clip_grad_value_(params, 0.1)
        opt.step()
        for p in wm.net.parameters():
            p.requires_grad_(True)
        critic.zero_grad(set_to_none=True)
        self.policy.zero_grad(set_to_none=True)
        st = {'pi_J': J.mean().item(), 'pi_gnorm': gn.item(), 'img_survival': c.mean().item(), 'align': float(align)}
        st.update(self._fit_critic(critic, self.opt_vcrit if velocity else self.opt_v, rec, xH.detach(), valid))
        return st

    def _fit_critic(self, critic, opt, rec, xH, valid):
        cfg, H = self.cfg, self.cfg.horizon
        X = torch.stack(rec['x'] + [xH], dim=1)
        critic.norm.update(X)
        with torch.no_grad():
            V = critic(X) * valid
            R = torch.zeros_like(V)
            R[:, H] = V[:, H]
            for t in range(H - 1, -1, -1):
                r, d, m = rec['r'][t], rec['d'][t], rec['m'][t]
                R[:, t] = r + cfg.gamma * (1 - d) * m * ((1 - cfg.lam) * V[:, t + 1] + cfg.lam * R[:, t + 1])
            w = torch.stack(rec['c'], dim=1) * valid[:, :H]
        Xf, Rf, wf = X[:, :H].reshape(-1, X.shape[-1]), R[:, :H].reshape(-1), w.reshape(-1)
        n = Xf.shape[0]
        mb = n // cfg.critic_minibatches
        tot = 0.0
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(cfg.critic_minibatches):
                j = perm[i * mb : (i + 1) * mb]
                loss = (((critic(Xf[j]) - Rf[j]) ** 2) * wf[j]).sum() / wf[j].sum().clamp_min(1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                self.sync_grads(list(critic.net.parameters()))
                torch.nn.utils.clip_grad_norm_(critic.net.parameters(), 1.0)
                opt.step()
                tot += loss.item()
        return {'v_loss': tot / (cfg.critic_epochs * cfg.critic_minibatches)}

    def _extra_state(self):
        return {
            'vel_enc': self.vel_enc.state_dict(),
            'critic_vel': self.critic_vel.state_dict(),
            'opt_vel': self.opt_vel.state_dict(),
            'opt_vcrit': self.opt_vcrit.state_dict(),
            'vel_mode': self._vel_mode,
        }

    def _load_extra(self, ck):
        if 'vel_enc' in ck and hasattr(self, 'vel_enc'):
            self.vel_enc.load_state_dict(ck['vel_enc'])
            self.critic_vel.load_state_dict(ck['critic_vel'])
            self.opt_vel.load_state_dict(ck['opt_vel'])
            self.opt_vcrit.load_state_dict(ck['opt_vcrit'])

    def run_eval(self):
        """Motion-tracking benchmark on the motion interface (unchanged protocol)."""
        super().run_eval()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(VelCfg):
        t = f.type if not isinstance(f.type, str) else eval(f.type)
        if t is bool:
            ap.add_argument(
                f'--{f.name.replace("_", "-")}', type=lambda x: x.lower() in ('1', 'true', 'yes'), default=f.default
            )
        else:
            ap.add_argument(f'--{f.name.replace("_", "-")}', type=t, default=f.default)
    cfg = VelCfg(**vars(ap.parse_args()))
    print(cfg)
    VelocityTrainer(cfg).run()


if __name__ == '__main__':
    main()
