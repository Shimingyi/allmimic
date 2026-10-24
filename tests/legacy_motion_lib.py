"""Frozen copy of allmimic/motion_lib.py before allmimic #6 (upstream dhmbb2 behavior), for equivalence tests only.

Do not edit: tests/test_motion_lib.py checks the new library against this file bit for bit.
"""

from __future__ import annotations

import numpy as np
import torch

from allmimic.rotations import quat_normalize, quat_to_mat


class MotionLibrary:
    def __init__(self, path: str, device: str, bin_seconds: float = 1.0, clip_ids=None, prefixes: str = ''):
        d = np.load(path, allow_pickle=False)
        starts = d['starts'].astype(np.int64)
        lengths = d['lengths'].astype(np.int64)
        names = d['names']
        if prefixes:
            pre = tuple(p for p in prefixes.split(',') if p)
            clip_ids = [i for i, n in enumerate(names) if str(n).split('/')[-1].startswith(pre)]
        qpos = d['qpos']
        qvel = d['qvel']
        if clip_ids is not None:
            clip_ids = np.asarray(clip_ids)
            sel = np.concatenate([np.arange(starts[c], starts[c] + lengths[c]) for c in clip_ids])
            qpos, qvel = qpos[sel], qvel[sel]
            lengths = lengths[clip_ids]
            names = names[clip_ids]
            starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        self.fps = float(d['fps'])
        self.dt = 1.0 / self.fps
        self.device = device
        self.names = [str(n) for n in names]
        self.num_clips = len(lengths)
        self.num_frames = int(lengths.sum())
        self.hours = self.num_frames / self.fps / 3600
        self.starts = torch.tensor(starts, device=device)
        self.lengths = torch.tensor(lengths, device=device)
        self.ends = self.starts + self.lengths  # exclusive
        self.root_pos = torch.tensor(qpos[:, :3], dtype=torch.float32, device=device)
        self.root_quat = quat_normalize(torch.tensor(qpos[:, 3:7], dtype=torch.float32, device=device))
        self.q = torch.tensor(qpos[:, 7:], dtype=torch.float32, device=device)
        self.root_lin_vel = torch.tensor(qvel[:, :3], dtype=torch.float32, device=device)
        self.root_ang_vel = torch.tensor(qvel[:, 3:6], dtype=torch.float32, device=device)
        self.qd = torch.tensor(qvel[:, 6:], dtype=torch.float32, device=device)
        self.clip_of_frame = torch.repeat_interleave(torch.arange(self.num_clips, device=device), self.lengths)
        # 1 s bins over the whole library for return-based reset sampling.
        bin_len = int(round(bin_seconds * self.fps))
        nb = ((lengths + bin_len - 1) // bin_len).astype(np.int64)
        bin_clip = np.repeat(np.arange(self.num_clips), nb)
        bin_k = np.concatenate([np.arange(n) for n in nb])
        bin_start = starts[bin_clip] + bin_k * bin_len
        bin_end = np.minimum(bin_start + bin_len, starts[bin_clip] + lengths[bin_clip])
        self.num_bins = len(bin_clip)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.bin_end = torch.tensor(bin_end, device=device)
        self.bin_of_frame = torch.empty(self.num_frames, dtype=torch.long, device=device)
        for_b = torch.repeat_interleave(torch.arange(self.num_bins, device=device), self.bin_end - self.bin_start)
        self.bin_of_frame[:] = for_b

    def frame(self, idx: torch.Tensor) -> dict:
        """Reference quantities at global frame indices (any shape)."""
        return {
            'pos': self.root_pos[idx],
            'rot': quat_to_mat(self.root_quat[idx]),
            'quat': self.root_quat[idx],
            'q': self.q[idx],
            'v': self.root_lin_vel[idx],
            'w': self.root_ang_vel[idx],
            'qd': self.qd[idx],
        }

    def clamp_to_clip(self, idx: torch.Tensor, ref_idx: torch.Tensor) -> torch.Tensor:
        """Clamp frame indices to the clip of ref_idx (holds the last frame)."""
        clip = self.clip_of_frame[ref_idx]
        end = self.ends[clip] - 1
        while end.dim() < idx.dim():
            end = end.unsqueeze(-1)
        return torch.minimum(idx, end)

    def remaining(self, idx: torch.Tensor) -> torch.Tensor:
        """Number of frames after idx in its clip (0 on the last frame)."""
        return self.ends[self.clip_of_frame[idx]] - 1 - idx
