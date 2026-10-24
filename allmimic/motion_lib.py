"""Motion library: all clips concatenated on the GPU, indexed by global frame.

Two storage formats, chosen by the file:
  upstream npz      qpos (N,36) / qvel (N,35) kept as float32 tensors -- bit-identical to dhmbb2.
  compact_v1 npz    root_pos (N,3) float32 + rest (N,68) float16 = [quat wxyz 4, q 29, root lin vel 3 (world),
                    root ang vel 3 (world), qd 29]; written by tools/build_allmimic_library.py.
                    ~148 B/frame instead of ~300 B, so the 240 h BONES-SEED train split (43 M frames) is ~6.4 GB.
Root position stays float32: at ~10 m from the origin float16 resolves only ~8 mm.

There are no per-frame index tables (they were int64 x 2 = 16 B/frame): clip_of() is a searchsorted over clip
starts and bin_of() is arithmetic within the clip (CHANGES #6; equal to the old tables, checked
in tests/test_motion_lib.py). frame() always returns float32.
"""

from __future__ import annotations

import numpy as np
import torch

from .rotations import quat_normalize, quat_to_mat

COMPACT = 'compact_v1'
REST = {'quat': (0, 4), 'q': (4, 33), 'v': (33, 36), 'w': (36, 39), 'qd': (39, 68)}  # column ranges of `rest`


class MotionLibrary:
    def __init__(self, path: str, device: str, bin_seconds: float = 1.0, clip_ids=None, prefixes: str = '', keep=None):
        d = np.load(path, allow_pickle=False)
        self.compact = 'storage' in d.files and str(d['storage']) == COMPACT
        starts = d['starts'].astype(np.int64)
        lengths = d['lengths'].astype(np.int64)
        names = d['names']
        if prefixes:
            pre = tuple(p for p in prefixes.split(',') if p)
            clip_ids = [i for i, n in enumerate(names) if str(n).split('/')[-1].startswith(pre)]
        if keep is not None:  # clip-name whitelist (e.g. tools/filter_infeasible.py <dataset>_feasible.txt)
            clip_ids = [
                i for i, n in enumerate(names) if str(n).split('/')[-1] in keep and (clip_ids is None or i in clip_ids)
            ]
        cols = ('root_pos', 'rest') if self.compact else ('qpos', 'qvel')
        a, b = d[cols[0]], d[cols[1]]
        if clip_ids is not None:
            clip_ids = np.asarray(clip_ids)
            sel = np.concatenate([np.arange(starts[c], starts[c] + lengths[c]) for c in clip_ids])
            a, b = a[sel], b[sel]
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
        if self.compact:
            self.root_pos = torch.tensor(a, dtype=torch.float32, device=device)
            rest = torch.tensor(b, dtype=torch.float16, device=device)
            # views into one float16 tensor; frame() casts to float32 (and renormalizes the quaternion)
            self.root_quat, self.q, self.root_lin_vel, self.root_ang_vel, self.qd = (
                rest[:, lo:hi] for lo, hi in REST.values()
            )
        else:
            self.root_pos = torch.tensor(a[:, :3], dtype=torch.float32, device=device)
            self.root_quat = quat_normalize(torch.tensor(a[:, 3:7], dtype=torch.float32, device=device))
            self.q = torch.tensor(a[:, 7:], dtype=torch.float32, device=device)
            self.root_lin_vel = torch.tensor(b[:, :3], dtype=torch.float32, device=device)
            self.root_ang_vel = torch.tensor(b[:, 3:6], dtype=torch.float32, device=device)
            self.qd = torch.tensor(b[:, 6:], dtype=torch.float32, device=device)
        # 1 s bins over the whole library for return-based reset sampling.
        self.bin_len = int(round(bin_seconds * self.fps))
        nb = ((lengths + self.bin_len - 1) // self.bin_len).astype(np.int64)
        bin_clip = np.repeat(np.arange(self.num_clips), nb)
        bin_k = np.concatenate([np.arange(n) for n in nb])
        bin_start = starts[bin_clip] + bin_k * self.bin_len
        bin_end = np.minimum(bin_start + self.bin_len, starts[bin_clip] + lengths[bin_clip])
        self.num_bins = len(bin_clip)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.bin_end = torch.tensor(bin_end, device=device)
        self.clip_bin0 = torch.tensor(np.concatenate([[0], np.cumsum(nb)[:-1]]), device=device)  # first bin per clip

    def clip_of(self, idx: torch.Tensor) -> torch.Tensor:
        """global frame indices (any shape) -> clip indices (same shape)"""
        return torch.searchsorted(self.starts, idx.contiguous(), right=True) - 1

    def bin_of(self, idx: torch.Tensor) -> torch.Tensor:
        """global frame indices (any shape) -> 1 s bin indices (same shape)"""
        clip = self.clip_of(idx)
        return self.clip_bin0[clip] + torch.div(idx - self.starts[clip], self.bin_len, rounding_mode='floor')

    @property
    def clip_of_frame(self) -> torch.Tensor:
        """(num_frames,) clip index of every frame, built on demand (velocity / dev scripts on small libraries)"""
        return torch.repeat_interleave(torch.arange(self.num_clips, device=self.device), self.lengths)

    @property
    def bin_of_frame(self) -> torch.Tensor:
        """(num_frames,) bin index of every frame, built on demand"""
        return self.bin_of(torch.arange(self.num_frames, device=self.device))

    def frame(self, idx: torch.Tensor) -> dict:
        """Reference quantities at global frame indices (any shape), float32."""
        quat = self.root_quat[idx].float()
        if self.compact:
            quat = quat_normalize(quat)
        return {
            'pos': self.root_pos[idx],
            'rot': quat_to_mat(quat),
            'quat': quat,
            'q': self.q[idx].float(),
            'v': self.root_lin_vel[idx].float(),
            'w': self.root_ang_vel[idx].float(),
            'qd': self.qd[idx].float(),
        }

    def clamp_to_clip(self, idx: torch.Tensor, ref_idx: torch.Tensor) -> torch.Tensor:
        """Clamp frame indices to the clip of ref_idx (holds the last frame)."""
        end = self.ends[self.clip_of(ref_idx)] - 1
        while end.dim() < idx.dim():
            end = end.unsqueeze(-1)
        return torch.minimum(idx, end)

    def remaining(self, idx: torch.Tensor) -> torch.Tensor:
        """Number of frames after idx in its clip (0 on the last frame)."""
        return self.ends[self.clip_of(idx)] - 1 - idx
