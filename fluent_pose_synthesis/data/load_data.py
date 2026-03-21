import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

from pose_format import Pose


def resample_sequence(seq, target_len):
    t = seq.shape[0]
    if t == target_len:
        return seq.copy()

    old_idx = np.linspace(0, t - 1, t)
    new_idx = np.linspace(0, t - 1, target_len)

    _, k, d = seq.shape
    out = np.zeros((target_len, k, d), dtype=seq.dtype)

    for ki in range(k):
        for di in range(d):
            out[:, ki, di] = np.interp(new_idx, old_idx, seq[:, ki, di])

    return out


class SignLanguagePoseDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        chunk_len: int,
        dtype=np.float32,
        history_len: int = 10,
        limited_num: int = -1,
        min_condition_length: int = 0,
        fixed_condition_length: int = -1,
    ):
        self.data_dir = data_dir
        self.split = split
        self.chunk_len = chunk_len
        self.history_len = history_len
        self.window_len = chunk_len + history_len
        self.dtype = dtype

        self.pose_header = None
        self.keypoints = None
        self.dims = None

        split_dir = self.data_dir / split
        self.examples = []

        fluent_files = sorted(list(split_dir.glob(f"{split}_*_original.pose")))
        if limited_num > 0:
            fluent_files = fluent_files[:limited_num]

        print(f"Found {len(fluent_files)} pose files")

        for fluent_file in tqdm(fluent_files, desc=f"Loading {split} examples"):
            disfluent_file = fluent_file.with_name(
                fluent_file.name.replace("_original.pose", "_updated.pose")
            )
            metadata_file = fluent_file.with_name(
                fluent_file.name.replace("_original.pose", "_metadata.json")
            )

            if not disfluent_file.exists():
                continue

            if not metadata_file.exists():
                continue

            self.examples.append(
                {
                    "fluent_path": fluent_file,
                    "disfluent_path": disfluent_file,
                    "metadata_path": metadata_file,
                }
            )

        print(f"Valid examples: {len(self.examples)}")

        self.fluent_clip_list = []
        self.disfluent_clip_list = []

        for example in tqdm(self.examples, desc="Processing pose files"):
            try:
                with open(example["fluent_path"], "rb") as f:
                    fluent_pose = Pose.read(f.read())

                with open(example["disfluent_path"], "rb") as f:
                    disfluent_pose = Pose.read(f.read())

                if self.pose_header is None:
                    self.pose_header = fluent_pose.header

                fluent_data = np.asarray(fluent_pose.body.data.astype(self.dtype))
                disfluent_data = np.asarray(disfluent_pose.body.data.astype(self.dtype))

                fluent_seq = fluent_data[:, 0]
                disfluent_seq = disfluent_data[:, 0]

                if fluent_seq.ndim != 3 or disfluent_seq.ndim != 3:
                    continue

                tf, kf, df = fluent_seq.shape
                td, kd, dd = disfluent_seq.shape

                if kf != kd or df != dd:
                    continue

                if self.keypoints is None:
                    self.keypoints = kf
                    self.dims = df

                disfluent_seq = resample_sequence(disfluent_seq, tf)

                self.fluent_clip_list.append(fluent_seq.astype(self.dtype))
                self.disfluent_clip_list.append(disfluent_seq.astype(self.dtype))

            except Exception:
                continue

        if len(self.fluent_clip_list) == 0:
            raise RuntimeError("No valid pose sequences found")

        concatenated_fluent = np.concatenate(self.fluent_clip_list, axis=0)
        self.input_mean = concatenated_fluent.mean(axis=0, keepdims=True)
        self.input_std = concatenated_fluent.std(axis=0, keepdims=True)

        concatenated_disfluent = np.concatenate(self.disfluent_clip_list, axis=0)
        self.condition_mean = concatenated_disfluent.mean(axis=0, keepdims=True)
        self.condition_std = concatenated_disfluent.std(axis=0, keepdims=True)

        self.input_std = np.where(self.input_std == 0, 1e-6, self.input_std)
        self.condition_std = np.where(self.condition_std == 0, 1e-6, self.condition_std)

        for i in range(len(self.fluent_clip_list)):
            self.fluent_clip_list[i] = (
                self.fluent_clip_list[i] - self.input_mean
            ) / self.input_std

            self.disfluent_clip_list[i] = (
                self.disfluent_clip_list[i] - self.condition_mean
            ) / self.condition_std

        self.train_indices = []

        for motion_idx in range(len(self.fluent_clip_list)):
            seq_len = self.fluent_clip_list[motion_idx].shape[0]

            if seq_len < self.window_len:
                continue

            for start in range(seq_len - self.window_len + 1):
                self.train_indices.append((motion_idx, start))

        print("Total training samples:", len(self.train_indices))

    def __len__(self):
        return len(self.train_indices)

    def __getitem__(self, idx):
        motion_idx, start = self.train_indices[idx]
        end = start + self.window_len

        full_seq = self.fluent_clip_list[motion_idx][start:end]
        disfluent_seq = self.disfluent_clip_list[motion_idx][start:end]

        full_seq = torch.from_numpy(full_seq.astype(np.float32))
        disfluent_seq = torch.from_numpy(disfluent_seq.astype(np.float32))

        history_len = self.history_len

        previous_output = full_seq[:history_len]
        target_seq = full_seq[history_len:]

        return {
            "data": target_seq,
            "conditions": {
                "input_sequence": disfluent_seq,
                "previous_output": previous_output,
            },
        }
