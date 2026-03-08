import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import pickle
import hashlib

from pose_format import Pose


class SignLanguagePoseDataset(Dataset):
    """Dataset for sign language pose sequences."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        chunk_len: int,
        dtype=np.float32,
        history_len: int = 10,
        limited_num: int = -1,
        use_cache: bool = True,
        cache_dir: Path = None,
        force_reload: bool = False,
        min_condition_length: int = 0,
        fixed_condition_length: int = -1,
    ):

        self.data_dir = data_dir
        self.split = split
        self.chunk_len = chunk_len
        self.history_len = history_len
        self.window_len = chunk_len + history_len
        self.dtype = dtype

        self.use_cache = use_cache
        self.force_reload = force_reload

        self.min_condition_length = min_condition_length
        self.fixed_condition_length = fixed_condition_length

        if cache_dir is None:
            self.cache_dir = self.data_dir / "cache"
        else:
            self.cache_dir = cache_dir

        self.cache_dir.mkdir(exist_ok=True)

        split_dir = self.data_dir / self.split

        all_files = (
            list(split_dir.glob(f"{split}_*_original.pose"))
            + list(split_dir.glob(f"{split}_*_updated.pose"))
            + list(split_dir.glob(f"{split}_*_metadata.json"))
        )

        mtimes = [f.stat().st_mtime for f in all_files if f.exists()]
        data_mtime = max(mtimes) if mtimes else 0

        cache_params = {
            "data_dir": str(data_dir),
            "split": split,
            "chunk_len": chunk_len,
            "history_len": history_len,
            "dtype": str(dtype),
            "limited_num": limited_num,
            "data_mtime": data_mtime,
            "min_condition_length": self.min_condition_length,
            "fixed_condition_length": self.fixed_condition_length,
        }

        cache_key = hashlib.md5(str(cache_params).encode()).hexdigest()

        self.cache_file = self.cache_dir / f"dataset_cache_{split}_{cache_key}.pkl"

        if self.use_cache and not self.force_reload and self.cache_file.exists():

            print(f"Loading dataset from cache: {self.cache_file}")

            self._load_from_cache()

            print(f"Dataset loaded from cache: {len(self.examples)} samples, split={split}")

            return

        self.examples = []

        split_dir = self.data_dir / split

        fluent_files = sorted(list(split_dir.glob(f"{split}_*_original.pose")))

        if limited_num > 0:
            fluent_files = fluent_files[:limited_num]

        for fluent_file in tqdm(fluent_files, desc=f"Loading {split} examples"):

            disfluent_file = fluent_file.with_name(
                fluent_file.name.replace("_original.pose", "_updated.pose")
            )

            metadata_file = fluent_file.with_name(
                fluent_file.name.replace("_original.pose", "_metadata.json")
            )

            if self.min_condition_length > 0:

                with open(metadata_file, "r", encoding="utf-8") as f:

                    metadata = json.load(f)

                disfluent_len = metadata.get("disfluent_pose_length", 0)

                if disfluent_len < self.min_condition_length:

                    continue

            self.examples.append(
                {
                    "fluent_path": fluent_file,
                    "disfluent_path": disfluent_file,
                    "metadata_path": metadata_file,
                }
            )

        if self.examples:

            first_fluent_path = self.examples[0]["fluent_path"]

            try:

                with open(first_fluent_path, "rb") as f:

                    first_pose = Pose.read(f)

                    self.pose_header = first_pose.header

            except Exception as e:

                print(f"[WARNING] Failed to read pose_header from {first_fluent_path}: {e}")

                self.pose_header = None

        else:

            self.pose_header = None

        self.fluent_clip_list = []
        self.fluent_mask_list = []
        self.disfluent_clip_list = []

        self.train_indices = []

        for example_idx, example in enumerate(
            tqdm(self.examples, desc=f"Processing pose files for {split}", total=len(self.examples))
        ):

            with open(example["fluent_path"], "rb") as f:
                fluent_pose = Pose.read(f)

            with open(example["disfluent_path"], "rb") as f:
                disfluent_pose = Pose.read(f)

            fluent_data = np.array(fluent_pose.body.data.astype(self.dtype))
            fluent_mask = fluent_pose.body.data.mask

            disfluent_data = np.array(disfluent_pose.body.data.astype(self.dtype))

            self.fluent_clip_list.append(fluent_data[:, 0])
            self.fluent_mask_list.append(fluent_mask[:, 0])

            disfluent_seq = disfluent_data[:, 0]

            if self.fixed_condition_length > 0:

                current_len = disfluent_seq.shape[0]

                target_len = self.fixed_condition_length

                if current_len > target_len:

                    disfluent_seq = disfluent_seq[:target_len]

                elif current_len < target_len:

                    padding_len = target_len - current_len

                    k_dim = disfluent_seq.shape[1]
                    d_dim = disfluent_seq.shape[2]

                    padding = np.zeros((padding_len, k_dim, d_dim), dtype=self.dtype)

                    disfluent_seq = np.concatenate([disfluent_seq, padding], axis=0)

            self.disfluent_clip_list.append(disfluent_seq)

        concatenated_fluent_clips = np.concatenate(self.fluent_clip_list, axis=0)

        self.input_mean = concatenated_fluent_clips.mean(axis=0, keepdims=True)

        self.input_std = concatenated_fluent_clips.std(axis=0, keepdims=True)

        concatenated_disfluent_clips = np.concatenate(self.disfluent_clip_list, axis=0)

        self.condition_mean = concatenated_disfluent_clips.mean(axis=0, keepdims=True)

        self.condition_std = concatenated_disfluent_clips.std(axis=0, keepdims=True)

        # FIX division by zero
        self.input_std = np.where(self.input_std == 0, 1e-6, self.input_std)
        self.condition_std = np.where(self.condition_std == 0, 1e-6, self.condition_std)

        for i in range(len(self.examples)):

            self.fluent_clip_list[i] = (
                self.fluent_clip_list[i] - self.input_mean
            ) / self.input_std

            self.disfluent_clip_list[i] = (
                self.disfluent_clip_list[i] - self.condition_mean
            ) / self.condition_std

        if self.use_cache:

            print(f"Saving dataset to cache: {self.cache_file}")

            self._save_to_cache()

        print("Dataset initialized with {} samples. Split: {}".format(len(self.examples), split))

    def _save_to_cache(self):

        data = {
            "examples": self.examples,
            "pose_header": self.pose_header,
            "fluent_clip_list": self.fluent_clip_list,
            "fluent_mask_list": self.fluent_mask_list,
            "disfluent_clip_list": self.disfluent_clip_list,
            "train_indices": self.train_indices,
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "condition_mean": self.condition_mean,
            "condition_std": self.condition_std,
        }

        with open(self.cache_file, "wb") as f:

            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_from_cache(self):

        with open(self.cache_file, "rb") as f:

            data = pickle.load(f)

        self.examples = data["examples"]
        self.pose_header = data["pose_header"]
        self.fluent_clip_list = data["fluent_clip_list"]
        self.fluent_mask_list = data["fluent_mask_list"]
        self.disfluent_clip_list = data["disfluent_clip_list"]
        self.train_indices = data["train_indices"]
        self.input_mean = data["input_mean"]
        self.input_std = data["input_std"]
        self.condition_mean = data["condition_mean"]
        self.condition_std = data["condition_std"]

    def __len__(self):

        return len(self.train_indices)

    def __getitem__(self, idx):

        motion_idx = self.train_indices[idx][0]

        disfluent_seq = torch.from_numpy(
            self.disfluent_clip_list[motion_idx].astype(np.float32)
        )

        full_seq = torch.from_numpy(
            self.fluent_clip_list[motion_idx].astype(np.float32)
        )

        history_len = self.history_len

        num_keypoints = full_seq.shape[1]
        num_dims = full_seq.shape[2]

        previous_output = torch.zeros((history_len, num_keypoints, num_dims))

        return {
            "data": full_seq,
            "conditions": {
                "input_sequence": disfluent_seq,
                "previous_output": previous_output,
            },
        }