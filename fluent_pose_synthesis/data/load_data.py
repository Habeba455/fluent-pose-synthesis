import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from pose_format import Pose


class SignLanguagePoseDataset(Dataset):
    """
    Sign-language pose dataset.

    Assumes disfluent files (`*{updated_suffix}`) are already v6-cleaned —
    i.e. they have the SAME length as the corresponding fluent (`*_original.pose`)
    file. Pairs whose lengths don't match are skipped with a warning (they
    indicate an un-processed video).

    Output of __getitem__:
        {
            "data":        [T_chunk, K, D]     target fluent chunk (normalized)
            "conditions": {
                "input_sequence":  [T_window, K, D]  disfluent (normalized)
                "previous_output": [T_hist, K, D]    fluent history (normalized)
            }
            "motion_idx":  int
            "start":       int
            "end":         int
        }

    Important:
        - Stride controls how much consecutive windows overlap. stride=1
          generates a sample per frame and causes severe sample duplication —
          avoid. Default: chunk_len // 2 (50% overlap).
        - Validation / test datasets MUST use training-set statistics.
          Build the training dataset first, then pass its stats to val/test
          via `set_normalization_stats(...)` BEFORE accessing items.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        chunk_len: int,
        dtype=np.float32,
        history_len: int = 10,
        limited_num: int = -1,
        updated_suffix: str = "_updated_v7_cleaned.pose",
        stride: Optional[int] = None,
        compute_stats: bool = True,
        external_stats: Optional[dict] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.chunk_len = chunk_len
        self.history_len = history_len
        self.window_len = chunk_len + history_len
        self.dtype = dtype
        self.updated_suffix = updated_suffix

        # ★ FIX #2: stride defaults to 50% overlap instead of 1-frame overlap.
        # This drops ~chunk_len× the number of training samples without losing
        # information the model can actually learn from.
        self.stride = stride if stride is not None else max(1, chunk_len // 2)

        self.pose_header = None
        self.keypoints = None
        self.dims = None

        split_dir = self.data_dir / split
        self.examples = []

        fluent_files = sorted(list(split_dir.glob(f"{split}_*_original.pose")))
        if limited_num > 0:
            fluent_files = fluent_files[:limited_num]

        print(f"Found {len(fluent_files)} original pose files in {split_dir}")

        skipped_no_disfluent = 0
        skipped_no_metadata = 0
        for fluent_file in tqdm(fluent_files, desc=f"Loading {split} examples"):
            disfluent_file = fluent_file.with_name(
                fluent_file.name.replace("_original.pose", updated_suffix)
            )
            metadata_file = fluent_file.with_name(
                fluent_file.name.replace("_original.pose", "_metadata.json")
            )

            if not disfluent_file.exists():
                skipped_no_disfluent += 1
                continue
            if not metadata_file.exists():
                skipped_no_metadata += 1
                continue

            self.examples.append(
                {
                    "fluent_path": fluent_file,
                    "disfluent_path": disfluent_file,
                    "metadata_path": metadata_file,
                }
            )

        if skipped_no_disfluent:
            print(f"  Skipped {skipped_no_disfluent} without disfluent counterpart")
        if skipped_no_metadata:
            print(f"  Skipped {skipped_no_metadata} without metadata")
        print(f"Valid examples: {len(self.examples)}")

        # ★ Keep raw (pre-normalization) arrays so we can re-normalize with
        # external stats later if needed.
        self.fluent_clip_list = []
        self.disfluent_clip_list = []
        self.metadata_list = []

        skipped_length_mismatch = 0
        skipped_bad_shape = 0
        skipped_other = 0

        for example in tqdm(self.examples, desc="Processing pose files"):
            try:
                with open(example["fluent_path"], "rb") as f:
                    fluent_pose = Pose.read(f.read())

                with open(example["disfluent_path"], "rb") as f:
                    disfluent_pose = Pose.read(f.read())

                with open(example["metadata_path"], "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                if self.pose_header is None:
                    self.pose_header = fluent_pose.header

                fluent_data = np.asarray(fluent_pose.body.data.astype(self.dtype))
                disfluent_data = np.asarray(disfluent_pose.body.data.astype(self.dtype))

                fluent_seq = fluent_data[:, 0]
                disfluent_seq = disfluent_data[:, 0]

                if fluent_seq.ndim != 3 or disfluent_seq.ndim != 3:
                    skipped_bad_shape += 1
                    continue

                tf, kf, df = fluent_seq.shape
                td, kd, dd = disfluent_seq.shape

                if kf != kd or df != dd:
                    skipped_bad_shape += 1
                    continue

                # ★ FIX #1: require that v6 cleaning was applied — fluent and
                # disfluent must have the SAME length. No resampling here,
                # because uniform resampling mis-aligns the stitched padding
                # present in raw updated files.
                if tf != td:
                    skipped_length_mismatch += 1
                    continue

                if self.keypoints is None:
                    self.keypoints = kf
                    self.dims = df

                self.fluent_clip_list.append(fluent_seq.astype(self.dtype))
                self.disfluent_clip_list.append(disfluent_seq.astype(self.dtype))
                self.metadata_list.append(metadata)

            except Exception as e:                        # noqa: BLE001
                print(f"Skipping {example['fluent_path'].name}: {e}")
                skipped_other += 1
                continue

        if skipped_length_mismatch:
            print(
                f"  Skipped {skipped_length_mismatch} pairs with length mismatch "
                "(disfluent not v6-cleaned?). Re-run v6 on those before training."
            )
        if skipped_bad_shape:
            print(f"  Skipped {skipped_bad_shape} with bad shape")
        if skipped_other:
            print(f"  Skipped {skipped_other} other errors")

        if len(self.fluent_clip_list) == 0:
            raise RuntimeError("No valid pose sequences found")

        # ★ FIX #3: compute normalization statistics only if requested.
        # Validation/test datasets should pass external_stats so they use
        # training-set statistics (no data leakage).
        if external_stats is not None:
            self.set_normalization_stats(
                input_mean=external_stats["input_mean"],
                input_std=external_stats["input_std"],
                condition_mean=external_stats["condition_mean"],
                condition_std=external_stats["condition_std"],
            )
        elif compute_stats:
            self._compute_and_apply_own_stats()
        else:
            # Stats will be set later via set_normalization_stats()
            self.input_mean = None
            self.input_std = None
            self.condition_mean = None
            self.condition_std = None
            self._normalized = False

        # ★ FIX #2: windowing with stride
        self.train_indices = []
        for motion_idx in range(len(self.fluent_clip_list)):
            seq_len = self.fluent_clip_list[motion_idx].shape[0]
            if seq_len < self.window_len:
                continue
            for start in range(0, seq_len - self.window_len + 1, self.stride):
                self.train_indices.append((motion_idx, start))

        # ★ FIX #6: explicit error if no windows were produced
        if len(self.train_indices) == 0:
            shortest = min(c.shape[0] for c in self.fluent_clip_list)
            longest = max(c.shape[0] for c in self.fluent_clip_list)
            raise RuntimeError(
                f"No training windows produced. window_len={self.window_len} "
                f"(chunk={self.chunk_len} + history={self.history_len}), "
                f"sequence lengths range {shortest}-{longest}. "
                f"Reduce chunk_len/history_len, or check your data."
            )

        print(
            f"Total training windows: {len(self.train_indices)} "
            f"(stride={self.stride}, window_len={self.window_len})"
        )

    # -----------------------------------------------------------------
    # Normalization helpers
    # -----------------------------------------------------------------
    def _compute_and_apply_own_stats(self):
        """Compute input_mean/std and condition_mean/std from this dataset."""
        concatenated_fluent = np.concatenate(self.fluent_clip_list, axis=0)
        self.input_mean = concatenated_fluent.mean(axis=0, keepdims=True)
        self.input_std = concatenated_fluent.std(axis=0, keepdims=True)

        concatenated_disfluent = np.concatenate(self.disfluent_clip_list, axis=0)
        self.condition_mean = concatenated_disfluent.mean(axis=0, keepdims=True)
        self.condition_std = concatenated_disfluent.std(axis=0, keepdims=True)

        self.input_std = np.where(self.input_std == 0, 1e-6, self.input_std)
        self.condition_std = np.where(self.condition_std == 0, 1e-6, self.condition_std)

        self._apply_normalization()

    def _apply_normalization(self):
        """Apply stored mean/std in-place to all stored sequences."""
        for i in range(len(self.fluent_clip_list)):
            self.fluent_clip_list[i] = (
                (self.fluent_clip_list[i] - self.input_mean) / self.input_std
            ).astype(self.dtype)

            self.disfluent_clip_list[i] = (
                (self.disfluent_clip_list[i] - self.condition_mean) / self.condition_std
            ).astype(self.dtype)
        self._normalized = True

    def set_normalization_stats(
        self,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        condition_mean: np.ndarray,
        condition_std: np.ndarray,
    ):
        """
        Apply external (e.g. training-set) normalization statistics.
        Call this on val/test datasets BEFORE iterating.
        """
        self.input_mean = np.asarray(input_mean, dtype=self.dtype)
        self.input_std = np.asarray(input_std, dtype=self.dtype)
        self.condition_mean = np.asarray(condition_mean, dtype=self.dtype)
        self.condition_std = np.asarray(condition_std, dtype=self.dtype)

        self.input_std = np.where(self.input_std == 0, 1e-6, self.input_std)
        self.condition_std = np.where(self.condition_std == 0, 1e-6, self.condition_std)

        self._apply_normalization()

    def get_stats_dict(self) -> dict:
        """Return current normalization stats for passing to another dataset."""
        return {
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "condition_mean": self.condition_mean,
            "condition_std": self.condition_std,
        }

    # -----------------------------------------------------------------
    # Dataset API
    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.train_indices)

    def __getitem__(self, idx):
        motion_idx, start = self.train_indices[idx]
        end = start + self.window_len

        full_seq = self.fluent_clip_list[motion_idx][start:end]
        disfluent_seq = self.disfluent_clip_list[motion_idx][start:end]

        full_seq = torch.from_numpy(full_seq.astype(np.float32))
        disfluent_seq = torch.from_numpy(disfluent_seq.astype(np.float32))

        previous_output = full_seq[: self.history_len]
        target_seq = full_seq[self.history_len:]

        return {
            "data": target_seq,
            "conditions": {
                "input_sequence": disfluent_seq,
                "previous_output": previous_output,
            },
            "metadata": self.metadata_list[motion_idx],
            "motion_idx": motion_idx,
            "start": start,
            "end": end,
        }
