# pylint: disable=protected-access, arguments-renamed
import itertools
import time
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
import torch.nn as nn
from tqdm import tqdm

from pose_format import Pose
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.numpy.pose_body import NumPyPoseBody

try:
    from pose_evaluation.metrics.distance_metric import DistanceMetric
    from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
    from pose_evaluation.metrics.pose_processors import NormalizePosesProcessor
    HAS_POSE_EVAL = True
except ModuleNotFoundError:
    HAS_POSE_EVAL = False
    DistanceMetric = None
    DTWDTAIImplementationDistanceMeasure = None
    NormalizePosesProcessor = None

from CAMDM.diffusion.gaussian_diffusion import GaussianDiffusion
from CAMDM.network.training import BaseTrainingPortal
from CAMDM.utils.common import mkdir


class _ConditionalWrapper(nn.Module):
    """Wrap a model with a fixed conditioning dict, exposing forward(x, t)."""

    def __init__(self, base_model: nn.Module, cond: dict):
        super().__init__()
        self.base_model = base_model
        self.cond = cond

    def forward(self, x, t, **kwargs):
        return self.base_model.interface(x, t, self.cond)


def move_to_device(val, device):
    if torch.is_tensor(val):
        return val.to(device)
    if isinstance(val, dict):
        return {k: move_to_device(v, device) for k, v in val.items()}
    return val


def masked_l2_per_sample(
    x: Tensor,
    y: Tensor,
    mask: Optional[Tensor] = None,
    reduce: bool = True,
) -> Tensor:
    """
    x, y   : [B, K, D, T]
    mask   : [B, K, D, T], True = invalid/masked
    """
    diff_sq = (x - y) ** 2

    if mask is not None:
        valid = (~mask.bool()).float()
        diff_sq = diff_sq * valid
    else:
        valid = torch.ones_like(diff_sq)

    per_sample_sum = diff_sq.flatten(start_dim=1).sum(dim=1)
    valid_count = valid.flatten(start_dim=1).sum(dim=1).clamp(min=1)
    per_sample_loss = per_sample_sum / valid_count

    if reduce:
        return per_sample_loss.mean()
    return per_sample_loss


class PoseTrainingPortal(BaseTrainingPortal):
    """
    Expected dataloader output:
      - batch["data"] -> [B, T_chunk, K, D]
      - batch["conditions"]["input_sequence"] -> [B, T_cond, K, D]
      - batch["conditions"]["previous_output"] -> [B, T_hist, K, D]

    Model interface expects:
      - x_t -> [B, K, D, T_chunk]
      - input_sequence -> [B, K, D, T_cond]
      - previous_output -> [B, K, D, T_hist]
    """

    def __init__(
        self,
        config: Any,
        model: torch.nn.Module,
        diffusion: GaussianDiffusion,
        dataloader: DataLoader,
        logger: Optional[Any],
        tb_writer: Optional[Any],
        validation_dataloader: Optional[DataLoader] = None,
        prior_loader: Optional[DataLoader] = None,
    ):
        super().__init__(config, model, diffusion, dataloader, logger, tb_writer, prior_loader)

        self.pose_header = None
        self.device = config.device
        self.validation_dataloader = validation_dataloader
        self.best_validation_metric = float("inf")

        if not hasattr(dataloader.dataset, "input_mean") or not hasattr(dataloader.dataset, "input_std"):
            raise AttributeError("Training dataset must expose input_mean and input_std")

        self.train_input_mean = torch.tensor(
            dataloader.dataset.input_mean,
            device=self.device,
            dtype=torch.float32,
        ).squeeze()

        self.train_input_std = torch.tensor(
            dataloader.dataset.input_std,
            device=self.device,
            dtype=torch.float32,
        ).squeeze()

        self.val_pose_header = None
        if self.validation_dataloader is not None:
            self.val_pose_header = getattr(self.validation_dataloader.dataset, "pose_header", None)
            if self.val_pose_header is not None and self.logger:
                self.logger.info("Pose header loaded from validation dataset.")

        if HAS_POSE_EVAL:
            self.validation_metric_calculator = DistanceMetric(
                name="Validation DTW",
                distance_measure=DTWDTAIImplementationDistanceMeasure(
                    name="dtaiDTW",
                    use_fast=True,
                    default_distance=0.0,
                ),
                pose_preprocessors=[NormalizePosesProcessor()],
            )
            if self.logger:
                self.logger.info("Initialized DTW validation metric")
        else:
            self.validation_metric_calculator = None
            if self.logger:
                self.logger.info("pose_evaluation not installed; DTW validation disabled")

    def diffuse(
        self,
        x_start: Tensor,
        t: Tensor,
        cond: Dict[str, Tensor],
        noise: Optional[Tensor] = None,
        return_loss: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        x_start from loader: [B, T_chunk, K, D]
        cond["input_sequence"]: [B, T_cond, K, D]
        cond["previous_output"]: [B, T_hist, K, D]
        """
        x_start = x_start.permute(0, 2, 3, 1).contiguous().to(self.device)  # [B, K, D, T]

        if noise is None:
            noise = torch.randn_like(x_start)

        t_device = t.to(self.device)
        x_t = self.diffusion.q_sample(x_start, t_device, noise=noise)

        processed_cond = {}
        for key, val in cond.items():
            processed_cond[key] = move_to_device(val, self.device)

        if "input_sequence" not in processed_cond:
            raise KeyError("conditions must contain 'input_sequence'")

        processed_cond["input_sequence"] = processed_cond["input_sequence"].permute(0, 2, 3, 1).contiguous()

        if "previous_output" in processed_cond and processed_cond["previous_output"] is not None:
            processed_cond["previous_output"] = processed_cond["previous_output"].permute(0, 2, 3, 1).contiguous()

        model_output = self.model.interface(
            x_t,
            self.diffusion._scale_timesteps(t_device),
            processed_cond,
        )  # [B, K, D, T]

        model_output_loader_shape = model_output.permute(0, 3, 1, 2).contiguous()  # [B, T, K, D]

        if not return_loss:
            return model_output_loader_shape, None

        loss_terms = {}

        mmt = self.diffusion.model_mean_type
        if mmt.name == "PREVIOUS_X":
            target = self.diffusion.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t_device)[0]
        elif mmt.name == "START_X":
            target = x_start
        elif mmt.name == "EPSILON":
            target = noise
        else:
            raise ValueError(f"Unsupported model_mean_type: {mmt}")

        if model_output.shape != target.shape or model_output.shape != x_start.shape:
            raise ValueError(
                f"Shape mismatch: model_output={tuple(model_output.shape)}, "
                f"target={tuple(target.shape)}, x_start={tuple(x_start.shape)}"
            )

        mask_from_loader = processed_cond.get("target_mask", None)
        if mask_from_loader is not None:
            mask = mask_from_loader.permute(0, 2, 3, 1).contiguous().bool()
        else:
            mask = torch.zeros_like(x_start, dtype=torch.bool)

        if getattr(self.config.trainer, "use_loss_mse", True):
            loss_terms["loss_data"] = masked_l2_per_sample(target, model_output, mask, reduce=True)

        if getattr(self.config.trainer, "use_loss_vel", True):
            target_vel = target[..., 1:] - target[..., :-1]
            model_output_vel = model_output[..., 1:] - model_output[..., :-1]
            mask_vel = mask[..., 1:] if mask is not None else None
            loss_terms["loss_data_vel"] = masked_l2_per_sample(
                target_vel,
                model_output_vel,
                mask_vel,
                reduce=True,
            )

            if getattr(self.config.trainer, "use_loss_accel", False):
                target_accel = target_vel[..., 1:] - target_vel[..., :-1]
                model_output_accel = model_output_vel[..., 1:] - model_output_vel[..., :-1]
                mask_accel = mask_vel[..., 1:] if mask_vel is not None else None
                loss_terms["loss_data_accel"] = masked_l2_per_sample(
                    target_accel,
                    model_output_accel,
                    mask_accel,
                    reduce=True,
                )

        total_loss = torch.tensor(0.0, device=self.device)

        if "loss_data" in loss_terms:
            total_loss = total_loss + loss_terms["loss_data"]

        lambda_vel = getattr(self.config.trainer, "lambda_vel", 1.0)
        if "loss_data_vel" in loss_terms:
            total_loss = total_loss + lambda_vel * loss_terms["loss_data_vel"]

        lambda_accel = getattr(self.config.trainer, "lambda_accel", 1.0)
        if "loss_data_accel" in loss_terms:
            total_loss = total_loss + lambda_accel * loss_terms["loss_data_accel"]

        loss_terms["loss"] = total_loss

        return model_output_loader_shape, loss_terms

    def _compute_dtw_score(self, predictions: List[Pose], references: List[Pose]) -> float:
        if self.validation_metric_calculator is None:
            if self.logger:
                self.logger.info("DTW skipped because pose_evaluation is unavailable.")
            return float("inf")

        start_time = time.time()
        wrapped_refs = [references]
        mean_score = float(self.validation_metric_calculator.corpus_score(predictions, wrapped_refs))
        elapsed = time.time() - start_time

        if self.logger:
            self.logger.info(f"Validation DTW corpus_score time: {elapsed:.4f}s")
            self.logger.info(f"=== Validation DTW (corpus_score): {mean_score:.4f} ===")

        if self.tb_writer:
            self.tb_writer.add_scalar("validation/DTW_distance", mean_score, self.epoch)

        return mean_score

    def _sample_chunk_with_cfg(
        self,
        input_sequence_bkdt: Tensor,
        previous_output_bkdt: Tensor,
        target_shape: Tuple[int, int, int, int],
    ) -> Tensor:
        guidance_scale = getattr(self.config.trainer, "guidance_scale", 2.0)

        cond_dict = {
            "input_sequence": input_sequence_bkdt,
            "previous_output": previous_output_bkdt,
        }

        uncond_dict = {
            "input_sequence": torch.zeros_like(input_sequence_bkdt),
            "previous_output": previous_output_bkdt,
        }

        wrapped_model_cond = _ConditionalWrapper(self.model, cond_dict)
        cond_chunk = self.diffusion.p_sample_loop(
            model=wrapped_model_cond,
            shape=target_shape,
            clip_denoised=getattr(self.config.diff, "clip_denoised", False),
            model_kwargs={"y": cond_dict},
            progress=False,
        )

        wrapped_model_uncond = _ConditionalWrapper(self.model, uncond_dict)
        uncond_chunk = self.diffusion.p_sample_loop(
            model=wrapped_model_uncond,
            shape=target_shape,
            clip_denoised=getattr(self.config.diff, "clip_denoised", False),
            model_kwargs={"y": uncond_dict},
            progress=False,
        )

        return uncond_chunk + guidance_scale * (cond_chunk - uncond_chunk)

    def _process_validation_batch(
        self,
        batch_data: Dict[str, Any],
        batch_idx: int,
    ) -> Tuple[List[Pose], List[Pose]]:
        """
        Chunk-level validation.
        """
        with torch.no_grad():
            gt_loader = batch_data["data"].to(self.device)  # [B, T, K, D]
            disfluent_loader = batch_data["conditions"]["input_sequence"].to(self.device)
            history_loader = batch_data["conditions"]["previous_output"].to(self.device)

            B, T_chunk, K, D_feat = gt_loader.shape

            disfluent_bkdt = disfluent_loader.permute(0, 2, 3, 1).contiguous()
            history_bkdt = history_loader.permute(0, 2, 3, 1).contiguous()

            pred_bkdt = self._sample_chunk_with_cfg(
                input_sequence_bkdt=disfluent_bkdt,
                previous_output_bkdt=history_bkdt,
                target_shape=(B, K, D_feat, T_chunk),
            )

            pred_loader = pred_bkdt.permute(0, 3, 1, 2).contiguous()

            val_ds = self.validation_dataloader.dataset
            if not hasattr(val_ds, "input_mean") or not hasattr(val_ds, "input_std"):
                raise AttributeError("Validation dataset must expose input_mean and input_std")

            val_mean = torch.tensor(val_ds.input_mean, device=self.device).view(1, 1, K, D_feat)
            val_std = torch.tensor(val_ds.input_std, device=self.device).view(1, 1, K, D_feat)

            gt_unnorm = gt_loader * val_std + val_mean
            pred_unnorm = pred_loader * val_std + val_mean

            refs, preds = [], []
            fps = getattr(self.val_pose_header, "fps", 25.0) if self.val_pose_header is not None else 25.0

            for i in range(B):
                ref_np = gt_unnorm[i].cpu().numpy().reshape(T_chunk, 1, K, D_feat).astype(np.float64)
                pred_np = pred_unnorm[i].cpu().numpy().reshape(T_chunk, 1, K, D_feat).astype(np.float64)

                ref_body = NumPyPoseBody(
                    fps=fps,
                    data=ref_np,
                    confidence=np.ones((T_chunk, 1, K), dtype=np.float32),
                )
                pred_body = NumPyPoseBody(
                    fps=fps,
                    data=pred_np,
                    confidence=np.ones((T_chunk, 1, K), dtype=np.float32),
                )

                refs.append(Pose(self.val_pose_header, ref_body))
                preds.append(Pose(self.val_pose_header, pred_body))

            return refs, preds

    def _run_validation_epoch(self) -> Optional[float]:
        if self.validation_dataloader is None:
            if self.logger:
                self.logger.info("Validation dataloader not provided. Skipping validation.")
            return None

        self.model.eval()
        with torch.no_grad():
            references, predictions = [], []
            for batch_idx, batch_data in enumerate(self.validation_dataloader):
                batch_refs, batch_preds = self._process_validation_batch(batch_data, batch_idx)
                references.extend(batch_refs)
                predictions.extend(batch_preds)

        if not references:
            if self.logger:
                self.logger.warning("No poses collected during validation for DTW calculation.")
            self.model.train()
            return float("inf")

        if self.logger:
            self.logger.info(f"Calculating DTW for {len(references)} validation samples...")

        dtw_score = self._compute_dtw_score(predictions, references)
        self.model.train()
        return dtw_score

    def run_loop(self, enable_profiler=False, profiler_directory="./logs/tb_profiler"):
        use_amp = getattr(self.config.trainer, "use_amp", False)
        scaler = GradScaler("cuda") if use_amp else None

        if enable_profiler:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_directory),
            )
            profiler.start()
        else:
            profiler = None

        sampling_num = min(4, len(self.dataloader.dataset))
        sampling_idx = np.random.randint(0, len(self.dataloader.dataset), sampling_num)
        sampling_subset = DataLoader(
            Subset(self.dataloader.dataset, sampling_idx),
            batch_size=min(2, sampling_num),
            shuffle=False,
            num_workers=0,
            collate_fn=zero_pad_collator,
        )

        if self.validation_dataloader is not None:
            num_to_save = getattr(self.config.trainer, "validation_save_num", 30)
            val_dataset = self.validation_dataloader.dataset
            if len(val_dataset) > num_to_save:
                self.validation_sample_indices = np.random.choice(
                    len(val_dataset), num_to_save, replace=False
                ).tolist()
            else:
                self.validation_sample_indices = list(range(len(val_dataset)))
        else:
            self.validation_sample_indices = []

        epoch_process_bar = tqdm(range(self.epoch, self.num_epochs), desc=f"Epoch {self.epoch}")

        for epoch_idx in epoch_process_bar:
            self.model.train()
            self.model.training = True
            self.epoch = epoch_idx
            epoch_losses = {}

            data_len = len(self.dataloader)

            for _, datas in enumerate(tqdm(self.dataloader, desc=f"Epoch {epoch_idx}")):
                datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
                cond = {
                    key: val.to(self.device) if torch.is_tensor(val) else val
                    for key, val in datas["conditions"].items()
                }
                x_start = datas["data"]

                self.opt.zero_grad()
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)

                if use_amp:
                    with autocast("cuda"):
                        _, losses = self.diffuse(x_start, t, cond, noise=None, return_loss=True)
                        total_loss = (losses["loss"] * weights).mean()
                    scaler.scale(total_loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    _, losses = self.diffuse(x_start, t, cond, noise=None, return_loss=True)
                    total_loss = (losses["loss"] * weights).mean()
                    total_loss.backward()
                    self.opt.step()

                if profiler:
                    profiler.step()

                if self.config.trainer.ema:
                    self.ema.update()

                for key_name, val in losses.items():
                    if "loss" in key_name:
                        epoch_losses.setdefault(key_name, []).append(val.mean().item())

            if profiler:
                profiler.stop()
                profiler = None

            if self.prior_loader is not None:
                for prior_datas in itertools.islice(self.prior_loader, data_len):
                    prior_datas = {
                        key: val.to(self.device) if torch.is_tensor(val) else val
                        for key, val in prior_datas.items()
                    }
                    prior_cond = {
                        key: val.to(self.device) if torch.is_tensor(val) else val
                        for key, val in prior_datas["conditions"].items()
                    }
                    prior_x_start = prior_datas["data"]

                    self.opt.zero_grad()
                    t, weights = self.schedule_sampler.sample(prior_x_start.shape[0], self.device)

                    if use_amp:
                        with autocast("cuda"):
                            _, prior_losses = self.diffuse(prior_x_start, t, prior_cond, noise=None, return_loss=True)
                            total_loss = (prior_losses["loss"] * weights).mean()
                        scaler.scale(total_loss).backward()
                        scaler.step(self.opt)
                        scaler.update()
                    else:
                        _, prior_losses = self.diffuse(prior_x_start, t, prior_cond, noise=None, return_loss=True)
                        total_loss = (prior_losses["loss"] * weights).mean()
                        total_loss.backward()
                        self.opt.step()

                    for key_name, val in prior_losses.items():
                        if "loss" in key_name:
                            epoch_losses.setdefault(key_name, []).append(val.mean().item())

            avg_loss_data = np.mean(epoch_losses["loss_data"]) if "loss_data" in epoch_losses else 0.0
            avg_loss_vel = np.mean(epoch_losses["loss_data_vel"]) if "loss_data_vel" in epoch_losses else 0.0
            avg_loss_total = np.mean(epoch_losses["loss"]) if "loss" in epoch_losses else 0.0

            loss_str = (
                f"loss_data: {avg_loss_data:.6f}, "
                f"loss_data_vel: {avg_loss_vel:.6f}, "
                f"loss: {avg_loss_total:.6f}"
            )

            epoch_avg_loss = avg_loss_total

            if self.epoch > 10 and epoch_avg_loss < self.best_loss:
                self.save_checkpoint(filename="best")

            if epoch_avg_loss < self.best_loss:
                self.best_loss = epoch_avg_loss

            epoch_process_bar.set_description(
                f"Epoch {epoch_idx}/{self.config.trainer.epoch} | "
                f"loss: {epoch_avg_loss:.6f} | best_loss: {self.best_loss:.6f}"
            )

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch_idx}/{self.config.trainer.epoch} | "
                    f"{loss_str} | best_loss: {self.best_loss:.6f}"
                )

            save_freq = max(1, int(getattr(self.config.trainer, "save_freq", 1)))
            if epoch_idx > 0 and epoch_idx % save_freq == 0:
                self.save_checkpoint(filename=f"weights_{epoch_idx}")

            if self.tb_writer:
                for key_name, vals in epoch_losses.items():
                    if "loss" in key_name:
                        self.tb_writer.add_scalar(f"train/{key_name}", np.mean(vals), epoch_idx)

            self.scheduler.step()

            eval_freq = max(1, int(getattr(self.config.trainer, "eval_freq", 1)))
            if self.validation_dataloader is not None and epoch_idx % eval_freq == 0:
                current_validation_metric = self._run_validation_epoch()

                if self.tb_writer and current_validation_metric is not None:
                    self.tb_writer.add_scalar("validation/DTW_distance", current_validation_metric, self.epoch)

                if (
                    current_validation_metric is not None
                    and current_validation_metric < self.best_validation_metric
                ):
                    self.best_validation_metric = current_validation_metric
                    if self.logger:
                        self.logger.info(
                            f"*** New best validation metric: "
                            f"{self.best_validation_metric:.4f} at epoch {self.epoch}. "
                            f"Saving best validation model. ***"
                        )
                    self.save_checkpoint(filename="best_model_validation")

                val_losses = []
                for val_batch in self.validation_dataloader:
                    val_batch_device = {}
                    for key, v_item in val_batch.items():
                        if torch.is_tensor(v_item):
                            val_batch_device[key] = v_item.to(self.device)
                        elif isinstance(v_item, dict):
                            val_batch_device[key] = {
                                sk: sv.to(self.device) if torch.is_tensor(sv) else sv
                                for sk, sv in v_item.items()
                            }
                        else:
                            val_batch_device[key] = v_item

                    current_cond_for_diffuse = {
                        k: v for k, v in val_batch_device.get("conditions", {}).items()
                    }

                    x_start = val_batch_device["data"]
                    t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                    with torch.no_grad():
                        _, losses = self.diffuse(
                            x_start,
                            t,
                            current_cond_for_diffuse,
                            noise=None,
                            return_loss=True,
                        )
                    batch_loss = (losses["loss"] * weights).mean().item()
                    val_losses.append(batch_loss)

                if self.tb_writer and val_losses:
                    avg_val_loss = np.mean(val_losses)
                    self.tb_writer.add_scalar("validation/loss", avg_val_loss, self.epoch)

                if self.validation_sample_indices:
                    save_dir = Path(self.config.save) / "validation_samples" / f"epoch_{self.epoch}"
                    mkdir(save_dir)

                    val_dataset = self.validation_dataloader.dataset
                    val_save_loader = DataLoader(
                        Subset(val_dataset, self.validation_sample_indices),
                        batch_size=1,
                        shuffle=False,
                        num_workers=self.config.trainer.workers,
                        pin_memory=True,
                        collate_fn=zero_pad_collator,
                    )

                    for loader_idx, batch_data in enumerate(val_save_loader):
                        refs, preds = self._process_validation_batch(batch_data, batch_idx=0)
                        idx = self.validation_sample_indices[loader_idx]

                        ref = refs[0]
                        pred = preds[0]

                        ref_path = save_dir / f"ref_epoch{self.epoch}_idx{idx}.pose"
                        with open(ref_path, "wb") as f:
                            ref.write(f)

                        pred_path = save_dir / f"pred_epoch{self.epoch}_idx{idx}.pose"
                        with open(pred_path, "wb") as f:
                            pred.write(f)

                    if self.logger:
                        self.logger.info(
                            f"Saved {len(self.validation_sample_indices)} "
                            f"validation GT and predictions to {save_dir}"
                        )

    def evaluate_sampling(self, dataloader: DataLoader, save_folder_name: str = "init_samples"):
        """
        Real sampling from diffusion using conditions.
        """
        self.model.eval()
        self.model.training = False

        mkdir(f"{self.save_dir}/{save_folder_name}")

        patched_dataloader = DataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=self.config.trainer.workers,
            collate_fn=zero_pad_collator,
            pin_memory=True,
        )

        datas = next(iter(patched_dataloader))

        def get_original_dataset(dataset):
            while isinstance(dataset, torch.utils.data.Subset):
                dataset = dataset.dataset
            return dataset

        dataset = get_original_dataset(patched_dataloader.dataset)
        self.pose_header = getattr(dataset, "pose_header", None)

        gt_chunk = datas["data"].to(self.device)

        cond = {
            key: (val.to(self.device) if torch.is_tensor(val) else val)
            for key, val in datas["conditions"].items()
        }

        B, T_chunk, K, D_feat = gt_chunk.shape
        input_sequence_bkdt = cond["input_sequence"].permute(0, 2, 3, 1).contiguous()
        previous_output_bkdt = cond["previous_output"].permute(0, 2, 3, 1).contiguous()

        pred_bkdt = self._sample_chunk_with_cfg(
            input_sequence_bkdt=input_sequence_bkdt,
            previous_output_bkdt=previous_output_bkdt,
            target_shape=(B, K, D_feat, T_chunk),
        )

        pred_output_tensor = pred_bkdt.permute(0, 3, 1, 2).contiguous()

        gt_chunk_array = gt_chunk.cpu().numpy()
        pred_output_array = pred_output_tensor.cpu().numpy()

        if not hasattr(dataset, "input_mean") or not hasattr(dataset, "input_std"):
            raise AttributeError("Dataset must expose input_mean and input_std")

        unnormed_gt = gt_chunk_array * dataset.input_std + dataset.input_mean
        unnormed_pred = pred_output_array * dataset.input_std + dataset.input_mean

        self.export_samples(unnormed_gt, f"{self.save_dir}/{save_folder_name}", "gt")
        self.export_samples(unnormed_pred, f"{self.save_dir}/{save_folder_name}", "pred")

        np.save(f"{self.save_dir}/{save_folder_name}/gt_output_normed.npy", gt_chunk_array)
        np.save(f"{self.save_dir}/{save_folder_name}/pred_output_normed.npy", pred_output_array)
        np.save(f"{self.save_dir}/{save_folder_name}/gt_output.npy", np.stack(unnormed_gt, axis=0))
        np.save(f"{self.save_dir}/{save_folder_name}/pred_output.npy", np.stack(unnormed_pred, axis=0))

        if self.logger:
            self.logger.info(f"Evaluate sampling {save_folder_name} at epoch {self.epoch}")
        else:
            print(f"Evaluate sampling {save_folder_name} at epoch {self.epoch}")

    def export_samples(self, pose_output_np: np.ndarray, save_path: str, prefix: str) -> list:
        """
        pose_output_np shape: (B, T, K, D)
        """
        pose_header = self.pose_header if self.pose_header is not None else self.val_pose_header
        if pose_header is None:
            raise ValueError("pose_header is required to export .pose files")

        for i in range(pose_output_np.shape[0]):
            pose_array = pose_output_np[i]
            time_len, keypoints, dim = pose_array.shape
            pose_array = pose_array.reshape(time_len, 1, keypoints, dim)

            confidence = np.ones((time_len, 1, keypoints), dtype=np.float32)

            pose_body = NumPyPoseBody(fps=25, data=pose_array, confidence=confidence)
            pose_obj = Pose(pose_header, pose_body)

            file_path = f"{save_path}/pose_{i}.{prefix}.pose"
            with open(file_path, "wb") as f:
                pose_obj.write(f)

            with open(file_path, "rb") as f_check:
                Pose.read(f_check.read())

        return pose_output_np
