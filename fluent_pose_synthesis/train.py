"""
Simple Kaggle-ready training script for sign-language pose diffusion.

Usage in Kaggle:
    !python train_lite.py

Or from a notebook cell:
    exec(open('train_lite.py').read())

Edit the CONFIG section below to change hyperparameters.
"""

import sys
import time
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.serialization
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pose_format.torch.masked.collator import zero_pad_collator
from CAMDM.diffusion.create_diffusion import create_gaussian_diffusion
from CAMDM.utils.common import fixseed
from CAMDM.utils.logger import Logger

# ★ Import the FIXED modules
from fluent_pose_synthesis.core.models import SignLanguagePoseDiffusion
from fluent_pose_synthesis.core.training import PoseTrainingPortal
from fluent_pose_synthesis.data.load_data import SignLanguagePoseDataset


# =========================================================
# CONFIG — EDIT THESE VALUES
# =========================================================
CONFIG = {
    # Paths
    "data": "/kaggle/working/final_dataset",
    "save": "/kaggle/working/train_output",
    "resume": None,                       # or path to checkpoint

    # Architecture
    "arch": {
        "decoder": "trans_enc",           # "trans_enc" | "trans_dec" | "gru"
        "chunk_len": 40,
        "history_len": 10,
        "keypoints": 178,                 # ← CHECK THIS matches your data
        "dims": 3,                        # ← CHECK THIS (2 or 3)
        "latent_dim": 256,
        "ff_size": 1024,
        "num_layers": 8,
        "num_heads": 4,
        "dropout": 0.1,
        "activation": "gelu",
    },

    # Training
    "trainer": {
        "batch_size": 32,
        "workers": 2,
        "epoch": 500,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "ema": True,
        "ema_rate": 0.9999,
        "cond_mask_prob": 0.1,
        "save_freq": 10,
        "eval_freq": 5,
        "validation_save_num": 10,
        "guidance_scale": 2.0,
        "use_amp": True,
        "use_loss_mse": True,
        "use_loss_vel": True,
        "use_loss_accel": False,
        "lambda_vel": 1.0,
        "lambda_accel": 1.0,
        "load_num": -1,
        "stride": 20,                     # windowing stride (chunk_len // 2)
    },

    # Diffusion
    "diff": {
        "noise_schedule": "cosine",
        "diffusion_steps": 8,
        "sigma_small": True,
        "predict_xstart": True,
        "use_kl": False,
        "rescale_timesteps": False,
        "timestep_respacing": "",
        "learn_sigma": False,
        "rescale_learned_sigmas": False,
        "clip_denoised": False,
    },

    # Misc
    "seed": 1024,
}


# =========================================================
# TORCH LOAD PATCH
# =========================================================
torch.serialization.add_safe_globals([
    SimpleNamespace, np.int64, np.int32, np.float64, np.float32, np.bool_,
])

_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    if not torch.cuda.is_available():
        kwargs.setdefault("map_location", torch.device("cpu"))
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


# =========================================================
# CONFIG HELPERS
# =========================================================
def dict_to_namespace(d):
    """Recursively convert dict -> SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def config_to_dict(cfg):
    """Recursively convert SimpleNamespace / Path -> JSON-safe dict."""
    if isinstance(cfg, SimpleNamespace):
        return {k: config_to_dict(v) for k, v in vars(cfg).items()}
    if isinstance(cfg, Path):
        return str(cfg)
    if isinstance(cfg, torch.device):
        return str(cfg)
    if isinstance(cfg, (list, tuple)):
        return [config_to_dict(i) for i in cfg]
    return cfg


# =========================================================
# TRAIN
# =========================================================
def train(config, resume_path, logger, tb_writer):
    """Main training loop."""
    np_dtype = np.float32

    # --- Training dataset (computes its own stats) ---
    logger.info("Loading training dataset...")
    train_dataset = SignLanguagePoseDataset(
        data_dir=config.data,
        split="train",
        chunk_len=config.arch.chunk_len,
        history_len=getattr(config.arch, "history_len", 10),
        dtype=np_dtype,
        limited_num=config.trainer.load_num,
        stride=getattr(config.trainer, "stride", None),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=zero_pad_collator,
    )
    logger.info(
        f"Training Dataset: {len(train_dataset)} windows "
        f"(chunk_len={config.arch.chunk_len}, history_len={config.arch.history_len})"
    )

    # --- Validation dataset (uses training stats — no data leakage) ---
    logger.info("Loading validation dataset...")
    train_stats = train_dataset.get_stats_dict()
    validation_dataset = SignLanguagePoseDataset(
        data_dir=config.data,
        split="val",                              # ★ matches your folder name
        chunk_len=config.arch.chunk_len,
        history_len=getattr(config.arch, "history_len", 10),
        dtype=np_dtype,
        limited_num=config.trainer.load_num,
        stride=getattr(config.trainer, "stride", None),
        external_stats=train_stats,               # ★ critical fix
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.trainer.workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=zero_pad_collator,
    )
    logger.info(f"Validation Dataset: {len(validation_dataset)} windows")

    # --- Diffusion + Model ---
    diffusion = create_gaussian_diffusion(config)
    input_feats = config.arch.keypoints * config.arch.dims

    # ★ only pass parameters the fixed model accepts
    model = SignLanguagePoseDiffusion(
        input_feats=input_feats,
        chunk_len=config.arch.chunk_len,
        keypoints=config.arch.keypoints,
        dims=config.arch.dims,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        dropout=getattr(config.arch, "dropout", 0.1),
        activation=getattr(config.arch, "activation", "gelu"),
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        batch_first=True,
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {type(model).__name__} | {n_params:,} trainable params")

    # --- Trainer ---
    trainer = PoseTrainingPortal(
        config, model, diffusion, train_dataloader, logger, tb_writer,
        validation_dataloader=validation_dataloader,
    )

    if resume_path is not None:
        try:
            trainer.load_checkpoint(str(resume_path))
            logger.info(f"Resumed from {resume_path} at epoch {trainer.epoch}")
        except FileNotFoundError:
            logger.info(f"No checkpoint found at {resume_path} — starting from scratch")

    profiler_dir = config.save / "profiler_logs"
    profiler_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training from epoch {trainer.epoch}")
    trainer.run_loop(
        enable_profiler=False,                # set True for profiling
        profiler_directory=str(profiler_dir),
    )


# =========================================================
# MAIN
# =========================================================
def main():
    start_time = time.time()

    # Build config namespace
    config = dict_to_namespace(CONFIG)
    config.data = Path(config.data)
    config.save = Path(config.save)

    fixseed(config.seed)

    # Handle existing save folder
    if config.save.exists() and config.resume is None:
        print(f"Save folder exists: {config.save}")
        print("Deleting it to start fresh.")
        shutil.rmtree(config.save, ignore_errors=True)

    config.save.mkdir(parents=True, exist_ok=True)

    # Resume path
    resume_path = None
    if config.resume:
        resume_path = Path(config.resume)
    elif config.save.exists():
        best_ckpt = config.save / "best.pt"
        if best_ckpt.exists():
            resume_path = best_ckpt
        else:
            ckpts = list(config.save.glob("weights_*.pt"))
            if ckpts:
                resume_path = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))

    logger = Logger(config.save / "log.txt")
    tb_writer = SummaryWriter(log_dir=config.save / "runtime")

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {config.device}")

    with open(config.save / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_dict(config), f, indent=4)
    logger.info(f"Saved config to {config.save / 'config.json'}")

    logger.info("Launching training")
    train(config, resume_path, logger, tb_writer)

    total_min = (time.time() - start_time) / 60
    logger.info(f"Total training time: {total_min:.2f} minutes")


if __name__ == "__main__":
    main()
