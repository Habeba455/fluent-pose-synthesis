import sys
import time
import shutil
import argparse
import json
from pathlib import Path, PosixPath
from types import SimpleNamespace

import numpy as np
import torch
import torch.serialization
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pose_format.torch.masked.collator import zero_pad_collator

from CAMDM.diffusion.create_diffusion import create_gaussian_diffusion
from CAMDM.utils.common import fixseed, select_platform
from CAMDM.utils.logger import Logger

from fluent_pose_synthesis.core.models import SignLanguagePoseDiffusion
from fluent_pose_synthesis.core.training import PoseTrainingPortal
from fluent_pose_synthesis.data.load_data import SignLanguagePoseDataset
from fluent_pose_synthesis.config.option import (
    add_model_args,
    add_train_args,
    add_diffusion_args,
    config_parse,
)

# Safe globals
torch.serialization.add_safe_globals([
    SimpleNamespace,
    PosixPath,
    np.int64,
    np.int32,
    np.float64,
    np.float32,
    np.bool_,
])

_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    if not torch.cuda.is_available():
        kwargs.setdefault("map_location", torch.device("cpu"))
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load


def train(config, resume_path, logger, tb_writer):

    np_dtype = select_platform(32)

    logger.info("Loading training dataset...")

    train_dataset = SignLanguagePoseDataset(
        data_dir=config.data,
        split="train",
        chunk_len=config.arch.chunk_len,
        history_len=getattr(config.arch, "history_len", 10),
        dtype=np_dtype,
        limited_num=config.trainer.load_num,
        min_condition_length=10,
        fixed_condition_length=-1
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=zero_pad_collator,
    )

    logger.info(
        f"Training Dataset includes {len(train_dataset)} samples "
        f"with {config.arch.chunk_len} frames each."
    )

    logger.info("Loading validation dataset...")

    validation_dataset = SignLanguagePoseDataset(
        data_dir=config.data,
        split="validation",
        chunk_len=config.arch.chunk_len,
        history_len=getattr(config.arch, "history_len", 10),
        dtype=np_dtype,
        limited_num=config.trainer.load_num,
        min_condition_length=10,
        fixed_condition_length=-1
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.trainer.workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=zero_pad_collator,
    )

    logger.info(f"Validation Dataset includes {len(validation_dataset)} samples.")

    diffusion = create_gaussian_diffusion(config)

    input_feats = config.arch.keypoints * config.arch.dims

    model = SignLanguagePoseDiffusion(
        input_feats=input_feats,
        chunk_len=config.arch.chunk_len,
        keypoints=config.arch.keypoints,
        dims=config.arch.dims,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        dropout=getattr(config.arch, "dropout", 0.2),
        ablation=getattr(config.arch, "ablation", None),
        activation=getattr(config.arch, "activation", "gelu"),
        legacy=getattr(config.arch, "legacy", False),
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        device=config.device,
    ).to(config.device)

    logger.info(f"Model created successfully")

    trainer = PoseTrainingPortal(
        config,
        model,
        diffusion,
        train_dataloader,
        logger,
        tb_writer,
        validation_dataloader=validation_dataloader
    )

    if resume_path is not None:
        try:
            trainer.load_checkpoint(str(resume_path))
        except FileNotFoundError:
            print(f"No checkpoint found at {resume_path}")
            sys.exit(1)

    profiler_dir = config.save / "profiler_logs"
    profiler_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training loop...")

    trainer.run_loop(
        enable_profiler=False,
        profiler_directory=str(profiler_dir)
    )


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--name", default="debug")
    parser.add_argument("-c", "--config", default="./fluent_pose_synthesis/config/default.json")
    parser.add_argument("-i", "--data", default="../dataset")
    parser.add_argument("-r", "--resume", default=None)
    parser.add_argument("-s", "--save", default="save")
    parser.add_argument("--seed", type=int, default=1024)

    add_model_args(parser)
    add_diffusion_args(parser)
    add_train_args(parser)

    args = parser.parse_args()

    fixseed(args.seed)

    config = config_parse(args)

    config.data = Path(config.data)
    config.save = Path(config.save)

    if "debug" in args.name:
        config.trainer.workers = 1
        config.trainer.load_num = -1
        config.trainer.batch_size = 16
        config.trainer.epoch = 300

    resume_path = Path(args.resume) if args.resume else None

    config.save.mkdir(parents=True, exist_ok=True)

    logger = Logger(config.save / "log.txt")

    tb_writer = SummaryWriter(log_dir=config.save / "runtime")

    if torch.cuda.is_available():
        config.device = torch.device("cuda")
    else:
        config.device = torch.device("cpu")

    logger.info(f"\nLaunching training with config:\n{config}")

    train(config, resume_path, logger, tb_writer)

    logger.info(f"\nTotal training time: {(time.time() - start_time) / 60:.2f} mins")


if __name__ == "__main__":
    main()