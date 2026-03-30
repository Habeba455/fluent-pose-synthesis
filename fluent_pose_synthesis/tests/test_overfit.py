import random
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

from fluent_pose_synthesis.core.models import SignLanguagePoseDiffusion
from fluent_pose_synthesis.core.training import PoseTrainingPortal
from CAMDM.diffusion.create_diffusion import create_gaussian_diffusion


class DummyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"data": torch.tensor(0)}


def get_toy_batch(batch_size=2, seq_len=40, keypoints=178, dims=3):
    """
    Returns shapes compatible with PoseTrainingPortal.diffuse():
      data: [B, T, K, D]
      conditions["input_sequence"]: [B, T_cond, K, D]
      conditions["previous_output"]: [B, T_hist, K, D]
      conditions["target_mask"]: [B, T, K, D]  (True = masked/invalid)
    """
    assert batch_size == 2, "This toy batch currently assumes batch_size=2"

    base_linear = torch.linspace(0, 1, seq_len * keypoints * dims).reshape(seq_len, keypoints, dims)

    sine_t = torch.sin(torch.linspace(0, 4 * np.pi, seq_len)).view(seq_len, 1, 1)
    base_sine = sine_t.expand(seq_len, keypoints, dims)

    pose_data = []
    target_mask = []
    input_sequence = []
    previous_output = []

    cond_len = max(1, seq_len // 4)
    hist_len = max(1, seq_len // 4)

    for i in range(batch_size):
        if i == 0:
            sample = base_linear + torch.randn_like(base_linear) * 0.01
            frame_valid = torch.ones(seq_len, dtype=torch.bool)
            frame_valid[1::2] = False
            cond_sample = torch.zeros(cond_len, keypoints, dims)
            hist_sample = torch.zeros(hist_len, keypoints, dims)
        else:
            sample = 0.5 + 0.2 * base_sine + torch.randn_like(base_sine) * 0.01
            frame_valid = torch.ones(seq_len, dtype=torch.bool)
            frame_valid[::2] = False
            cond_sample = torch.ones(cond_len, keypoints, dims)
            hist_sample = torch.ones(hist_len, keypoints, dims) * 0.5

        pose_data.append(sample)

        # target_mask: True = masked/invalid
        # frame_valid True means valid -> mask False
        frame_mask = (~frame_valid).view(seq_len, 1, 1).expand(seq_len, keypoints, dims)
        target_mask.append(frame_mask)

        input_sequence.append(cond_sample)
        previous_output.append(hist_sample)

    batch = {
        "data": torch.stack(pose_data, dim=0),  # [B, T, K, D]
        "conditions": {
            "target_mask": torch.stack(target_mask, dim=0),          # [B, T, K, D]
            "input_sequence": torch.stack(input_sequence, dim=0),    # [B, T_cond, K, D]
            "previous_output": torch.stack(previous_output, dim=0),  # [B, T_hist, K, D]
        },
    }
    return batch


def create_minimal_config(device="cpu"):
    return Namespace(
        device=torch.device(device),
        save="./test_overfit_output",
        data="./dummy",
        trainer=Namespace(
            use_loss_mse=True,
            use_loss_vel=True,
            use_loss_3d=True,
            workers=0,
            batch_size=2,
            cond_mask_prob=0.15,
            load_num=1,
            lr=1e-3,
            epoch=100,
            lr_anneal_steps=0,
            weight_decay=0,
            ema=False,
            save_freq=5,
            lambda_vel=1.0,
        ),
        arch=Namespace(
            keypoints=178,
            dims=3,
            chunk_len=40,
            latent_dim=32,
            ff_size=64,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            decoder="trans_enc",
            ablation=None,
            activation="gelu",
            legacy=False,
        ),
        diff=Namespace(
            noise_schedule="cosine",
            diffusion_steps=4,
            sigma_small=True,
            clip_denoised=False,
        ),
    )


def plot_loss_curve(losses, save_path="loss_curve.png"):
    plt.figure()
    plt.plot(losses, label="Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Overfitting Loss Curve")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved loss plot to {save_path}")


def compute_average_keypoint_error(pose1, pose2):
    """
    pose shape expected: [B, K, D, T]
    """
    assert pose1.shape == pose2.shape, "Shape mismatch"
    # move time to second dim for readability if needed is not necessary;
    # just compute per-keypoint L2 over D
    diff = torch.norm(pose1 - pose2, dim=2)  # [B, K, T]
    return diff.mean().item()


def compute_cosine_distance(pose1, pose2):
    v1 = pose1.flatten()
    v2 = pose2.flatten()
    cos = F.cosine_similarity(v1, v2, dim=0)
    return 1 - cos.item()


def test_overfit_toy_batch():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = create_minimal_config(device="cpu")
    dummy_dataloader = DataLoader(DummyDataset())

    batch = get_toy_batch(
        batch_size=config.trainer.batch_size,
        seq_len=config.arch.chunk_len,
        keypoints=config.arch.keypoints,
        dims=config.arch.dims,
    )

    batch = {
        k: (
            v.to(config.device)
            if isinstance(v, torch.Tensor)
            else {kk: vv.to(config.device) for kk, vv in v.items()}
        )
        for k, v in batch.items()
    }

    diffusion = create_gaussian_diffusion(config)

    model = SignLanguagePoseDiffusion(
        input_feats=config.arch.keypoints * config.arch.dims,
        chunk_len=config.arch.chunk_len,
        keypoints=config.arch.keypoints,
        dims=config.arch.dims,
        latent_dim=config.arch.latent_dim,
        ff_size=config.arch.ff_size,
        num_layers=config.arch.num_layers,
        num_heads=config.arch.num_heads,
        dropout=config.arch.dropout,
        arch=config.arch.decoder,
        cond_mask_prob=config.trainer.cond_mask_prob,
        device=config.device,
    ).to(config.device)

    trainer = PoseTrainingPortal(
        config,
        model,
        diffusion,
        dataloader=dummy_dataloader,
        logger=None,
        tb_writer=None,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.trainer.lr)

    print("Start overfitting...")
    losses = []

    for step in range(config.trainer.epoch):
        t, weights = trainer.schedule_sampler.sample(config.trainer.batch_size, config.device)
        _, loss_dict = trainer.diffuse(batch["data"], t, batch["conditions"], return_loss=True)
        loss = (loss_dict["loss"] * weights).mean()
        losses.append(loss.item())
        print(f"[Step {step}] Loss: {loss.item():.6f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert losses[-1] < 1e-3, f"Final loss is too high: {losses[-1]:.6f}"
    plot_loss_curve(losses, save_path="overfit_loss_curve.png")

    model.eval()
    with torch.no_grad():
        t, _ = trainer.schedule_sampler.sample(1, config.device)

        # Direct model call expects [B, K, D, T]
        fluent_1 = batch["data"][0:1].permute(0, 2, 3, 1).contiguous()
        fluent_2 = batch["data"][1:2].permute(0, 2, 3, 1).contiguous()

        cond_1 = batch["conditions"]["input_sequence"][0:1].permute(0, 2, 3, 1).contiguous()
        cond_2 = batch["conditions"]["input_sequence"][1:2].permute(0, 2, 3, 1).contiguous()

        prev_1 = batch["conditions"]["previous_output"][0:1].permute(0, 2, 3, 1).contiguous()
        prev_2 = batch["conditions"]["previous_output"][1:2].permute(0, 2, 3, 1).contiguous()

        out1 = model(
            fluent_clip=fluent_1,
            disfluent_seq=cond_1,
            t=t,
            previous_output=prev_1,
        )

        out2 = model(
            fluent_clip=fluent_2,
            disfluent_seq=cond_2,
            t=t,
            previous_output=prev_2,
        )

        print(f"out1 shape: {out1.shape}, out2 shape: {out2.shape}")

        expected_shape = (
            1,
            config.arch.keypoints,
            config.arch.dims,
            config.arch.chunk_len,
        )
        assert out1.shape == out2.shape == expected_shape, (
            f"Unexpected output shape, expected {expected_shape}, "
            f"got out1={out1.shape}, out2={out2.shape}"
        )

        l2_diff = torch.norm(out1 - out2).item()
        avg_kpt_error = compute_average_keypoint_error(out1, out2)
        cosine_dist = compute_cosine_distance(out1, out2)

        print(f"Output L2 norm diff: {l2_diff:.6f}")
        print(f"Average keypoint error: {avg_kpt_error:.6f}")
        print(f"Cosine distance: {cosine_dist:.6f}")

        assert (avg_kpt_error > 0.01 or cosine_dist > 0.01), (
            "Outputs are too similar. Possible collapse."
        )

    print("Overfitting test passed.")


if __name__ == "__main__":
    test_overfit_toy_batch()
