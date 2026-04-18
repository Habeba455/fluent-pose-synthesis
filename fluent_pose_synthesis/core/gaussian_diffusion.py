"""
PoseGaussianDiffusion — يستبدل create_gaussian_diffusion بتاع CAMDM.
الفرق الأساسي: بيقرا predict_xstart من الـ config فعلاً (CAMDM كان بيتجاهله).
"""

from typing import Optional, Dict, Any
import torch as th

from CAMDM.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from CAMDM.diffusion.respace import SpacedDiffusion, space_timesteps


class PoseGaussianDiffusion(SpacedDiffusion):
    """
    SpacedDiffusion (زي اللي بيعمله CAMDM.create_gaussian_diffusion)
    بس بيحترم predict_xstart اللي في config، مش hard-coded.
    """
    pass


def create_pose_gaussian_diffusion(config) -> PoseGaussianDiffusion:
    """
    Drop-in replacement for CAMDM's create_gaussian_diffusion.

    الفرق الأساسي:
    - CAMDM: predict_xstart = True (hard-coded — بيتجاهل الـ config)
    - ده  : predict_xstart = getattr(config.diff, "predict_xstart", False)

    وده اللي بيخلي الموديل يتعلم حركة حقيقية بدلاً من mean pose ثابتة.
    """
    # ★ FIX: قراءة من الـ config (default = False عشان epsilon prediction)
    predict_xstart = getattr(config.diff, "predict_xstart", False)
    learn_sigma = getattr(config.diff, "learn_sigma", False)
    rescale_timesteps = getattr(config.diff, "rescale_timesteps", False)
    timestep_respacing = getattr(config.diff, "timestep_respacing", "")
    use_kl = getattr(config.diff, "use_kl", False)
    rescale_learned_sigmas = getattr(config.diff, "rescale_learned_sigmas", False)

    steps = config.diff.diffusion_steps
    scale_beta = 1.0

    betas = get_named_beta_schedule(config.diff.noise_schedule, steps, scale_beta)

    # Loss type
    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    # Model mean type: EPSILON (predict noise) vs START_X (predict x_0)
    model_mean_type = (
        ModelMeanType.EPSILON if not predict_xstart
        else ModelMeanType.START_X
    )

    # Variance type
    if learn_sigma:
        model_var_type = ModelVarType.LEARNED_RANGE
    elif getattr(config.diff, "sigma_small", True):
        model_var_type = ModelVarType.FIXED_SMALL
    else:
        model_var_type = ModelVarType.FIXED_LARGE

    diffusion = PoseGaussianDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_3d=10,
        lambda_vel=getattr(config.trainer, "lambda_vel", 1),
        lambda_r_vel=1,
    )

    print("=" * 60)
    print("PoseGaussianDiffusion created with:")
    print(f"  predict_xstart  = {predict_xstart}")
    print(f"  model_mean_type = {model_mean_type}")
    print(f"  model_var_type  = {model_var_type}")
    print(f"  loss_type       = {loss_type}")
    print(f"  num_timesteps   = {diffusion.num_timesteps}")
    print(f"  lambda_vel      = {getattr(config.trainer, 'lambda_vel', 1)}")
    print("=" * 60)

    return diffusion
