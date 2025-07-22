# gaussian_diffusion.py
import enum
import math
import numpy as np
import torch
from typing import Callable, Dict, Optional, Tuple, Union

from .resample import create_named_schedule_sampler
from .loss import normal_kl, continuous_gaussian_log_likelihood


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean over all non-batch dimensions.

    :param tensor: A PyTorch tensor of arbitrary shape.
    :return: Mean of the tensor across all dimensions except the first (batch dimension).
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> np.ndarray:
    """
    Get a beta schedule by name. Supports linear and cosine schedules.

    :param schedule_name: Name of the schedule ("linear" or "cosine").
    :param num_diffusion_timesteps: Total number of timesteps in the diffusion process.
    :return: A numpy array of beta values for each timestep.
    """
    if schedule_name == "linear":
        # Linear schedule scales beta values linearly over the timesteps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        # Cosine schedule smoothly varies beta using a cosine function.
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "sigmoid":
        # Sigmoid schedule provides a smooth, S-shaped transition.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        # Generate values for the sigmoid function, centered around 0.
        sig_range = np.linspace(-6, 6, num_diffusion_timesteps)
        # Apply the sigmoid function.
        sig = 1 / (1 + np.exp(-sig_range))
        # Scale the sigmoid output to the desired beta range.
        return beta_start + (beta_end - beta_start) * sig
    elif schedule_name == "exponential":
        # Exponential schedule increases beta values geometrically.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        # Creates a geometric progression from beta_start to beta_end.
        return np.geomspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "karras":
        # Karras (EDM) schedule, defined in terms of noise levels (sigma).
        # See https://arxiv.org/abs/2206.00364 for details.
        sigma_min = 0.002
        sigma_max = 80.0
        rho = 7.0

        # Generate the noise level schedule (sigmas)
        steps = np.arange(num_diffusion_timesteps, dtype=np.float64)
        ramp = steps / (num_diffusion_timesteps - 1)
        sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        # Convert sigmas to alpha_bars
        alpha_bars = 1.0 / (sigmas ** 2 + 1)

        # Convert alpha_bars to betas
        alpha_bars_prev = np.append(1.0, alpha_bars[:-1])
        betas = 1 - alpha_bars / alpha_bars_prev

        return np.clip(betas, 0.0, 0.999)  # Clip for numerical stability
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar: Callable[[float], float], max_beta: float = 0.999) -> np.ndarray:
    """
    Create a beta schedule that discretizes an alpha_t_bar function, representing
    cumulative product of (1 - beta) over time for t in [0, 1].

    :param num_diffusion_timesteps: Number of betas to produce.
    :param alpha_bar: Lambda function taking argument t from 0 to 1 and producing the cumulative product.
    :param max_beta: Maximum beta value to avoid singularities.
    :return: Array of beta values.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    PREVIOUS_X = enum.auto()  # Model predicts x_{t-1}
    START_X = enum.auto()     # Model predicts x_0
    EPSILON = enum.auto()     # Model predicts epsilon
    # <<< START NEW CODE >>>
    V_PREDICTION = enum.auto() # Model predicts v = sqrt(alpha_bar) * eps - sqrt(1-alpha_bar) * x_0
    # <<< END NEW CODE >>>


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making it learn the variance.
    """
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    """
    Which type of loss criterion to use.
    """
    MSE = enum.auto()  # Use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # Use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # Use the variational lower-bound
    RESCALED_KL = enum.auto()  # Like KL, but rescale to estimate the full VLB

    def is_vb(self):
        """
        Whether to use variational bound loss.
        """
        return self in [LossType.KL, LossType.RESCALED_KL]


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas_zooms: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    :param diffusion_steps: number of steps T
    :param diffusion_step_scheduler: method for betas calculation
    :param diffusion_step_sampler: sampler for diffusion steps; default - uniform
    """

    def __init__(
        self,
        *,
        betas_zooms: Optional[dict] = None,
        model_mean_type: ModelMeanType = "epsilon", # Default remains EPSILON
        model_var_type: ModelVarType = ModelVarType.FIXED_LARGE,
        loss_type: LossType = LossType.MSE,
        rescale_timesteps: bool = False,
        clip_denoised: bool = False,
        use_dynamic_clipping: bool = False,
        diffusion_steps: int = 1000,
        diffusion_step_scheduler: dict | str = "linear",
        diffusion_step_sampler: Optional[Callable] = None,
        uncertainty_diffusion = False,
        density_diffusion=False
    ):
        if model_mean_type == "epsilon":
            self.model_mean_type = ModelMeanType.EPSILON
        elif model_mean_type == "v_prediction":
            self.model_mean_type = ModelMeanType.V_PREDICTION
        elif model_mean_type == "previous_x":
            self.model_mean_type = ModelMeanType.PREVIOUS_X
        elif model_mean_type == "start_x":
            self.model_mean_type = ModelMeanType.START_X
        else:
            raise NotImplementedError
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_steps = rescale_timesteps
        self.clip_denoised = clip_denoised
        self.use_dynamic_clipping = use_dynamic_clipping

        # Use float64 for accuracy.
        if betas_zooms is None:
            betas_zooms = {}
            for zoom in diffusion_step_scheduler.keys():
                betas_zooms[zoom] = get_named_beta_schedule(diffusion_step_scheduler[zoom], diffusion_steps)
                betas_zooms[zoom] = np.array(betas_zooms[zoom], dtype=np.float64)
        self.betas_zooms = betas_zooms

        if diffusion_step_sampler is None:
            diffusion_step_sampler = create_named_schedule_sampler("uniform", diffusion_steps)
        self.diffusion_step_sampler = diffusion_step_sampler
        self.diffusion_steps = int(betas_zooms[list(betas_zooms.keys())[0]].shape[0])

        alphas_zooms = {int(zoom): 1.0 - betas_zooms[zoom] for zoom in betas_zooms.keys()}
        self.alphas_cumprod_zooms = {int(zoom): np.cumprod(alphas_zooms[zoom], axis=0) for zoom in betas_zooms.keys()}
        self.alphas_cumprod_prev_zooms = {int(zoom): np.append(1.0, self.alphas_cumprod_zooms[zoom][:-1]) for zoom in betas_zooms.keys()}
        self.alphas_cumprod_next_zooms = {int(zoom): np.append(self.alphas_cumprod_zooms[zoom][1:], 0.0) for zoom in betas_zooms.keys()}

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod_zooms = {int(zoom): np.sqrt(self.alphas_cumprod_zooms[zoom]) for zoom in betas_zooms.keys()}
        self.sqrt_one_minus_alphas_cumprod_zooms = {int(zoom): np.sqrt(1.0 - self.alphas_cumprod_zooms[zoom]) for zoom in betas_zooms.keys()}
        self.log_one_minus_alphas_cumprod_zooms = {int(zoom): np.log(1.0 - self.alphas_cumprod_zooms[zoom]) for zoom in betas_zooms.keys()}
        self.sqrt_recip_alphas_cumprod_zooms = {int(zoom): np.sqrt(1.0 / self.alphas_cumprod_zooms[zoom]) for zoom in betas_zooms.keys()}
        self.sqrt_recipm1_alphas_cumprod_zooms = {int(zoom): np.sqrt(1.0 / self.alphas_cumprod_zooms[zoom] - 1) for zoom in betas_zooms.keys()}

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance_zooms = {int(zoom): (
                betas_zooms[zoom] * (1.0 - self.alphas_cumprod_prev_zooms[zoom]) / (1.0 - self.alphas_cumprod_zooms[zoom])
        ) for zoom in betas_zooms.keys()}
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped_zooms = {int(zoom): np.log(
            np.append(self.posterior_variance_zooms[zoom][1], self.posterior_variance_zooms[zoom][1:])
        ) for zoom in betas_zooms.keys()}
        self.posterior_mean_coef1_zooms = {int(zoom): (
                betas_zooms[zoom] * np.sqrt(self.alphas_cumprod_prev_zooms[zoom]) / (1.0 - self.alphas_cumprod_zooms[zoom])
        ) for zoom in betas_zooms.keys()}
        self.posterior_mean_coef2_zooms = {int(zoom): (
            (1.0 - self.alphas_cumprod_prev_zooms[zoom])
            * np.sqrt(alphas_zooms[zoom])
            / (1.0 - self.alphas_cumprod_zooms[zoom])
        ) for zoom in betas_zooms.keys()}
        self.uncertainty_diffusion = uncertainty_diffusion
        self.density_diffusion = density_diffusion

    def q_mean_variance(
        self, x_start: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod_zooms, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod_zooms, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod_zooms, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(
        self, x_start_zooms: dict, diff_steps: torch.Tensor, noise_zooms: Optional[dict] = None
    ) -> dict:
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start_zooms: the initial data batch.
        :param diff_steps: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise_zooms: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise_zooms is None:
            noise_zooms = {int(zoom): torch.randn_like(x_start_zooms[zoom]) for zoom in x_start_zooms.keys()}
        return {int(zoom): (
                extract_into_tensor(self.sqrt_alphas_cumprod_zooms[zoom], diff_steps, x_start_zooms[zoom].shape) * x_start_zooms[zoom]
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod_zooms[zoom], diff_steps, x_start_zooms[zoom].shape) * noise_zooms[zoom]
        ) for zoom in x_start_zooms.keys()}

    def q_posterior_mean_variance(
        self, x_start_zooms: dict, x_t_zooms: dict, diff_steps: torch.Tensor
    ) -> Tuple[dict, dict, dict]:
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        posterior_mean_zooms = {int(zoom): (
                extract_into_tensor(self.posterior_mean_coef1_zooms[zoom], diff_steps, x_t_zooms[zoom].shape) * x_start_zooms[zoom]
                + extract_into_tensor(self.posterior_mean_coef2_zooms[zoom], diff_steps, x_t_zooms[zoom].shape) * x_t_zooms[zoom]
        ) for zoom in x_t_zooms.keys()}
        posterior_variance_zooms = {int(zoom): extract_into_tensor(self.posterior_variance_zooms[zoom], diff_steps, x_t_zooms[zoom].shape) for zoom in x_t_zooms.keys()}
        posterior_log_variance_clipped_zooms = {int(zoom): extract_into_tensor(
            self.posterior_log_variance_clipped_zooms[zoom], diff_steps, x_t_zooms[zoom].shape
        ) for zoom in x_t_zooms.keys()}
        return posterior_mean_zooms, posterior_variance_zooms, posterior_log_variance_clipped_zooms

    def process_xstart(self, in_zooms: dict, t: torch.Tensor, denoised_fn: Optional[Callable] = None) -> dict:
        """
        Applies optional denoising function and clipping to the predicted x_start.

        Clipping behavior depends on `self.clip_denoised` and `self.use_dynamic_clipping`:
        - If `self.clip_denoised` is False, no clipping is applied.
        - If `self.clip_denoised` is True and `self.use_dynamic_clipping` is True,
          applies SNR-weighted clipping (original dynamic behavior).
        - If `self.clip_denoised` is True and `self.use_dynamic_clipping` is False,
          applies standard hard clamping to [-1.0, 1.0].

        :param in_zooms: The predicted x_start tensor.
        :param t: The timestep tensor.
        :param denoised_fn: An optional function to apply to the tensor first.
        :return: The processed x_start tensor.
        """
        if denoised_fn is not None:
            # Apply custom denoising function if provided
            in_zooms = {int(zoom): denoised_fn(in_zooms) for zoom in in_zooms.keys()}

        # Check if any form of clipping should be applied
        if self.clip_denoised:
            for zoom in in_zooms.keys():
                if self.use_dynamic_clipping:
                    sqrt_alpha_cumprod_t = extract_into_tensor(self.sqrt_alphas_cumprod_zooms[zoom], t, in_zooms[zoom].shape)
                    denominator = 1.0 - sqrt_alpha_cumprod_t + 1e-5  # Add epsilon for numerical stability
                    snr = sqrt_alpha_cumprod_t / denominator

                    f_t = snr / (snr + 1.0)

                    in_tensor_clamp = torch.clamp(in_zooms[zoom], -1.0, 1.0)

                    in_zooms[zoom] = f_t * in_zooms[zoom] + (1.0 - f_t) * in_tensor_clamp
                else:
                    in_zooms[zoom] = torch.clamp(in_zooms[zoom], -1.0, 1.0)

        return in_zooms

    def p_mean_variance(
        self,
        model: Callable,
        x_zooms: dict,
        diff_steps: torch.Tensor,
        mask_zooms: dict = None,
        emb: Dict = None,
        denoised_fn: Optional[Callable] = None,
        **model_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Apply the model to get p(x_{t-1} | x_t) and predict the model output.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x_zooms: the [N x C x ...] tensor at time t.
        :param diff_steps: a 1-D Tensor of timesteps.
        :param mask_zooms: a [N x C x ...] tensor applied to x_t
        :param emb: an embedding dictionary
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to compute means and variances.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the predicted x_start.
        """
        if emb is None:
            emb = {}
        emb["DiffusionStepEmbedder"] = self._scale_steps(diff_steps)
        if 'condition' in model_kwargs.keys():
            x_input_zooms = {int(zoom): torch.cat([x_zooms[zoom], model_kwargs.pop('condition')][zoom], dim=-1) for zoom in x_zooms.keys()}
        else:
            x_input_zooms = x_zooms
        model_output_zooms = model(x_input_zooms.copy(), emb=emb.copy(), mask=mask_zooms.copy(), **model_kwargs)

        if self.model_var_type in [ModelVarType.FIXED_LARGE, ModelVarType.FIXED_SMALL]:
            # Use pre-calculated fixed variance schedules
            model_variance_zooms, model_log_variance_zooms = {
                # for fixedlarge, we set the initial variance estimates to B.
                # at the same time, we need to make sure we don't include beta_0.
                ModelVarType.FIXED_LARGE: (
                    {int(zoom): np.append(self.posterior_variance_zooms[zoom][1], self.betas_zooms[zoom][1:]) for zoom in x_zooms.keys()},
                    {int(zoom): np.log(np.append(self.posterior_variance_zooms[zoom][1], self.betas_zooms[zoom][1:])) for zoom in x_zooms.keys()},
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance_zooms,
                    self.posterior_log_variance_clipped_zooms,
                ),
            }[self.model_var_type]
            model_variance_zooms = {int(zoom): extract_into_tensor(model_variance_zooms[zoom], diff_steps, x_zooms[zoom].shape) for zoom in x_zooms.keys()}
            model_log_variance_zooms = {int(zoom): extract_into_tensor(model_log_variance_zooms[zoom], diff_steps, x_zooms[zoom].shape) for zoom in x_zooms.keys()}
        else:
            raise NotImplementedError

        # --- Calculate predicted x_start and model mean based on model_mean_type ---
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart_zooms = self.process_xstart(
                self._predict_xstart_from_xprev(x_t_zooms=x_zooms, t=diff_steps, xprev_zooms=model_output_zooms), diff_steps, denoised_fn
            )
            model_mean_zooms = model_output_zooms
        elif self.model_mean_type == ModelMeanType.START_X:
            pred_xstart_zooms = self.process_xstart(model_output_zooms, diff_steps, denoised_fn)
            model_mean_zooms, _, _ = self.q_posterior_mean_variance(
                x_start_zooms=pred_xstart_zooms, x_t_zooms=x_zooms, diff_steps=diff_steps
            )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart_zooms = self.process_xstart(
                self._predict_xstart_from_eps(x_t_zooms=x_zooms, t=diff_steps, eps_zooms=model_output_zooms), diff_steps, denoised_fn
            )
            model_mean_zooms, _, _ = self.q_posterior_mean_variance(
                x_start_zooms=pred_xstart_zooms, x_t_zooms=x_zooms, diff_steps=diff_steps
            )
        elif self.model_mean_type == ModelMeanType.V_PREDICTION:
             # Model outputs v_pred
            v_pred = model_output_zooms
            pred_xstart_zooms = self.process_xstart(
                self._predict_xstart_from_v(x_t_zooms=x_zooms, t=diff_steps, v_zooms=v_pred), diff_steps, denoised_fn
            )
            model_mean_zooms, _, _ = self.q_posterior_mean_variance(
                x_start_zooms=pred_xstart_zooms, x_t_zooms=x_zooms, diff_steps=diff_steps
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        return {
            "mean": model_mean_zooms,
            "variance": model_variance_zooms,
            "log_variance": model_log_variance_zooms,
            "pred_xstart": pred_xstart_zooms,
        }

    def _predict_xstart_from_eps(self, x_t_zooms: dict, t: torch.Tensor, eps_zooms: dict) -> dict:
        return {int(zoom): (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * x_t_zooms[zoom]
                - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * eps_zooms[zoom]
        ) for zoom in x_t_zooms.keys()}

    def _predict_xstart_from_xprev(self, x_t_zooms: dict, t: torch.Tensor, xprev_zooms: dict) -> dict:
        return {int(zoom): (
                extract_into_tensor(1.0 / self.posterior_mean_coef1_zooms[zoom], t, x_t_zooms[zoom].shape) * xprev_zooms[zoom]
                - extract_into_tensor(
            self.posterior_mean_coef2_zooms[zoom] / self.posterior_mean_coef1_zooms[zoom], t, x_t_zooms[zoom].shape
            )
                * x_t_zooms[zoom]
        ) for zoom in x_t_zooms.keys()}

    def _predict_xstart_from_v(self, x_t_zooms: dict, t: torch.Tensor, v_zooms: dict) -> dict:
        return {int(zoom): (
                extract_into_tensor(self.sqrt_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * x_t_zooms[zoom]
                - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * v_zooms[zoom]
        ) for zoom in x_t_zooms.keys()}

    def _predict_eps_from_v(self, x_t_zooms: dict, t: torch.Tensor, v_zooms: dict) -> dict:
        return {int(zoom): (
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * x_t_zooms[zoom]
                + extract_into_tensor(self.sqrt_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * v_zooms
        ) for zoom in x_t_zooms.keys()}

    def _predict_v_from_eps_and_xstart(self, eps_zooms: dict, x_start_zooms: dict, t: torch.Tensor) -> dict:
        return {int(zoom): (
                extract_into_tensor(self.sqrt_alphas_cumprod_zooms[zoom], t, eps_zooms[zoom].shape) * eps_zooms[zoom]
                - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod_zooms[zoom], t, eps_zooms[zoom].shape) * x_start_zooms[zoom]
        ) for zoom in eps_zooms.keys()}

    def predict_eps_from_xstart(self, x_t_zooms: dict, t: torch.Tensor, pred_xstart_zooms: dict) -> dict:
        return {int(zoom): (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * x_t_zooms[zoom]
                - pred_xstart_zooms[zoom]
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) for zoom in x_t_zooms.keys()}

    def _scale_steps(self, t: torch.Tensor) -> torch.Tensor:
        """
        Optional: Scale t to [0, 1000] if rescale_timesteps is True.
        """
        if self.rescale_steps:
            return t.float() * (1000.0 / self.diffusion_steps)
        return t

    def _vb_terms_bpd(
        self,
        model: Callable,
        x_start_zooms: dict,
        x_t_zooms: dict,
        mask_zooms: dict,
        emb: Dict,
        diff_steps: torch.Tensor,
        **model_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for easier comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start_zooms=x_start_zooms, x_t_zooms=x_t_zooms, diff_steps=diff_steps
        )
        out_zooms = self.p_mean_variance(
            model=model,
            x_zooms=x_t_zooms,
            diff_steps=diff_steps,
            mask_zooms=mask_zooms,
            emb=emb,
            **model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out_zooms["mean"], out_zooms["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -continuous_gaussian_log_likelihood(
            x_start_zooms, means=out_zooms["mean"], log_scales=0.5 * out_zooms["log_variance"]
        )
        assert decoder_nll.shape == x_start_zooms.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where(diff_steps == 0, decoder_nll, kl)
        return {"output": output, "pred_xstart": out_zooms["pred_xstart"]}

    def training_losses(
        self,
        model: Callable,
        gt_zooms: dict,
        diff_steps: torch.Tensor,
        mask_zooms: dict = None,
        emb: Dict = None,
        noise_zooms: dict = None,
        create_pred_xstart: bool = False,
        **model_kwargs
    ) -> Tuple[dict, dict, dict]:
        if noise_zooms is None:
            noise_zooms = {int(zoom): torch.randn_like(gt_zooms[zoom]) for zoom in gt_zooms.keys()}
        if mask_zooms is not None:
            noise_zooms = {int(zoom): torch.where(~mask_zooms[zoom], torch.zeros_like(gt_zooms[zoom]), noise_zooms[zoom]) for zoom in gt_zooms.keys()}
        x_t_zooms = self.q_sample(gt_zooms, diff_steps, noise_zooms=noise_zooms)

        if self.loss_type in {LossType.MSE, LossType.RESCALED_MSE}:
            # Prepare model inputs
            if emb is None:
                emb = {}
            emb["DiffusionStepEmbedder"] = self._scale_steps(diff_steps)
            if 'condition' in model_kwargs.keys():
                x_input_zooms = torch.cat([x_t_zooms, model_kwargs.pop('condition')], dim=-1)
            else:
                x_input_zooms = x_t_zooms
            model_output = model(x_input_zooms.copy(), emb=emb.copy(), mask=mask_zooms.copy(), **model_kwargs)

            # Determine the target for the MSE loss based on model_mean_type
            if self.model_mean_type == ModelMeanType.PREVIOUS_X:
                target_zooms = self.q_posterior_mean_variance(
                    x_start_zooms=gt_zooms, x_t_zooms=x_t_zooms, diff_steps=diff_steps
                )[0]
            elif self.model_mean_type == ModelMeanType.START_X:
                target_zooms = gt_zooms
            elif self.model_mean_type == ModelMeanType.EPSILON:
                target_zooms = noise_zooms
            elif self.model_mean_type == ModelMeanType.V_PREDICTION:
                 target_zooms = self._predict_v_from_eps_and_xstart(noise_zooms, gt_zooms, diff_steps)
            else:
                raise NotImplementedError(self.model_mean_type)

            model_output = {int(zoom): torch.where(~mask_zooms[zoom], target_zooms[zoom], model_output[zoom]) for zoom in target_zooms.keys()}

            if create_pred_xstart:
                if self.model_mean_type == ModelMeanType.START_X:
                    pred_xstart_zooms = self.process_xstart(model_output, diff_steps, None)
                elif self.model_mean_type == ModelMeanType.EPSILON:
                    pred_xstart_zooms = self.process_xstart(
                        self._predict_xstart_from_eps(x_t_zooms=x_t_zooms, t=diff_steps, eps_zooms=model_output), diff_steps, None
                    )
                elif self.model_mean_type == ModelMeanType.V_PREDICTION:
                    pred_xstart_zooms = self.process_xstart(
                        self._predict_xstart_from_v(x_t_zooms=x_t_zooms, t=diff_steps, v_zooms=model_output), diff_steps, None
                    )
                elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
                     pred_xstart_zooms = self.process_xstart(
                         self._predict_xstart_from_xprev(x_t_zooms=x_t_zooms, t=diff_steps, xprev_zooms=model_output), diff_steps, None
                     )
            else:
                pred_xstart_zooms = None
        else:
            raise NotImplementedError(self.loss_type)

        return target_zooms, model_output, pred_xstart_zooms # Return loss dict and predicted x0


    def get_diffusion_steps(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """ samples diffusion steps """
        t, weights = self.diffusion_step_sampler.sample(batch_size)
        return t.to(device), weights.to(device)

    def get_uncertainty_timesteps(self, uncertainty_embedding):
        """ Maps uncertainty [0, 1] to discrete timesteps [0, num_timesteps-1]. """
        # Ensure uncertainty is clipped
        # Linear mapping: u=0 -> t=0, u=1 -> t = num_timesteps - 1
        t_map = (uncertainty_embedding * (self.diffusion_steps - 1)).round().long()
        return t_map


def extract_into_tensor(arr: Union[np.ndarray, torch.Tensor], diffusion_steps: torch.Tensor, broadcast_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract values from a 1-D numpy array or tensor for a batch of indices.

    :param arr: the 1-D numpy array or torch tensor.
    :param diffusion_steps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # Convert numpy array to tensor if necessary
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)

    # Ensure array is on the same device as diffusion_steps
    dev = diffusion_steps.device
    arr = arr.to(dev)

    # Extract values using diffusion_steps as indices
    res = arr[diffusion_steps].float()

    # Reshape and expand to match broadcast_shape
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)