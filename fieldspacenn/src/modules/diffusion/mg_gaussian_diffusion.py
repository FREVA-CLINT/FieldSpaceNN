import enum
import math
import numpy as np
import torch
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from .resample import create_named_schedule_sampler


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean over all non-batch dimensions.

    :param tensor: A tensor of shape ``(b, ...)``.
    :return: Mean of the tensor across all dimensions except the first, shape ``(b,)``.
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
    V_PREDICTION = enum.auto() # Model predicts v = sqrt(alpha_bar) * eps - sqrt(1-alpha_bar) * x_0


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


class MGGaussianDiffusion:
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
        betas_zooms: Optional[Mapping[int, np.ndarray]] = None,
        model_mean_type: Union[ModelMeanType, str] = "epsilon",
        model_var_type: ModelVarType = ModelVarType.FIXED_LARGE,
        loss_type: LossType = LossType.MSE,
        rescale_timesteps: bool = False,
        clip_denoised: bool = False,
        use_dynamic_clipping: bool = False,
        diffusion_steps: int = 1000,
        diffusion_step_scheduler: Mapping[int, str] | str = "linear",
        diffusion_step_sampler: Optional[Callable] = None,
        uncertainty_diffusion: bool = False,
        density_diffusion: bool = False,
        separate_noise_on_zoom: bool = True
    ) -> None:
        """
        Initialize diffusion schedules and precompute per-zoom coefficients.

        :param betas_zooms: Optional mapping from zoom to beta schedules.
        :param model_mean_type: Model output type ("epsilon", "v_prediction", etc.).
        :param model_var_type: Variance prediction type.
        :param loss_type: Loss criterion used during training.
        :param rescale_timesteps: Whether to rescale timesteps to [0, 1000].
        :param clip_denoised: Whether to clip x_0 predictions.
        :param use_dynamic_clipping: Whether to use dynamic clipping.
        :param diffusion_steps: Number of diffusion steps.
        :param diffusion_step_scheduler: Scheduler name(s) per zoom or a single string.
        :param diffusion_step_sampler: Optional sampler for diffusion steps.
        :param uncertainty_diffusion: Whether to use uncertainty-aware diffusion.
        :param density_diffusion: Whether to use density-aware diffusion.
        :param separate_noise_on_zoom: Whether to sample noise per zoom level.
        :return: None.
        """
        self.model_mean_type: ModelMeanType
        if model_mean_type == "epsilon":
            self.model_mean_type = ModelMeanType.EPSILON
        elif model_mean_type == "v_prediction":
            self.model_mean_type = ModelMeanType.V_PREDICTION
        elif model_mean_type == "previous_x":
            self.model_mean_type = ModelMeanType.PREVIOUS_X
        elif model_mean_type == "start_x":
            self.model_mean_type = ModelMeanType.START_X
        elif isinstance(model_mean_type, ModelMeanType):
            self.model_mean_type = model_mean_type
        else:
            raise NotImplementedError
        self.model_var_type: ModelVarType = model_var_type
        self.loss_type: LossType = loss_type
        self.rescale_steps: bool = rescale_timesteps
        self.clip_denoised: bool = clip_denoised
        self.use_dynamic_clipping: bool = use_dynamic_clipping
        self.separate_noise_on_zoom: bool = separate_noise_on_zoom

        # Use float64 for accuracy.
        if betas_zooms is None:
            betas_zooms = {}
            for zoom in diffusion_step_scheduler.keys():
                betas_zooms[zoom] = get_named_beta_schedule(diffusion_step_scheduler[zoom], diffusion_steps)
                betas_zooms[zoom] = np.array(betas_zooms[zoom], dtype=np.float64)
        self.betas_zooms: Dict[int, np.ndarray] = betas_zooms

        if diffusion_step_sampler is None:
            diffusion_step_sampler = create_named_schedule_sampler("uniform", diffusion_steps)
        self.diffusion_step_sampler: Any = diffusion_step_sampler
        self.diffusion_steps: int = int(betas_zooms[list(betas_zooms.keys())[0]].shape[0])

        alphas_zooms = {int(zoom): 1.0 - betas_zooms[zoom] for zoom in betas_zooms.keys()}
        self.alphas_cumprod_zooms: Dict[int, np.ndarray] = {
            int(zoom): np.cumprod(alphas_zooms[zoom], axis=0) for zoom in betas_zooms.keys()
        }
        self.alphas_cumprod_prev_zooms: Dict[int, np.ndarray] = {
            int(zoom): np.append(1.0, self.alphas_cumprod_zooms[zoom][:-1]) for zoom in betas_zooms.keys()
        }
        self.alphas_cumprod_next_zooms: Dict[int, np.ndarray] = {
            int(zoom): np.append(self.alphas_cumprod_zooms[zoom][1:], 0.0) for zoom in betas_zooms.keys()
        }

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod_zooms: Dict[int, np.ndarray] = {
            int(zoom): np.sqrt(self.alphas_cumprod_zooms[zoom]) for zoom in betas_zooms.keys()
        }
        self.sqrt_one_minus_alphas_cumprod_zooms: Dict[int, np.ndarray] = {
            int(zoom): np.sqrt(1.0 - self.alphas_cumprod_zooms[zoom]) for zoom in betas_zooms.keys()
        }
        self.log_one_minus_alphas_cumprod_zooms: Dict[int, np.ndarray] = {
            int(zoom): np.log(1.0 - self.alphas_cumprod_zooms[zoom]) for zoom in betas_zooms.keys()
        }
        self.sqrt_recip_alphas_cumprod_zooms: Dict[int, np.ndarray] = {
            int(zoom): np.sqrt(1.0 / self.alphas_cumprod_zooms[zoom]) for zoom in betas_zooms.keys()
        }
        self.sqrt_recipm1_alphas_cumprod_zooms: Dict[int, np.ndarray] = {
            int(zoom): np.sqrt(1.0 / self.alphas_cumprod_zooms[zoom] - 1) for zoom in betas_zooms.keys()
        }

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance_zooms: Dict[int, np.ndarray] = {int(zoom): (
                betas_zooms[zoom] * (1.0 - self.alphas_cumprod_prev_zooms[zoom]) / (1.0 - self.alphas_cumprod_zooms[zoom])
        ) for zoom in betas_zooms.keys()}
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped_zooms: Dict[int, np.ndarray] = {int(zoom): np.log(
            np.append(self.posterior_variance_zooms[zoom][1], self.posterior_variance_zooms[zoom][1:])
        ) for zoom in betas_zooms.keys()}
        self.posterior_mean_coef1_zooms: Dict[int, np.ndarray] = {int(zoom): (
                betas_zooms[zoom] * np.sqrt(self.alphas_cumprod_prev_zooms[zoom]) / (1.0 - self.alphas_cumprod_zooms[zoom])
        ) for zoom in betas_zooms.keys()}
        self.posterior_mean_coef2_zooms: Dict[int, np.ndarray] = {int(zoom): (
            (1.0 - self.alphas_cumprod_prev_zooms[zoom])
            * np.sqrt(alphas_zooms[zoom])
            / (1.0 - self.alphas_cumprod_zooms[zoom])
        ) for zoom in betas_zooms.keys()}
        self.uncertainty_diffusion: bool = uncertainty_diffusion
        self.density_diffusion: bool = density_diffusion

    def q_mean_variance(
        self, x_start: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the distribution q(x_t | x_0).

        :param x_start: Noiseless input tensor of shape ``(b, v, t, n, d, f)``.
        :param t: Diffusion step indices of shape ``(b,)``.
        :return: A tuple (mean, variance, log_variance), each of shape
            ``(b, v, t, n, d, f)``.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod_zooms, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod_zooms, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod_zooms, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(
        self,
        x_start_zooms: Mapping[int, torch.Tensor],
        diff_steps: torch.Tensor,
        noise_zooms: Optional[Mapping[int, torch.Tensor]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start_zooms: Initial data batch per zoom with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param diff_steps: Diffusion step indices of shape ``(b,)``.
        :param noise_zooms: Optional noise per zoom with tensors of shape
            ``(b, v, t, n, d, f)``.
        :return: A noisy version of x_start per zoom, shape ``(b, v, t, n, d, f)``.
        """
        if noise_zooms is None:
            noise_zooms = self.generate_noise(x_start_zooms)
        return {int(zoom): (
                extract_into_tensor(self.sqrt_alphas_cumprod_zooms[zoom], diff_steps, x_start_zooms[zoom].shape) * x_start_zooms[zoom]
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod_zooms[zoom], diff_steps, x_start_zooms[zoom].shape) * noise_zooms[zoom]
        ) for zoom in x_start_zooms.keys()}

    def q_posterior_mean_variance(
        self,
        x_start_zooms: Mapping[int, torch.Tensor],
        x_t_zooms: Mapping[int, torch.Tensor],
        diff_steps: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        :param x_start_zooms: Predicted x_start per zoom with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param x_t_zooms: Noisy inputs per zoom with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param diff_steps: Diffusion step indices of shape ``(b,)``.
        :return: Tuple of (mean, variance, log_variance) per zoom, each matching
            the input tensor shapes.
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

    def process_xstart(
        self,
        in_zooms: Mapping[int, torch.Tensor],
        t: torch.Tensor,
        denoised_fn: Optional[Callable] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Applies optional denoising function and clipping to the predicted x_start.

        Clipping behavior depends on `self.clip_denoised` and `self.use_dynamic_clipping`:
        - If `self.clip_denoised` is False, no clipping is applied.
        - If `self.clip_denoised` is True and `self.use_dynamic_clipping` is True,
          applies SNR-weighted clipping (original dynamic behavior).
        - If `self.clip_denoised` is True and `self.use_dynamic_clipping` is False,
          applies standard hard clamping to [-1.0, 1.0].

        :param in_zooms: Predicted x_start per zoom with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param t: The timestep tensor of shape ``(b,)``.
        :param denoised_fn: An optional function to apply to the tensor first.
        :return: The processed x_start per zoom of shape ``(b, v, t, n, d, f)``.
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
        x_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        diff_steps: torch.Tensor,
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        denoised_fn: Optional[Callable] = None,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Apply the model to get p(x_{t-1} | x_t) and predict the model output.

        :param model: the model, which takes a list of groups and a batch of timesteps as input.
        :param x_groups: List of per-group zoom mappings with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param diff_steps: Diffusion step indices of shape ``(b,)``.
        :param mask_groups: Optional list of per-group mask dictionaries with tensors
            of shape ``(b, v, t, n, d, f)`` or broadcastable to it.
        :param emb_groups: Optional list of embedding dictionaries.
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to compute means and variances.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': a list of model mean output dictionaries.
                 - 'variance': a list of model variance output dictionaries.
                 - 'log_variance': a list of log of 'variance' dictionaries.
                 - 'pred_xstart': a list of predicted x_start dictionaries.
                 Each dictionary maps zoom to tensors of shape ``(b, v, t, n, d, f)``.
        """
        if emb_groups is None:
            emb_groups = [{} for _ in x_groups]
        for emb in emb_groups:
            if emb is not None:
                emb["DiffusionStepEmbedder"] = self._scale_steps(diff_steps)

        if 'condition' in model_kwargs.keys():
            condition = model_kwargs.pop('condition')
            x_input_groups = [torch.cat([x_t_zooms, condition], dim=-1) if x_t_zooms else None for x_t_zooms in x_groups]
        else:
            x_input_groups = x_groups

        model_kwargs.update({"emb_groups": emb_groups, "mask_zooms_groups": mask_groups})
        model_output_groups = model(x_input_groups, **model_kwargs)

        output_mean_groups, output_variance_groups, output_log_variance_groups, output_pred_xstart_groups = [], [], [], []

        for i, (x_zooms, model_output_zooms) in enumerate(zip(x_groups, model_output_groups)):
            if not x_zooms:
                output_mean_groups.append(None)
                output_variance_groups.append(None)
                output_log_variance_groups.append(None)
                output_pred_xstart_groups.append(None)
                continue

            if self.model_var_type in [ModelVarType.FIXED_LARGE, ModelVarType.FIXED_SMALL]:
                model_variance_schedule, model_log_variance_schedule = {
                    ModelVarType.FIXED_LARGE: (
                        {int(zoom): np.append(self.posterior_variance_zooms[zoom][1], self.betas_zooms[zoom][1:]) for zoom in x_zooms.keys()},
                        {int(zoom): np.log(np.append(self.posterior_variance_zooms[zoom][1], self.betas_zooms[zoom][1:])) for zoom in x_zooms.keys()},
                    ),
                    ModelVarType.FIXED_SMALL: (
                        self.posterior_variance_zooms,
                        self.posterior_log_variance_clipped_zooms,
                    ),
                }[self.model_var_type]
                model_variance = {int(zoom): extract_into_tensor(model_variance_schedule[zoom], diff_steps, x_zooms[zoom].shape) for zoom in x_zooms.keys()}
                model_log_variance = {int(zoom): extract_into_tensor(model_log_variance_schedule[zoom], diff_steps, x_zooms[zoom].shape) for zoom in x_zooms.keys()}
            else:
                raise NotImplementedError(self.model_var_type)

            if self.model_mean_type == ModelMeanType.PREVIOUS_X:
                pred_xstart = self.process_xstart(
                    self._predict_xstart_from_xprev(x_t_zooms=x_zooms, t=diff_steps, xprev_zooms=model_output_zooms), diff_steps, denoised_fn
                )
                model_mean = model_output_zooms
            elif self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = self.process_xstart(model_output_zooms, diff_steps, denoised_fn)
                model_mean, _, _ = self.q_posterior_mean_variance(
                    x_start_zooms=pred_xstart, x_t_zooms=x_zooms, diff_steps=diff_steps
                )
            elif self.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = self.process_xstart(
                    self._predict_xstart_from_eps(x_t_zooms=x_zooms, t=diff_steps, eps_zooms=model_output_zooms), diff_steps, denoised_fn
                )
                model_mean, _, _ = self.q_posterior_mean_variance(
                    x_start_zooms=pred_xstart, x_t_zooms=x_zooms, diff_steps=diff_steps
                )
            elif self.model_mean_type == ModelMeanType.V_PREDICTION:
                pred_xstart = self.process_xstart(
                    self._predict_xstart_from_v(x_t_zooms=x_zooms, t=diff_steps, v_zooms=model_output_zooms), diff_steps, denoised_fn
                )
                model_mean, _, _ = self.q_posterior_mean_variance(
                    x_start_zooms=pred_xstart, x_t_zooms=x_zooms, diff_steps=diff_steps
                )
            else:
                raise NotImplementedError(self.model_mean_type)

            output_mean_groups.append(model_mean)
            output_variance_groups.append(model_variance)
            output_log_variance_groups.append(model_log_variance)
            output_pred_xstart_groups.append(pred_xstart)

        return {
            "mean": output_mean_groups,
            "variance": output_variance_groups,
            "log_variance": output_log_variance_groups,
            "pred_xstart": output_pred_xstart_groups,
        }

    def _predict_xstart_from_eps(
        self,
        x_t_zooms: Mapping[int, torch.Tensor],
        t: torch.Tensor,
        eps_zooms: Mapping[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Recover x_start from epsilon predictions.

        :param x_t_zooms: Noisy inputs per zoom of shape ``(b, v, t, n, d, f)``.
        :param t: Diffusion step indices of shape ``(b,)``.
        :param eps_zooms: Predicted epsilon per zoom of shape ``(b, v, t, n, d, f)``.
        :return: Predicted x_start per zoom of shape ``(b, v, t, n, d, f)``.
        """
        return {int(zoom): (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * x_t_zooms[zoom]
                - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * eps_zooms[zoom]
        ) for zoom in x_t_zooms.keys()}

    def _predict_xstart_from_xprev(
        self,
        x_t_zooms: Mapping[int, torch.Tensor],
        t: torch.Tensor,
        xprev_zooms: Mapping[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Recover x_start from predictions of x_{t-1}.

        :param x_t_zooms: Noisy inputs per zoom of shape ``(b, v, t, n, d, f)``.
        :param t: Diffusion step indices of shape ``(b,)``.
        :param xprev_zooms: Predicted x_{t-1} per zoom of shape ``(b, v, t, n, d, f)``.
        :return: Predicted x_start per zoom of shape ``(b, v, t, n, d, f)``.
        """
        return {int(zoom): (
                extract_into_tensor(1.0 / self.posterior_mean_coef1_zooms[zoom], t, x_t_zooms[zoom].shape) * xprev_zooms[zoom]
                - extract_into_tensor(
            self.posterior_mean_coef2_zooms[zoom] / self.posterior_mean_coef1_zooms[zoom], t, x_t_zooms[zoom].shape
            )
                * x_t_zooms[zoom]
        ) for zoom in x_t_zooms.keys()}

    def _predict_xstart_from_v(
        self,
        x_t_zooms: Mapping[int, torch.Tensor],
        t: torch.Tensor,
        v_zooms: Mapping[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Recover x_start from v-prediction outputs.

        :param x_t_zooms: Noisy inputs per zoom of shape ``(b, v, t, n, d, f)``.
        :param t: Diffusion step indices of shape ``(b,)``.
        :param v_zooms: Predicted v per zoom of shape ``(b, v, t, n, d, f)``.
        :return: Predicted x_start per zoom of shape ``(b, v, t, n, d, f)``.
        """
        return {int(zoom): (
                extract_into_tensor(self.sqrt_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * x_t_zooms[zoom]
                - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * v_zooms[zoom]
        ) for zoom in x_t_zooms.keys()}

    def _predict_eps_from_v(
        self,
        x_t_zooms: Mapping[int, torch.Tensor],
        t: torch.Tensor,
        v_zooms: Mapping[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Recover epsilon from v-prediction outputs.

        :param x_t_zooms: Noisy inputs per zoom of shape ``(b, v, t, n, d, f)``.
        :param t: Diffusion step indices of shape ``(b,)``.
        :param v_zooms: Predicted v per zoom of shape ``(b, v, t, n, d, f)``.
        :return: Predicted epsilon per zoom of shape ``(b, v, t, n, d, f)``.
        """
        return {int(zoom): (
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * x_t_zooms[zoom]
                + extract_into_tensor(self.sqrt_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * v_zooms[zoom]
        ) for zoom in x_t_zooms.keys()}

    def _predict_v_from_eps_and_xstart(
        self,
        eps_zooms: Mapping[int, torch.Tensor],
        x_start_zooms: Mapping[int, torch.Tensor],
        t: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute v-prediction from epsilon and x_start.

        :param eps_zooms: Predicted epsilon per zoom of shape ``(b, v, t, n, d, f)``.
        :param x_start_zooms: Predicted x_start per zoom of shape ``(b, v, t, n, d, f)``.
        :param t: Diffusion step indices of shape ``(b,)``.
        :return: Predicted v per zoom of shape ``(b, v, t, n, d, f)``.
        """
        return {int(zoom): (
                extract_into_tensor(self.sqrt_alphas_cumprod_zooms[zoom], t, eps_zooms[zoom].shape) * eps_zooms[zoom]
                - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod_zooms[zoom], t, eps_zooms[zoom].shape) * x_start_zooms[zoom]
        ) for zoom in eps_zooms.keys()}

    def predict_eps_from_xstart(
        self,
        x_t_zooms: Mapping[int, torch.Tensor],
        t: torch.Tensor,
        pred_xstart_zooms: Mapping[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Recover epsilon from x_start predictions.

        :param x_t_zooms: Noisy inputs per zoom of shape ``(b, v, t, n, d, f)``.
        :param t: Diffusion step indices of shape ``(b,)``.
        :param pred_xstart_zooms: Predicted x_start per zoom of shape ``(b, v, t, n, d, f)``.
        :return: Predicted epsilon per zoom of shape ``(b, v, t, n, d, f)``.
        """
        return {int(zoom): (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) * x_t_zooms[zoom]
                - pred_xstart_zooms[zoom]
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod_zooms[zoom], t, x_t_zooms[zoom].shape) for zoom in x_t_zooms.keys()}

    def _scale_steps(self, t: torch.Tensor) -> torch.Tensor:
        """
        Scale timesteps to match the original [0, 1000] convention.

        :param t: Diffusion step indices of shape ``(b,)``.
        :return: Rescaled timesteps of shape ``(b,)``.
        """
        if self.rescale_steps:
            return t.float() * (1000.0 / self.diffusion_steps)
        return t

    def training_losses(
        self,
        model: Callable,
        gt_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        diff_steps: torch.Tensor,
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        noise_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        create_pred_xstart: bool = False,
        **model_kwargs
    ) -> list:
        """
        Compute per-group diffusion training losses.

        :param model: Model callable that consumes grouped inputs and embeddings.
        :param gt_groups: List of per-group zoom mappings with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param diff_steps: Diffusion step indices of shape ``(b,)``.
        :param mask_groups: Optional list of mask dictionaries per group with tensors
            of shape ``(b, v, t, n, d, f)`` or broadcastable to it.
        :param emb_groups: Optional list of embedding dictionaries per group.
        :param noise_groups: Optional list of noise tensors per group with shape
            ``(b, v, t, n, d, f)``.
        :param create_pred_xstart: Whether to also return predicted x_start.
        :param model_kwargs: Additional keyword arguments forwarded to the model.
        :return: List of tuples ``(target, model_output, pred_xstart)`` per group,
            where each element is a zoom-to-tensor mapping with shape
            ``(b, v, t, n, d, f)`` or ``None``.
        """
        if noise_groups is None:
            noise_groups = [self.generate_noise(group) if group else None for group in gt_groups]

        if mask_groups is not None:
            for i, (mask_zooms, gt_zooms, noise_zooms) in enumerate(zip(mask_groups, gt_groups, noise_groups)):
                if mask_zooms and gt_zooms and noise_zooms:
                    # Zero out noise where the mask disables supervision.
                    noise_groups[i] = {int(zoom): torch.where(~mask_zooms[zoom], torch.zeros_like(gt_zooms[zoom]), noise_zooms[zoom]) for zoom in gt_zooms.keys()}

        x_t_groups = [self.q_sample(gt_zooms, diff_steps, noise_zooms=noise_zooms) if gt_zooms else None
                      for gt_zooms, noise_zooms in zip(gt_groups, noise_groups)]

        # Prepare model inputs
        if emb_groups is None:
            emb_groups = [{} for _ in gt_groups]
        
        for emb in emb_groups:
            if emb is not None:
                emb["DiffusionStepEmbedder"] = self._scale_steps(diff_steps)

        if 'condition' in model_kwargs:
            condition = model_kwargs.pop('condition')
            x_input_groups = [torch.cat([x_t_zooms, condition], dim=-1) if x_t_zooms else None for x_t_zooms in x_t_groups]
        else:
            x_input_groups = x_t_groups

        model_output_groups = model(x_input_groups, emb_groups=emb_groups, mask_zooms_groups=mask_groups, **model_kwargs)

        target_groups = []
        pred_xstart_groups = []
        
        if self.loss_type in {LossType.MSE, LossType.RESCALED_MSE}:
            for i, (gt_zooms, x_t_zooms, noise_zooms, model_output, mask_zooms) in enumerate(zip(gt_groups, x_t_groups, noise_groups, model_output_groups, mask_groups)):
                if not gt_zooms:
                    target_groups.append(None)
                    pred_xstart_groups.append(None)
                    continue

                # Determine the target for the MSE loss based on model_mean_type
                if self.model_mean_type == ModelMeanType.PREVIOUS_X:
                    target_zooms = self.q_posterior_mean_variance(x_start_zooms=gt_zooms, x_t_zooms=x_t_zooms, diff_steps=diff_steps)[0]
                elif self.model_mean_type == ModelMeanType.START_X:
                    target_zooms = gt_zooms
                elif self.model_mean_type == ModelMeanType.EPSILON:
                    target_zooms = noise_zooms
                elif self.model_mean_type == ModelMeanType.V_PREDICTION:
                    target_zooms = self._predict_v_from_eps_and_xstart(noise_zooms, gt_zooms, diff_steps)
                else:
                    raise NotImplementedError(self.model_mean_type)
                target_groups.append(target_zooms)

                if mask_zooms:
                    model_output_groups[i] = {int(zoom): torch.where(~mask_zooms[zoom], target_zooms[zoom], model_output[zoom]) for zoom in target_zooms.keys()}

                if create_pred_xstart:
                    if self.model_mean_type == ModelMeanType.START_X:
                        pred_xstart_zooms = self.process_xstart(model_output, diff_steps, None)
                    elif self.model_mean_type == ModelMeanType.EPSILON:
                        pred_xstart_zooms = self.process_xstart(self._predict_xstart_from_eps(x_t_zooms=x_t_zooms, t=diff_steps, eps_zooms=model_output), diff_steps, None)
                    elif self.model_mean_type == ModelMeanType.V_PREDICTION:
                        pred_xstart_zooms = self.process_xstart(self._predict_xstart_from_v(x_t_zooms=x_t_zooms, t=diff_steps, v_zooms=model_output), diff_steps, None)
                    elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
                        pred_xstart_zooms = self.process_xstart(self._predict_xstart_from_xprev(x_t_zooms=x_t_zooms, t=diff_steps, xprev_zooms=model_output), diff_steps, None)
                    pred_xstart_groups.append(pred_xstart_zooms)
                else:
                    pred_xstart_groups.append(None)
        else:
            raise NotImplementedError(self.loss_type)

        return list(zip(target_groups, model_output_groups, pred_xstart_groups))


    def get_diffusion_steps(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample diffusion steps and associated weights.

        :param batch_size: Number of samples to draw.
        :param device: Target device for the returned tensors.
        :return: Tuple of (diffusion_steps, weights), both of shape ``(batch_size,)``.
        """
        t, weights = self.diffusion_step_sampler.sample(batch_size)
        return t.to(device), weights.to(device)

    def get_uncertainty_timesteps(self, uncertainty_embedding: torch.Tensor) -> torch.Tensor:
        """
        Map uncertainty values in [0, 1] to discrete timesteps.

        :param uncertainty_embedding: Uncertainty tensor of shape ``(b,)``.
        :return: Timestep indices of shape ``(b,)``.
        """
        # Ensure uncertainty is clipped
        # Linear mapping: u=0 -> t=0, u=1 -> t = num_timesteps - 1
        t_map = (uncertainty_embedding * (self.diffusion_steps - 1)).round().long()
        return t_map

    def generate_noise(self, x_zooms: Mapping[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Generate Gaussian noise per zoom level.

        :param x_zooms: Input tensors per zoom of shape ``(b, v, t, n, d, f)``.
        :return: Noise tensors per zoom of shape ``(b, v, t, n, d, f)``.
        """
        if self.separate_noise_on_zoom:
            noise_zooms = {int(zoom): torch.randn_like(x_zooms[zoom]) for zoom in x_zooms.keys()}
        else:
            max_zoom = max(x_zooms.keys())
            noise = torch.randn_like(x_zooms[max_zoom])
            noise_zooms = {}
            for zoom in x_zooms.keys():
                # Reshape to share noise across zooms by spatially pooling.
                noise_zooms[zoom] = noise.view(*x_zooms[zoom].shape[:3], -1, 4 ** (max_zoom - zoom),
                                               x_zooms[zoom].shape[-2], x_zooms[zoom].shape[-1]).mean(dim=-3)
        return noise_zooms


def extract_into_tensor(
    arr: Union[np.ndarray, torch.Tensor],
    diffusion_steps: torch.Tensor,
    broadcast_shape: Tuple[int, ...],
) -> torch.Tensor:
    """
    Extract values from a 1-D numpy array or tensor for a batch of indices.

    :param arr: the 1-D numpy array or torch tensor.
    :param diffusion_steps: a tensor of indices into the array to extract,
        of shape ``(b,)``.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor with shape ``broadcast_shape``.
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
