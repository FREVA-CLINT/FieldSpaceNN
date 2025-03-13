"""
This code originally started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

The code was further modified from OpenAI's Guided diffusion model:
https://github.com/openai/guided-diffusion
"""

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
    """Defines model prediction output type: previous step, starting state, or epsilon."""
    PREVIOUS_X = enum.auto()  # Model predicts x_{t-1}
    START_X = enum.auto()     # Model predicts x_0
    EPSILON = enum.auto()     # Model predicts epsilon


class ModelVarType(enum.Enum):
    """Defines the type of variance output that the model will produce."""
    LEARNED = enum.auto()        # Model learns to output variance directly
    FIXED_SMALL = enum.auto()    # Model outputs a small fixed variance
    FIXED_LARGE = enum.auto()    # Model outputs a large fixed variance
    LEARNED_RANGE = enum.auto()  # Model learns to output variance within a range


class LossType(enum.Enum):
    """Defines the type of loss function used in the diffusion model training."""
    MSE = enum.auto()            # Mean Squared Error loss
    RESCALED_MSE = enum.auto()    # Rescaled MSE loss
    KL = enum.auto()             # Kullback-Leibler divergence loss
    RESCALED_KL = enum.auto()     # Rescaled KL loss

    def is_vb(self) -> bool:
        """Returns True if the loss type is variational bound (VB) related."""
        return self in (LossType.KL, LossType.RESCALED_KL)


class GaussianDiffusion:
    """
    Core class for training and sampling diffusion models.

    :param betas: Array of beta values per timestep in diffusion process.
    :param model_mean_type: Specifies what the model is learning to predict (e.g., epsilon or x_0).
    :param model_var_type: Specifies how the model's output variance is computed.
    :param loss_type: Specifies the type of loss function for training.
    :param rescale_timesteps: If True, rescale timesteps between 0 and 1000 as in the original paper.
    :param diffusion_steps: Total number of steps in the diffusion process.
    :param diffusion_step_scheduler: Type of beta schedule to use.
    :param diffusion_step_sampler: Custom sampler function for diffusion steps; defaults to uniform.
    """

    def __init__(
        self,
        *,
        betas: Optional[np.ndarray] = None,
        model_mean_type: ModelMeanType = ModelMeanType.EPSILON,
        model_var_type: ModelVarType = ModelVarType.FIXED_LARGE,
        loss_type: LossType = LossType.MSE,
        rescale_timesteps: bool = False,
        clip_denoised: bool = True,
        diffusion_steps: int = 1000,
        diffusion_step_scheduler: str = "linear",
        diffusion_step_sampler: Optional[Callable] = None
    ):
        # Initialize model parameters
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_steps = rescale_timesteps
        self.clip_denoised = clip_denoised

        # Initialize beta schedule if not provided
        if betas is None:
            betas = get_named_beta_schedule(diffusion_step_scheduler, diffusion_steps)
            betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        # Setup default diffusion step sampler if not provided
        if diffusion_step_sampler is None:
            diffusion_step_sampler = create_named_schedule_sampler("uniform", diffusion_steps)
        self.diffusion_step_sampler = diffusion_step_sampler
        self.diffusion_steps = int(betas.shape[0])

        # Pre-compute various quantities for the diffusion process
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # Quantities for the forward process q(x_t | x_{t-1}) and posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior variance, log variance, and coefficients
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(
        self, x_start: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute q(x_t | x_0) distribution.

        :param x_start: Tensor of noiseless inputs of shape [N x C x ...].
        :param t: Tensor representing the timestep.
        :return: Tuple of (mean, variance, log_variance) of shape matching x_start.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(
        self, x_start: torch.Tensor, diff_steps: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample q(x_t | x_0) by applying diffusion steps.

        :param x_start: Initial data batch, tensor of shape [N x C x ...].
        :param diff_steps: Diffusion step tensor of shape [N].
        :param noise: Optional Gaussian noise tensor of shape matching x_start.
        :return: Noisy version of x_start after diffusion steps.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, diff_steps, x_start.shape) * x_start
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, diff_steps, x_start.shape) * noise
        )

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, diff_steps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance of the posterior q(x_{t-1} | x_t, x_0).

        :param x_start: Starting (noiseless) data tensor.
        :param x_t: Tensor at diffusion step t.
        :param diff_steps: Diffusion step tensor.
        :return: Tuple of (posterior_mean, posterior_variance, posterior_log_variance_clipped).
        """
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, diff_steps, x_t.shape) * x_start
                + extract_into_tensor(self.posterior_mean_coef2, diff_steps, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, diff_steps, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, diff_steps, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def process_xstart(self, in_tensor, t, denoised_fn):
        if denoised_fn is not None:
            in_tensor = denoised_fn(in_tensor)
        if self.clip_denoised:
            # Compute f(diff_step) based on the noise schedule
            snr = extract_into_tensor(self.sqrt_alphas_cumprod, t, in_tensor.shape) / (
                        1 - extract_into_tensor(self.sqrt_alphas_cumprod, t, in_tensor.shape) + 1e-5)
            f_t = snr / (snr + 1)  # Alternative: use self.sqrt_alphas_cumprod[diff_steps]

            # Compute a clamped version of x_0
            in_tensor_clamp = torch.clamp(in_tensor, -1, 1)

            # Weighted combination of raw prediction and clamped version
            in_tensor = f_t * in_tensor + (1 - f_t) * in_tensor_clamp
        return in_tensor

    def p_mean_variance(
        self,
        model: Callable,
        x: torch.Tensor,
        diff_steps: torch.Tensor,
        mask: torch.Tensor = None,
        emb: Dict = None,
        denoised_fn: Optional[Callable] = None,
        **model_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Apply the model to get p(x_{t-1} | x_t) and predict x_0.

        :param model: The model function to generate predictions.
        :param x: Input tensor at diffusion step t.
        :param mask: Mask tensor applied to x_t.
        :param emb: embedding dictionary
        :param diff_steps: Diffusion step tensor.
        :param denoised_fn: Optional function applied to x_start.
        :param model_kwargs: Extra arguments for the model.
        :return: Dictionary with keys "mean", "variance", "log_variance", and "pred_xstart".
        """

        # Ensure correct input shapes
        b, c = x.shape[0], x.shape[-1]
        assert diff_steps.shape == (b,)
        if emb is None:
            emb = {}
        emb["DiffusionStepEmbedder"] = self._scale_steps(diff_steps)
        model_output = model(x, emb=emb, mask=mask, **model_kwargs)
        model_output = model_output.view(x.shape)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # Splits the output into predicted mean and log variance
            assert model_output.shape == (*x.shape[:-1], c * 2)
            model_output, model_var_values = torch.split(model_output, c, dim=-1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = extract_into_tensor(
                    self.posterior_log_variance_clipped, diff_steps, x.shape
                )
                max_log = extract_into_tensor(np.log(self.betas), diff_steps, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            # For fixed variance types, fetch values directly from class attributes
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = extract_into_tensor(model_variance, diff_steps, x.shape)
            model_log_variance = extract_into_tensor(model_log_variance, diff_steps, x.shape)

        # Predict initial x depending on mean type
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = self.process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=diff_steps, xprev=model_output), diff_steps, denoised_fn
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = self.process_xstart(model_output, diff_steps, denoised_fn)
            else:
                pred_xstart = self.process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=diff_steps, eps=model_output), diff_steps, denoised_fn
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, diff_steps=diff_steps
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Predict x_start from epsilon and x_t.

        :param x_t: The data tensor at time t.
        :param t: Timestep tensor.
        :param eps: The predicted noise.
        :return: Predicted x_start.
        """
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t: torch.Tensor, t: torch.Tensor, xprev: torch.Tensor) -> torch.Tensor:
        """
        Predict x_start from xprev and x_t.

        :param x_t: Data tensor at time t.
        :param t: Diffusion step tensor.
        :param xprev: Previous timestep tensor.
        :return: Predicted x_start.
        """
        return (
            extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract_into_tensor(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
        )

    def predict_eps_from_xstart(self, x_t: torch.Tensor, t: torch.Tensor, pred_xstart: torch.Tensor) -> torch.Tensor:
        """
        Predict epsilon from x_start and x_t.

        :param x_t: Data tensor at time t.
        :param t: Diffusion step tensor.
        :param pred_xstart: Predicted x_start tensor.
        :return: Predicted epsilon.
        """
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_steps(self, t: torch.Tensor) -> torch.Tensor:
        """
        Scale diffusion steps if required.

        This function scales the input timestep tensor `t` to the range [0, 1000] if
        `rescale_timesteps` is set to True. This is often necessary to match the timestep
        scaling used in certain diffusion model implementations.

        :param t: A tensor of diffusion steps to be scaled, of shape [N].
        :return: A scaled tensor of t.
        """
        # If rescaling is enabled, multiply by the scaling factor 1000 / diffusion_steps
        if self.rescale_steps:
            return t.float() * (1000.0 / self.diffusion_steps)
        return t

    def _vb_terms_bpd(
            self,
            model: Callable,
            x_start: torch.Tensor,
            x_t: torch.Tensor,
            mask: torch.Tensor,
            emb: Dict,
            diff_steps: torch.Tensor,
            **model_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute variational bound terms for training using KL divergence.

        This function calculates the KL divergence or negative log-likelihood (NLL) for the
        variational lower bound (VLB), measured in bits per dimension (bpd). The VB terms
        are essential for training with KL loss types.

        :param model: The model function used to predict mean and variance.
        :param x_start: Ground truth tensor for x_0, shape [N x ...].
        :param x_t: Noisy tensor at diffusion step t, shape [N x ...].
        :param mask: Mask tensor applied to x_t.
        :param diff_steps: Diffusion step tensor.
        :param model_kwargs: Additional arguments for the model.
        :return: A dictionary with the following keys:
                 - "output": Tensor of shape [N] representing VLB term.
                 - "pred_xstart": Prediction for x_0 as calculated by the model.
        """
        # Compute the true mean and log variance of the posterior q(x_{t-1} | x_t, x_0)
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, diff_steps=diff_steps
        )

        # Apply the model to predict mean and variance for p(x_{t-1} | x_t)
        out = self.p_mean_variance(
            model=model,
            x=x_t,
            diff_steps=diff_steps,
            mask=mask,
            emb=emb,
            **model_kwargs
        )

        # Calculate KL divergence between true posterior q and model's distribution p
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)  # Convert to bits per dimension

        # Calculate negative log-likelihood (decoder NLL) for the first timestep
        decoder_nll = -continuous_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)  # Convert to bits per dimension

        # At the first timestep, use NLL instead of KL, otherwise use KL
        output = torch.where(diff_steps == 0, decoder_nll, kl)

        return {
            "output": output,  # Final VB term
            "pred_xstart": out["pred_xstart"],  # Predicted x_0 for use in loss computation
        }

    def training_losses(
            self,
            model: Callable,
            gt_data: torch.Tensor,
            diff_steps: torch.Tensor,
            mask: torch.Tensor = None,
            emb: Dict = None,
            noise: Optional[torch.Tensor] = None,
            **model_kwargs
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute training losses for a single timestep.

        :param model: The model function to evaluate loss on.
        :param gt_data: Ground truth tensor of shape [N x ...].
        :param diff_steps: Diffusion step tensor of shape [N].
        :param mask: Mask data applied to x_t.
        :param emb: embedding dictionary
        :param model_kwargs: Extra arguments for the model.
        :param noise: Optional Gaussian noise tensor of the same shape as gt_data.
        :return: A tuple where the first element is a dictionary with keys:
                 - "loss": Training loss tensor of shape [N].
                 - "mse": Mean squared error (if applicable).
                 - "vb": Variational bound (if applicable).
                 The second element is a tensor containing the intermediate results.
        """
        if noise is None:
            noise = torch.randn_like(gt_data)

        # Diffuse ground truth data for the given diffusion steps
        x_t = self.q_sample(gt_data, diff_steps, noise=noise)
        if torch.is_tensor(mask):
            x_t = torch.where(~mask, gt_data, x_t)

        terms = {}

        # Compute KL or variational bound (VB) losses for applicable loss types
        if self.loss_type in {LossType.KL, LossType.RESCALED_KL}:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=gt_data,
                x_t=x_t,
                diff_steps=diff_steps,
                mask=mask,
                emb=emb,
                **model_kwargs
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.diffusion_steps  # Rescale the KL term

        # Compute MSE or Rescaled MSE losses for applicable loss types
        elif self.loss_type in {LossType.MSE, LossType.RESCALED_MSE}:
            if emb is None:
                emb = {}
            emb["DiffusionStepEmbedder"] = self._scale_steps(diff_steps)
            model_output = model(x_t, emb=emb, mask=mask, **model_kwargs)

            if self.model_var_type in {ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE}:
                b, c = x_t.shape[0], x_t.shape[-1]
                assert model_output.shape == (b, c * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, c, dim=-1)
                # Variational bound for learning variance without affecting mean prediction
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=-1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=gt_data,
                    x_t=x_t,
                    diff_steps=diff_steps,
                    mask=mask,
                    emb=emb,
                    **model_kwargs
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Rescale VB term to keep MSE term unaffected
                    terms["vb"] *= self.diffusion_steps / 1000.0

            # Define target based on model mean type
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=gt_data, x_t=x_t, diff_steps=diff_steps
                )[0],
                ModelMeanType.START_X: gt_data,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            target = target.view(model_output.shape)
            mask = mask.view(model_output.shape)
            x_t = x_t.view(model_output.shape)

            # Calculate mean squared error loss and add to terms
            if torch.is_tensor(mask):
                terms["mse"] = mean_flat((torch.where(mask, target - model_output, 0)) ** 2)
            else:
                terms["mse"] = mean_flat((target  - model_output) ** 2)
            terms["loss"] = terms["mse"] + terms.get("vb", 0)

        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        # Compute intermediate results if model is predicting epsilon
        if self.model_mean_type == ModelMeanType.EPSILON:
            results = (x_t - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, diff_steps, x_t.shape)
                       * model_output) / extract_into_tensor(self.sqrt_alphas_cumprod, diff_steps, x_t.shape)
            if torch.is_tensor(mask):
                results = torch.where(mask, results, x_t)
            results = self.process_xstart(results, diff_steps, None)
        else:
            results = model_output

        return terms, results

    def get_diffusion_steps(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sampled diffusion steps.

        :param batch_size: Batch size.
        :param device: Device for the resulting tensors.
        :return: Tuple of diffusion steps and weights.
        """
        t, weights = self.diffusion_step_sampler.sample(batch_size)
        return t.to(device), weights.to(device)


def extract_into_tensor(arr: np.ndarray, diffusion_steps: torch.Tensor, broadcast_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: The 1-D numpy array of values.
    :param diffusion_steps: Diffusion step tensor for array indexing.
    :param broadcast_shape: Shape to broadcast the output tensor.
    :return: Tensor of extracted values expanded to `broadcast_shape`.
    """
    res = torch.from_numpy(arr).to(device=diffusion_steps.device)[diffusion_steps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)