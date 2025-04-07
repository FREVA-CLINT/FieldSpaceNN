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

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
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
        betas: Optional[np.ndarray] = None,
        model_mean_type: ModelMeanType = ModelMeanType.EPSILON, # Default remains EPSILON
        model_var_type: ModelVarType = ModelVarType.FIXED_LARGE,
        loss_type: LossType = LossType.MSE,
        rescale_timesteps: bool = False,
        clip_denoised: bool = True,
        use_dynamic_clipping: bool = True,
        diffusion_steps: int = 1000,
        diffusion_step_scheduler: str = "linear",
        diffusion_step_sampler: Optional[Callable] = None,
        unmask_existing = True
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_steps = rescale_timesteps
        self.clip_denoised = clip_denoised
        self.use_dynamic_clipping = use_dynamic_clipping
        self.unmask_existing = unmask_existing

        # Use float64 for accuracy.
        if betas is None:
            betas = get_named_beta_schedule(diffusion_step_scheduler, diffusion_steps)
            betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        if diffusion_step_sampler is None:
            diffusion_step_sampler = create_named_schedule_sampler("uniform", diffusion_steps)
        self.diffusion_step_sampler = diffusion_step_sampler
        self.diffusion_steps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.diffusion_steps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(
        self, x_start: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(
        self, x_start: torch.Tensor, diff_steps: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param diff_steps: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
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
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, diff_steps, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, diff_steps, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, diff_steps, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, diff_steps, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def process_xstart(self, in_tensor: torch.Tensor, t: torch.Tensor, denoised_fn: Optional[Callable] = None) -> torch.Tensor:
        """
        Applies optional denoising function and clipping to the predicted x_start.

        Clipping behavior depends on `self.clip_denoised` and `self.use_dynamic_clipping`:
        - If `self.clip_denoised` is False, no clipping is applied.
        - If `self.clip_denoised` is True and `self.use_dynamic_clipping` is True,
          applies SNR-weighted clipping (original dynamic behavior).
        - If `self.clip_denoised` is True and `self.use_dynamic_clipping` is False,
          applies standard hard clamping to [-1.0, 1.0].

        :param in_tensor: The predicted x_start tensor.
        :param t: The timestep tensor.
        :param denoised_fn: An optional function to apply to the tensor first.
        :return: The processed x_start tensor.
        """
        if denoised_fn is not None:
            # Apply custom denoising function if provided
            in_tensor = denoised_fn(in_tensor)

        # Check if any form of clipping should be applied
        if self.clip_denoised:
            if self.use_dynamic_clipping:
                # --- Dynamic SNR-Weighted Clipping Logic ---
                # Calculate a Signal-to-Noise Ratio (SNR) like term based on the schedule.
                sqrt_alpha_cumprod_t = extract_into_tensor(self.sqrt_alphas_cumprod, t, in_tensor.shape)
                # Denominator uses the specific `1 - sqrt_alphas_cumprod` formulation.
                denominator = 1.0 - sqrt_alpha_cumprod_t + 1e-5  # Add epsilon for numerical stability
                snr = sqrt_alpha_cumprod_t / denominator

                # Compute a weight f(t) based on the SNR-like term.
                f_t = snr / (snr + 1.0)

                # Compute a standard clamped version of the input tensor.
                in_tensor_clamp = torch.clamp(in_tensor, -1.0, 1.0)

                # Combine the original prediction and the clamped version using the calculated weight f_t.
                in_tensor = f_t * in_tensor + (1.0 - f_t) * in_tensor_clamp
                # --- End Dynamic Clipping Logic ---
            else:
                # --- Standard Hard Clipping Logic ---
                in_tensor = torch.clamp(in_tensor, -1.0, 1.0)
                # --- End Standard Clipping Logic ---

        # If self.clip_denoised is False, the original in_tensor (potentially modified by denoised_fn) is returned.
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
        Apply the model to get p(x_{t-1} | x_t) and predict the model output.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param diff_steps: a 1-D Tensor of timesteps.
        :param mask: a [N x C x ...] tensor applied to x_t
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
        b, c = x.shape[0], x.shape[-1] if len(x.shape) > 1 else 1 # Handle potential 1D data
        assert diff_steps.shape == (b,)
        if emb is None:
            emb = {}
        emb["DiffusionStepEmbedder"] = self._scale_steps(diff_steps)
        model_output = model(x, emb=emb.copy(), mask=mask, **model_kwargs)

        # Reshape if necessary (e.g. if model outputs channels last)
        if model_output.shape != x.shape:
            model_output = model_output.view(x.shape)
        elif model_output.shape != x.shape and self.model_var_type not in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # If shapes don't match and we are not learning variance (which expects double channels)
             raise ValueError(f"Model output shape {model_output.shape} does not match input shape {x.shape}")


        # Calculate variance
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # Assuming the model outputs mean and variance parameters concatenated
            expected_shape = list(x.shape)
            expected_shape[1] = expected_shape[1] * 2 # Expect double channels/features
            assert model_output.shape == tuple(expected_shape), \
                   f"Expected shape {tuple(expected_shape)} for learned variance, got {model_output.shape}"

            # Split model output into mean prediction part and variance part
            model_output, model_var_values = torch.split(model_output, c, dim=1) # Split along channel dim

            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else: # ModelVarType.LEARNED_RANGE
                min_log = extract_into_tensor(
                    self.posterior_log_variance_clipped, diff_steps, x.shape
                )
                max_log = extract_into_tensor(np.log(self.betas), diff_steps, x.shape)
                # The model_var_values are assumed to be in [-1, 1] range
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            # Use pre-calculated fixed variance schedules
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial variance estimates to B.
                # at the same time, we need to make sure we don't include beta_0.
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

        # --- Calculate predicted x_start and model mean based on model_mean_type ---
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = self.process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=diff_steps, xprev=model_output), diff_steps, denoised_fn
            )
            model_mean = model_output
        elif self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = self.process_xstart(model_output, diff_steps, denoised_fn)
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, diff_steps=diff_steps
            )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = self.process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=diff_steps, eps=model_output), diff_steps, denoised_fn
            )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, diff_steps=diff_steps
            )
        elif self.model_mean_type == ModelMeanType.V_PREDICTION:
             # Model outputs v_pred
            v_pred = model_output
            pred_xstart = self.process_xstart(
                self._predict_xstart_from_v(x_t=x, t=diff_steps, v=v_pred), diff_steps, denoised_fn
            )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, diff_steps=diff_steps
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == x.shape
        assert pred_xstart.shape == x.shape
        assert model_variance.shape == x.shape
        assert model_log_variance.shape == x.shape

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """ eq. 15 """
        assert x_t.shape == eps.shape
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t: torch.Tensor, t: torch.Tensor, xprev: torch.Tensor) -> torch.Tensor:
        assert x_t.shape == xprev.shape
        # (xprev - coef2*x_t) / coef1
        return (
            extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    # <<< START NEW CODE >>>
    def _predict_xstart_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """ Predict x_0 from v = sqrt(alpha_bar) * eps - sqrt(1-alpha_bar) * x_0 """
        assert x_t.shape == v.shape
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def _predict_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """ Predict eps from v = sqrt(alpha_bar) * eps - sqrt(1-alpha_bar) * x_0 """
        assert x_t.shape == v.shape
        return (
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
            + extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v
        )

    def _predict_v_from_eps_and_xstart(self, eps: torch.Tensor, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ Compute the target v given the true epsilon and true x_0 """
        assert eps.shape == x_start.shape
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, eps.shape) * eps
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, eps.shape) * x_start
        )
    # <<< END NEW CODE >>>

    def predict_eps_from_xstart(self, x_t: torch.Tensor, t: torch.Tensor, pred_xstart: torch.Tensor) -> torch.Tensor:
        """ eq. 7 """
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

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
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        mask: torch.Tensor,
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
            x_start=x_start, x_t=x_t, diff_steps=diff_steps
        )
        out = self.p_mean_variance(
            model=model,
            x=x_t,
            diff_steps=diff_steps,
            mask=mask,
            emb=emb,
            **model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -continuous_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where(diff_steps == 0, decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

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

        :param model: the model to evaluate loss on.
        :param gt_data: the [N x C x ...] tensor of ground-truth data.
        :param diff_steps: a batch of timestep indices.
        :param mask: a [N x C x ...] tensor applied to x_t
        :param emb: an embedding dictionary
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the split-out normal noise.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 And a tensor of shape [N x C x ...] storing intermediate results.
        """
        if noise is None:
            noise = torch.randn_like(gt_data)
        x_t = self.q_sample(gt_data, diff_steps, noise=noise)

        # Apply mask if provided (useful for inpainting/imputation)
        if torch.is_tensor(mask):
            x_t = torch.where(~mask * self.unmask_existing, gt_data, x_t) # Keep unmasked areas as ground truth

        terms = {}

        if self.loss_type in {LossType.KL, LossType.RESCALED_KL}:
            # Calculate KL divergence based loss
            vb_output = self._vb_terms_bpd(
                model=model,
                x_start=gt_data,
                x_t=x_t,
                diff_steps=diff_steps,
                mask=mask,
                emb=emb,
                **model_kwargs
            )
            terms["loss"] = vb_output["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.diffusion_steps
            # Store pred_xstart for potential logging/analysis
            pred_xstart_for_results = vb_output["pred_xstart"]

        elif self.loss_type in {LossType.MSE, LossType.RESCALED_MSE}:
            # Prepare model inputs
            if emb is None:
                emb = {}
            emb["DiffusionStepEmbedder"] = self._scale_steps(diff_steps)
            model_output = model(x_t, emb=emb, mask=mask, **model_kwargs)

            # Handle learned variance if applicable
            if self.model_var_type in {ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE}:
                # Assuming model outputs mean and variance parts concatenated
                b, c = x_t.shape[0], x_t.shape[1] # Get channel dimension
                expected_shape = list(x_t.shape)
                expected_shape[1] = c * 2
                assert model_output.shape == tuple(expected_shape), \
                    f"Expected shape {tuple(expected_shape)} for learned variance, got {model_output.shape}"

                model_output_mean_part, model_var_values = torch.split(model_output, c, dim=1)

                # Calculate VB loss term for variance (using detached mean prediction)
                frozen_out = torch.cat([model_output_mean_part.detach(), model_var_values], dim=1)
                vb_output = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out, **kwargs: r, # Pass frozen output directly
                    x_start=gt_data,
                    x_t=x_t,
                    diff_steps=diff_steps,
                    mask=mask,
                    emb=emb,
                    **model_kwargs # Pass other kwargs through
                )
                terms["vb"] = vb_output["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Rescale VB term appropriately if using RESCALED_MSE
                    terms["vb"] *= self.diffusion_steps / 1000.0 # Scaling factor used in some implementations

                # The main model output for MSE loss is just the mean part
                model_output = model_output_mean_part

            # Determine the target for the MSE loss based on model_mean_type
            if self.model_mean_type == ModelMeanType.PREVIOUS_X:
                target = self.q_posterior_mean_variance(
                    x_start=gt_data, x_t=x_t, diff_steps=diff_steps
                )[0]
            elif self.model_mean_type == ModelMeanType.START_X:
                target = gt_data
            elif self.model_mean_type == ModelMeanType.EPSILON:
                target = noise
            elif self.model_mean_type == ModelMeanType.V_PREDICTION:
                 # Calculate the target velocity v
                 target = self._predict_v_from_eps_and_xstart(noise, gt_data, diff_steps)
            else:
                raise NotImplementedError(self.model_mean_type)

            target = target.view(model_output.shape)
            x_t = x_t.view(model_output.shape)

            # Calculate MSE loss, applying mask if provided
            if torch.is_tensor(mask):
                mask = mask.view(model_output.shape)
                squared_error = (torch.where(~mask * self.unmask_existing, torch.zeros_like(target), target - model_output)) ** 2
                terms["mse"] = mean_flat(squared_error) # Mean over non-batch dims
            else:
                 # Compute standard MSE loss over the whole input
                terms["mse"] = mean_flat((target - model_output) ** 2)

            # Combine MSE loss with VB term if variance is learned
            terms["loss"] = terms["mse"] + terms.get("vb", 0)

            # Store predicted x_start for results (needs to be calculated based on model output type)
            # <<< START MODIFIED CODE >>>
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart_for_results = self.process_xstart(model_output, diff_steps, None)
            elif self.model_mean_type == ModelMeanType.EPSILON:
                 pred_xstart_for_results = self.process_xstart(
                    self._predict_xstart_from_eps(x_t=x_t, t=diff_steps, eps=model_output), diff_steps, None
                 )
            elif self.model_mean_type == ModelMeanType.V_PREDICTION:
                 pred_xstart_for_results = self.process_xstart(
                     self._predict_xstart_from_v(x_t=x_t, t=diff_steps, v=model_output), diff_steps, None
                 )
            # Note: PREVIOUS_X case needs careful handling if results are needed,
            # as model_output is x_{t-1}, not directly related to x_0 for results tensor.
            # For simplicity, we might assume results are primarily used/meaningful
            # when predicting START_X, EPSILON, or V. If PREVIOUS_X results needed,
            # calculate pred_xstart from xprev:
            elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
                 pred_xstart_for_results = self.process_xstart(
                     self._predict_xstart_from_xprev(x_t=x_t, t=diff_steps, xprev=model_output), diff_steps, None
                 )
            # <<< END MODIFIED CODE >>>

        else:
            raise NotImplementedError(self.loss_type)

        # Return the loss terms and the predicted x_start (or equivalent)
        # The 'results' tensor now consistently holds the predicted x_start regardless of loss type
        # <<< START MODIFIED CODE >>>
        # Ensure results are assigned even if loss is KL based
        if 'pred_xstart_for_results' not in locals(): # Handle KL loss case where it wasn't calculated in MSE block
             if self.loss_type in {LossType.KL, LossType.RESCALED_KL}:
                 # Re-run p_mean_variance minimally if needed, or use the one from _vb_terms_bpd
                 # Using the one from _vb_terms_bpd is more efficient
                  pred_xstart_for_results = vb_output["pred_xstart"] # Already computed in _vb_terms_bpd
             else:
                 # Fallback or error if logic doesn't cover a case
                 raise RuntimeError("Could not determine intermediate results (pred_xstart)")

        if torch.is_tensor(mask):
            pred_xstart_for_results = torch.where(~mask * self.unmask_existing, x_t, pred_xstart_for_results)

        return terms, pred_xstart_for_results # Return loss dict and predicted x0


    def get_diffusion_steps(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """ samples diffusion steps """
        t, weights = self.diffusion_step_sampler.sample(batch_size)
        return t.to(device), weights.to(device)


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