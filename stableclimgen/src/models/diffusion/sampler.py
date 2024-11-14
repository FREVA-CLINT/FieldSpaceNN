import torch
from .gaussian_diffusion import extract_into_tensor, GaussianDiffusion


class Sampler:
    def __init__(self, gaussian_diffusion: GaussianDiffusion):
        """
        Initialize the Sampler class with a Gaussian diffusion process.

        :param gaussian_diffusion: Gaussian diffusion object with necessary properties and methods.
        """
        self.gaussian_diffusion = gaussian_diffusion

    def sample_loop(
        self,
        model: torch.nn.Module,
        gt_data: torch.Tensor,
        mask_data: torch.Tensor,
        cond_data: torch.Tensor,
        coords: torch.Tensor,
        noise: torch.Tensor = None,
        denoised_fn: callable = None,
        cond_fn: callable = None,
        model_kwargs: dict = None,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Generate samples from the model.

        :param model: The model module used for sampling.
        :param gt_data: Ground truth data tensor.
        :param mask_data: Mask data tensor, applied to control data.
        :param cond_data: Conditioning data tensor for sampling.
        :param coords: Coordinate tensor for conditional sampling.
        :param noise: Initial noise tensor. If None, random noise is used.
        :param denoised_fn: Optional function applied to the predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param model_kwargs: Extra arguments for the model, used for conditioning.
        :param eta: Hyperparameter controlling noise level.
        :return: Final batch of non-differentiable samples.
        """
        x_0 = noise if noise is not None else torch.randn_like(gt_data).to(gt_data.device)

        final = None
        # Progressive sampling loop
        for sample in self.sample_loop_progressive(
            model,
            gt_data,
            x_0,
            mask_data,
            cond_data,
            coords,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=gt_data.device,
            eta=eta
        ):
            final = sample

        return final["sample"]

    def sample_loop_progressive(
        self,
        model: torch.nn.Module,
        gt_data: torch.Tensor,
        x_0: torch.Tensor,
        mask_data: torch.Tensor,
        cond_data: torch.Tensor,
        coords: torch.Tensor,
        denoised_fn: callable = None,
        cond_fn: callable = None,
        model_kwargs: dict = None,
        device: torch.device = None,
        eta: float = 0.0,
    ) -> dict:
        """
        Perform a progressive sampling loop over diffusion steps.

        :param model: The model module used for sampling.
        :param gt_data: Ground truth data tensor.
        :param x_0: Initial noise tensor.
        :param mask_data: Mask data tensor for controlling data.
        :param cond_data: Conditioning data tensor.
        :param coords: Coordinate tensor.
        :param denoised_fn: Optional function applied to predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param model_kwargs: Extra arguments for the model, used for conditioning.
        :param device: Device to perform sampling on (e.g., 'cuda' or 'cpu').
        :param eta: Hyperparameter controlling noise level.
        :return: A generator yielding the sample dictionary for each diffusion step.
        """
        x_0 = (1 - mask_data) * x_0 + mask_data * gt_data  # Apply initial mask to the noise
        indices = list(range(self.gaussian_diffusion.diffusion_steps))[::-1]  # Reverse the diffusion steps

        # Iterate over diffusion steps in reverse
        for i in indices:
            diffusion_steps = torch.tensor([i] * x_0.shape[0], device=device)
            with torch.no_grad():
                out = self.sample(
                    model,
                    x_0,
                    cond_data,
                    mask_data,
                    coords,
                    diffusion_steps,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                # Reapply mask to the sample output
                out["sample"] = (1 - mask_data) * out["sample"] + mask_data * gt_data
                yield out
                x_0 = out["sample"]

    def sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        cond_data: torch.Tensor,
        mask_data: torch.Tensor,
        coords: torch.Tensor,
        diffusion_steps: torch.Tensor,
        denoised_fn: callable = None,
        cond_fn: callable = None,
        model_kwargs: dict = None,
        eta: float = 0.0
    ) -> dict:
        """
        Placeholder method for sampling at each timestep. Intended to be implemented by subclasses.

        :param model: The model to sample from.
        :param x_t: The current tensor at x_t.
        :param cond_data: Conditioning data tensor.
        :param mask_data: Mask data tensor.
        :param coords: Coordinate tensor.
        :param diffusion_steps: Current diffusion step tensor.
        :param denoised_fn: Optional function applied to predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param model_kwargs: Extra arguments for the model.
        :param eta: Hyperparameter controlling noise level.
        :return: A dictionary containing sample results.
        """
        return {}


class DDPMSampler(Sampler):
    def __init__(self, gaussian_diffusion):
        """
        Initialize the DDPM Sampler with a Gaussian diffusion process.

        :param gaussian_diffusion: Gaussian diffusion object with necessary properties and methods.
        """
        super().__init__(gaussian_diffusion)

    def sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        cond_data: torch.Tensor,
        mask_data: torch.Tensor,
        coords: torch.Tensor,
        diffusion_steps: torch.Tensor,
        denoised_fn: callable = None,
        cond_fn: callable = None,
        model_kwargs: dict = None,
        eta: float = 0.0
    ) -> dict:
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: The model to sample from.
        :param x_t: The current tensor at x_t.
        :param cond_data: Conditioning data tensor.
        :param mask_data: Mask data tensor.
        :param coords: Coordinate tensor.
        :param diffusion_steps: Current diffusion step tensor.
        :param denoised_fn: Optional function applied to predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param model_kwargs: Extra arguments for the model.
        :param eta: Hyperparameter controlling noise level.
        :return: A dictionary containing sample and predicted x_start.
        """
        out = self.gaussian_diffusion.p_mean_variance(
            model,
            x_t,
            cond_data,
            mask_data,
            coords,
            diffusion_steps,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x_t)
        nonzero_mask = (diffusion_steps != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


class DDIMSampler(Sampler):
    def __init__(self, gaussian_diffusion):
        """
        Initialize the DDIM Sampler with a Gaussian diffusion process.

        :param gaussian_diffusion: Gaussian diffusion object with necessary properties and methods.
        """
        super().__init__(gaussian_diffusion)

    def sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        cond_data: torch.Tensor,
        mask_data: torch.Tensor,
        coords: torch.Tensor,
        diffusion_steps: torch.Tensor,
        denoised_fn: callable = None,
        cond_fn: callable = None,
        model_kwargs: dict = None,
        eta: float = 0.0
    ) -> dict:
        """
        Sample x_{t-1} from the model using DDIM.

        :param model: The model to sample from.
        :param x_t: The current tensor at x_t.
        :param cond_data: Conditioning data tensor.
        :param mask_data: Mask data tensor.
        :param coords: Coordinate tensor.
        :param diffusion_steps: Current diffusion step tensor.
        :param denoised_fn: Optional function applied to predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param model_kwargs: Extra arguments for the model.
        :param eta: Hyperparameter controlling noise level.
        :return: A dictionary containing sample and predicted x_start.
        """
        out = self.gaussian_diffusion.p_mean_variance(
            model,
            x_t,
            cond_data,
            mask_data,
            coords,
            diffusion_steps,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        eps = self.gaussian_diffusion.predict_eps_from_xstart(x_t, diffusion_steps, out["pred_xstart"])

        alpha_bar = extract_into_tensor(self.gaussian_diffusion.alphas_cumprod, diffusion_steps, x_t.shape)
        alpha_bar_prev = extract_into_tensor(self.gaussian_diffusion.alphas_cumprod_prev, diffusion_steps, x_t.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)

        noise = torch.randn_like(x_t)
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        nonzero_mask = (diffusion_steps != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
