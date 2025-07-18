import torch
from .mg_gaussian_diffusion import extract_into_tensor, GaussianDiffusion


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
        input_zooms: dict,
        mask_zooms: dict = None,
        denoised_fn: callable = None,
        cond_fn: callable = None,
        eta: float = 0.0,
        progress: bool = False,
        **model_kwargs
    ) -> torch.Tensor:
        """
        Generate samples from the model.

        :param model: The model module used for sampling.
        :param input_zooms: Input data tensor.
        :param mask_zooms: Mask data tensor, applied to control data.
        :param noise: Initial noise tensor. If None, random noise is used.
        :param denoised_fn: Optional function applied to the predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param eta: Hyperparameter controlling noise level.
        :param progress: Add progress bar.
        :param model_kwargs: Extra arguments for the model, used for conditioning.
        :return: Final batch of non-differentiable samples.
        """
        x_0_zooms = {int(zoom): torch.randn_like(input_zooms[zoom]) for zoom in input_zooms.keys()}

        final = None
        # Progressive sampling loop
        for sample in self.sample_loop_progressive(
            model,
            input_zooms,
            x_0_zooms,
            mask_zooms,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            device=x_0_zooms[list(x_0_zooms.keys())[0]].device,
            eta=eta,
            progress=progress,
            **model_kwargs
        ):
            final = sample

        return final["sample"]

    def sample_loop_progressive(
        self,
        model: torch.nn.Module,
        input_zooms: dict,
        x_0_zooms: dict,
        mask_zooms: dict = None,
        denoised_fn: callable = None,
        cond_fn: callable = None,
        device: torch.device = None,
        eta: float = 0.0,
        progress: bool = False,
        **model_kwargs
    ) -> dict:
        """
        Perform a progressive sampling loop over diffusion steps.

        :param model: The model module used for sampling.
        :param input_zooms: Input data tensor.
        :param x_0_zooms: Initial noise tensor.
        :param mask_zooms: Mask data tensor for controlling data.
        :param denoised_fn: Optional function applied to predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param device: Device to perform sampling on (e.g., 'cuda' or 'cpu').
        :param eta: Hyperparameter controlling noise level.
        :param progress: Add progress bar.
        :param model_kwargs: Extra arguments for the model, used for conditioning.
        :return: A generator yielding the sample dictionary for each diffusion step.
        """
        if mask_zooms is not None:
            x_0_zooms = {int(zoom): torch.where(~mask_zooms[zoom], torch.zeros_like(input_zooms[zoom]), x_0_zooms[zoom]) for zoom in x_0_zooms.keys()}

        indices = list(range(self.gaussian_diffusion.diffusion_steps))[::-1]  # Reverse the diffusion steps

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        # Iterate over diffusion steps in reverse
        for i in indices:
            diffusion_steps = torch.tensor([i] * x_0_zooms[list(x_0_zooms.keys())[0]].shape[0], device=device)
            with torch.no_grad():
                out = self.sample(
                    model,
                    x_0_zooms,
                    diffusion_steps,
                    mask_zooms,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    eta=eta,
                    **model_kwargs
                )
                # Reapply mask to the sample output
                out["sample"] = {int(zoom): torch.where(~mask_zooms[zoom], input_zooms[zoom], out["sample"][zoom]) for zoom in input_zooms.keys()} if mask_zooms is not None else out["sample"]
                yield out
                x_0_zooms = out["sample"]

    def sample(
        self,
        model: torch.nn.Module,
        x_t_zooms: dict,
        diffusion_steps: torch.Tensor,
        mask_zooms: dict = None,
        denoised_fn: callable = None,
        cond_fn: callable = None,
        eta: float = 0.0,
        **model_kwargs
    ) -> dict:
        """
        Placeholder method for sampling at each timestep. Intended to be implemented by subclasses.

        :param model: The model to sample from.
        :param x_t_zooms: The current tensor at x_t.
        :param diffusion_steps: Current diffusion step tensor.
        :param mask_zooms: Mask data tensor.
        :param denoised_fn: Optional function applied to predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param eta: Hyperparameter controlling noise level.
        :param model_kwargs: Extra arguments for the model.
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
        x_t_zooms: dict,
        diffusion_steps: torch.Tensor,
        mask_zooms: dict = None,
        denoised_fn: callable = None,
        cond_fn: callable = None,
        eta: float = 0.0,
        **model_kwargs
    ) -> dict:
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: The model to sample from.
        :param x_t_zooms: The current tensor at x_t.
        :param diffusion_steps: Current diffusion step tensor.
        :param mask_zooms: Mask data tensor.
        :param denoised_fn: Optional function applied to predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param eta: Hyperparameter controlling noise level.
        :param model_kwargs: Extra arguments for the model.
        :return: A dictionary containing sample and predicted x_start.
        """
        out_zooms = self.gaussian_diffusion.p_mean_variance(
            model,
            x_t_zooms.copy(),
            diffusion_steps,
            mask_zooms,
            denoised_fn=denoised_fn,
            **model_kwargs
        )
        noise_zooms = {int(zoom): torch.randn_like(x_t_zooms[zoom]) for zoom in x_t_zooms.keys()}
        nonzero_mask_zooms = {int(zoom): (diffusion_steps != 0).float().view(-1, *([1] * (len(x_t_zooms[zoom].shape) - 1))) for zoom in x_t_zooms.keys()}
        sample_zooms = {int(zoom): out_zooms["mean"][zoom] + nonzero_mask_zooms[zoom] * torch.exp(0.5 * out_zooms["log_variance"][zoom]) * noise_zooms[zoom] for zoom in x_t_zooms.keys()}
        return {"sample": sample_zooms, "pred_xstart": out_zooms["pred_xstart"]}


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
        x_t_zooms: dict,
        diffusion_steps: torch.Tensor,
        mask_zooms: dict = None,
        denoised_fn: callable = None,
        cond_fn: callable = None,
        eta: float = 0.0,
        **model_kwargs
    ) -> dict:
        """
        Sample x_{t-1} from the model using DDIM.

        :param model: The model to sample from.
        :param x_t_zooms: The current tensor at x_t.
        :param diffusion_steps: Current diffusion step tensor.
        :param mask_zooms: Mask data tensor.
        :param denoised_fn: Optional function applied to predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param eta: Hyperparameter controlling noise level.
        :param model_kwargs: Extra arguments for the model.
        :return: A dictionary containing sample and predicted x_start.
        """
        out_zooms = self.gaussian_diffusion.p_mean_variance(
            model,
            x_t_zooms,
            diffusion_steps,
            mask_zooms,
            denoised_fn=denoised_fn,
            **model_kwargs
        )

        eps_zooms = self.gaussian_diffusion.predict_eps_from_xstart(x_t_zooms, diffusion_steps, out_zooms["pred_xstart"])

        alpha_bar_zooms = {int(zoom): extract_into_tensor(self.gaussian_diffusion.alphas_cumprod, diffusion_steps, x_t_zooms.shape) for zoom in x_t_zooms.keys()}
        alpha_bar_prev_zooms = {int(zoom): extract_into_tensor(self.gaussian_diffusion.alphas_cumprod_prev, diffusion_steps, x_t_zooms.shape) for zoom in x_t_zooms.keys()}
        sigma_zooms = {int(zoom): eta * torch.sqrt((1 - alpha_bar_prev_zooms) / (1 - alpha_bar_zooms)) * torch.sqrt(1 - alpha_bar_zooms / alpha_bar_prev_zooms) for zoom in x_t_zooms.keys()}

        noise_zooms = {int(zoom): torch.randn_like(x_t_zooms[zoom]) for zoom in x_t_zooms.keys()}
        mean_pred_zooms = {int(zoom): out_zooms["pred_xstart"][zoom] * torch.sqrt(alpha_bar_prev_zooms[zoom]) + torch.sqrt(1 - alpha_bar_prev_zooms[zoom] - sigma_zooms[zoom] ** 2) * eps_zooms[zoom] for zoom in x_t_zooms.keys()}
        nonzero_mask_zooms = {int(zoom): (diffusion_steps != 0).float().view(-1, *([1] * (len(x_t_zooms[zoom].shape) - 1))) for zoom in x_t_zooms.keys()}
        sample_zooms = {int(zoom): mean_pred_zooms[zoom] + nonzero_mask_zooms[zoom] * sigma_zooms[zoom] * noise_zooms[zoom] for zoom in x_t_zooms.keys()}
        return {"sample": sample_zooms, "pred_xstart": out_zooms["pred_xstart"]}
