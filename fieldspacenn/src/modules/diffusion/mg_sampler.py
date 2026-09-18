import torch
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from .mg_gaussian_diffusion import extract_into_tensor, MGGaussianDiffusion


class Sampler:
    def __init__(self, gaussian_diffusion: MGGaussianDiffusion) -> None:
        """
        Initialize the Sampler class with a Gaussian diffusion process.

        :param gaussian_diffusion: Gaussian diffusion object with necessary properties and methods.
        :return: None.
        """
        self.gaussian_diffusion: MGGaussianDiffusion = gaussian_diffusion

    def sample_loop(
        self,
        model: torch.nn.Module,
        input_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        denoised_fn: Optional[Callable] = None,
        cond_fn: Optional[Callable] = None,
        eta: float = 0.0,
        progress: bool = False,
        **model_kwargs: Any
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """
        Generate samples from the model.

        :param model: The model module used for sampling.
        :param input_groups: List of input data groups with zoom tensors of shape
            ``(b, v, t, n, d, f)``.
        :param mask_groups: Optional list of mask data groups applied to control data,
            with tensors of shape ``(b, v, t, n, d, f)`` or broadcastable to it.
        :param denoised_fn: Optional function applied to the predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param eta: Hyperparameter controlling noise level.
        :param progress: Add progress bar.
        :param model_kwargs: Extra arguments for the model, used for conditioning.
        :return: Final batch of non-differentiable samples as zoom tensors of
            shape ``(b, v, t, n, d, f)``.
        """
        x_t_groups = [self.gaussian_diffusion.generate_noise(g) if g else None for g in input_groups]

        first_valid_group = next((g for g in x_t_groups if g), None)
        if not first_valid_group:
            return x_t_groups

        device = first_valid_group[list(first_valid_group.keys())[0]].device

        final = None
        for sample in self.sample_loop_progressive(
            model,
            input_groups,
            x_t_groups,
            mask_groups=mask_groups,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            device=device,
            eta=eta,
            progress=progress,
            **model_kwargs
        ):
            final = sample

        return final["sample"]

    def sample_loop_progressive(
        self,
        model: torch.nn.Module,
        input_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        x_t_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        denoised_fn: Optional[Callable] = None,
        cond_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        eta: float = 0.0,
        progress: bool = False,
        **model_kwargs: Any
    ) -> Iterator[Dict[str, Any]]:
        """
        Perform a progressive sampling loop over diffusion steps.

        :param model: The model module used for sampling.
        :param input_groups: List of input data groups with zoom tensors of shape
            ``(b, v, t, n, d, f)``.
        :param x_t_groups: List of initial noise tensor groups with shape
            ``(b, v, t, n, d, f)``.
        :param mask_groups: Optional list of mask data groups for controlling data.
        :param denoised_fn: Optional function applied to predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param device: Device to perform sampling on (e.g., 'cuda' or 'cpu').
        :param eta: Hyperparameter controlling noise level.
        :param progress: Add progress bar.
        :param model_kwargs: Extra arguments for the model, used for conditioning.
        :return: A generator yielding the sample dictionary for each diffusion step.
        """
        if mask_groups is not None:
            for i, (mask_zooms, input_zooms, x_t_zooms) in enumerate(zip(mask_groups, input_groups, x_t_groups)):
                if mask_zooms and input_zooms and x_t_zooms:
                    # Preserve known inputs where masks are disabled.
                    x_t_groups[i] = {int(zoom): torch.where(~mask_zooms[zoom], input_zooms[zoom], x_t_zooms[zoom]) for zoom in x_t_zooms.keys()}

        indices = list(range(self.gaussian_diffusion.diffusion_steps))[::-1]  # Reverse the diffusion steps

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        first_valid_group = next((g for g in x_t_groups if g), None)
        batch_size = first_valid_group[list(first_valid_group.keys())[0]].shape[0]

        # Iterate over diffusion steps in reverse
        for i in indices:
            diffusion_steps = torch.tensor([i] * batch_size, device=device)
            with torch.no_grad():
                out = self.sample(
                    model,
                    x_t_groups,
                    diffusion_steps,
                    mask_groups=mask_groups,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    eta=eta,
                    **model_kwargs
                )
                # Reapply mask to the sample output
                if mask_groups is not None:
                    for j, (mask_zooms, input_zooms, sample_zooms) in enumerate(zip(mask_groups, input_groups, out["sample"])):
                        if mask_zooms and input_zooms and sample_zooms:
                            # Reapply masks to keep conditioning regions fixed.
                            out["sample"][j] = {int(zoom): torch.where(~mask_zooms[zoom], input_zooms[zoom], sample_zooms[zoom]) for zoom in input_zooms.keys()}
                yield out
                x_t_groups = out["sample"]

    def sample(
        self,
        model: torch.nn.Module,
        x_t_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        diffusion_steps: torch.Tensor,
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        denoised_fn: Optional[Callable] = None,
        cond_fn: Optional[Callable] = None,
        eta: float = 0.0,
        **model_kwargs: Any
    ) -> Dict[str, Any]:
        """
        Placeholder method for sampling at each timestep. Intended to be implemented by subclasses.

        :param model: The model to sample from.
        :param x_t_groups: The current list of tensor groups at x_t with shape
            ``(b, v, t, n, d, f)``.
        :param diffusion_steps: Current diffusion step indices of shape ``(b,)``.
        :param mask_groups: Optional list of mask data groups.
        :param denoised_fn: Optional function applied to predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param eta: Hyperparameter controlling noise level.
        :param model_kwargs: Extra arguments for the model.
        :return: A dictionary containing sample results.
        """
        return {}


class DDPMSampler(Sampler):
    def __init__(self, gaussian_diffusion: MGGaussianDiffusion) -> None:
        """
        Initialize the DDPM Sampler with a Gaussian diffusion process.

        :param gaussian_diffusion: Gaussian diffusion object with necessary properties and methods.
        :return: None.
        """
        super().__init__(gaussian_diffusion)

    def sample(
        self,
        model: torch.nn.Module,
        x_t_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        diffusion_steps: torch.Tensor,
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        denoised_fn: Optional[Callable] = None,
        cond_fn: Optional[Callable] = None,
        eta: float = 0.0,
        **model_kwargs: Any
    ) -> Dict[str, Any]:
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: The model to sample from.
        :param x_t_groups: The current list of tensor groups at x_t with shape
            ``(b, v, t, n, d, f)``.
        :param diffusion_steps: Current diffusion step indices of shape ``(b,)``.
        :param mask_groups: Optional list of mask data groups.
        :param denoised_fn: Optional function applied to predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param eta: Hyperparameter controlling noise level.
        :param model_kwargs: Extra arguments for the model.
        :return: A dictionary containing sample and predicted x_start tensors.
        """
        output_groups = self.gaussian_diffusion.p_mean_variance(
            model,
            x_t_groups,
            diffusion_steps,
            mask_groups=mask_groups,
            denoised_fn=denoised_fn,
            **model_kwargs
        )
        
        sample_groups = []
        for i, x_t_zooms in enumerate(x_t_groups):
            if not x_t_zooms:
                sample_groups.append(None)
                continue
            noise_zooms = self.gaussian_diffusion.generate_noise(x_t_zooms)
            # Only inject noise when diffusion_steps > 0.
            nonzero_mask_zooms = {int(zoom): (diffusion_steps != 0).float().view(-1, *([1] * (len(x_t_zooms[zoom].shape) - 1))) for zoom in x_t_zooms.keys()}
            sample_zooms = {int(zoom): output_groups["mean"][i][zoom] + nonzero_mask_zooms[zoom] * torch.exp(0.5 * output_groups["log_variance"][i][zoom]) * noise_zooms[zoom] for zoom in x_t_zooms.keys()}
            sample_groups.append(sample_zooms)

        return {"sample": sample_groups, "pred_xstart": output_groups["pred_xstart"]}


class DDIMSampler(Sampler):
    def __init__(self, gaussian_diffusion: MGGaussianDiffusion) -> None:
        """
        Initialize the DDIM Sampler with a Gaussian diffusion process.

        :param gaussian_diffusion: Gaussian diffusion object with necessary properties and methods.
        :return: None.
        """
        super().__init__(gaussian_diffusion)

    def sample(
        self,
        model: torch.nn.Module,
        x_t_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        diffusion_steps: torch.Tensor,
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        denoised_fn: Optional[Callable] = None,
        cond_fn: Optional[Callable] = None,
        eta: float = 0.0,
        **model_kwargs: Any
    ) -> Dict[str, Any]:
        """
        Sample x_{t-1} from the model using DDIM.

        :param model: The model to sample from.
        :param x_t_groups: The current list of tensor groups at x_t with shape
            ``(b, v, t, n, d, f)``.
        :param diffusion_steps: Current diffusion step indices of shape ``(b,)``.
        :param mask_groups: Optional list of mask data groups.
        :param denoised_fn: Optional function applied to predicted x_start.
        :param cond_fn: Optional gradient function acting similarly to the model.
        :param eta: Hyperparameter controlling noise level.
        :param model_kwargs: Extra arguments for the model.
        :return: A dictionary containing sample and predicted x_start tensors.
        """
        output_groups = self.gaussian_diffusion.p_mean_variance(
            model,
            x_t_groups,
            diffusion_steps,
            mask_groups=mask_groups,
            denoised_fn=denoised_fn,
            **model_kwargs
        )

        sample_groups = []
        for i, x_t_zooms in enumerate(x_t_groups):
            if not x_t_zooms:
                sample_groups.append(None)
                continue

            pred_xstart_zooms = output_groups["pred_xstart"][i]
            eps_zooms = self.gaussian_diffusion.predict_eps_from_xstart(x_t_zooms, diffusion_steps, pred_xstart_zooms)

            alpha_bar_zooms = {int(zoom): extract_into_tensor(self.gaussian_diffusion.alphas_cumprod_zooms[zoom], diffusion_steps, x_t_zooms[zoom].shape) for zoom in x_t_zooms.keys()}
            alpha_bar_prev_zooms = {int(zoom): extract_into_tensor(self.gaussian_diffusion.alphas_cumprod_prev_zooms[zoom], diffusion_steps, x_t_zooms[zoom].shape) for zoom in x_t_zooms.keys()}
            sigma_zooms = {int(zoom): eta * torch.sqrt((1 - alpha_bar_prev_zooms[zoom]) / (1 - alpha_bar_zooms[zoom])) * torch.sqrt(1 - alpha_bar_zooms[zoom] / alpha_bar_prev_zooms[zoom]) for zoom in x_t_zooms.keys()}

            noise_zooms = self.gaussian_diffusion.generate_noise(x_t_zooms)
            mean_pred_zooms = {int(zoom): pred_xstart_zooms[zoom] * torch.sqrt(alpha_bar_prev_zooms[zoom]) + torch.sqrt(1 - alpha_bar_prev_zooms[zoom] - sigma_zooms[zoom] ** 2) * eps_zooms[zoom] for zoom in x_t_zooms.keys()}
            nonzero_mask_zooms = {int(zoom): (diffusion_steps != 0).float().view(-1, *([1] * (len(x_t_zooms[zoom].shape) - 1))) for zoom in x_t_zooms.keys()}
            sample_zooms = {int(zoom): mean_pred_zooms[zoom] + nonzero_mask_zooms[zoom] * sigma_zooms[zoom] * noise_zooms[zoom] for zoom in x_t_zooms.keys()}
            sample_groups.append(sample_zooms)

        return {"sample": sample_groups, "pred_xstart": output_groups["pred_xstart"]}
