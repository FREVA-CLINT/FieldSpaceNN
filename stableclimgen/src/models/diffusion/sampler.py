import torch

from .gaussian_diffusion import extract_into_tensor


class Sampler:
    def __init__(self, gaussian_diffusion):
        self.gaussian_diffusion = gaussian_diffusion

    def sample_loop(
            self,
            model,
            gt_data,
            mask_data,
            cond_data,
            coords,
            noise=None,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0
    ):
        """
                Generate samples from the model.

                :param model: the model module.
                :param shape: the shape of the samples, (N, C, H, W).
                :param noise: if specified, the noise from the encoder to sample.
                              Should be of the same shape as `shape`.
                :param denoised_fn: if not None, a function which applies to the
                    x_start prediction before it is used to sample.
                :param cond_fn: if not None, this is a gradient function that acts
                                similarly to the model.
                :param model_kwargs: if not None, a dict of extra keyword arguments to
                    pass to the model. This can be used for conditioning.
                :return: a non-differentiable batch of samples.
                """

        if noise is not None:
            x_0 = noise
        else:
            x_0 = torch.randn_like(gt_data).to(gt_data.device)

        final = None
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
            model,
            gt_data,
            x_0,
            mask_data,
            cond_data,
            coords,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            eta=0.0,
    ):
        x_0 = (1 - mask_data) * x_0 + mask_data * gt_data
        indices = list(range(self.gaussian_diffusion.diffusion_steps))[::-1]

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
                out["sample"] = (1 - mask_data) * out["sample"] + mask_data * gt_data
                yield out
                x_0 = out["sample"]

    def sample(
            self,
            model,
            x_t,
            cond_data,
            mask_data,
            coords,
            diffusion_steps,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0
    ):
        return


class DDPMSampler(Sampler):
    def __init__(self, gaussian_diffusion):
        super().__init__(gaussian_diffusion)

    def sample(
            self,
            model,
            x_t,
            cond_data,
            mask_data,
            coords,
            diffusion_steps,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x_t: the current tensor at x_{t-1}.
        :param diffusion_steps: the value of t, starting at 0 for the first diffusion step.
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
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
        nonzero_mask = (
            (diffusion_steps != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


class DDIMSampler(Sampler):
    def __init__(self, gaussian_diffusion):
        super().__init__(gaussian_diffusion)

    def sample(
            self,
            model,
            x_t,
            cond_data,
            mask_data,
            coords,
            diffusion_steps,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
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
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self.gaussian_diffusion.predict_eps_from_xstart(x_t, diffusion_steps, out["pred_xstart"])

        alpha_bar = extract_into_tensor(self.gaussian_diffusion.alphas_cumprod, diffusion_steps, x_t.shape)
        alpha_bar_prev = extract_into_tensor(self.gaussian_diffusion.alphas_cumprod_prev, diffusion_steps, x_t.shape)
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x_t)
        mean_pred = (
                out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (diffusion_steps != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
