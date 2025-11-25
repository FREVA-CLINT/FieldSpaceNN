import numpy as np
import torch
from typing import Union, List, Set, Dict
from .mg_gaussian_diffusion import GaussianDiffusion


def space_diffusion_steps(num_diffusion_steps: int, section_counts: Union[str, List[int]]) -> Set[int]:
    """
    Create a set of diffusion steps from an original diffusion process based on specified counts or
    a DDIM-inspired striding scheme.

    This function divides the total number of diffusion steps into sections, with each section having
    a specified count of diffusion steps. If a DDIM-style striding (specified by "ddimN") is used,
    it attempts to create an exact number of steps as specified.

    :param num_diffusion_steps: Total number of diffusion steps in the original process.
    :param section_counts: Either a list of integers or a string with comma-separated values that
                           specify the number of steps per section. Use "ddimN" to apply DDIM-inspired
                           striding with N steps.
    :return: A set of diffusion steps to use from the original process.
    """
    # Check if section_counts is a string and if it uses "ddim" striding
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])  # Extract step count for DDIM
            for i in range(1, num_diffusion_steps):
                if len(range(0, num_diffusion_steps, i)) == desired_count:
                    return set(range(0, num_diffusion_steps, i))
            raise ValueError(
                f"Cannot create exactly {num_diffusion_steps} steps with an integer stride"
            )
        elif section_counts.startswith("trailer"):
            desired_count = int(section_counts[len("trailer"):])  # Extract step count for DDIM
            for i in range(1, num_diffusion_steps):
                if len(range(0, num_diffusion_steps, i)) == desired_count:
                    return set(range(num_diffusion_steps-1, 0, -i)[::-1])
            raise ValueError(
                f"Cannot create exactly {num_diffusion_steps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_diffusion_steps // len(section_counts)  # Size of each section
    extra = num_diffusion_steps % len(section_counts)  # Extra steps if not divisible evenly
    start_idx = 0
    all_steps = []

    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)  # Calculate size for current section
        if size < section_count:
            raise ValueError(
                f"Cannot divide section of {size} steps into {section_count} steps"
            )

        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []

        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))  # Select rounded step
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size  # Update start index for next section

    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process class that enables step-skipping in the diffusion process.
    Allows for diffusion steps to be spaced out according to specified criteria.
    """

    def __init__(self, respacing: Union[None, str, List[int]] = None, diffusion_steps: int = 1000, **kwargs):
        """
        Initialize the SpacedDiffusion object with specified step spacing.

        :param respacing: Specifies how to space out diffusion steps. Accepts a string for DDIM striding
                          or a list for custom section counts.
        :param diffusion_steps: Total number of diffusion steps in the base process.
        """
        if respacing is None:
            respacing = [diffusion_steps]
        use_steps = space_diffusion_steps(diffusion_steps, respacing)

        self.use_steps = set(use_steps)  # Set of diffusion steps to use
        self.diffusion_step_map = []  # Mapping for diffusion steps
        self.original_num_steps = diffusion_steps

        base_diffusion = GaussianDiffusion(diffusion_steps=diffusion_steps, **kwargs)  # Initialize base GaussianDiffusion
        last_alpha_cumprod_zooms = {int(zoom): 1.0 for zoom in base_diffusion.alphas_cumprod_zooms.keys()}
        new_betas_zooms = {int(zoom): [] for zoom in base_diffusion.alphas_cumprod_zooms.keys()}

        # Calculate new betas for selected diffusion steps
        for k, zoom in enumerate(base_diffusion.alphas_cumprod_zooms.keys()):
            for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod_zooms[zoom]):
                if i in self.use_steps:
                    new_betas_zooms[zoom].append(1 - alpha_cumprod / last_alpha_cumprod_zooms[zoom])
                    last_alpha_cumprod_zooms[zoom] = alpha_cumprod
                    if k==0:
                        self.diffusion_step_map.append(i)
            new_betas_zooms[zoom] = np.array(new_betas_zooms[zoom])

        kwargs["betas_zooms"] = new_betas_zooms
        super().__init__(diffusion_steps=diffusion_steps, **kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        """
        Wrap the p_mean_variance method with a custom model for spaced diffusion steps.
        """
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        """
        Wrap the training_losses method with a custom model for spaced diffusion steps.
        """
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        """
        Wrap the given model with a diffusion step mapping.

        :param model: The model to wrap for diffusion.
        :return: Wrapped model with custom step mapping and rescaling.
        """
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.diffusion_step_map, self.rescale_steps, self.original_num_steps
        )

    def _scale_steps(self, t: torch.Tensor) -> torch.Tensor:
        """
        Rescale diffusion steps. Actual scaling is handled in the wrapped model.

        :param t: Original diffusion steps.
        :return: Scaled diffusion steps.
        """
        return t


class _WrappedModel:
    """
    A wrapper class for models in the spaced diffusion process. Rescales and remaps steps as needed.
    """

    def __init__(self, model, diffusion_step_map: List[int], rescale_steps: bool, original_num_steps: int):
        """
        Initialize the wrapped model with the given parameters.

        :param model: The original model to wrap.
        :param diffusion_step_map: List of diffusion steps mapped from the original process.
        :param rescale_steps: Whether to rescale steps.
        :param original_num_steps: The original number of diffusion steps.
        """
        self.model = model
        self.diffusion_step_map = diffusion_step_map
        self.rescale_steps = rescale_steps
        self.original_num_steps = original_num_steps

    def __call__(self, x: torch.Tensor, emb: Dict, mask_zooms: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Call the wrapped model with rescaled diffusion steps.

        :param x: Input tensor.
        :param emb: Embedding tensor.
        :param mask_zooms: Mask tensor for diffusion.
        :param cond: Conditional input tensor.
        :return: Output from the model with rescaled diffusion steps.
        """
        # Convert diffusion_step_map to a tensor on the correct device and type
        diffusion_steps = emb["DiffusionStepEmbedder"]
        map_tensor = torch.tensor(self.diffusion_step_map, device=diffusion_steps.device, dtype=diffusion_steps.dtype)
        new_ts = map_tensor[diffusion_steps]  # Map diffusion steps

        # Rescale steps if necessary
        if self.rescale_steps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        emb["DiffusionStepEmbedder"] = new_ts
        return self.model(x, emb=emb, mask_zooms=mask_zooms, **kwargs)
