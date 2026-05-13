from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from .mg_flow_matching import MGFlowMatching


class EulerFlowSampler:
    """
    Deterministic Euler sampler for continuous flow-matching models.
    """

    def __init__(self, flow_matching: MGFlowMatching) -> None:
        """
        Initialize the sampler.

        :param flow_matching: Flow-matching process helper.
        :return: None.
        """
        self.flow_matching: MGFlowMatching = flow_matching

    @staticmethod
    def _known_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype == torch.bool:
            return ~mask
        return mask <= 0

    def _apply_known_values(
        self,
        sample_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        input_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """
        Preserve known input values according to the repository mask convention.
        """
        if mask_groups is None:
            return list(sample_groups)

        output_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        for mask_zooms, input_zooms, sample_zooms in zip(mask_groups, input_groups, sample_groups):
            if mask_zooms and input_zooms and sample_zooms:
                output_groups.append({
                    int(zoom): torch.where(
                        self._known_mask(mask_zooms[zoom]),
                        input_zooms[zoom],
                        sample_zooms[zoom],
                    )
                    for zoom in sample_zooms.keys()
                })
            else:
                output_groups.append(sample_zooms)
        return output_groups

    def sample_loop(
        self,
        model: torch.nn.Module,
        input_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        x_t_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        sample_configs: Optional[Dict[int, Any]] = None,
        time_range: Tuple[float, float] = (0.0, 1.0),
        n_steps: int = 50,
        progress: bool = False,
        **model_kwargs: Any,
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """
        Generate samples by integrating predicted velocity.

        :param model: Velocity model.
        :param input_groups: Input groups used for shape and known-value preservation.
        :param x_t_groups: Optional initial state. If omitted, Gaussian noise is used.
        :param mask_groups: Optional mask groups.
        :param emb_groups: Optional embedding groups.
        :param sample_configs: Sampling configuration forwarded to the model.
        :param time_range: Start/end integration times.
        :param n_steps: Number of Euler steps.
        :param progress: Whether to show a progress bar.
        :param model_kwargs: Additional model keyword arguments.
        :return: Final sampled groups.
        """
        final = None
        for sample in self.sample_loop_progressive(
            model=model,
            input_groups=input_groups,
            x_t_groups=x_t_groups,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
            time_range=time_range,
            n_steps=n_steps,
            progress=progress,
            **model_kwargs,
        ):
            final = sample

        if final is None:
            return list(x_t_groups) if x_t_groups is not None else [None] * len(input_groups)
        return final["sample"]

    def sample_loop_progressive(
        self,
        model: torch.nn.Module,
        input_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        x_t_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        sample_configs: Optional[Dict[int, Any]] = None,
        time_range: Tuple[float, float] = (0.0, 1.0),
        n_steps: int = 50,
        progress: bool = False,
        **model_kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield Euler integration states.
        """
        if n_steps <= 0:
            raise ValueError(f"`n_steps` must be > 0, got {n_steps}.")

        start, end = float(time_range[0]), float(time_range[1])
        if start < 0.0 or end > 1.0 or start > end:
            raise ValueError(f"`time_range` must satisfy 0 <= start <= end <= 1, got {time_range}.")

        if x_t_groups is None:
            x_t_groups = [self.flow_matching.generate_noise(group) if group else None for group in input_groups]
        x_t_groups = self._apply_known_values(x_t_groups, input_groups, mask_groups)

        first_valid_group = next((group for group in x_t_groups if group), None)
        if not first_valid_group:
            return

        first_tensor = next(iter(first_valid_group.values()))
        batch_size = first_tensor.shape[0]
        device = first_tensor.device
        times = torch.linspace(start, end, n_steps + 1, device=device)
        step_indices = range(n_steps)

        if progress:
            from tqdm.auto import tqdm
            step_indices = tqdm(step_indices)

        model_kwargs = dict(model_kwargs)
        if sample_configs is not None:
            model_kwargs["sample_configs"] = sample_configs

        for step_idx in step_indices:
            t = times[step_idx]
            dt = times[step_idx + 1] - t
            t_batch = torch.full((batch_size,), float(t.item()), device=device, dtype=first_tensor.dtype)

            with torch.no_grad():
                velocity_groups = self.flow_matching.model_velocity(
                    model,
                    x_t_groups,
                    t_batch,
                    mask_groups=mask_groups,
                    emb_groups=emb_groups,
                    **model_kwargs,
                )
                sample_groups: List[Optional[Dict[int, torch.Tensor]]] = []
                for x_t_zooms, velocity_zooms in zip(x_t_groups, velocity_groups):
                    if not x_t_zooms or not velocity_zooms:
                        sample_groups.append(None)
                        continue
                    sample_groups.append({
                        int(zoom): x_t_zooms[zoom] + dt.to(dtype=x_t_zooms[zoom].dtype) * velocity_zooms[zoom]
                        for zoom in x_t_zooms.keys()
                    })

                sample_groups = self._apply_known_values(sample_groups, input_groups, mask_groups)
                out = {"sample": sample_groups, "velocity": velocity_groups, "time": t_batch}
                yield out
                x_t_groups = sample_groups
