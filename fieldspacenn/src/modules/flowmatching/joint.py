"""Trainer-independent joint sampling of monthly latent climate sequences."""

import torch

from ..embedding.temporal import monthly_time_inputs


@torch.no_grad()
def generate_joint(model, timestamps, forcings, future_length, initial_latents=None,
                   seed=0, sampling_steps=None, initial_states=None, encode_initial_states=None):
    """Generate one ensemble member per batch item, returning normalized latents.

    ``model`` is LightningMGFlowMatchingModel. Timestamps are Unix seconds (T,)
    or (B,T), covering history AND future. Forcings are already normalized using
    training statistics, with named tensors (B,T,profile); scalar CO2 may be
    (B,T). Initial latents have shape (B,V,K,12*4**zoom,1,F) at every input zoom.

    Physical inputs require an explicit frozen-AE preprocessing callback that
    returns latents normalized exactly like the training dataset. No physical
    decoding or multigrid composition is performed by this helper.
    """
    if not model.flow_matching.prefix_conditioning:
        raise ValueError("generate_joint requires a prefix-conditioned flow model")
    if isinstance(future_length, bool) or int(future_length) != future_length or future_length < 0:
        raise ValueError("future_length must be a nonnegative integer")
    future_length = int(future_length)
    if initial_states is not None:
        if initial_latents is not None or encode_initial_states is None:
            raise ValueError("Physical initial states require encode_initial_states and no initial_latents")
        initial_latents = encode_initial_states(initial_states)
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    calendar, positions = monthly_time_inputs(timestamps)
    calendar, positions = calendar.to(device), positions.to(device)
    batch, length = positions.shape
    block = model.model.get_block(0)
    if len(block.n_groups_variables) != 1:
        raise ValueError("generate_joint currently supports one climate variable group")
    variables = block.n_groups_variables[0]
    zooms = list(model.model.in_zooms)
    initial_latents = {} if initial_latents is None else initial_latents
    if initial_latents and set(initial_latents) != set(zooms):
        raise ValueError("Initial latents must include exactly the model input zooms")
    prefix = next(iter(initial_latents.values())).shape[2] if initial_latents else 0
    if prefix + future_length != length:
        raise ValueError("Timestamp length must equal initial history plus future_length")
    inputs, masks = {}, {}
    for zoom in zooms:
        shape = (batch, variables, length, 12 * 4**zoom, 1, model.model.in_features)
        inputs[zoom] = torch.zeros(shape, device=device, dtype=dtype)
        if initial_latents:
            known = initial_latents[zoom]
            expected = (*shape[:2], prefix, *shape[3:])
            if tuple(known.shape) != expected or known.dtype != dtype or not torch.isfinite(known).all():
                raise ValueError(f"Zoom {zoom}: initial latents must be finite {dtype} tensors of shape {expected}")
            inputs[zoom][:, :, :prefix] = known.to(device)
        masks[zoom] = torch.ones(shape, device=device, dtype=torch.bool)
        masks[zoom][:, :, :prefix] = False

    expected_names = next((set(m.forcing_names) for m in block.modules() if hasattr(m, "forcing_names")), set())
    if set(forcings) != expected_names:
        raise ValueError(f"Expected forcing names {sorted(expected_names)}, got {sorted(forcings)}")
    forcing_inputs = {}
    for name, values in forcings.items():
        values = torch.as_tensor(values, device=device, dtype=dtype)
        if values.ndim == 2:
            values = values.unsqueeze(-1)
        if values.ndim != 3 or tuple(values.shape[:2]) != (batch, length) or not values.shape[-1] or not torch.isfinite(values).all():
            raise ValueError(f"Forcing {name} must be finite with shape (B,T,profile)")
        forcing_inputs[name] = values
    if future_length == 0:
        return inputs

    variable_ids = torch.arange(variables, device=device).expand(batch, -1)
    embeddings = [{
        "TimeEmbedder": {z: calendar for z in zooms},
        "TemporalPosition": {z: positions for z in zooms},
        "ForcingEmbedder": {z: forcing_inputs for z in zooms},
        "VariableEmbedder": variable_ids,
        "MGEmbedder": variable_ids,
    }]
    configs = {z: {"n_past_ts": 0, "n_future_ts": length - 1,
                   "zoom_patch_sample": -1, "patch_index": torch.zeros(batch, device=device, dtype=torch.long)}
               for z in range(max(zooms) + 1)}
    generator = torch.Generator(device=device).manual_seed(int(seed))
    current = [{z: torch.randn(x.shape, device=device, dtype=dtype, generator=generator) for z, x in inputs.items()}]
    steps = model.sampling_steps_per_block if sampling_steps is None else int(sampling_steps)
    was_training = model.training
    model.eval()
    try:
        for i in range(model.model.n_blocks):
            current = model.sampler.sample_loop(
                model.model.get_block(i), input_groups=[inputs], x_t_groups=current,
                mask_groups=[masks], emb_groups=embeddings, sample_configs=configs,
                time_range=model.model.get_time_range(i, inference=True), n_steps=steps,
            )
    finally:
        model.train(was_training)
    return current[0]
