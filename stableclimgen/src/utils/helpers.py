import torch


def create_zero_tensor_like_shape(tensor, shape, keep_dims=None):
    if keep_dims is None:
        keep_dims = []

    # Ensure keep_dims is sorted for consistent behavior
    keep_dims = sorted(keep_dims)

    # Start with the target shape
    new_shape = list(shape)

    # Replace dimensions in new_shape based on keep_dims
    for dim in keep_dims:
        if dim < len(tensor.shape):
            new_shape[dim] = tensor.shape[dim]

    # Create a zero tensor with the modified shape
    zero_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)

    return zero_tensor


def expand_tensor_to_shape(tensor, shape, keep_dims=None):
    if keep_dims is None:
        keep_dims = []

    # Ensure keep_dims is sorted for consistent behavior
    keep_dims = sorted(keep_dims)

    while len(shape) > len(tensor.shape):
        current_dim = len(tensor.shape)

        if current_dim - 1 in keep_dims:
            tensor = tensor[..., None]
        else:
            # Add the new dimension at the end of the current shape
            tensor = tensor[..., None, :]

    return create_zero_tensor_like_shape(tensor, shape, keep_dims=keep_dims) + tensor