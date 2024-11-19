import omegaconf


def check_value(value, n_repeat):
    if not isinstance(value, list) and not isinstance(value, omegaconf.listconfig.ListConfig):
        value = [value]*n_repeat
    return value

def expand_tensor(tensor, dims=5, keep_dims=None):
    if keep_dims is None:
        keep_dims = []

    # Convert negative dimensions in keep_dims to positive ones based on the target shape length
    keep_dims = [(dim if dim >= 0 else dims + dim) for dim in keep_dims]

    while dims > len(tensor.shape):
        current_dim = len(tensor.shape)

        if current_dim - 1 in keep_dims:
            tensor = tensor[..., None]
        else:
            # Add the new dimension at the end of the current shape
            tensor = tensor[..., None, :]

    return tensor