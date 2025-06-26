
import datetime

import torch
from einops import rearrange
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import healpy as hp
import numpy as np

config_path = '/container/da/genai_data/models/vae_16compress_vonmises_crosstucker_unetlike'
with initialize_config_dir(config_dir=config_path, job_name="your_job"):
    cfg = compose(config_name="config_frevagpt", strict=False)

norm_dict = {"tas": {"normalizer": {"class": "QuantileNormalizer", "quantile": 0.01},
                     "stats": {"quantiles": {"0.01": 234.30984497,
                                             "0.99": 305.74804688}}},
             "pr": {"normalizer": {"class": "QuantileNormalizer", "quantile": 0.01},
                    "stats": {"quantiles": {"0.01": 0.0, "0.99": 0.00055823}}},
             "pres_sfc": {"normalizer": {"class": "QuantileNormalizer", "quantile": 0.01},
                          "stats": {"quantiles": {"0.01": 64049.64453125,
                                                  "0.99": 103337.53125}}},
             "uas": {"normalizer": {"class": "QuantileNormalizer", "quantile": 0.01},
                     "stats": {"quantiles": {"0.01": -12.04504204,
                                             "0.99": 17.22920036}}},
             "vas": {"normalizer": {"class": "QuantileNormalizer", "quantile": 0.01},
                     "stats": {"quantiles": {"0.01": -12.20680428,
                                             "0.99": 11.26358509}}}}

cfg.dataloader.dataset.norm_dict = norm_dict
cfg.model.mode = "decode"

# Initialize model and trainer
model = instantiate(cfg.model)
trainer = instantiate(cfg.trainer)

START_DATE = datetime.datetime.strptime(cfg.start_date_str, "%Y-%m-%d").date()
END_DATE = datetime.datetime.strptime(cfg.end_date_str, "%Y-%m-%d").date()
MAX_INDEX = (END_DATE - START_DATE).days


def decode(timesteps, variables, lon=None, lat=None):
    timesteps = [date_to_index(ts) for ts in timesteps]

    data_dict = {
        "test": {
            "source":
                {"files": [cfg.data_file],
                 "variables": cfg.variables},
            "target": {"files": [None],
                       "variables": cfg.variables},
            "timesteps": timesteps
        }
    }

    cfg.dataloader.dataset.data_dict = data_dict

    if lon and lat:
        zoom_level = 1
        nside = 2 ** zoom_level

        theta = np.radians(90.0 - lat)
        phi = np.radians(lon)

        region = int(hp.ang2pix(nside, theta, phi, nest=True))
    else:
        region = -1

    cfg.dataloader.dataset.region = region
    test_dataset = instantiate(cfg.dataloader.dataset,
                               data_dict=data_dict["test"],
                               variables_source=data_dict["test"]["source"]["variables"],
                               variables_target=data_dict["test"]["target"]["variables"])
    data_module = instantiate(cfg.dataloader.datamodule, dataset_test=test_dataset)

    predictions = trainer.predict(model=model, dataloaders=data_module.test_dataloader(), ckpt_path=cfg.ckpt_path)
    output = torch.cat([batch["output"] for batch in predictions], dim=0)
    output = rearrange(output, "(b2 b1) n t s ... -> b2 n t (b1 s) ... ",
                       b1=1 if test_dataset.region != -1 else (test_dataset.global_cells.shape[0]
                       if test_dataset.coarsen_lvl_single_map == -1
                       else test_dataset.global_cells.reshape(-1, 4 ** test_dataset.coarsen_lvl_single_map).shape[0]))
    output = rearrange(output, "b n t s ... -> (b t) n s ... ")
    for k, var in enumerate(test_dataset.variables_target):
        output[:, :, :, k] = test_dataset.var_normalizers[var].denormalize(output[:, :, :, k])

    output_dict = dict(zip(test_dataset.variables_target, output.split(1, dim=3)))
    final_output_dict = {}
    for var in variables:
        final_output_dict[var] = output_dict[var]

    coords = None if not lon or not lat else test_dataset.__getitem__(0)[3].squeeze()
    return final_output_dict, coords


def date_to_index(date_str: str) -> int | None:
    """
    Converts a date string (YYYY-MM-DD) to its corresponding zero-based index.

    Args:
        date_str: The date string to convert.

    Returns:
        The integer index if the date is within the valid range, otherwise None.
    """
    try:
        current_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"Error: Invalid date format for '{date_str}'. Please use YYYY-MM-DD.")
        return None

    if not (START_DATE <= current_date <= END_DATE):
        print(f"Error: Date '{date_str}' is outside the valid range ({cfg.start_date_str} to {cfg.end_date_str}).")
        return None

    # Calculate the difference in days from the start date
    index = (current_date - START_DATE).days
    return index