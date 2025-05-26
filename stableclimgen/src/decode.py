import json
import os
import sys
import torch
from einops import rearrange
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)

def decode(timesteps, variables, region=-1, compression_factor=16, data_file="/work/bk1318/k204233/stableclimgen/evaluations/mgno_ngc/vae_{}compress_vonmises_crosstucker_unetlike/local_singlemap.zarr") -> None:
    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    config_path = '/work/bk1318/k204233/stableclimgen/snapshots/mgno_ngc/vae_16compress_vonmises_crosstucker_unetlike'
    with initialize_config_dir(config_dir=config_path, job_name="your_job"):
        cfg = compose(config_name="composed_config", strict=False)

    data_dict = {
        "test": {
            "source":
                {"files": [data_file.format(compression_factor)],
                 "variables": variables},
            "target": {"files": [None],
                       "variables": variables},
            "timesteps": timesteps
        }
    }

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

    cfg.dataloader.dataset.data_dict = data_dict
    cfg.dataloader.dataset.norm_dict = norm_dict
    cfg.dataloader.datamodule.num_workers = 1
    cfg.dataloader.datamodule.batch_size = 1
    cfg.dataloader.dataset.p_dropout = 0
    cfg.dataloader.dataset.random_p = False
    cfg.dataloader.dataset.in_nside = 256
    cfg.dataloader.dataset.search_radius = 4
    cfg.dataloader.dataset.nh_input = 3
    cfg.dataloader.dataset.coarsen_sample_level = 7
    cfg.dataloader.dataset.deterministic = True
    cfg.dataloader.dataset.n_sample_vars = 1
    cfg.dataloader.dataset.bottleneck_in_nside = 32
    cfg.dataloader.dataset.region = region
    cfg.ckpt_path = os.path.join(config_path, 'last.ckpt')
    cfg.trainer.accelerator = 'gpu'
    cfg.trainer.precision = 32
    cfg.model.mode = "decode"
    cfg.export_to_zarr = True

    # Initialize model and trainer
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.trainer)

    test_dataset = instantiate(cfg.dataloader.dataset,
                               data_dict=data_dict["test"],
                               variables_source=data_dict["test"]["source"]["variables"],
                               variables_target=data_dict["test"]["target"]["variables"])
    data_module = instantiate(cfg.dataloader.datamodule, dataset_test=test_dataset)

    predictions = trainer.predict(model=model, dataloaders=data_module.test_dataloader(), ckpt_path=cfg.ckpt_path)
    output = torch.cat([batch["output"] for batch in predictions], dim=0)
    output = rearrange(output, "(b2 b1) n t s ... -> b2 n t (b1 s) ... ",
                       b1=test_dataset.global_cells.shape[0]
                       if test_dataset.coarsen_lvl_single_map == -1
                       else test_dataset.global_cells.reshape(-1, 4 ** test_dataset.coarsen_lvl_single_map).shape[0])
    output = rearrange(output, "b n t s ... -> (b t) n s ... ")
    for k, var in enumerate(test_dataset.variables_target):
        output[..., k] = test_dataset.var_normalizers[var].denormalize(output[..., k])

    output_dict = dict(zip(test_dataset.variables_target, output.split(1, dim=3)))
    sys.stdout = save_stdout
    return output_dict