#!/usr/bin/env python
# coding: utf-8

# In[1]:


import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import compose, initialize_config_dir
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from fieldspacenn.src.test import test
import xarray as xr
import json
import zarr
import healpy as hp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import healpy as hp
import numpy as np
import torch
from einops import rearrange
from fieldspacenn.src.utils.helpers import load_from_state_dict


# ### Mapping RCP26

# In[2]:


config_path = '/work/bk1318/k204233/stableclimgen/snapshots/newtests/mg_healconv_64comp_lvl6/'
with initialize_config_dir(config_dir=config_path, job_name="your_job"):
    cfg = compose(config_name="composed_config", strict=False)
cfg.ckpt_path = os.path.join(config_path,'last.ckpt')
cfg.output_path = '{}/era5_test_sr16x.pt'.format(config_path.replace("snapshots", "evaluations"))
cfg.trainer.accelerator='gpu'
cfg.trainer.precision=32
cfg.trainer.devices=1
cfg.dataloader.datamodule.num_workers=8


# In[3]:


# Initialize model and trainer
model = instantiate(cfg.model)
trainer = instantiate(cfg.trainer)


# In[ ]:


if not os.path.exists(os.path.dirname(cfg.output_path)):
    os.makedirs(os.path.dirname(cfg.output_path))

cfg.dataloader.dataset.p_dropout=0.0
cfg.dataloader.dataset.p_dropout_all=1.0
cfg.dataloader.dataset.sampling_zooms[3].p_drop=0.0
cfg.dataloader.dataset.sampling_zooms[6].p_drop=0.0
cfg.dataloader.dataset.sampling_zooms[7].p_drop=1.0
cfg.dataloader.dataset.sampling_zooms[8].p_drop=1.0

cfg.data_split['test'].timesteps = ["29703-31167"]#31167
test_dataset = instantiate(cfg.dataloader.dataset,
                           data_dict=cfg.data_split['test'])
data_module = instantiate(cfg.dataloader.datamodule, dataset_test=test_dataset, batch_size=1)


# In[6]:


import time
if cfg.ckpt_path is not None:
    model = load_from_state_dict(model, cfg.ckpt_path, print_keys=True, device=model.device)[0]
start_time = time.time()
predictions = trainer.predict(model=model, dataloaders=data_module.test_dataloader())
end_time = time.time()

print(f"Predicted time: {end_time - start_time:.2f} seconds")


# In[8]:


max_zoom = np.max(test_dataset.zooms).item()
sampling = test_dataset.sampling_zooms_collate or test_dataset.sampling_zooms.copy()
sampling = sampling[max_zoom]['zoom_patch_sample']
if sampling == -1:
    n_patches = 1
else:
    npix = hp.nside2npix(2 ** max_zoom)
    n_patches = npix // 4**(max_zoom - sampling)

target_variables = []
for grid_type, variables_grid_type in test_dataset.grid_types_vars.items():
    target_variables += list(set(variables_grid_type))
# Aggregate outputs from multiple devices
output = torch.cat([batch["output"][0][max_zoom] for batch in predictions], dim=0)
if output.dim() == 5:
    output = rearrange(output, "(b2 b1) v t n ... -> b2 v t (b1 n) ... ", b1=n_patches)
else:
    output = rearrange(output, "(b2 b1) m v t n ... -> b2 v m t (b1 n) ... ", b1=n_patches)

target_variables = []
for grid_type, variables_grid_type in test_dataset.grid_types_vars.items():
    target_variables += list(set(variables_grid_type))
print(target_variables)
target_variables=["tas", "uas", "vas", "ps", "pr"]
#target_variables=["tas"]
for k, var in enumerate(target_variables):
    output[:,k] = test_dataset.var_normalizers[max_zoom][var].denormalize(output[:, k])
output = dict(zip(target_variables, output.split(1, dim=1)))
torch.save(output, cfg.output_path)
