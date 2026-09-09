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


config_path = '/work/bk1318/k204233/stableclimgen/snapshots/newtests_tas/healconv_64comp/'
with initialize_config_dir(config_dir=config_path, job_name="your_job"):
    cfg = compose(config_name="composed_config", strict=False)
cfg.ckpt_path = os.path.join(config_path,'last.ckpt')
cfg.output_path = '{}/era5_latent.pt'.format(config_path.replace("snapshots", "evaluations"))
cfg.trainer.accelerator='gpu'
cfg.trainer.precision=32
cfg.trainer.devices=1
cfg.model.mode="encode"
cfg.dataloader.datamodule.num_workers=8


# In[3]:


# Initialize model and trainer
model = instantiate(cfg.model)
trainer = instantiate(cfg.trainer)


# In[4]:


if not os.path.exists(os.path.dirname(cfg.output_path)):
    os.makedirs(os.path.dirname(cfg.output_path))

cfg.data_split['test'].timesteps = ["0-31167"]#31167
test_dataset = instantiate(cfg.dataloader.dataset,
                           data_dict=cfg.data_split['test'])
data_module = instantiate(cfg.dataloader.datamodule, dataset_test=test_dataset, batch_size=32)


# In[ ]:


import time
if cfg.ckpt_path is not None:
    model = load_from_state_dict(model, cfg.ckpt_path, print_keys=True, device=model.device)[0]
start_time = time.time()
predictions = trainer.predict(model=model, dataloaders=data_module.test_dataloader())
end_time = time.time()

print(f"Predicted time: {end_time - start_time:.2f} seconds")


# In[ ]:


max_zoom = max(predictions[0]["output"][0].keys())
min_zoom = min(predictions[0]["output"][0].keys())
output_coarse = torch.cat([batch["output"][0][min_zoom] for batch in predictions], dim=0)
output_fine = torch.cat([batch["output"][0][max_zoom] for batch in predictions], dim=0)
torch.save(output_coarse, cfg.output_path.replace("latent", "latent_coarse"))
torch.save(output_fine, cfg.output_path.replace("latent", "latent_fine"))

