# FieldSpaceNN

FieldSpaceNN contains the code and configurations for **Field-Space Attention for Structure-Preserving Earth System Transformers**.

Field-Space Attention is designed for high-resolution Earth-system modelling with strong spatial compression. It decomposes each field into a fixed hierarchy of coarse values and fine-scale residuals, projects the resulting multiscale patches into tokens, and repeatedly returns to field space during Transformer processing. The repository contains the experiments from the article:

- near-surface temperature downscaling on HEALPix grids;
- six-hour multivariable ERA5 prediction with channel mixing or variable attention;
- single-scale Vision Transformer, CNN, and U-Net reference configurations;
- ablations of multiscale tokenization and field-update frequency.

The codebase also includes the Field-Space Autoencoder and experimental diffusion and flow-matching components.

## Installation

FieldSpaceNN requires Python 3.11 or newer. To install the paper branch in an isolated environment:

```bash
git clone --branch FST_initial_paper https://github.com/FREVA-CLINT/FieldSpaceNN.git
cd FieldSpaceNN
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Repository structure

- `fieldspacenn/src/models/` contains the Field-Space Transformer, autoencoder, CNN, U-Net, diffusion, and flow-matching models.
- `fieldspacenn/src/modules/field_space/` implements the multiscale field representation and Field-Space Attention.
- `fieldspacenn/src/data/` contains the datasets and Lightning data modules.
- `fieldspacenn/configs/` contains the Hydra experiment, model, data, trainer, and logging configurations.
- `fieldspacenn/src/train.py` is the main training entry point.
- `fieldspacenn/src/test.py` provides prediction and evaluation support.

## Configuration and training

Experiments are configured with [Hydra](https://hydra.cc/). Top-level experiment files live directly in `fieldspacenn/configs/`; reusable model, data, trainer, and logger configurations live in their corresponding subdirectories. Hydra overrides can select a different model, logger, batch size, accelerator, or data configuration without editing the YAML files.

Before starting a run, replace the example data and checkpoint paths in the selected configuration and adjust the trainer and logger for the local system. In particular, the HEALPix experiments require the configured zoom levels in the model, data loader, and input data to agree.

### Field-space models from the paper

The paper experiments use the following model families:

- `SPA_single_*` and `SPA_multi_*` are the single-scale and multiscale temperature-downscaling Transformers.
- `CM_single_*` and `CM_multi_*` are the channel-mixing prediction models.
- `VA_single_*` and `VA_multi_*` are the variable-attention prediction models.
- `CNN_*` and `UNET_*` are the convolutional reference models.

Use `era5_downscaling_7_train` or `era5_downscaling_357_train` for temperature downscaling, and `era5_prediction_7_train`, `era5_prediction_357_train`, or `era5_prediction_457_train` for six-hour ERA5 prediction. For example:

```bash
python -m fieldspacenn.src.train \
  --config-name era5_downscaling_357_train \
  model=SPA_multi_3_4_4 \
  data_zooms=default

python -m fieldspacenn.src.train \
  --config-name era5_prediction_357_train \
  model=VA_multi_2_16_16 \
  data_zooms=default \
  data_variables=default_2D
```

Swap the `model` override for another compatible member of the families above. The selected experiment configuration determines the task-specific data loader and trainer defaults.

### Autoencoder models

`mg_autoencoder_train` trains the Field-Space Autoencoder on HEALPix data. Its default model is `mg_autoencoder`; `mg_healpix_conv_ae` provides the convolutional multi-zoom alternative:

```bash
python -m fieldspacenn.src.train \
  --config-name mg_autoencoder_train \
  model=mg_autoencoder

python -m fieldspacenn.src.train \
  --config-name mg_autoencoder_train \
  model=mg_healpix_conv_ae
```

Both supplied multi-grid autoencoder model configurations use zooms `3`, `5`, and `6`; point the data configuration and loader at those same levels.

For regular latitude-longitude data, use the CNN variational autoencoder configuration. Update the NetCDF file lists in `fieldspacenn/configs/data_split/regular.yaml` first:

```bash
python -m fieldspacenn.src.train --config-name cnn_vae_train
```

The related deterministic CNN baseline can be trained with `--config-name cnn_train`.

### Flow matching

The multi-grid flow-matching model is defined by `model/mg_flowmatching.yaml`. It can be selected from the general multi-grid training configuration:

```bash
python -m fieldspacenn.src.train \
  --config-name mg_transformer_train \
  model=mg_flowmatching \
  run_name=mg_flowmatching \
  model.model.pretrained_block_ckpt_path=/path/to/checkpoint.ckpt
```


## References

If you use Field-Space Attention, please cite:

```bibtex
@article{witte2025field,
  title={Field-Space Attention for Structure-Preserving Earth System Transformers},
  author={Witte, Maximilian and Meuer, Johannes and Pl{\'e}siat, {\'E}tienne and Baehr, Johanna and Kadow, Christopher},
  journal={arXiv preprint arXiv:2512.20350},
  year={2025}
}
```

The Field-Space Autoencoder is described in:

```bibtex
@article{meuer2026field,
  title={Field-Space Autoencoder for Scalable Climate Emulators},
  author={Meuer, Johannes and Witte, Maximilian and Pl{\'e}siat, {\'E}ti{\'e}nne and Ludwig, Thomas and Kadow, Christopher},
  journal={arXiv preprint arXiv:2601.15102},
  year={2026}
}
```

## License

See [LICENSE](LICENSE).
