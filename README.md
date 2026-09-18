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

## Configuration

Experiments are configured with [Hydra](https://hydra.cc/). The top-level configurations in `fieldspacenn/configs/` cover temperature downscaling and six-hour prediction. Model configurations are grouped by the terminology used in the article:

- `SPA_single_*` and `SPA_multi_*`: single-scale and multiscale temperature-downscaling Transformers;
- `CM_single_*` and `CM_multi_*`: channel-mixing prediction models;
- `VA_single_*` and `VA_multi_*`: variable-attention prediction models;
- `CNN_*` and `UNET_*`: convolutional reference models.

Before training, point `fieldspacenn/configs/data_zooms/default.yaml` to the required preprocessed ERA5 Zarr data and adjust the trainer and logger settings for the local system.

A run is launched with a top-level experiment configuration, for example:

```bash
python -m fieldspacenn.src.train \
  --config-name era5_downscaling_357_train \
  model=SPA_multi_3_4_4 \
  data_zooms=default
```

Hydra overrides can be supplied on the command line to select another model, logger, batch size, accelerator, or data configuration.

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
