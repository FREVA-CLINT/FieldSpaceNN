# FieldSpaceNN

FieldSpaceNN is a research codebase for climate data generation and emulation across multi-grid spatial layouts. It includes multi-grid transformers, field-space attention modules, autoencoders, diffusion models, and CNN baselines, with flexible data loaders and visualization utilities.

**Key Methods**
- Field-Space Attention for structure-preserving Earth system transformers. (Witte et al., 2025)
- Field-Space Autoencoder for scalable climate emulation. (Meuer et al., 2026)
- Multi-grid transformer, autoencoder, and diffusion pipelines for HEALPix and regular grids.
- CNN and CNN-VAE baselines for fast experimentation.

**Repository Layout**
- `fieldspacenn/src/models/` model definitions (CNN, CNN-VAE, multi-grid transformer, autoencoder).
- `fieldspacenn/src/modules/` core modules (field-space attention, diffusion, CNN blocks, grids, embeddings).
- `fieldspacenn/src/data/` datasets and PyTorch Lightning datamodules.
- `fieldspacenn/src/utils/` logging, plotting, losses, normalizers, schedulers.
- `fieldspacenn/configs/` Hydra configs for models, data, trainer, and logging.
- `scripts/` example training scripts.

**Setup**
From the repo root (`/home/joe/PycharmProjects/fieldspacenn/src/fieldspacenn`):

```bash
pip install -e .
```

**Training (CLI Examples)**
The training entrypoint is `fieldspacenn/src/train.py` and uses Hydra configs in `fieldspacenn/configs/`.

Example based on `scripts/train_transformer.sh`:

```bash
python -m fieldspacenn.src.train \
  -cp fieldspacenn/configs \
  -cn mg_transformer_train \
  trainer.accelerator="cpu" \
  dataloader.datamodule.batch_size=1
```

Other training configs (matching scripts):

```bash
python -m fieldspacenn.src.train -cp fieldspacenn/configs -cn mg_autoencoder_train
python -m fieldspacenn.src.train -cp fieldspacenn/configs -cn mg_transformer_diffusion_train
python -m fieldspacenn.src.train -cp fieldspacenn/configs -cn cnn_train
python -m fieldspacenn.src.train -cp fieldspacenn/configs -cn cnn_vae_train
```

You can also use the provided scripts:

```bash
bash scripts/train_transformer.sh
bash scripts/train_autoencoder.sh
bash scripts/train_diffusion.sh
bash scripts/train_cnn.sh
bash scripts/train_cnn_vae.sh
```

**Validation / Inference (CLI Examples)**
For prediction-style validation, use `fieldspacenn/src/test.py` with a test config and checkpoint:

```bash
python -m fieldspacenn.src.test \
  -cp fieldspacenn/configs \
  -cn mg_transformer_test \
  ckpt_path=/path/to/checkpoint.ckpt \
  output_path=/path/to/output.pt \
  trainer.accelerator="cpu" \
  dataloader.datamodule.batch_size=1
```

During training, validation is performed automatically by Lightning based on the `trainer` config.

**Logging & Visualization**
- Validation images are written to `validation_images/` under the logger's save directory.
- HEALPix and regular-grid plots are supported in `fieldspacenn/src/utils/visualization.py`.

**References**

```bibtex
@article{witte2025field,
  title={Field-Space Attention for Structure-Preserving Earth System Transformers},
  author={Witte, Maximilian and Meuer, Johannes and Pl{\'e}siat, {\'E}tienne and Kadow, Christopher},
  journal={arXiv preprint arXiv:2512.20350},
  year={2025}
}

@article{meuer2026field,
  title={Field-Space Autoencoder for Scalable Climate Emulators},
  author={Meuer, Johannes and Witte, Maximilian and Pl{\'e}siat, {\'E}ti{\'e}nne and Ludwig, Thomas and Kadow, Christopher},
  journal={arXiv preprint arXiv:2601.15102},
  year={2026}
}
```
