# FieldSpaceNN

This repository contains the code base for `witte2025field`:

```bibtex
@article{witte2025field,
  title={Field-Space Attention for Structure-Preserving Earth System Transformers},
  author={Witte, Maximilian and Meuer, Johannes and Pl{\'e}siat, {\'E}tienne and Kadow, Christopher},
  journal={arXiv preprint arXiv:2512.20350},
  year={2025}
}
```

`stableclimgen` was the former name of the repository and was later renamed to `FieldSpaceNN`. In this code snapshot, parts of the codebase and entrypoints still use the `stableclimgen` package path.

## Installation

```bash
git clone https://github.com/FREVA-CLINT/FieldSpaceNN.git
cd FieldSpaceNN
git checkout FST_initial_paper
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Configuration

Training and evaluation are configured with Hydra config files under `stableclimgen/configs/`.

- Main experiment configs such as `FST_MSViT_train.yaml` live in `stableclimgen/configs/`.
- Dataloader configs live in `stableclimgen/configs/dataloader/`.
- Trainer, logger, embedding, and grid configs live in their respective subdirectories under `stableclimgen/configs/`.
- The different model configs can be found in `stableclimgen/configs/model/`.

For example, `stableclimgen/configs/FST_MSViT_train.yaml` selects defaults for the trainer, logger, model, grid, data split, dataloader, and embedding setup.

## Running

The main training entrypoint is `stableclimgen/src/train.py`.

Example:

```bash
python -m stableclimgen.src.train \
  -cp stableclimgen/configs \
  -cn FST_MSViT_train
```

You can change the selected experiment by pointing `-cn` to another config in `stableclimgen/configs/`, and you can override Hydra values on the command line as needed.
