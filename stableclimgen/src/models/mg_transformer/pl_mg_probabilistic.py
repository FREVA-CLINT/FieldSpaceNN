import torch

import lightning.pytorch as pl

from stableclimgen.src.models.mg_transformer.pl_mg_model import merge_sampling_dicts


class LightningProbabilisticModel(pl.LightningModule):
    def __init__(self, n_samples=1, max_batchsize=-1):
        super().__init__()
        self.n_samples = n_samples
        self.max_batchsize = max_batchsize

    def predict_step(self, batch, batch_index):
        source, target, patch_index_zooms, mask, emb = batch

        max_zoom = max(target.keys())
        batch_size = target[max_zoom].shape[0]
        total_samples = batch_size * self.n_samples  # Total expanded batch size

        # Repeat each sample n_samples times
        source = {int(zoom): source[zoom].repeat_interleave(self.n_samples, dim=0) for zoom in source.keys()}
        target = {int(zoom): target[zoom].repeat_interleave(self.n_samples, dim=0) for zoom in target.keys()}
        mask = {int(zoom): mask[zoom].repeat_interleave(self.n_samples, dim=0) for zoom in mask.keys()}

        patch_index_zooms = {k: v.repeat_interleave(self.n_samples, dim=0) for k, v in patch_index_zooms.items()}

        emb['GroupEmbedder'] = emb['GroupEmbedder'].repeat_interleave(self.n_samples, dim=0)
        emb['DensityEmbedder'] = (mask.copy(), emb['DensityEmbedder'][1].repeat_interleave(self.n_samples, dim=0))
        emb['TimeEmbedder'] = {int(zoom): emb['TimeEmbedder'][zoom].repeat_interleave(self.n_samples, dim=0) for zoom in emb['TimeEmbedder'].keys()}

        output_zooms = {int(zoom): [] for zoom in target.keys()}
        max_batchsize = self.max_batchsize if self.max_batchsize != -1 else total_samples
        for start in range(0, total_samples, max_batchsize):
            end = min(start + max_batchsize, total_samples)

            # Slice dictionaries properly
            emb_chunk = {}
            emb_chunk['GroupEmbedder'] = emb['GroupEmbedder'][start:end]
            emb_chunk['DensityEmbedder'] = (
                {int(zoom): emb['DensityEmbedder'][0][zoom][start:end] for zoom in emb['DensityEmbedder'][0].keys()},
                emb['DensityEmbedder'][1][start:end])
            emb_chunk['TimeEmbedder'] = {int(zoom): emb['TimeEmbedder'][zoom][start:end] for zoom in emb['TimeEmbedder'].keys()}

            patch_index_zooms_chunk = {k: v[start:end] for k, v in patch_index_zooms.items()}
            
            output_chunk = self._predict_step(
                source={int(zoom): source[zoom][start:end] for zoom in source.keys()},
                target={int(zoom): target[zoom][start:end] for zoom in target.keys()},
                patch_index_zooms=patch_index_zooms_chunk,
                mask={int(zoom): mask[zoom][start:end] for zoom in mask.keys()},
                emb=emb_chunk
            )
            output_zooms = {int(zoom): output_zooms[zoom].append(output_chunk[zoom]) for zoom in output_chunk.keys()}

        # Concatenate all output chunks
        output_zooms = {int(zoom): torch.cat(output_zooms[zoom], dim=0) for zoom in output_zooms.keys()}
        output_zooms = {int(zoom): output_zooms[zoom].view(batch_size, self.n_samples, *output_zooms[zoom].shape[1:]) for zoom in output_zooms.keys()}

        return {"output": output_zooms, "mask": mask}