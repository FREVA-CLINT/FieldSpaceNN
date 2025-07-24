import torch

import lightning.pytorch as pl

class LightningProbabilisticModel(pl.LightningModule):
    def __init__(self, n_samples=1, max_batchsize=-1):
        super().__init__()
        self.n_samples = n_samples
        self.max_batchsize = max_batchsize

    def predict_step(self, batch, batch_index):
        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch
        max_zoom = max(target.keys())
        batch_size = target[max_zoom].shape[0]
        total_samples = batch_size * self.n_samples  # Total expanded batch size

        # Repeat each sample n_samples times
        source = {int(zoom): source[zoom].repeat_interleave(self.n_samples, dim=0) for zoom in source.keys()}
        target = {int(zoom): target[zoom].repeat_interleave(self.n_samples, dim=0) for zoom in target.keys()}
        coords_input = coords_input.repeat_interleave(self.n_samples, dim=0)
        coords_output = coords_output.repeat_interleave(self.n_samples, dim=0)
        rel_dists_input = rel_dists_input.repeat_interleave(self.n_samples, dim=0)
        mask = {int(zoom): mask[zoom].repeat_interleave(self.n_samples, dim=0) for zoom in mask.keys()}

        indices = {k: v.repeat_interleave(self.n_samples, dim=0) for k, v in sample_dict.items()}
        emb = {k: (v.repeat_interleave(self.n_samples, dim=0) if torch.is_tensor(v)
        else ({ik: iv.repeat_interleave(self.n_samples, dim=0) for ik, iv in v[0].items()}, v[1].repeat_interleave(self.n_samples, dim=0))) for k, v in emb.items()}

        outputs = []
        max_batchsize = self.max_batchsize if self.max_batchsize != -1 else total_samples
        for start in range(0, total_samples, max_batchsize):
            end = min(start + max_batchsize, total_samples)

            # Slice dictionaries properly
            emb_chunk = {k: v[start:end] if torch.is_tensor(v) else ({ik: iv[start:end] for ik, iv in v[0].items()}, v[1][start:end]) for k, v in emb.items()}
            sample_dict_chunk = indices[start:end] if torch.is_tensor(indices) else {k: v[start:end] for k, v in indices.items()}
            
            output_chunk = self._predict_step(
                source={int(zoom): source[zoom][start:end] for zoom in source.keys()},
                target={int(zoom): target[zoom][start:end] for zoom in target.keys()},
                mask={int(zoom): mask[zoom][start:end] for zoom in mask.keys()},
                emb=emb_chunk,
                coords_input=coords_input[start:end],
                coords_output=coords_output[start:end],
                sample_dict=sample_dict_chunk,
                dists_input=rel_dists_input[start:end]
            )
            outputs.append(output_chunk)

        # Concatenate all output chunks
        output = torch.cat(outputs, dim=0)

        output = output.view(batch_size, self.n_samples, *output.shape[1:])
        return {"output": output, "mask": torch.zeros_like(output)}