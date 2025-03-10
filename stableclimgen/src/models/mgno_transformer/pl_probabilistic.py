import torch

import lightning.pytorch as pl

class LightningProbabilisticModel(pl.LightningModule):
    def __init__(self, n_samples=1, max_batchsize=-1):
        super().__init__()
        self.n_samples = n_samples
        self.max_batchsize = max_batchsize

    def predict_step(self, batch, batch_index):
        source, target, coords_input, coords_output, indices, mask, emb = batch
        batch_size = source.shape[0]
        total_samples = batch_size * self.n_samples  # Total expanded batch size

        # Repeat each sample n_samples times
        source = source.repeat_interleave(self.n_samples, dim=0)
        coords_input = coords_input.repeat_interleave(self.n_samples, dim=0)
        coords_output = coords_output.repeat_interleave(self.n_samples, dim=0)
        mask = mask.repeat_interleave(self.n_samples, dim=0).unsqueeze(-1)

        indices = indices.repeat_interleave(self.n_samples, dim=0) if torch.is_tensor(indices) else {k: v.repeat_interleave(self.n_samples, dim=0) for k, v in indices.items()}
        emb = {k: v.repeat_interleave(self.n_samples, dim=0) for k, v in emb.items()}

        outputs = []
        max_batchsize = self.max_batchsize if self.max_batchsize != -1 else total_samples
        for start in range(0, total_samples, max_batchsize):
            end = min(start + max_batchsize, total_samples)

            # Slice dictionaries properly
            emb_chunk = {k: v[start:end] for k, v in emb.items()}
            indices_chunk = indices[start:end] if torch.is_tensor(indices) else {k: v[start:end] for k, v in indices.items()}

            output_chunk = self._predict_step(
                source=source[start:end],
                mask=mask[start:end],
                emb=emb_chunk,
                coords_input=coords_input[start:end],
                coords_output=coords_output[start:end],
                indices_sample=indices_chunk,
            )
            outputs.append(output_chunk)

        # Concatenate all output chunks
        output = torch.cat(outputs, dim=0)

        output = output.view(batch_size, self.n_samples, *output.shape[1:])
        mask = mask.view(batch_size, self.n_samples, *mask.shape[1:])
        return {"output": output, "mask": mask}