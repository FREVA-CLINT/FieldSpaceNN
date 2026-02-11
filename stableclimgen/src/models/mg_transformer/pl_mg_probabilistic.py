from typing import Any, Dict, Tuple

import torch

import lightning.pytorch as pl


class LightningProbabilisticModel(pl.LightningModule):
    """
    LightningModule mixin that expands batches for probabilistic sampling.
    """

    def __init__(self, n_samples: int = 1, max_batchsize: int = -1) -> None:
        """
        Initialize the probabilistic wrapper.

        :param n_samples: Number of samples per input in prediction.
        :param max_batchsize: Maximum expanded batch size per chunk (-1 for no limit).
        :return: None.
        """
        super().__init__()
        self.n_samples: int = n_samples
        self.max_batchsize: int = max_batchsize

    def predict_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_index: int,
    ) -> Dict[str, Any]:
        """
        Expand the batch for probabilistic sampling and call the model-specific prediction.

        :param batch: Tuple ``(source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms)``
            with tensors shaped ``(b, v, t, n, d, f)`` per zoom.
        :param batch_index: Index of the current batch.
        :return: Dictionary with output groups and masks.
        """
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        first_valid_target = next((g for g in target_groups if g), None)
        if not first_valid_target:
            return {"output": target_groups, "mask": mask_groups}

        batch_size = first_valid_target[max(first_valid_target.keys())].shape[0]
        total_samples = batch_size * self.n_samples  # Total expanded batch size

        # Repeat each sample n_samples times for each group.
        source_groups_rep = [{int(z): g[z].repeat_interleave(self.n_samples, dim=0) for z in g} if g else None for g in source_groups]
        target_groups_rep = [{int(z): g[z].repeat_interleave(self.n_samples, dim=0) for z in g} if g else None for g in target_groups]
        mask_groups_rep = [{int(z): g[z].repeat_interleave(self.n_samples, dim=0) for z in g} if g else None for g in mask_groups] if mask_groups else [None] * len(source_groups)
        patch_index_zooms_rep = {k: v.repeat_interleave(self.n_samples, dim=0) for k, v in patch_index_zooms.items()}
        
        emb_groups_rep = []
        if emb_groups:
            for emb in emb_groups:
                if emb:
                    emb_rep = {
                        'VariableEmbedder': emb['VariableEmbedder'].repeat_interleave(self.n_samples, dim=0),
                        'TimeEmbedder': {int(z): emb['TimeEmbedder'][z].repeat_interleave(self.n_samples, dim=0) for z in emb['TimeEmbedder']}
                    }
                    emb_groups_rep.append(emb_rep)
                else:
                    emb_groups_rep.append(None)

        output_groups_chunks = []
        max_batchsize = self.max_batchsize if self.max_batchsize != -1 else total_samples
        for start in range(0, total_samples, max_batchsize):
            end = min(start + max_batchsize, total_samples)

            # Slice all group structures for the current chunk
            source_chunk = [{int(z): g[z][start:end] for z in g} if g else None for g in source_groups_rep]
            target_chunk = [{int(z): g[z][start:end] for z in g} if g else None for g in target_groups_rep]
            mask_chunk = [{int(z): g[z][start:end] for z in g} if g else None for g in mask_groups_rep]
            patch_chunk = {k: v[start:end] for k, v in patch_index_zooms_rep.items()}
            emb_chunk = []
            if emb_groups_rep:
                for emb_rep in emb_groups_rep:
                    if emb_rep:
                        chunk = {'VariableEmbedder': emb_rep['VariableEmbedder'][start:end],
                                 'TimeEmbedder': {int(z): emb_rep['TimeEmbedder'][z][start:end] for z in emb_rep['TimeEmbedder']}}
                        emb_chunk.append(chunk)
                    else:
                        emb_chunk.append(None)

            output_chunk = self._predict_step(source_chunk, target_chunk, patch_chunk, mask_chunk, emb_chunk)
            output_groups_chunks.append(output_chunk)

        # Concatenate chunks for each group
        num_groups = len(source_groups)
        output_groups = [{} for _ in range(num_groups)]
        for i in range(num_groups):
            if source_groups[i] is None: continue
            # Concatenate all chunks for the i-th group
            group_chunks = [chunk[i] for chunk in output_groups_chunks if chunk and chunk[i]]
            if group_chunks:
                concatenated_group = {int(z): torch.cat([c[z] for c in group_chunks], dim=0) for z in group_chunks[0]}
                # Reshape to (batch_size, n_samples, ...)
                output_groups[i] = {int(z): v.view(batch_size, self.n_samples, *v.shape[1:]) for z, v in concatenated_group.items()}

        return {"output": output_groups, "mask": mask_groups}
