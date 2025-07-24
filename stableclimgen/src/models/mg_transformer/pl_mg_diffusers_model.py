import torch
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from stableclimgen.src.models.mg_transformer.pl_mg_model import MGMultiLoss
from stableclimgen.src.modules.multi_grid.input_output import MG_Difference_Encoder
from .pl_mg_probabilistic import LightningProbabilisticModel
from ..mgno_transformer.pl_mgno_base_model import LightningMGNOBaseModel, check_empty
from ...modules.grids.grid_utils import decode_zooms


class Lightning_MG_diffusion_transformer(LightningMGNOBaseModel, LightningProbabilisticModel):
    def __init__(self,
                 model,
                 lr_groups,
                 lambda_loss_dict,
                 schedulers,
                 weight_decay=0,
                 noise_std=0.0,
                 composed_loss=True,
                 n_samples=1,
                 max_batchsize=-1,
                 mg_encoder_config=None,
                 sampling_steps=None):

        super().__init__(
            model,
            lr_groups,
            {},
            weight_decay=weight_decay,
            noise_std=noise_std
        )

        self.loss = MGMultiLoss(lambda_loss_dict, grid_layers=model.grid_layers, max_zoom=max(model.in_zooms))
        self.composed_loss = composed_loss
        self.model = model
        self.n_samples = n_samples
        self.max_batchsize = max_batchsize
        self.schedulers = schedulers
        self.sampling_steps = sampling_steps

        if mg_encoder_config:
            self.encoder = MG_Difference_Encoder(out_zooms=mg_encoder_config.out_zooms)
        else:
            self.encoder = None

    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch
        timesteps = torch.randint(0, self.schedulers[list(target.keys())[0]].config.num_train_timesteps,
                                  (target[list(target.keys())[0]].shape[0],),
                                  device=self.device).long()

        noise_zooms = {int(zoom): torch.where(~mask[zoom], torch.zeros_like(target[zoom]), torch.randn_like(target[zoom])) for zoom in target.keys()}
        noisy_target_zooms = {zoom: self.schedulers[zoom].add_noise(target[zoom], noise_zooms[zoom], timesteps) for zoom in target.keys()}

        emb["DiffusionStepEmbedder"] = timesteps
        model_pred_zooms = self(noisy_target_zooms, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict,
                                mask=mask, emb=emb, dists_input=rel_dists_input)
        loss, loss_dict = self.loss(model_pred_zooms, noise_zooms, mask=mask, sample_dict=sample_dict, prefix='train/')

        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch
        timesteps = torch.randint(0, self.schedulers[list(target.keys())[0]].config.num_train_timesteps,
                                  (target[list(target.keys())[0]].shape[0],),
                                  device=self.device).long()
        coords_input, coords_output, mask, rel_dists_input = check_empty(coords_input), check_empty(coords_output), check_empty(mask), check_empty(rel_dists_input)
        sample_dict = self.prepare_sample_dict(sample_dict)

        noise_zooms = {int(zoom): torch.where(~mask[zoom], torch.zeros_like(target[zoom]), torch.randn_like(target[zoom])) for zoom in target.keys()}
        noisy_target_zooms = {zoom: self.schedulers[zoom].add_noise(target[zoom], noise_zooms[zoom], timesteps) for zoom in
                              target.keys()}

        emb["DiffusionStepEmbedder"] = timesteps
        model_pred_zooms = self(noisy_target_zooms.copy(), coords_input=coords_input, coords_output=coords_output,
                                sample_dict=sample_dict,
                                mask=mask, emb=emb, dists_input=rel_dists_input)
        loss, loss_dict = self.loss(model_pred_zooms, noise_zooms, mask=mask, sample_dict=sample_dict, prefix='val/')

        self.log_dict({"val/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        max_zoom = max(target.keys())
        if batch_idx == 0 and rank_zero_only and (source[max_zoom].device in ['cuda:0','cpu','mps']):
            # We can get the predicted original sample from the first step's output
            pred_xstart = {}
            for zoom in noisy_target_zooms.keys():
                pred_zoom = self.schedulers[zoom].step(
                    model_pred_zooms[zoom][0:1],  # Slice to get a single sample
                    timesteps[0].item(),  # Get the integer timestep
                    noisy_target_zooms[zoom][0:1]  # Slice to get a single sample
                ).pred_original_sample
                pred_xstart[zoom] = pred_zoom.unsqueeze(0)
            source_p = decode_zooms(source, max_zoom)
            output_p = decode_zooms(pred_xstart, max_zoom)
            target_p = decode_zooms(target, max_zoom)
            mask = mask[max_zoom] if mask is not None else None
            self.log_tensor_plot(source_p, output_p, target_p, coords_input, coords_output, mask, sample_dict,
                                 f"tensor_plot_{int(self.current_epoch)}_diff{timesteps[0].item()}", emb)

        return loss

    def predict_step(self, batch, batch_idx):
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source, target, mask, emb, coords_input, coords_output, sample_dict, dists_input):
        image_zooms = {
            int(zoom): torch.where(~mask[zoom], torch.zeros_like(target[zoom]), torch.randn_like(target[zoom])) for zoom
            in target.keys()}

        # Set timesteps for the sampling loop
        for zoom in image_zooms.keys():
            self.schedulers[zoom].set_timesteps(self.schedulers[zoom].config.num_train_timesteps if self.sampling_steps is None else self.sampling_steps)

        for i, t in enumerate(tqdm(self.schedulers[zoom].timesteps, desc="Sampling")):
            timesteps = torch.full((target[list(target.keys())[0]].shape[0],), t, device=self.device, dtype=torch.long)
            emb["DiffusionStepEmbedder"] = timesteps
            noise_pred_zooms = self.model(image_zooms.copy(), coords_input=coords_input, coords_output=coords_output,
                                          sample_dict=sample_dict,
                                          mask_zooms=mask, emb=emb)

            # Compute the previous noisy sample x_{t-1}
            stepped_zooms = {}
            for zoom in image_zooms.keys():
                stepped_zooms[zoom] = self.schedulers[zoom].step(noise_pred_zooms[zoom], t, image_zooms[zoom]).prev_sample

            if i < len(self.schedulers[zoom].timesteps) - 1:
                # 1. Noise the ground truth to the `prev_timestep` level
                noised_ground_truth_zooms = {}
                for zoom in target.keys():
                    prev_timestep = self.schedulers[zoom].timesteps[i + 1]
                    noise = torch.randn_like(target[zoom])
                    noised_ground_truth_zooms[zoom] = self.schedulers[zoom].add_noise(target[zoom], noise, prev_timestep)

                # 2. Replace the known areas in the model's prediction with the noised ground truth
                for zoom in image_zooms.keys():
                    stepped_zooms[zoom] = torch.where(
                        ~mask[zoom],
                        noised_ground_truth_zooms[zoom],
                        stepped_zooms[zoom]
                    )

            image_zooms = stepped_zooms

        outputs = decode_zooms(image_zooms, max(image_zooms.keys()))
        return outputs