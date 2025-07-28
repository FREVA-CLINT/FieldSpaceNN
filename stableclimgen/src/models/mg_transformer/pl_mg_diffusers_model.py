import torch
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from stableclimgen.src.models.mg_transformer.pl_mg_model import MGMultiLoss
from stableclimgen.src.modules.multi_grid.input_output import MG_Difference_Encoder
from .pl_mg_probabilistic import LightningProbabilisticModel
from ..mgno_transformer.pl_mgno_base_model import LightningMGNOBaseModel, check_empty
from ...modules.grids.grid_utils import decode_zooms


def get_repaint_paper_schedule(num_inference_steps: int, jump_length: int, jumps: int):
    """
    Creates a faithful RePaint schedule as described in the paper.
    """
    # This is a port of the original get_schedule_jump_paper()
    jump_n_sample = jumps

    jumps_dict = {}
    for j in range(0, num_inference_steps - jump_length, jump_length):
        jumps_dict[j] = jump_n_sample - 1

    t = num_inference_steps
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if jumps_dict.get(t, 0) > 0:
            jumps_dict[t] = jumps_dict[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)

    # The last step is -1 to signify the end, which we don't need for the loop
    # so we return the pairs of (current_t, next_t)
    return list(zip(ts[:-1], ts[1:]))


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
                 sampling_steps=None,
                 use_repaint=False,
                 repaint_jumps=10,
                 repaint_jump_length=10):

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
        self.use_repaint = use_repaint
        self.repaint_jumps = repaint_jumps
        self.repaint_jump_length = repaint_jump_length

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
        mask_zooms = mask
        image_zooms = {zoom: torch.randn_like(t_tensor) for zoom, t_tensor in target.items()}
        batch_size = next(iter(image_zooms.values())).shape[0]
        max_zoom = max(image_zooms.keys())

        for zoom in self.schedulers.keys():
            self.schedulers[zoom].set_timesteps(self.sampling_steps)

        # Use one scheduler's timesteps as the reference for the loop
        reference_timesteps = self.schedulers[max_zoom].timesteps

        if self.use_repaint:
            time_pairs = get_repaint_paper_schedule(self.sampling_steps, self.repaint_jump_length, self.repaint_jumps)
            time_pairs.append((0, -1))
            for i_last, i_cur in tqdm(time_pairs, desc="RePainting (Corrected)"):
                t_last = reference_timesteps[self.sampling_steps - i_last - 1]

                emb['DiffusionStepEmbedder'] = torch.full((batch_size,), t_last, device=self.device, dtype=torch.long)

                if i_cur < i_last:  # Reverse Step
                    noise_pred_zooms =  self.model(image_zooms.copy(), coords_input=coords_input, coords_output=coords_output,
                                          sample_dict=sample_dict,
                                          mask_zooms=mask, emb=emb)
                    predicted_sample_zooms = {}
                    for zoom in image_zooms.keys():
                        predicted_sample_zooms[zoom] = self.schedulers[zoom].step(noise_pred_zooms[zoom], t_last,
                                                                                  image_zooms[zoom]).prev_sample

                    t_next_tensor = torch.full((batch_size,), t_last, device=self.device, dtype=torch.long)
                    noised_ground_truth_zooms = {}
                    for zoom in target.keys():
                        noise = torch.randn_like(target[zoom])
                        noised_ground_truth_zooms[zoom] = self.schedulers[zoom].add_noise(target[zoom], noise,
                                                                                          t_next_tensor)

                    for zoom in image_zooms.keys():
                        image_zooms[zoom] = torch.where(~mask_zooms[zoom], noised_ground_truth_zooms[zoom],
                                                        predicted_sample_zooms[zoom])

                else:  # Forward Step
                    renoised_zooms = {}
                    for zoom in image_zooms.keys():
                        # Use t_next (the destination time) to get the correct beta
                        beta_t_next = self.schedulers[zoom].betas[self.sampling_steps - i_last - 1]
                        alpha_t_next = 1.0 - beta_t_next
                        noise = torch.randn_like(image_zooms[zoom])
                        renoised_zooms[zoom] = alpha_t_next.sqrt() * image_zooms[zoom] + (
                                    1.0 - alpha_t_next).sqrt() * noise
                    image_zooms = renoised_zooms

        else:
            # Use the timesteps from the scheduler with the max zoom for the main loop
            for i, t in enumerate(tqdm(self.schedulers[max_zoom].timesteps, desc="Inpainting")):
                timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                emb["DiffusionStepEmbedder"] = timesteps

                noise_pred_zooms = self.model(image_zooms.copy(), coords_input=coords_input, coords_output=coords_output,
                                          sample_dict=sample_dict,
                                          mask_zooms=mask, emb=emb)

                stepped_zooms = {}
                for zoom in image_zooms.keys():
                    stepped_zooms[zoom] = self.schedulers[zoom].step(noise_pred_zooms[zoom], t,
                                                                     image_zooms[zoom]).prev_sample

                if mask_zooms is not None and i < len(self.schedulers[max_zoom].timesteps) - 1:
                    noised_ground_truth_zooms = {}
                    for zoom in target.keys():
                        prev_timestep = self.schedulers[zoom].timesteps[i + 1]
                        noise = torch.randn_like(target[zoom])
                        noised_ground_truth_zooms[zoom] = self.schedulers[zoom].add_noise(target[zoom], noise,
                                                                                          prev_timestep)

                    for zoom in image_zooms.keys():
                        # Using ~mask to match your new convention (False = known region)
                        stepped_zooms[zoom] = torch.where(~mask_zooms[zoom], noised_ground_truth_zooms[zoom],
                                                          stepped_zooms[zoom])

                image_zooms = stepped_zooms

        outputs = decode_zooms(image_zooms, max_zoom)
        return outputs