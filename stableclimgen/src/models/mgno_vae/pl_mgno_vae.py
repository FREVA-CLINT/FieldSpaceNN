import torch
import torch.nn as nn

from ..mgno_transformer.pl_mgno_base_model import LightningMGNOBaseModel
from ..mgno_transformer.pl_mgno_base_model import check_empty
from ..mgno_transformer.pl_probabilistic import LightningProbabilisticModel
from ...modules.grids.grid_layer import Interpolator
from pytorch_lightning.utilities import rank_zero_only

class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target):
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss


class Lightning_MGNO_VAE(LightningMGNOBaseModel, LightningProbabilisticModel):
    def __init__(self, model, lr_groups, kl_weight: float = 1e-6, n_samples=1, max_batchsize=-1, residual_interpolator_settings=None, **base_model_args):
        super().__init__(model, lr_groups, **base_model_args)
        # maybe create multi_grid structure here?
        self.model = model

        self.lr_groups = lr_groups
        self.kl_weight = kl_weight

        self.n_samples = n_samples
        self.max_batchsize = max_batchsize

        if residual_interpolator_settings:
            self.residual_interpolator_down = Interpolator(self.model.grid_layers,
                             residual_interpolator_settings.get("search_zoom_rel", 3),
                             0,
                             residual_interpolator_settings.get("bottleneck_level", 3),
                             residual_interpolator_settings.get("precompute", True),
                             residual_interpolator_settings.get("nh_inter", 3),
                             residual_interpolator_settings.get("power", 1),
                             residual_interpolator_settings.get("cutoff_dist_zoom", None),
                             residual_interpolator_settings.get("cutoff_dist", None),
                             residual_interpolator_settings.get("search_zoom_compute", None)
                             )
            self.residual_interpolator_up = Interpolator(self.model.grid_layers,
                             residual_interpolator_settings.get("search_zoom_rel", 3),
                             residual_interpolator_settings.get("bottleneck_level", 3),
                             0,
                             residual_interpolator_settings.get("precompute", True),
                             residual_interpolator_settings.get("nh_inter", 3),
                             residual_interpolator_settings.get("power", 1),
                             residual_interpolator_settings.get("cutoff_dist_zoom", None),
                             residual_interpolator_settings.get("cutoff_dist", None),
                             residual_interpolator_settings.get("search_zoom_compute", None)
                             )
        else:
            self.residual_interpolator_up = self.residual_interpolator_down = None

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x, coords_input, coords_output, sample_dict={}, mask=None, emb=None, dists_input=None):
        b, nt, n = x.shape[:3]
        coords_input, coords_output, mask, dists_input = check_empty(coords_input), check_empty(coords_output), check_empty(mask), check_empty(dists_input)
        sample_dict = self.prepare_sample_dict(sample_dict)

        if self.interpolator:
            x, density_map = self.interpolator(x,
                                              mask=mask,
                                              calc_density=True,
                                              sample_dict=sample_dict,
                                              input_coords=coords_input,
                                              input_dists=dists_input)
            emb["DensityEmbedder"] = 1 - density_map.transpose(-2, -1)
            emb["UncertaintyEmbedder"] = (density_map.transpose(-2, -1), emb['VariableEmbedder'])

            x = x.unsqueeze(-3)
            mask = None

        interp_x = 0
        if self.residual_interpolator_up and self.residual_interpolator_down:
            interp_x, _ = self.residual_interpolator_down(x,
                                                          calc_density=False,
                                                          sample_dict=sample_dict)
            interp_x, _ = self.residual_interpolator_up(interp_x.unsqueeze(dim=-3),
                                                        calc_density=False,
                                                        sample_dict=sample_dict)
        emb['CoordinateEmbedder'] = self.model.grid_layer_max.get_coordinates(**sample_dict)
        x, posterior = self.model(x, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, residual=interp_x)
        return x, posterior

    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch
        output, posterior = self(source, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, dists_input=rel_dists_input)
        rec_loss, loss_dict = self.loss(output, target, mask=mask, sample_dict=sample_dict, prefix='train/')

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = rec_loss + self.kl_weight * kl_loss
            loss_dict['train/kl_loss'] = self.kl_weight * kl_loss
        else:
            loss = rec_loss

        loss_dict['train/total_loss'] = loss.item()

        self.log_dict(loss_dict, prog_bar=True, sync_dist=False)
        return loss


    def validation_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch
        coords_input, coords_output, mask, rel_dists_input = check_empty(coords_input), check_empty(coords_output), check_empty(mask), check_empty(rel_dists_input)
        sample_dict = self.prepare_sample_dict(sample_dict)

        output, posterior = self(source, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, dists_input=rel_dists_input)

        rec_loss, loss_dict = self.loss(output, target, mask=mask, sample_dict=sample_dict, prefix='val/')

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = rec_loss + self.kl_weight * kl_loss
            loss_dict['val/kl_loss'] = self.kl_weight * kl_loss
        else:
            loss = rec_loss

        loss_dict['val/total_loss'] = loss.item()

        # Calculate total loss

        if batch_idx == 0 and rank_zero_only:
            if hasattr(self, "interpolator") and self.interpolator is not None:
                input_inter, input_density = self.interpolator(source, mask=mask, input_coords=coords_input, sample_dict=sample_dict, calc_density=True, input_dists=rel_dists_input)
            else:
                input_inter = None
                input_density = None
            has_var = hasattr(self.model, 'predict_var') and self.model.predict_var
            self.log_tensor_plot(source, output, target, coords_input, coords_output, mask, sample_dict,
                                 f"tensor_plot_{int(self.current_epoch)}", emb, input_inter=input_inter,
                                 input_density=input_density, has_var=has_var)

        self.log_dict(loss_dict, prog_bar=False, sync_dist=True)

        return loss

    def predict_step(self, batch, batch_idx):
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source, mask, emb, coords_input, coords_output, sample_dict, input_dists):
        output, posterior, _ = self(source, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, dists_input=input_dists)
        return output