import torch
import torch.nn as nn

from fieldspacenn.src.modules.flowmatching.mg_flow_matching import MGFlowMatching
from fieldspacenn.src.modules.flowmatching.mg_sampler import EulerFlowSampler
from fieldspacenn.src.models.mg_flowmatching import mg_flowmatching_model
from fieldspacenn.src.models.mg_flowmatching.mg_flowmatching_model import MGFlowMatchingModel
from fieldspacenn.src.models.mg_flowmatching.pl_mg_flowmatching_model import LightningMGFlowMatchingModel


class ConstantVelocityModel(nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = value
        self.last_emb_groups = None

    def forward(self, x_zooms_groups, emb_groups=None, **kwargs):
        self.last_emb_groups = emb_groups
        return [
            {int(zoom): torch.full_like(tensor, self.value) for zoom, tensor in group.items()}
            if group
            else None
            for group in x_zooms_groups
        ]


def _group(value: float, batch_size: int = 2):
    return {3: torch.full((batch_size, 1, 1, 1, 1, 1), value)}


def test_sample_times_stay_in_range():
    flow_matching = MGFlowMatching()

    times = flow_matching.sample_times(8, torch.device("cpu"), time_range=(0.25, 0.5))

    assert times.shape == (8,)
    assert torch.all(times >= 0.25)
    assert torch.all(times <= 0.5)


def test_interpolation_and_target_velocity():
    flow_matching = MGFlowMatching()
    data = _group(1.0)
    noise = _group(0.0)
    times = torch.tensor([0.25, 0.75])

    x_t = flow_matching.interpolate(data, noise, times)
    target = flow_matching.target_velocity(data, noise)

    assert torch.allclose(x_t[3].flatten(), times)
    assert torch.allclose(target[3], torch.ones_like(target[3]))


def test_training_losses_inject_time_and_return_endpoint_estimate():
    flow_matching = MGFlowMatching()
    model = ConstantVelocityModel(value=1.0)
    data_groups = [_group(1.0)]
    noise_groups = [_group(0.0)]
    times = torch.tensor([0.25, 0.75])

    outputs = flow_matching.training_losses(
        model,
        data_groups,
        times,
        noise_groups=noise_groups,
        create_pred_x1=True,
    )

    target, model_output, pred_x1 = outputs[0]
    assert torch.allclose(target[3], torch.ones_like(target[3]))
    assert torch.allclose(model_output[3], torch.ones_like(model_output[3]))
    assert torch.allclose(pred_x1[3], torch.ones_like(pred_x1[3]))
    assert model.last_emb_groups[0]["DiffusionStepEmbedder"] is times


def test_training_losses_preserve_known_mask_regions():
    flow_matching = MGFlowMatching()
    model = ConstantVelocityModel(value=5.0)
    data_groups = [_group(1.0)]
    noise_groups = [_group(0.0)]
    mask_groups = [{3: torch.zeros_like(data_groups[0][3], dtype=torch.bool)}]
    times = torch.tensor([0.25, 0.75])

    outputs = flow_matching.training_losses(
        model,
        data_groups,
        times,
        mask_groups=mask_groups,
        noise_groups=noise_groups,
        create_pred_x1=True,
    )

    target, model_output, pred_x1 = outputs[0]
    assert torch.allclose(target[3], torch.zeros_like(target[3]))
    assert torch.allclose(model_output[3], torch.zeros_like(model_output[3]))
    assert torch.allclose(pred_x1[3], torch.ones_like(pred_x1[3]))


def test_euler_sampler_integrates_constant_velocity():
    flow_matching = MGFlowMatching()
    sampler = EulerFlowSampler(flow_matching)
    model = ConstantVelocityModel(value=1.0)
    input_groups = [_group(0.0)]
    x_t_groups = [_group(0.0)]

    output = sampler.sample_loop(
        model,
        input_groups=input_groups,
        x_t_groups=x_t_groups,
        n_steps=4,
        time_range=(0.0, 1.0),
    )

    assert torch.allclose(output[0][3], torch.ones_like(output[0][3]))


def test_flowmatching_model_and_lightning_wrapper_instantiation(monkeypatch):
    class TinyTransformer(nn.Module):
        def __init__(self, mgrids, block_configs, in_zooms, in_features=1, **kwargs):
            super().__init__()
            self.in_zooms = in_zooms
            self.in_features = in_features
            self.grid_layers = nn.ModuleDict()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, x_zooms_groups=None, **kwargs):
            return x_zooms_groups

    monkeypatch.setattr(mg_flowmatching_model, "MG_Transformer", TinyTransformer)

    model = MGFlowMatchingModel(
        n_blocks=2,
        block_time_ranges=[[0.0, 0.5], [0.5, 1.0]],
        mgrids=[],
        block_configs={},
        in_zooms=[3],
    )
    lightning_model = LightningMGFlowMatchingModel(
        model=model,
        flow_matching=MGFlowMatching(),
        lr_groups={"default": {"lr": 0.001}},
        lambda_loss_dict={"zooms": {}},
    )

    assert model.n_blocks == 2
    assert model.get_time_range(1, inference=True) == (0.5, 1.0)
    assert lightning_model.sampling_steps_per_block == 50
