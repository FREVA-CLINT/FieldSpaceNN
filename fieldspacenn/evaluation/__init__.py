"""Native-HEALPix, monthly ensemble evaluation; no training dependencies."""
from .config import EvaluationConfig, Scenario, load_manifest


def evaluate_scenario(scenario, config=None, *, output_dir):
    from .suite import evaluate_scenario as run
    return run(scenario, config, output_dir=output_dir)


def evaluate_scenario_collection(manifest, config=None, *, output_dir):
    from .suite import evaluate_scenario_collection as run
    return run(manifest, config, output_dir=output_dir)


__all__ = ['EvaluationConfig', 'Scenario', 'load_manifest',
           'evaluate_scenario', 'evaluate_scenario_collection']
