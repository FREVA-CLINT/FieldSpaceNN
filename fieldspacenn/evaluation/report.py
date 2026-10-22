"""Complete registry, tidy scalar records, provenance and scientific acceptance."""
from dataclasses import dataclass, field
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import xarray as xr


FAMILIES = {
 'coverage': ['input_coverage', 'forcing_coverage', 'region_coverage', 'leakage_audit', 'training_support', 'generation_cost'],
 'forced': ['trajectory', 'climatology', 'seasonal_cycle_amplitude', 'seasonal_cycle_phase', 'mean_map',
            'mean_bias', 'mae', 'rmse', 'pattern_correlation', 'trend', 'trend_error',
            'land_ocean_contrast', 'hemispheric_contrast', 'temperature_change_pattern', 'significance_map'],
 'calibration': ['crps', 'fair_crps', 'crps_skill', 'coverage', 'interval_width', 'rank_histogram',
                 'quantile_error', 'wasserstein', 'spread_ratio', 'variance_ratio', 'size_matched_crps',
                 'reference_benchmark', 'qq_distribution'],
 'variability': ['residual_variance', 'autocorrelation', 'temporal_spectrum', 'averaging_variance',
                 'raw_trend_distribution', 'residual_trend_distribution', 'anomaly_persistence',
                 'climate_state_variability'],
 'dependence': ['spatial_spectrum', 'semivariogram', 'regional_correlation', 'long_range_correlation',
                'eof_basis', 'eof_subspace', 'eof_explained_variance', 'pc_spectrum',
                'temperature_precipitation_correlation', 'lagged_temperature_precipitation',
                'cross_region_dependence', 'energy_score', 'variogram_score'],
 'extremes': ['monthly_tail', 'annual_monthly_temperature_extrema', 'seasonal_rainfall_anomaly',
              'event_probability', 'event_duration', 'event_severity', 'event_area_fraction',
              'regional_average_extreme', 'regional_cooccurrence', 'precipitation_deficit',
              'multi_season_persistence', 'threshold_ties', 'climate_state_extremes', 'return_level',
              'Rx1day', 'Rx5day', 'daily_wet_day_frequency', 'consecutive_dry_days', 'TXx', 'daily_heatwave_duration'],
 'eruption': ['controlled_response', 'peak_response', 'time_to_peak', 'integrated_response', 'recovery',
              'seasonal_response', 'response_contrasts', 'eruption_variability', 'eruption_event_risk',
              'precipitation_temperature_response'],
 'cross_scenario': ['co2_only_response', 'amplitude_response', 'background_interaction', 'eruption_spacing',
                    'history_dependence', 'stationary_drift', 'interpolation', 'extrapolation',
                    'error_support_distance', 'future_forcing_perturbation'],
 'nested': ['nested_variance', 'negative_temperature_trend', 'warming_crossing',
            'nested_precipitation_deficit', 'nested_compound_probability'],
 'baselines': ['baseline_comparison', 'simple_baseline', 'ablation_comparison', 'compression_error'],
}
REGISTRY = {metric: family for family, metrics in FAMILIES.items() for metric in metrics}
DAILY = set(FAMILIES['extremes'][-6:])
COLLECTION = set(FAMILIES['eruption'] + FAMILIES['cross_scenario'] + FAMILIES['nested'])


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+'\n')


@dataclass
class ScenarioResult:
    scenario_id: str
    metrics: pd.DataFrame
    artifacts: dict
    provenance: dict
    figures: list = field(default_factory=list)
    validation: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class CollectionResult:
    scenarios: dict
    metrics: pd.DataFrame
    comparison: pd.DataFrame
    provenance: dict
    figures: list = field(default_factory=list)


class Recorder:
    def __init__(self, scenario_id, output_dir, config):
        self.id, self.config = scenario_id, config
        self.root = Path(output_dir); self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'diagnostics').mkdir(exist_ok=True)
        self.rows, self.artifacts = [], {}
        self.source_counts = {}

    def add(self, metric, value=None, *, units='1', aggregation='not applicable', variable='both',
            region='Global', period='evaluation', comparator='', n_members=0, n_targets=0,
            n_time=0, n_independent=None, ci=None, ci_method='', status='Computed', reason='',
            artifact='', details=None):
        if metric not in REGISTRY:
            raise KeyError(f'Unregistered metric: {metric}')
        details=dict(details or {})
        if self.source_counts:
            details.setdefault('source_population_counts',self.source_counts)
            if status=='Computed':
                n_members=n_members or self.source_counts.get('reference_members' if comparator=='reference' else 'members',0)
                n_targets=n_targets or (self.source_counts.get('reference_members',0) if comparator!='reference' else 0)
                n_time=n_time or self.source_counts.get('months',0)
        if value is not None and not np.isfinite(value):
            value, status, reason = None, 'Undefined statistic', reason or 'Zero variance or unsupported numeric domain'
        lo, hi = ci if ci is not None else (None, None)
        ci_valid = lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi)
        acceptance = 'Not assessed'
        tolerance = self.config.tolerances.get(metric)
        if tolerance is not None and status == 'Computed':
            if not ci_valid:
                acceptance = 'Not assessable without interval'
            else:
                lower, upper = tolerance
                acceptance = 'Within tolerance' if lower <= lo and hi <= upper else 'Equivalence not demonstrated'
        self.rows.append(dict(scenario=self.id, family=REGISTRY[metric], metric=metric,
            variable=variable, region=region, period=period, aggregation=aggregation, comparator=comparator,
            value=value, units=units, n_members=n_members, n_targets=n_targets, n_time=n_time,
            n_independent=n_independent, ci_low=lo if ci_valid else None, ci_high=hi if ci_valid else None,
            ci_level=self.config.confidence if ci_valid else None, ci_method=ci_method if ci_valid else '',
            status=status, reason=reason, acceptance=acceptance, artifact=artifact,
            details=json.dumps(clean_json(details), sort_keys=True)))

    def skip(self, metrics, status, reason, **kwargs):
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            self.add(metric, status=status, reason=reason, **kwargs)

    def array(self, name, data, **attrs):
        """Persist arrays immediately; registry records link to these values."""
        if isinstance(data, xr.DataArray):
            data = data.to_dataset(name=data.name or 'value')
        data = data.copy(deep=False)
        def sanitize(attributes):
            return {k: (int(v) if isinstance(v, bool) else v if isinstance(v, (str, int, float, np.number))
                        else json.dumps(clean_json(v))) for k, v in attributes.items() if v is not None}
        data.attrs = sanitize({**data.attrs, **attrs})
        for var in data.variables:
            data[var].attrs = sanitize(data[var].attrs)
            if data[var].dtype == bool:
                data[var] = data[var].astype('int8')
        filename = ''.join(c if c.isalnum() or c in '_-' else '_' for c in name)+'.nc'
        path = self.root/'diagnostics'/filename
        data.to_netcdf(path, engine='netcdf4')
        self.artifacts[name] = str(path)
        return str(path)

    def finalize(self, *, collection=False):
        present = {r['metric'] for r in self.rows}
        expected = set(REGISTRY) if not collection else COLLECTION
        missing = expected-present
        # Completeness is a programming invariant: do not invent a skip for a
        # forgotten calculation. Collection-dependent rows are explicitly gated.
        if missing:
            raise RuntimeError(f'Metric dispatcher omitted registry entries: {sorted(missing)}')
        frame = pd.DataFrame(self.rows)
        frame.to_csv(self.root/'metrics.csv', index=False)
        frame.to_parquet(self.root/'metrics.parquet', index=False)
        scorecard = frame.groupby(['family', 'status'], dropna=False).size().reset_index(name='results')
        scorecard.to_csv(self.root/'scorecard.csv', index=False)
        write_json(self.root/'metric_registry.json', REGISTRY)
        return frame


def provenance(config, scenario):
    packages = {}
    for name in ['numpy', 'scipy', 'xarray', 'dask', 'zarr', 'healpy', 'cftime', 'pandas',
                 'matplotlib', 'scikit-learn', 'regionmask', 'geopandas', 'netCDF4', 'pyarrow', 'nbclient', 'nbformat']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    digest = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(file.name.encode()); digest.update(file.read_bytes())
    return {'python': platform.python_version(), 'packages': packages,
            'implementation_sha256': digest.hexdigest(), 'config': config.to_dict(),
            'scenario': scenario.to_dict() if hasattr(scenario, 'to_dict') else scenario,
            'bootstrap_unit': 'complete member trajectories; documented lineage clusters when supplied',
            'test_data': config.allow_test_grid,
            'quick_mode_disclosure': 'Exploratory 99-replicate intervals and sampled expensive diagnostics; full registry retained'
                if config.mode == 'quick' else 'Full budgets; expensive sample selections remain explicit',
            'references': {'ar6': 'https://doi.org/10.5194/essd-12-2959-2020',
                           'fair_crps': 'https://doi.org/10.1002/qj.2270'},
            'event_windows': 'half-open monthly intervals relative to eruption month',
            'spectrum': 'Welch Hann, 50% overlap, segment linear detrend; cycles/year',
            'fdr': 'Benjamini-Yekutieli within variable/metric/period spatial family',
            'history_bootstrap': 'resampled forcing-history support sensitivity, not exhaustive finite-support uncertainty',
            'synthetic_substitution': False}
