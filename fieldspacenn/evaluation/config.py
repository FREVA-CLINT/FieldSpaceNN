"""Explicit configuration. Physical metadata are never inferred from magnitudes."""
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
import json


@dataclass
class EvaluationConfig:
    mode: str = 'quick'
    seed: int = 1729
    confidence: float = .95
    bootstrap: int | None = None
    size_matching_repetitions: int | None = None
    spatial_pairs: int | None = None
    lmax: int = 64
    nside: int = 32
    allow_test_grid: bool = False
    max_memory_mb: int = 1024
    spatial_snapshots: int | None = None
    eof_samples: int | None = None
    eof_modes: int = 6
    monthly_lags: list = field(default_factory=lambda: [1, 3, 6, 12, 24, 60])
    annual_lags: list = field(default_factory=lambda: [1, 2, 5, 10])
    averaging_months: list = field(default_factory=lambda: [1, 12, 60, 120, 240])
    trend_years: list = field(default_factory=lambda: [5, 10, 20])
    latitude_bands: list = field(default_factory=lambda: [[-90, -60], [-60, -30], [-30, 30], [30, 60], [60, 90]])
    plot_regions: list = field(default_factory=lambda: ['Global', 'NEU', 'MED', 'WAF', 'SAS', 'SEA'])
    ar6: bool = True
    polygon_path: str | None = None
    polygon_id: str = 'Acronym'
    fractional_mask_path: str | None = None
    land_fraction_path: str | None = None
    land_fraction_variable: str = 'land_fraction'
    cache_dir: str = '.cache/climate-evaluation'
    intervals: list = field(default_factory=lambda: [.5, .8, .9, .95])
    quantiles: list = field(default_factory=lambda: [.01, .05, .1, .5, .9, .95, .99])
    min_tail_observations: int = 10
    min_independent_blocks: int = 8
    residual_block_months: int = 24
    welch_segment_months: int = 120
    fair_crps: bool = False
    independent_members: bool = False
    min_coverage: float = .95
    missing_policy: str = 'error'
    temporal_alignment: str = 'exact'
    thresholds: dict = field(default_factory=lambda: {'hot': .9, 'cold': .1, 'dry': .1, 'wet': .9})
    deficit_years: list = field(default_factory=lambda: [2, 5, 10])
    deficit_fraction: float = .1
    warming_levels: list = field(default_factory=lambda: [1.5, 2., 3.])
    sustained_crossing_years: int = 20
    event_windows: dict = field(default_factory=lambda: {
        'pre': [-120, 0], 'response': [0, 36], 'early_recovery': [36, 72],
        'late_recovery': [72, 132], 'persistent': [132, 252]})
    variogram_exponent: float = .5
    vector_time_lags: list = field(default_factory=lambda: [0, 1, 2])
    selected_cells: list = field(default_factory=list)
    long_range_pairs: list = field(default_factory=list)
    tolerances: dict = field(default_factory=dict)
    significance_maps: bool = False
    fdr_alpha: float = .05
    return_period_years: list = field(default_factory=list)
    min_return_years: int = 30
    figures: bool = True

    def __post_init__(self):
        if self.mode not in ('quick', 'full'):
            raise ValueError('mode must be quick or full')
        for key, values in {'bootstrap': (99, 999), 'size_matching_repetitions': (20, 100),
                            'spatial_pairs': (10000, 100000), 'spatial_snapshots': (12, 120),
                            'eof_samples': (240, 1200)}.items():
            if getattr(self, key) is None:
                setattr(self, key, values[self.mode == 'full'])
        if self.nside != 32 and not self.allow_test_grid:
            raise ValueError('Production evaluation requires NSIDE=32; other grids are test-only')
        if self.nside < 1 or self.nside & (self.nside - 1):
            raise ValueError('NSIDE must be a positive power of two')
        self.lmax = min(self.lmax, 3 * self.nside - 1)
        if self.bootstrap < 2 or not 0 < self.confidence < 1:
            raise ValueError('bootstrap >= 2 and 0 < confidence < 1 required')
        if self.missing_policy not in ('error', 'common_mask'):
            raise ValueError('missing_policy must be error or common_mask')
        if self.temporal_alignment not in ('exact', 'intersection'):
            raise ValueError('temporal_alignment must be exact or intersection')
        if self.fair_crps and not self.independent_members:
            raise ValueError('Fair CRPS requires independent_members=True and supporting provenance')
        if not 0 < self.variogram_exponent < 2:
            raise ValueError('variogram exponent must lie in (0,2)')

    def to_dict(self):
        return asdict(self)


@dataclass
class Scenario:
    id: str
    generated: dict
    description: str = ''
    reference: dict | None = None
    calibration: dict | None = None
    training: dict | None = None
    reconstruction: dict | None = None
    baselines: list = field(default_factory=list)
    forcings: dict = field(default_factory=dict)
    eruptions: list = field(default_factory=list)
    control_id: str | None = None
    co2_group: str | None = None
    volcanic_group: str | None = None
    history_probability: float | None = None
    initialization: dict = field(default_factory=dict)
    periods: dict = field(default_factory=dict)
    baseline_period: list | None = None
    climate_windows: dict = field(default_factory=dict)
    splits: dict = field(default_factory=dict)
    training_support: dict = field(default_factory=dict)
    held_out: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)
    pairing: dict = field(default_factory=dict)
    recovery: dict = field(default_factory=dict)
    comparisons: list = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    model_variant: str = 'generator'
    physical_scenario: str | None = None
    role: str = 'generated'
    reference_independent: bool = False
    stationary: bool = False
    spinup_end: str | None = None

    def __post_init__(self):
        if not self.id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in self.id):
            raise ValueError('Scenario ID must contain only letters, digits, underscores or hyphens')
        if not isinstance(self.generated, dict) or not any(k in self.generated for k in ('path', 'member_paths')):
            raise ValueError(f'{self.id}: generated needs path or member_paths')
        if self.role not in ('generated', 'control', 'baseline', 'ablation', 'reconstruction'):
            raise ValueError(f'Unsupported scenario role {self.role}')

    def to_dict(self):
        return asdict(self)


def load_manifest(value):
    """Read a JSON manifest or dictionary; resolve input paths relative to its file."""
    base = None
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        value, base = json.loads(path.read_text()), path.parent
    if isinstance(value, list):
        value = {'scenarios': value}
    value = json.loads(json.dumps(value))
    def resolve(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if base and (k == 'path' or k.endswith('_path')) and isinstance(v, str) and '://' not in v:
                    obj[k] = str((base / Path(v).expanduser()).resolve())
                elif base and k == 'member_paths':
                    obj[k] = [str((base / Path(p).expanduser()).resolve()) for p in v]
                else:
                    resolve(v)
        elif isinstance(obj, list):
            for v in obj:
                resolve(v)
    resolve(value)
    scenarios = [Scenario(**s) for s in value['scenarios']]
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError('Scenario IDs must be unique')
    return scenarios, EvaluationConfig(**value.get('config', {})), value
