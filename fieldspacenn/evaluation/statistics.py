"""Small-array statistical kernels. Dimensions and estimands are explicit.

Space-time arrays are reduced/chunked by the orchestrator before these kernels.
Members are never concatenated into a time series. Bootstrap units are complete
members (or documented lineage clusters), except explicitly stationary blocks.
"""
import numpy as np
from scipy import signal, stats
from scipy.spatial.distance import cdist, pdist, squareform
from .data import Unavailable


def crps(ensemble, targets, fair=False):
    """(M, ...) forecast versus (R, ...) targets -> (R, ...), physical units.

    Empirical: mean|X-y| - sum_(i<j)|Xi-Xj|/M².
    Fair: replace M² with M(M-1), for iid conditional draws only.
    Sorting avoids a quadratic member-by-member field array.
    """
    x, y = np.asarray(ensemble), np.asarray(targets)
    m = len(x)
    if fair and m < 2:
        raise Unavailable('Insufficient members', 'Fair CRPS needs at least two forecast members')
    ordered = np.sort(x, axis=0)
    coefficients = (2*np.arange(1, m+1)-m-1).reshape((m,)+(1,)*(x.ndim-1))
    pair = np.sum(coefficients*ordered, axis=0)/(m*(m-1) if fair else m*m)
    return np.stack([np.mean(np.abs(x-target), axis=0)-pair for target in y])


def rank_histogram(x, y):
    """Fractionally allocate tied ranks; output sums to one over ranks."""
    out = np.zeros(len(x)+1)
    valid_total = 0
    for target in y:
        valid = np.isfinite(target) & np.isfinite(x).all(axis=0)
        lower, equal = (x < target).sum(axis=0), (x == target).sum(axis=0)
        for rank in range(len(x)+1):
            out[rank] += np.sum(np.where(valid, ((lower <= rank) & (rank <= lower+equal))/(equal+1), 0))
        valid_total += valid.sum()
    return out/valid_total if valid_total else out*np.nan


def coverage_width(x, y, probability):
    lo, hi = np.quantile(x, [(1-probability)/2, (1+probability)/2], axis=0)
    return ((y >= lo) & (y <= hi)).astype(float), hi-lo


def correlation(a, b, axis=-1):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a-a.mean(axis=axis, keepdims=True), b-b.mean(axis=axis, keepdims=True)
    den = np.sqrt((a*a).sum(axis=axis)*(b*b).sum(axis=axis))
    return np.divide((a*b).sum(axis=axis), den, out=np.full(np.shape(den), np.nan), where=den > 0)


def slope(a, years, axis=-1):
    """OLS slope in physical units per decade; no iid-time standard errors."""
    x = np.asarray(years, float); x = x-x.mean()
    if len(x) < 2 or not np.sum(x*x):
        return np.full(np.asarray(a).shape[:axis] if axis == -1 else (), np.nan)
    return np.sum(np.moveaxis(a, axis, -1)*x, axis=-1)/np.sum(x*x)*10


def autocorrelation(x, lags):
    x = np.asarray(x)
    return np.stack([correlation(x[..., :-lag], x[..., lag:]) if 0 < lag < x.shape[-1]-2
                     else np.full(x.shape[:-1], np.nan) for lag in lags], axis=-1)


def welch_spectrum(x, segment=120, sampling_per_year=12):
    """Hann-window Welch PSD, linear detrend per segment, 50% overlap.

    Need at least four non-overlapping segment equivalents; overlapping Welch
    segments improve estimation but do not count as independent replicates.
    """
    if x.shape[-1] < 4*segment:
        raise Unavailable('Insufficient record length', f'Welch needs at least {4*segment} observations for segment={segment}')
    f, p = signal.welch(x, fs=sampling_per_year, window='hann', nperseg=segment,
                        noverlap=segment//2, detrend='linear', axis=-1, scaling='density')
    return f[1:], p[..., 1:]


def block_means(x, width, days=None):
    n = x.shape[-1]//width
    shaped=x[..., :n*width].reshape(x.shape[:-1]+(n, width))
    if days is None:
        return shaped.mean(axis=-1)
    weights=np.asarray(days)[:n*width].reshape(n,width)
    return np.sum(shaped*weights,axis=-1)/weights.sum(axis=-1)


def run_lengths(mask):
    """Return durations and starts for one boolean time series."""
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return ends-starts, starts


def member_indices(n, rng, clusters=None):
    if clusters is None:
        return rng.integers(n, size=n)
    clusters = np.asarray(clusters)
    if len(clusters) != n:
        raise ValueError('Lineage clusters must align with member IDs')
    unique = np.unique(clusters)
    chosen = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(clusters == c) for c in chosen])


def bootstrap_members(function, x, y=None, *, repetitions=99, confidence=.95,
                      seed=1729, clusters=None, target_clusters=None, paired=False):
    """Resample full trajectories. Pairing must be validated by the caller."""
    if len(x) < 2 or (y is not None and len(y) < 2):
        return np.nan, np.nan
    if (clusters is not None and len(np.unique(clusters))<2) or (target_clusters is not None and len(np.unique(target_clusters))<2):
        return np.nan,np.nan
    if paired and (y is None or len(x) != len(y)):
        raise ValueError('Paired resampling requires equal, aligned member counts')
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        ix = member_indices(len(x), rng, clusters)
        if y is None:
            values.append(function(x[ix]))
        else:
            iy = ix if paired else member_indices(len(y), rng, target_clusters)
            values.append(function(x[ix], y[iy]))
    return tuple(np.nanquantile(values, [(1-confidence)/2, (1+confidence)/2], axis=0))


def stationary_block_indices(length, block, rng):
    """Circular fixed-length blocks; only for declared stationary residual windows."""
    starts = rng.integers(length, size=int(np.ceil(length/block)))
    return ((starts[:, None]+np.arange(block)) % length).ravel()[:length]


def benjamini_yekutieli(pvalues, alpha=.05):
    """FDR correction valid under arbitrary dependence within a declared family."""
    p = np.asarray(pvalues); out = np.full(p.shape, np.nan)
    good = np.isfinite(p); values = p[good]; n = len(values)
    if not n:
        return out, np.zeros(p.shape, bool)
    order = np.argsort(values)
    adjusted = values[order]*n*np.sum(1/np.arange(1, n+1))/np.arange(1, n+1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    undo = np.empty(n); undo[order] = adjusted; out[good] = undo
    return out, out <= alpha


def sampled_semivariogram(x, lon, lat, count=10000, seed=1729, bins=None):
    """x=(sample,cell); random unique unordered pairs; great-circle distance km."""
    rng = np.random.default_rng(seed); cells = x.shape[-1]
    # Generate a bounded number of candidates, not a full cell covariance matrix.
    pairs = np.sort(rng.integers(cells, size=(count*2, 2)), axis=1)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)[:count]
    a, b = pairs.T
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    hav = np.sin((lat[a]-lat[b])/2)**2+np.cos(lat[a])*np.cos(lat[b])*np.sin((lon[a]-lon[b])/2)**2
    distance = 6371.0088*2*np.arcsin(np.sqrt(hav.clip(0, 1)))
    bins = np.linspace(0, np.pi*6371.0088, 21) if bins is None else np.asarray(bins)
    semivariance = np.mean((x[:, a]-x[:, b])**2, axis=0)/2
    assigned = np.minimum(np.digitize(distance, bins)-1, len(bins)-2)
    values = np.array([semivariance[assigned == i].mean() if (assigned == i).any() else np.nan for i in range(len(bins)-1)])
    counts = np.bincount(assigned, minlength=len(bins)-1)
    return (bins[1:]+bins[:-1])/2, values, counts


def spatial_spectrum(maps, nside, lmax):
    import healpy as hp
    if not np.isfinite(maps).all():
        raise Unavailable('Undefined statistic', 'Full-sky spectra require complete maps; zero-filling a mask would bias power')
    return np.stack([hp.anafast(hp.reorder(m-m.mean(), n2r=True), lmax=min(lmax, 3*nside-1),
                                pol=False, iter=3) for m in maps])


def fit_eofs(calibration, modes=6, seed=1729):
    """Randomized thin SVD on (samples,cells), equal-area weighted implicitly."""
    from sklearn.utils.extmath import randomized_svd
    if not np.isfinite(calibration).all():
        raise Unavailable('Undefined statistic', 'EOF input needs complete common spatial support')
    center = calibration.mean(axis=0)
    x = calibration-center
    rank = min(modes, min(x.shape)-1)
    if rank < 1:
        raise Unavailable('Insufficient independent information', 'EOF needs at least two samples and cells')
    _, s, vt = randomized_svd(x, n_components=rank, random_state=seed)
    total = np.sum(x*x)
    if total == 0:
        raise Unavailable('Undefined statistic', 'Zero calibration variance')
    return {'center': center, 'basis': vt, 'explained': s*s/total, 'singular_values': s}


def eof_subspace(basis_a, basis_b):
    singular = np.linalg.svd(basis_a @ basis_b.T, compute_uv=False).clip(0, 1)
    return np.mean(singular**2), np.arccos(singular)


def energy_score(x, y):
    """(forecast_member, feature), (target_member, feature), fixed scaled space."""
    return cdist(y, x).mean(axis=1)-cdist(x, x).mean()/2


def variogram_score(x, y, exponent=.5, weights=None):
    i, j = np.triu_indices(x.shape[-1], 1)
    weights = np.ones(len(i))/max(1, len(i)) if weights is None else np.asarray(weights)
    predicted = np.mean(np.abs(x[:, i]-x[:, j])**exponent, axis=0)
    return np.sum(weights*(np.abs(y[:, i]-y[:, j])**exponent-predicted)**2, axis=-1)


def eruption_summary(response, month_offsets, days, *, peak_interval, tolerance, sustained):
    """Signed response: negative cooling/suppression; integration in unit-days.

    Peak is the minimum in [start,end). Recovery search starts after the peak.
    Censored duration is last observed elapsed month, not a fictitious recovery.
    """
    response = np.asarray(response); t = np.asarray(month_offsets)
    selected = (t >= peak_interval[0]) & (t < peak_interval[1])
    if selected.sum() != peak_interval[1]-peak_interval[0]:
        raise Unavailable('Incomplete window', 'Peak search interval is incomplete')
    indices = np.flatnonzero(selected)
    peak_index = indices[np.argmin(response[selected])]
    recovered = None
    for i in range(peak_index+1, len(response)-sustained+1):
        if np.all(np.abs(response[i:i+sustained]) <= tolerance):
            recovered = int(t[i]); break
    result = {'peak': float(response[peak_index]), 'time_to_peak_months': int(t[peak_index]),
              'recovery_months': recovered, 'censored': recovered is None,
              'followup_months': int(t[-1])+1}
    for width in (60, 120):
        take = (t >= 0) & (t < width)
        result[f'integral_{width}m'] = float(np.sum(response[take]*days[take])) if take.sum() == width else None
    return result


def nested_variance(histories, weights=None):
    """Finite discrete distribution of forcing histories; arrays are (member,...).

    Between correction accounts for noise in both conditional and overall means.
    Negative estimates are retained. A one-member history cannot estimate noise.
    """
    if len(histories) < 2:
        raise Unavailable('Insufficient independent information', 'At least two forcing histories required')
    counts = np.array([len(x) for x in histories])
    if np.any(counts < 2):
        raise Unavailable('Insufficient members', 'Every history needs at least two members for finite-size correction')
    w = np.ones(len(histories))/len(histories) if weights is None else np.asarray(weights, float)
    if w.shape != counts.shape or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('History probabilities must be finite nonnegative weights with positive sum')
    w = w/w.sum(); shape = (len(w),)+(1,)*(histories[0].ndim-1)
    ww, nn = w.reshape(shape), counts.reshape(shape)
    means = np.stack([x.mean(axis=0) for x in histories])
    variances = np.stack([x.var(axis=0, ddof=1) for x in histories])
    grand = np.sum(ww*means, axis=0)
    within = np.sum(ww*variances, axis=0)
    raw = np.sum(ww*(means-grand)**2, axis=0)
    correction = np.sum(ww*(1-ww)*variances/nn, axis=0)
    between = raw-correction
    return {'within': within, 'between': between, 'between_uncorrected': raw,
            'finite_member_correction': correction, 'total': within+between,
            'negative_between': between < 0, 'weights': w}


def hierarchical_variance_interval(histories, weights, repetitions=99, confidence=.95, seed=1729):
    """History bootstrap estimates sensitivity to sampled support, then members.

    Draw history IDs using supplied probabilities; their resampled empirical
    distribution has equal slot weights (do not apply probabilities twice).
    """
    rng = np.random.default_rng(seed); w = np.asarray(weights, float); w /= w.sum()
    draws = {k: [] for k in ('within', 'between', 'total')}
    for _ in range(repetitions):
        ix = rng.choice(len(histories), size=len(histories), replace=True, p=w)
        sample = [histories[h][rng.integers(len(histories[h]), size=len(histories[h]))] for h in ix]
        estimate = nested_variance(sample)
        for k in draws:
            draws[k].append(estimate[k])
    return {k: np.quantile(v, [(1-confidence)/2, (1+confidence)/2], axis=0) for k, v in draws.items()}


def threshold_crossings(annual, threshold, sustained=1):
    """Trailing-mean crossing, dated at the final year of its averaging window."""
    x = np.asarray(annual)
    if sustained > x.shape[-1]:
        raise Unavailable('Insufficient record length', 'Sustained warming window exceeds record')
    windows = np.lib.stride_tricks.sliding_window_view(x, sustained, axis=-1).mean(axis=-1)
    crossed = windows >= threshold
    event = crossed.any(axis=-1)
    timing = np.where(event, crossed.argmax(axis=-1)+sustained-1, x.shape[-1]-1)
    return timing, ~event


def reference_benchmark(reference, forecast_size, target_size, repetitions=20, seed=1729):
    """Disjoint held-out reference forecasts/targets; preserve complete trajectories."""
    if len(reference) < forecast_size+target_size:
        raise Unavailable('Insufficient members', f'Reference benchmark needs {forecast_size+target_size} independent members')
    rng = np.random.default_rng(seed); scores = []; bias = []; rmse = []
    for _ in range(repetitions):
        ix = rng.permutation(len(reference))[:forecast_size+target_size]
        x, y = reference[ix[:forecast_size]], reference[ix[forecast_size:]]
        scores.append(np.mean(crps(x, y)))
        difference = x.mean(axis=0)-y.mean(axis=0)
        bias.append(difference.mean()); rmse.append(np.sqrt(np.mean(difference**2)))
    return {'crps': np.asarray(scores), 'mean_bias': np.asarray(bias), 'rmse': np.asarray(rmse)}


def hierarchical_risk_interval(histories,weights,function=np.mean,*,repetitions=99,confidence=.95,seed=1729):
    """Mixture risk uncertainty with full member records retained per history."""
    rng=np.random.default_rng(seed);w=np.asarray(weights,float);w=w/w.sum()
    estimates=[]
    for _ in range(repetitions):
        selected=rng.choice(len(histories),len(histories),replace=True,p=w)
        estimates.append(np.mean([function(histories[h][rng.integers(len(histories[h]),size=len(histories[h]))]) for h in selected],axis=0))
    return np.quantile(estimates,[(1-confidence)/2,(1+confidence)/2],axis=0)


def return_level(annual_extremes, periods, *, min_years=30, repetitions=99, seed=1729):
    """GEV per member, then aggregate; bootstrap complete member records.

    Reject return periods longer than half the per-member record. This is an
    explicit conservative support policy, not a universal statistical bound.
    """
    x = np.asarray(annual_extremes)
    if x.shape[-1] < min_years or max(periods) > x.shape[-1]/2:
        raise Unavailable('Insufficient record length', 'Return-level support rule not satisfied')
    estimates = np.array([stats.genextreme.ppf(1-1/np.asarray(periods), *stats.genextreme.fit(v)) for v in x])
    lo, hi = bootstrap_members(lambda a: a.mean(axis=0), estimates, repetitions=repetitions, seed=seed)
    return estimates.mean(axis=0), lo, hi


def fit_lagged_baseline(training_fields, forcing, lags=(0, 1, 3, 6, 12, 24), ridge=1e-6):
    """training_fields=(M,T,V,C), forcing=(T,2), separate calibration only.

    Regress ensemble-mean fields on intercept, calendar harmonics and lagged
    CO2/AOD. Keep full multivariate residual fields for joint block resampling.
    """
    x = np.asarray(training_fields); f = np.asarray(forcing)
    if len(f) <= max(lags)+12:
        raise Unavailable('Insufficient record length', 'Baseline training record too short for lagged predictors')
    design, start = forcing_design(f, lags)
    target = x[:, start:].mean(axis=0)
    penalty = np.eye(design.shape[1])*ridge; penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T@design+penalty, design.T@target.reshape(len(design), -1))
    fitted = (design@coefficients).reshape(target.shape)
    residual = x[:, start:]-fitted
    return {'coefficients': coefficients, 'residual': residual, 'lags': tuple(lags),
            'shape': target.shape[1:], 'training_start': start}


def forcing_design(forcing, lags, months=None):
    start = max(lags); t = np.arange(start, len(forcing))
    # CO2 is log-transformed relative to a fixed 280 ppm convention.
    f = np.asarray(forcing, float).copy()
    if np.any(f[:, 0] <= 0):
        raise ValueError('CO2 concentration must be positive ppm')
    f[:, 0] = np.log(f[:, 0]/280.)
    phase=t if months is None else np.asarray(months)[t]-1
    design = np.column_stack([np.ones(len(t)), np.sin(2*np.pi*phase/12), np.cos(2*np.pi*phase/12),
                              *[f[t-lag, j] for j in range(f.shape[1]) for lag in lags]])
    return design, start


def generate_baseline(model, forcing, members, block=24, seed=1729):
    design, start = forcing_design(forcing, model['lags'])
    mean = (design@model['coefficients']).reshape((len(design),)+model['shape'])
    residual = model['residual']; rng = np.random.default_rng(seed)
    generated = []
    for _ in range(members):
        pieces = []
        for j in range(0, len(mean), block):
            member = rng.integers(len(residual)); begin = rng.integers(residual.shape[1])
            ix = (begin+np.arange(min(block, len(mean)-j))) % residual.shape[1]
            pieces.append(residual[member, ix])
        generated.append(mean+np.concatenate(pieces, axis=0))
    return np.stack(generated), start
