"""Diagnostics for one decoded conditional ensemble and its verifying trajectory."""

import numpy as np


def evaluate_decoded_ensemble(ensemble, reference, months, history_length=0,
                             area_weights=None, eruption_index=None):
    """Score physical fields shaped (members,time,cells) against (time,cells).

    Call separately for each variable, scenario, initial history, and held-out
    trajectory. Scores exclude known history; boundary and volcanic diagnostics
    use it explicitly. Equal weights are appropriate for HEALPix cells.
    """
    x, y = np.asarray(ensemble, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    month = np.asarray(months)
    if x.ndim != 3 or x.shape[1:] != y.shape or not x.shape[0] or not x.shape[2]:
        raise ValueError("Expected ensemble (members,time,cells) and reference (time,cells)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Decoded fields must be finite")
    if month.shape != (x.shape[1],) or not np.all((month >= 0) & (month < 12) & (month == month.astype(int))):
        raise ValueError("months must contain one integer 0..11 per timestep")
    if not 0 <= history_length < x.shape[1]:
        raise ValueError("At least one generated timestep is needed for evaluation")
    weights = np.ones(x.shape[2]) if area_weights is None else np.asarray(area_weights, dtype=np.float64)
    if weights.shape != (x.shape[2],) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("area_weights must be finite, nonnegative cell weights with a positive sum")
    weights = weights / weights.sum()
    full_x, full_y = x, y
    x, y, month = x[:, history_length:], y[history_length:], month[history_length:]
    members, length, _ = x.shape
    def average(values):
        return float(np.mean(np.sum(values * weights, axis=-1)))
    mean = x.mean(axis=0)
    # Equivalent to E|X-y| - 0.5 E|X-X'|, without quadratic member storage.
    ordered = np.sort(x, axis=0)
    coeff = (2 * np.arange(1, members + 1) - members - 1)[:, None, None]
    crps = np.abs(x - y).mean(axis=0) - (coeff * ordered).sum(axis=0) / members**2
    result = {
        'members': members, 'future_length': length, 'history_length': history_length,
        'crps': average(crps), 'ensemble_mean_rmse': average((mean - y)**2)**0.5,
        'ensemble_spread': average(x.var(axis=0, ddof=1))**0.5 if members > 1 else None,
        'seasonal_mean_bias': {str(m + 1): average(mean[month == m] - y[month == m]) for m in np.unique(month)},
        'temporal_quantile_bias': {str(q): average(np.quantile(x, q, axis=1).mean(axis=0) - np.quantile(y, q, axis=0)) for q in (0.95, 0.99)},
    }
    # Split ties uniformly across their possible ranks for a deterministic histogram.
    below, equal = (x < y).sum(axis=0), (x == y).sum(axis=0)
    result['rank_histogram_fraction'] = [average(((below <= r) & (r <= below + equal)) / (equal + 1)) for r in range(members + 1)]
    def autocorrelation(values, lag):
        centered = values - values.mean(axis=-2, keepdims=True)
        variance = np.mean(centered**2, axis=-2)
        covariance = np.mean(centered[..., :-lag, :] * centered[..., lag:, :], axis=-2)
        valid = variance > 0
        if not valid.any():
            return None
        corr = np.divide(covariance, variance, out=np.zeros_like(covariance), where=valid)
        weighted_valid = valid * weights
        return float((corr * weights).sum() / weighted_valid.sum())
    result['autocorrelation'] = {str(lag): {'generated': autocorrelation(x, lag), 'reference': autocorrelation(y, lag)} for lag in (1, 12, 24) if lag < length}
    gx, gy = np.sum(x * weights, axis=-1), np.sum(y * weights, axis=-1)
    result['global_mean_spectrum'] = {
        'cycles_per_month': np.fft.rfftfreq(length).tolist(),
        'generated': (np.abs(np.fft.rfft(gx - gx.mean(axis=-1, keepdims=True), axis=-1))**2 / length).mean(axis=0).tolist(),
        'reference': (np.abs(np.fft.rfft(gy - gy.mean()))**2 / length).tolist(),
    }
    if history_length:
        k = history_length
        jumps = full_x[:, k] - full_x[:, k - 1]
        truth_jump = full_y[k] - full_y[k - 1]
        result['boundary'] = {'generated_rms_change': average(jumps**2)**0.5,
                              'reference_rms_change': average(truth_jump**2)**0.5,
                              'ensemble_mean_jump_rmse': average((jumps.mean(axis=0) - truth_jump)**2)**0.5}
    if eruption_index is not None:
        if not 12 <= eruption_index < full_y.shape[0]:
            raise ValueError("eruption_index needs 12 preceding months and at least one response month")
        e = eruption_index
        gx, gy = np.sum(full_x * weights, axis=-1), np.sum(full_y * weights, axis=-1)
        result['volcanic_response_global_mean'] = {
            'baseline_months': 12,
            'generated': (gx[:, e:] - gx[:, e-12:e].mean(axis=1, keepdims=True)).mean(axis=0).tolist(),
            'reference': (gy[e:] - gy[e-12:e].mean()).tolist(),
        }
    return result
