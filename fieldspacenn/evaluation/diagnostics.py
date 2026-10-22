"""Scenario diagnostics. Regional products fit in memory; native maps are chunked."""
import numpy as np
import xarray as xr
from scipy.stats import wasserstein_distance
from . import statistics as st
from .data import (Unavailable, aggregate, regional_average, select_period, month_number,
                   parse_month, load_fields, align_fields)
from .report import FAMILIES, DAILY


UNITS = {'temperature': 'degC', 'precipitation': 'mm/day'}


def _ci(config, fn, x, y=None, clusters=None, target_clusters=None):
    return st.bootstrap_members(fn, x, y, repetitions=config.bootstrap,
                                confidence=config.confidence, seed=config.seed, clusters=clusters,target_clusters=target_clusters)


def _clusters(scenario, members, source='generated'):
    info = scenario.lineage.get(source, {})
    if info:
        if any(str(m) not in info for m in members):
            raise ValueError(f'{source} lineage must identify all evaluated members')
        return [info[str(m)] for m in members]
    return None


def _years(time):
    return np.asarray(time.dt.year+(time.dt.month-.5)/12., float)


def _array_record(rec, metric, name, array, *, units='1', variable='both', aggregation='monthly',
                  n_members=0, n_targets=0, n_time=0, reason='', details=None, comparator=''):
    sizes=array.sizes
    details={**(details or {}),'artifact_dimensions':dict(sizes)}
    n_members=n_members or sizes.get('member',0)
    n_targets=n_targets or sizes.get('target_member',0)
    n_time=n_time or sizes.get('time',sizes.get('time_index',0))
    artifact = rec.array(name, array, units=units, metric=metric, scenario=rec.id)
    # Empty event populations are unavailable summaries, not computed successes.
    if any(size==0 for size in sizes.values()):
        rec.skip(metric,'Undefined statistic','No samples for this array-valued diagnostic',units=units,
                 variable=variable,aggregation=aggregation,artifact=artifact,comparator=comparator)
        return artifact
    rec.add(metric, units=units, variable=variable, aggregation=aggregation,
            n_members=n_members, n_targets=n_targets, n_time=n_time, artifact=artifact,
            reason=reason, details=details, comparator=comparator)
    return artifact


def forced(g, r, regional, reference_regional, weights, scenario, config, rec):
    clusters = _clusters(scenario, g.member.values)
    target_clusters=_clusters(scenario,r.member.values,'reference') if r is not None else None
    for variable, unit in UNITS.items():
        series = regional[variable]
        ens = series.mean('member')
        trajectory = xr.Dataset({'ensemble_mean': ens, 'q05': series.quantile(.05, 'member').drop_vars('quantile'),
                                 'q95': series.quantile(.95, 'member').drop_vars('quantile')})
        if reference_regional is not None:
            y = reference_regional[variable]
            trajectory['reference_mean'] = y.mean('member')
            trajectory['reference_q05'] = y.quantile(.05, 'member').drop_vars('quantile')
            trajectory['reference_q95'] = y.quantile(.95, 'member').drop_vars('quantile')
        _array_record(rec, 'trajectory', f'{variable}_trajectory', trajectory, units=unit, variable=variable,
                      n_members=len(g.member), n_time=len(g.time), details={'bands': '5th–95th ensemble quantiles; not confidence intervals'})
        # Historical calibration baseline is distinct from the period whose
        # generated/reference seasonal cycle is being described.
        clim_source = select_period(series, scenario.periods.get('climatology'))
        if len(clim_source.time) < 12:
            rec.skip(['climatology', 'seasonal_cycle_amplitude', 'seasonal_cycle_phase'], 'Insufficient record length', 'Need all twelve calendar months', variable=variable)
        else:
            climatology = clim_source.groupby('time.month').mean(('member', 'time'))
            _array_record(rec, 'climatology', f'{variable}_climatology', climatology, units=unit, variable=variable,
                          aggregation='calendar-month climatology', reason='Evaluation-period descriptive climatology' if not scenario.periods.get('climatology') else 'Explicit climatology period')
            for region in series.region.values:
                cycle = climatology.sel(region=region)
                rec.add('seasonal_cycle_amplitude', float(cycle.max()-cycle.min()), units=unit, variable=variable, region=str(region), aggregation='peak-to-trough climatology', n_time=len(clim_source.time))
                rec.add('seasonal_cycle_phase', float(cycle.month.values[int(cycle.argmax('month'))]) if np.isfinite(cycle).all() else np.nan,
                        units='calendar month', variable=variable, region=str(region), aggregation='month of climatological maximum', n_time=len(clim_source.time))
            if reference_regional is not None:
                reference_clim_source=select_period(reference_regional[variable],scenario.periods.get('climatology'))
                if len(reference_clim_source.time)>=12:
                    reference_clim=reference_clim_source.groupby('time.month').mean(('member','time'))
                    _array_record(rec,'climatology',f'{variable}_reference_climatology',reference_clim,units=unit,variable=variable,aggregation='calendar-month climatology',comparator='reference')
                    for region in reference_clim.region.values:
                        cycle=reference_clim.sel(region=region)
                        rec.add('seasonal_cycle_amplitude',float(cycle.max()-cycle.min()),units=unit,variable=variable,region=str(region),aggregation='peak-to-trough climatology',comparator='reference',n_time=len(reference_clim_source.time))
                        rec.add('seasonal_cycle_phase',float(cycle.month.values[int(cycle.argmax('month'))]) if np.isfinite(cycle).all() else np.nan,units='calendar month',variable=variable,region=str(region),aggregation='month of climatological maximum',comparator='reference',n_time=len(reference_clim_source.time))
        for frequency in ('seasonal', 'annual', '5year', '10year', '20year'):
            a = aggregate(g[variable].mean('member'), frequency)
            if not len(a.time):
                rec.skip('mean_map', 'Insufficient record length', 'No complete aggregates', variable=variable, aggregation=frequency)
            else:
                maps = xr.Dataset({'generated': a})
                if r is not None:
                    maps['reference'] = aggregate(r[variable].mean('member'), frequency)
                    maps['difference'] = maps.generated-maps.reference
                _array_record(rec, 'mean_map', f'{variable}_{frequency}_maps', maps, units=unit, variable=variable, aggregation=frequency, n_time=len(a.time))
        for frequency in ('monthly', 'seasonal', 'annual', '5year', '10year', '20year'):
            a = aggregate(series, frequency)
            if len(a.time) < 2:
                rec.skip('trend', 'Insufficient record length', 'Need at least two complete time aggregates', variable=variable, aggregation=frequency)
                continue
            yy = _years(a.time)
            for region in a.region.values:
                values = a.sel(region=region).transpose('member', 'time').values
                member_slopes = st.slope(values, yy)
                ci = _ci(config, lambda z: st.slope(z.mean(0), yy), values, clusters=clusters)
                rec.add('trend', float(np.mean(member_slopes)), units=unit+'/decade', aggregation=frequency,
                        variable=variable, region=str(region), n_members=len(values), n_time=len(yy), ci=ci,
                        ci_method='whole-member/lineage bootstrap')
                if reference_regional is not None:
                    b = aggregate(reference_regional[variable], frequency).sel(region=region).transpose('member', 'time').values
                    reference_ci=_ci(config,lambda z:st.slope(z.mean(0),yy),b,clusters=target_clusters)
                    rec.add('trend',float(st.slope(b.mean(0),yy)),units=unit+'/decade',variable=variable,region=str(region),aggregation=frequency,comparator='reference',n_members=len(b),n_time=len(yy),ci=reference_ci,ci_method='whole-member/lineage bootstrap')
                    delta = float(st.slope(values.mean(0)-b.mean(0), yy))
                    ci = _ci(config, lambda x,y: st.slope(x.mean(0)-y.mean(0), yy), values, b,clusters,target_clusters)
                    rec.add('trend_error', delta, units=unit+'/decade', variable=variable, region=str(region), aggregation=frequency,
                            n_members=len(values), n_targets=len(b), n_time=len(yy), ci=ci, ci_method='independent whole-member bootstrap')
        for name, pair in [('land_ocean_contrast', ('Land', 'Ocean')), ('hemispheric_contrast', ('NH', 'SH'))]:
            if all(p in series.region for p in pair):
                diff = series.sel(region=pair[0])-series.sel(region=pair[1])
                _array_record(rec, name, f'{variable}_{name}', diff, units=unit, variable=variable,
                              details={'subtraction': f'{pair[0]} minus {pair[1]}'})
                if reference_regional is not None:
                    reference_diff=reference_regional[variable].sel(region=pair[0])-reference_regional[variable].sel(region=pair[1])
                    _array_record(rec,name,f'{variable}_reference_{name}',reference_diff,units=unit,variable=variable,comparator='reference',details={'subtraction':f'{pair[0]} minus {pair[1]}'})
            else:
                rec.skip(name, 'Missing region data', 'Land fraction mask not supplied', variable=variable)
        annual = aggregate(series, 'annual')
        global_temp = aggregate(regional.temperature.sel(region='Global'), 'annual').mean('member')
        if len(annual.time) < 10 or float(global_temp.var()) == 0:
            rec.skip('temperature_change_pattern', 'Insufficient record length', 'Need >=10 complete years and varying global mean temperature', variable=variable)
        else:
            coefficients = ((annual.mean('member')-annual.mean(('member', 'time')))*(global_temp-global_temp.mean())).sum('time')/((global_temp-global_temp.mean())**2).sum()
            _array_record(rec, 'temperature_change_pattern', f'{variable}_change_pattern', coefficients, units=unit+'/degC', variable=variable, aggregation='annual regression')
        if r is None:
            rec.skip(['mean_bias', 'mae', 'rmse', 'pattern_correlation', 'trend_error', 'significance_map'], 'Missing reference', 'No matching reference ensemble configured', variable=variable)
            continue
        difference = g[variable].mean('member')-r[variable].mean('member')
        diagnostic = xr.Dataset({'bias': difference.mean('time'), 'mae': abs(difference).mean('time'),
                                'rmse': np.sqrt((difference**2).mean('time'))}).compute()
        rec.array(f'{variable}_error_maps', diagnostic, units=unit)
        for region in weights.region.values:
            w = weights.sel(region=region)
            area_bias = float(diagnostic.bias.weighted(w).mean('cell'))
            area_mae = float(diagnostic.mae.weighted(w).mean('cell'))
            area_rmse = float(np.sqrt((diagnostic.rmse**2).weighted(w).mean('cell')))
            x = series.sel(region=region).transpose('member', 'time').values
            y = reference_regional[variable].sel(region=region).transpose('member', 'time').values
            for metric, value in [('mean_bias', area_bias), ('mae', area_mae), ('rmse', area_rmse)]:
                # Bias commutes with spatial averaging; MAE/RMSE do not. Their
                # map estimands are not assigned CIs for a different estimand.
                ci = _ci(config, lambda a,b: np.mean(a.mean(0)-b.mean(0)), x, y,clusters,target_clusters) if metric == 'mean_bias' else None
                rec.add(metric, value, units=unit, variable=variable, region=str(region), aggregation='native-cell monthly errors, area then time',
                        n_members=len(x), n_targets=len(y), n_time=x.shape[1], ci=ci, ci_method='independent whole-member bootstrap',
                        artifact=rec.artifacts[f'{variable}_error_maps'])
        gm = g[variable].mean(('member', 'time')).values
        rm = r[variable].mean(('member', 'time')).values
        valid = np.isfinite(gm) & np.isfinite(rm)
        rec.add('pattern_correlation', float(st.correlation(gm[valid], rm[valid])), variable=variable,
                aggregation='native-grid climatological spatial pattern', n_time=len(g.time))
        if not config.significance_maps:
            rec.skip('significance_map','Not requested','significance_maps=False',variable=variable)
        elif not scenario.reference_independent:
            rec.skip('significance_map','Not verifiable','Independent reference targets must be documented',variable=variable)
        elif len(g.member) < 2 or len(r.member) < 2:
            rec.skip('significance_map','Insufficient members','Need independent member or lineage replicates',variable=variable)
        else:
            from scipy.stats import ttest_ind
            # Independent trajectory means, not grid cells/time as replicates.
            def replicates(a,groups):
                x=a.mean('time').values
                if groups is not None:
                    groups=np.asarray(groups)
                    x=np.stack([x[groups==label].mean(0) for label in np.unique(groups)])
                return x
            x=replicates(g[variable],clusters);y=replicates(r[variable],target_clusters)
            if min(len(x),len(y))<2:
                rec.skip('significance_map','Insufficient independent information','Need >=2 independent lineages per ensemble',variable=variable)
                continue
            p = ttest_ind(x,y,axis=0,equal_var=False).pvalue
            q, reject = st.benjamini_yekutieli(p, config.fdr_alpha)
            _array_record(rec,'significance_map',f'{variable}_significance', xr.Dataset({'p': ('cell', p), 'q_BY': ('cell', q), 'reject': ('cell', reject)}, coords={'cell':g.cell}),variable=variable,aggregation='Welch test of trajectory/lineage means; BY across cells',n_members=len(x),n_targets=len(y))


def calibration(g, r, regional, reference_regional, scenario, config, rec):
    if r is None:
        rec.skip(FAMILIES['calibration'], 'Missing reference', 'No matching reference ensemble')
        return
    clusters=_clusters(scenario,g.member.values)
    target_clusters=_clusters(scenario,r.member.values,'reference')
    for variable, unit in UNITS.items():
        for frequency in ('monthly', 'seasonal', 'annual'):
            aa = aggregate(regional[variable], frequency)
            bb = aggregate(reference_regional[variable], frequency)
            if not len(aa.time):
                rec.skip(['crps', 'coverage', 'interval_width'], 'Insufficient record length', 'No complete aggregates', variable=variable, aggregation=frequency)
                continue
            for region in aa.region.values:
                x = aa.sel(region=region).transpose('member', 'time').values
                y = bb.sel(region=region).transpose('member', 'time').values
                kwargs = dict(variable=variable, region=str(region), units=unit, aggregation=frequency,
                              n_members=len(x), n_targets=len(y), n_time=x.shape[-1])
                ci = _ci(config, lambda a,b: st.crps(a,b).mean(), x, y,clusters,target_clusters)
                rec.add('crps', float(st.crps(x, y).mean()), ci=ci, ci_method='independent whole-member bootstrap', **kwargs)
                if frequency == 'monthly':
                    if config.fair_crps and len(x) > 1:
                        rec.add('fair_crps', float(st.crps(x,y,True).mean()), **kwargs)
                    else:
                        rec.skip('fair_crps', 'Insufficient members' if len(x)<2 else 'Not verifiable', 'Requires >=2 independent conditional forecast draws and fair_crps=True', **kwargs)
                    ranks = st.rank_histogram(x,y)
                    _array_record(rec, 'rank_histogram', f'{variable}_{region}_ranks', xr.DataArray(ranks, dims='rank'), variable=variable,
                                  details={'ties': 'fractionally allocated over all admissible ranks'})
                for probability in config.intervals:
                    covered, width = st.coverage_width(x,y,probability)
                    ci = _ci(config, lambda a,b: st.coverage_width(a,b,probability)[0].mean(), x, y,clusters,target_clusters)
                    kw = {**kwargs, 'units': '1', 'details': {'nominal': probability}}
                    rec.add('coverage', float(covered.mean()), ci=ci, ci_method='independent whole-member bootstrap', **kw)
                    rec.add('interval_width', float(width.mean()), details={'nominal': probability}, **kwargs)
                if len(x)>1 and len(y)>1:
                    vx = np.var(x,axis=0,ddof=1).mean(); vy = np.var(y,axis=0,ddof=1).mean()
                    ratio = vx/vy if vy>0 else np.nan
                    rec.add('variance_ratio', float(ratio), **{**kwargs,'units':'1'}, details={'correction': 'unbiased sample variance ddof=1'})
                    rec.add('spread_ratio', float(np.sqrt(ratio)), **{**kwargs,'units':'1'})
                else:
                    rec.skip(['variance_ratio','spread_ratio'], 'Insufficient members', 'Both ensembles require >=2 members', **kwargs)
        # Native-grid proper score and spread maps, bounded cell blocks.
        maps = {k: [] for k in ('crps','variance_ratio')}
        chunk_cells = max(1, min(256, config.max_memory_mb*1024**2//(8*len(g.time)*(len(g.member)+len(r.member))*4)))
        for first in range(0, len(g.cell), chunk_cells):
            x = g[variable].isel(cell=slice(first,first+chunk_cells)).values
            y = r[variable].isel(cell=slice(first,first+chunk_cells)).values
            maps['crps'].append(st.crps(x,y).mean((0,1)))
            if len(x)>1 and len(y)>1:
                vy = y.var(0,ddof=1).mean(0)
                maps['variance_ratio'].append(np.divide(x.var(0,ddof=1).mean(0), vy, out=np.full(vy.shape,np.nan),where=vy>0))
            else:
                maps['variance_ratio'].append(np.full(x.shape[-1],np.nan))
        rec.array(f'{variable}_calibration_maps', xr.Dataset({k: ('cell', np.concatenate(v)) for k,v in maps.items()}, coords={'cell':g.cell}), crps_units=unit)
        # Pool only within declared climate windows AND calendar strata.
        windows = scenario.climate_windows or {'evaluation_nonstationary_descriptive': [str(g.time.values[0])[:7], f'{int(g.time.dt.year[-1])+1:04d}-01']}
        for window, bounds in windows.items():
            a, b = select_period(regional[variable],bounds), select_period(reference_regional[variable],bounds)
            if not len(a.time):
                rec.skip(['quantile_error','wasserstein','qq_distribution'], 'Incomplete window','No data in climate window',variable=variable,period=window)
                continue
            for month in np.unique(a.time.dt.month):
                aa = a.isel(time=np.flatnonzero(a.time.dt.month.values==month))
                bb = b.isel(time=np.flatnonzero(b.time.dt.month.values==month))
                for region in a.region.values:
                    x,y = aa.sel(region=region).values.ravel(),bb.sel(region=region).values.ravel()
                    kw = dict(variable=variable,region=str(region),period=window,units=unit,aggregation=f'calendar month {month}',n_members=len(g.member),n_targets=len(r.member),n_time=len(aa.time))
                    rec.add('wasserstein', float(wasserstein_distance(x,y)) if np.isfinite(x).all() and np.isfinite(y).all() else np.nan, **kw)
                    quantiles=[]
                    for q in config.quantiles:
                        # Count conservatively by member x nonoverlapping block-years.
                        support=min(len(g.member),len(r.member))*max(1,len(aa.time)//max(1,config.residual_block_months//12))
                        if support*min(q,1-q)<config.min_tail_observations:
                            rec.skip('quantile_error','Insufficient independent information',f'Tail support {support*min(q,1-q):.1f} < {config.min_tail_observations}',details={'quantile':q},**kw)
                        else:
                            rec.add('quantile_error',float(np.quantile(x,q)-np.quantile(y,q)),details={'quantile':q},**kw)
                        quantiles.append([np.quantile(y,q),np.quantile(x,q)])
                    _array_record(rec,'qq_distribution',f'{variable}_{region}_{window}_month{month}_qq',xr.DataArray(quantiles,dims=('quantile','source'),coords={'quantile':config.quantiles,'source':['reference','generated']}),units=unit,variable=variable,aggregation=kw['aggregation'])
        x = regional[variable].transpose('member','time','region').values
        y = reference_regional[variable].transpose('member','time','region').values
        m = min(len(x),len(y)); rng=np.random.default_rng(config.seed)
        scores=[st.crps(x[rng.choice(len(x),m,False)],y[rng.choice(len(y),m,False)]).mean() for _ in range(config.size_matching_repetitions)]
        rec.add('size_matched_crps',float(np.mean(scores)),units=unit,variable=variable,aggregation='monthly regional means; equal sizes',n_members=m,n_targets=m,n_time=len(g.time),details={'subsampling_repetitions':len(scores),'subsampling_sd':np.std(scores),'not_a_confidence_interval':True})
        if not scenario.reference_independent:
            rec.skip('reference_benchmark','Not verifiable','Independent held-out reference membership not documented',variable=variable)
        else:
            try:
                benchmark=st.reference_benchmark(y,min(len(x),len(y)//2),max(1,len(y)//2),config.size_matching_repetitions,config.seed)
                _array_record(rec,'reference_benchmark',f'{variable}_reference_benchmark',xr.Dataset({k:('repeat',v) for k,v in benchmark.items()}),units=unit,variable=variable,details={'finite_sample_benchmark':True,'forecast_size':min(len(x),len(y)//2),'target_size':max(1,len(y)//2)})
            except Unavailable as exc:
                rec.skip('reference_benchmark',exc.status,exc.reason,variable=variable)
    # Filled by the baseline adapter if eligible.
    if not scenario.baselines:
        rec.skip('crps_skill','Missing baseline','No separately supplied or calibrated baseline')


def variability(regional, reference_regional, scenario, config, rec):
    if len(regional.member)<2:
        rec.skip(FAMILIES['variability'],'Insufficient members','At least two generated members required to estimate residual variability')
        return
    for variable,unit in UNITS.items():
        for label,source in [('generated',regional),('reference',reference_regional)]:
            if source is None or len(source.member)<2:
                continue
            raw=source[variable]; residual=raw-raw.mean('member')
            for frequency in ('monthly','seasonal','annual'):
                values=aggregate(residual,frequency)
                if not len(values.time):
                    rec.skip('residual_variance','Insufficient record length','No complete aggregates',variable=variable,aggregation=frequency,comparator=label)
                    continue
                variance=values.var('member',ddof=1).mean('time')
                _array_record(rec,'residual_variance',f'{variable}_{label}_{frequency}_variance',variance,units=f'({unit})^2',variable=variable,aggregation=frequency,comparator=label,details={'finite_ensemble_correction':'ddof=1'})
            for region in raw.region.values:
                x=residual.sel(region=region).transpose('member','time').values
                kwargs=dict(variable=variable,region=str(region),n_members=len(x),n_time=x.shape[-1],comparator=label)
                for frequency,lags in [('monthly',config.monthly_lags),('annual',config.annual_lags)]:
                    a=aggregate(residual.sel(region=region),frequency).transpose('member','time').values
                    for lag in lags:
                        if a.shape[-1]<max(3*lag,8):
                            rec.skip('autocorrelation','Insufficient record length',f'Need >= max(3*lag,8) observations at lag {lag}',aggregation=frequency,details={'lag':lag},**kwargs)
                        else:
                            corr=st.autocorrelation(a,[lag])[:,0]
                            ci=_ci(config,lambda z: np.nanmean(st.autocorrelation(z,[lag])),a,clusters=_clusters(scenario,source.member.values,label))
                            rec.add('autocorrelation',float(np.nanmean(corr)),ci=ci,ci_method='whole-member bootstrap',aggregation=frequency,details={'lag':lag},**kwargs)
                try:
                    f,p=st.welch_spectrum(x,config.welch_segment_months)
                    correction=len(x)/(len(x)-1);p=p*correction
                    _array_record(rec,'temporal_spectrum',f'{variable}_{label}_{region}_psd',xr.DataArray(p,dims=('member','frequency'),coords={'frequency':f}),units=f'({unit})^2/(cycles/year)',variable=variable,details={'region':str(region),'estimator':'Welch Hann, 50% overlap; member spectra kept separate','finite_ensemble_power_factor':correction},comparator=label)
                except Unavailable as exc:
                    rec.skip('temporal_spectrum',exc.status,exc.reason,**kwargs)
                for width in config.averaging_months:
                    n=x.shape[-1]//width
                    if n<config.min_independent_blocks:
                        rec.skip('averaging_variance','Insufficient record length',f'Only {n} nonoverlapping windows; require {config.min_independent_blocks}',aggregation=f'{width} month means',**kwargs)
                    else:
                        b=st.block_means(x,width,residual.time.dt.days_in_month.values)
                        # Ensemble centering reduces marginal residual variance by (M-1)/M.
                        value=b.var(0,ddof=1).mean()
                        rec.add('averaging_variance',float(value),units=f'({unit})^2',aggregation=f'{width} month means',n_independent=n,details={'overlap':False,'day_weighted':True},**kwargs)
                for years in config.trend_years:
                    width=years*12; n=x.shape[-1]//width
                    for metric,data in [('raw_trend_distribution',raw.sel(region=region).transpose('member','time').values),('residual_trend_distribution',x)]:
                        if n<2:
                            rec.skip(metric,'Insufficient record length','Need at least two nonoverlapping trend windows',aggregation=f'{years} year trends',**kwargs)
                        else:
                            slopes=st.slope(data[:,:n*width].reshape(len(data),n,width),np.arange(width)/12)
                            _array_record(rec,metric,f'{variable}_{label}_{region}_{metric}_{years}',xr.DataArray(slopes,dims=('member','window')),units=unit+'/decade',variable=variable,aggregation=f'nonoverlapping {years} year trends',comparator=label)
                for direction in ('positive','negative'):
                    durations=np.concatenate([st.run_lengths(row>0 if direction=='positive' else row<0)[0] for row in x])
                    _array_record(rec,'anomaly_persistence',f'{variable}_{label}_{region}_{direction}_persistence',xr.DataArray(durations,dims='event'),units='months',variable=variable,details={'sign':direction,'region':str(region),'edge_runs':'observed length, boundary-censored'})
            if scenario.climate_windows:
                for name,bounds in scenario.climate_windows.items():
                    part=select_period(residual,bounds)
                    if len(part.time)<12:
                        rec.skip('climate_state_variability','Insufficient record length','Need >=12 months',variable=variable,period=name)
                    else:
                        _array_record(rec,'climate_state_variability',f'{variable}_{label}_{name}_variance',part.var('member',ddof=1).mean('time'),units=f'({unit})^2',variable=variable,comparator=label,details={'window':name})
            else:
                rec.skip('climate_state_variability','Missing forcing metadata','Configure climate_windows for climate-state comparisons',variable=variable,comparator=label)
