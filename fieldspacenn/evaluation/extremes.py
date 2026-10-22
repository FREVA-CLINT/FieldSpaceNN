"""Monthly extremes only. Calibration thresholds are separate from evaluation."""
import numpy as np
import xarray as xr
from . import statistics as st
from .data import Unavailable, aggregate, regional_average, select_period
from .diagnostics import UNITS, _array_record, _clusters
from .report import FAMILIES, DAILY


def thresholds(calibration, probabilities):
    if set(np.unique(calibration.time.dt.month)) != set(range(1,13)):
        raise Unavailable('Insufficient record length','Calibration must cover every calendar month')
    if hasattr(calibration.temperature.data,'chunks'):
        calibration=calibration.chunk({'member':-1,'time':-1, **({'cell':256} if 'cell' in calibration.dims else {})})
    return xr.Dataset({name:calibration['temperature' if name in ('hot','cold') else 'precipitation'].groupby('time.month').quantile(q,dim=('member','time'),skipna=False).drop_vars('quantile',errors='ignore') for name,q in probabilities.items()})


def indicators(fields,threshold):
    th=threshold.sel(month=fields.time.dt.month)
    hot=fields.temperature>th.hot; cold=fields.temperature<th.cold
    dry=fields.precipitation<th.dry; wet=fields.precipitation>th.wet
    valid=fields.temperature.notnull() & fields.precipitation.notnull() & th.to_array().notnull().all('variable')
    return xr.Dataset({'hot':hot,'cold':cold,'dry':dry,'wet':wet,
                       'hot_dry':hot & dry,'cold_dry':cold & dry,'cold_wet':cold & wet}).where(valid)


def run(g,r,regional,reference_regional,calibration,weights,scenario,config,rec):
    rec.skip(sorted(DAILY),'Unsupported at monthly resolution','Requires daily observations; monthly fields cannot recover daily indices')
    for variable,unit in UNITS.items():
        # Calendar-month tails are descriptive; not stationary century-wide tails.
        q=g[variable].chunk({'member':-1,'time':-1,'cell':256}) if hasattr(g[variable].data,'chunks') else g[variable]
        quant=q.groupby('time.month').quantile([.1,.9],dim=('member','time'))
        _array_record(rec,'monthly_tail',f'{variable}_monthly_tail',quant,units=unit,variable=variable,aggregation='calendar-month 10th/90th percentiles',reason='Evaluation-period exposure; potentially nonstationary')
    a=regional.temperature
    years=[]; mins=[]; maxs=[]
    for y in np.unique(a.time.dt.year):
        part=a.isel(time=np.flatnonzero(a.time.dt.year.values==y))
        if len(part.time)==12:
            years.append(y); mins.append(part.min('time')); maxs.append(part.max('time'))
    if years:
        data=xr.Dataset({'minimum_monthly_mean':xr.concat(mins,xr.IndexVariable('year',years)),
                         'maximum_monthly_mean':xr.concat(maxs,xr.IndexVariable('year',years))})
        _array_record(rec,'annual_monthly_temperature_extrema','annual_monthly_temperature_extrema',data,units='degC',variable='temperature',aggregation='annual extrema of monthly regional means')
    else:
        rec.skip('annual_monthly_temperature_extrema','Insufficient record length','No complete calendar years')
    if not config.return_period_years:
        rec.skip('return_level','Not requested','Optional return_period_years not configured')
    elif not years:
        rec.skip('return_level','Insufficient record length','No complete years')
    else:
        for region in a.region.values:
            try:
                values=data.maximum_monthly_mean.sel(region=region).transpose('member','year').values
                level,lo,hi=st.return_level(values,config.return_period_years,min_years=config.min_return_years,repetitions=config.bootstrap,seed=config.seed)
                for i,period in enumerate(config.return_period_years):
                    rec.add('return_level',float(level[i]),units='degC',variable='temperature',region=str(region),aggregation=f'{period} year return of annual maximum monthly-mean temperature',n_members=len(values),n_time=values.shape[-1],ci=(lo[i],hi[i]),ci_method='whole-member GEV bootstrap')
            except Unavailable as exc:
                rec.skip('return_level',exc.status,exc.reason,region=str(region))
    conditional=[m for m in FAMILIES['extremes'] if m not in DAILY|{'monthly_tail','annual_monthly_temperature_extrema','return_level'}]
    if calibration is None:
        rec.skip(conditional,'Missing calibration','Independent reference calibration thresholds are required')
        return None
    if scenario.baseline_period:
        fixed_cal=select_period(calibration,scenario.baseline_period)
    else:
        fixed_cal=calibration
    try:
        fixed=thresholds(fixed_cal,config.thresholds)
    except Unavailable as exc:
        rec.skip(conditional,exc.status,exc.reason)
        return None
    calreg=xr.Dataset({v:regional_average(calibration[v],weights,config.min_coverage)[0] for v in UNITS}).compute()
    fixed_calreg=select_period(calreg,scenario.baseline_period) if scenario.baseline_period else calreg
    fixed_reg=thresholds(fixed_calreg,config.thresholds)
    rec.array('fixed_thresholds',fixed,definition='strict comparisons; calendar-month calibration quantiles')
    reg_events=None
    for label,source,rr in [('generated',g,regional),('reference',r,reference_regional)]:
        if source is None:
            continue
        native=indicators(source,fixed)
        events=indicators(rr,fixed_reg).compute()
        if label=='generated':
            reg_events=events
        for event in events.data_vars:
            area,coverage=regional_average(native[event].astype(float),weights,config.min_coverage)
            area=area.compute()
            _array_record(rec,'event_area_fraction',f'{label}_{event}_area',area,aggregation='fraction of native cells exceeding fixed local thresholds',comparator=label,details={'event':event})
            probability=native[event].mean(('member','time')).compute()
            _array_record(rec,'event_probability',f'{label}_{event}_probability_map',probability,aggregation='monthly cell-event frequency',comparator=label,details={'event':event})
            _array_record(rec,'regional_average_extreme',f'{label}_{event}_regional_probability',events[event].mean(('member','time')),aggregation='extreme of regional-average fields',comparator=label,details={'event':event})
            for region in rr.region.values:
                mask=events[event].sel(region=region).transpose('member','time').values
                if not np.isfinite(mask).all():
                    rec.skip(['regional_average_extreme','event_duration','event_severity'],'Missing region data',
                             'Insufficient common area or calibration support; missing values are not non-events',region=str(region),comparator=label,details={'event':event})
                    continue
                ci=st.bootstrap_members(lambda a:a.mean(),mask,repetitions=config.bootstrap,
                    confidence=config.confidence,seed=config.seed,clusters=_clusters(scenario,source.member.values,label))
                rec.add('regional_average_extreme',float(mask.mean()),region=str(region),aggregation='monthly regional-average event probability',
                        comparator=label,n_members=len(mask),n_time=mask.shape[1],ci=ci,ci_method='whole-member/lineage bootstrap',details={'event':event})
                # Severity = sum of normalized distance beyond the active threshold.
                th=fixed_reg.sel(month=rr.time.dt.month,region=region)
                temp=rr.temperature.sel(region=region); pr=rr.precipitation.sel(region=region)
                tscale=float(fixed_calreg.temperature.sel(region=region).std())
                pscale=float(fixed_calreg.precipitation.sel(region=region).std())
                components={'hot':(temp-th.hot)/tscale if tscale else temp*np.nan,
                            'cold':(th.cold-temp)/tscale if tscale else temp*np.nan,
                            'dry':(th.dry-pr)/pscale if pscale else pr*np.nan,
                            'wet':(pr-th.wet)/pscale if pscale else pr*np.nan}
                severity=sum(components[k].clip(min=0) for k in event.split('_')).transpose('member','time').values
                records=[]
                for member,row in enumerate(mask):
                    durations,starts=st.run_lengths(row)
                    for duration,start in zip(durations,starts):
                        records.append((member,start,duration,float(np.sum(severity[member,start:start+duration])),int(start==0 or start+duration==len(row))))
                array=np.asarray(records).reshape(-1,5)
                artifact=rec.array(f'{label}_{event}_{region}_durations',xr.Dataset({key:('event_index',array[:,i]) for i,key in enumerate(['member_index','start_index','duration_months','severity_standardized_months','boundary_censored'])}))
                for metric,column,units in [('event_duration',2,'months'),('event_severity',3,'standardized anomaly-months')]:
                    if len(array):
                        rec.add(metric,float(array[:,column].mean()),units=units,region=str(region),aggregation='regional-average event runs',comparator=label,artifact=artifact,n_members=len(mask),n_time=mask.shape[1],details={'event':event,'events':len(array),'censored_runs_included_as_observed_lengths':True})
                    else:
                        rec.skip(metric,'Undefined statistic','No events observed; duration/severity undefined',region=str(region),comparator=label,details={'event':event})
            # Same-month co-occurrence is a probability, not correlation of indicators.
            x=events[event].transpose('member','time','region').values.astype(float)
            co=np.einsum('mti,mtj->ij',x,x)/(x.shape[0]*x.shape[1])
            _array_record(rec,'regional_cooccurrence',f'{label}_{event}_cooccurrence',xr.DataArray(co,dims=('region','region_other'),coords={'region':rr.region,'region_other':rr.region.values}),aggregation='simultaneous monthly regional-average events',comparator=label,details={'event':event})
        ties=xr.Dataset({k:(source['temperature' if k in ('hot','cold') else 'precipitation']==fixed[k].sel(month=source.time.dt.month)).where(source['temperature' if k in ('hot','cold') else 'precipitation'].notnull()).mean(('member','time')) for k in config.thresholds})
        _array_record(rec,'threshold_ties',f'{label}_threshold_ties',ties,aggregation='probability mass exactly at threshold',comparator=label)
        seasonal=aggregate(rr.precipitation,'seasonal')
        seasonal_cal=aggregate(fixed_calreg.precipitation,'seasonal')
        if len(seasonal.time) and len(seasonal_cal.time):
            clim=seasonal_cal.groupby('time.season').mean(('member','time'))
            anomalies=seasonal.groupby('time.season')-clim
            _array_record(rec,'seasonal_rainfall_anomaly',f'{label}_seasonal_rainfall_anomaly',anomalies,units='mm/day',variable='precipitation',aggregation='day-weighted seasonal rate minus fixed seasonal mean',comparator=label)
            durations=[]
            for region in anomalies.region.values:
                for row in anomalies.sel(region=region).transpose('member','time').values:
                    durations.extend(st.run_lengths(row<0)[0].tolist())
            _array_record(rec,'multi_season_persistence',f'{label}_multi_season_persistence',xr.DataArray(durations,dims='event'),units='seasons',variable='precipitation',aggregation='consecutive below-baseline seasons',comparator=label)
        else:
            rec.skip(['seasonal_rainfall_anomaly','multi_season_persistence'],'Insufficient record length','No complete seasons in evaluation/calibration',comparator=label)
        for years in config.deficit_years:
            annual=aggregate(rr.precipitation,'annual')
            base=aggregate(fixed_calreg.precipitation,'annual').mean(('member','time'))
            if len(annual.time)<years:
                rec.skip('precipitation_deficit','Insufficient record length',f'Need {years} complete years',comparator=label)
            else:
                rolling=annual.rolling(time=years,min_periods=years).mean().isel(time=slice(years-1,None))
                risk=(rolling<(1-config.deficit_fraction)*base).mean(('member','time'))
                _array_record(rec,'precipitation_deficit',f'{label}_{years}year_deficit',risk,variable='precipitation',aggregation=f'{years} year rolling deficit probability',comparator=label,details={'deficit_fraction':config.deficit_fraction,'overlapping_windows':'dependent; not independent replicates'})
    if scenario.climate_windows:
        for name,bounds in scenario.climate_windows.items():
            c=select_period(calibration,bounds); x=select_period(g,bounds)
            if len(c.time)<12 or len(x.time)<12:
                rec.skip('climate_state_extremes','Insufficient record length','Matching state calibration/evaluation window unavailable',period=name)
            else:
                th=thresholds(c,config.thresholds)
                _array_record(rec,'climate_state_extremes',f'{name}_climate_state_event_probabilities',indicators(x,th).mean(('member','time')),aggregation='state-dependent calendar-month thresholds',details={'climate_window':name})
    else:
        rec.skip('climate_state_extremes','Missing calibration','No matched climate_windows configured for state-dependent thresholds')
    return reg_events
