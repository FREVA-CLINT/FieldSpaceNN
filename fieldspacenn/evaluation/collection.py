"""Controlled events, explicit cross-scenario experiments and nested uncertainty."""
from itertools import combinations
from contextlib import ExitStack
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import wasserstein_distance
from . import statistics as st
from .data import (Unavailable,load_forcings,scalar_aod,month_number,parse_month,event_window,
                   aggregate,select_period)
from .diagnostics import UNITS,_array_record
from .report import FAMILIES,COLLECTION,Recorder,write_json


def _regional(result,reference=False):
    key='reference_regional' if reference else 'regional'
    if key not in result.artifacts:
        raise Unavailable('Missing reference','Matching reference ensemble not supplied')
    with xr.open_dataset(result.artifacts[key]) as ds:
        return ds.load()


def _compatible_time(a,b):
    if a.time.dt.calendar!=b.time.dt.calendar or not np.array_equal(month_number(a.time),month_number(b.time)):
        raise Unavailable('Incompatible scenario metadata','Scenarios must cover identical months and calendars')
    return b.assign_coords(time=a.time)


def _compatible_initialization(a,b):
    required=('protocol','history_id')
    if not all(k in a.initialization and k in b.initialization for k in required):
        raise Unavailable('Missing forcing metadata','Control matching requires initialization.protocol and history_id')
    if any(a.initialization[k]!=b.initialization[k] for k in required):
        raise Unavailable('Incompatible scenario metadata','Initialization/history protocols differ')


def _same_thresholds(results,identifiers):
    paths=[results[key].artifacts.get('fixed_thresholds') for key in identifiers]
    if not all(paths):
        return False
    with xr.open_dataset(paths[0]) as first:
        for path in paths[1:]:
            with xr.open_dataset(path) as other:
                if not first.equals(other):
                    return False
    return True


def _control(scenario,control,g,c):
    _compatible_initialization(scenario,control)
    c=_compatible_time(g,c)
    fg=load_forcings(scenario.forcings,g.time);fc=load_forcings(control.forcings,g.time)
    if not np.allclose(fg.co2,fc.co2,rtol=0,atol=1e-8):
        raise Unavailable('Incompatible scenario metadata','Controlled response requires identical CO2 history')
    if control.eruptions or not control.forcings.get('no_eruptions',False):
        raise Unavailable('Missing control','Control must be explicitly documented as no-eruption')
    return c,fg,fc


def eruption_analyses(scenarios,results,config,rec):
    lookup={s.id:s for s in scenarios};responses={}
    for scenario in scenarios:
        if not scenario.eruptions:
            continue
        if not scenario.control_id or scenario.control_id not in results:
            rec.skip(FAMILIES['eruption'],'Missing control',f'{scenario.id}: matched no-eruption control unavailable',comparator=scenario.id)
            continue
        control=lookup[scenario.control_id]
        g=_regional(results[scenario.id]);c=_regional(results[control.id])
        try:
            c,fg,fc=_control(scenario,control,g,c)
        except Unavailable as exc:
            rec.skip(FAMILIES['eruption'],exc.status,exc.reason,comparator=scenario.id)
            continue
        response=g.mean('member')-c.mean('member')
        data=response.rename({v:f'generated_{v}' for v in UNITS})
        paired=scenario.pairing.get('physically_paired',False)
        if paired and (not scenario.pairing.get('evidence') or not np.array_equal(g.member,c.member)):
            raise ValueError('Paired response intervals require physical pairing evidence and aligned member IDs')
        for variable in UNITS:
            x=g[variable].transpose('member','time','region').values
            y=c[variable].transpose('member','time','region').values
            lo,hi=st.bootstrap_members(lambda a,b:a.mean(0)-b.mean(0),x,y,repetitions=config.bootstrap,confidence=config.confidence,seed=config.seed,paired=paired)
            if np.ndim(lo):
                data[f'generated_{variable}_ci_low']=(('time','region'),lo)
                data[f'generated_{variable}_ci_high']=(('time','region'),hi)
        try:
            r=_regional(results[scenario.id],True);rc=_compatible_time(r,_regional(results[control.id],True))
            reference=r.mean('member')-rc.mean('member')
            data=data.assign({f'reference_{v}':reference[v] for v in UNITS})
        except Unavailable:
            reference=None
        artifact=_array_record(rec,'controlled_response',f'{scenario.id}_controlled_response',data,aggregation='eruption ensemble mean minus matched control mean',n_members=len(g.member),n_time=len(g.time),comparator=scenario.id)
        responses[scenario.id]={'response':response,'reference':reference,'scenario':scenario,'artifact':artifact}
        # Spatial products are opened lazily and persisted before closing.
        if 'ensemble_mean' in results[scenario.id].artifacts and 'ensemble_mean' in results[control.id].artifacts:
            with xr.open_dataset(results[scenario.id].artifacts['ensemble_mean'],chunks='auto') as gm,xr.open_dataset(results[control.id].artifacts['ensemble_mean'],chunks='auto') as cm:
                native=gm-cm.assign_coords(time=gm.time)
                for event in scenario.eruptions:
                    for name,bounds in config.event_windows.items():
                        part,complete=event_window(native,event['date'],bounds)
                        if complete:
                            rec.array(f'{scenario.id}_{event["id"]}_{name}_response_map',part.weighted(part.time.dt.days_in_month).mean('time'),window_months=bounds,complete=True)
        dates=[parse_month(e['date']) for e in scenario.eruptions]
        for event in scenario.eruptions:
            offset=month_number(g.time)-parse_month(event['date'])
            overlapping=any(0<abs(parse_month(event['date'])-other)<252 for other in dates)
            for variable,unit in UNITS.items():
                tolerance=scenario.recovery.get(variable,{}).get('tolerance')
                sustained=scenario.recovery.get('sustained_months')
                search=scenario.recovery.get('peak_search_months')
                for region in g.region.values:
                    kw=dict(variable=variable,region=str(region),period=event['id'],comparator=scenario.id,
                            n_members=len(g.member),n_targets=len(c.member),n_time=len(g.time),
                            details={'event_overlap':overlapping,'not_independent_event_replicate':overlapping})
                    if tolerance is None or sustained is None or search is None:
                        rec.skip(['peak_response','time_to_peak','integrated_response','recovery'],'Missing forcing metadata','Configure recovery tolerances, sustained_months, peak_search_months',**kw)
                    else:
                        trajectory=response[variable].sel(region=region).values
                        try:
                            summary=st.eruption_summary(trajectory,offset,g.time.dt.days_in_month.values,peak_interval=search,tolerance=tolerance,sustained=sustained)
                            x=g[variable].sel(region=region).transpose('member','time').values
                            y=c[variable].sel(region=region).transpose('member','time').values
                            paired=scenario.pairing.get('physically_paired',False)
                            if paired and (not scenario.pairing.get('evidence') or not np.array_equal(g.member,c.member)):
                                raise ValueError('Paired uncertainty requires evidence and aligned physical member IDs')
                            take=(offset>=search[0]) & (offset<search[1])
                            ci=st.bootstrap_members(lambda a,b: np.min((a.mean(0)-b.mean(0))[take]),x,y,repetitions=config.bootstrap,confidence=config.confidence,seed=config.seed,paired=paired)
                            rec.add('peak_response',summary['peak'],units=unit,aggregation='minimum controlled response in configured search interval',ci=ci,ci_method='physically paired members' if paired else 'independent whole-member bootstrap',**kw)
                            rec.add('time_to_peak',summary['time_to_peak_months'],units='months after eruption',aggregation='minimum response timing',**kw)
                            rec.add('recovery',summary['recovery_months'],units='months after eruption',aggregation='sustained return within tolerance',**{**kw,'details':{**kw['details'],**summary,'tolerance':tolerance,'sustained_months':sustained,'right_censored':summary['censored']}})
                            for width in (60,120):
                                value=summary[f'integral_{width}m']
                                if value is None:
                                    rec.skip('integrated_response','Incomplete window',f'Missing months in [0,{width})',**kw)
                                else:
                                    rec.add('integrated_response',value,units='degC day' if variable=='temperature' else 'mm',aggregation=f'calendar-day integral [0,{width}) months',**kw)
                        except Unavailable as exc:
                            rec.skip(['peak_response','time_to_peak','integrated_response','recovery'],exc.status,exc.reason,**kw)
                for name,bounds in config.event_windows.items():
                    part,complete=event_window(response[variable],event['date'],bounds)
                    if not complete:
                        rec.skip('seasonal_response','Incomplete window',f'{name}: missing months',variable=variable,period=event['id'],comparator=scenario.id)
                        continue
                    season=aggregate(part,'seasonal')
                    _array_record(rec,'seasonal_response',f'{scenario.id}_{event["id"]}_{name}_{variable}_seasonal_response',season,units=unit,variable=variable,aggregation='complete day-weighted seasons',comparator=scenario.id)
                if reference is not None and tolerance is not None and sustained is not None and search is not None:
                    for region in reference.region.values:
                        try:
                            rs=st.eruption_summary(reference[variable].sel(region=region).values,offset,r.time.dt.days_in_month.values,peak_interval=search,tolerance=tolerance,sustained=sustained)
                            kwref=dict(variable=variable,region=str(region),period=event['id'],comparator=f'{scenario.id}:reference',n_members=len(r.member),n_targets=len(rc.member),n_time=len(r.time))
                            rec.add('peak_response',rs['peak'],units=unit,aggregation='reference controlled response peak',**kwref)
                            rec.add('time_to_peak',rs['time_to_peak_months'],units='months after eruption',aggregation='reference controlled response',**kwref)
                            rec.add('recovery',rs['recovery_months'],units='months after eruption',aggregation='reference sustained recovery',details={'right_censored':rs['censored'],'followup_months':rs['followup_months']},**kwref)
                            for width in (60,120):
                                if rs[f'integral_{width}m'] is not None:
                                    rec.add('integrated_response',rs[f'integral_{width}m'],units='degC day' if variable=='temperature' else 'mm',aggregation=f'reference calendar-day integral [0,{width}) months',**kwref)
                                else:
                                    rec.skip('integrated_response','Incomplete window',f'Reference integral [0,{width}) incomplete',**kwref)
                        except Unavailable as exc:
                            rec.skip(['peak_response','time_to_peak','integrated_response','recovery'],exc.status,exc.reason,variable=variable,comparator=f'{scenario.id}:reference')
                for pair in [('NH','SH'),('Land','Ocean')]:
                    if all(p in response.region for p in pair):
                        _array_record(rec,'response_contrasts',f'{scenario.id}_{event["id"]}_{variable}_{pair[0]}_{pair[1]}',response[variable].sel(region=pair[0])-response[variable].sel(region=pair[1]),units=unit,variable=variable,details={'contrast':pair},comparator=scenario.id)
                    else:
                        rec.skip('response_contrasts','Missing region data','Land/ocean fractions required',variable=variable,comparator=scenario.id)
                if len(g.member)>1 and len(c.member)>1:
                    pre,complete_pre=event_window(g[variable],event['date'],config.event_windows['pre'])
                    post,complete_post=event_window(g[variable],event['date'],config.event_windows['response'])
                    cp,complete_control=event_window(c[variable],event['date'],config.event_windows['response'])
                    if complete_pre and complete_post and complete_control:
                        values=xr.Dataset({'pre_variance':pre.var('member',ddof=1).mean('time'),'post_variance':post.var('member',ddof=1).mean('time'),'control_post_variance':cp.var('member',ddof=1).mean('time')})
                        _array_record(rec,'eruption_variability',f'{scenario.id}_{event["id"]}_{variable}_spread_change',values,units=f'({unit})^2',variable=variable,comparator=scenario.id)
                    else:
                        rec.skip('eruption_variability','Incomplete window','Pre/response variability windows incomplete',variable=variable,comparator=scenario.id)
                else:
                    rec.skip('eruption_variability','Insufficient members','Need >=2 members in eruption and control',variable=variable,comparator=scenario.id)
            _array_record(rec,'precipitation_temperature_response',f'{scenario.id}_{event["id"]}_tp_response',response,units='temperature: degC; precipitation: mm/day',aggregation='controlled response trajectory',comparator=scenario.id)
            ea=results[scenario.id].artifacts.get('regional_events');eb=results[control.id].artifacts.get('regional_events')
            if ea and eb and _same_thresholds(results,[scenario.id,control.id]):
                with xr.open_dataset(ea) as a,xr.open_dataset(eb) as b:
                    ap,acomp=event_window(a,event['date'],config.event_windows['response'])
                    bp,bcomp=event_window(b,event['date'],config.event_windows['response'])
                    if acomp and bcomp:
                        risk=ap.mean(('member','time'))-bp.mean(('member','time'))
                        _array_record(rec,'eruption_event_risk',f'{scenario.id}_{event["id"]}_event_risk',risk,aggregation='controlled event probability difference',comparator=scenario.id)
                    else:
                        rec.skip('eruption_event_risk','Incomplete window','Response event window incomplete',comparator=scenario.id)
            else:
                rec.skip('eruption_event_risk','Missing calibration','Both scenarios need identical calibrated event thresholds',comparator=scenario.id)
    if not any(s.eruptions for s in scenarios):
        rec.skip(FAMILIES['eruption'],'Missing forcing metadata','No eruption dates/event IDs supplied')
    return responses


def cross_scenario(scenarios,results,responses,config,rec):
    lookup={s.id:s for s in scenarios}
    # Explicit experiment definitions prevent accidental attribution to unmatched histories.
    for scenario in scenarios:
        g=_regional(results[scenario.id])
        if scenario.stationary:
            try:
                f=load_forcings(scenario.forcings,g.time)
                if not np.allclose(f.co2,f.co2.isel(time=0)) or not np.allclose(f.aod,f.aod.isel(time=0)):
                    raise Unavailable('Incompatible scenario metadata','Stationary control has time-varying forcings')
                start=scenario.spinup_end
                if start:
                    g=g.isel(time=np.flatnonzero(month_number(g.time)>=parse_month(start)))
                if len(g.time)<120:
                    raise Unavailable('Insufficient record length','Drift requires >=10 post-spinup years')
                for variable,unit in UNITS.items():
                    values=g[variable].transpose('member','region','time').values
                    trend=st.slope(values,np.arange(len(g.time))/12)
                    out=xr.Dataset({'generated_member_trend':(('member','region'),trend)},coords={'region':g.region})
                    if scenario.reference:
                        reference=_regional(results[scenario.id],True)
                        ref_start=scenario.reference.get('spinup_end',start)
                        if ref_start:
                            reference=reference.isel(time=np.flatnonzero(month_number(reference.time)>=parse_month(ref_start)))
                        common=np.intersect1d(month_number(g.time),month_number(reference.time))
                        xx=g[variable].isel(time=np.flatnonzero(np.isin(month_number(g.time),common))).transpose('member','region','time').values
                        yy=reference[variable].isel(time=np.flatnonzero(np.isin(month_number(reference.time),common))).transpose('member','region','time').values
                        if len(common)<120:
                            raise Unavailable('Insufficient record length','Need >=10 common post-spinup years for reference drift correction')
                        gx=st.slope(xx,np.arange(len(common))/12);ry=st.slope(yy,np.arange(len(common))/12)
                        out['reference_mean_trend']=('region',ry.mean(0));out['drift_error']=('region',gx.mean(0)-ry.mean(0))
                    _array_record(rec,'stationary_drift',f'{scenario.id}_{variable}_drift',out,units=unit+'/decade',variable=variable,comparator=scenario.id,details={'spinup_end':start,'reference_drift':'reported and subtracted when available'})
            except Unavailable as exc:
                rec.skip('stationary_drift',exc.status,exc.reason,comparator=scenario.id)
        experiments=list(scenario.comparisons)
        if scenario.reference and scenario.training_support.get('forcing_ranges') and not any(e['type']=='error_support_distance' for e in experiments):
            experiments.append({'type':'error_support_distance','other_id':scenario.id})
        # Explicit group identities are sufficient to discover background-state
        # experiments without asking users to enumerate every pair again.
        for other in scenarios:
            if scenario.id<other.id and scenario.id in responses and other.id in responses and scenario.volcanic_group and scenario.volcanic_group==other.volcanic_group and scenario.co2_group!=other.co2_group:
                if not any(e['type']=='background_interaction' and e.get('other_id')==other.id for e in experiments):
                    experiments.append({'type':'background_interaction','other_id':other.id})
            if scenario.id<other.id and scenario.model_variant==other.model_variant and scenario.forcings.get('no_eruptions') and other.forcings.get('no_eruptions') and scenario.co2_group!=other.co2_group:
                if not any(e['type']=='co2_only_response' and e.get('other_id')==other.id for e in experiments):
                    experiments.append({'type':'co2_only_response','other_id':other.id})
        for experiment in experiments:
            metric=experiment['type'];other_id=experiment.get('other_id')
            if metric not in FAMILIES['cross_scenario']:
                raise ValueError(f'Unknown comparison type {metric}')
            if other_id not in results:
                rec.skip(metric,'Missing scenario combination',f'Other scenario {other_id!r} unavailable',comparator=scenario.id)
                continue
            other=lookup[other_id];a=_regional(results[scenario.id]);b=_regional(results[other_id])
            try:
                b=_compatible_time(a,b)
                fa=load_forcings(scenario.forcings,a.time);fb=load_forcings(other.forcings,a.time)
                if metric=='future_forcing_perturbation':
                    _compatible_initialization(scenario,other)
                    cutoff=parse_month(experiment['cutoff']);pre=month_number(a.time)<cutoff
                    if not pre.any():
                        raise Unavailable('Insufficient record length','No pre-cutoff months')
                    if not np.allclose(fa.co2.values[pre],fb.co2.values[pre]) or not np.allclose(fa.aod.values[pre],fb.aod.values[pre]):
                        raise Unavailable('Incompatible scenario metadata','Pre-cutoff forcing histories differ')
                    if np.allclose(fa.co2,fb.co2) and np.allclose(fa.aod,fb.aod):
                        raise Unavailable('Incompatible scenario metadata','No future forcing perturbation')
                    for variable,unit in UNITS.items():
                        x=a[variable].isel(time=np.flatnonzero(pre)).transpose('member','time','region').values
                        y=b[variable].isel(time=np.flatnonzero(pre)).transpose('member','time','region').values
                        if min(len(x),len(y))<2:
                            raise Unavailable('Insufficient members','Distribution comparisons need >=2 members per scenario')
                        distance=np.array([[wasserstein_distance(x[:,t,j],y[:,t,j]) for j in range(x.shape[-1])] for t in range(x.shape[1])])
                        _array_record(rec,metric,f'{scenario.id}_{other_id}_{variable}_anticipation',xr.DataArray(distance,dims=('precutoff_month','region'),coords={'region':a.region}),units=unit,variable=variable,comparator=f'{scenario.id}:{other_id}',aggregation='same-month ensemble W1 before cutoff',details={'cutoff':experiment['cutoff'],'individual_trajectory_equality_not_required':True})
                elif metric=='background_interaction':
                    if scenario.id not in responses or other_id not in responses:
                        raise Unavailable('Missing control','Both eruption scenarios require valid matched controls')
                    if scenario.volcanic_group!=other.volcanic_group or scenario.volcanic_group is None:
                        raise Unavailable('Missing forcing metadata','Matching volcanic-history group required')
                    if not np.allclose(fa.aod,fb.aod):
                        raise Unavailable('Incompatible scenario metadata','Background interaction requires identical eruption AOD trajectories')
                    if np.all(fa.co2.values>=fb.co2.values) and np.any(fa.co2.values>fb.co2.values):
                        higher,lower=scenario.id,other_id
                    elif np.all(fb.co2.values>=fa.co2.values) and np.any(fb.co2.values>fa.co2.values):
                        higher,lower=other_id,scenario.id
                    else:
                        raise Unavailable('Incompatible scenario metadata','Crossing/equal CO2 pathways do not define an unambiguous high-minus-low interaction')
                    aresp,bresp=responses[higher],responses[lower]
                    interaction=aresp['response']-bresp['response']
                    out=interaction.rename({v:f'generated_{v}' for v in UNITS})
                    if aresp['reference'] is not None and bresp['reference'] is not None:
                        ri=aresp['reference']-bresp['reference']
                        out=out.assign({f'reference_{v}':ri[v] for v in UNITS})
                        out=out.assign({f'error_{v}':interaction[v]-ri[v] for v in UNITS})
                    else:
                        rec.skip(metric,'Missing reference','Generated interaction descriptive; reference interaction unavailable',comparator=f'{scenario.id}:{other_id}')
                    _array_record(rec,metric,f'{scenario.id}_{other_id}_interaction',out,units='temperature: degC; precipitation: mm/day',aggregation='high-CO2 controlled response minus low-CO2 controlled response',comparator=f'{scenario.id}:{other_id}',details={'zero_is_not_assumed_correct':True,'higher_id':higher,'lower_id':lower})
                    native_ids=[higher,lookup[higher].control_id,lower,lookup[lower].control_id]
                    with ExitStack() as stack:
                        native=[stack.enter_context(xr.open_dataset(results[k].artifacts['ensemble_mean'],chunks='auto')) for k in native_ids]
                        native=[d.assign_coords(time=native[0].time) for d in native]
                        imap=(native[0]-native[1])-(native[2]-native[3])
                        outmap=imap.mean('time').rename({v:f'generated_{v}' for v in UNITS})
                        if all('reference_ensemble_mean' in results[k].artifacts for k in native_ids):
                            nr=[stack.enter_context(xr.open_dataset(results[k].artifacts['reference_ensemble_mean'],chunks='auto')) for k in native_ids]
                            nr=[d.assign_coords(time=native[0].time) for d in nr]
                            refmap=((nr[0]-nr[1])-(nr[2]-nr[3])).mean('time')
                            outmap=outmap.assign({f'reference_{v}':refmap[v] for v in UNITS})
                            outmap=outmap.assign({f'error_{v}':outmap[f'generated_{v}']-refmap[v] for v in UNITS})
                        _array_record(rec,metric,f'{higher}_{lower}_interaction_map',outmap,units='temperature: degC; precipitation: mm/day',aggregation='evaluation-period native-grid mean interaction',comparator=f'{higher}:{lower}')
                elif metric=='co2_only_response':
                    if not np.allclose(fa.aod,fb.aod) or not np.allclose(fa.aod,fa.aod.isel(time=0)):
                        raise Unavailable('Incompatible scenario metadata','CO2-only comparison requires identical constant background AOD')
                    _compatible_initialization(scenario,other)
                    _array_record(rec,metric,f'{scenario.id}_{other_id}_co2_only',a.mean('member')-b.mean('member'),units='temperature: degC; precipitation: mm/day',aggregation='ensemble-mean response',comparator=f'{scenario.id}:{other_id}')
                elif metric=='history_dependence':
                    if 'matched_co2_ppm' not in experiment or 'co2_tolerance_ppm' not in experiment:
                        raise Unavailable('Missing forcing metadata','Configure matched CO2 concentrations and tolerance')
                    for ppm in experiment['matched_co2_ppm']:
                        ia=np.flatnonzero(abs(fa.co2.values-ppm)<=experiment['co2_tolerance_ppm'])
                        ib=np.flatnonzero(abs(fb.co2.values-ppm)<=experiment['co2_tolerance_ppm'])
                        if not len(ia) or not len(ib):
                            rec.skip(metric,'Incomplete window',f'No matched concentration near {ppm} ppm',comparator=f'{scenario.id}:{other_id}')
                            continue
                        delta=a.isel(time=ia).mean(('member','time'))-b.isel(time=ib).mean(('member','time'))
                        _array_record(rec,metric,f'{scenario.id}_{other_id}_history_{ppm}',delta,units='temperature: degC; precipitation: mm/day',aggregation='matched-CO2 histories, descriptive difference',details={'co2_ppm':ppm,'not_instantaneous_forcing_attribution':True},comparator=f'{scenario.id}:{other_id}')
                elif metric in ('amplitude_response','eruption_spacing'):
                    key='amplitude' if metric=='amplitude_response' else 'separation_months'
                    if scenario.id not in responses or other_id not in responses or key not in experiment:
                        raise Unavailable('Missing forcing metadata',f'Controlled responses and explicit {key} required')
                    _array_record(rec,metric,f'{scenario.id}_{other_id}_{metric}',responses[scenario.id]['response']-responses[other_id]['response'],units='temperature: degC; precipitation: mm/day',aggregation='controlled-response difference',details={key:experiment[key]},comparator=f'{scenario.id}:{other_id}')
                elif metric in ('interpolation','extrapolation','error_support_distance'):
                    ranges=scenario.training_support.get('forcing_ranges')
                    if not ranges:
                        raise Unavailable('Not verifiable','Documented training forcing ranges required')
                    distance=[]
                    for key,values in [('co2',fa.co2.values),('aod',scalar_aod(fa,scenario.forcings).values)]:
                        lo,hi=ranges[key]
                        distance.append(np.maximum.reduce([lo-values,values-hi,np.zeros_like(values)])/(hi-lo))
                    d=np.sqrt(np.sum(np.array(distance)**2,axis=0))
                    expected='extrapolation' if (d>0).any() else 'interpolation'
                    if metric in ('interpolation','extrapolation') and expected!=metric:
                        raise Unavailable('Incompatible scenario metadata',f'Documented ranges classify scenario as {expected}')
                    reference=_regional(results[scenario.id],True)
                    errors=(a.mean('member')-reference.mean('member'))
                    out=errors.assign(support_distance=('time',d))
                    _array_record(rec,metric,f'{scenario.id}_{metric}',out,units='temperature: degC; precipitation: mm/day; distance: normalized',aggregation='monthly mean errors versus marginal forcing-range distance',details={'classification':expected,'joint_support_requires_combination_metadata':True},comparator=scenario.id)
                    if metric=='error_support_distance':
                        rec.add(expected,float(np.max(d)),aggregation='maximum normalized distance outside marginal training ranges',comparator=scenario.id,details={'holdout_independence':'see split audit; range classification alone does not establish OOD generalization'})
                elif metric=='stationary_drift':
                    if not scenario.stationary:
                        raise Unavailable('Missing forcing metadata','Set stationary=True and spinup_end when appropriate')
            except Unavailable as exc:
                rec.skip(metric,exc.status,exc.reason,comparator=f'{scenario.id}:{other_id}')
    # Automatic amplitude response across documented eruption amplitudes.
    points=[]
    for scenario in scenarios:
        if scenario.id in responses:
            for event in scenario.eruptions:
                if 'amplitude' in event:
                    points.append((scenario,event))
    if len(points)>=2:
        for variable,unit in UNITS.items():
            for co2_group in set(s.co2_group for s,e in points):
                subset=[(s,e) for s,e in points if s.co2_group==co2_group]
                if co2_group is None or len(subset)<2:
                    continue
                vals=[]
                for scenario,event in subset:
                    search=scenario.recovery.get('peak_search_months')
                    if search is None:
                        continue
                    part,complete=event_window(responses[scenario.id]['response'][variable],event['date'],search)
                    if complete:
                        vals.append((event['amplitude'],float(part.sel(region='Global').min()),scenario.id,event['id']))
                if len(vals)>=2:
                    artifact=rec.root/f'amplitude_response_{variable}_{co2_group}.csv'
                    pd.DataFrame(vals,columns=['amplitude','peak_response','scenario','event']).to_csv(artifact,index=False)
                    rec.add('amplitude_response',units=unit,variable=variable,aggregation='peak controlled response versus supplied eruption amplitude',artifact=str(artifact),details={'co2_group':co2_group,'overlapping_events_not_independent':True})
    present={row['metric'] for row in rec.rows}
    for metric in FAMILIES['cross_scenario']:
        if metric not in present:
            rec.skip(metric,'Missing scenario combination','No eligible explicitly matched experiment supplied')


def nested(scenarios,results,config,rec):
    groups={}
    for s in scenarios:
        if s.co2_group is not None and s.volcanic_group is not None and s.role in ('generated','control'):
            groups.setdefault((s.co2_group,s.model_variant),[]).append(s)
    eligible=False
    for (pathway,variant),members in groups.items():
        if len(members)<2:
            continue
        eligible=True
        if len({s.volcanic_group for s in members})!=len(members):
            rec.skip(FAMILIES['nested'],'Incompatible scenario metadata','Multiple outputs for the same volcanic history; distinguish model_variant',comparator=pathway)
            continue
        probabilities=[s.history_probability for s in members]
        if any(p is None for p in probabilities) and not all(p is None for p in probabilities):
            raise ValueError('Supply probabilities for every volcanic history or none')
        weights=np.ones(len(members))/len(members) if all(p is None for p in probabilities) else np.asarray(probabilities,float)
        if np.any(weights<0) or not np.isfinite(weights).all() or weights.sum()<=0:
            raise ValueError('Invalid volcanic-history probabilities')
        weights/=weights.sum()
        histories=[_regional(results[s.id]) for s in members]
        try:
            for i in range(1,len(histories)):
                histories[i]=_compatible_time(histories[0],histories[i])
            for s in members[1:]:
                _compatible_initialization(members[0],s)
            forcings=[load_forcings(s.forcings,h.time) for s,h in zip(members,histories)]
            if any(not np.allclose(forcings[0].co2,f.co2) for f in forcings[1:]):
                raise Unavailable('Incompatible scenario metadata','CO2 must be identical within a nested pathway group')
            info={'pathway':pathway,'model_variant':variant,'history_weights':weights,
                  'weight_assumption':'explicit probabilities' if probabilities[0] is not None else 'EQUAL HISTORY PROBABILITIES ASSUMED',
                  'negative_components':'retained, sampling-unstable','bootstrap':'history support sensitivity plus member resampling'}
            for variable,unit in UNITS.items():
                arrays=[h[variable].transpose('member','time','region').values for h in histories]
                comp=st.nested_variance(arrays,weights)
                intervals=st.hierarchical_variance_interval(arrays,weights,config.bootstrap,config.confidence,config.seed)
                out=xr.Dataset({k:(('time','region'),v) for k,v in comp.items() if k!='weights'},coords={'time':histories[0].time,'region':histories[0].region})
                for k,bounds in intervals.items():
                    out[k+'_ci_low']=(('time','region'),bounds[0]);out[k+'_ci_high']=(('time','region'),bounds[1])
                out['within_fraction']=out.within/out.total.where(out.total>0)
                out['between_fraction']=out.between/out.total.where(out.total>0)
                _array_record(rec,'nested_variance',f'{pathway}_{variant}_{variable}_variance_components',out,units=f'({unit})^2',variable=variable,aggregation='conditional forcing-history variance decomposition',details=info,comparator=pathway)
            annual=[aggregate(h.temperature,'annual') for h in histories]
            for years in (10,20):
                if len(annual[0].time)<years:
                    rec.skip('negative_temperature_trend','Insufficient record length',f'Need {years} annual values',comparator=pathway)
                    continue
                risks=[];indicators=[]
                for a in annual:
                    x=a.transpose('member','region','time').values
                    windows=np.lib.stride_tricks.sliding_window_view(x,years,axis=-1)
                    slopes=st.slope(windows,np.arange(years))
                    indicator=(slopes<0).mean(2)
                    indicators.append(indicator);risks.append(indicator.mean(0))
                mixture=np.sum(weights[:,None]*np.stack(risks),axis=0)
                ci=st.hierarchical_risk_interval(indicators,weights,lambda a:a.mean(0),repetitions=config.bootstrap,confidence=config.confidence,seed=config.seed)
                _array_record(rec,'negative_temperature_trend',f'{pathway}_{variant}_{years}year_negative_trend',xr.Dataset({'probability':('region',mixture),'ci_low':('region',ci[0]),'ci_high':('region',ci[1])},coords={'region':annual[0].region}),aggregation=f'probability of negative {years}-year raw temperature trend',details={**info,'overlapping_windows':'dependent','ci_level':config.confidence},comparator=pathway)
            baseline_values=[results[s.id].provenance.get('warming_baseline_degC') for s in members]
            if any(v is None for v in baseline_values) or not np.allclose(baseline_values,baseline_values[0]):
                rec.skip('warming_crossing','Missing calibration','A shared independently calibrated warming baseline is required',comparator=pathway)
            else:
                for level in config.warming_levels:
                    for sustained in (1,config.sustained_crossing_years):
                        if len(annual[0].time)<sustained:
                            rec.skip('warming_crossing','Insufficient record length',f'Need {sustained} years',comparator=pathway)
                            continue
                        rows=[];crossed_histories=[]
                        for h,a in enumerate(annual):
                            x=a.sel(region='Global').transpose('member','time').values-baseline_values[0]
                            timing,censored=st.threshold_crossings(x,level,sustained)
                            crossed_histories.append((~censored).astype(float))
                            for j,(t,c) in enumerate(zip(timing,censored)):
                                rows.append((members[h].id,j,int(a.time.dt.year[t]),bool(c),float(weights[h]/len(x))))
                        frame=pd.DataFrame(rows,columns=['history','member_index','year','right_censored','probability_weight'])
                        artifact=rec.root/f'{pathway}_{variant}_{level}_{sustained}_crossings.csv';frame.to_csv(artifact,index=False)
                        ci=st.hierarchical_risk_interval(crossed_histories,weights,repetitions=config.bootstrap,confidence=config.confidence,seed=config.seed)
                        rec.add('warming_crossing',float(frame.loc[~frame.right_censored,'probability_weight'].sum()),aggregation=f'{sustained}-year trailing mean crossing {level} degC',artifact=str(artifact),comparator=pathway,details=info,ci=ci,ci_method='hierarchical history/member bootstrap')
            # Event indicator variance and risk mixture use the same fixed thresholds.
            event_paths=[results[s.id].artifacts.get('regional_events') for s in members]
            if all(event_paths) and _same_thresholds(results,[s.id for s in members]):
                event_data=[]
                for p in event_paths:
                    with xr.open_dataset(p) as e:
                        event_data.append(e.load())
                for event in ('hot_dry','cold_dry','cold_wet'):
                    selected=[str(r) for r in config.plot_regions if r in event_data[0].region and r!='Global'] or ['Global']
                    arrays=[e[event].sel(region=selected).all('region').transpose('member','time').values.astype(float) for e in event_data]
                    risk=sum(w*x.mean() for w,x in zip(weights,arrays))
                    ci=st.hierarchical_risk_interval(arrays,weights,repetitions=config.bootstrap,confidence=config.confidence,seed=config.seed)
                    rec.add('nested_compound_probability',float(risk),aggregation=f'simultaneous regional {event} events',comparator=pathway,details={**info,'regions':selected},ci=ci,ci_method='hierarchical history/member bootstrap')
                    component=st.nested_variance(arrays,weights)
                    _array_record(rec,'nested_variance',f'{pathway}_{variant}_{event}_indicator_variance',xr.Dataset({k:('time',v) for k,v in component.items() if k!='weights'},coords={'time':event_data[0].time}),aggregation='compound-event indicator variance',details=info,comparator=pathway)
            else:
                rec.skip('nested_compound_probability','Missing calibration','All histories require shared calibrated event indicators',comparator=pathway)
            for years in config.deficit_years:
                paths=[results[s.id].artifacts.get(f'generated_{years}year_deficit') for s in members]
                if not all(paths):
                    rec.skip('nested_precipitation_deficit','Missing calibration',f'All histories require {years}-year deficit estimates',comparator=pathway)
                else:
                    values=[]
                    for p in paths:
                        with xr.open_dataset(p) as data:
                            values.append(next(iter(data.data_vars.values())).load())
                    mixture=sum(w*v for w,v in zip(weights,values))
                    _array_record(rec,'nested_precipitation_deficit',f'{pathway}_{variant}_{years}year_deficit',mixture,aggregation=f'mixture {years}-year precipitation-deficit risk',comparator=pathway,details=info)
        except Unavailable as exc:
            rec.skip(FAMILIES['nested'],exc.status,exc.reason,comparator=pathway)
    if not eligible:
        rec.skip(FAMILIES['nested'],'Missing scenario combination','Need >=2 volcanic histories per CO2 pathway and model variant')


def run(scenarios,results,config,output_dir):
    rec=Recorder('collection',output_dir,config)
    responses=eruption_analyses(scenarios,results,config,rec)
    cross_scenario(scenarios,results,responses,config,rec)
    nested(scenarios,results,config,rec)
    return rec.finalize(collection=True),rec.artifacts
