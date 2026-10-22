"""External adapters and a calibration-only lagged forcing baseline."""
from pathlib import Path
import numpy as np
import xarray as xr
from . import statistics as st
from .data import (load_fields,load_forcings,scalar_aod,align_fields,regional_average,
                   select_period,month_number,Unavailable)
from .diagnostics import UNITS


def _check_separate(spec,target_spec,label):
    def paths(s):
        return set(s.get('member_paths',[s.get('path')]))
    if paths(spec)&paths(target_spec):
        raise ValueError(f'{label} cannot use held-out target stores')


def build_simple_baseline(training,training_forcings,future_forcings,members,config,output_dir):
    """Fit lazy native-grid coefficients and resample whole field blocks jointly.

    Predictors operate only on time. Small predictor inversion is eager; field
    fitting and output writing are lazy. Same sampled member/time indices serve
    temperature, precipitation and every cell.
    """
    lags=(0,1,3,6,12,24)
    ff=np.column_stack([training_forcings.co2.values,scalar_aod(training_forcings,{'aod_reduction':'latitude_band_area'}).values])
    design,start=st.forcing_design(ff,lags,training.time.dt.month.values)
    if len(design)<24:
        raise Unavailable('Insufficient record length','Baseline training needs >=48 months')
    penalty=np.eye(design.shape[1])*1e-6; penalty[0,0]=0
    inverse=np.linalg.solve(design.T@design+penalty,design.T)
    inverse=xr.DataArray(inverse,dims=('predictor','time'),coords={'time':training.time[start:]})
    coefficients=xr.Dataset({v:xr.dot(inverse,training[v].isel(time=slice(start,None)).mean('member'),dim='time') for v in UNITS}).compute()
    design_da=xr.DataArray(design,dims=('time','predictor'),coords={'time':training.time[start:]})
    fitted=xr.Dataset({v:xr.dot(design_da,coefficients[v],dim='predictor') for v in UNITS})
    residual=training.isel(time=slice(start,None))-fitted
    future=np.column_stack([future_forcings.co2.values,scalar_aod(future_forcings,{'aod_reduction':'latitude_band_area'}).values])
    pred,offset=st.forcing_design(future,lags,future_forcings.time.dt.month.values)
    time=future_forcings.time[offset:]
    future_design=xr.DataArray(pred,dims=('time','predictor'),coords={'time':time})
    mean=xr.Dataset({v:xr.dot(future_design,coefficients[v],dim='predictor') for v in UNITS})
    rng=np.random.default_rng(config.seed); simulations=[]; all_indices=[]
    for member in range(members):
        mi=[];ti=[]
        for j in range(0,len(time),config.residual_block_months):
            length=min(config.residual_block_months,len(time)-j)
            selected_member=int(rng.integers(training.sizes['member']))
            candidates=np.flatnonzero((residual.time.dt.month.values==int(time.dt.month[j])) &
                                     (np.arange(residual.sizes['time'])<=residual.sizes['time']-length))
            if not len(candidates):
                raise Unavailable('Insufficient record length','No calendar-aligned complete training residual block')
            selected_time=int(rng.choice(candidates))
            mi.extend([selected_member]*length)
            ti.extend(((selected_time+np.arange(length))%residual.sizes['time']).tolist())
        sampled=residual.isel(member=xr.DataArray(mi,dims='draw'),time=xr.DataArray(ti,dims='draw')).drop_vars(['member','time'],errors='ignore').rename(draw='time').assign_coords(time=time)
        simulations.append((sampled+mean).expand_dims(member=[member]))
        all_indices.append({'member':mi,'time':ti})
    out=xr.concat(simulations,dim='member').transpose('member','time','cell')
    for variable in UNITS:
        out[variable].attrs.update(training[variable].attrs)
    path=Path(output_dir)/'lagged_baseline.nc'
    out.to_netcdf(path)
    return out, {'path':str(path),'lag_months':lags,'discarded_initial_months':offset,
                 'resampling':'same calendar-aligned member/time blocks across variables and native space',
                 'seed':config.seed,'no_precipitation_clipping':True}


def run(g,r,regional,reference_regional,weights,scenario,config,rec):
    if not scenario.baselines:
        rec.skip(['baseline_comparison','ablation_comparison'],'Missing baseline','No separately generated baseline/ablation outputs configured')
    has_baseline=has_ablation=False
    for entry in scenario.baselines:
        kind=entry.get('kind','external'); name=entry.get('id',kind)
        metric='ablation_comparison' if kind in ('aod_removed','co2_removed') else 'baseline_comparison'
        has_ablation|=metric=='ablation_comparison'; has_baseline|=metric=='baseline_comparison'
        if r is None:
            rec.skip([metric,'crps_skill'],'Missing reference','Matching targets required',comparator=name)
            continue
        baseline=load_fields(entry['dataset'],config)
        try:
            _,b,align=align_fields(g,baseline,config)
            yy=r.sel(time=b.time)
            for variable,unit in UNITS.items():
                x=regional[variable].sel(time=b.time).transpose('member','time','region').values
                z=regional_average(b[variable],weights,config.min_coverage)[0].transpose('member','time','region').values
                y=regional_average(yy[variable],weights,config.min_coverage)[0].transpose('member','time','region').values
                m=min(len(x),len(z));rng=np.random.default_rng(config.seed)
                bs=[];gs=[]
                for _ in range(config.size_matching_repetitions):
                    bs.append(st.crps(z[rng.choice(len(z),m,False)],y).mean())
                    gs.append(st.crps(x[rng.choice(len(x),m,False)],y).mean())
                rec.add(metric,float(np.mean(bs)),units=unit,variable=variable,comparator=name,aggregation='matched-size monthly regional CRPS',n_members=m,n_targets=len(y),n_time=x.shape[1],details={'kind':kind,'subsampling_repetitions':config.size_matching_repetitions,'alignment':align})
                rec.add('crps_skill',float(1-np.mean(gs)/np.mean(bs)) if np.mean(bs)>0 else np.nan,variable=variable,comparator=name,aggregation='1 - generated/baseline CRPS',n_members=m,n_targets=len(y))
        finally:
            baseline.close()
    if scenario.baselines and not has_baseline:
        rec.skip('baseline_comparison','Missing baseline','Only ablation outputs supplied')
    if scenario.baselines and not has_ablation:
        rec.skip('ablation_comparison','Missing baseline','No separately generated forcing ablation supplied')
    if scenario.training is None:
        rec.skip('simple_baseline','Missing calibration','Separate training dataset and training forcing paths required')
    elif r is None:
        rec.skip('simple_baseline','Missing reference','No held-out targets for baseline evaluation')
    else:
        _check_separate(scenario.training,scenario.reference,'Baseline training')
        train=load_fields(scenario.training,config)
        try:
            train_forcing=load_forcings(scenario.training.get('forcings',{}),train.time)
            future=load_forcings(scenario.forcings,g.time)
            baseline,metadata=build_simple_baseline(train,train_forcing,future,len(g.member),config,rec.root)
            for variable,unit in UNITS.items():
                z=regional_average(baseline[variable],weights,config.min_coverage)[0].transpose('member','time','region').values
                y=reference_regional[variable].sel(time=baseline.time).transpose('member','time','region').values
                x=regional[variable].sel(time=baseline.time).transpose('member','time','region').values
                bs=float(st.crps(z,y).mean());gs=float(st.crps(x,y).mean())
                rec.add('simple_baseline',bs,units=unit,variable=variable,aggregation='native-field baseline; monthly regional CRPS',details=metadata,n_members=len(z),n_targets=len(y),n_time=z.shape[1])
                rec.add('crps_skill',1-gs/bs if bs>0 else np.nan,variable=variable,comparator='lagged_forcing_joint_blocks',aggregation='1 - generated/baseline CRPS')
        except Unavailable as exc:
            rec.skip('simple_baseline',exc.status,exc.reason)
        finally:
            train.close()
    if scenario.reconstruction is None:
        rec.skip('compression_error','Missing reconstruction','No reconstruction with identified original supplied')
    else:
        spec=scenario.reconstruction
        if 'original' not in spec:
            rec.skip('compression_error','Missing reference','Reconstruction needs its original dataset and exact member correspondence')
            return
        reconstruction=load_fields(spec['dataset'],config);original=load_fields(spec['original'],config)
        try:
            x,y,_=align_fields(reconstruction,original,config)
            if not np.array_equal(x.member.values,y.member.values):
                raise ValueError('Reconstruction member IDs must match originals exactly')
            for variable,unit in UNITS.items():
                diff=x[variable]-y[variable]
                rec.add('compression_error',float(np.sqrt((diff**2).mean().compute())),units=unit,variable=variable,aggregation='paired original-reconstruction native-grid RMSE',n_members=len(x.member),n_time=len(x.time),details={'not_generation_error':True})
        finally:
            reconstruction.close();original.close()
