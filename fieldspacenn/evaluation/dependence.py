"""Native spatial structure and calibrated multivariate scores."""
import numpy as np
import xarray as xr
from . import statistics as st
from .data import Unavailable, regional_average, aggregate
from .diagnostics import UNITS, _array_record
from .report import FAMILIES


def _snapshots(a, budget, seed):
    rng=np.random.default_rng(seed)
    count=min(budget,a.sizes['member']*a.sizes['time'])
    flat=np.sort(rng.choice(a.sizes['member']*a.sizes['time'],count,False))
    values=a.isel(member=xr.DataArray(flat//a.sizes['time'],dims='sample'),
                  time=xr.DataArray(flat%a.sizes['time'],dims='sample')).transpose('sample','cell').values
    return values,flat.tolist()


def run(g,r,regional,reference_regional,calibration,weights,scenario,config,rec):
    import healpy as hp
    if len(g.member)<2:
        rec.skip(FAMILIES['dependence'],'Insufficient members','Residual dependence needs >=2 members')
        return
    lon,lat=hp.pix2ang(config.nside,g.cell.values.astype(int),nest=True,lonlat=True)
    source_sets=[('generated',g,regional),('reference',r,reference_regional)]
    for label,source,regional_source in source_sets:
        if source is None or len(source.member)<2:
            continue
        residual=source-source.mean('member')
        rr=regional_source-regional_source.mean('member')
        for variable,unit in UNITS.items():
            _array_record(rec,'residual_variance',f'{variable}_{label}_residual_variance_map',source[variable].var('member',ddof=1).mean('time').compute(),
                          units=f'({unit})^2',variable=variable,aggregation='mean monthly native-cell conditional variance',comparator=label,n_members=len(source.member),n_time=len(source.time))
            samples,indices=_snapshots(residual[variable],config.spatial_snapshots,config.seed)
            try:
                correction=len(source.member)/(len(source.member)-1)
                power=st.spatial_spectrum(samples,config.nside,config.lmax)*correction
                _array_record(rec,'spatial_spectrum',f'{variable}_{label}_spatial_spectrum',xr.DataArray(power,dims=('sample','degree')),units=f'({unit})^2',variable=variable,comparator=label,details={'sample_indices':indices,'lmax':config.lmax,'monopole_removed':True,'pixel_window_deconvolution':False,'ordering':'NESTED -> RING','finite_ensemble_power_factor':correction})
            except Unavailable as exc:
                rec.skip('spatial_spectrum',exc.status,exc.reason,variable=variable,comparator=label)
            distance,semivariance,counts=st.sampled_semivariogram(samples,lon,lat,config.spatial_pairs,config.seed)
            correction=len(source.member)/(len(source.member)-1)
            semivariance=semivariance*correction
            _array_record(rec,'semivariogram',f'{variable}_{label}_semivariogram',xr.Dataset({'semivariance':('distance_km',semivariance),'pair_count':('distance_km',counts)},coords={'distance_km':distance}),units=f'({unit})^2',variable=variable,comparator=label,details={'sample_indices':indices,'seed':config.seed,'finite_ensemble_variance_factor':correction})
            x=rr[variable].transpose('member','time','region').values
            corr=np.stack([np.corrcoef(a.T) for a in x])
            _array_record(rec,'regional_correlation',f'{variable}_{label}_regional_correlation',xr.DataArray(np.nanmean(corr,axis=0),dims=('region','region_other'),coords={'region':rr.region,'region_other':rr.region.values}),variable=variable,comparator=label,details={'member_correlations':'averaged; never concatenate member time axes'})
            if config.long_range_pairs:
                for first,second in config.long_range_pairs:
                    if first not in rr.region or second not in rr.region:
                        rec.skip('long_range_correlation','Missing region data',f'{first} or {second} unavailable',variable=variable)
                    else:
                        rec.add('long_range_correlation',float(np.nanmean(st.correlation(rr[variable].sel(region=first).values,rr[variable].sel(region=second).values))),variable=variable,region=f'{first}:{second}',aggregation='monthly residuals',comparator=label,n_members=len(x),n_time=x.shape[1])
            else:
                rec.skip('long_range_correlation','Missing region data','No long_range_pairs configured',variable=variable,comparator=label)
        for anomaly in ('residual','raw_monthly_anomaly'):
            # Calendar climatology removal is within each scenario, not reference centering.
            fields=residual if anomaly=='residual' else source.groupby('time.month')-source.groupby('time.month').mean(('member','time'))
            a=fields.temperature; b=fields.precipitation
            ac=a-a.mean('time'); bc=b-b.mean('time')
            den=np.sqrt((ac**2).sum('time')*(bc**2).sum('time'))
            correlations=((ac*bc).sum('time')/den.where(den>0)).mean('member')
            _array_record(rec,'temperature_precipitation_correlation',f'{label}_{anomaly}_tp_map',correlations.compute(),aggregation=anomaly,comparator=label)
            reg=rr if anomaly=='residual' else regional_source.groupby('time.month')-regional_source.groupby('time.month').mean(('member','time'))
            a=reg.temperature.transpose('member','region','time').values
            b=reg.precipitation.transpose('member','region','time').values
            lags=np.array([-12,-6,-3,-1,0,1,3,6,12])
            curves=[]
            for lag in lags:
                if abs(lag)>=a.shape[-1]-2:
                    curves.append(np.full(a.shape[:2],np.nan))
                elif lag<0:
                    curves.append(st.correlation(a[..., -lag:],b[..., :lag]))
                elif lag>0:
                    curves.append(st.correlation(a[..., :-lag],b[..., lag:]))
                else:
                    curves.append(st.correlation(a,b))
            _array_record(rec,'lagged_temperature_precipitation',f'{label}_{anomaly}_tp_lag',xr.DataArray(np.stack(curves).mean(1),dims=('lag','region'),coords={'lag':lags,'region':reg.region}),aggregation=anomaly,comparator=label,details={'positive_lag':'temperature leads precipitation, months'})
            matrix=np.stack([[st.correlation(a[:,i],b[:,j]).mean() for j in range(a.shape[1])] for i in range(a.shape[1])])
            _array_record(rec,'cross_region_dependence',f'{label}_{anomaly}_tp_crossregion',xr.DataArray(matrix,dims=('temperature_region','precipitation_region'),coords={'temperature_region':reg.region.values,'precipitation_region':reg.region.values}),aggregation=anomaly,comparator=label)
    for variable,unit in UNITS.items():
        fit_source=calibration if calibration is not None else (r if r is not None else g)
        label='independent calibration' if calibration is not None else 'DESCRIPTIVE evaluation-data fallback'
        if len(fit_source.member)<2:
            rec.skip(['eof_basis','eof_subspace','eof_explained_variance','pc_spectrum'],'Insufficient members','EOF residual basis requires >=2 source members',variable=variable)
            continue
        samples,fit_ix=_snapshots(fit_source[variable]-fit_source[variable].mean('member'),config.eof_samples,config.seed)
        try:
            basis=st.fit_eofs(samples,config.eof_modes,config.seed)
            _array_record(rec,'eof_basis',f'{variable}_eof_basis',xr.Dataset({'basis':(('mode','cell'),basis['basis']),'center':('cell',basis['center']),'calibration_explained_fraction':('mode',basis['explained'])},coords={'cell':g.cell}),units=unit,variable=variable,reason=label,details={'fit_sample_indices':fit_ix})
            for source_label,source,_ in source_sets:
                if source is None or len(source.member)<2:
                    continue
                residual=source[variable]-source[variable].mean('member')
                # Matrix product is sample-by-mode, never cell-by-cell covariance.
                modes=xr.DataArray(basis['basis'],dims=('mode','cell'),coords={'cell':g.cell})
                pc=xr.dot(residual-xr.DataArray(basis['center'],dims='cell'),modes,dim='cell').transpose('member','mode','time').compute()
                v=pc.var(('member','time'))
                total=float(residual.var(('member','time')).sum('cell').compute())
                _array_record(rec,'eof_explained_variance',f'{variable}_{source_label}_eof_explained',v/total,variable=variable,reason=label,comparator=source_label)
                other,other_ix=_snapshots(residual,config.eof_samples,config.seed)
                fit_other=st.fit_eofs(other,config.eof_modes,config.seed)
                similarity,angles=st.eof_subspace(basis['basis'],fit_other['basis'])
                rec.add('eof_subspace',float(similarity),variable=variable,aggregation='leading-mode subspace',comparator=source_label,reason=label,details={'principal_angles_radians':angles,'near_degenerate_modes':'compared jointly as a subspace','sample_indices':other_ix})
                try:
                    f,p=st.welch_spectrum(pc.values,config.welch_segment_months)
                    p=p*len(source.member)/(len(source.member)-1)
                    _array_record(rec,'pc_spectrum',f'{variable}_{source_label}_pc_spectrum',xr.DataArray(p,dims=('member','mode','frequency'),coords={'frequency':f}),units=f'({unit})^2/(cycles/year)',variable=variable,comparator=source_label,reason=label)
                except Unavailable as exc:
                    rec.skip('pc_spectrum',exc.status,exc.reason,variable=variable,comparator=source_label)
        except Unavailable as exc:
            rec.skip(['eof_basis','eof_subspace','eof_explained_variance','pc_spectrum'],exc.status,exc.reason,variable=variable)
    if r is None:
        rec.skip(['energy_score','variogram_score'],'Missing reference','Independent reference targets required')
    elif calibration is None:
        rec.skip(['energy_score','variogram_score'],'Missing calibration','Fixed calibration-reference scales required')
    else:
        selected=[str(v) for v in config.plot_regions if v in regional.region]
        if not selected:
            selected=['Global']
        cal=xr.Dataset({v:regional_average(calibration[v],weights,config.min_coverage)[0] for v in UNITS}).compute()
        for representation in ('regional','space_time','selected_points'):
            if representation=='selected_points':
                if not config.selected_cells:
                    rec.skip(['energy_score','variogram_score'],'Missing region data','No selected_cells configured',aggregation=representation)
                    continue
                sources=[d.sel(cell=config.selected_cells).to_array('variable').transpose('member','time','variable','cell').values for d in (g,r,calibration)]
            else:
                sources=[d.sel(region=selected).to_array('variable').transpose('member','time','variable','region').values for d in (regional,reference_regional,cal)]
            x,y,c=[a.reshape(a.shape[:2]+(-1,)) for a in sources]
            center=c.mean((0,1)); scale=c.std((0,1),ddof=1)
            if np.any(scale<=0) or not np.isfinite(scale).all():
                rec.skip(['energy_score','variogram_score'],'Undefined statistic','Calibration has zero/nonfinite scale',aggregation=representation)
                continue
            x,y=(x-center)/scale,(y-center)/scale
            lags=config.vector_time_lags if representation=='space_time' else [0]
            start=max(lags)
            if len(g.time)<=start:
                rec.skip(['energy_score','variogram_score'],'Insufficient record length','Space-time vector exceeds available time',aggregation=representation)
                continue
            es,vs=[],[]
            for t in range(start,x.shape[1]):
                xx=np.concatenate([x[:,t-lag] for lag in lags],axis=-1)
                yy=np.concatenate([y[:,t-lag] for lag in lags],axis=-1)
                es.append(st.energy_score(xx,yy)); vs.append(st.variogram_score(xx,yy,config.variogram_exponent))
            for metric,values in [('energy_score',es),('variogram_score',vs)]:
                _array_record(rec,metric,f'{metric}_{representation}',xr.DataArray(np.stack(values),dims=('time_index','target_member')),aggregation=representation,n_members=len(x),n_targets=len(y),n_time=len(es),details={'calibration_center':center,'calibration_scale':scale,'regions':selected,'time_lags_months':lags,'pair_weights':'uniform normalized unordered feature pairs','variogram_exponent':config.variogram_exponent})
