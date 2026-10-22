"""Publication figures; display projections never enter metric calculations."""
from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
GROUPS={1:('Scenario and evaluation coverage',['coverage']),
        2:('Forced and volcanic response',['forced','eruption']),
        3:('Distributional calibration',['calibration']),
        4:('Internal variability and spatial–temporal dependence',['variability','dependence']),
        5:('Monthly extremes and compound events',['extremes']),
        6:('Volcanic versus internal uncertainty',['nested'])}


def style():
    plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
        'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.prop_cycle':plt.cycler(color=COLORS)})


def save(fig,root,name):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for ax in fig.axes:
        title=ax.get_title()
        if title:
            # Long scenario IDs must not overlap neighbouring publication panels.
            width=max(35,int(fig.get_figwidth()*ax.get_position().width*10))
            ax.set_title('\n'.join(textwrap.fill(line,width=width,break_long_words=False) for line in title.splitlines()))
    paths=[]
    for suffix in ('png','pdf'):
        path=root/f'{name}.{suffix}'
        fig.savefig(path,dpi=250,bbox_inches='tight');paths.append(str(path))
    plt.close(fig)
    return paths


def _load(path):
    with xr.open_dataset(path) as d:
        return d.load()


def _first(ds):
    return next(iter(ds.data_vars.values()))


def _time(time):
    return time.dt.year.values+(time.dt.month.values-.5)/12


def _missing(ax,rows,title):
    ax.set_axis_off();ax.set_title(title)
    if len(rows):
        counts=rows.groupby('status').size()
        text='\n'.join(f'{k}: {v}' for k,v in counts.items())
        reasons=rows.loc[rows.status!='Computed','reason'].drop_duplicates().tolist()[:3]
        text+='\n\n'+'\n'.join(str(r)[:100] for r in reasons)
    else:
        text='No eligible inputs for this panel'
    ax.text(.03,.9,text,va='top',wrap=True,transform=ax.transAxes,fontsize=9)


def _map(values,title,unit,path,root,difference=False):
    import healpy as hp
    a=np.asarray(values,float)
    if not np.isfinite(a).any():
        return []
    fig=plt.figure(figsize=(9,4.5))
    limits={}
    if difference:
        bound=max(float(np.nanmax(abs(a))),1e-12);limits={'min':-bound,'max':bound}
    hp.mollview(a,nest=True,title=title,unit=unit,cmap='RdBu_r' if difference else 'viridis',
                badcolor='lightgray',fig=fig.number,**limits)
    hp.graticule(color='gray',alpha=.3)
    return save(fig,root,path)


def plot_scenario(result,config,root):
    style();paths=[];metrics=result.metrics;art=result.artifacts
    with xr.open_dataset(art['regional']) as timing:
        period=f'{int(timing.time.dt.year[0])}–{int(timing.time.dt.year[-1])}'
    s=f'{result.scenario_id} | {period}'
    # 1: registry coverage and supplied forcings.
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    counts=metrics.groupby(['family','status']).size().unstack(fill_value=0)
    counts.plot.barh(stacked=True,ax=axes[0],color=COLORS+[ '#999999','#666666','#aaaaaa'])
    axes[0].set(xlabel='Metric records (count)',ylabel='',title=f'{s} | full registry coverage');axes[0].legend(fontsize=6,loc='upper left',bbox_to_anchor=(1,1))
    if 'forcings' in art:
        d=_load(art['forcings']);t=_time(d.time)
        axes[1].plot(t,d.co2,color=COLORS[0],label='CO₂');axes[1].set(ylabel='CO₂ (ppm)',xlabel='Calendar year',title=f'{s} | supplied monthly forcings')
        ax=axes[1].twinx();a=d.aod
        if 'lat' in a.dims:
            # Latitude profiles retained; display envelope is not an amplitude estimator.
            ax.fill_between(t,a.min('lat'),a.max('lat'),alpha=.2,color=COLORS[1],label='AOD latitude range')
            ax.plot(t,a.mean('lat'),color=COLORS[1],lw=.7,label='AOD unweighted latitude mean (display)')
        else:
            ax.plot(t,a,color=COLORS[1])
        ax.set_ylabel('AOD (1)')
    else:
        _missing(axes[1],metrics[metrics.metric=='training_support'],f'{s} | forcing coverage')
    paths+=save(fig,root,'01_scenario_coverage')
    # 2: physical trajectories and monthly climatologies.
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for row,(v,unit) in enumerate([('temperature','°C'),('precipitation','mm/day')]):
        d=_load(art[f'{v}_trajectory']).sel(region='Global');t=_time(d.time)
        axes[row,0].plot(t,d.ensemble_mean,label='Generated ensemble mean')
        axes[row,0].fill_between(t,d.q05,d.q95,alpha=.22,label='Generated 5–95% ensemble spread')
        if 'reference_mean' in d:
            axes[row,0].plot(t,d.reference_mean,color=COLORS[1],label='Reference ensemble mean')
            axes[row,0].fill_between(t,d.reference_q05,d.reference_q95,color=COLORS[1],alpha=.15,label='Reference 5–95% ensemble spread')
        axes[row,0].set(title=f'{s} | global {v} | monthly',ylabel=unit,xlabel='Calendar year');axes[row,0].legend(fontsize=7)
        key=f'{v}_climatology'
        if key in art:
            d=_first(_load(art[key]))
            for region in config.plot_regions:
                if region in d.region:
                    axes[row,1].plot(d.month,d.sel(region=region),label=region)
            axes[row,1].set(title=f'{s} | {v} monthly climatology | evaluation/baseline period',xlabel='Calendar month',ylabel=unit);axes[row,1].legend(fontsize=7)
        else:
            _missing(axes[row,1],metrics[metrics.metric=='climatology'],f'{s} | {v} climatology')
    paths+=save(fig,root,'02_forced_response')
    # 3: calibration and spread.
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for row,v in enumerate(('temperature','precipitation')):
        sub=metrics[(metrics.variable==v)&(metrics.region=='Global')&(metrics.aggregation=='monthly')]
        cover=sub[(sub.metric=='coverage')&(sub.status=='Computed')]
        if len(cover):
            nominal=[json.loads(d)['nominal'] for d in cover.details]
            axes[row,0].plot(nominal,cover.value,'o-',label='Empirical coverage')
            axes[row,0].plot([0,1],[0,1],'k--',lw=1,label='Nominal')
            axes[row,0].set(xlabel='Nominal central interval probability',ylabel='Empirical coverage (1)',title=f'{s} | global {v} | monthly evaluation');axes[row,0].legend()
            ranks=_first(_load(art[f'{v}_Global_ranks']))
            axes[row,1].bar(np.arange(len(ranks)),ranks,color=COLORS[row]);axes[row,1].axhline(1/len(ranks),color='black',ls='--')
            axes[row,1].set(title=f'{s} | global {v} | tie-aware ranks',xlabel='Rank',ylabel='Fraction (1)')
        else:
            for ax in axes[row]:
                _missing(ax,metrics[metrics.family=='calibration'],f'{s} | {v} calibration')
    paths+=save(fig,root,'03_distributional_calibration')
    # 4: regional residual ACF and spectra.
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,v in enumerate(('temperature','precipitation')):
        sub=metrics[(metrics.metric=='autocorrelation')&(metrics.variable==v)&(metrics.region=='Global')&(metrics.aggregation=='monthly')&(metrics.status=='Computed')]
        if len(sub):
            for label,a in sub.groupby('comparator'):
                lags=[json.loads(d)['lag'] for d in a.details]
                axes[row,0].plot(lags,a.value,'o-',label=label)
                if a.ci_low.notna().any():
                    axes[row,0].fill_between(lags,a.ci_low.astype(float),a.ci_high.astype(float),alpha=.15)
            axes[row,0].set(title=f'{s} | global {v} residual ACF | evaluation',xlabel='Lag (months)',ylabel='Correlation (1)');axes[row,0].legend()
        else:
            _missing(axes[row,0],metrics[metrics.metric=='autocorrelation'],f'{s} | {v} residual ACF')
        plotted=False
        for label in ('generated','reference'):
            key=f'{v}_{label}_Global_psd'
            if key in art:
                d=_first(_load(art[key])).mean('member');axes[row,1].loglog(d.frequency,d,label=label);plotted=True
        if plotted:
            axes[row,1].set(title=f'{s} | global {v} residual Welch PSD',xlabel='Frequency (cycles/year)',ylabel=f'{"°C" if v=="temperature" else "mm/day"}² / (cycles/year)');axes[row,1].legend()
        else:
            _missing(axes[row,1],metrics[metrics.metric=='temporal_spectrum'],f'{s} | {v} residual spectrum')
    paths+=save(fig,root,'04_internal_variability_dependence')
    # 5: exposure versus area fractions, distinct quantities.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    key='generated_hot_dry_regional_probability'
    if key in art:
        d=_first(_load(art[key]));selected=[r for r in config.plot_regions if r in d.region]
        d=d.sel(region=selected);axes[0].bar(d.region.values,d.values,color=COLORS[1]);axes[0].tick_params(axis='x',rotation=45)
        axes[0].set(title=f'{s} | hot–dry regional-average events | monthly evaluation',ylabel='Event probability (1)')
        area=_first(_load(art['generated_hot_dry_area'])).sel(region='Global')
        axes[1].hist(area.values.ravel(),bins=20,color=COLORS[0]);axes[1].set(title=f'{s} | global hot–dry affected area | monthly evaluation',xlabel='Fraction of cells exceeding local thresholds',ylabel='Member-month count (dependent)')
    else:
        for ax in axes:
            _missing(ax,metrics[(metrics.family=='extremes')&(~metrics.metric.isin(['Rx1day','Rx5day','TXx']))],f'{s} | calibrated monthly extremes')
    paths+=save(fig,root,'05_monthly_extremes_compounds')
    fig,ax=plt.subplots(figsize=(10,4));_missing(ax,metrics[metrics.family=='nested'],f'{s} | volcanic versus internal uncertainty')
    paths+=save(fig,root,'06_volcanic_internal_uncertainty')
    # Native-grid maps supplement the six principal groups.
    for v,unit in [('temperature','°C'),('precipitation','mm/day')]:
        key=f'{v}_error_maps'
        if key in art:
            d=_load(art[key]);paths+=_map(d.bias.values,f'{s} | {v} ensemble-mean bias | monthly evaluation',unit,f'{v}_bias_map',root,True)
        key=f'{v}_spatial_spectrum'
        data_key=f'{v}_generated_spatial_spectrum'
        if data_key in art:
            fig,ax=plt.subplots(figsize=(7,4))
            for label in ('generated','reference'):
                k=f'{v}_{label}_spatial_spectrum'
                if k in art:
                    d=_first(_load(art[k])).mean('sample');ax.loglog(np.arange(1,len(d)),d.values[1:],label=label)
            ax.set(title=f'{s} | {v} residual spherical spectrum | sampled monthly maps',xlabel='Spherical harmonic degree',ylabel=unit+'²');ax.legend();paths+=save(fig,root,key)
    paths+=supplemental(result,config,root)
    return paths


def supplemental(result,config,root):
    """Selected-region scientific panels from exported distributions and arrays."""
    paths=[];art=result.artifacts;frame=result.metrics
    with xr.open_dataset(art['regional']) as timing:
        period=f'{int(timing.time.dt.year[0])}–{int(timing.time.dt.year[-1])}'
    s=f'{result.scenario_id} | {period}'
    for variable,unit in [('temperature','°C'),('precipitation','mm/day')]:
        for region in config.plot_regions:
            sub=frame[(frame.variable==variable)&(frame.region==region)&(frame.status=='Computed')]
            trends=sub[(sub.metric=='trend')&(sub.aggregation=='annual')]
            if len(trends):
                fig,ax=plt.subplots(figsize=(6,4))
                labels=['Reference' if label=='reference' else 'Generated' for label in trends.comparator]
                ax.bar(np.arange(len(trends)),trends.value,color=COLORS[:len(trends)])
                ax.set_xticks(np.arange(len(trends)),labels)
                for i,(_,trend) in enumerate(trends.iterrows()):
                    if np.isfinite(trend.ci_low):ax.vlines(i,trend.ci_low,trend.ci_high,color='black',lw=2)
                ax.axhline(0,color='gray',lw=.8);ax.set(title=f'{s} | {region} {variable} trend | annual evaluation',ylabel=unit+'/decade')
                paths+=save(fig,root,f'{variable}_{region}_trend_uncertainty')
            averaging=sub[sub.metric=='averaging_variance']
            if len(averaging):
                fig,ax=plt.subplots(figsize=(6,4))
                for label,rows in averaging.groupby('comparator'):
                    x=[int(a.split()[0]) for a in rows.aggregation]
                    ax.loglog(x,rows.value,'o-',label=label)
                ax.set(title=f'{s} | {region} {variable} residual averages | evaluation',xlabel='Averaging timescale (months, day weighted)',ylabel=unit+'²');ax.legend()
                paths+=save(fig,root,f'{variable}_{region}_variance_timescale')
            coverage=sub[(sub.metric=='coverage')&(sub.aggregation=='monthly')]
            width=sub[(sub.metric=='interval_width')&(sub.aggregation=='monthly')]
            if len(coverage) and len(width):
                fig,ax=plt.subplots(figsize=(6,4));ax.plot(coverage.value,width.value,'o-')
                for (_,a),(_,b) in zip(coverage.iterrows(),width.iterrows()):
                    ax.annotate(f'{json.loads(a.details)["nominal"]:.0%}',(a.value,b.value),fontsize=8)
                ax.set(title=f'{s} | {region} {variable} | monthly intervals',xlabel='Empirical coverage (1)',ylabel=f'Interval width ({unit})')
                paths+=save(fig,root,f'{variable}_{region}_width_coverage')
            # January provides a calendar-conditioned QQ example; all months remain exported.
            qq=[k for k in art if k.startswith(f'{variable}_{region}_') and k.endswith('_month1_qq')]
            for key in qq[:1]:
                d=_first(_load(art[key]));fig,ax=plt.subplots(figsize=(5,5))
                ax.plot(d.sel(source='reference'),d.sel(source='generated'),'o-')
                limits=[float(d.min()),float(d.max())];ax.plot(limits,limits,'k--',lw=1)
                ax.set(title=f'{s} | {region} {variable} | January configured window',xlabel=f'Reference quantiles ({unit})',ylabel=f'Generated quantiles ({unit})')
                paths+=save(fig,root,f'{variable}_{region}_qq')
            for years in config.trend_years:
                keys=[f'{variable}_generated_{region}_{kind}_{years}' for kind in ('raw_trend_distribution','residual_trend_distribution')]
                if all(k in art for k in keys):
                    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
                    for ax,kind in zip(axes,('raw_trend_distribution','residual_trend_distribution')):
                        for label in ('generated','reference'):
                            key=f'{variable}_{label}_{region}_{kind}_{years}'
                            if key in art:
                                a=_first(_load(art[key])).values.ravel();ax.hist(a,bins=15,density=True,alpha=.4,label=label)
                        ax.set(title=f'{region} | {kind.replace("_"," ")}',xlabel=unit+'/decade',ylabel='Density');ax.legend()
                    fig.suptitle(f'{s} | {variable} | nonoverlapping {years}-year trends, evaluation')
                    paths+=save(fig,root,f'{variable}_{region}_{years}year_trend_distributions')
        # Full native-field spectrum and semivariogram comparisons.
        key=f'{variable}_generated_semivariogram'
        if key in art:
            fig,ax=plt.subplots(figsize=(7,4))
            for label in ('generated','reference'):
                key=f'{variable}_{label}_semivariogram'
                if key in art:
                    d=_load(art[key]);ax.plot(d.distance_km,d.semivariance,label=label)
            ax.set(title=f'{s} | {variable} residual semivariogram | sampled monthly fields',xlabel='Great-circle distance (km)',ylabel=unit+'²');ax.legend()
            paths+=save(fig,root,f'{variable}_semivariogram')
        matrices=[];labels=[]
        for label in ('generated','reference'):
            key=f'{variable}_{label}_regional_correlation'
            if key in art:
                matrices.append(_first(_load(art[key])));labels.append(label)
        if matrices:
            if len(matrices)==2:
                matrices.append(matrices[0]-matrices[1]);labels.append('generated minus reference')
            fig,axes=plt.subplots(1,len(matrices),figsize=(6*len(matrices),5),squeeze=False,layout='constrained')
            for ax,d,label in zip(axes[0],matrices,labels):
                selected=[r for r in config.plot_regions if r in d.region]
                d=d.sel(region=selected,region_other=selected)
                im=ax.imshow(d,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(np.arange(len(selected)),selected,rotation=45,ha='right');ax.set_yticks(np.arange(len(selected)),selected)
                ax.set_title(f'{s} | {variable} residual correlation\n{label} | monthly evaluation');fig.colorbar(im,ax=ax,label='Correlation (1)')
            paths+=save(fig,root,f'{variable}_regional_correlations')
        for label in ('generated','reference'):
            key=f'{variable}_{label}_residual_variance_map'
            if key in art:
                paths+=_map(_first(_load(art[key])).values,f'{s} | {label} {variable} conditional variance | monthly evaluation',unit+'²',key,root)
        key=f'{variable}_eof_basis'
        if key in art:
            d=_load(art[key])
            for mode in range(min(2 if config.mode=='quick' else config.eof_modes,d.sizes['mode'])):
                paths+=_map(d.basis.isel(mode=mode).values,f'{s} | {variable} EOF {mode+1} basis | designated fit period','orthonormal loading (1)',f'{variable}_eof_{mode+1}',root,True)
        key=f'{variable}_annual_maps'
        if key in art:
            import healpy as hp
            d=_load(art[key]).mean('time')
            fields=['generated','reference','difference'] if 'reference' in d else ['generated']
            shared=[float(d[k].min()) for k in fields if k!='difference']+[float(d[k].max()) for k in fields if k!='difference']
            fig=plt.figure(figsize=(6*len(fields),4))
            for index,field in enumerate(fields):
                values=d[field].values
                if field=='difference':
                    limit=max(float(np.nanmax(abs(values))),1e-12);lo,hi=-limit,limit
                else:lo,hi=min(shared),max(shared)
                hp.mollview(values,nest=True,sub=(1,len(fields),index+1),fig=fig.number,min=lo,max=hi,
                    title=f'{s} | {field} {variable}\nannual evaluation climatology',unit=unit,cmap='RdBu_r' if field=='difference' else 'viridis')
            paths+=save(fig,root,f'{variable}_generated_reference_maps')
    for key in ['generated_residual_tp_lag','reference_residual_tp_lag']:
        if key in art:
            d=_first(_load(art[key]));fig,ax=plt.subplots(figsize=(7,4))
            for region in config.plot_regions:
                if region in d.region:ax.plot(d.lag,d.sel(region=region),label=region)
            ax.axhline(0,color='gray',lw=.8);ax.set(title=f'{s} | {key.replace("_"," ")} | monthly evaluation',xlabel='Lag (months; positive = temperature leads)',ylabel='Correlation (1)');ax.legend(fontsize=8)
            paths+=save(fig,root,key)
    for event in ['hot_dry','cold_dry','cold_wet']:
        key=f'generated_{event}_probability_map'
        if key in art:
            paths+=_map(_first(_load(art[key])).values,f'{s} | {event} event probability | monthly fixed thresholds','Probability (1)',key,root)
        key=f'generated_{event}_Global_durations'
        if key in art:
            d=_load(art[key]);fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
            for ax,name,unit in zip(axes,['duration_months','severity_standardized_months'],['months','standardized anomaly-months']):
                if d.sizes.get('event_index',0):ax.hist(d[name],bins=15)
                ax.set(title=f'{s} | global {event} | monthly event {name}',xlabel=unit,ylabel='Event count (dependent)')
            paths+=save(fig,root,f'{event}_duration_severity')
    # Joint density is explicitly month-conditioned and window-limited.
    if 'regional' in art:
        d=_load(art['regional']).sel(region='Global');january=d.isel(time=np.flatnonzero(d.time.dt.month.values==1)[:30])
        if january.sizes['time']:
            fig,ax=plt.subplots(figsize=(6,5));h=ax.hist2d(january.temperature.values.ravel(),january.precipitation.values.ravel(),bins=25,cmap='viridis')
            years=january.time.dt.year.values
            ax.set(title=f'{s} | global temperature–precipitation\nJanuary {years[0]}–{years[-1]} (potentially nonstationary)',xlabel='Temperature (°C)',ylabel='Precipitation (mm/day)');fig.colorbar(h[3],ax=ax,label='Member-January count')
            paths+=save(fig,root,'global_january_joint_density')
    return paths


def plot_collection(metrics,artifacts,config,root):
    style();paths=[]
    for number,(title,families) in GROUPS.items():
        fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
        sub=metrics[metrics.family.isin(families)]
        if number==1:
            counts=sub.groupby(['scenario','status']).size().unstack(fill_value=0)
            if len(counts):
                counts.plot.barh(stacked=True,ax=ax,color=COLORS);ax.set(xlabel='Registry rows',ylabel='',title='Collection | evaluation coverage');ax.legend(fontsize=7)
            else:
                _missing(ax,sub,title)
        elif number==2 and any('controlled_response' in k for k in artifacts):
            for key,path in artifacts.items():
                if key.endswith('controlled_response'):
                    d=_load(path)
                    ax.plot(_time(d.time),d.generated_temperature.sel(region='Global'),label=key.removesuffix('_controlled_response'))
            ax.axhline(0,color='gray',lw=.8);ax.set(title='Collection | global controlled temperature response | monthly evaluation',xlabel='Calendar year',ylabel='Eruption minus control (°C)');ax.legend(fontsize=7)
        elif number==6 and any('temperature_variance_components' in k for k in artifacts):
            for key,path in artifacts.items():
                if key.endswith('temperature_variance_components'):
                    d=_load(path).sel(region='Global');t=_time(d.time)
                    for component,ls in [('within','-'),('between','--')]:
                        ax.plot(t,d[component],ls=ls,label=f'{key}: {component}')
            ax.axhline(0,color='gray',lw=.8);ax.set(title='Collection | global temperature variance components | monthly evaluation',xlabel='Calendar year',ylabel='Variance (°C²), negative corrected estimates retained');ax.legend(fontsize=6)
        else:
            chosen={2:'mean_bias',3:'crps',4:'spread_ratio',5:'regional_average_extreme'}.get(number)
            rows=sub[(sub.metric==chosen)&(sub.status=='Computed')&(sub.variable=='temperature')] if chosen else pd.DataFrame()
            if len(rows) and rows.value.notna().any():
                table=rows.pivot_table(index='region',columns='scenario',values='value',aggfunc='mean')
                vals=table.values
                bound=np.nanmax(abs(vals)) or 1
                im=ax.imshow(vals,aspect='auto',cmap='RdBu_r' if chosen=='mean_bias' else 'viridis',vmin=-bound if chosen=='mean_bias' else None,vmax=bound if chosen=='mean_bias' else None)
                ax.set_xticks(np.arange(len(table.columns)),table.columns,rotation=45,ha='right');ax.set_yticks(np.arange(len(table.index)),table.index)
                ax.set_title(f'Collection | temperature {chosen} | evaluated periods/aggregations');fig.colorbar(im,ax=ax,label='°C' if chosen in ('crps','mean_bias') else '1')
            else:
                _missing(ax,sub,'Collection | '+title)
        paths+=save(fig,root,f'{number:02d}_{title.lower().replace(" ","_").replace("–","-")}')
    return paths


def update_collection_panel(result,scenario,artifacts,config,root):
    """Replace the provisional sixth panel after collection eligibility is known."""
    keys=[k for k in artifacts if scenario.co2_group and k.startswith(f'{scenario.co2_group}_{scenario.model_variant}_') and k.endswith('temperature_variance_components')]
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    if keys:
        d=_load(artifacts[keys[0]]).sel(region='Global');t=_time(d.time)
        for component,label,color in [('within','Conditional internal variability',COLORS[0]),('between','Volcanic forcing uncertainty (corrected)',COLORS[1])]:
            ax.plot(t,d[component],label=label,color=color)
            if component+'_ci_low' in d:
                ax.fill_between(t,d[component+'_ci_low'],d[component+'_ci_high'],color=color,alpha=.15)
        ax.axhline(0,color='gray',lw=.8);ax.set(title=f'{scenario.id} | {scenario.co2_group} | global temperature variance, monthly evaluation',xlabel='Calendar year',ylabel='Variance (°C²); bootstrap uncertainty shaded');ax.legend(fontsize=8)
    else:
        _missing(ax,result.metrics[result.metrics.family=='nested'],f'{scenario.id} | volcanic versus internal uncertainty')
    return save(fig,root,'06_volcanic_internal_uncertainty')
