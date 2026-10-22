"""Public orchestration and sequential scenario execution."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import xarray as xr
from .config import Scenario,EvaluationConfig,load_manifest
from .data import (load_fields,align_fields,regions,regional_average,select_period,month_number,
                   parse_month,load_forcings,Unavailable)
from .report import (Recorder,ScenarioResult,CollectionResult,provenance,write_json,
                     FAMILIES,COLLECTION)
from . import diagnostics,dependence,extremes,audit,baselines


def _restrict(ds,scenario):
    bounds=scenario.periods.get('evaluation')
    out=select_period(ds,bounds)
    history_end=scenario.initialization.get('history_end')
    if history_end:
        out=out.isel(time=np.flatnonzero(month_number(out.time)>=parse_month(history_end)))
    if not out.sizes['time']:
        raise ValueError(f'{scenario.id}: evaluation period contains no generated months')
    return out


def evaluate_scenario(scenario,config=None,*,output_dir):
    """Validate, evaluate and persist one physical scenario.

    Primary schema errors raise. Optional input absence skips dependent analyses;
    corrupt explicitly configured optional inputs raise rather than disappear.
    Inputs and result paths remain explicit; there is no synthetic fallback.
    """
    if isinstance(scenario,dict):
        scenario=Scenario(**scenario)
    if config is None:
        config=EvaluationConfig()
    elif isinstance(config,dict):
        config=EvaluationConfig(**config)
    rec=Recorder(scenario.id,output_dir,config)
    prov=provenance(config,scenario);started=time.monotonic()
    opened=[]
    try:
        print(f'[{scenario.id}] validating generated fields',flush=True)
        raw=load_fields(scenario.generated,config);opened.append(raw)
        import json
        if json.loads(raw.attrs.get('source_attributes','{}')).get('synthetic_verification_only') and not config.allow_test_grid:
            raise ValueError('Synthetic verification data require explicit verification configuration; never use them as real evaluation inputs')
        g=_restrict(raw,scenario);r=None;cal=None
        if scenario.reference:
            ref=load_fields(scenario.reference,config);opened.append(ref)
            g,r,alignment=align_fields(g,_restrict(ref,scenario),config)
            prov['alignment']=alignment
        elif config.missing_policy=='common_mask':
            mask=g.to_array().notnull().all(('variable','member','time'))
            g=g.where(mask)
        if scenario.calibration:
            if scenario.reference:
                baselines._check_separate(scenario.calibration,scenario.reference,'Threshold/EOF calibration')
            cal=load_fields(scenario.calibration,config);opened.append(cal)
            if cal.temperature.attrs['standard_name']!=g.temperature.attrs['standard_name']:
                raise ValueError('Calibration temperature definition differs from evaluation')
        rec.source_counts={'members':len(g.member),'months':len(g.time),'cells':len(g.cell),
                           'reference_members':len(r.member) if r is not None else 0}
        weights,notes=regions(g,config);prov['regions']=notes
        if notes.get('ar6_status')=='Missing region data':
            rec.skip('region_coverage','Missing region data',notes.get('reason','AR6 polygons unavailable'),region='AR6')
        rr={};coverage={}
        for variable in diagnostics.UNITS:
            print(f'[{scenario.id}] aggregating {variable}',flush=True)
            rr[variable],coverage[variable]=regional_average(g[variable],weights,config.min_coverage)
        regional=xr.Dataset(rr).compute()
        reference_regional=xr.Dataset({v:regional_average(r[v],weights,config.min_coverage)[0] for v in diagnostics.UNITS}).compute() if r is not None else None
        rec.array('regional',regional,definition='native-area regional means; uncentered physical fields')
        rec.array('ensemble_mean',g.mean('member'))
        rec.array('region_weights',weights)
        if r is not None:
            rec.array('reference_regional',reference_regional)
            rec.array('reference_ensemble_mean',r.mean('member'))
        for variable in diagnostics.UNITS:
            cover=coverage[variable].min(('member','time')).compute()
            for region in cover.region.values:
                rec.add('region_coverage',float(cover.sel(region=region)),variable=variable,region=str(region),aggregation='minimum valid area fraction across members/months',n_members=len(g.member),n_time=len(g.time))
        rec.add('input_coverage',len(g.time),units='months',aggregation='validated monthly record',n_members=len(g.member),n_time=len(g.time),details={'cells':len(g.cell),'calendar':g.time.dt.calendar,'start':str(g.time.values[0]),'end':str(g.time.values[-1]),'ordering':'NESTED','history_excluded_until':scenario.initialization.get('history_end')})
        prov['input_metadata']={'generated':dict(raw.attrs),'reference':dict(r.attrs) if r is not None else None,
                                'generated_sizes':dict(g.sizes),'reference_sizes':dict(r.sizes) if r is not None else None}
        if scenario.forcings:
            f=load_forcings(scenario.forcings,g.time,require_complete=False)
            rec.array('forcings',f)
            prov['forcing_nonfinite_counts']=f.attrs['nonfinite_counts']
            prov['forcing_metadata_issues']=f.attrs['metadata_issues']
            for variable in ('co2','aod'):
                fraction=float(np.isfinite(f[variable]).mean())
                complete=fraction==1 and not json.loads(f.attrs['metadata_issues'])
                rec.add('forcing_coverage',fraction,variable=variable,aggregation='finite supplied forcing values',
                        n_time=len(f.time),status='Computed' if complete else 'Missing forcing metadata',
                        reason='' if complete else 'Missing forcing values/units retained; dependent comparisons disabled; no imputation or unit inference',details={'unit_issues':json.loads(f.attrs['metadata_issues'])})
        else:
            rec.skip('forcing_coverage','Missing forcing metadata','No forcing trajectories supplied')
        if cal is not None and scenario.baseline_period:
            base=select_period(cal,scenario.baseline_period)
            if len(base.time):
                regional_base=regional_average(base.temperature,weights.sel(region=['Global']),config.min_coverage)[0]
                prov['warming_baseline_degC']=float(regional_base.mean('member').weighted(base.time.dt.days_in_month).mean(('time','region')).compute())
                prov['warming_baseline_period']=scenario.baseline_period
        audit.audit(scenario,g,rec)
        print(f'[{scenario.id}] forced response and calibration',flush=True)
        diagnostics.forced(g,r,regional,reference_regional,weights,scenario,config,rec)
        diagnostics.calibration(g,r,regional,reference_regional,scenario,config,rec)
        print(f'[{scenario.id}] variability and spatial/joint dependence',flush=True)
        diagnostics.variability(regional,reference_regional,scenario,config,rec)
        dependence.run(g,r,regional,reference_regional,cal,weights,scenario,config,rec)
        print(f'[{scenario.id}] monthly extremes and baselines',flush=True)
        events=extremes.run(g,r,regional,reference_regional,cal,weights,scenario,config,rec)
        if events is not None:
            rec.array('regional_events',events)
        baselines.run(g,r,regional,reference_regional,weights,scenario,config,rec)
        rec.skip(sorted(COLLECTION),'Requires scenario collection','Evaluated by evaluate_scenario_collection when matching experiments are supplied')
        metrics=rec.finalize()
        prov.update(elapsed_seconds=time.monotonic()-started,real_data_evaluation=not config.allow_test_grid,
                    execution_status='Completed',artifacts=rec.artifacts)
        write_json(rec.root/'provenance.json',prov)
        write_json(rec.root/'configuration.json',{'config':config.to_dict(),'scenario':scenario.to_dict()})
        validation=metrics[metrics.family=='coverage'].copy()
        result=ScenarioResult(scenario.id,metrics,rec.artifacts,prov,validation=validation)
        if config.figures:
            from .plotting import plot_scenario
            result.figures=plot_scenario(result,config,rec.root/'figures')
        write_json(rec.root/'artifacts.json',{'diagnostics':result.artifacts,'figures':result.figures})
        print(f'[{scenario.id}] finished: {len(metrics)} metric rows',flush=True)
        return result
    except Exception as exc:
        prov.update(execution_status='Failed',error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-started)
        write_json(rec.root/'provenance.json',prov)
        raise
    finally:
        for ds in opened:
            ds.close()


def evaluate_scenario_collection(manifest,config=None,*,output_dir):
    """Evaluate every scenario once, then eligible controlled/collection analyses."""
    scenarios,default,resolved=load_manifest(manifest)
    config=default if config is None else EvaluationConfig(**config) if isinstance(config,dict) else config
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    write_json(root/'manifest.resolved.json',resolved)
    results={}
    for scenario in scenarios:
        results[scenario.id]=evaluate_scenario(scenario,config,output_dir=root/scenario.id)
    from .collection import run
    comparison,artifacts=run(scenarios,results,config,root/'collection')
    singles=pd.concat([r.metrics[~r.metrics.metric.isin(COLLECTION)] for r in results.values()],ignore_index=True)
    metrics=pd.concat([singles,comparison],ignore_index=True)
    metrics.to_csv(root/'all_metrics.csv',index=False);metrics.to_parquet(root/'all_metrics.parquet',index=False)
    comparison.to_csv(root/'cross_scenario_comparison.csv',index=False)
    for scenario in scenarios:
        result=results[scenario.id]
        def belongs(row):
            comparator=str(row['comparator'])
            return comparator=='' or scenario.id in comparator.split(':') or (scenario.co2_group is not None and comparator==scenario.co2_group)
        attached=comparison[comparison.apply(belongs,axis=1)].copy()
        attached['scenario']=scenario.id
        case=Recorder(scenario.id,root/scenario.id,config)
        case.rows=result.metrics[~result.metrics.metric.isin(COLLECTION)].to_dict('records')+attached.to_dict('records')
        present={r['metric'] for r in case.rows}
        for metric in COLLECTION-present:
            case.skip(metric,'Missing scenario combination','No eligible collection experiment applies to this scenario')
        result.metrics=case.finalize()
        result.provenance['collection_results']=str(root/'cross_scenario_comparison.csv')
        write_json(root/scenario.id/'provenance.json',result.provenance)
    manifest_table=[]
    for s in scenarios:
        manifest_table.append({'scenario':s.id,'description':s.description,'role':s.role,'model_variant':s.model_variant,
            'co2_group':s.co2_group,'volcanic_group':s.volcanic_group,'reference_available':bool(s.reference),
            'independent_reference_documented':s.reference_independent,'control_id':s.control_id,
            'forcing_metadata_available':bool(s.forcings),'training_support_available':bool(s.training_support),
            'supported_claim':'reference agreement assessable' if s.reference and s.reference_independent else 'descriptive; independence/agreement not established',
            'splits':json_string(s.splits)})
    pd.DataFrame(manifest_table).to_csv(root/'scenario_manifest.csv',index=False)
    score=metrics.groupby(['scenario','family','status']).size().reset_index(name='results')
    score.to_csv(root/'scorecards.csv',index=False)
    prov=provenance(config,resolved);prov.update(execution_status='Completed',scenarios=list(results),collection_artifacts=artifacts)
    figures=[]
    if config.figures:
        from .plotting import plot_collection
        figures=plot_collection(metrics,artifacts,config,root/'figures')
        from .plotting import update_collection_panel
        for scenario in scenarios:
            update_collection_panel(results[scenario.id],scenario,artifacts,config,root/scenario.id/'figures')
    write_json(root/'provenance.json',prov)
    return CollectionResult(results,metrics,comparison,prov,figures)


def json_string(value):
    import json
    return json.dumps(value,sort_keys=True)
