"""Evidence-based split audit. Absence of metadata is not independence."""
import numpy as np
from .data import month_number, parse_month, load_forcings, scalar_aod, Unavailable


def audit(scenario,generated,rec):
    training=scenario.training_support
    splits=scenario.splits
    held=scenario.held_out
    for dimension in ('member','time','eruption','combination','pathway','amplitude'):
        if dimension not in training:
            rec.skip('leakage_audit','Not verifiable',f'No training {dimension} metadata',aggregation=dimension)
            continue
        train=training[dimension]
        evaluation=held.get(dimension,splits.get(dimension))
        if dimension=='time':
            tm=month_number(generated.time)
            intervals=train if train and isinstance(train[0],list) else [train]
            overlap=sum(int(((tm>=parse_month(start)) & (tm<parse_month(end))).sum()) for start,end in intervals)
            rec.add('leakage_audit',overlap,units='months',aggregation='time overlap',reason='Overlap is evidence to inspect; not automatically leakage for a member holdout')
        elif dimension=='amplitude':
            if evaluation is None:
                rec.skip('leakage_audit','Not verifiable','No evaluated amplitude documented',aggregation=dimension)
            else:
                values=np.atleast_1d(evaluation).astype(float)
                distance=np.maximum.reduce([np.full(values.shape,float(train[0]))-values,values-float(train[1]),np.zeros(values.shape)])
                rec.add('leakage_audit',float(distance.max()),units='configured amplitude units',aggregation='amplitude support distance',details={'training_range':train})
        elif evaluation is None:
            rec.skip('leakage_audit','Not verifiable',f'No held-out {dimension} IDs documented',aggregation=dimension)
        else:
            overlap=set(map(str,np.atleast_1d(train))) & set(map(str,np.atleast_1d(evaluation)))
            rec.add('leakage_audit',len(overlap),units='identifiers',aggregation=dimension,reason='Shared identifiers found' if overlap else 'No identifier overlap in supplied metadata',details={'overlap':sorted(overlap)})
    branch=scenario.lineage.get('generated',{})
    training_branches=scenario.lineage.get('training',{})
    if branch and training_branches:
        overlap=set(branch.values()) & set(training_branches.values())
        rec.add('leakage_audit',len(overlap),units='lineages',aggregation='historical/future branch lineage',details={'shared':sorted(overlap)})
    else:
        rec.skip('leakage_audit','Not verifiable','Generated and training branch lineage metadata required',aggregation='branch lineage')
    ranges=training.get('forcing_ranges')
    if not ranges:
        rec.skip('training_support','Not verifiable','No documented CO2/AOD support ranges')
    else:
        try:
            forcing=load_forcings(scenario.forcings,generated.time)
            values={'co2':forcing.co2.values,'aod':scalar_aod(forcing,scenario.forcings).values}
            for key,a in values.items():
                if key not in ranges:
                    rec.skip('training_support','Not verifiable',f'No {key} support range',variable=key)
                    continue
                lo,hi=ranges[key]
                if hi<=lo:
                    raise ValueError('Training support upper limit must exceed lower limit')
                distance=np.maximum.reduce([lo-a,a-hi,np.zeros_like(a)])/(hi-lo)
                rec.add('training_support',float(np.mean(distance)),variable=key,aggregation='mean normalized distance outside documented range',n_time=len(a),details={'maximum_distance':distance.max(),'outside_fraction':np.mean(distance>0),'range':[lo,hi],'does_not_verify_generalization':True})
        except Unavailable as exc:
            rec.skip('training_support',exc.status,exc.reason)
    cost=scenario.timing
    if not all(k in cost for k in ('generation_seconds','generated_member_years')):
        rec.skip('generation_cost','Missing timing metadata','Supply generation_seconds and generated_member_years; training separate')
    elif cost['generated_member_years']<=0:
        raise ValueError('generated_member_years must be positive')
    else:
        rec.add('generation_cost',cost['generation_seconds']/cost['generated_member_years'],units='seconds/member-year',aggregation='generation only',details=cost)
