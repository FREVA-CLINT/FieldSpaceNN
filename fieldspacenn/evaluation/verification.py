"""Explicit synthetic fixtures for verification only; never a real-data fallback."""
from pathlib import Path
import numpy as np
import xarray as xr
from .report import write_json


def synthetic_fields(*,members=8,months=120,nside=1,seed=1,start='2000-01-01',bias=0.,scale=1.,eruption=None):
    import healpy as hp
    rng=np.random.default_rng(seed);cells=12*nside*nside
    lon,lat=hp.pix2ang(nside,np.arange(cells),nest=True,lonlat=True)
    # Low-rank fields have known spatial and temperature-precipitation dependence.
    innovation=rng.normal(size=(members,months,3));latent=np.zeros_like(innovation)
    for t in range(1,months):
        latent[:,t]=.7*latent[:,t-1]+innovation[:,t]
    basis=np.stack([np.ones(cells),np.sin(np.deg2rad(lat)),np.cos(np.deg2rad(lon))])
    noise=np.einsum('mtk,kc->mtc',latent,basis)*scale
    month=np.arange(months)
    forced=.002*month+2*np.sin(2*np.pi*month/12)
    if eruption is not None:
        offset=month-eruption
        pulse=np.where(offset>=0,-1.5*np.exp(-np.maximum(offset,0)/18),0.)
        forced=forced+pulse
    tas=280+forced[None,:,None]+10*np.sin(np.deg2rad(lat))[None,None,:]+noise+bias
    pr=np.maximum(.01,3+.25*forced[None,:,None]-.2*noise+.1*rng.normal(size=noise.shape))
    d=xr.Dataset({'tas':(('ensemble','time','cell'),tas.astype('float32')),
                  'pr':(('ensemble','time','cell'),(pr/86400).astype('float32'))},
                 coords={'ensemble':np.arange(members),'time':xr.date_range(start,periods=months,freq='MS'), 'cell':np.arange(cells)},
                 attrs={'synthetic_verification_only':1,'ordering':'nested'})
    d.tas.attrs={'units':'K','standard_name':'air_temperature'}
    d.pr.attrs={'units':'kg m-2 s-1','standard_name':'precipitation_flux'}
    return d


def specification(path):
    return {'path':str(path),'dimensions':{'ensemble':'member'},
            'variables':{'temperature':'tas','precipitation':'pr'},'ordering':'nested',
            'units':{'temperature':'K','precipitation':'kg m-2 s-1'},
            'temperature_definition':'air_temperature','precipitation_representation':'flux'}


def fixture_manifest(root,*,months=120,members=6,collection=True,figures=True):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    calibration=synthetic_fields(members=10,months=120,seed=91,start='1980-01-01')
    calpath=root/'calibration.nc';calibration.to_netcdf(calpath)
    scenarios=[]
    cases=[('synthetic_control',None),('synthetic_eruption',36)] if collection else [('synthetic_case',None)]
    for i,(name,eruption) in enumerate(cases):
        generated=synthetic_fields(members=members,months=months,seed=10+i,eruption=eruption,bias=.3)
        reference=synthetic_fields(members=members+2,months=months,seed=30+i,eruption=eruption)
        gp=root/f'{name}.nc';rp=root/f'{name}_reference.nc'
        generated.to_netcdf(gp);reference.to_netcdf(rp)
        aod=np.zeros(months)
        if eruption is not None:
            offset=np.arange(months)-eruption
            aod=np.where(offset>=0,.15*np.exp(-np.maximum(offset,0)/12),0)
        forcing=xr.Dataset({'co2c':('time',350+np.arange(months)*.1),'aod550':('time',aod)},coords={'time':generated.time})
        forcing.co2c.attrs['units']='ppm';forcing.aod550.attrs['units']='1'
        fp=root/f'{name}_forcing.nc';forcing.to_netcdf(fp)
        s={'id':name,'description':'SYNTHETIC VERIFICATION ONLY','generated':specification(gp),
            'reference':specification(rp),'calibration':specification(calpath),'reference_independent':True,
            'forcings':{'path':str(fp),'no_eruptions':eruption is None},
            'initialization':{'protocol':'synthetic AR1','history_id':'synthetic_initialization'},
            'co2_group':'synthetic_pathway','volcanic_group':name,'history_probability':.5 if collection else 1.,
            'baseline_period':['1980-01','1990-01'],
            'recovery':{'temperature':{'tolerance':.2},'precipitation':{'tolerance':.1},'sustained_months':12,'peak_search_months':[0,36]}}
        if eruption is not None:
            s['control_id']='synthetic_control';s['eruptions']=[{'id':'synthetic_event','date':str(generated.time.values[eruption])[:7],'amplitude':.15}]
        scenarios.append(s)
    manifest={'config':{'mode':'quick','nside':1,'allow_test_grid':True,'ar6':False,'latitude_bands':[],
                        'bootstrap':9,'size_matching_repetitions':3,'spatial_pairs':50,'spatial_snapshots':3,
                        'eof_samples':24,'eof_modes':2,'welch_segment_months':12,'min_tail_observations':1,
                        'plot_regions':['Global','NH','SH'],'long_range_pairs':[['NH','SH']],
                        'figures':figures,'min_independent_blocks':3},'scenarios':scenarios}
    write_json(root/'manifest.json',manifest)
    return manifest
