"""Lazy data adapters, calendar-safe aggregation and native-grid region masks."""
from pathlib import Path
import hashlib
import json
import numpy as np
import xarray as xr


class Unavailable(Exception):
    """An expected unmet scientific prerequisite, never an implementation error."""
    def __init__(self, status, reason):
        self.status, self.reason = status, reason
        super().__init__(reason)


def open_store(path, *, chunks='auto', decode_times=True):
    path = str(path)
    if '://' not in path and not Path(path).exists():
        raise FileNotFoundError(f'Input does not exist: {path}')
    if path.endswith(('.nc', '.nc4', '.netcdf')):
        return xr.open_dataset(path, chunks=chunks, decode_times=decode_times)
    return xr.open_zarr(path, chunks=chunks, decode_times=decode_times, consolidated=None)


def inspect_schema(spec):
    """Small samples only; does not compute entire fields."""
    with open_store(spec.get('path', spec.get('member_paths', [None])[0]), chunks=None,
                    decode_times=False) as ds:
        return {'sizes': dict(ds.sizes), 'attributes': dict(ds.attrs),
                'variables': {n: {'dimensions': list(a.dims), 'dtype': str(a.dtype),
                    'attributes': dict(a.attrs),
                    'sample': a.isel({d: slice(0, 2) for d in a.dims}).values.tolist()}
                    for n, a in ds.variables.items()}}


def month_number(time):
    return np.asarray(time.dt.year * 12 + time.dt.month - 1, dtype=int)


def parse_month(value):
    if hasattr(value, 'year'):
        return value.year * 12 + value.month - 1
    text = str(value)
    return int(text[:4]) * 12 + int(text[5:7]) - 1


def select_period(ds, bounds):
    """Start inclusive, end exclusive; strings specify calendar months."""
    if not bounds:
        return ds
    m = month_number(ds.time)
    return ds.isel(time=np.flatnonzero((m >= parse_month(bounds[0])) & (m < parse_month(bounds[1]))))


def event_window(ds, date, bounds):
    m = month_number(ds.time) - parse_month(date)
    index = np.flatnonzero((m >= bounds[0]) & (m < bounds[1]))
    complete = len(index) == bounds[1] - bounds[0]
    return ds.isel(time=index), complete


def _unit(unit):
    return str(unit).lower().replace(' ', '').replace('**', '^').replace('degrees_', '').replace('degree_', '')


def convert_temperature(a, units):
    u = _unit(units)
    if u in ('k', 'kelvin'):
        a = a - 273.15
    elif u not in ('c', 'degc', 'celsius', '°c'):
        raise ValueError(f'Unsupported temperature units {units!r}; specify K or degC')
    return a.assign_attrs(units='degC', standard_name='air_temperature')


def convert_precipitation(a, units, representation):
    """Water-equivalent units; monthly totals use each calendar's actual days."""
    u = _unit(units)
    factors = {'kgm-2s-1': 86400., 'kgm^-2s^-1': 86400., 'mm/s': 86400.,
               'mms-1': 86400., 'm/s': 86400000., 'ms-1': 86400000.,
               'mm/day': 1., 'mmd-1': 1., 'mmday-1': 1., 'mmday^-1': 1.,
               'm/day': 1000., 'md-1': 1000., 'kgm-2day-1': 1.}
    if representation in ('flux', 'mean_rate') and u in factors:
        a = a * factors[u]
    elif representation == 'monthly_total' and u in ('mm', 'm', 'kgm-2', 'kgm^-2'):
        a = a * (1000. if u == 'm' else 1.) / a.time.dt.days_in_month
    else:
        raise ValueError(f'Ambiguous/unsupported precipitation units={units!r}, representation={representation!r}')
    return a.assign_attrs(units='mm/day', representation='mean_rate')


def _read_one(spec, config):
    ds = open_store(spec['path'], chunks=spec.get('chunks', 'auto'), decode_times=False)
    raw_attrs = dict(ds.attrs)
    if raw_attrs.get('stage') == 'encode' or 'latent_zoom' in raw_attrs:
        raise ValueError('Latent stores cannot be evaluated as physical climate fields')
    dims = spec.get('dimensions', {})  # actual -> canonical
    ds = ds.rename({a: b for a, b in dims.items() if a != b})
    for dim, selector in spec.get('selectors', {}).items():
        ds = ds.sel({dim: selector}, drop=True)
    variables = spec.get('variables', {})  # canonical -> actual
    if set(variables) != {'temperature', 'precipitation'}:
        raise ValueError('variables must explicitly map temperature and precipitation to stored names')
    if any(v not in ds for v in variables.values()):
        raise ValueError(f'Required fields absent: {variables}; available={list(ds.data_vars)}')
    ordering = str(spec.get('ordering', '')).lower()
    normalize = lambda v: {'nest': 'nested', 'nested': 'nested', 'ring': 'ring'}.get(str(v).lower(), str(v).lower())
    metadata = [raw_attrs] + [dict(ds[k].attrs) for k in ds.variables]
    for attrs in metadata:
        for key in ('healpix_order', 'ordering', 'pixel_ordering'):
            if key in attrs and normalize(attrs[key]) != normalize(ordering):
                raise ValueError(f'Configured ordering {ordering!r} conflicts with {key}={attrs[key]!r}')
        if 'healpix_nside' in attrs and int(attrs['healpix_nside']) != config.nside:
            raise ValueError('HEALPix NSIDE metadata conflicts with evaluation grid')
    if normalize(ordering) != 'nested':
        raise ValueError('Explicit ordering="nested" is required; convert RING input before evaluation')
    time_spec = spec.get('time', {})
    if 'source_path' in time_spec:
        with open_store(time_spec['source_path'], chunks=None) as source:
            time = source[time_spec.get('variable', 'time')].load()
        if time.size != ds.sizes['time']:
            raise ValueError('External time coordinate length does not match decoded store')
        ds = ds.assign_coords(time=('time', time.values))
    elif 'start' in time_spec:
        if 'calendar' not in time_spec:
            raise ValueError('Explicit start requires explicit calendar')
        ds = ds.assign_coords(time=xr.date_range(time_spec['start'], periods=ds.sizes['time'],
                             freq='MS', calendar=time_spec['calendar'], use_cftime=True))
    else:
        if np.issubdtype(ds.time.dtype, np.number) and 'since' not in ds.time.attrs.get('units', ''):
            raise ValueError('Integer time is not a date. Configure time.source_path or time.start/calendar')
        ds = xr.decode_cf(ds)
    if 'member' not in ds.dims:
        if 'single_member_id' not in spec:
            raise ValueError('No member dimension; supply single_member_id for an explicit single trajectory')
        ds = ds.expand_dims(member=[str(spec['single_member_id'])])
    pixel_name = spec.get('pixel_id', 'cell')
    expected = 12 * config.nside ** 2
    pixel_id_source = f'stored variable: {pixel_name}'
    if pixel_name not in ds.coords and pixel_name not in ds:
        # MPI-GE rechunked stores use an unindexed full-grid cell dimension.
        # Its positions are pixel IDs only when the stored HEALPix CRS declares
        # the resolution and ordering. Never infer geometry from length alone.
        grid_sources = [('dataset', raw_attrs)] + [(k, dict(ds[k].attrs)) for k in ds.variables]
        declared_grids = [name for name, attrs in grid_sources
                          if attrs.get('grid_mapping_name') == 'healpix'
                          and attrs.get('healpix_nside') == config.nside
                          and normalize(attrs.get('healpix_order', '')) == 'nested']
        if pixel_name != 'cell' or 'cell' in ds.variables or not declared_grids:
            raise ValueError('Explicit HEALPix pixel IDs are required unless an unindexed cell dimension '
                             'has stored HEALPix CRS metadata declaring healpix_nside and healpix_order="nest"')
        if ds.sizes.get('cell') != expected:
            raise ValueError(f'Implicit HEALPix indexing requires the complete {expected}-cell grid; '
                             f'found {ds.sizes.get("cell")} cells')
        ids = np.arange(expected, dtype=np.int64)
        pixel_id_source = ('implicit full-grid cell positions from stored HEALPix CRS: '
                           + ', '.join(declared_grids))
    else:
        ids = ds[pixel_name].values
    if ids.shape != (expected,) or not np.array_equal(np.sort(ids), np.arange(expected)):
        raise ValueError(f'Need exactly {expected} unique integer NESTED pixel IDs 0..{expected-1}')
    ds = ds.assign_coords(cell=('cell', ids)).sortby('cell')
    for names,axis in [(('latitude','lat'),'latitude'),(('longitude','lon'),'longitude')]:
        for name in names:
            if name not in ds or ds[name].dims!=('cell',):
                continue
            import healpy as hp
            lon,lat=hp.pix2ang(config.nside,ds.cell.values.astype(int),nest=True,lonlat=True)
            angular_units=spec.get('coordinate_units',ds[name].attrs.get('units',''))
            observed=ds[name].values.astype(float)
            if angular_units in ('radian','radians','rad'):
                observed=np.rad2deg(observed)
            elif angular_units not in ('degrees','degree','degrees_north','degrees_east'):
                raise ValueError(f'{name}: specify coordinate_units or stored angular units to verify NESTED geometry')
            expected_coordinate=lat if axis=='latitude' else lon
            difference=observed-expected_coordinate
            if axis=='longitude':difference=(difference+180)%360-180
            if not np.allclose(difference,0,atol=1e-5,rtol=0):
                raise ValueError(f'{name} does not agree with NESTED HEALPix pixel IDs')
    result = {}
    for canonical, name in variables.items():
        a = ds[name]
        if set(a.dims) != {'member', 'time', 'cell'}:
            raise ValueError(f'{name}: expected member,time,cell after selectors, found {a.dims}')
        a = a.transpose('member', 'time', 'cell')
        units = spec.get('units', {}).get(canonical, a.attrs.get('units'))
        if units is None:
            raise ValueError(f'{name}: units absent; configure units.{canonical}')
        stored = a.attrs.get('units')
        if stored and _unit(stored) != _unit(units) and not spec.get('override_units_reason'):
            raise ValueError(f'{name}: configured and stored units disagree; explicit override_units_reason required')
        if canonical == 'temperature':
            definition = spec.get('temperature_definition', a.attrs.get('standard_name'))
            if definition not in ('air_temperature', 'surface_temperature', 'skin_temperature'):
                raise ValueError('Specify temperature_definition: air_temperature or skin/surface_temperature')
            stored_definition=a.attrs.get('standard_name')
            if stored_definition and stored_definition!=definition:
                raise ValueError(f'Temperature definition {definition!r} conflicts with stored {stored_definition!r}')
            result[canonical] = convert_temperature(a, units).assign_attrs(standard_name=definition)
        else:
            representation = spec.get('precipitation_representation')
            if representation is None:
                raise ValueError('precipitation_representation must explicitly be flux, mean_rate or monthly_total')
            result[canonical] = convert_precipitation(a, units, representation)
    out = xr.Dataset(result, attrs={'source_path': str(spec['path']), 'source_attributes': json.dumps(raw_attrs, default=str),
                                  'ordering': 'nested', 'nside': config.nside,
                                  'pixel_id_source': pixel_id_source})
    # The derived lazy arrays retain their underlying store references.
    out.set_close(ds.close)
    return out


def load_fields(spec, config):
    if 'member_paths' in spec:
        parts = []
        ids = spec.get('member_ids', [Path(p).stem for p in spec['member_paths']])
        if len(ids) != len(spec['member_paths']):
            raise ValueError('member_ids and member_paths differ in length')
        for p, ident in zip(spec['member_paths'], ids):
            part = _read_one({**spec, 'path': p, 'single_member_id': ident}, config)
            if part.sizes['member'] != 1:
                raise ValueError('member_paths entries must each contain one member')
            parts.append(part.assign_coords(member=[str(ident)]))
        ds = xr.concat(parts, dim='member', join='exact', coords='minimal', compat='equals')
        ds.set_close(lambda: [p.close() for p in parts])
    else:
        ds = _read_one(spec, config)
    validate_fields(ds, config)
    return ds


def validate_fields(ds, config):
    months = month_number(ds.time)
    if len(months) == 0 or (np.diff(months) != 1).any():
        raise ValueError('Monthly timestamps must be unique, ordered and gap-free; do not fill missing months silently')
    members = ds.member.values
    if len(set(map(str, members))) != len(members):
        raise ValueError('Duplicate member identifiers')
    for variable in ('temperature', 'precipitation'):
        a = ds[variable]
        counts = xr.Dataset({'infinite': np.isinf(a).sum(), 'missing': a.isnull().sum()}).compute()
        if int(counts.infinite):
            raise ValueError(f'{variable}: nonfinite infinite values found')
        if int(counts.missing) and config.missing_policy == 'error':
            raise ValueError(f'{variable}: {int(counts.missing)} missing values; supply an explicit common_mask policy if intended')


def align_fields(generated, reference, config):
    if generated.temperature.attrs.get('standard_name') != reference.temperature.attrs.get('standard_name'):
        raise ValueError('Generated/reference temperature definitions differ (air versus skin)')
    if not np.array_equal(generated.cell, reference.cell):
        raise ValueError('Generated/reference pixel IDs differ')
    cal = lambda d: d.time.dt.calendar
    if cal(generated) != cal(reference):
        raise ValueError('Calendars differ; an explicit scientifically justified calendar conversion is required')
    gm, rm = month_number(generated.time), month_number(reference.time)
    if config.temporal_alignment == 'exact' and not np.array_equal(gm, rm):
        raise ValueError('Generated/reference calendar months differ; select matching evaluation periods')
    common = np.intersect1d(gm, rm)
    if not len(common):
        raise ValueError('No common evaluation months')
    generated = generated.isel(time=np.flatnonzero(np.isin(gm, common)))
    reference = reference.isel(time=np.flatnonzero(np.isin(rm, common))).assign_coords(time=generated.time)
    if config.missing_policy == 'common_mask':
        # Static support across variables, members and time avoids changing-area artefacts.
        mask = generated.to_array().notnull().all(('variable', 'member', 'time')) & reference.to_array().notnull().all(('variable', 'member', 'time'))
        generated, reference = generated.where(mask), reference.where(mask)
    return generated, reference, {'generated_excluded_months': len(gm)-len(common), 'reference_excluded_months': len(rm)-len(common)}


def aggregate(a, frequency='annual'):
    """Complete calendar aggregates only; mean rates stay rates, weighted by days."""
    if frequency == 'monthly':
        return a
    m, y = np.asarray(a.time.dt.month), np.asarray(a.time.dt.year)
    if frequency == 'seasonal':
        # DJF=0, MAM=1, JJA=2, SON=3; season year is January's year.
        sy, season = y + (m == 12), (m % 12) // 3
        keys, expected = sy * 4 + season, 3
    elif frequency == 'annual':
        keys, expected = y, 12
    else:
        years = int(str(frequency).removesuffix('year'))
        keys, expected = (y - y[0]) // years, 12 * years
    arrays, times = [], []
    for key in np.unique(keys):
        ix = np.flatnonzero(keys == key)
        if len(ix) != expected:
            continue
        part = a.isel(time=ix)
        arrays.append(part.weighted(part.time.dt.days_in_month.astype(float)).mean('time', keep_attrs=True))
        # Middle observed month preserves datetime/CFTime type, including DJF.
        times.append(a.time.values[ix[len(ix)//2]])
    if not arrays:
        out = a.isel(time=slice(0, 0))
    else:
        out = xr.concat(arrays, xr.IndexVariable('time', times)).transpose(*a.dims)
    out.attrs.update(aggregation=frequency, day_weighted=True,
                     excluded_months=int(len(a.time) - len(arrays)*expected))
    return out


def regions(ds, config):
    import healpy as hp
    lon, lat = hp.pix2ang(config.nside, ds.cell.values.astype(int), nest=True, lonlat=True)
    masks = {'Global': np.ones(len(lat)), 'NH': (lat >= 0).astype(float), 'SH': (lat < 0).astype(float)}
    for lo, hi in config.latitude_bands:
        masks[f'lat_{lo}_{hi}'] = ((lat >= lo) & ((lat <= hi) if hi == 90 else (lat < hi))).astype(float)
    notes = {'method': 'native HEALPix equal area; cell-centre polygons', 'ar6_status': 'Not requested'}
    if config.land_fraction_path:
        with open_store(config.land_fraction_path, chunks=None) as land:
            f = land[config.land_fraction_variable].sel(cell=ds.cell).values
        if f.shape != lat.shape or not np.isfinite(f).all() or (f < 0).any() or (f > 1).any():
            raise ValueError('Land fractions must align by cell ID and lie in [0,1]')
        masks.update(Land=f, Ocean=1-f)
    if config.fractional_mask_path:
        with open_store(config.fractional_mask_path, chunks=None) as supplied:
            a = supplied['fraction'].sel(cell=ds.cell).transpose('region', 'cell').load()
        if not np.isfinite(a).all() or float(a.min()) < 0 or float(a.max()) > 1:
            raise ValueError('Region fractions must be finite and in [0,1]')
        masks.update({str(r): a.sel(region=r).values for r in a.region.values})
        notes.update(method='supplied fractional overlap masks', ar6_status='Supplied; verify polygon provenance')
    elif config.ar6:
        cache = Path(config.cache_dir); cache.mkdir(parents=True, exist_ok=True)
        source_id = hashlib.sha256(Path(config.polygon_path).read_bytes()).hexdigest() if config.polygon_path else 'regionmask-ar6-v4'
        fingerprint = hashlib.sha256((source_id + str(config.nside) + config.polygon_id).encode()).hexdigest()[:20]
        cached = cache / f'regions-{fingerprint}.npz'
        if cached.exists():
            with np.load(cached) as data:
                masks.update({k: data[k] for k in data.files})
            notes['ar6_status'] = 'Cached polygons'
        else:
            try:
                if config.polygon_path:
                    import geopandas as gpd
                    polygons = gpd.read_file(config.polygon_path).to_crs('EPSG:4326')
                    names, geometry = polygons[config.polygon_id], polygons.geometry
                else:
                    import regionmask
                    with regionmask.set_options(cache_dir=cache / 'polygons'):
                        reg = regionmask.defined_regions.ar6.all
                    names, geometry = reg.abbrevs, reg.polygons
                import shapely
                points = shapely.points(((lon + 180) % 360)-180, lat)
                extra = {str(name): shapely.intersects(poly, points).astype(float) for name, poly in zip(names, geometry)}
                masks.update(extra)
                np.savez_compressed(cached, **extra)
                notes.update(ar6_status='Computed from AR6 polygons', polygon_source=source_id)
            except (ImportError, OSError, ConnectionError) as exc:
                notes.update(ar6_status='Missing region data', reason=str(exc))
    w = xr.DataArray(np.stack(list(masks.values())), dims=('region', 'cell'),
                     coords={'region': list(masks), 'cell': ds.cell}, name='region_weight')
    return w, notes


def regional_average(a, weights, min_coverage=.95):
    denominator = weights.sum('cell')
    valid = a.notnull()
    support = xr.dot(valid.astype(float), weights, dim='cell')
    coverage = support / denominator.where(denominator > 0)
    mean = xr.dot(a.fillna(0), weights, dim='cell') / support.where(support > 0)
    mean = mean.where(coverage >= min_coverage)
    mean.attrs.update(a.attrs)
    return mean.transpose(*[d for d in a.dims if d != 'cell'], 'region'), coverage


def load_forcings(spec, target_time=None, require_complete=True):
    """Require physical forcing arrays and an explicit AOD reduction for amplitudes."""
    if not spec:
        raise Unavailable('Missing forcing metadata', 'No forcing trajectories configured')
    if 'path' in spec:
        with open_store(spec['path'], chunks=None) as source:
            out = source[[spec.get('co2', 'co2c'), spec.get('aod', 'aod550')]].load()
        out = out.rename({spec.get('co2', 'co2c'): 'co2', spec.get('aod', 'aod550'): 'aod'})
    else:
        if target_time is None or not all(k in spec for k in ('co2_values', 'aod_values')):
            raise ValueError('Inline forcings need co2_values, aod_values and a time coordinate')
        out = xr.Dataset({'co2': ('time', spec['co2_values']), 'aod': ('time', spec['aod_values'])}, coords={'time': target_time})
    co2_unit=str(spec.get('co2_units',out.co2.attrs.get('units',''))).lower()
    aod_unit=str(spec.get('aod_units',out.aod.attrs.get('units',''))).lower()
    metadata_issues=[]
    if co2_unit in ('mol mol-1','mol/mol','1'):
        out['co2']=out.co2*1e6;out.co2.attrs['units']='ppm'
    elif co2_unit not in ('ppm','ppmv','parts per million','1e-6'):
        metadata_issues.append('CO2 units absent/unsupported; specify co2_units (normally ppm)')
    else:
        out.co2.attrs['units']='ppm'
    if aod_unit not in ('1','unitless','dimensionless'):
        metadata_issues.append('AOD units absent/unsupported; specify aod_units="1"')
    else:
        out.aod.attrs['units']='1'
    if target_time is not None:
        fm, tm = month_number(out.time), month_number(target_time)
        positions = {int(m): i for i, m in enumerate(fm)}
        if len(positions) != len(fm) or any(int(m) not in positions for m in tm):
            raise ValueError('Forcings do not uniquely cover evaluation months')
        out = out.isel(time=[positions[int(m)] for m in tm]).assign_coords(time=target_time)
    missing={v:int((~np.isfinite(out[v])).sum()) for v in out.data_vars}
    out.attrs['nonfinite_counts']=json.dumps(missing)
    out.attrs['metadata_issues']=json.dumps(metadata_issues)
    if require_complete and (any(missing.values()) or metadata_issues):
        raise Unavailable('Missing forcing metadata',f'Forcing nonfinite counts: {missing}; metadata issues: {metadata_issues}; no imputation or unit inference performed')
    return out


def scalar_aod(forcing, spec):
    a = forcing.aod
    if a.dims == ('time',):
        return a
    if a.dims != ('time', 'lat') or spec.get('aod_reduction') != 'latitude_band_area':
        raise Unavailable('Missing forcing metadata', 'Latitude-dependent AOD requires aod_reduction=latitude_band_area')
    latitude = a.lat.values
    order = np.argsort(latitude)
    a = a.isel(lat=order); latitude = latitude[order]
    edges = np.r_[-90., (latitude[1:]+latitude[:-1])/2, 90.]
    weight = np.diff(np.sin(np.deg2rad(edges)))
    return a.weighted(xr.DataArray(weight, dims='lat', coords={'lat': a.lat})).mean('lat')
