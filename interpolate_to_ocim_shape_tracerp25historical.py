import numpy as np
import xarray as xr
import xesmf as xe
import scipy as sp
from scipy.interpolate import interp1d, CubicSpline


load_path = '/Volumes/CHEN/GFDLdata/tracers_sigma2/'
save_path_intermediate = '/Volumes/CHEN/GFDLdata/historical_p25/intermediate/'
save_path_final = '/Volumes/CHEN/GFDLdata/historical_p25/final/'

'''
Load CYCLOCIM depth coordinate
'''
# data = sp.io.loadmat('/Users/chen/coding/CM4X/CYCLOCIM/2x2.mat')
data = sp.io.loadmat('/Volumes/CHEN/GFDLdata/2x2.mat')

msk = data['M3d']
grd = data['grd']

new_lon = grd[0][0]['xt'][0]
new_lat = grd[0][0]['yt'][0]
new_depth = grd[0][0]['zt'][0]

'''
Density coord to Depth
'''

def to_depth_coord(thickness, concentration):
    valid_idx = ~np.isnan(thickness) & ~np.isnan(concentration)
    if valid_idx.sum() == 0:
        return np.nan
    valid_thickness = thickness[valid_idx]
    valid_concentration = concentration[valid_idx]
    
    layer_bottom = np.concatenate(([0], np.cumsum(valid_thickness)[:-1]))
    layer_top = np.cumsum(valid_thickness)
    # Calculate mid-depth of every layer
    layer_mid = (layer_bottom + layer_top) / 2
    return layer_mid, valid_concentration

def density_interpolation(thickness, concentration, interp_depths): 
    valid_idx = ~np.isnan(thickness) & ~np.isnan(concentration)
    if valid_idx.sum() == 0:
        return np.full(interp_depths.shape, np.nan)
    elif valid_idx.sum() == 1:
        return np.full(interp_depths.shape, concentration[valid_idx])
    valid_thickness = thickness[valid_idx]
    valid_concentration = concentration[valid_idx]
    
    layer_bottom = np.concatenate(([0], np.cumsum(valid_thickness)[:-1]))
    layer_top = np.cumsum(valid_thickness)
    # Calculate mid-depth of every layer
    layer_mid = (layer_bottom + layer_top) / 2
    
    # target interpolation depth (CYCLOCIM depth)
    interp_func = interp1d(layer_mid, valid_concentration, kind='linear', bounds_error=False, 
                           fill_value=(valid_concentration[0], valid_concentration[-1]))  # fill_value='extrapolate'
    
    interp_concentration = interp_func(interp_depths)
    return interp_concentration

for i in np.arange(1850, 2011, 5): 
    # ds = xr.open_zarr('/Users/chen/coding/CM4X/data/CM4Xp25_piControl_tracers_sigma2_0291-0295.zarr')
    ds = xr.open_zarr(load_path + f'CM4Xp25_historical_tracers_sigma2_{i}-{i+4}.zarr')

    '''
    Sea Water Salinity
    '''
    interp_so = xr.apply_ufunc(
        density_interpolation,
        ds['thkcello'], 
        ds['so'], 
        input_core_dims=[['sigma2_l'], ['sigma2_l']],
        output_core_dims=[['depth']],
        kwargs={'interp_depths': new_depth}, 
        vectorize=True, 
        dask='parallelized', 
        dask_gufunc_kwargs={'output_sizes': {'depth': len(new_depth)}},
        output_dtypes=[ds['so'].dtype]
    )

    interp_so = interp_so.assign_coords(depth=new_depth)

    '''
    CFC11
    '''
    interp_cfc11 = xr.apply_ufunc(
        density_interpolation,
        ds['thkcello'], 
        ds['cfc11'], 
        input_core_dims=[['sigma2_l'], ['sigma2_l']],
        output_core_dims=[['depth']],
        kwargs={'interp_depths': new_depth}, 
        vectorize=True, 
        dask='parallelized', 
        dask_gufunc_kwargs={'output_sizes': {'depth': len(new_depth)}},
        output_dtypes=[ds['cfc11'].dtype]
    )

    interp_cfc11 = interp_cfc11.assign_coords(depth=new_depth)

    '''
    CFC12
    '''
    interp_cfc12 = xr.apply_ufunc(
        density_interpolation,
        ds['thkcello'], 
        ds['cfc12'], 
        input_core_dims=[['sigma2_l'], ['sigma2_l']],
        output_core_dims=[['depth']],
        kwargs={'interp_depths': new_depth}, 
        vectorize=True, 
        dask='parallelized', 
        dask_gufunc_kwargs={'output_sizes': {'depth': len(new_depth)}},
        output_dtypes=[ds['cfc12'].dtype]
    )

    interp_cfc12 = interp_cfc12.assign_coords(depth=new_depth)

    '''
    SF6
    '''
    interp_sf6 = xr.apply_ufunc(
        density_interpolation,
        ds['thkcello'], 
        ds['sf6'], 
        input_core_dims=[['sigma2_l'], ['sigma2_l']],
        output_core_dims=[['depth']],
        kwargs={'interp_depths': new_depth}, 
        vectorize=True, 
        dask='parallelized', 
        dask_gufunc_kwargs={'output_sizes': {'depth': len(new_depth)}},
        output_dtypes=[ds['sf6'].dtype]
    )

    interp_sf6 = interp_sf6.assign_coords(depth=new_depth)

    '''
    thetao
    Sea Water Potential Temperature
    '''
    interp_thetao = xr.apply_ufunc(
        density_interpolation,
        ds['thkcello'], 
        ds['thetao'], 
        input_core_dims=[['sigma2_l'], ['sigma2_l']],
        output_core_dims=[['depth']],
        kwargs={'interp_depths': new_depth}, 
        vectorize=True, 
        dask='parallelized', 
        dask_gufunc_kwargs={'output_sizes': {'depth': len(new_depth)}},
        output_dtypes=[ds['thetao'].dtype]
    )

    interp_thetao = interp_thetao.assign_coords(depth=new_depth)

    '''
    agessc
    ideal_age_tracer
    '''
    interp_agessc = xr.apply_ufunc(
        density_interpolation,
        ds['thkcello'], 
        ds['agessc'], 
        input_core_dims=[['sigma2_l'], ['sigma2_l']],
        output_core_dims=[['depth']],
        kwargs={'interp_depths': new_depth}, 
        vectorize=True, 
        dask='parallelized', 
        dask_gufunc_kwargs={'output_sizes': {'depth': len(new_depth)}},
        output_dtypes=[ds['agessc'].dtype]
    )

    interp_agessc = interp_agessc.assign_coords(depth=new_depth)

    ### Density coord to Depth
    ds1 = xr.Dataset(
        {
            'so': interp_so, 
            'cfc11': interp_cfc11, 
            'cfc12': interp_cfc12, 
            'sf6': interp_sf6, 
            'thetao': interp_thetao, 
            'agessc': interp_agessc
        },
        coords={
            'time': ds['time'],
            'yh': ds['yh'],
            'xh': ds['xh'],
            'depth': new_depth,

            'geolon': ds['geolon'],
            'geolat': ds['geolat']
        }
    )

    '''
    Save intermediate dataset
    '''
    ds1.to_netcdf(save_path_intermediate + f'CM4Xp25_historical_tracers_sigma2_{i}-{i+4}.nc')

    '''
    Load intermediate dataset
    '''
    ds2 = xr.open_dataset(save_path_intermediate + f'CM4Xp25_historical_tracers_sigma2_{i}-{i+4}.nc')
    ds2 = ds2.assign_coords(lat=ds2.geolat, lon=ds2.geolon)

    '''
    New lat & lon coord
    '''
    new_grid = xr.Dataset({
        'lat': (['lat'], new_lat),
        'lon': (['lon'], new_lon)
    })

    regridder = xe.Regridder(ds2, new_grid, method='nearest_s2d', periodic=True)  # bilinear, nearest_s2d

    so_regridded = regridder(ds2['so'])
    cfc11_regridded = regridder(ds2['cfc11'])
    cfc12_regridded = regridder(ds2['cfc12'])
    sf6_regridded = regridder(ds2['sf6'])
    thetao_regridded = regridder(ds2['thetao'])
    agessc_regridded = regridder(ds2['agessc'])

    new_ds = xr.Dataset({
        'so': so_regridded, 
        'cfc11': cfc11_regridded, 
        'cfc12': cfc12_regridded, 
        'sf6': sf6_regridded, 
        'thetao': thetao_regridded, 
        'agessc': agessc_regridded
    }, coords={
        'time': ds2['time'],
        'depth': ds2['depth'],
        'lat': new_grid['lat'],
        'lon': new_grid['lon']
    })

    '''
    Save new dataset
    '''
    new_ds.to_netcdf(save_path_final + f'CM4Xp25_historical_tracers_sigma2_{i}-{i+4}.nc')

    print(f'Finished {i} to {i+4}')
    print('-------------------------------------')
