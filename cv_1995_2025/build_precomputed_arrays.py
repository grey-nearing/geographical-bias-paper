"""Build pre-unioned memory-mapped float32 arrays for the 8,811 post-2016 public Caravan basins.

Produces ./data/data/precomputed/:
- basins.txt (8,811 basins)
- dates.npy (15,712 dates: 1981-01-01 .. 2024-01-07)
- statics.npy (8811, 200) float32
- streamflow.npy (8811, 15712) float32
- hindcast_2d.npy (8811, 15712, 9) float32
- forecast_3d_post2016.npy (8811, 2936, 7, 7) float32 [for dates 2015-12-25 .. 2024-01-07]
- forecast_3d_sum.npy (8811, 15712, 7) float64 [sum over 7 lead times per (basin, date)]
- forecast_3d_sq_sum.npy (8811, 15712, 7) float64 [sum of squares over 7 lead times per (basin, date)]
- forecast_3d_count.npy (8811, 15712, 7) int32 [valid count over 7 lead times per (basin, date)]
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import xarray as xr
import yaml

HINDCAST_FEATURES = [
    'hres_surface_net_solar_radiation',
    'hres_surface_net_thermal_radiation',
    'hres_surface_pressure',
    'hres_temperature_2m',
    'hres_total_precipitation',
    'graphcast_temperature_2m',
    'graphcast_total_precipitation',
    'imerg_precipitation',
    'cpc_precipitation',
]

FORECAST_FEATURES = [
    'hres_surface_net_solar_radiation',
    'hres_surface_net_thermal_radiation',
    'hres_surface_pressure',
    'hres_temperature_2m',
    'hres_total_precipitation',
    'graphcast_temperature_2m',
    'graphcast_total_precipitation',
]

UNION_MAPPING = {
    'cpc_precipitation': 'era5land_total_precipitation',
    'imerg_precipitation': 'era5land_total_precipitation',
    'graphcast_temperature_2m': 'era5land_temperature_2m',
    'graphcast_total_precipitation': 'era5land_total_precipitation',
    'hres_surface_net_solar_radiation': 'era5land_surface_net_solar_radiation',
    'hres_surface_net_thermal_radiation': 'era5land_surface_net_thermal_radiation',
    'hres_surface_pressure': 'era5land_surface_pressure',
    'hres_temperature_2m': 'era5land_temperature_2m',
    'hres_total_precipitation': 'era5land_total_precipitation',
}


def main() -> None:
    t_start = time.time()
    root = Path('./data')
    out_dir = root / 'data' / 'precomputed'
    out_dir.mkdir(parents=True, exist_ok=True)

    basins = (root / 'splits' / 'caravan_post2016_basins.txt').read_text().strip().splitlines()
    n_basins = len(basins)
    dates = pd.date_range('1981-01-01', '2024-01-07', freq='D', name='date')
    n_dates = len(dates)
    post2016_start = pd.Timestamp('2015-12-25')
    post2016_offset = int((post2016_start - dates[0]).days)
    dates_post2016 = dates[post2016_offset:]
    n_dates_post2016 = len(dates_post2016)
    print(
        f'Building precomputed arrays for {n_basins} basins, {n_dates} days '
        f'(post2016 3D window: {dates_post2016[0].strftime("%Y-%m-%d")}..{dates_post2016[-1].strftime("%Y-%m-%d")}, '
        f'{n_dates_post2016} days, offset={post2016_offset})',
        flush=True,
    )

    (out_dir / 'basins.txt').write_text('\n'.join(basins) + '\n')
    np.save(out_dir / 'dates.npy', dates.values)

    with open(root / 'configs' / 'base_v2_config.yml', 'r') as f:
        base_cfg = yaml.safe_load(f)
    static_attrs = base_cfg['static_attributes']
    (out_dir / 'static_attributes.txt').write_text('\n'.join(static_attrs) + '\n')

    # 1. Statics
    if not (out_dir / 'statics.npy').exists():
        t0 = time.time()
        attr_ds = xr.open_zarr(root / 'data' / 'attributes.zarr', consolidated=False).sel(basin=basins)
        statics = np.column_stack([attr_ds[a].values.astype(np.float32) for a in static_attrs])
        np.save(out_dir / 'statics.npy', statics)
        print(f'[1/4] Saved statics.npy {statics.shape} in {time.time() - t0:.2f}s', flush=True)
    else:
        print('[1/4] statics.npy already exists.', flush=True)

    # 2. Streamflow
    if not (out_dir / 'streamflow.npy').exists():
        t0 = time.time()
        sf_ds = xr.open_zarr(root / 'data' / 'streamflow.zarr', consolidated=False).sel(basin=basins, date=dates)
        sf = sf_ds['streamflow'].transpose('basin', 'date').values.astype(np.float32)
        np.save(out_dir / 'streamflow.npy', sf)
        print(f'[2/4] Saved streamflow.npy {sf.shape} in {time.time() - t0:.2f}s', flush=True)
        del sf
    else:
        print('[2/4] streamflow.npy already exists.', flush=True)

    # 3. Load ERA5_LAND fallback bands for 1981-01-01 .. 2024-01-14
    t0 = time.time()
    dates_ext = pd.date_range('1981-01-01', '2024-01-14', freq='D', name='date')
    era5_bands = sorted(set(UNION_MAPPING.values()))
    era5_ds = (
        xr.open_zarr(root / 'data' / 'dynamics' / 'ERA5_LAND' / 'timeseries.zarr', consolidated=False)[era5_bands]
        .sel(basin=basins, date=dates_ext)
        .compute()
    )
    era5_dict = {
        b: era5_ds[b].transpose('basin', 'date').values.astype(np.float32)
        for b in era5_bands
    }
    del era5_ds
    print(f'[3/4] Loaded ERA5_LAND fallbacks {len(era5_bands)} bands in {time.time() - t0:.2f}s', flush=True)

    # Precompute 7-day rolling sum and sq_sum of each ERA5_LAND band for pre-2016 dates
    era5_7d_sum = {}
    era5_7d_sq_sum = {}
    for b, arr in era5_dict.items():
        arr64 = arr.astype(np.float64)
        s = np.zeros((n_basins, n_dates), dtype=np.float64)
        sq = np.zeros((n_basins, n_dates), dtype=np.float64)
        for k in range(7):
            s += arr64[:, k : k + n_dates]
            sq += arr64[:, k : k + n_dates] ** 2
        era5_7d_sum[b] = s
        era5_7d_sq_sum[b] = sq
        del arr64

    # 4. Build hindcast_2d.npy (8811, 15712, 9) and forecast_3d_post2016.npy (8811, 2936, 7, 7)
    t0 = time.time()
    hindcast_2d = np.empty((n_basins, n_dates, len(HINDCAST_FEATURES)), dtype=np.float32)
    forecast_3d_post2016 = np.empty((n_basins, n_dates_post2016, 7, len(FORECAST_FEATURES)), dtype=np.float32)
    forecast_3d_sum = np.zeros((n_basins, n_dates, len(FORECAST_FEATURES)), dtype=np.float64)
    forecast_3d_sq_sum = np.zeros((n_basins, n_dates, len(FORECAST_FEATURES)), dtype=np.float64)
    forecast_3d_count = np.full((n_basins, n_dates, len(FORECAST_FEATURES)), 7, dtype=np.int32)

    lead_slice = slice(pd.Timedelta(days=1), pd.Timedelta(days=7))

    # 3D products (HRES, GRAPHCAST) only exist from 2016-01-01 onwards!
    for prod_name, prod_features in [
        ('HRES', [f for f in HINDCAST_FEATURES if f.startswith('hres_')]),
        ('GRAPHCAST', [f for f in HINDCAST_FEATURES if f.startswith('graphcast_')]),
    ]:
        tp = time.time()
        p_ds = (
            xr.open_zarr(
                root / 'data' / 'dynamics' / prod_name / 'timeseries.zarr',
                consolidated=False,
                decode_timedelta=True,
            )[prod_features]
            .sel(basin=basins, lead_time=lead_slice)
            .reindex(date=dates_post2016)
            .compute()
        )
        for feat in prod_features:
            h_idx = HINDCAST_FEATURES.index(feat)
            f_idx = FORECAST_FEATURES.index(feat)
            fb_name = UNION_MAPPING[feat]
            fallback = era5_dict[fb_name]

            # Pre-2016 (0 .. post2016_offset) is 100% fallback
            hindcast_2d[:, :post2016_offset, h_idx] = fallback[:, :post2016_offset]
            forecast_3d_sum[:, :post2016_offset, f_idx] = era5_7d_sum[fb_name][:, :post2016_offset]
            forecast_3d_sq_sum[:, :post2016_offset, f_idx] = era5_7d_sq_sum[fb_name][:, :post2016_offset]

            # Post-2016 (dates_post2016, length 2936)
            raw_3d = p_ds[feat].transpose('basin', 'date', 'lead_time').values.astype(np.float32)
            unioned_3d = np.empty_like(raw_3d)
            for k in range(7):
                rk = raw_3d[:, :, k]
                fb_k = fallback[:, post2016_offset + k : post2016_offset + k + n_dates_post2016]
                unioned_3d[:, :, k] = np.where(np.isnan(rk), fb_k, rk)

            assert not np.isnan(unioned_3d).any(), f'NaNs remain in unioned 3D {feat}'
            hindcast_2d[:, post2016_offset:, h_idx] = unioned_3d[:, :, 0]
            forecast_3d_post2016[:, :, :, f_idx] = unioned_3d
            u64 = unioned_3d.astype(np.float64)
            forecast_3d_sum[:, post2016_offset:, f_idx] = u64.sum(axis=2)
            forecast_3d_sq_sum[:, post2016_offset:, f_idx] = (u64 ** 2).sum(axis=2)
            del raw_3d, unioned_3d, u64

        del p_ds
        print(f'  Processed {prod_name} ({len(prod_features)} vars) in {time.time() - tp:.2f}s', flush=True)

    # 2D products (IMERG, CPC)
    for prod_name, prod_features in [
        ('IMERG', ['imerg_precipitation']),
        ('CPC', ['cpc_precipitation']),
    ]:
        tp = time.time()
        p_ds = (
            xr.open_zarr(
                root / 'data' / 'dynamics' / prod_name / 'timeseries.zarr',
                consolidated=False,
                decode_timedelta=True,
            )[prod_features]
            .sel(basin=basins)
            .reindex(date=dates)
            .compute()
        )
        for feat in prod_features:
            h_idx = HINDCAST_FEATURES.index(feat)
            fallback = era5_dict[UNION_MAPPING[feat]]
            raw_2d = p_ds[feat].transpose('basin', 'date').values.astype(np.float32)
            unioned_2d = np.where(np.isnan(raw_2d), fallback[:, :n_dates], raw_2d)
            assert not np.isnan(unioned_2d).any(), f'NaNs remain in unioned 2D {feat}'
            hindcast_2d[:, :, h_idx] = unioned_2d
            del raw_2d, unioned_2d
        del p_ds
        print(f'  Processed {prod_name} ({len(prod_features)} vars) in {time.time() - tp:.2f}s', flush=True)

    np.save(out_dir / 'hindcast_2d.npy', hindcast_2d)
    np.save(out_dir / 'forecast_3d_post2016.npy', forecast_3d_post2016)
    np.save(out_dir / 'forecast_3d_sum.npy', forecast_3d_sum)
    np.save(out_dir / 'forecast_3d_sq_sum.npy', forecast_3d_sq_sum)
    np.save(out_dir / 'forecast_3d_count.npy', forecast_3d_count)
    print(f'[4/4] Saved all precomputed arrays in {time.time() - t0:.2f}s (total {time.time() - t_start:.2f}s)', flush=True)


if __name__ == '__main__':
    main()
