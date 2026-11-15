"""Fast builder for attributes.zarr and streamflow.zarr for the 8,811 post-2016 public Caravan basins."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import time
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
import yaml

CARAVAN_NC_DIR = Path('./data/data/Caravan-nc')
BASE_DAY = np.datetime64('1980-01-01', 'D')
N_DATES = len(pd.date_range('1980-01-01', '2024-12-31', freq='D'))


def _read_one_nc(idx_basin: tuple[int, str]) -> tuple[int, np.ndarray, np.ndarray]:
    idx, basin = idx_basin
    sub = basin.partition('_')[0]
    nc_path = CARAVAN_NC_DIR / 'timeseries' / 'netcdf' / sub / f'{basin}.nc'
    with netCDF4.Dataset(nc_path, 'r') as ds:
        sf_var = ds.variables['streamflow']
        sf_raw = sf_var[:]
        if np.ma.isMaskedArray(sf_raw):
            sf = sf_raw.filled(np.nan).astype(np.float32)
        else:
            sf = np.asarray(sf_raw, dtype=np.float32)
        d_var = ds.variables['date']
        d_raw = np.asarray(d_var[:])
        units = getattr(d_var, 'units', 'days since 1950-01-01')
        origin_str = units.split('since')[1].strip().split()[0]
        origin = np.datetime64(origin_str, 'D')
        file_days = origin + d_raw.astype('timedelta64[D]')
        offsets = (file_days - BASE_DAY).astype(np.int32)
        valid = (offsets >= 0) & (offsets < N_DATES)
        return idx, offsets[valid], sf[valid]


def main() -> None:
    t_start = time.time()
    root = Path('./data')
    cov_df = pd.read_csv(root / 'data' / 'caravan_basin_streamflow_coverage.csv')
    post2016_df = cov_df[cov_df['n_valid_post2016'] > 0].copy()
    basins = sorted(post2016_df['gauge_id'].tolist())
    print(f'Selected {len(basins)} public Caravan basins with post-2016 streamflow data.', flush=True)

    attr_out = root / 'data' / 'attributes.zarr'
    if not attr_out.exists():
        with open(root / 'configs' / 'base_v2_config.yml', 'r') as f:
            base_cfg = yaml.safe_load(f)
        static_attrs = base_cfg['static_attributes']
        t0 = time.time()
        attr_root = CARAVAN_NC_DIR / 'attributes'
        subdatasets = sorted({b.partition('_')[0] for b in basins})
        sub_dfs = []
        for sub in subdatasets:
            sub_dir = attr_root / sub
            csvs = sorted(sub_dir.glob('*.csv'))
            part_dfs = [pd.read_csv(c, index_col='gauge_id') for c in csvs]
            sub_df = pd.concat(part_dfs, axis=1)
            sub_df = sub_df.loc[:, ~sub_df.columns.duplicated()]
            sub_dfs.append(sub_df)
        full_attr_df = pd.concat(sub_dfs, axis=0)
        full_attr_df.index.name = 'basin'
        attr_df = full_attr_df.loc[basins, static_attrs].astype(np.float32)
        attr_ds = xr.Dataset.from_dataframe(attr_df)
        for v in attr_ds.data_vars:
            attr_ds[v] = attr_ds[v].astype(np.float32)
        attr_ds.chunk({'basin': len(basins)}).to_zarr(attr_out, mode='w', consolidated=False)
        print(f'Saved {attr_out} in {time.time() - t0:.2f}s', flush=True)
    else:
        print(f'{attr_out} already exists.', flush=True)

    # 2. Fast parallel netCDF4 load of streamflow -> streamflow.zarr
    t0 = time.time()
    dates = pd.date_range('1980-01-01', '2024-12-31', freq='D', name='date')
    sf_matrix = np.full((len(basins), N_DATES), np.nan, dtype=np.float32)

    with ProcessPoolExecutor(max_workers=16) as ex:
        for idx, offs, vals in ex.map(_read_one_nc, enumerate(basins), chunksize=64):
            sf_matrix[idx, offs] = vals

    print(f'Read all 8,811 NetCDFs into float32 matrix {sf_matrix.shape} in {time.time() - t0:.2f}s', flush=True)

    sf_ds = xr.Dataset(
        data_vars={'streamflow': (('basin', 'date'), sf_matrix)},
        coords={'basin': np.array(basins, dtype=str), 'date': dates},
    )
    sf_out = root / 'data' / 'streamflow.zarr'
    if sf_out.exists():
        shutil.rmtree(sf_out)
    sf_ds.chunk({'basin': len(basins), 'date': 3650}).to_zarr(sf_out, mode='w', consolidated=False)
    print(f'Saved {sf_out} in {time.time() - t0:.2f}s (total elapsed {time.time() - t_start:.2f}s)', flush=True)


if __name__ == '__main__':
    main()
