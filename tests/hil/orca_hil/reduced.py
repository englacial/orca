"""Reduced results: small xarray Datasets stored as reduced.zarr.zip (zarr ZipStore)."""
import xarray as xr
import zarr.storage

from . import yamlio

REQUIRED_ATTRS = ("test_id", "run_id", "git_sha", "fixture_id", "tier")


def write_reduced(ds: xr.Dataset, path, **attrs):
    """Write ds to a zarr ZipStore at ``path``. ``attrs`` are added to the Dataset attrs."""
    ds = ds.copy()
    ds.attrs = yamlio.to_builtin({**ds.attrs, **attrs})
    for var in list(ds.variables):
        ds[var].attrs = yamlio.to_builtin(ds[var].attrs)
    store = zarr.storage.ZipStore(str(path), mode="w")
    try:
        ds.to_zarr(store, mode="w", consolidated=False)
    finally:
        store.close()
    return path


def read_reduced(path) -> xr.Dataset:
    store = zarr.storage.ZipStore(str(path), mode="r")
    try:
        return xr.open_zarr(store, consolidated=False).load()
    finally:
        store.close()
