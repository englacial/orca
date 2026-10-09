"""Reduced results: small xarray Datasets stored as reduced.zarr.zip (zarr ZipStore)."""
import tempfile
import zipfile
from pathlib import Path

import xarray as xr
import zarr.storage

from . import yamlio

REQUIRED_ATTRS = ("test_id", "run_id", "git_sha", "fixture_id", "tier")


def write_reduced(ds: xr.Dataset, path, **attrs):
    """Write ds as a zipped zarr store at ``path``. ``attrs`` are added to the Dataset attrs.

    The store is written to a temporary directory and then zipped (uncompressed):
    writing a ZipStore directly leaves duplicate entries, because xarray rewrites
    metadata documents during a write.
    """
    ds = ds.copy()
    ds.attrs = yamlio.to_builtin({**ds.attrs, **attrs})
    for var in list(ds.variables):
        ds[var].attrs = yamlio.to_builtin(ds[var].attrs)
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        ds.to_zarr(tmp, mode="w", consolidated=False)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(Path(tmp).rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(tmp).as_posix())
    return path


def read_reduced(path) -> xr.Dataset:
    store = zarr.storage.ZipStore(str(path), mode="r")
    try:
        return xr.open_zarr(store, consolidated=False).load()
    finally:
        store.close()
