"""Read source RGB pixels without resize, reprojection or radiometric guessing."""

from __future__ import annotations
import hashlib
from pathlib import Path
from .optical_contracts import RasterSource


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tile_windows(width: int, height: int, size: int = 512, overlap: int = 128):
    if width <= 0 or height <= 0 or size < 32 or size % 32 or not 0 <= overlap < size:
        raise ValueError(
            "positive raster, tile size multiple of 32, and 0 <= overlap < size required"
        )

    def origins(n):
        if n <= size:
            return [0]
        return sorted(set(list(range(0, n - size + 1, size - overlap)) + [n - size]))

    for y in origins(height):
        for x in origins(width):
            yield (x, y, min(size, width - x), min(size, height - y))


class NativeRaster:
    def __init__(self, path: Path, source: RasterSource):
        import rasterio

        self.path = Path(path)
        if sha256_file(self.path) != source.sha256:
            raise ValueError("source checksum mismatch")
        sidecar = Path(str(self.path) + ".aux.xml")
        if source.sidecar_sha256 and (
            not sidecar.is_file() or sha256_file(sidecar) != source.sidecar_sha256
        ):
            raise ValueError("source georeferencing sidecar checksum mismatch")
        self.dataset = rasterio.open(self.path)
        try:
            ds = self.dataset
            if ds.width != source.width or ds.height != source.height:
                raise ValueError("source raster dimensions mismatch")
            if ds.count != 3 or any(t != "uint8" for t in ds.dtypes):
                raise ValueError(
                    "AW-01 accepts only explicit uint8 three-band RGB; prepare other radiometry explicitly"
                )
            from rasterio.enums import ColorInterp

            if tuple(ds.colorinterp) != (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
            ):
                raise ValueError("explicit RGB channel interpretation required")
            if source.crs and (ds.crs is None or ds.crs.to_string() != source.crs):
                raise ValueError("source CRS mismatch")
            if source.transform and any(
                abs(a - b) > 1e-8
                for a, b in zip(tuple(ds.transform)[:6], source.transform)
            ):
                raise ValueError("source geotransform mismatch")
        except BaseException:
            self.dataset.close()
            raise

    def read(self, window, size):
        import numpy as np
        from rasterio.windows import Window

        x, y, w, h = window
        pixels = self.dataset.read(window=Window(x, y, w, h))
        valid = self.dataset.dataset_mask(window=Window(x, y, w, h)) > 0
        # Padding only: pixel i remains pixel i, including non-square edge tiles.
        out = np.zeros((3, size, size), dtype=np.uint8)
        out[:, :h, :w] = np.where(valid[None], pixels, 0)
        return out, valid

    def close(self):
        self.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def pixel_spacing_m(crs, affine, width, height):
    """Local source-grid spacing at raster centre; transform points, never pixels."""
    import math
    from rasterio.warp import transform

    x, y = width / 2, height / 2
    coords = [affine * (x, y), affine * (x + 1, y), affine * (x, y + 1)]
    lon, lat = transform(crs, "EPSG:4326", [coords[0][0]], [coords[0][1]])
    local = f"+proj=aeqd +lat_0={lat[0]} +lon_0={lon[0]} +datum=WGS84 +units=m +no_defs"
    xs, ys = transform(crs, local, [p[0] for p in coords], [p[1] for p in coords])
    return (
        math.hypot(xs[1] - xs[0], ys[1] - ys[0]),
        math.hypot(xs[2] - xs[0], ys[2] - ys[0]),
    )
