"""Map source pixel edges through the original affine transform into WGS84."""

from .schemas import GeoJSONGeometry


def pixel_polygon(source, box):
    if source.crs is None or source.transform is None:
        return None
    from rasterio.warp import transform

    a, b, c, d, e, f = source.transform
    x1, y1, x2, y2 = box
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    xs = [a * x + b * y + c for x, y in pts]
    ys = [d * x + e * y + f for x, y in pts]
    lon, lat = transform(source.crs, "EPSG:4326", xs, ys)
    return GeoJSONGeometry(
        type="Polygon", coordinates=[[[x, y] for x, y in zip(lon, lat)]]
    )
