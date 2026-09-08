from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import settings
from .models import ImageMetadata


class ImageValidationError(ValueError):
    pass


@dataclass
class PreparedImage:
    array: np.ndarray
    metadata: ImageMetadata


def validate_and_prepare(data: bytes, filename: str) -> PreparedImage:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        raise ImageValidationError("Поддерживаются только JPG, PNG и GeoTIFF.")
    if not data:
        raise ImageValidationError("Файл пуст.")
    if len(data) > settings.max_file_mb * 1024 * 1024:
        raise ImageValidationError(f"Файл превышает лимит {settings.max_file_mb} МБ.")

    crs = None
    bounds = None
    transform = None
    try:
        if suffix in {".tif", ".tiff"}:
            try:
                import rasterio
                from rasterio.io import MemoryFile
            except ImportError as exc:
                raise ImageValidationError("Для GeoTIFF установите optional dependency rasterio.") from exc
            with MemoryFile(data) as mem, mem.open() as ds:
                if ds.width*ds.height>settings.max_pixels:
                    raise ImageValidationError(f"Изображение слишком большое ({ds.width}×{ds.height}).")
                bands = ds.read(list(range(1, min(ds.count, 3) + 1)))
                if bands.shape[0] == 1:
                    bands = np.repeat(bands, 3, axis=0)
                bands = bands.astype(np.float32)
                for i in range(bands.shape[0]):
                    lo, hi = np.percentile(bands[i], [2, 98])
                    bands[i] = np.clip((bands[i] - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
                array = np.moveaxis(bands[:3], 0, -1).astype(np.uint8)
                crs = str(ds.crs) if ds.crs else None
                bounds = list(ds.bounds) if ds.bounds else None
                transform = list(ds.transform)[:6]
                fmt = "GeoTIFF"
        else:
            Image.MAX_IMAGE_PIXELS=settings.max_pixels
            with Image.open(io.BytesIO(data)) as image:
                if image.width*image.height>settings.max_pixels:
                    raise ImageValidationError(f"Изображение слишком большое ({image.width}×{image.height}).")
                image.load()
                fmt = image.format or suffix.lstrip(".").upper()
                array = np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        if isinstance(exc,ImageValidationError): raise
        raise ImageValidationError("Файл повреждён или не является корректным изображением.") from exc

    height, width = array.shape[:2]
    if width * height > settings.max_pixels:
        raise ImageValidationError(
            f"Изображение слишком большое ({width}×{height}). Лимит: {settings.max_pixels:,} пикселей."
        )
    metadata = ImageMetadata(
        filename=Path(filename).name,
        format=fmt,
        width=width,
        height=height,
        sha256=hashlib.sha256(data).hexdigest(),
        crs=crs,
        bounds=bounds,
        transform=transform,
    )
    return PreparedImage(array=array, metadata=metadata)
