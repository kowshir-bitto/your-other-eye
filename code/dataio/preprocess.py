
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _retina_mask(bgr: np.ndarray, thresh_frac: float = 0.06) -> np.ndarray:
    m = bgr.max(axis=2)
    thr = max(float(m.max()) * thresh_frac, 8.0)
    mask = m > thr

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return mask


def circle_crop(bgr: np.ndarray, pad_frac: float = 0.0) -> np.ndarray:
    mask = _retina_mask(bgr)
    if mask.sum() < 0.05 * mask.size:
        return bgr

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return bgr

    y0, y1 = rows[0], rows[-1] + 1
    x0, x1 = cols[0], cols[-1] + 1

    if pad_frac:
        ph = int((y1 - y0) * pad_frac)
        pw = int((x1 - x0) * pad_frac)
        y0, y1 = max(0, y0 - ph), min(bgr.shape[0], y1 + ph)
        x0, x1 = max(0, x0 - pw), min(bgr.shape[1], x1 + pw)

    out = bgr[y0:y1, x0:x1]
    return out if out.size else bgr


def pad_to_square(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h == w:
        return bgr
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=bgr.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = bgr
    return canvas


def ben_graham(bgr: np.ndarray, sigma_frac: float = 0.1, weight: float = 4.0) -> np.ndarray:
    side = max(bgr.shape[:2])
    sigma = max(side * sigma_frac, 1.0)
    blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
    out = cv2.addWeighted(bgr, weight, blur, -weight, 128)

    mask = _retina_mask(bgr)
    out[~mask] = 0
    return out


def canonicalise_laterality(bgr: np.ndarray, eye: str) -> np.ndarray:
    if str(eye).upper() in {"OS", "LEFT", "L"}:
        return cv2.flip(bgr, 1)
    return bgr


def load_fundus(
    path: str | Path,
    eye: str = "",
    size: int = 224,
    do_circle_crop: bool = True,
    do_laterality: bool = True,
    do_ben_graham: bool = False,
    mean: np.ndarray | tuple = tuple(IMAGENET_MEAN),
    std: np.ndarray | tuple = tuple(IMAGENET_STD),
) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not decode image: {path}")

    if do_circle_crop:
        bgr = circle_crop(bgr)
    if do_laterality and eye:
        bgr = canonicalise_laterality(bgr, eye)

    bgr = pad_to_square(bgr)
    work = max(size, 448)
    if max(bgr.shape[:2]) > work:
        bgr = cv2.resize(bgr, (work, work), interpolation=cv2.INTER_AREA)
    if do_ben_graham:
        bgr = ben_graham(bgr)

    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def load_fundus_rgb(
    path: str | Path,
    eye: str = "",
    size: int = 448,
    **kw,
) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not decode image: {path}")
    if kw.get("do_circle_crop", True):
        bgr = circle_crop(bgr)
    if kw.get("do_laterality", True) and eye:
        bgr = canonicalise_laterality(bgr, eye)
    bgr = pad_to_square(bgr)
    work = max(size, 448)
    if max(bgr.shape[:2]) > work:
        bgr = cv2.resize(bgr, (work, work), interpolation=cv2.INTER_AREA)
    if kw.get("do_ben_graham", False):
        bgr = ben_graham(bgr)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_preview(paths_eyes: list[tuple[str, str]], out_png: Path, size: int = 256) -> None:
    tiles = []
    for p, eye in paths_eyes:
        try:
            tiles.append(load_fundus_rgb(p, eye, size=size))
        except (FileNotFoundError, cv2.error):
            tiles.append(np.zeros((size, size, 3), np.uint8))
    if not tiles:
        return
    cols = min(6, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * size, cols * size, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * size:(r + 1) * size, c * size:(c + 1) * size] = t
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out_png)
