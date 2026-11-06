"""Ввод-вывод в формате ТЗ.

  * Входы: 4 PNG 128×128, 8 бит, 1 канал.
  * ДОЭ: PNG 128×128, 8 бит; пиксель 0 = −π, 255 = +π.
  * Восстановления: 4 PNG 128×128, 8 бит, 1 канал.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def phase_to_uint8(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    phi = np.mod(phi + math.pi, 2 * math.pi) - math.pi        # в [-π, π)
    return np.clip(np.round((phi + math.pi) / (2 * math.pi) * 255), 0, 255).astype(np.uint8)


def uint8_to_phase(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) / 255 * 2 * math.pi - math.pi


def load_image(path, n: int = 128) -> np.ndarray:
    """PNG -> float32 (n, n) в [0, 1]. Цветные переводятся в серый, размер проверяется."""
    img = Image.open(path).convert("L")
    if img.size != (n, n):
        img = img.resize((n, n), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def list_images(folder) -> list:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def load_targets(source, n: int = 128) -> np.ndarray:
    """Папка с 4 картинками (по алфавиту = по плоскостям) или список путей -> (4, n, n)."""
    paths = list_images(source) if isinstance(source, (str, Path)) and Path(source).is_dir() else list(source)
    if len(paths) != 4:
        raise ValueError(f"нужно ровно 4 изображения, найдено {len(paths)}: {paths}")
    return np.stack([load_image(p, n) for p in paths])


def save_doe(phi: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(phase_to_uint8(phi)).save(path)


def load_doe(path) -> np.ndarray:
    return uint8_to_phase(np.asarray(Image.open(path).convert("L")))


def save_gray(img_u8: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(img_u8, dtype=np.uint8)).save(path)


def save_result(out_dir, phi, recons_u8, metrics: dict, extra: dict | None = None) -> None:
    """Всё для сдачи: doe.png, recon_1..4.png, metrics.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_doe(phi, out / "doe.png")
    for i, r in enumerate(recons_u8, start=1):
        save_gray(r, out / f"recon_{i}.png")
    payload = dict(metrics)
    if extra:
        payload.update(extra)
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
