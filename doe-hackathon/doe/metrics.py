"""Метрики из ТЗ: коэффициент корреляции Пирсона (КК) и индекс структурного сходства (SSIM).

Две группы функций:
  * torch-версии (pearson, ssim_global) — дифференцируемые, для лосса;
  * evaluate() — «как у жюри»: на 8-битных картинках, numpy.

ВНИМАНИЕ: SSIM на слайде записан глобально (по всему кадру). skimage считает его
в скользящем окне — числа будут другими. evaluate() выдаёт оба варианта,
пока организаторы не скажут, какой используют.
"""
from __future__ import annotations

import numpy as np

try:  # numpy-часть работает и без PyTorch
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .config import Config, DEFAULT

_C1 = 0.01 ** 2  # для данных в диапазоне [0, 1]
_C2 = 0.03 ** 2


# ---------- torch, для обучения ----------

def pearson(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """КК по последним двум осям. Не зависит от масштаба и сдвига яркости."""
    a = a - a.mean((-2, -1), keepdim=True)
    b = b - b.mean((-2, -1), keepdim=True)
    num = (a * b).sum((-2, -1))
    den = torch.sqrt((a * a).sum((-2, -1)) * (b * b).sum((-2, -1)) + eps)
    return num / den


def ssim_global(a: torch.Tensor, b: torch.Tensor, c1: float = _C1, c2: float = _C2) -> torch.Tensor:
    """SSIM по формуле со слайда, по всему кадру. Ожидает данные в [0, 1]."""
    ma, mb = a.mean((-2, -1)), b.mean((-2, -1))
    va = a.var((-2, -1), unbiased=False)
    vb = b.var((-2, -1), unbiased=False)
    cov = ((a - ma[..., None, None]) * (b - mb[..., None, None])).mean((-2, -1))
    return (2 * ma * mb + c1) * (2 * cov + c2) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))


def normalize(x: torch.Tensor, mode: str = "max", eps: float = 1e-8) -> torch.Tensor:
    """Привести восстановление к [0, 1] так же, как при сохранении в PNG."""
    if mode == "max":
        return x / (x.amax((-2, -1), keepdim=True) + eps)
    if mode == "p99":
        q = torch.quantile(x.flatten(-2), 0.99, dim=-1)[..., None, None]
        return (x / (q + eps)).clamp(0, 1)
    raise ValueError(f"unknown norm mode: {mode}")


# ---------- numpy, «как у жюри» ----------

def to_uint8(x: np.ndarray, mode: str = "max") -> np.ndarray:
    """Восстановление (H, W) -> 8-битная картинка."""
    x = np.asarray(x, dtype=np.float64)
    if mode == "max":
        x = x / (x.max() + 1e-12)
    elif mode == "p99":
        x = np.clip(x / (np.percentile(x, 99) + 1e-12), 0, 1)
    else:
        raise ValueError(mode)
    return np.round(x * 255).astype(np.uint8)


def pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def ssim_global_np(a: np.ndarray, b: np.ndarray, L: float = 255.0) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return float((2 * ma * mb + c1) * (2 * cov + c2) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))


def evaluate(recons_u8, targets_u8) -> dict:
    """Попарные метрики для 4 пар «исходное — восстановленное» (оба uint8)."""
    from skimage.metrics import structural_similarity

    rows = []
    for i, (r, t) in enumerate(zip(recons_u8, targets_u8), start=1):
        rows.append({
            "plane": i,
            "pearson": pearson_np(t, r),
            "ssim_global": ssim_global_np(t, r),
            "ssim_windowed": float(structural_similarity(t, r, data_range=255)),
        })
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("pearson", "ssim_global", "ssim_windowed")}
    return {"planes": rows, "mean": mean}


def format_table(result: dict) -> str:
    lines = ["плоскость  КК      SSIM(глоб.)  SSIM(окно)"]
    for r in result["planes"]:
        lines.append(f"{r['plane']:^9}  {r['pearson']:.4f}  {r['ssim_global']:.4f}       {r['ssim_windowed']:.4f}")
    m = result["mean"]
    lines.append(f"{'среднее':^9}  {m['pearson']:.4f}  {m['ssim_global']:.4f}       {m['ssim_windowed']:.4f}")
    return "\n".join(lines)
