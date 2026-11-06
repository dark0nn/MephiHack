"""Финальная оценка «как у жюри» — общая для всех методов.

Берёт фазу ДОЭ, квантует её в 8 бит (как при сохранении PNG), заново считает
восстановление независимой numpy-моделью, переводит в 8 бит и считает метрики.
Так любой метод (GS, Adam, нейросеть) оценивается одинаково.
"""
from __future__ import annotations

import numpy as np

from . import optics_np
from .config import Config, DEFAULT
from .io import phase_to_uint8, uint8_to_phase, save_result
from .metrics import to_uint8, evaluate


def finalize(phi: np.ndarray, targets: np.ndarray, cfg: Config = DEFAULT):
    """phi (N, N) радианы, targets (4, N, N) в [0, 1] -> (фаза после 8 бит, восстановления uint8, метрики)."""
    phi_q = uint8_to_phase(phase_to_uint8(phi))
    amp = optics_np.reconstruct(phi_q, cfg)
    img = amp if cfg.recon_quantity == "amplitude" else amp ** 2
    recons_u8 = np.stack([to_uint8(r, cfg.recon_norm) for r in img])
    targets_u8 = np.round(np.clip(targets, 0, 1) * 255).astype(np.uint8)
    return phi_q, recons_u8, evaluate(recons_u8, targets_u8)


def finalize_and_save(phi, targets, out_dir, cfg: Config = DEFAULT, extra: dict | None = None) -> dict:
    phi_q, recons_u8, result = finalize(phi, targets, cfg)
    save_result(out_dir, phi_q, recons_u8, result, extra)
    return result
