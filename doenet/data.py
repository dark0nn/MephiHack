"""Данные: поток для обучения и две фиксированные валидации.

Обучение — doe.data.TargetSets: синтетика на лету + MNIST/Fashion из data/images (50/50),
тёмный фон, светлый объект (светлый фон инвертируется в FolderImages).
Валидации (фиксированные, детерминированные):
  * "syn"   — только синтетика (фигуры, линии, буквы, цифры);
  * "mnist" — только картинки из data/images (MNIST + Fashion-MNIST).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from doe.data import FolderImages, TargetSets, fixed_set


def train_stream(n: int, planes: int, images: str | None, seed: int = 0) -> TargetSets:
    return TargetSets(n, planes, folder=images, p_folder=0.5, seed=seed)


def val_sets(n: int, planes: int, count: int, images: str | None, seed: int = 12345) -> dict:
    sets = {"syn": fixed_set(count, n, planes, seed=seed)}
    if images and Path(images).exists():
        fi = FolderImages(images, n)
        if len(fi) > 0:
            rng = np.random.default_rng(seed + 1)
            sets["mnist"] = torch.from_numpy(
                np.stack([np.stack([fi.sample(rng) for _ in range(planes)]) for _ in range(count)]))
    return sets
