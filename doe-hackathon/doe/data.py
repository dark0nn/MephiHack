"""Датасет для обучения без учителя.

Ответы GS не нужны: каждый пример — просто 4 случайные картинки-цели.
Источники (смешиваются):
  * синтетика: фигуры, линии, буквы и цифры — генерируется на лету;
  * своя папка с картинками (MNIST, буквы, фото): data/images/** — опционально.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .io import IMG_EXT
from .synth import synth_image


class FolderImages:
    """Все картинки из папки (рекурсивно), приведённые к n×n серым."""

    def __init__(self, root, n: int = 128, max_files: int | None = None):
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)[:max_files]
        self.n = n

    def __len__(self):
        return len(self.paths)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        p = self.paths[int(rng.integers(len(self.paths)))]
        img = Image.open(p).convert("L").resize((self.n, self.n), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
        if a.mean() > 0.5:  # тёмный фон, светлый объект — как у голограмм
            a = 1.0 - a
        return a


class TargetSets(torch.utils.data.IterableDataset):
    """Бесконечный поток примеров (P, n, n): P независимых картинок для P плоскостей."""

    def __init__(self, n: int = 128, planes: int = 4, folder=None, p_folder: float = 0.5, seed: int = 0):
        super().__init__()
        self.n, self.planes, self.p_folder, self.seed = n, planes, p_folder, seed
        self.folder = FolderImages(folder, n) if folder and Path(folder).exists() else None
        if self.folder is not None and len(self.folder) == 0:
            self.folder = None

    def one(self, rng):
        if self.folder is not None and rng.random() < self.p_folder:
            return self.folder.sample(rng)
        return synth_image(rng, self.n)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed + 1000 * wid + int(torch.randint(0, 10**6, (1,)).item()))
        while True:
            yield torch.from_numpy(np.stack([self.one(rng) for _ in range(self.planes)]))


def fixed_set(count: int, n: int = 128, planes: int = 4, seed: int = 12345) -> torch.Tensor:
    """Детерминированный набор для валидации: (count, P, n, n)."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(np.stack([np.stack([synth_image(rng, n) for _ in range(planes)]) for _ in range(count)]))
