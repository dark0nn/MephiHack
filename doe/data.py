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
    """Все картинки из папки (рекурсивно), приведённые к n×n серым.

    Понимает два формата:
      * отдельные файлы картинок (png, jpg, ...);
      * массивы .npy формы (K, h, w) uint8 — целые датасеты одним файлом
        (MNIST, Fashion-MNIST: см. scripts/get_datasets.py). Открываются через mmap,
        поэтому в память не грузятся и работают с несколькими workers.
    Вероятность выбрать источник пропорциональна числу картинок в нём.
    """

    def __init__(self, root, n: int = 128, max_files: int | None = None):
        root = Path(root)
        self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)[:max_files]
        self.arrays = []                                  # (путь, число картинок); сам массив — лениво
        for p in sorted(root.rglob("*.npy")):
            arr = np.load(p, mmap_mode="r")
            if arr.ndim == 3 and len(arr) > 0:
                self.arrays.append((p, len(arr)))
        self._mm = {}
        self.n = n

    def __getstate__(self):
        # в процессы DataLoader передаём только пути: mmap откроется в каждом заново,
        # а не будет копироваться целиком (на Mac workers запускаются через spawn)
        st = self.__dict__.copy()
        st["_mm"] = {}
        return st

    def _array(self, path):
        if path not in self._mm:
            self._mm[path] = np.load(path, mmap_mode="r")
        return self._mm[path]

    def __len__(self):
        return len(self.paths) + sum(k for _, k in self.arrays)

    def describe(self) -> str:
        parts = [f"{p.stem}: {k:,}" for p, k in self.arrays]
        if self.paths:
            parts.append(f"файлы: {len(self.paths):,}")
        return ", ".join(parts)

    def _raw(self, rng: np.random.Generator) -> Image.Image:
        i = int(rng.integers(len(self)))
        if i < len(self.paths):
            return Image.open(self.paths[i]).convert("L")
        i -= len(self.paths)
        for path, k in self.arrays:
            if i < k:
                return Image.fromarray(np.ascontiguousarray(self._array(path)[i], dtype=np.uint8))
            i -= k
        raise IndexError(i)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        img = self._raw(rng).resize((self.n, self.n), Image.BILINEAR)
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
