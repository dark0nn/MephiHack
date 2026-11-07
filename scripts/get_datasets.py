"""Скачать MNIST и Fashion-MNIST в data/images/ (по одному .npy на датасет).

    python scripts/get_datasets.py              # оба датасета
    python scripts/get_datasets.py mnist        # только один

Результат: data/images/mnist.npy и data/images/fashion_mnist.npy — массивы (70000, 28, 28) uint8
(train + test вместе: нам нужны только картинки, метки не используются).
После этого train.py сам подмешивает их в обучение (50% примеров — из папки, 50% — синтетика).
Нужен только numpy, без torchvision. Папка data/images/ не попадает в git и в Docker-образ.
"""
import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "images"

FILES = ["train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"]
SOURCES = {
    "mnist": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
    ],
    "fashion_mnist": [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    ],
}


def download(name: str, fname: str) -> bytes:
    last = None
    for base in SOURCES[name]:
        url = base + fname
        try:
            print(f"  {url}")
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # пробуем следующее зеркало
            print(f"    не вышло: {e}")
            last = e
    raise RuntimeError(f"Не удалось скачать {fname} ни с одного зеркала: {last}")


def parse_idx_images(raw: bytes) -> np.ndarray:
    data = gzip.decompress(raw)
    magic, count, h, w = np.frombuffer(data[:16], dtype=">u4")
    assert magic == 2051, f"неверный формат idx (magic {magic})"
    return np.frombuffer(data[16:], dtype=np.uint8).reshape(count, h, w)


def main():
    names = sys.argv[1:] or list(SOURCES)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in SOURCES:
            raise SystemExit(f"Неизвестный датасет {name}. Доступны: {', '.join(SOURCES)}")
        target = OUT / f"{name}.npy"
        if target.exists():
            print(f"{name}: уже есть ({target.relative_to(ROOT)}), пропускаю")
            continue
        print(f"{name}: скачиваю")
        imgs = np.concatenate([parse_idx_images(download(name, f)) for f in FILES])
        np.save(target, imgs)
        print(f"{name}: {imgs.shape[0]:,} картинок {imgs.shape[1]}×{imgs.shape[2]} -> {target.relative_to(ROOT)}")
    print("Готово. train.py подхватит их автоматически из data/images/.")


if __name__ == "__main__":
    main()
