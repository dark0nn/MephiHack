"""Синтетические картинки-цели: фигуры, линии, буквы и цифры. Без PyTorch."""
from __future__ import annotations

import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

_CHARS = string.ascii_uppercase + string.digits + "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЭЮЯ"


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size) if hasattr(ImageFont, "load_default") else ImageFont.load_default()


def synth_image(rng: np.random.Generator, n: int = 128) -> np.ndarray:
    """Одна синтетическая картинка (n, n) в [0, 1]."""
    img = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(img)
    kind = rng.choice(["shapes", "text", "lines", "mixed"], p=[0.35, 0.3, 0.15, 0.2])

    def rnd_val():
        return int(rng.choice([255, 255, 255, rng.integers(80, 256)]))  # чаще бинарные, иногда серые

    def shape():
        x0, y0 = rng.integers(8, n - 40, 2)
        w, h = rng.integers(16, n // 2, 2)
        box = [int(x0), int(y0), int(min(x0 + w, n - 4)), int(min(y0 + h, n - 4))]
        t = rng.integers(0, 4)
        if t == 0:
            d.ellipse(box, fill=rnd_val())
        elif t == 1:
            d.rectangle(box, fill=rnd_val())
        elif t == 2:
            pts = [tuple(int(v) for v in rng.integers(8, n - 8, 2)) for _ in range(3)]
            d.polygon(pts, fill=rnd_val())
        else:
            d.rectangle(box, outline=rnd_val(), width=int(rng.integers(3, 10)))

    def line():
        p = [int(v) for v in rng.integers(4, n - 4, 4)]
        d.line(p, fill=rnd_val(), width=int(rng.integers(3, 12)))

    def text():
        k = int(rng.integers(1, 4))
        s = "".join(rng.choice(list(_CHARS), k))
        size = int(rng.integers(n // 4, int(n / (0.6 * k + 0.4))))
        f = _font(size)
        x, y = rng.integers(0, n // 4, 2)
        d.text((int(x), int(y)), s, fill=rnd_val(), font=f)

    if kind == "shapes":
        for _ in range(rng.integers(1, 4)):
            shape()
    elif kind == "text":
        text()
    elif kind == "lines":
        for _ in range(rng.integers(1, 5)):
            line()
    else:
        shape(); text()
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 2.0))))
    return np.asarray(img, dtype=np.float32) / 255.0
