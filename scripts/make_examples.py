"""Создаёт тестовые наборы из 4 картинок в data/examples/ (формат как у организаторов).

    python scripts/make_examples.py
"""
import _path  # noqa: F401
from _path import ROOT

import numpy as np
from PIL import Image, ImageDraw

from doe.synth import synth_image, _font

N = 128


def save_set(name, imgs):
    out = ROOT / "data" / "examples" / name
    out.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(imgs, start=1):
        Image.fromarray(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)).save(out / f"{i}.png")
    print("  ", out.relative_to(ROOT))


def shapes():
    imgs = []
    for draw in (
        lambda d: d.ellipse([34, 34, 94, 94], fill=255),
        lambda d: (d.rectangle([30, 30, 98, 98], fill=255), d.rectangle([50, 50, 78, 78], fill=0)),
        lambda d: d.polygon([(64, 20), (20, 108), (108, 108)], fill=255),
        lambda d: (d.rectangle([58, 20, 70, 108], fill=255), d.rectangle([20, 58, 108, 70], fill=255)),
    ):
        im = Image.new("L", (N, N), 0)
        draw(ImageDraw.Draw(im))
        imgs.append(np.asarray(im, np.float32) / 255)
    return imgs


def letters(text="ДОЭ1"):
    imgs = []
    f = _font(84)
    for ch in text:
        im = Image.new("L", (N, N), 0)
        d = ImageDraw.Draw(im)
        l, t, r, b = d.textbbox((0, 0), ch, font=f)
        d.text(((N - (r - l)) / 2 - l, (N - (b - t)) / 2 - t), ch, fill=255, font=f)
        imgs.append(np.asarray(im, np.float32) / 255)
    return imgs


if __name__ == "__main__":
    print("Создаю примеры:")
    save_set("shapes", shapes())
    save_set("letters", letters())
    rng = np.random.default_rng(7)
    save_set("random", [synth_image(rng, N) for _ in range(4)])
