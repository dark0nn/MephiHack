"""Проверка окружения: версии, устройство, быстрый тест физики.

    python scripts/check_env.py
"""
import platform

import _path  # noqa: F401

import numpy as np


def main():
    print(f"Python {platform.python_version()} на {platform.system()} {platform.machine()}")
    try:
        import torch
    except ImportError:
        print("PyTorch НЕ установлен: выполните python -m pip install -r requirements.txt (см. README)")
        return
    from doe import DEFAULT, optics_np
    from doe.baselines import run
    from doe.optics import FresnelPropagator, get_device

    dev = get_device()
    print(f"PyTorch {torch.__version__}; устройство: {dev}")
    if dev.type == "cpu":
        print("  (видеокарты нет — для обучения сети лучше Google Colab с GPU)")

    phi = np.random.default_rng(0).uniform(-np.pi, np.pi, (DEFAULT.n, DEFAULT.n))
    ref = np.abs(optics_np.propagate(phi))
    got = FresnelPropagator().to(dev)(torch.tensor(phi, dtype=torch.float32, device=dev))[0].abs().cpu().numpy()
    err = np.abs(ref - got).max()
    print(f"Модель Френеля: torch и numpy совпадают, макс. расхождение {err:.1e}", "OK" if err < 2e-4 else "ОШИБКА")

    y, x = np.mgrid[:128, :128]
    t = np.stack([((x - 64) ** 2 + (y - 64) ** 2 < r * r).astype(np.float32) for r in (30, 24, 18, 12)])
    _, sec = run("gs_parallel", torch.from_numpy(t), 50, device=dev)
    print(f"GS, 50 итераций: {sec:.2f} с")
    print("Окружение готово.")


if __name__ == "__main__":
    main()
