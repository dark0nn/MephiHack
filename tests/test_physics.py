"""Проверки физики и формата. Запуск: python -m pytest -q

numpy-тесты работают без PyTorch; torch-тесты сверяют torch-модель с numpy-эталоном.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doe import DEFAULT  # noqa: E402
from doe import optics_np  # noqa: E402
from doe.io import phase_to_uint8, uint8_to_phase  # noqa: E402
from doe.metrics import pearson_np, ssim_global_np  # noqa: E402

RNG = np.random.default_rng(0)


def disk_targets():
    n = DEFAULT.n
    y, x = np.mgrid[:n, :n]
    t = np.zeros((4, n, n), np.float32)
    for i, r in enumerate((30, 24, 18, 12)):
        t[i] = ((x - 64) ** 2 + (y - 64) ** 2 < r * r)
    return t


def test_critical_distance():
    assert abs(DEFAULT.critical_distance - 24.06e-3) < 0.01e-3
    assert abs(DEFAULT.out_pixel(24.1e-3) - 10.02e-6) < 0.01e-6


def test_energy_conserved():
    phi = RNG.uniform(-math.pi, math.pi, (DEFAULT.n, DEFAULT.n))
    K = optics_np.propagate(phi)
    assert np.allclose((np.abs(K) ** 2).sum((-2, -1)), DEFAULT.n ** 2, rtol=1e-6)


def test_back_is_inverse():
    phi = RNG.uniform(-math.pi, math.pi, (DEFAULT.n, DEFAULT.n))
    H = optics_np.backpropagate(optics_np.propagate(phi))
    assert np.allclose(H, np.exp(1j * phi)[None], atol=1e-8)


def test_phase_png_roundtrip():
    phi = RNG.uniform(-math.pi, math.pi, (16, 16))
    q = phase_to_uint8(phi)
    assert q.dtype == np.uint8 and q.min() >= 0 and q.max() <= 255
    assert phase_to_uint8(np.array([-math.pi]))[0] == 0
    assert phase_to_uint8(np.array([math.pi - 1e-9]))[0] == 255
    err = np.angle(np.exp(1j * (uint8_to_phase(q) - phi)))
    assert np.abs(err).max() <= math.pi / 255 + 1e-9


def test_metrics_identity():
    a = RNG.integers(0, 256, (32, 32)).astype(np.uint8)
    assert pearson_np(a, a) == pytest.approx(1.0)
    assert ssim_global_np(a, a) == pytest.approx(1.0)


def test_gs_reaches_reasonable_quality():
    t = disk_targets()
    phi = optics_np.gs_parallel(t, iters=30)
    amp = optics_np.reconstruct(phi)
    pcc = np.mean([pearson_np(a, b) for a, b in zip(amp, t)])
    assert pcc > 0.6, pcc


# ---------- torch ----------

def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    from doe.optics import FresnelPropagator

    phi = RNG.uniform(-math.pi, math.pi, (DEFAULT.n, DEFAULT.n))
    K_np = optics_np.propagate(phi)
    K_t = FresnelPropagator()(torch.tensor(phi, dtype=torch.float32))[0].numpy()
    assert np.allclose(np.abs(K_t), np.abs(K_np), atol=2e-4)


def test_torch_metrics_match_numpy():
    torch = pytest.importorskip("torch")
    from doe.metrics import pearson, ssim_global

    a = RNG.random((32, 32)); b = RNG.random((32, 32))
    assert float(pearson(torch.tensor(a), torch.tensor(b))) == pytest.approx(pearson_np(a, b), abs=1e-6)
    assert float(ssim_global(torch.tensor(a), torch.tensor(b))) == pytest.approx(ssim_global_np(a, b, L=1.0), abs=1e-6)


def test_unet_shapes_and_grad():
    torch = pytest.importorskip("torch")
    from doe.losses import HoloLoss
    from doe.model import UNet
    from doe.optics import FresnelPropagator

    model = UNet(base=8, depth=3)
    t = torch.rand(2, 4, DEFAULT.n, DEFAULT.n)
    phi = model(t)
    assert phi.shape == (2, DEFAULT.n, DEFAULT.n)
    loss, _ = HoloLoss()(FresnelPropagator().recon_image(phi), t)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_baselines_run():
    torch = pytest.importorskip("torch")
    from doe.baselines import METHODS

    t = torch.from_numpy(disk_targets())
    for name, fn in METHODS.items():
        phi = fn(t, iters=3)
        assert phi.shape == (1, DEFAULT.n, DEFAULT.n), name
