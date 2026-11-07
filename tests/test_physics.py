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


def test_holonet_unrolled_gs_matches_baseline_and_has_finite_grads():
    torch = pytest.importorskip("torch")
    from doe.baselines import gs_parallel
    from doe.losses import HoloLoss
    from doe.model import HoloNet, UNet
    from doe.optics import FresnelPropagator, target_amplitude

    torch.manual_seed(0)
    prop = FresnelPropagator()
    t = (torch.rand(2, 4, 128, 128) > 0.7).float()          # много нулей — проверка на NaN
    amp = target_amplitude(t)
    net = HoloNet(UNet(base=8, depth=2), prop, gs_iters=3)

    # развёрнутый GS внутри модели совпадает с бейзлайном GS от той же стартовой фазы
    with torch.no_grad():
        phi0 = net.unet(amp)
        ref = gs_parallel(t, iters=3, prop=prop, init=phi0)
        got = net(amp)
    d = torch.remainder(got - ref + math.pi, 2 * math.pi) - math.pi
    assert d.abs().max() < 1e-3

    loss, s = HoloLoss(w_worst=1.0)(prop.recon_image(net(amp)), t)
    loss.backward()
    assert torch.isfinite(loss) and s["worst"] <= s["pcc"] + 1e-6
    assert all(torch.isfinite(p.grad).all() for p in net.unet.parameters() if p.grad is not None)


def test_physics_input_starts_from_gs_and_trains():
    torch = pytest.importorskip("torch")
    from doe.baselines import gs_parallel
    from doe.losses import HoloLoss
    from doe.model import build_model
    from doe.optics import FresnelPropagator, target_amplitude

    torch.manual_seed(0)
    prop = FresnelPropagator()
    t = (torch.rand(2, 4, 128, 128) > 0.7).float()
    amp = target_amplitude(t)
    net = build_model(DEFAULT, prop, base=8, depth=2, net_input="physics", pre_gs=2, head_std=0.0)
    assert net.unet.downs[0].net[0].in_channels == 2 * DEFAULT.planes + 2

    # необученная сеть (поправка = 0) даёт ровно pre_gs шагов GS от фиксированной стартовой фазы:
    # 1-й шаг — обратное распространение целей, 2-й — обычная итерация GS
    with torch.no_grad():
        K0 = torch.polar(amp, net.init_phase.expand_as(amp))
        phi1 = prop.backward_field(K0).sum(1).angle()
        ref = gs_parallel(t, iters=1, prop=prop, init=phi1)
        got = net(amp)
    d = torch.remainder(got - ref + math.pi, 2 * math.pi) - math.pi
    # расхождения ~1e-3 рад только в пикселях с почти нулевой амплитудой (фаза там неустойчива)
    assert d.abs().max() < 1e-2 and d.abs().mean() < 1e-4

    loss, _ = HoloLoss()(prop.recon_image(net(amp)), t)
    loss.backward()
    assert torch.isfinite(loss)
    assert net.unet.head.weight.grad is not None and net.unet.head.weight.grad.abs().sum() > 0

    # с обычной (маленькой, но ненулевой) головой градиент доходит до первого слоя сети
    net2 = build_model(DEFAULT, prop, base=8, depth=2, net_input="physics")
    loss2, _ = HoloLoss()(prop.recon_image(net2(amp)), t)
    loss2.backward()
    assert net2.unet.downs[0].net[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in net.unet.parameters() if p.grad is not None)


def test_folder_images_reads_npy_datasets(tmp_path):
    pytest.importorskip("torch")
    import pickle
    from doe.data import FolderImages, TargetSets

    np.save(tmp_path / "mnist.npy", (RNG.random((7, 28, 28)) * 60).astype(np.uint8))      # тёмный фон
    np.save(tmp_path / "bright.npy", (200 + RNG.random((3, 28, 28)) * 55).astype(np.uint8))  # светлый фон
    np.save(tmp_path / "labels.npy", np.arange(10))                                         # не картинки — пропуск
    fi = FolderImages(tmp_path, n=DEFAULT.n)
    assert len(fi) == 10 and "mnist: 7" in fi.describe()
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = fi.sample(rng)
        assert a.shape == (DEFAULT.n, DEFAULT.n) and a.dtype == np.float32
        assert 0 <= a.min() and a.max() <= 1 and a.mean() <= 0.5    # светлый фон инвертирован
    fi2 = pickle.loads(pickle.dumps(fi))                             # как в workers DataLoader
    assert fi2.sample(rng).shape == (DEFAULT.n, DEFAULT.n)
    ds = TargetSets(DEFAULT.n, DEFAULT.planes, folder=tmp_path)
    assert next(iter(ds)).shape == (DEFAULT.planes, DEFAULT.n, DEFAULT.n)


def test_target_phase_mode_shapes_and_grads():
    torch = pytest.importorskip("torch")
    from doe.losses import HoloLoss
    from doe.model import build_model
    from doe.optics import FresnelPropagator, target_amplitude

    torch.manual_seed(0)
    prop = FresnelPropagator()
    t = (torch.rand(2, 4, 128, 128) > 0.7).float()
    amp = target_amplitude(t)
    net = build_model(DEFAULT, prop, base=8, depth=2, net_input="target_phase", gs_iters=2)
    assert net.unet(amp).shape == (2, DEFAULT.planes, DEFAULT.n, DEFAULT.n)
    phi = net(amp)
    assert phi.shape == (2, DEFAULT.n, DEFAULT.n)
    loss, _ = HoloLoss()(prop.recon_image(phi), t)
    loss.backward()
    g = net.unet.downs[0].net[0].weight.grad
    assert torch.isfinite(loss) and g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_windowed_ssim_matches_direct_formula():
    """Оконный SSIM через кумулятивные суммы = прямой подсчёт по каждому окну 7×7 (как в skimage)."""
    from doe.metrics import ssim_windowed_np

    def direct(a, b, L=255.0, w=7):
        a = a.astype(float); b = b.astype(float)
        c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
        s = []
        for i in range(a.shape[0] - w + 1):
            for j in range(a.shape[1] - w + 1):
                x, y = a[i:i + w, j:j + w], b[i:i + w, j:j + w]
                cov = ((x - x.mean()) * (y - y.mean())).sum() / (w * w - 1)
                s.append((2 * x.mean() * y.mean() + c1) * (2 * cov + c2)
                         / ((x.mean() ** 2 + y.mean() ** 2 + c1) * (x.var(ddof=1) + y.var(ddof=1) + c2)))
        return float(np.mean(s))

    a = RNG.integers(0, 256, (24, 21)).astype(np.uint8)
    b = np.clip(a + RNG.normal(0, 40, a.shape), 0, 255).astype(np.uint8)
    assert ssim_windowed_np(a, b) == pytest.approx(direct(a, b), abs=1e-10)
    assert ssim_windowed_np(a, a) == pytest.approx(1.0)
