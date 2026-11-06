"""Эталонная модель на numpy.

Делает то же, что doe/optics.py, но без PyTorch. Нужна, чтобы:
  * сверять torch-версию в тестах;
  * быстро проверять идеи без GPU.
"""
import numpy as np

from .config import Config, DEFAULT


def coords(cfg: Config = DEFAULT):
    u = (np.arange(cfg.n) - cfg.n / 2) * cfg.pixel
    return np.meshgrid(u, u, indexing="ij")


def chirps(cfg: Config = DEFAULT) -> np.ndarray:
    """Квадратичные фазы exp(jπ/(λz)·(u²+v²)) для каждой плоскости, форма (P, N, N)."""
    U, V = coords(cfg)
    r2 = U ** 2 + V ** 2
    z = np.asarray(cfg.distances)[:, None, None]
    return np.exp(1j * np.pi / (cfg.wavelength * z) * r2)


def propagate(phi: np.ndarray, cfg: Config = DEFAULT, ch=None) -> np.ndarray:
    """Фаза ДОЭ (N, N) -> комплексные поля на всех плоскостях (P, N, N).

    Формула Френеля с одним БПФ без внешнего фазового множителя
    (он не влияет на |K|). Нормировка 1/N делает преобразование унитарным:
    энергия на каждой плоскости равна энергии на ДОЭ.
    """
    ch = chirps(cfg) if ch is None else ch
    H = np.exp(1j * phi)[None] * ch
    F = np.fft.fft2(np.fft.ifftshift(H, axes=(-2, -1)))
    return np.fft.fftshift(F, axes=(-2, -1)) / cfg.n


def backpropagate(K: np.ndarray, cfg: Config = DEFAULT, ch=None) -> np.ndarray:
    """Обратное распространение (P, N, N) -> поле на ДОЭ для каждой плоскости (P, N, N)."""
    ch = chirps(cfg) if ch is None else ch
    G = np.fft.ifft2(np.fft.ifftshift(K, axes=(-2, -1)))
    return np.fft.fftshift(G, axes=(-2, -1)) * cfg.n * np.conj(ch)


def reconstruct(phi: np.ndarray, cfg: Config = DEFAULT) -> np.ndarray:
    """Амплитуды |K| на всех плоскостях (P, N, N)."""
    return np.abs(propagate(phi, cfg))


def target_amplitude(targets: np.ndarray, cfg: Config = DEFAULT) -> np.ndarray:
    """Желаемая амплитуда поля: картинка сравнивается с |K| или с |K|², см. cfg.recon_quantity."""
    t = np.clip(targets, 0, None)
    return t if cfg.recon_quantity == "amplitude" else np.sqrt(t)


def gs_parallel(targets: np.ndarray, iters: int = 100, cfg: Config = DEFAULT, seed: int = 0) -> np.ndarray:
    """Многоплоскостной GS: обратные поля со всех плоскостей суммируются."""
    rng = np.random.default_rng(seed)
    ch = chirps(cfg)
    amp = target_amplitude(targets, cfg)
    phi = rng.uniform(-np.pi, np.pi, (cfg.n, cfg.n))
    for _ in range(iters):
        K = propagate(phi, cfg, ch)
        K = amp * np.exp(1j * np.angle(K))
        phi = np.angle(backpropagate(K, cfg, ch).sum(0))
    return phi
