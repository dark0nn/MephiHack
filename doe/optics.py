"""Дифференцируемая модель распространения света (PyTorch).

Формула со слайда: прямой расчёт дифракции Френеля с одним БПФ

    K(x,y) = exp(jkz)/(jλz) · exp(jπ/(λz)(x²+y²)) · FFT[ H(u,v) · exp(jπ/(λz)(u²+v²)) ]

Для обучения и метрик нужна только |K|, а внешние множители — чистая фаза
и константа, поэтому по умолчанию они отброшены (full=False).
Нормировка 1/N делает БПФ унитарным: энергия на плоскости = энергии на ДОЭ.

Все функции работают с батчем: phi формы (B, N, N) -> поля (B, P, N, N),
где P = 4 плоскости.
"""
import math

import torch

from .config import Config, DEFAULT


def get_device(prefer: str = "auto") -> torch.device:
    """cuda -> mps (Apple Silicon) -> cpu. MPS проверяется на поддержку комплексного БПФ."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            x = torch.polar(torch.ones(4, 4, device="mps"), torch.zeros(4, 4, device="mps"))
            torch.fft.fft2(x).abs().sum().item()
            return torch.device("mps")
        except Exception:  # старые версии torch не умеют complex FFT на MPS
            pass
    return torch.device("cpu")


class FresnelPropagator(torch.nn.Module):
    """Фаза ДОЭ -> поля на P плоскостях. Параметров не имеет, только буферы."""

    def __init__(self, cfg: Config = DEFAULT):
        super().__init__()
        self.cfg = cfg
        n = cfg.n
        u = (torch.arange(n, dtype=torch.float64) - n / 2) * cfg.pixel
        U, V = torch.meshgrid(u, u, indexing="ij")
        r2 = U ** 2 + V ** 2
        z = torch.tensor(cfg.distances, dtype=torch.float64)[:, None, None]
        arg = math.pi / (cfg.wavelength * z) * r2           # (P, N, N), считаем в float64
        arg = torch.remainder(arg, 2 * math.pi).float()    # сворачиваем до точности float32
        self.register_buffer("chirp_re", torch.cos(arg))
        self.register_buffer("chirp_im", torch.sin(arg))

    @property
    def chirp(self) -> torch.Tensor:
        return torch.complex(self.chirp_re, self.chirp_im)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """phi: (B, N, N) или (N, N), радианы -> комплексные поля (B, P, N, N)."""
        if phi.dim() == 2:
            phi = phi[None]
        return self.forward_field(torch.polar(torch.ones_like(phi), phi))

    def forward_field(self, H: torch.Tensor) -> torch.Tensor:
        """Комплексное поле на ДОЭ (B, N, N) -> поля на плоскостях (B, P, N, N)."""
        F = torch.fft.fft2(torch.fft.ifftshift(H[:, None] * self.chirp, dim=(-2, -1)))
        return torch.fft.fftshift(F, dim=(-2, -1)) / self.cfg.n

    def backward_field(self, K: torch.Tensor) -> torch.Tensor:
        """Обратное распространение (B, P, N, N) -> поля на ДОЭ (B, P, N, N). Для GS."""
        G = torch.fft.ifft2(torch.fft.ifftshift(K, dim=(-2, -1)))
        return torch.fft.fftshift(G, dim=(-2, -1)) * self.cfg.n * self.chirp.conj()

    def amplitude(self, phi: torch.Tensor) -> torch.Tensor:
        """|K| на всех плоскостях, (B, P, N, N)."""
        return self.forward(phi).abs()

    def recon_image(self, phi: torch.Tensor) -> torch.Tensor:
        """То, что сравнивается с картинкой: |K| или |K|² (cfg.recon_quantity)."""
        a = self.amplitude(phi)
        return a if self.cfg.recon_quantity == "amplitude" else a ** 2


def target_amplitude(targets: torch.Tensor, cfg: Config = DEFAULT) -> torch.Tensor:
    """Желаемая амплитуда поля из картинки-цели в [0, 1]."""
    t = targets.clamp_min(0)
    return t if cfg.recon_quantity == "amplitude" else t.sqrt()


def wrap(phi: torch.Tensor) -> torch.Tensor:
    """Свернуть фазу в [-π, π)."""
    return torch.remainder(phi + math.pi, 2 * math.pi) - math.pi
