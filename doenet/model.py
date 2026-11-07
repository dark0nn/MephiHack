"""Модель «фаза в плоскостях изображения» (идея HoloNet, Peng et al. 2020).

    амплитуды целей A (B, P, N, N)
      -> U-Net -> P фаз ψ_p в плоскостях изображения (пары cos/sin -> atan2, без разрыва на ±π)
      -> K_p = A_p·exp(iψ_p) -> обратное распространение на ДОЭ -> сумма по p -> H
      -> [K итераций дифференцируемого GS]
      -> фаза ДОЭ φ = arg(H)

Нормировки z/(|z|+eps) вместо angle(): без NaN в градиентах на нулевой амплитуде.
Последний слой НЕ инициализируется нулём (через нулевые веса градиент не доходит до сети).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class Block(nn.Module):
    """conv3×3 (циклическое дополнение: поле периодично по БПФ) -> GroupNorm -> SiLU, дважды.
    GroupNorm, а не BatchNorm: при batch 4 статистика батча шумная, а в eval другая."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(_groups(cout), cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(_groups(cout), cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base: int = 32, depth: int = 4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.downs.append(Block(c, ch))
            c = ch
        self.mid = Block(chs[-2], chs[-1])
        self.ups = nn.ModuleList(Block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chs[0], out_ch, 1)          # обычная (ненулевая) инициализация

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.mid(x)
        for up, s in zip(self.ups, reversed(skips)):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = up(torch.cat([x, s], 1))
        return self.head(x)


class PlanePhaseNet(nn.Module):
    """U-Net в плоскостях изображения + физика внутри модели.

    prop — doe.optics.FresnelPropagator (эталонная физика, без параметров).
    gs_iters — сколько итераций дифференцируемого GS выполнить после сети.
    in_norm — нормировка входа сети: "raw" (амплитуда как есть, [0, 1]) или "std"
              (каждая картинка к нулевому среднему и единичному СКО). Физика всегда
              использует исходную амплитуду.
    psi_offset — "none" или "random": к ψ сети прибавляется фиксированная случайная фаза
              (диффузор). Свёрточная сеть выдаёт гладкие ψ, а гладкая фаза в плоскости изображения
              даёт плохой ДОЭ; со случайной добавкой сеть учит поправку к хорошему старту.
              Добавка задаётся seed и не хранится в чекпоинте (восстанавливается одинаково).
    """

    def __init__(self, prop: nn.Module, planes: int = 4, base: int = 32, depth: int = 4,
                 gs_iters: int = 0, in_norm: str = "raw", eps: float = 1e-4,
                 psi_offset: str = "none", offset_seed: int = 0):
        super().__init__()
        assert in_norm in ("raw", "std"), in_norm
        assert psi_offset in ("none", "random"), psi_offset
        self.prop, self.P, self.gs_iters, self.in_norm, self.eps = prop, planes, gs_iters, in_norm, eps
        self.unet = UNet(planes, 2 * planes, base, depth)
        n = prop.chirp_re.shape[-1]
        off = torch.zeros(planes, n, n)
        if psi_offset == "random":
            g = torch.Generator().manual_seed(offset_seed)
            off = (torch.rand(planes, n, n, generator=g) * 2 - 1) * torch.pi
        self.register_buffer("psi_off", off.to(prop.chirp_re.device), persistent=False)

    def _unit(self, re: torch.Tensor, im: torch.Tensor):
        r = torch.sqrt(re * re + im * im) + self.eps
        return re / r, im / r

    def net_input(self, amp: torch.Tensor) -> torch.Tensor:
        if self.in_norm == "std":
            m = amp.mean((-2, -1), keepdim=True)
            s = amp.std((-2, -1), keepdim=True) + 1e-3
            return (amp - m) / s
        return amp

    def plane_phases(self, amp: torch.Tensor) -> torch.Tensor:
        """ψ_p (B, P, N, N) в (-π, π]."""
        cs = self.unet(self.net_input(amp))
        cs = cs.view(cs.shape[0], self.P, 2, *cs.shape[-2:])
        return torch.atan2(cs[:, :, 1], cs[:, :, 0]) + self.psi_off

    def doe_field(self, amp: torch.Tensor, gs_iters: int | None = None):
        """Поле на ДОЭ единичной амплитуды (re, im), каждое (B, N, N)."""
        psi = self.plane_phases(amp)
        K = torch.complex(amp * torch.cos(psi), amp * torch.sin(psi))
        G = self.prop.backward_field(K).sum(1)
        re, im = self._unit(G.real, G.imag)
        for _ in range(self.gs_iters if gs_iters is None else gs_iters):
            Kf = self.prop.forward_field(torch.complex(re, im))
            kr, ki = self._unit(Kf.real, Kf.imag)
            G = self.prop.backward_field(torch.complex(amp * kr, amp * ki)).sum(1)
            re, im = self._unit(G.real, G.imag)
        return re, im

    def forward(self, amp: torch.Tensor, gs_iters: int | None = None) -> torch.Tensor:
        """amp (B, P, N, N) -> фаза ДОЭ (B, N, N)."""
        re, im = self.doe_field(amp, gs_iters)
        return torch.atan2(im, re)

    def recon(self, amp: torch.Tensor, quantity: str = "amplitude") -> tuple[torch.Tensor, torch.Tensor]:
        """Для обучения: (фаза ДОЭ, восстановления |K| или |K|² на P плоскостях).
        Восстановление считается из поля единичной амплитуды, без лишнего atan2 -> exp."""
        re, im = self.doe_field(amp)
        a = self.prop.forward_field(torch.complex(re, im)).abs()
        return torch.atan2(im, re), (a if quantity == "amplitude" else a * a)


def build(prop, cfg, base=32, depth=4, gs_iters=0, in_norm="raw", psi_offset="none",
          offset_seed=0) -> PlanePhaseNet:
    return PlanePhaseNet(prop, cfg.planes, base, depth, gs_iters, in_norm,
                         psi_offset=psi_offset, offset_seed=offset_seed)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
