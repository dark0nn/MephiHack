"""Итеративные методы — бейзлайны для сравнения с нейросетью.

  * gs_parallel   — многоплоскостной GS: обратные поля всех плоскостей суммируются;
  * gs_sequential — GS «по очереди»: плоскость за плоскостью (обычно хуже, «помнит» последнюю);
  * adam_phase    — градиентный спуск по фазе через дифференцируемую модель (Adam).

Все работают с батчем: targets (B, P, N, N) или (P, N, N) в [0, 1] -> фаза (B, N, N).
"""
import math
import time

import torch

from .config import Config, DEFAULT
from .losses import HoloLoss
from .optics import FresnelPropagator, target_amplitude, wrap


def _prep(targets, prop):
    t = targets if targets.dim() == 4 else targets[None]
    return t.to(prop.chirp_re.device, torch.float32)


def random_phase(shape, device, generator=None):
    return (torch.rand(shape, device=device, generator=generator) * 2 - 1) * math.pi


@torch.no_grad()
def gs_parallel(targets, iters=100, cfg: Config = DEFAULT, prop=None, init=None, seed=0):
    prop = prop if prop is not None else FresnelPropagator(cfg)
    t = _prep(targets, prop)
    amp = target_amplitude(t, cfg)
    g = torch.Generator(device="cpu").manual_seed(seed)
    phi = init if init is not None else random_phase(t.shape[:1] + t.shape[-2:], "cpu", g).to(t.device)
    for _ in range(iters):
        K = prop(phi)
        K = torch.polar(amp, K.angle())
        phi = prop.backward_field(K).sum(1).angle()
    return phi


@torch.no_grad()
def gs_sequential(targets, iters=100, cfg: Config = DEFAULT, prop=None, init=None, seed=0):
    prop = prop if prop is not None else FresnelPropagator(cfg)
    t = _prep(targets, prop)
    amp = target_amplitude(t, cfg)
    g = torch.Generator(device="cpu").manual_seed(seed)
    phi = init if init is not None else random_phase(t.shape[:1] + t.shape[-2:], "cpu", g).to(t.device)
    P = t.shape[1]
    for _ in range(iters):
        for p in range(P):
            K = prop(phi)[:, p:p + 1]
            K = torch.polar(amp[:, p:p + 1], K.angle())
            # обратное распространение только для плоскости p
            G = torch.fft.ifft2(torch.fft.ifftshift(K, dim=(-2, -1)))
            G = torch.fft.fftshift(G, dim=(-2, -1)) * cfg.n * prop.chirp[p:p + 1].conj()
            phi = G[:, 0].angle()
    return phi


def adam_phase(targets, iters=300, lr=0.05, cfg: Config = DEFAULT, prop=None, init=None, seed=0, loss_fn=None):
    """Прямая оптимизация фазы. Это и бейзлайн, и «дотюнивание» после сети."""
    prop = prop if prop is not None else FresnelPropagator(cfg)
    t = _prep(targets, prop)
    loss_fn = loss_fn if loss_fn is not None else HoloLoss(cfg)
    g = torch.Generator(device="cpu").manual_seed(seed)
    phi0 = init.detach().clone() if init is not None else random_phase(t.shape[:1] + t.shape[-2:], "cpu", g).to(t.device)
    phi = phi0.requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=lr)
    for _ in range(iters):
        loss, _ = loss_fn(prop.recon_image(phi), t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return wrap(phi.detach())


METHODS = {"gs_parallel": gs_parallel, "gs_sequential": gs_sequential, "adam": adam_phase}


def run(method: str, targets, iters: int, cfg: Config = DEFAULT, device="cpu", **kw):
    """Запустить метод и замерить время. Возвращает (фаза (N, N) на cpu, секунды)."""
    prop = FresnelPropagator(cfg).to(device)
    t0 = time.perf_counter()
    phi = METHODS[method](targets, iters=iters, cfg=cfg, prop=prop, **kw)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return phi[0].cpu(), time.perf_counter() - t0
