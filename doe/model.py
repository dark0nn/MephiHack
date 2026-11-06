"""U-Net: 4 картинки-цели -> фаза ДОЭ.

Выход — два канала (c, s), фаза = atan2(s, c). Так у сети нет «разрыва» на ±π,
в отличие от π·tanh(x). На вход подаётся желаемая амплитуда (sqrt интенсивности,
если сравнение идёт по |K|²).
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular"),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular"),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Классический U-Net. depth=4, base=32 -> около 7,8 млн параметров."""

    def __init__(self, in_ch: int = 4, base: int = 32, depth: int = 4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.downs.append(ConvBlock(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chs[-2], chs[-1])
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.up_convs.append(ConvBlock(ch * 2, ch))
        self.head = nn.Conv2d(chs[0], 2, 1)

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        """targets (B, P, N, N) -> фаза (B, N, N) в (-π, π]."""
        x = targets
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = conv(torch.cat([x, skip], 1))
        cs = self.head(x)
        return torch.atan2(cs[:, 1], cs[:, 0])


class HoloNet(nn.Module):
    """U-Net + K итераций параллельного GS внутри модели («развёрнутый» GS).

    Сеть даёт стартовую фазу, затем K шагов GS её уточняют. Шаги GS дифференцируемы,
    поэтому сеть учится выдавать не готовый ответ, а лучший старт для GS.
    Нормировки z/(|z|+eps) вместо angle() — чтобы не было NaN в градиентах на нулевой амплитуде.
    """

    def __init__(self, unet: UNet, prop: nn.Module, gs_iters: int = 0, eps: float = 1e-6):
        super().__init__()
        self.unet, self.prop, self.gs_iters, self.eps = unet, prop, gs_iters, eps

    def forward(self, amp: torch.Tensor) -> torch.Tensor:
        """amp — целевая амплитуда (B, P, N, N) -> фаза (B, N, N)."""
        phi = self.unet(amp)
        if self.gs_iters == 0:
            return phi
        H = torch.polar(torch.ones_like(phi), phi)
        for _ in range(self.gs_iters):
            K = self.prop.forward_field(H)
            K = amp * K / (K.abs() + self.eps)
            G = self.prop.backward_field(K).sum(1)
            H = G / (G.abs() + self.eps)
        return torch.atan2(H.imag, H.real)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
