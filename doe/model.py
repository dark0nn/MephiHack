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

    def __init__(self, in_ch: int = 4, base: int = 32, depth: int = 4, out_planes: int = 1):
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
        self.out_planes = out_planes
        self.head = nn.Conv2d(chs[0], 2 * out_planes, 1)    # (c, s) на каждую выходную фазу

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        """targets (B, P, N, N) -> фаза (B, N, N) в (-π, π]; при out_planes > 1 — (B, out_planes, N, N)."""
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
        if self.out_planes == 1:
            return torch.atan2(cs[:, 1], cs[:, 0])
        cs = cs.view(cs.shape[0], self.out_planes, 2, *cs.shape[-2:])
        return torch.atan2(cs[:, :, 1], cs[:, :, 0])


class HoloNet(nn.Module):
    """Полная модель: (физический старт) -> U-Net -> (K итераций GS).

    Три режима (net_input):

    * "target_phase" — сеть смотрит на картинки и предсказывает фазу В ПЛОСКОСТЯХ ИЗОБРАЖЕНИЯ
      (по одной на каждую из P плоскостей). Затем физика: амплитуда цели с этой фазой обратно
      распространяется на ДОЭ, поля P плоскостей суммируются, фаза суммы = маска.
      Задача для свёрток локальная: фаза лежит пиксель в пиксель с картинкой (как в HoloNet,
      Peng et al. 2020). Обратное распространение — внутри графа, сеть учится через него.

    * "targets" — сеть получает 4 картинки и сразу выдаёт фазу (старый режим).
    * "physics" — сначала физикой считается приближение: цели с фиксированной случайной фазой
      обратно распространяются на плоскость ДОЭ и суммируются (= один шаг GS; pre_gs > 1 даёт
      ещё несколько шагов GS). Сеть получает это поле — для каждой плоскости Re/Im обратного поля
      и cos/sin суммарной фазы, всего 2P+2 канала — и выдаёт ПОПРАВКУ к фазе.
      Всё, что видит сеть, уже лежит в плоскости ДОЭ и выровнено по пикселям, поэтому задача
      для свёрток становится локальной. Голова сети инициализируется так, что поправка ≈ 0:
      необученная модель сразу даёт результат pre_gs шагов GS, и обучение только улучшает его.

    После сети — K итераций параллельного GS (gs_iters), они дифференцируемы, поэтому сеть
    учится давать лучший старт для GS. Нормировки z/(|z|+eps) вместо angle() — чтобы не было NaN
    в градиентах на нулевой амплитуде.
    """

    def __init__(self, unet: UNet, prop: nn.Module, gs_iters: int = 0, eps: float = 1e-6,
                 net_input: str = "targets", pre_gs: int = 1, seed: int = 0):
        super().__init__()
        assert net_input in ("targets", "physics", "target_phase"), net_input
        self.unet, self.prop, self.gs_iters, self.eps = unet, prop, gs_iters, eps
        self.net_input, self.pre_gs = net_input, max(1, pre_gs)
        if net_input == "physics":
            P, n = prop.chirp_re.shape[0], prop.chirp_re.shape[-1]
            g = torch.Generator().manual_seed(seed)
            init = (torch.rand(P, n, n, generator=g) * 2 - 1) * torch.pi
            # не сохраняется в чекпоинт: при том же seed восстанавливается одинаково
            self.register_buffer("init_phase", init.to(prop.chirp_re.device), persistent=False)

    @staticmethod
    def in_channels(net_input: str, planes: int) -> int:
        return 2 * planes + 2 if net_input == "physics" else planes

    @staticmethod
    def zero_correction_head(unet: UNet, std: float = 1e-2):
        """Голова выдаёт (c, s) ≈ (1, 0) -> поправка atan2(s, c) ≈ 0 для любого входа.

        Веса не ровно нулевые (std): через нулевую голову градиент не доходит до остальной сети,
        и она не учится вообще. С std = 0.01 поправка на старте ~0.01 рад — на качество не влияет.
        """
        with torch.no_grad():
            unet.head.weight.normal_(0.0, std)
            unet.head.bias.copy_(torch.tensor([1.0, 0.0]))

    def _unit(self, z: torch.Tensor) -> torch.Tensor:
        return z / (z.abs() + self.eps)

    @torch.no_grad()
    def physics_start(self, amp: torch.Tensor):
        """amp (B, P, N, N) -> стартовая фаза phi0 (B, N, N) и признаки для сети (B, 2P+2, N, N).

        Считается без градиента: параметров тут нет, а комплексная арифметика в графе
        на Apple MPS теряет градиент (сеть «стоит» — val loss не меняется вообще).
        """
        K = torch.polar(amp, self.init_phase.expand_as(amp))
        G = self.prop.backward_field(K)                     # (B, P, N, N)
        H = self._unit(G.sum(1))
        for _ in range(self.pre_gs - 1):
            K = amp * self._unit(self.prop.forward_field(H))
            G = self.prop.backward_field(K)
            H = self._unit(G.sum(1))
        scale = G.abs().pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt() + self.eps
        Gn = G / scale
        feats = torch.cat([Gn.real, Gn.imag, H.real[:, None], H.imag[:, None]], 1)
        return torch.atan2(H.imag, H.real), feats

    def forward(self, amp: torch.Tensor) -> torch.Tensor:
        """amp — целевая амплитуда (B, P, N, N) -> фаза (B, N, N)."""
        if self.net_input == "target_phase":
            psi = self.unet(amp)                            # (B, P, N, N) — фазы на плоскостях изображения
            H = self.prop.backward_field(torch.polar(amp, psi)).sum(1)
            phi = torch.atan2(H.imag, H.real)
        elif self.net_input == "physics":
            phi0, feats = self.physics_start(amp)
            phi = phi0 + self.unet(feats)                   # поправка к фазе, только вещественные операции
            phi = torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi   # обратно в [-π, π)
        else:
            phi = self.unet(amp)
        if self.gs_iters == 0:
            return phi
        H = torch.polar(torch.ones_like(phi), phi)
        for _ in range(self.gs_iters):
            K = amp * self._unit(self.prop.forward_field(H))
            H = self._unit(self.prop.backward_field(K).sum(1))
        return torch.atan2(H.imag, H.real)


def build_model(cfg, prop: nn.Module, base: int = 32, depth: int = 4, gs_iters: int = 0,
                net_input: str = "targets", pre_gs: int = 1, zero_head: bool = True,
                head_std: float = 1e-2) -> HoloNet:
    """Собрать U-Net + HoloNet с нужным числом входных каналов (общая точка для train и infer)."""
    unet = UNet(in_ch=HoloNet.in_channels(net_input, cfg.planes), base=base, depth=depth,
                out_planes=cfg.planes if net_input == "target_phase" else 1)
    if net_input == "physics" and zero_head:
        HoloNet.zero_correction_head(unet, head_std)
    return HoloNet(unet, prop, gs_iters=gs_iters, net_input=net_input, pre_gs=pre_gs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
