"""Функция потерь: напрямую оптимизируем то, по чему оценивает жюри."""
import torch

from .config import Config, DEFAULT
from .metrics import pearson, ssim_global, normalize


def scale_invariant_mse(rec: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """MSE после подбора оптимального масштаба s = <rec,tgt>/<rec,rec> (масштаб не обучается)."""
    s = (rec * tgt).sum((-2, -1), keepdim=True) / ((rec * rec).sum((-2, -1), keepdim=True) + eps)
    return ((s.detach() * rec - tgt) ** 2).mean((-2, -1))


class HoloLoss(torch.nn.Module):
    """loss = w_pcc·(1−КК) + w_ssim·(1−SSIM) + w_mse·MSE, усреднение по плоскостям и батчу.

    rec — восстановления (B, P, N, N) в «единицах картинки» (|K| или |K|²),
    tgt — цели (B, P, N, N) в [0, 1].
    """

    def __init__(self, cfg: Config = DEFAULT, w_pcc: float = 1.0, w_ssim: float = 1.0, w_mse: float = 1.0):
        super().__init__()
        self.cfg, self.w_pcc, self.w_ssim, self.w_mse = cfg, w_pcc, w_ssim, w_mse

    def forward(self, rec: torch.Tensor, tgt: torch.Tensor):
        rec_n = normalize(rec, self.cfg.recon_norm)
        pcc = pearson(rec, tgt)
        ssim = ssim_global(rec_n, tgt)
        mse = scale_invariant_mse(rec, tgt)
        loss = self.w_pcc * (1 - pcc).mean() + self.w_ssim * (1 - ssim).mean() + self.w_mse * mse.mean()
        return loss, {"loss": loss.item(), "pcc": pcc.mean().item(), "ssim": ssim.mean().item()}
