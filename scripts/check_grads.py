"""Проверка, что градиенты доходят до первого слоя сети, на CPU и на MPS/CUDA, и что устройства совпадают.

    python scripts/check_grads.py
"""
import _path  # noqa: F401

import torch

from doe import DEFAULT
from doe.data import fixed_set
from doe.losses import HoloLoss
from doe.optics import FresnelPropagator, get_device, target_amplitude
from doenet.model import build


def run(device, t, gs_iters):
    torch.manual_seed(0)
    prop = FresnelPropagator(DEFAULT).to(device)
    m = build(prop, DEFAULT, base=8, depth=3, gs_iters=gs_iters).to(device)
    tt = t.to(device)
    _, rec = m.recon(target_amplitude(tt, DEFAULT))
    loss, s = HoloLoss(DEFAULT, w_worst=1.0)(rec, tt)
    loss.backward()
    g_first = m.unet.downs[0].net[0].weight.grad
    g_head = m.unet.head.weight.grad
    finite = all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None)
    return loss.item(), s["pcc"], g_first.norm().item(), g_head.norm().item(), finite, g_first.detach().cpu()


def main():
    t = fixed_set(2, DEFAULT.n, DEFAULT.planes, seed=7)
    t[:, :, :20] = 0                                  # много нулевой амплитуды — проверка на NaN
    devs = ["cpu"] + ([str(get_device())] if get_device().type != "cpu" else [])
    ok = True
    for gs in (0, 3):
        res = {}
        for d in devs:
            loss, pcc, gf, gh, fin, g = run(torch.device(d), t, gs)
            res[d] = (loss, g)
            print(f"GS={gs} {d:>4}: loss {loss:.6f} КК {pcc:.4f} |grad первого слоя| {gf:.3e} "
                  f"|grad головы| {gh:.3e} конечные {fin}")
            ok &= fin and gf > 0 and gh > 0
        if len(devs) == 2:
            dl = abs(res[devs[0]][0] - res[devs[1]][0])
            g0, g1 = res[devs[0]][1], res[devs[1]][1]
            rel = ((g0 - g1).norm() / g0.norm()).item()
            print(f"  cpu vs {devs[1]}: |Δloss| {dl:.2e}, отн. расхождение градиента первого слоя {rel:.2e}")
            ok &= dl < 1e-3 and rel < 5e-2
    print("OK" if ok else "ПРОБЛЕМА")


if __name__ == "__main__":
    main()
