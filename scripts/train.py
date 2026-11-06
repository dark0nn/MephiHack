"""Обучение U-Net без учителя через дифференцируемую модель Френеля.

    python scripts/train.py --steps 3000 --batch 16              # на GPU (Colab) ~ 10–30 мин
    python scripts/train.py --steps 200 --batch 4 --base 16      # быстрая проверка на ноутбуке

Дообучение готовой сети (ширина сети берётся из чекпоинта):
    python scripts/train.py --resume checkpoints/<run>/best.pt --steps 2000 --lr 3e-4 --w-worst 1 --gs-iters 5
  --w-worst  штраф за худшую из 4 плоскостей (не даёт «жертвовать» дальней картинкой)
  --gs-iters K итераций GS внутри модели после сети (сеть учится давать старт для GS)

Цикл: 4 цели -> U-Net -> фаза -> 4 восстановления -> лосс (1−КК, 1−SSIM, MSE) -> backprop.
Чекпоинты: checkpoints/<run>/last.pt и best.pt. Лог: checkpoints/<run>/log.csv
"""
import argparse
import csv
import time
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT

import torch

from doe import DEFAULT
from doe.data import TargetSets, fixed_set
from doe.losses import HoloLoss
from doe.model import HoloNet, UNet, count_params
from doe.optics import FresnelPropagator, get_device, target_amplitude


@torch.no_grad()
def validate(model, prop, loss_fn, val, cfg, device, batch=8):
    model.eval()
    stats = {"loss": 0.0, "pcc": 0.0, "ssim": 0.0, "worst": 0.0}
    k = 0
    for i in range(0, len(val), batch):
        t = val[i:i + batch].to(device)
        phi = model(target_amplitude(t, cfg))
        _, s = loss_fn(prop.recon_image(phi), t)
        for key in stats:
            stats[key] += s[key] * len(t)
        k += len(t)
    model.train()
    return {key: v / k for key, v in stats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32, help="ширина U-Net")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--images", default=str(ROOT / "data/images"), help="своя папка картинок (необязательно)")
    ap.add_argument("--val-size", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run", default=time.strftime("unet_%Y%m%d_%H%M"))
    ap.add_argument("--resume", default=None, help="путь к .pt, чтобы дообучить готовую сеть")
    ap.add_argument("--gs-iters", type=int, default=0, help="итераций GS внутри модели после сети")
    ap.add_argument("--w-worst", type=float, default=0.0, help="вес штрафа за худшую плоскость")
    args = ap.parse_args()

    cfg = DEFAULT
    device = get_device(args.device)
    torch.manual_seed(0)
    out = ROOT / "checkpoints" / args.run
    out.mkdir(parents=True, exist_ok=True)

    ck_resume = None
    if args.resume:
        ck_resume = torch.load(args.resume, map_location=device)
        a = ck_resume.get("args", {})
        args.base, args.depth = a.get("base", args.base), a.get("depth", args.depth)
    unet = UNet(in_ch=cfg.planes, base=args.base, depth=args.depth).to(device)
    if ck_resume is not None:
        unet.load_state_dict(ck_resume["model"])
        print(f"Дообучаю {args.resume} (шаг {ck_resume.get('step')}, val КК {ck_resume.get('val', {}).get('pcc', 0):.3f})")
    prop = FresnelPropagator(cfg).to(device)
    model = HoloNet(unet, prop, gs_iters=args.gs_iters).to(device)
    loss_fn = HoloLoss(cfg, w_worst=args.w_worst)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    data = TargetSets(cfg.n, cfg.planes, folder=args.images)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, num_workers=args.workers,
                                         persistent_workers=args.workers > 0)
    val = fixed_set(args.val_size, cfg.n, cfg.planes)

    print(f"Устройство: {device}; параметров: {count_params(unet):,}; GS внутри: {args.gs_iters}; "
          f"штраф худшей плоскости: {args.w_worst}; "
          f"картинки из папки: {'да' if data.folder else 'нет, только синтетика'}")
    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "train_loss", "train_pcc", "train_ssim", "val_loss", "val_pcc", "val_ssim",
                     "val_worst_pcc", "sec"])

    best = float("inf")
    t0 = time.time()
    run_stats = {"loss": 0.0, "pcc": 0.0, "ssim": 0.0, "worst": 0.0}
    it = iter(loader)
    for step in range(1, args.steps + 1):
        t = next(it).to(device, non_blocking=True)
        phi = model(target_amplitude(t, cfg))
        loss, s = loss_fn(prop.recon_image(phi), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        sched.step()
        for key in run_stats:
            run_stats[key] += s[key]

        if step % args.val_every == 0 or step == args.steps:
            n = args.val_every if step % args.val_every == 0 else step % args.val_every
            tr = {k: v / n for k, v in run_stats.items()}
            run_stats = {k: 0.0 for k in run_stats}
            v = validate(model, prop, loss_fn, val, cfg, device)
            sec = time.time() - t0
            writer.writerow([step, tr["loss"], tr["pcc"], tr["ssim"], v["loss"], v["pcc"], v["ssim"], v["worst"],
                             round(sec, 1)])
            log.flush()
            print(f"шаг {step:>6} | train loss {tr['loss']:.4f} КК {tr['pcc']:.3f} | "
                  f"val loss {v['loss']:.4f} КК {v['pcc']:.3f} (худшая {v['worst']:.3f}) SSIM {v['ssim']:.3f} | {sec:.0f} с")
            ck = {"model": unet.state_dict(), "args": vars(args), "step": step, "val": v}
            torch.save(ck, out / "last.pt")
            if v["loss"] < best:
                best = v["loss"]
                torch.save(ck, out / "best.pt")
    log.close()
    print("Готово. Лучшая модель:", out / "best.pt")


if __name__ == "__main__":
    main()
