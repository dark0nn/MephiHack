"""Обучение U-Net без учителя через дифференцируемую модель Френеля.

    python scripts/train.py --steps 3000 --batch 16              # на GPU (Colab) ~ 10–30 мин
    python scripts/train.py --steps 200 --batch 4 --base 16      # быстрая проверка на ноутбуке

Продолжить оборванный запуск с того же шага (те же параметры, тот же лог):
    python scripts/train.py --continue checkpoints/<run>/last.pt

Новая версия модели поверх готовой (ширина сети берётся из чекпоинта, совпадающие слои
загружаются, новые учатся с нуля; результат пишется в новый запуск):
    python scripts/train.py --resume checkpoints/<run>/best.pt --steps 2000 --lr 3e-4 --w-worst 1 --gs-iters 5
  --freeze-loaded N  первые N шагов учить только новые слои, загруженные заморожены
  --w-worst  штраф за худшую из 4 плоскостей (не даёт «жертвовать» дальней картинкой)
  --gs-iters K итераций GS внутри модели после сети (сеть учится давать старт для GS)

Вход сети (--net-input):
  target_phase (по умолч.) — сеть предсказывает фазу на каждой плоскости изображения, физика переносит
                           поле на ДОЭ (как HoloNet). Самый перспективный режим.
  physics                — сеть получает поле, посчитанное физикой (обратное распространение целей
                           на ДОЭ = 1 шаг GS), и выдаёт поправку к его фазе. До обучения = результат GS.
  targets                — сеть получает 4 картинки и сразу выдаёт фазу (старые чекпоинты).

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
from doe.checkpoint import load_matching, save_checkpoint
from doe.data import TargetSets, fixed_set
from doe.losses import HoloLoss
from doe.model import HoloNet, build_model, count_params
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
    ap.add_argument("--resume", default=None, help="путь к .pt: веса как старт для новой версии модели")
    ap.add_argument("--continue", dest="cont", default=None,
                    help="путь к last.pt: продолжить тот же запуск с того же шага")
    ap.add_argument("--freeze-loaded", type=int, default=0,
                    help="при --resume: столько первых шагов не трогать загруженные слои")
    ap.add_argument("--net-input", choices=["target_phase", "physics", "targets"], default="target_phase",
                    help="target_phase: сеть даёт фазу на плоскостях изображения, физика переносит на ДОЭ; "
                         "physics: поправка к полю от обратного распространения; targets: картинки -> фаза")
    ap.add_argument("--pre-gs", type=int, default=1,
                    help="при physics: сколько шагов GS считается до сети (1 = одно обратное распространение)")
    ap.add_argument("--gs-iters", type=int, default=0, help="итераций GS внутри модели после сети")
    ap.add_argument("--w-worst", type=float, default=0.0, help="вес штрафа за худшую плоскость")
    args = ap.parse_args()

    cfg = DEFAULT
    ck_cont = None
    if args.cont:
        # Продолжение: все параметры берём из чекпоинта, менять можно только устройство и workers
        ck_cont = torch.load(args.cont, map_location="cpu")
        if "opt" not in ck_cont:
            raise SystemExit("В этом чекпоинте нет состояния оптимизатора (старый формат). "
                             "Используйте --resume, чтобы дообучить с этих весов.")
        keep = {"device": args.device, "workers": args.workers, "cont": args.cont}
        old = {"net_input": "targets", "pre_gs": 1}         # чекпоинты до появления physics-входа
        args = argparse.Namespace(**{**vars(ap.parse_args([])), **old, **ck_cont["args"], **keep})
    device = get_device(args.device)
    torch.manual_seed(0)
    out = ROOT / "checkpoints" / args.run
    out.mkdir(parents=True, exist_ok=True)

    ck_resume = None
    if args.resume and not args.cont:
        ck_resume = torch.load(args.resume, map_location=device)
        a = ck_resume.get("args", {})
        args.base, args.depth = a.get("base", args.base), a.get("depth", args.depth)
    prop = FresnelPropagator(cfg).to(device)
    model = build_model(cfg, prop, base=args.base, depth=args.depth, gs_iters=args.gs_iters,
                        net_input=args.net_input, pre_gs=args.pre_gs).to(device)
    unet = model.unet
    loaded = []
    if ck_cont is not None:
        unet.load_state_dict(ck_cont["model"])
    elif ck_resume is not None:
        print(f"Старт с весов {args.resume} (шаг {ck_resume.get('step')}, "
              f"val КК {ck_resume.get('val', {}).get('pcc', 0):.3f})")
        loaded = load_matching(unet, ck_resume["model"])["loaded"]
        old_input = ck_resume.get("args", {}).get("net_input", "targets")
        if old_input != args.net_input and args.net_input == "physics":  # target_phase: голова другой формы
            # голова старой сети выдавала фазу целиком, а здесь нужна поправка: начинаем с нуля
            HoloNet.zero_correction_head(unet)
            loaded = [k for k in loaded if not k.startswith("head.")]
            print(f"  вход сети сменился ({old_input} -> physics): голова обнулена, первый слой учится заново")
    loss_fn = HoloLoss(cfg, w_worst=args.w_worst)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    start, best = 1, float("inf")
    if ck_cont is not None:
        opt.load_state_dict(ck_cont["opt"])
        sched.load_state_dict(ck_cont["sched"])
        start, best = ck_cont["step"] + 1, ck_cont.get("best", float("inf"))
        print(f"Продолжаю запуск {args.run} с шага {start} из {args.steps}")
        if start > args.steps:
            raise SystemExit("Этот запуск уже завершён. Для дообучения используйте --resume.")

    frozen = []
    if args.freeze_loaded > 0 and loaded:
        params = dict(unet.named_parameters())
        frozen = [params[k] for k in loaded if k in params]
        for p in frozen:
            p.requires_grad_(False)
        print(f"Заморожено {len(frozen)} загруженных тензоров на {args.freeze_loaded} шагов")

    data = TargetSets(cfg.n, cfg.planes, folder=args.images)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, num_workers=args.workers,
                                         persistent_workers=args.workers > 0)
    val = fixed_set(args.val_size, cfg.n, cfg.planes)

    print(f"Устройство: {device}; вход сети: {args.net_input}"
          f"{f' (GS до сети: {args.pre_gs})' if args.net_input == 'physics' else ''}; "
          f"параметров: {count_params(unet):,}; GS после сети: {args.gs_iters}; "
          f"штраф худшей плоскости: {args.w_worst}; "
          f"картинки из папки: {data.folder.describe() + ' (50% примеров)' if data.folder else 'нет, только синтетика'}")
    log = open(out / "log.csv", "a" if ck_cont is not None else "w", newline="")
    writer = csv.writer(log)
    if ck_cont is None:
        writer.writerow(["step", "train_loss", "train_pcc", "train_ssim", "val_loss", "val_pcc", "val_ssim",
                         "val_worst_pcc", "sec"])

    t0 = time.time()
    run_stats = {"loss": 0.0, "pcc": 0.0, "ssim": 0.0, "worst": 0.0}
    it = iter(loader)
    for step in range(start, args.steps + 1):
        if frozen and step == start + args.freeze_loaded:
            for p in frozen:
                p.requires_grad_(True)
            frozen = []
            print(f"шаг {step}: размораживаю загруженные слои")
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
            n = max(1, min(step - start + 1, args.val_every if step % args.val_every == 0 else step % args.val_every))
            tr = {k: v / n for k, v in run_stats.items()}
            run_stats = {k: 0.0 for k in run_stats}
            v = validate(model, prop, loss_fn, val, cfg, device)
            sec = time.time() - t0
            writer.writerow([step, tr["loss"], tr["pcc"], tr["ssim"], v["loss"], v["pcc"], v["ssim"], v["worst"],
                             round(sec, 1)])
            log.flush()
            print(f"шаг {step:>6} | train loss {tr['loss']:.4f} КК {tr['pcc']:.3f} | "
                  f"val loss {v['loss']:.4f} КК {v['pcc']:.3f} (худшая {v['worst']:.3f}) SSIM {v['ssim']:.3f} | {sec:.0f} с")
            if v["loss"] < best:
                best = v["loss"]
                save_checkpoint(out / "best.pt", unet, opt, sched, step, best, v, vars(args))
            save_checkpoint(out / "last.pt", unet, opt, sched, step, best, v, vars(args))
    log.close()
    print("Готово. Лучшая модель:", out / "best.pt")


if __name__ == "__main__":
    main()
