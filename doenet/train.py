"""Обучение PlanePhaseNet без учителя через дифференцируемую физику.

    python scripts/train_net.py --steps 300 --batch 4 --base 16 --val-every 50 --run smoke
    python scripts/train_net.py --continue checkpoints/<run>/last.pt      # тот же запуск, с того же шага
    python scripts/train_net.py --resume checkpoints/<run>/best.pt ...    # веса как старт новой версии

Лог: checkpoints/<run>/log.csv. Чекпоинты: last.pt, best.pt (по среднему val loss двух валидаций).
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from doe import DEFAULT
from doe.baselines import gs_parallel
from doe.checkpoint import load_matching
from doe.losses import HoloLoss
from doe.optics import FresnelPropagator, get_device, target_amplitude

from .data import train_stream, val_sets
from .model import build, count_params

ROOT = Path(__file__).resolve().parents[1]
KIND = "doenet"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--gs-iters", type=int, default=0, help="итераций дифференцируемого GS внутри модели")
    ap.add_argument("--in-norm", choices=["raw", "std"], default="raw")
    ap.add_argument("--psi-offset", choices=["none", "random"], default="none",
                    help="фиксированная случайная добавка к фазам плоскостей изображения")
    ap.add_argument("--w-worst", type=float, default=0.0, help="вес штрафа за худшую плоскость")
    ap.add_argument("--images", default=str(ROOT / "data/images"))
    ap.add_argument("--val-size", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default=None)
    ap.add_argument("--continue", dest="cont", default=None)
    ap.add_argument("--resume", default=None)
    return ap


def lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))
    return f


@torch.no_grad()
def validate(model, prop, loss_fn, data, cfg, device, batch=8, with_gs5=True):
    model.eval()
    acc = {"loss": 0.0, "pcc": 0.0, "ssim": 0.0, "worst": 0.0, "pcc_gs5": 0.0}
    for i in range(0, len(data), batch):
        t = data[i:i + batch].to(device)
        amp = target_amplitude(t, cfg)
        phi, rec = model.recon(amp, cfg.recon_quantity)
        _, s = loss_fn(rec, t)
        for k in ("loss", "pcc", "ssim", "worst"):
            acc[k] += s[k] * len(t)
        if with_gs5:
            phi5 = gs_parallel(t, iters=5, cfg=cfg, prop=prop, init=phi)
            _, s5 = loss_fn(prop.recon_image(phi5), t)
            acc["pcc_gs5"] += s5["pcc"] * len(t)
    model.train()
    return {k: v / len(data) for k, v in acc.items()}


@torch.no_grad()
def gs_reference(prop, loss_fn, data, cfg, device):
    """КК бейзлайнов GS-5 и GS-50 (случайный старт) на валидации — цель, которую надо побить."""
    out = {}
    for it in (5, 50):
        tot = 0.0
        for i in range(0, len(data), 8):
            t = data[i:i + 8].to(device)
            _, s = loss_fn(prop.recon_image(gs_parallel(t, iters=it, cfg=cfg, prop=prop, seed=i)), t)
            tot += s["pcc"] * len(t)
        out[f"gs{it}"] = tot / len(data)
    return out


def save(path, model, opt, sched, step, best, val, args):
    torch.save({"kind": KIND, "version": 1, "model": model.unet.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": step, "best": best, "val": val, "args": dict(args)}, path)


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    ck_cont = None
    if args.cont:
        ck_cont = torch.load(args.cont, map_location="cpu")
        if ck_cont.get("kind") != KIND:
            raise SystemExit("Это не чекпоинт doenet: --continue работает только с чекпоинтами scripts/train_net.py")
        keep = {"device": args.device, "workers": args.workers, "cont": args.cont}
        args = argparse.Namespace(**{**vars(ap.parse_args([])), **ck_cont["args"], **keep})
    args.run = args.run or time.strftime("net_%Y%m%d_%H%M")

    cfg = DEFAULT
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    out = ROOT / "checkpoints" / args.run
    out.mkdir(parents=True, exist_ok=True)

    prop = FresnelPropagator(cfg).to(device)
    if args.resume and not args.cont:
        ck_res = torch.load(args.resume, map_location="cpu")
        a = ck_res.get("args", {})
        args.base, args.depth = a.get("base", args.base), a.get("depth", args.depth)
    model = build(prop, cfg, args.base, args.depth, args.gs_iters, args.in_norm,
                  args.psi_offset, args.seed).to(device)
    if ck_cont is not None:
        model.unet.load_state_dict(ck_cont["model"])
    elif args.resume:
        load_matching(model.unet, ck_res["model"])

    loss_fn = HoloLoss(cfg, w_worst=args.w_worst)
    opt = torch.optim.AdamW(model.unet.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(args.warmup, args.steps))
    start, best = 1, float("inf")
    if ck_cont is not None:
        opt.load_state_dict(ck_cont["opt"])
        sched.load_state_dict(ck_cont["sched"])
        start, best = ck_cont["step"] + 1, ck_cont.get("best", float("inf"))
        if start > args.steps:
            raise SystemExit("Этот запуск уже завершён; для дообучения используйте --resume с новым --steps.")
        print(f"Продолжаю {args.run} с шага {start}")

    data = train_stream(cfg.n, cfg.planes, args.images, seed=args.seed)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, num_workers=args.workers,
                                         persistent_workers=args.workers > 0)
    vals = val_sets(cfg.n, cfg.planes, args.val_size, args.images)
    print(f"Устройство: {device}; параметров: {count_params(model):,}; GS в модели: {args.gs_iters}; "
          f"вход: {args.in_norm}; добавка к ψ: {args.psi_offset}; lr {args.lr}; валидации: {', '.join(f'{k} ({len(v)})' for k, v in vals.items())}")
    if data.folder is None:
        print("ВНИМАНИЕ: data/images пуст — обучение только на синтетике (python scripts/get_datasets.py)")
    for name, v in vals.items():
        ref = gs_reference(prop, loss_fn, v, cfg, device)
        print(f"  цель на val {name}: КК GS-5 {ref['gs5']:.3f}, GS-50 {ref['gs50']:.3f}")

    cols = ["step", "lr", "train_loss", "train_pcc", "train_ssim", "train_worst"]
    for name in vals:
        cols += [f"{name}_{k}" for k in ("loss", "pcc", "ssim", "worst", "pcc_gs5")]
    cols += ["grad_first", "sec"]
    new_log = ck_cont is None or not (out / "log.csv").exists()
    log = open(out / "log.csv", "w" if new_log else "a", newline="", encoding="utf-8")
    w = csv.writer(log)
    if new_log:
        w.writerow(cols)

    first = model.unet.downs[0].net[0].weight
    run = {k: 0.0 for k in ("loss", "pcc", "ssim", "worst", "grad")}
    k_run, t0 = 0, time.time()
    it = iter(loader)
    for step in range(start, args.steps + 1):
        t = next(it).to(device, non_blocking=True)
        _, rec = model.recon(target_amplitude(t, cfg), cfg.recon_quantity)
        loss, s = loss_fn(rec, t)
        if not torch.isfinite(loss):
            raise SystemExit(f"шаг {step}: loss = {loss.item()} — остановка")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        run["grad"] += first.grad.norm().item()
        torch.nn.utils.clip_grad_norm_(model.unet.parameters(), 1.0)
        opt.step()
        sched.step()
        for k in ("loss", "pcc", "ssim", "worst"):
            run[k] += s[k]
        k_run += 1

        if step % args.val_every == 0 or step == args.steps:
            tr = {k: v / k_run for k, v in run.items()}
            row = [step, sched.get_last_lr()[0], tr["loss"], tr["pcc"], tr["ssim"], tr["worst"]]
            vr = {}
            for name, v in vals.items():
                vr[name] = validate(model, prop, loss_fn, v, cfg, device)
                row += [vr[name][k] for k in ("loss", "pcc", "ssim", "worst", "pcc_gs5")]
            sec = time.time() - t0
            row += [tr["grad"], round(sec, 1)]
            w.writerow(row)
            log.flush()
            msg = " | ".join(f"{n}: loss {r['loss']:.4f} КК {r['pcc']:.3f} (худш. {r['worst']:.3f}) "
                             f"SSIM {r['ssim']:.3f} +GS5 {r['pcc_gs5']:.3f}" for n, r in vr.items())
            print(f"шаг {step:>6} | train {tr['loss']:.4f} КК {tr['pcc']:.3f} | {msg} | grad1 {tr['grad']:.2e} | {sec:.0f} с",
                  flush=True)
            score = sum(r["loss"] for r in vr.values()) / len(vr)
            if score < best:
                best = score
                save(out / "best.pt", model, opt, sched, step, best, vr, vars(args))
            save(out / "last.pt", model, opt, sched, step, best, vr, vars(args))
            run = {k: 0.0 for k in run}
            k_run = 0
    log.close()
    print("Готово:", out / "best.pt")


def load_for_inference(ck: dict, prop, cfg, device, gs_iters: int | None = None):
    a = ck["args"]
    model = build(prop, cfg, a["base"], a["depth"], a["gs_iters"] if gs_iters is None else gs_iters,
                  a.get("in_norm", "raw"), a.get("psi_offset", "none"), a.get("seed", 0)).to(device)
    model.unet.load_state_dict(ck["model"])
    return model.eval()
