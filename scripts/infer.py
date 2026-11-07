"""Финальный скрипт по формату ТЗ: 4 PNG -> ДОЭ + 4 восстановления + метрики.

    python scripts/infer.py --targets data/examples/letters            # последняя обученная сеть
    python scripts/infer.py --weights checkpoints/<run>/best.pt --targets data/examples/letters
    python scripts/infer.py --weights ... --targets ... --refine 50   # + 50 шагов Adam после сети

Результат: outputs/infer/<набор>/ — doe.png, recon_1..4.png, metrics.json
"""
import argparse
import time
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT

import torch

from doe import DEFAULT
from doe.baselines import adam_phase
from doe.io import load_targets
from doe.metrics import format_table
from doe.model import build_model
from doe.optics import FresnelPropagator, get_device, target_amplitude
from doe.pipeline import finalize_and_save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="latest",
                    help="путь к .pt; по умолчанию best.pt из самого свежего запуска в checkpoints/")
    ap.add_argument("--targets", required=True, help="папка с 4 картинками (по алфавиту = плоскости 1..4)")
    ap.add_argument("--refine", type=int, default=0, help="шагов Adam после сети (0 = чистая сеть)")
    ap.add_argument("--gs-iters", type=int, default=None, help="итераций GS после сети (по умолчанию как при обучении)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = DEFAULT
    device = get_device(args.device)
    if args.weights == "latest":
        found = sorted((ROOT / "checkpoints").glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise SystemExit("В checkpoints/ нет ни одного best.pt: сначала обучите сеть (scripts/train.py) "
                             "или укажите --weights путь/к/файлу.pt")
        args.weights = str(found[-1])
        print(f"Веса: {Path(args.weights).relative_to(ROOT)}")
    ck = torch.load(args.weights, map_location=device)
    a = ck.get("args", {})
    prop = FresnelPropagator(cfg).to(device)
    gs_iters = args.gs_iters if args.gs_iters is not None else a.get("gs_iters", 0)
    net_input = a.get("net_input", "targets")              # старые чекпоинты — без physics-входа
    model = build_model(cfg, prop, base=a.get("base", 32), depth=a.get("depth", 4), gs_iters=gs_iters,
                        net_input=net_input, pre_gs=a.get("pre_gs", 1), zero_head=False).to(device)
    model.unet.load_state_dict(ck["model"])
    model.eval()

    targets_np = load_targets(args.targets, cfg.n)
    t = torch.from_numpy(targets_np)[None].to(device)

    with torch.no_grad():
        model(target_amplitude(t, cfg))                   # прогрев
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        phi = model(target_amplitude(t, cfg))
        if device.type == "cuda":
            torch.cuda.synchronize()
        net_sec = time.perf_counter() - t0

    ref_sec = 0.0
    if args.refine > 0:
        t0 = time.perf_counter()
        phi = adam_phase(t, iters=args.refine, cfg=cfg, prop=prop, init=phi)
        ref_sec = time.perf_counter() - t0

    out = args.out or str(ROOT / "outputs/infer" / Path(args.targets).name)
    res = finalize_and_save(phi[0].cpu().numpy(), targets_np, out, cfg,
                            extra={"method": {"physics": "phys+unet", "target_phase": "tphase+unet"}.get(net_input, "unet")
                                             + (f"+gs{gs_iters}" if gs_iters else "")
                                             + (f"+adam{args.refine}" if args.refine else ""),
                                   "net_seconds": net_sec, "refine_seconds": ref_sec})
    print(f"Сеть{f' + {gs_iters} GS' if gs_iters else ''}: {net_sec * 1000:.1f} мс; дотюнивание: {ref_sec:.2f} с -> {out}")
    print(format_table(res))


if __name__ == "__main__":
    main()
