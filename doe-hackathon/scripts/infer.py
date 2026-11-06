"""Финальный скрипт по формату ТЗ: 4 PNG -> ДОЭ + 4 восстановления + метрики.

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
from doe.model import UNet
from doe.optics import FresnelPropagator, get_device, target_amplitude
from doe.pipeline import finalize_and_save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--targets", required=True, help="папка с 4 картинками (по алфавиту = плоскости 1..4)")
    ap.add_argument("--refine", type=int, default=0, help="шагов Adam после сети (0 = чистая сеть)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = DEFAULT
    device = get_device(args.device)
    ck = torch.load(args.weights, map_location=device)
    a = ck.get("args", {})
    model = UNet(in_ch=cfg.planes, base=a.get("base", 32), depth=a.get("depth", 4)).to(device).eval()
    model.load_state_dict(ck["model"])
    prop = FresnelPropagator(cfg).to(device)

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
                            extra={"method": "unet" + (f"+adam{args.refine}" if args.refine else ""),
                                   "net_seconds": net_sec, "refine_seconds": ref_sec})
    print(f"Сеть: {net_sec * 1000:.1f} мс; дотюнивание: {ref_sec:.2f} с -> {out}")
    print(format_table(res))


if __name__ == "__main__":
    main()
