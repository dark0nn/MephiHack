"""Итеративные бейзлайны: GS и градиентный спуск. Сохраняет результат в формате ТЗ.

    python scripts/run_baseline.py --targets data/examples/shapes --method gs_parallel --iters 100
    python scripts/run_baseline.py --targets data/examples/shapes --method all

Результат: outputs/baselines/<набор>/<метод>/ — doe.png, recon_1..4.png, metrics.json
"""
import argparse
import json
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT

import torch

from doe import DEFAULT
from doe.baselines import METHODS, run
from doe.io import load_targets
from doe.metrics import format_table
from doe.optics import get_device
from doe.pipeline import finalize_and_save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(ROOT / "data/examples/shapes"), help="папка с 4 картинками")
    ap.add_argument("--method", default="all", choices=["all", *METHODS])
    ap.add_argument("--iters", type=int, default=None, help="по умолчанию: GS 100, Adam 300")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "outputs/baselines"))
    args = ap.parse_args()

    cfg = DEFAULT
    device = get_device(args.device)
    targets_np = load_targets(args.targets, cfg.n)
    targets = torch.from_numpy(targets_np)
    set_name = Path(args.targets).name
    methods = list(METHODS) if args.method == "all" else [args.method]
    print(f"Устройство: {device}; набор: {args.targets}")

    summary = {}
    for m in methods:
        iters = args.iters or (300 if m == "adam" else 100)
        phi, sec = run(m, targets, iters, cfg, device=device)
        out = f"{args.out}/{set_name}/{m}"
        res = finalize_and_save(phi.numpy(), targets_np, out, cfg, extra={"method": m, "iters": iters, "seconds": sec})
        summary[m] = {"iters": iters, "seconds": round(sec, 3), **{k: round(v, 4) for k, v in res["mean"].items()}}
        print(f"\n== {m}: {iters} итераций, {sec:.2f} с -> {out}")
        print(format_table(res))

    print("\nСводка:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
