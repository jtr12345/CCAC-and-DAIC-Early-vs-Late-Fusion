# -*- coding: utf-8 -*-
"""DAIC 精简版：5 seed，即时输出。"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from _daic_fusion import load, strat_folds
from _daic_fusion2 import run_seed, make_models

ROOT = os.path.dirname(os.path.abspath(__file__))
N_SEEDS = 5


def main():
    X, z = load()
    ysev = z["severity"].astype(int)
    ybin = z["binary"].astype(int)
    print(f"n={len(ysev)}  severity={np.bincount(ysev, minlength=5)}  "
          f"binary阳性={ybin.mean()*100:.1f}%", flush=True)

    pairs = {
        "handcrafted(egemaps+openface)": ("audio_egemaps", "video_openface"),
        "deep(bow+vgg16)": ("audio_bow_egemaps", "video_vgg16"),
    }
    out_all = {}
    for task, yy in (("severity", ysev), ("binary", ybin)):
        print(f"\n{'='*62}\n目标：{task}\n{'='*62}", flush=True)
        for pname, (fa, fv) in pairs.items():
            for mk, model in make_models(task).items():
                if model is None:
                    continue
                rows, ws = [], []
                for seed in range(N_SEEDS):
                    folds = strat_folds(yy, 5, seed)
                    r, w = run_seed(task, yy, X[fa], X[fv], folds, model)
                    rows.append(r); ws.append(w)
                print(f"\n--- {pname} [{mk}] ---", flush=True)
                for k in rows[0]:
                    v = np.array([r[k] for r in rows])
                    print(f"  {k:12s} {v.mean():8.4f} ± {v.std():.4f}", flush=True)
                d = np.array([r["early"] - r["late_w"] for r in rows])
                d2 = np.array([r["early"] - r["late_eq"] for r in rows])
                print(f"  early-late_w  Δ={d.mean():+.4f}  为正 {int((d>0).sum())}/{N_SEEDS}", flush=True)
                print(f"  early-late_eq Δ={d2.mean():+.4f}  为正 {int((d2>0).sum())}/{N_SEEDS}", flush=True)
                print(f"  late 平均 audio 权重={np.mean(ws):.2f}", flush=True)
                out_all[f"{task}|{pname}|{mk}"] = {
                    k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))]
                    for k in rows[0]}

    json.dump(out_all, open(os.path.join(ROOT, "_daic_lean.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote _daic_lean.json", flush=True)


if __name__ == "__main__":
    main()
