# -*- coding: utf-8 -*-
"""合成数据功效校验：验证本实验设计能否检出真实交互效应。

三种生成机制，样本量与 DAIC 一致（n=274）：
  additive    : y 由两个模态的主效应相加决定
  interaction : y 只由 a0*v0 决定（无主效应）
  null        : y 与特征独立
预期：additive 下 early≈late；interaction 下 early(MLP) 明显优于 late；null 下都≈0。
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _daic_fusion import qwk, prep, softmax_reg, mlp, strat_folds, N_CLASS
from _daic_fusion2 import run_seed

sys.stdout.reconfigure(encoding="utf-8")
N = 274
D = 32
N_SEEDS = 10


def make(kind, seed):
    rng = np.random.RandomState(seed)
    a = rng.randn(N, D).astype(np.float32)
    v = rng.randn(N, D).astype(np.float32)
    if kind == "additive":
        s = 1.5 * a[:, 0] + 1.5 * v[:, 0] + 0.8 * rng.randn(N)
    elif kind == "interaction":
        s = 2.5 * (a[:, 0] * v[:, 0]) + 0.5 * rng.randn(N)
    else:
        s = rng.randn(N)
    qs = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
    y = np.digitize(s, qs)
    return a, v, y.astype(int)


def main():
    models = {"linear": lambda a, b, c: softmax_reg(a, b, c, l2=0.1),
              "mlp": lambda a, b, c: mlp(a, b, c, h=48, epochs=350)}
    print(f"n={N}  d={D}  5 类均衡  重复 {N_SEEDS} 次")
    for kind in ("additive", "interaction", "null"):
        print(f"\n=== 生成机制：{kind} ===")
        for mk, model in models.items():
            rows = []
            for seed in range(N_SEEDS):
                a, v, y = make(kind, 1000 + seed)
                folds = strat_folds(y, 5, seed)
                r, _ = run_seed("severity", y, a, v, folds, model, nested=True)
                rows.append(r)
            print(f"  [{mk}]")
            for k in ("audio", "video", "early", "late_eq", "late_w", "late_stack"):
                z = np.array([r[k] for r in rows])
                print(f"    {k:12s} {z.mean():+.4f} ± {z.std():.4f}")
            d = np.array([r["early"] - r["late_w"] for r in rows])
            print(f"    early - late_w = {d.mean():+.4f}   为正 {int((d>0).sum())}/{N_SEEDS}")


if __name__ == "__main__":
    main()
