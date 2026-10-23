# -*- coding: utf-8 -*-
"""基线校准：给分数模型和变化模型充分调参，确认"打不过持续性"不是欠拟合造成。"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fusion_probe import (HERE, N_CLASS, LEVEL_IDX, MODALITIES, qwk,
                           group_folds, prep, assemble)
from _capacity_matched import score_feats
from _change_model import ridge, apply_delta

sys.stdout.reconfigure(encoding="utf-8")


def softmax_reg_cw(Xtr, ytr, Xte, l2=0.05, epochs=500, lr=0.03, class_weight=True, seed=0):
    n, d = Xtr.shape
    Y = np.eye(N_CLASS)[ytr]
    if class_weight:
        cnt = np.bincount(ytr, minlength=N_CLASS).astype(float)
        cw = np.where(cnt > 0, n / (N_CLASS * np.maximum(cnt, 1)), 0.0)
        sw = cw[ytr][:, None]
    else:
        sw = 1.0
    rng = np.random.RandomState(seed)
    W = rng.randn(d, N_CLASS).astype(np.float32) * 0.01
    b = np.zeros(N_CLASS, np.float32)
    mW = np.zeros_like(W); vW = np.zeros_like(W); mb = np.zeros_like(b); vb = np.zeros_like(b)
    for t in range(1, epochs + 1):
        Z = Xtr @ W + b; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) * sw / n
        gW = Xtr.T @ G + l2 * W; gb = G.sum(0)
        for prm, g, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
            m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
            prm -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    Z = Xte @ W + b; Z -= Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    return P


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"), encoding="utf-8-sig")))
    lab = {(r["anon_school"], r["anon_class"], r["anon_person"]): r for r in rows}
    per_mod, subjects = assemble(dict(MODALITIES))
    y = np.array([LEVEL_IDX[lab[s]["t4_anxiety_level"]] for s in subjects])
    y3 = np.array([LEVEL_IDX[lab[s]["t3_anxiety_level"]] for s in subjects])
    groups = np.array([s[1] for s in subjects])
    schools = np.array([s[0] for s in subjects])
    folds = group_folds(groups, 5, seed=0)
    S = score_feats(subjects)
    delta = (y - y3).astype(np.float32)

    print(f"持续性 QWK={qwk(y, y3):.4f}\n")

    print("=== 分数模型：l2 × 类别加权 ===")
    print(f"{'l2':>8s} {'cw':>6s} {'QWK':>8s} {'acc':>8s}")
    for l2 in (0.001, 0.01, 0.03, 0.1, 0.3, 1.0):
        for cw in (True, False):
            p = np.zeros(len(y), int)
            for f in range(5):
                tr, te = folds != f, folds == f
                A, B = prep(S[tr], S[te], k=S.shape[1])
                p[te] = softmax_reg_cw(A, y[tr], B, l2=l2, class_weight=cw).argmax(1)
            print(f"{l2:8.3f} {str(cw):>6s} {qwk(y, p):8.4f} {np.mean(p == y):8.4f}")

    print("\n=== 变化量 ridge：λ 扫描（λ 越大越接近持续性）===")
    print(f"{'lam':>8s} {'QWK':>8s} {'acc':>8s} {'MAE(Δ)':>8s}")
    for lam in (1, 10, 50, 200, 1000, 5000):
        dh = np.zeros(len(y), np.float32)
        for f in range(5):
            tr, te = folds != f, folds == f
            A, B = prep(S[tr], S[te], k=S.shape[1])
            dh[te] = ridge(A, delta[tr], B, lam)
        p = apply_delta(y3, dh)
        print(f"{lam:8.0f} {qwk(y, p):8.4f} {np.mean(p == y):8.4f} {np.mean(np.abs(delta - dh)):8.4f}")

    print("\n=== 各校 t3→t4 结构与基线 ===")
    print(f"{'school':10s} {'n':>5s} {'t4分布':>28s} {'persist QWK':>12s} {'t3唯一值':>10s}")
    for sc in sorted(set(schools)):
        ii = schools == sc
        dist = np.bincount(y[ii], minlength=N_CLASS).tolist()
        print(f"{sc:10s} {ii.sum():5d} {str(dist):>28s} {qwk(y[ii], y3[ii]):12.4f} "
              f"{len(set(y3[ii].tolist())):10d}")

    print("\n=== 各校 t3 等级分布（看是否被截断）===")
    for sc in sorted(set(schools)):
        ii = schools == sc
        print(f"  {sc}: " + str({int(k): int(v) for k, v in zip(*np.unique(y3[ii], return_counts=True))}))


if __name__ == "__main__":
    main()
