# -*- coding: utf-8 -*-
"""第四轮：把目标从「预测 T4 等级」改成「预测相对 T3 的恶化」，检验多模态信号是否藏在变化量里。"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fusion_probe import (HERE, LEVEL_IDX, MODALITIES, group_folds, prep,
                           assemble, score_history)

sys.stdout.reconfigure(encoding="utf-8")


def logistic(Xtr, ytr, Xte, l2=0.05, epochs=400, lr=0.03):
    n = Xtr.shape[0]
    w = np.zeros(Xtr.shape[1], np.float32); b = 0.0
    mw = np.zeros_like(w); vw = np.zeros_like(w); mb = vb = 0.0
    pos = max(ytr.sum(), 1.0)
    cw = np.where(ytr > 0, n / (2 * pos), n / (2 * max(n - pos, 1.0)))
    for it in range(1, epochs + 1):
        p = 1 / (1 + np.exp(-np.clip(Xtr @ w + b, -30, 30)))
        g = (p - ytr) * cw / n
        gw, gb = Xtr.T @ g + l2 * w, g.sum()
        mw = 0.9 * mw + 0.1 * gw; vw = 0.999 * vw + 0.001 * gw * gw
        w -= lr * (mw / (1 - 0.9 ** it)) / (np.sqrt(vw / (1 - 0.999 ** it)) + 1e-8)
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb * gb
        b -= lr * (mb / (1 - 0.9 ** it)) / (np.sqrt(vb / (1 - 0.999 ** it)) + 1e-8)
    return 1 / (1 + np.exp(-np.clip(Xte @ w + b, -30, 30)))


def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s), float); r[order] = np.arange(1, len(s) + 1)
    pos = y == 1; npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"), encoding="utf-8-sig")))
    lab = {(r["anon_school"], r["anon_class"], r["anon_person"]): r for r in rows}
    per_mod, subjects = assemble(dict(MODALITIES))
    y4 = np.array([LEVEL_IDX[lab[s]["t4_anxiety_level"]] for s in subjects])
    y3 = np.array([LEVEL_IDX[lab[s]["t3_anxiety_level"]] for s in subjects])
    groups = np.array([s[1] for s in subjects])
    folds = group_folds(groups, 5)
    S, A, V = score_history(subjects), per_mod["audio_basic"], per_mod["video_clip"]

    for name, tgt in [("恶化 y4>y3", (y4 > y3).astype(np.float32)),
                      ("任何变化 y4!=y3", (y4 != y3).astype(np.float32))]:
        print(f"\n=== 目标：{name}   正例 {int(tgt.sum())}/{len(tgt)} ({tgt.mean()*100:.1f}%) ===")
        preds = {k: np.zeros(len(tgt), np.float32)
                 for k in ("score", "audio", "video", "early_add", "early_mlp")}
        for f in range(5):
            tr, te = folds != f, folds == f
            a_s, b_s = prep(S[tr], S[te], k=S.shape[1])
            a_a, b_a = prep(A[tr], A[te], k=32)
            a_v, b_v = prep(V[tr], V[te], k=32)
            preds["score"][te] = logistic(a_s, tgt[tr], b_s)
            preds["audio"][te] = logistic(a_a, tgt[tr], b_a)
            preds["video"][te] = logistic(a_v, tgt[tr], b_v)
            X1 = np.hstack([a_s, a_a, a_v]); X2 = np.hstack([b_s, b_a, b_v])
            preds["early_add"][te] = logistic(X1, tgt[tr], X2)
            # 显式交互项：音频×视频 低维交叉
            inter_tr = np.einsum("ij,ik->ijk", a_a[:, :8], a_v[:, :8]).reshape(tr.sum(), -1)
            inter_te = np.einsum("ij,ik->ijk", b_a[:, :8], b_v[:, :8]).reshape(te.sum(), -1)
            preds["early_mlp"][te] = logistic(np.hstack([X1, inter_tr]), tgt[tr],
                                              np.hstack([X2, inter_te]))
        for k, p in preds.items():
            print(f"  {k:10s} AUC={auc(tgt, p):.4f}")
        print(f"  {'chance':10s} AUC=0.5000")


if __name__ == "__main__":
    main()
