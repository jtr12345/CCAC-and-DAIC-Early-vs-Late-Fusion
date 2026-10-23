# -*- coding: utf-8 -*-
"""第三轮：序数回归基线 + 加权 late fusion 上界 + 交互效应 bootstrap 检验。"""
import os, sys, csv, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fusion_probe import (HERE, N_CLASS, LEVEL_IDX, TIMEPOINTS, MODALITIES,
                           scores, qwk, group_folds, prep, softmax_reg, mlp,
                           assemble, score_history)

sys.stdout.reconfigure(encoding="utf-8")


def ordinal_logreg(Xtr, ytr, Xte, l2=0.03, epochs=400, lr=0.03):
    """P(y>k) for k=0..3，再转成 5 类概率。保留序信息，适合有序目标。"""
    n = Xtr.shape[0]
    pk = np.zeros((Xte.shape[0], N_CLASS - 1), np.float32)
    for k in range(N_CLASS - 1):
        t = (ytr > k).astype(np.float32)
        w = np.zeros(Xtr.shape[1], np.float32); b = 0.0
        mw = np.zeros_like(w); vw = np.zeros_like(w); mb = vb = 0.0
        pos = max(t.sum(), 1.0)
        cw = np.where(t > 0, n / (2 * pos), n / (2 * max(n - pos, 1.0)))
        for it in range(1, epochs + 1):
            z = Xtr @ w + b
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            g = (p - t) * cw / n
            gw = Xtr.T @ g + l2 * w
            gb = g.sum()
            for prm, gr, m, v in ((w, gw, mw, vw), (None, gb, None, None)):
                if prm is None:
                    mb = 0.9 * mb + 0.1 * gr; vb = 0.999 * vb + 0.001 * gr * gr
                    b -= lr * (mb / (1 - 0.9 ** it)) / (np.sqrt(vb / (1 - 0.999 ** it)) + 1e-8)
                else:
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    prm -= lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
        pk[:, k] = 1 / (1 + np.exp(-np.clip(Xte @ w + b, -30, 30)))
    # 单调化
    pk = np.minimum.accumulate(pk, axis=1)
    P = np.zeros((Xte.shape[0], N_CLASS), np.float32)
    P[:, 0] = 1 - pk[:, 0]
    for k in range(1, N_CLASS - 1):
        P[:, k] = pk[:, k - 1] - pk[:, k]
    P[:, N_CLASS - 1] = pk[:, N_CLASS - 2]
    P = np.clip(P, 1e-6, None)
    return P / P.sum(1, keepdims=True)


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"), encoding="utf-8-sig")))
    label = {(r["anon_school"], r["anon_class"], r["anon_person"]): LEVEL_IDX[r["t4_anxiety_level"]]
             for r in rows}
    per_mod, subjects = assemble(dict(MODALITIES))
    y = np.array([label[s] for s in subjects])
    groups = np.array([s[1] for s in subjects])
    folds = group_folds(groups, 5)
    S, A, V = score_history(subjects), per_mod["audio_basic"], per_mod["video_clip"]

    P_ords = np.zeros((len(y), N_CLASS), np.float32)
    for f in range(5):
        tr, te = folds != f, folds == f
        a, b = prep(S[tr], S[te], k=S.shape[1])
        P_ords[te] = ordinal_logreg(a, y[tr], b)

    t3 = np.array([LEVEL_IDX[[r for r in rows
                              if (r['anon_school'], r['anon_class'], r['anon_person']) == s][0]
                            ['t3_anxiety_level']] for s in subjects])

    z = np.load(os.path.join(HERE, "_probe2_oof.npz"), allow_pickle=True)
    P = {k.split("::", 1)[1]: z[k] for k in z.files if k.startswith("P::")}

    print(f"{'model':40s} {'acc':>7s} {'mf1':>7s} {'qwk':>7s} {'mae':>7s}")
    show = {
        "persistence(t3)": t3,
        "score_only (softmax)": P["score_only"].argmax(1),
        "score_only (ordinal)": P_ords.argmax(1),
    }
    for k, p in show.items():
        s = scores(y, p); print(f"{k:40s} {s['acc']:7.3f} {s['mf1']:7.3f} {s['qwk']:7.3f} {s['mae']:7.3f}")

    # ---- 加权 late fusion 上界（权重在 OOF 上网格搜，乐观上界）----
    ps, pa, pv = P["score_only"], P["audio_only"], P["video_only"]
    best = (-9, None)
    grid = np.arange(0, 1.0001, 0.05)
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1 + 1e-9:
                continue
            w3 = 1 - w1 - w2
            q = ((w1 * ps + w2 * pa + w3 * pv).argmax(1))
            v = qwk(y, q)
            if v > best[0]:
                best = (v, (round(w1, 2), round(w2, 2), round(w3, 2)))
    # score+audio only
    best2 = (-9, None)
    for w1 in grid:
        w2 = 1 - w1
        q = ((w1 * ps + w2 * pa).argmax(1))
        v = qwk(y, q)
        if v > best2[0]:
            best2 = (v, (round(w1, 2), round(w2, 2)))
    print(f"\n加权 late 上界 score+audio          : QWK={best2[0]:.4f}  w={best2[1]}")
    print(f"加权 late 上界 score+audio+video    : QWK={best[0]:.4f}  w={best[1]}")

    # ---- 交互效应检验：同输入下 MLP - linear ----
    rng = np.random.RandomState(1)
    cls_all = np.array(sorted(set(groups)))
    idx_by_cls = {c: np.where(groups == c)[0] for c in cls_all}

    def boot_pair(pA, pB, n_boot=500):
        d = []
        for _ in range(n_boot):
            pick = rng.choice(cls_all, size=len(cls_all), replace=True)
            ii = np.concatenate([idx_by_cls[c] for c in pick])
            d.append(qwk(y[ii], pA[ii]) - qwk(y[ii], pB[ii]))
        d = np.array(d)
        return d.mean(), np.percentile(d, [2.5, 97.5]), (d <= 0).mean()

    pairs = [
        ("interaction: sav MLP - linear", P["early score+audio+video (MLP)"].argmax(1),
         P["early score+audio+video (linear)"].argmax(1)),
        ("interaction: sa  MLP - linear", P["early score+audio (MLP)"].argmax(1),
         P["early score+audio (linear)"].argmax(1)),
        ("early(MLP) - weighted late upper bound", P["early score+audio+video (MLP)"].argmax(1), None),
    ]
    print("\n=== bootstrap 配对检验（按班级重采样，500 次）===")
    for name, pA, pB in pairs:
        if pB is None:
            continue
        m, ci, pneg = boot_pair(pA, pB)
        print(f"{name:42s} ΔQWK={m:+.4f}  95%CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  P(Δ<=0)={pneg:.3f}")

    # 与加权 late 上界比
    wv = best[1]
    p_late_best = (wv[0] * ps + wv[1] * pa + wv[2] * pv).argmax(1)
    m, ci, pneg = boot_pair(P["early score+audio+video (MLP)"].argmax(1), p_late_best)
    print(f"{'early(MLP) - weighted late(best)':42s} ΔQWK={m:+.4f}  95%CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  P(Δ<=0)={pneg:.3f}")

    # 与 persistence 比
    m, ci, pneg = boot_pair(P["early score+audio+video (MLP)"].argmax(1), t3)
    print(f"{'early(MLP) - persistence':42s} ΔQWK={m:+.4f}  95%CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  P(Δ<=0)={pneg:.3f}")


if __name__ == "__main__":
    main()
