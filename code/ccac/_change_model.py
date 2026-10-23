# -*- coding: utf-8 -*-
"""变化量建模：预测 Δ = y4 - y3，再叠加到 T3 等级上。

好处：预测 Δ=0 即等价于持续性基线，模型不会比"抄 T3"更差。
用途：检验多模态信号是否藏在变化里，同时做容量匹配的 early/late 对照。
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fusion_probe import (HERE, N_CLASS, LEVEL_IDX, MODALITIES, qwk,
                           group_folds, prep, assemble)
from _capacity_matched import score_feats

sys.stdout.reconfigure(encoding="utf-8")


def ridge(Xtr, ytr, Xte, lam=10.0):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-8] = 1.0
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    ym = ytr.mean()
    d = A.shape[1]
    w = np.linalg.solve(A.T @ A + lam * np.eye(d), A.T @ (ytr - ym))
    return B @ w + ym


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = y == 1; npos, nneg = p.sum(), (~p).sum()
    return float((r[p].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def apply_delta(y3, dh, clip_round=True):
    d = np.clip(dh, -3, 3)
    if clip_round:
        d = np.round(d)
    return np.clip(y3 + d, 0, N_CLASS - 1).astype(int)


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"), encoding="utf-8-sig")))
    lab = {(r["anon_school"], r["anon_class"], r["anon_person"]): r for r in rows}
    per_mod, subjects = assemble(dict(MODALITIES))
    groups = np.array([s[1] for s in subjects])
    folds = group_folds(groups, 5)

    y4 = np.array([LEVEL_IDX[lab[s]["t4_anxiety_level"]] for s in subjects])
    y3 = np.array([LEVEL_IDX[lab[s]["t3_anxiety_level"]] for s in subjects])
    delta = (y4 - y3).astype(np.float32)
    S, A, V = score_feats(subjects), per_mod["audio_basic"], per_mod["video_clip"]

    print(f"n={len(y4)}  Δ分布: " + str({int(k): int(v) for k, v in zip(*np.unique(delta, return_counts=True))}))
    print(f"持续性 QWK={qwk(y4, y3):.4f}  acc={np.mean(y4 == y3):.4f}")

    def run(lam=20.0):
        dh = {k: np.zeros(len(y4), np.float32) for k in
              ("score", "audio", "video", "early", "late_equal", "late_weighted")}
        wsum = np.zeros(3)
        for f in range(5):
            tr, te = folds != f, folds == f
            Ss = prep(S[tr], S[te], k=S.shape[1])
            Aa = prep(A[tr], A[te], k=32)
            Vv = prep(V[tr], V[te], k=32)
            ds = ridge(Ss[0], delta[tr], Ss[1], lam)
            da = ridge(Aa[0], delta[tr], Aa[1], lam)
            dv = ridge(Vv[0], delta[tr], Vv[1], lam)
            de = ridge(np.hstack([Ss[0], Aa[0], Vv[0]]), delta[tr],
                       np.hstack([Ss[1], Aa[1], Vv[1]]), lam)
            dh["score"][te], dh["audio"][te], dh["video"][te], dh["early"][te] = ds, da, dv, de
            dh["late_equal"][te] = (ds + da + dv) / 3
            # 权重：用训练折的拟合内预测选（3 个参数，过拟合风险低）
            ds_tr = ridge(Ss[0], delta[tr], Ss[0], lam)
            da_tr = ridge(Aa[0], delta[tr], Aa[0], lam)
            dv_tr = ridge(Vv[0], delta[tr], Vv[0], lam)
            best = (-9, None)
            for w1 in np.arange(0, 1.001, 0.1):
                for w2 in np.arange(0, 1.001 - w1 + 1e-9, 0.1):
                    w3 = 1 - w1 - w2
                    pred = apply_delta(y3[tr], w1 * ds_tr + w2 * da_tr + w3 * dv_tr)
                    v = qwk(y4[tr], pred)
                    if v > best[0]:
                        best = (v, (w1, w2, w3))
            w = best[1]; wsum += np.array(w)
            dh["late_weighted"][te] = w[0] * ds + w[1] * da + w[2] * dv
        return dh, wsum / 5

    dh, w = run()
    print(f"\n{'model (Δ→level)':26s} {'QWK':>7s} {'acc':>7s} {'MAE(Δ)':>8s} {'AUC恶化':>8s}")
    results = {"persistence": y3}
    for k in dh:
        results[k] = apply_delta(y3, dh[k])
    for k, p in results.items():
        aucv = auc((y4 > y3).astype(int), dh[k]) if k in dh else float("nan")
        print(f"{k:26s} {qwk(y4, p):7.4f} {np.mean(p == y4):7.4f} "
              f"{np.mean(np.abs(delta - dh[k])) if k in dh else 0.0:8.4f} {aucv:8.4f}")
    print(f"权重×100 = {np.round(w * 100).astype(int).tolist()}")

    # bootstrap: early vs late_weighted
    rng = np.random.RandomState(3)
    cls = np.array(sorted(set(groups)))
    idx = {c: np.where(groups == c)[0] for c in cls}

    def boot(pa, pb, n=500):
        d = []
        for _ in range(n):
            pick = rng.choice(cls, size=len(cls), replace=True)
            ii = np.concatenate([idx[c] for c in pick])
            d.append(qwk(y4[ii], pa[ii]) - qwk(y4[ii], pb[ii]))
        d = np.array(d)
        return d.mean(), np.percentile(d, [2.5, 97.5]), (d <= 0).mean()

    print("\n配对 bootstrap（ΔQWK）")
    for name, a, b in [("early - late_weighted", results["early"], results["late_weighted"]),
                       ("early - late_equal", results["early"], results["late_equal"]),
                       ("early - persistence", results["early"], y3),
                       ("late_weighted - persistence", results["late_weighted"], y3)]:
        m, ci, pn = boot(a, b)
        print(f"{name:30s} {m:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  P(Δ<=0)={pn:.3f}")


if __name__ == "__main__":
    main()
