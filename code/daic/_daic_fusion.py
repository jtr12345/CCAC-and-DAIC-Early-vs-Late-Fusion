# -*- coding: utf-8 -*-
"""DAIC-WOZ 融合策略实验（单站点，无校际混杂）。

目标：PHQ 严重度 5 级（QWK）
对照：容量匹配的 early / late / stacking，重复分层 5 折 CV
"""
import os, sys, json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(ROOT, "_cache")
SEED = 0
K = 5
N_CLASS = 5
FEATS = ["audio_egemaps", "video_openface", "audio_bow_egemaps", "video_vgg16"]


def load():
    X = {f: np.load(os.path.join(C, f"daic_{f}.npy")) for f in FEATS}
    z = np.load(os.path.join(C, "daic_labels.npz"), allow_pickle=True)
    keep = z["row_index"]
    return {f: X[f][keep] for f in FEATS}, z


def strat_folds(y, n_folds=5, seed=0):
    rng = np.random.RandomState(seed)
    folds = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


def qwk(y, p, k=5):
    O = np.zeros((k, k))
    for a, b in zip(y, p):
        O[a, b] += 1
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den else 0.0


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = y == 1; npos, nneg = p.sum(), (~p).sum()
    return float((r[p].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def prep(Xtr, Xte, k=32):
    """标准化 + PCA(k)。d>n 时走 Gram 矩阵，避免 d×d 特征分解。"""
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-8] = 1.0
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    if A.shape[1] > k:
        n = len(A)
        if A.shape[1] > n:
            G = A @ A.T / max(n - 1, 1)
            w, U = np.linalg.eigh(G)
            idx = np.argsort(w)[::-1][:k]
            w, U = w[idx], U[:, idx]
            V = A.T @ U / np.sqrt(np.maximum(w, 1e-12) * max(n - 1, 1))
        else:
            Cv = A.T @ A / max(n - 1, 1)
            w, V = np.linalg.eigh(Cv)
            V = V[:, np.argsort(w)[::-1][:k]]
        A = A @ V; B = B @ V
        s = A.std(0); s[s < 1e-8] = 1.0
        A /= s; B /= s
    return A.astype(np.float32), B.astype(np.float32)


def softmax_reg(Xtr, ytr, Xte, l2=0.05, epochs=400, lr=0.03, seed=SEED):
    n, d = Xtr.shape
    Y = np.eye(N_CLASS)[ytr]
    cnt = np.bincount(ytr, minlength=N_CLASS).astype(float)
    cw = np.where(cnt > 0, n / (N_CLASS * np.maximum(cnt, 1)), 0.0)
    sw = cw[ytr][:, None]
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
            m *= .9; m += .1 * g; v *= .999; v += .001 * g * g
            prm -= lr * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
    Z = Xte @ W + b; Z -= Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    return P


def mlp(Xtr, ytr, Xte, h=48, l2=1e-2, epochs=350, lr=0.01, seed=SEED):
    n, d = Xtr.shape
    Y = np.eye(N_CLASS)[ytr]
    cnt = np.bincount(ytr, minlength=N_CLASS).astype(float)
    cw = np.where(cnt > 0, n / (N_CLASS * np.maximum(cnt, 1)), 0.0)
    sw = cw[ytr][:, None]
    rng = np.random.RandomState(seed)
    P1 = (rng.randn(d, h) * np.sqrt(2 / d)).astype(np.float32); b1 = np.zeros(h, np.float32)
    P2 = (rng.randn(h, N_CLASS) * np.sqrt(2 / h)).astype(np.float32); b2 = np.zeros(N_CLASS, np.float32)
    ps = [P1, b1, P2, b2]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    for t in range(1, epochs + 1):
        H = np.maximum(Xtr @ P1 + b1, 0)
        Z = H @ P2 + b2; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) * sw / n
        gP2 = H.T @ G + l2 * P2; gb2 = G.sum(0)
        GH = G @ P2.T; GH[H <= 0] = 0
        gP1 = Xtr.T @ GH + l2 * P1; gb1 = GH.sum(0)
        for prm, g, m, v in zip(ps, [gP1, gb1, gP2, gb2], ms, vs):
            m *= .9; m += .1 * g; v *= .999; v += .001 * g * g
            prm -= lr * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
    H = np.maximum(Xte @ P1 + b1, 0)
    Z = H @ P2 + b2; Z -= Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    return P


def run_once(y, Xa, Xv, folds, model):
    n = len(y)
    P = {k: np.zeros((n, N_CLASS), np.float32) for k in
         ("audio", "video", "early", "late_eq", "late_w", "late_stack")}
    wa = []
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        Aa, Ba = prep(Xa[tr], Xa[te], 32)
        Av, Bv = prep(Xv[tr], Xv[te], 32)
        pa = model(Aa, y[tr], Ba); pv = model(Av, y[tr], Bv)
        pe = model(np.hstack([Aa, Av]), y[tr], np.hstack([Ba, Bv]))
        P["audio"][te], P["video"][te], P["early"][te] = pa, pv, pe
        P["late_eq"][te] = (pa + pv) / 2
        # 训练折内选权重（内采样预测）
        pa_tr = model(Aa, y[tr], Aa); pv_tr = model(Av, y[tr], Av)
        best = (-9, None)
        for w in np.arange(0, 1.001, 0.1):
            q = (w * pa_tr + (1 - w) * pv_tr).argmax(1)
            v = qwk(y[tr], q)
            if v > best[0]:
                best = (v, w)
        w = best[1]; wa.append(w)
        P["late_w"][te] = w * pa + (1 - w) * pv
        # stacking：15 维概率（这里只有 10 维，两类器）
        Ftr = np.hstack([pa_tr, pv_tr]); Fte = np.hstack([pa, pv])
        P["late_stack"][te] = softmax_reg(Ftr, y[tr], Fte, l2=0.1)
    return P, np.mean(wa)


def main():
    X, z = load()
    y = z["severity"].astype(int)
    print(f"n={len(y)}  分布={np.bincount(y, minlength=5)}")

    pairs = {
        "handcrafted: audio_egemaps + video_openface": ("audio_egemaps", "video_openface"),
        "deep-ish: audio_bow + video_vgg16": ("audio_bow_egemaps", "video_vgg16"),
    }
    results = {}
    for pname, (fa, fv) in pairs.items():
        for mname, model in (("linear", lambda a, b, c: softmax_reg(a, b, c, l2=0.1)),
                             ("mlp", mlp)):
            qs = {k: [] for k in ("audio", "video", "early", "late_eq", "late_w", "late_stack")}
            ws = []
            for seed in range(5):
                folds = strat_folds(y, 5, seed)
                P, w = run_once(y, X[fa], X[fv], folds, model)
                ws.append(w)
                for k, pp in P.items():
                    qs[k].append(qwk(y, pp.argmax(1)))
            print(f"\n--- {pname} [{mname}] ---")
            for k, v in qs.items():
                v = np.array(v)
                print(f"  {k:12s} QWK={v.mean():.4f}±{v.std():.4f}")
            print(f"  late 学到的 audio 权重={np.mean(ws):.2f}")
            results[f"{pname}|{mname}"] = {k: (float(np.mean(v)), float(np.std(v)))
                                           for k, v in qs.items()}

    json.dump(results, open(os.path.join(ROOT, "_daic_fusion_results.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
