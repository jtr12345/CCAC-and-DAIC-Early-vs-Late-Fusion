# -*- coding: utf-8 -*-
"""CCAC 融合策略探针：单模态基线 / 加性 vs 非线性 / late fusion。

划分：GroupKFold，group = anon_class（同班学生不跨折，避免泄漏）
目标：t4_anxiety_level（5 级有序）
"""
import os, sys, json, itertools
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache")
SEED = 0
N_CLASS = 5
LEVELS = ["正常", "轻度", "中度", "重度", "非常严重"]
LEVEL_IDX = {v: i for i, v in enumerate(LEVELS)}

MODALITIES = {
    "audio_basic": "audio_basic",
    "audio_wavlm": "audio_wavlm_base",
    "video_clip": "video_clip_base",
    "video_dinov2": "video_dinov2_base",
}
TIMEPOINTS = ["T1", "T2", "T3"]


# ---------------- 指标 ----------------
def macro_f1(y, p):
    f = []
    for c in range(N_CLASS):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f)) if f else 0.0


def qwk(y, p):
    O = np.zeros((N_CLASS, N_CLASS))
    for a, b in zip(y, p):
        O[a, b] += 1
    w = np.array([[(i - j) ** 2 / (N_CLASS - 1) ** 2 for j in range(N_CLASS)]
                  for i in range(N_CLASS)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den else 0.0


def scores(y, p):
    return dict(acc=float(np.mean(y == p)), mf1=macro_f1(y, p), qwk=qwk(y, p),
                mae=float(np.mean(np.abs(y - p))))


# ---------------- 分组切分 ----------------
def group_folds(groups, n_folds=5, seed=SEED):
    uniq = np.array(sorted(set(groups)))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


# ---------------- 预处理 ----------------
def pca_fit(X, k):
    n = X.shape[0]
    mu = X.mean(0)
    Xc = X - mu
    if Xc.shape[1] > Xc.shape[0]:
        G = Xc @ Xc.T / max(n - 1, 1)
        w, U = np.linalg.eigh(G)
        idx = np.argsort(w)[::-1][:k]
        w, U = w[idx], U[:, idx]
        V = Xc.T @ U / np.sqrt(np.maximum(w, 1e-12) * max(n - 1, 1))
    else:
        C = Xc.T @ Xc / max(n - 1, 1)
        w, V = np.linalg.eigh(C)
        idx = np.argsort(w)[::-1][:k]
        V = V[:, idx]
    return mu, V.astype(np.float32)


def prep(Xtr, Xte, k=32):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    if A.shape[1] > k:
        pm, V = pca_fit(A, k)
        A = (A - pm) @ V
        B = (B - pm) @ V
        s2 = A.std(0)
        s2[s2 < 1e-8] = 1.0
        A, B = A / s2, B / s2
    return A.astype(np.float32), B.astype(np.float32)


# ---------------- 模型（numpy） ----------------
def softmax_reg(Xtr, ytr, Xte, l2=1e-2, epochs=400, lr=0.03, seed=SEED):
    n, d = Xtr.shape
    Y = np.eye(N_CLASS)[ytr]
    cnt = np.bincount(ytr, minlength=N_CLASS).astype(np.float64)
    cw = np.where(cnt > 0, n / (N_CLASS * np.maximum(cnt, 1)), 0.0)
    sw = cw[ytr][:, None]
    rng = np.random.RandomState(seed)
    W = rng.randn(d, N_CLASS).astype(np.float32) * 0.01
    b = np.zeros(N_CLASS, dtype=np.float32)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    for t in range(1, epochs + 1):
        Z = Xtr @ W + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) * sw / n
        gW = Xtr.T @ G + l2 * W
        gb = G.sum(0)
        for prm, g, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            prm -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    Z = Xte @ W + b
    Z -= Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    return P


def mlp(Xtr, ytr, Xte, hidden=64, l2=1e-2, epochs=500, lr=0.01, seed=SEED):
    n, d = Xtr.shape
    Y = np.eye(N_CLASS)[ytr]
    cnt = np.bincount(ytr, minlength=N_CLASS).astype(np.float64)
    cw = np.where(cnt > 0, n / (N_CLASS * np.maximum(cnt, 1)), 0.0)
    sw = cw[ytr][:, None]
    rng = np.random.RandomState(seed)
    P1 = (rng.randn(d, hidden) * np.sqrt(2.0 / d)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    P2 = (rng.randn(hidden, N_CLASS) * np.sqrt(2.0 / hidden)).astype(np.float32)
    b2 = np.zeros(N_CLASS, dtype=np.float32)
    ps = [P1, b1, P2, b2]
    ms = [np.zeros_like(p) for p in ps]
    vs = [np.zeros_like(p) for p in ps]
    for t in range(1, epochs + 1):
        H = np.maximum(Xtr @ P1 + b1, 0)
        Z = H @ P2 + b2
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) * sw / n
        gP2 = H.T @ G + l2 * P2
        gb2 = G.sum(0)
        GH = G @ P2.T
        GH[H <= 0] = 0
        gP1 = Xtr.T @ GH + l2 * P1
        gb1 = GH.sum(0)
        for prm, g, m, v in zip(ps, [gP1, gb1, gP2, gb2], ms, vs):
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            prm -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    H = np.maximum(Xte @ P1 + b1, 0)
    Z = H @ P2 + b2
    Z -= Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    return P


# ---------------- 数据装配 ----------------
def load_cache(name):
    z = np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=True)
    return z["X"], z["keys"]


def assemble(feats):
    """feats: {modality: cache_name} -> (X_by_subject, keys_subject)"""
    per_mod = {}
    key_index = None
    for mod, name in feats.items():
        X, keys = load_cache(name)
        idx = {}
        for i, k in enumerate(keys):
            idx[(k[0], k[1], k[2], k[3])] = i
        if key_index is None:
            subs = sorted({(k[0], k[1], k[2]) for k in keys})
            key_index = subs
            ri = {s: i for i, s in enumerate(subs)}
        M = np.zeros((len(key_index), X.shape[1] * len(TIMEPOINTS)), dtype=np.float32)
        for k, i in idx.items():
            s = (k[0], k[1], k[2])
            if k[3] not in TIMEPOINTS or s not in ri:
                continue
            j = TIMEPOINTS.index(k[3])
            M[ri[s], j * X.shape[1]:(j + 1) * X.shape[1]] = X[i]
        per_mod[mod] = M
    return per_mod, key_index


def score_history(subjects):
    import csv
    rows = {}
    for r in csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"), encoding="utf-8-sig")):
        rows[(r["anon_school"], r["anon_class"], r["anon_person"])] = r
    X = np.zeros((len(subjects), 18), dtype=np.float32)
    for i, s in enumerate(subjects):
        r = rows[s]
        v = []
        for t in TIMEPOINTS:
            for m in ("depression", "anxiety", "stress"):
                v.append(float(r[f"{t.lower()}_{m}_score"]))
                v.append(float(LEVEL_IDX[r[f"{t.lower()}_{m}_level"]]))
        X[i] = v
    return X


# ---------------- 主流程 ----------------
def run():
    import csv
    rows = list(csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"), encoding="utf-8-sig")))
    label = {(r["anon_school"], r["anon_class"], r["anon_person"]): LEVEL_IDX[r["t4_anxiety_level"]]
             for r in rows}

    feats = dict(MODALITIES)
    per_mod, subjects = assemble(feats)
    y = np.array([label[s] for s in subjects])
    groups = np.array([s[1] for s in subjects])
    folds = group_folds(groups, 5)
    print(f"n={len(y)}  classes={len(set(groups))}  folds={np.bincount(folds)}")
    print(f"标签分布: {np.bincount(y, minlength=N_CLASS)}")

    # 持续性基线
    t3 = np.array([LEVEL_IDX[[r for r in rows
                              if (r['anon_school'], r['anon_class'], r['anon_person']) == s][0]
                            ['t3_anxiety_level']] for s in subjects])
    print("\n[baseline] predict t4 = t3 :", {k: round(v, 4) for k, v in scores(y, t3).items()})

    all_mods = dict(per_mod)
    all_mods["score_history"] = score_history(subjects)

    # ---------- Step 1: 单模态 ----------
    print("\n=== Step 1  单模态基线 (GroupKFold by class) ===")
    print(f"{'modality':16s} {'d_raw':>6s} {'raw acc':>8s} {'raw mf1':>8s} {'raw qwk':>8s} "
          f"{'pca32 acc':>9s} {'pca32 mf1':>9s} {'pca32 qwk':>9s}")
    oof = {}
    for mod, X in all_mods.items():
        res_raw = np.zeros(len(y), dtype=int)
        res_pca = np.zeros(len(y), dtype=int)
        prob_pca = np.zeros((len(y), N_CLASS), dtype=np.float32)
        for f in range(5):
            tr, te = folds != f, folds == f
            # raw（高维只做标准化，强 L2）
            A, B = prep(X[tr], X[te], k=X.shape[1])
            res_raw[te] = softmax_reg(A, y[tr], B, l2=0.05).argmax(1)
            # pca32
            A, B = prep(X[tr], X[te], k=32)
            P = softmax_reg(A, y[tr], B)
            prob_pca[te] = P
            res_pca[te] = P.argmax(1)
        sr, sp = scores(y, res_raw), scores(y, res_pca)
        print(f"{mod:16s} {X.shape[1]:6d} {sr['acc']:8.3f} {sr['mf1']:8.3f} {sr['qwk']:8.3f} "
              f"{sp['acc']:9.3f} {sp['mf1']:9.3f} {sp['qwk']:9.3f}")
        oof[mod] = dict(raw=res_raw, pca=res_pca, prob=prob_pca)

    # 残差相关
    print("\n残差相关 (pca32 预测 - 真值):")
    mods = list(all_mods)
    R = np.array([y - oof[m]["pca"] for m in mods], dtype=float)
    C = np.corrcoef(R)
    print(f"{'':16s}" + "".join(f"{m[:10]:>11s}" for m in mods))
    for i, m in enumerate(mods):
        print(f"{m:16s}" + "".join(f"{C[i, j]:11.2f}" for j in range(len(mods))))

    # ---------- Step 2: 融合 ----------
    print("\n=== Step 2  融合对照（音频=audio_basic, 视频=video_clip_base）===")
    pair = ["audio_basic", "video_clip"]
    Xa, Xv = all_mods[pair[0]], all_mods[pair[1]]

    # early additive / nonlinear
    pe_add = np.zeros((len(y), N_CLASS), dtype=np.float32)
    pe_nl = np.zeros((len(y), N_CLASS), dtype=np.float32)
    for f in range(5):
        tr, te = folds != f, folds == f
        Aa, Ba = prep(Xa[tr], Xa[te], k=32)
        Av, Bv = prep(Xv[tr], Xv[te], k=32)
        A = np.hstack([Aa, Av]); B = np.hstack([Ba, Bv])
        pe_add[te] = softmax_reg(A, y[tr], B)
        pe_nl[te] = mlp(A, y[tr], B)

    # late fusion: soft 平均 / hard 投票
    p1, p2 = oof[pair[0]]["prob"], oof[pair[1]]["prob"]
    late_soft = (p1 + p2) / 2
    late_hard = np.zeros_like(late_soft)
    votes = np.stack([p1.argmax(1), p2.argmax(1)], axis=1)
    for i, v in enumerate(votes):
        late_hard[i] = np.bincount(v, minlength=N_CLASS)
    results = {
        "early_add(linear)": pe_add.argmax(1),
        "early_nonlinear(MLP)": pe_nl.argmax(1),
        "late_soft_equal": late_soft.argmax(1),
        "late_hard_vote": late_hard.argmax(1),
        "unimodal_audio": oof[pair[0]]["pca"],
        "unimodal_video": oof[pair[1]]["pca"],
        "score_history": oof["score_history"]["pca"],
        "persistence(t3)": t3,
    }
    print(f"{'model':24s} {'acc':>7s} {'mf1':>7s} {'qwk':>7s} {'mae':>7s}")
    for k, p in results.items():
        s = scores(y, p)
        print(f"{k:24s} {s['acc']:7.3f} {s['mf1']:7.3f} {s['qwk']:7.3f} {s['mae']:7.3f}")

    np.savez(os.path.join(HERE, "_probe_oof.npz"),
             y=y, folds=folds, t3=t3,
             **{f"{k}__{name}": oof[k][name] for k in oof for name in ("raw", "pca", "prob")})
    json.dump({k: ({kk: round(vv, 4) for kk, vv in scores(y, p).items()})
               for k, p in results.items()},
              open(os.path.join(HERE, "_probe_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nwrote _probe_oof.npz, _probe_results.json")


if __name__ == "__main__":
    run()
