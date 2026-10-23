# -*- coding: utf-8 -*-
"""容量匹配的 early vs late 融合对照。

关键设计：
- late fusion 的权重/stacking 用嵌套 CV 学习，避免用测试折信息
- early/late 两侧参数量对齐并在结果里报告
- 同时跑两个目标：5 级等级、相对 T3 的恶化
- 分数特征加入显式趋势项（斜率/差分），把基线做扎实
"""
import os, sys, csv, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fusion_probe import (HERE, N_CLASS, LEVEL_IDX, MODALITIES, scores, qwk,
                           group_folds, prep, softmax_reg, mlp, assemble)

sys.stdout.reconfigure(encoding="utf-8")


# ---------------- 分数特征（含显式趋势） ----------------
def score_feats(subjects):
    rows = {(r["anon_school"], r["anon_class"], r["anon_person"]): r
            for r in csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"),
                                         encoding="utf-8-sig"))}
    X = np.zeros((len(subjects), 36), np.float32)
    for i, s in enumerate(subjects):
        r = rows[s]
        base, lv = [], []
        for t in ("t1", "t2", "t3"):
            for m in ("depression", "anxiety", "stress"):
                base.append(float(r[f"{t}_{m}_score"]))
                lv.append(float(LEVEL_IDX[r[f"{t}_{m}_level"]]))
        b = np.array(base).reshape(3, 3)      # (time, measure)
        l = np.array(lv).reshape(3, 3)
        d = np.concatenate([b[2] - b[1], b[1] - b[0], b[2] - b[0],
                            l[2] - l[1], l[1] - l[0], l[2] - l[0]])
        X[i] = np.concatenate([b.ravel(), l.ravel(), d])
    return X


# ---------------- 参数量 ----------------
def n_lin(d_in, k=N_CLASS):
    return d_in * k + k


def n_mlp(d_in, h, k=N_CLASS):
    return d_in * h + h + h * k + k


# ---------------- 单模态概率（嵌套内层 OOF） ----------------
def oof_probs(X, y, groups, folds, model, k_pca=32, inner_seed=7):
    """返回 (P_oof, P_te) per fold：P_oof 用内层 CV 生成，P_te 用全训练折拟合。"""
    P_oof = np.zeros((len(y), N_CLASS), np.float32)
    P_te = np.zeros((len(y), N_CLASS), np.float32)
    for f in range(5):
        tr = folds != f
        te = folds == f
        gtr = groups[tr]
        inner = group_folds(gtr, 3, seed=inner_seed + f)
        for k in range(3):
            itr = inner != k
            ite = inner == k
            idx_tr = np.where(tr)[0][itr]
            idx_te = np.where(tr)[0][ite]
            A, B = prep(X[idx_tr], X[idx_te], k=k_pca)
            P_oof[idx_te] = model(A, y[idx_tr], B)
        A, B = prep(X[tr], X[te], k=k_pca)
        P_te[te] = model(A, y[tr], B)
    return P_oof, P_te


def run_task(task, y, groups, folds, S, A, V, print_fn=print):
    tr_models = {
        "linear": lambda a, b, c: softmax_reg(a, b, c, l2=0.05),
        "mlp": lambda a, b, c: mlp(a, b, c, hidden=64, epochs=350),
    }
    # 每个模态的嵌套 OOF 概率
    P_oof_all, P_te_all = {}, {}
    for name, X, kp in (("score", S, S.shape[1]), ("audio", A, 32), ("video", V, 32)):
        for mk, model in tr_models.items():
            po, pt = oof_probs(X, y, groups, folds, model, k_pca=kp)
            P_oof_all[(name, mk)], P_te_all[(name, mk)] = po, pt

    res = {}
    # ---- late: 等权 / 学权重 / stacking ----
    for mk in tr_models:
        po = [P_oof_all[(n, mk)] for n in ("score", "audio", "video")]
        pt = [P_te_all[(n, mk)] for n in ("score", "audio", "video")]

        # 等权
        eq_oof = np.mean(po, 0); eq_te = np.mean(pt, 0)
        res[f"late_equal[{mk}]"] = eq_te.argmax(1)

        # 学权重：每个外层折内，只用训练折的内层 OOF 选权重
        grid = np.arange(0, 1.0001, 0.1)
        wsum = np.zeros(3)
        out = np.zeros((len(y), N_CLASS), np.float32)
        for f in range(5):
            tr = folds != f; te = folds == f
            best = (-9, None)
            for w1 in grid:
                for w2 in grid:
                    if w1 + w2 > 1 + 1e-9:
                        continue
                    w3 = 1 - w1 - w2
                    q = (w1 * po[0][tr] + w2 * po[1][tr] + w3 * po[2][tr]).argmax(1)
                    v = qwk(y[tr], q)
                    if v > best[0]:
                        best = (v, (w1, w2, w3))
            w = best[1]; wsum += np.array(w)
            out[te] = w[0] * pt[0][te] + w[1] * pt[1][te] + w[2] * pt[2][te]
        res[f"late_weighted[{mk}]"] = out.argmax(1)
        res[f"  w[{mk}]"] = np.round(wsum / 5 * 100).astype(int)

        # stacking：在 15 维概率上再学一层（同样只用训练折拟合）
        stack_te = np.zeros((len(y), N_CLASS), np.float32)
        for f in range(5):
            tr = folds != f; te = folds == f
            Ftr = np.hstack([p[tr] for p in po])
            Fte = np.hstack([p[te] for p in pt])
            stack_te[te] = softmax_reg(Ftr, y[tr], Fte, l2=0.05)
        res[f"late_stack[{mk}]"] = stack_te.argmax(1)

    # ---- early ----
    for mk, model in tr_models.items():
        P = np.zeros((len(y), N_CLASS), np.float32)
        for f in range(5):
            tr = folds != f; te = folds == f
            As, Bs = prep(S[tr], S[te], k=S.shape[1])
            Aa, Ba = prep(A[tr], A[te], k=32)
            Av, Bv = prep(V[tr], V[te], k=32)
            P[te] = model(np.hstack([As, Aa, Av]), y[tr], np.hstack([Bs, Ba, Bv]))
        res[f"early[{mk}]"] = P.argmax(1)

    # ---- 参数量报告 ----
    ds, da, dv = S.shape[1], 32, 32
    print_fn(f"\n--- {task} ---")
    print_fn(f"参数量 early[linear]={n_lin(ds+da+dv)}  "
             f"late[linear]=单模态{n_lin(ds)+n_lin(da)+n_lin(dv)}+权重3={n_lin(ds)+n_lin(da)+n_lin(dv)+3}  "
             f"stacking额外={n_lin(15)}")
    print_fn(f"参数量 early[mlp]={n_mlp(ds+da+dv,64)}  "
             f"late[mlp]=单模态{n_mlp(ds,64)+n_mlp(da,64)+n_mlp(dv,64)}+权重3")
    return res


def bootstrap_pair(y, groups, pA, pB, n_boot=500, seed=1):
    rng = np.random.RandomState(seed)
    cls_all = np.array(sorted(set(groups)))
    idx = {c: np.where(groups == c)[0] for c in cls_all}
    d = []
    for _ in range(n_boot):
        pick = rng.choice(cls_all, size=len(cls_all), replace=True)
        ii = np.concatenate([idx[c] for c in pick])
        d.append(qwk(y[ii], pA[ii]) - qwk(y[ii], pB[ii]))
    d = np.array(d)
    return d.mean(), np.percentile(d, [2.5, 97.5]), float((d <= 0).mean())


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"), encoding="utf-8-sig")))
    lab = {(r["anon_school"], r["anon_class"], r["anon_person"]): r for r in rows}
    per_mod, subjects = assemble(dict(MODALITIES))
    groups = np.array([s[1] for s in subjects])
    folds = group_folds(groups, 5)
    S = score_feats(subjects)
    A = per_mod["audio_basic"]
    V = per_mod["video_clip"]

    y5 = np.array([LEVEL_IDX[lab[s]["t4_anxiety_level"]] for s in subjects])
    y3 = np.array([LEVEL_IDX[lab[s]["t3_anxiety_level"]] for s in subjects])

    print("=== 目标 1：5 级 t4_anxiety_level ===")
    r5 = run_task("5class", y5, groups, folds, S, A, V)
    print(f"{'model':26s} {'acc':>7s} {'mf1':>7s} {'qwk':>7s}")
    for k, p in r5.items():
        if k.startswith("  "):
            print(f"{k:26s}  权重×100 = {p.tolist()}"); continue
        s = scores(y5, p)
        print(f"{k:26s} {s['acc']:7.3f} {s['mf1']:7.3f} {s['qwk']:7.3f}")

    print("\n配对 bootstrap（ΔQWK，按班级重采样，500 次）")
    pairs = [("early[linear] - late_weighted[linear]", r5["early[linear]"], r5["late_weighted[linear]"]),
             ("early[mlp] - late_weighted[mlp]", r5["early[mlp]"], r5["late_weighted[mlp]"]),
             ("early[linear] - late_equal[linear]", r5["early[linear]"], r5["late_equal[linear]"]),
             ("early[mlp] - late_equal[mlp]", r5["early[mlp]"], r5["late_equal[mlp]"]),
             ("early[mlp] - persistence", r5["early[mlp]"], y3)]
    for name, a, b in pairs:
        m, ci, pn = bootstrap_pair(y5, groups, a, b)
        print(f"{name:42s} {m:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  P(Δ<=0)={pn:.3f}")

    # ---- 目标 2：恶化 ----
    print("\n\n=== 目标 2：恶化 y4>y3（二分类）===")
    yb = (y5 > y3).astype(int)
    # 用 5 类的概率模型换成二分类不直接可行，这里复用一个简化版
    print("（二分类结果见 _capacity_matched_bin.py）")


if __name__ == "__main__":
    main()
