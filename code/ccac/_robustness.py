# -*- coding: utf-8 -*-
"""稳健性：①按学校分层评估 ②特征健全性 ③音视频是否只反映当前状态(y3)而非变化。"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fusion_probe import (HERE, N_CLASS, LEVEL_IDX, MODALITIES, qwk,
                           group_folds, prep, softmax_reg, assemble)
from _capacity_matched import score_feats

sys.stdout.reconfigure(encoding="utf-8")


def fit_predict(Xtr, ytr, Xte, l2=0.05):
    A, B = prep(Xtr, Xte, k=min(32, Xtr.shape[1]) if Xtr.shape[1] > 36 else Xtr.shape[1])
    return softmax_reg(A, ytr, B, l2=l2).argmax(1)


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"), encoding="utf-8-sig")))
    lab = {(r["anon_school"], r["anon_class"], r["anon_person"]): r for r in rows}
    per_mod, subjects = assemble(dict(MODALITIES))
    y4 = np.array([LEVEL_IDX[lab[s]["t4_anxiety_level"]] for s in subjects])
    y3 = np.array([LEVEL_IDX[lab[s]["t3_anxiety_level"]] for s in subjects])
    groups = np.array([s[1] for s in subjects])
    schools = np.array([s[0] for s in subjects])
    folds = group_folds(groups, 5, seed=0)
    S, A, V = score_feats(subjects), per_mod["audio_basic"], per_mod["video_clip"]

    # ---------- ① 特征健全性 ----------
    print("=== 特征健全性 ===")
    for name, X in (("score", S), ("audio", A), ("video", V)):
        bad = int(np.sum(~np.isfinite(X)))
        const = int(np.sum(X.std(0) < 1e-8))
        print(f"{name:8s} shape={X.shape} 非有限值={bad} 常数列={const} "
              f"全局std={X.std():.4f}")

    # ---------- ② 同一模型分别预测 y3 与 y4 ----------
    print("\n=== 同一特征分别预测 T3（当前）与 T4（未来）===")
    print(f"{'target':8s} {'model':10s} {'QWK':>8s} {'acc':>8s}")
    for tname, yy in (("y3", y3), ("y4", y4)):
        for mname, X in (("score", S), ("audio", A), ("video", V)):
            p = np.zeros(len(yy), int)
            for f in range(5):
                tr, te = folds != f, folds == f
                p[te] = fit_predict(X[tr], yy[tr], X[te])
            print(f"{tname:8s} {mname:10s} {qwk(yy, p):8.4f} {np.mean(p == yy):8.4f}")

    # ---------- ③ 按学校分层（排除 n<50）----------
    print("\n=== 按学校分层 QWK（只用 n>=50 的学校）===")
    keep = [s for s in sorted(set(schools)) if (schools == s).sum() >= 50]
    models = {}
    for mname, X in (("score", S), ("audio", A), ("video", V),
                     ("early", np.hstack([S, A, V]))):
        p = np.zeros(len(y4), int)
        for f in range(5):
            tr, te = folds != f, folds == f
            pp = prep(X[tr], X[te], k=min(32, X.shape[1]))
            p[te] = softmax_reg(pp[0], y4[tr], pp[1], l2=0.05).argmax(1)
        models[mname] = p
    models["persistence"] = y3

    print(f"{'model':14s}" + "".join(f"{s[-3:]:>8s}" for s in keep) + f"{'学校均值':>10s}")
    for k, p in models.items():
        vals = [qwk(y4[schools == s], p[schools == s]) for s in keep]
        print(f"{k:14s}" + "".join(f"{v:8.3f}" for v in vals) + f"{np.mean(vals):10.3f}")

    # ---------- ④ 音视频能否预测"变化"（在有大方差的学校内）----------
    print("\n=== 变化预测 AUC（排除 SCH_006 这类近常数学校后）===")
    var_schools = [s for s in keep if len(set(y4[schools == s].tolist())) >= 3]
    mask = np.isin(schools, var_schools)
    print(f"纳入学校: {var_schools}  n={mask.sum()}")
    yb = (y4 > y3).astype(int)
    print(f"恶化正例: {int(yb[mask].sum())}/{int(mask.sum())} ({yb[mask].mean()*100:.1f}%)")
    for mname, X in (("score", S), ("audio", A), ("video", V)):
        p = np.zeros(len(y4), int)
        for f in range(5):
            tr, te = folds != f, folds == f
            pp = prep(X[tr], X[te], k=min(32, X.shape[1]))
            p[te] = softmax_reg(pp[0], y4[tr], pp[1], l2=0.05).argmax(1)
        print(f"  {mname:8s} 预测 y4 的 QWK(限定学校)={qwk(y4[mask], p[mask]):+.4f}")


if __name__ == "__main__":
    main()
