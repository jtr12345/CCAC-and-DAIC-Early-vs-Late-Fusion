# -*- coding: utf-8 -*-
"""混杂检验：音视频特征是否编码了学校/班级身份，从而借校际标签差异"刷"出汇总信号。"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fusion_probe import (HERE, N_CLASS, LEVEL_IDX, MODALITIES, qwk,
                           group_folds, prep, softmax_reg, assemble)
from _capacity_matched import score_feats

sys.stdout.reconfigure(encoding="utf-8")


def multiclass_cv(X, label, groups, n_folds=5, l2=0.1, seed=0):
    """按班级分组 CV 预测 label（label 可以是学校/班级）。"""
    classes = sorted(set(label.tolist()))
    ci = {c: i for i, c in enumerate(classes)}
    y = np.array([ci[v] for v in label])
    folds = group_folds(groups, n_folds, seed=seed)
    p = np.zeros(len(y), int)
    # 复用 softmax_reg 需要 N_CLASS=5，这里手写 K 类
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        A, B = prep(X[tr], X[te], k=min(32, X.shape[1]) if X.shape[1] > 36 else X.shape[1])
        K = len(classes)
        Y = np.eye(K)[y[tr]]
        n, d = A.shape
        W = np.zeros((d, K), np.float32); b = np.zeros(K, np.float32)
        mW = np.zeros_like(W); vW = np.zeros_like(W); mb = np.zeros_like(b); vb = np.zeros_like(b)
        for t in range(1, 401):
            Z = A @ W + b; Z -= Z.max(1, keepdims=True)
            P = np.exp(Z); P /= P.sum(1, keepdims=True)
            G = (P - Y) / n
            gW = A.T @ G + l2 * W; gb = G.sum(0)
            for prm, g, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
                m *= .9; m += .1 * g; v *= .999; v += .001 * g * g
                prm -= 0.03 * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
        Z = B @ W + b; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        p[te] = P.argmax(1)
    return np.mean(p == y), K, y, p


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"), encoding="utf-8-sig")))
    lab = {(r["anon_school"], r["anon_class"], r["anon_person"]): r for r in rows}
    per_mod, subjects = assemble(dict(MODALITIES))
    y4 = np.array([LEVEL_IDX[lab[s]["t4_anxiety_level"]] for s in subjects])
    y3 = np.array([LEVEL_IDX[lab[s]["t3_anxiety_level"]] for s in subjects])
    schools = np.array([s[0] for s in subjects])
    groups = np.array([s[1] for s in subjects])
    S, A, V = score_feats(subjects), per_mod["audio_basic"], per_mod["video_clip"]

    # 排除 n=1 的 SCH_005
    ok = schools != "SCH_005"
    print("=== 能否从特征识别「学校」？(班级分组 CV) ===")
    for name, X in (("score", S), ("audio", A), ("video", V)):
        acc, K, _, _ = multiclass_cv(X[ok], schools[ok], groups[ok])
        print(f"{name:8s} 学校识别 acc={acc:.3f}  (7 类 chance≈{1/K:.3f}, "
              f"多数类基线={np.max(np.bincount(np.unique(schools[ok], return_inverse=True)[1]))/ok.sum():.3f})")

    print("\n=== 能否从特征识别「班级」？(班级分组 CV，k 会很大会受限，作参考) ===")
    cl = np.array([s[1] for s in subjects])
    for name, X in (("audio", A), ("video", V)):
        acc, K, _, _ = multiclass_cv(X[ok], cl[ok], groups[ok])
        print(f"{name:8s} 班级识别 acc={acc:.3f} (K={K}, chance={1/K:.3f})")

    # ---------- 校际混杂的量化 ----------
    print("\n=== 校际混杂量化（n>=50 的学校）===")
    keep = [s for s in sorted(set(schools)) if (schools == s).sum() >= 50]
    print(f"{'school':10s} {'n':>4s} {'音视频pred均值':>14s} {'y4均值':>8s} {'y3均值':>8s}")
    rows_out = []
    for s in keep:
        ii = schools == s
        # 用跨校 CV 得到的视频单模态预测均值（这里简化为特征一阶统计与标签的关联）
        rows_out.append((s, int(ii.sum()), float(A[ii].mean()), float(V[ii].mean()),
                         float(y4[ii].mean()), float(y3[ii].mean())))
        print(f"{s:10s} {ii.sum():4d} {A[ii].mean():14.3f} {y4[ii].mean():8.3f} {y3[ii].mean():8.3f}")

    v = np.array([[r[2], r[3], r[4]] for r in rows_out])
    from itertools import combinations
    def pear(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"\n校均值相关：audio特征~y4  r={pear(v[:,0], v[:,2]):+.3f}  "
          f"video特征~y4  r={pear(v[:,1], v[:,2]):+.3f}")
    print("（学校样本少，仅为方向性参考；真正的证据是上面「按校分层后 QWK 归零」。）")


if __name__ == "__main__":
    main()
