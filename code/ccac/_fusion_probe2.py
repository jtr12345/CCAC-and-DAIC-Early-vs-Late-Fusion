# -*- coding: utf-8 -*-
"""第二轮：在分数历史基线上检验音视频的增量价值，以及 early/late 融合差异。"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fusion_probe import (HERE, N_CLASS, LEVELS, LEVEL_IDX, TIMEPOINTS, MODALITIES,
                           scores, group_folds, prep, softmax_reg, mlp,
                           assemble, score_history)

sys.stdout.reconfigure(encoding="utf-8")


def late_from_probs(probs, mode="soft"):
    P = np.stack(probs, axis=0)          # (m, n, C)
    if mode == "soft":
        return P.mean(0)
    votes = P.argmax(2).T                 # (n, m)
    out = np.zeros((votes.shape[0], N_CLASS))
    for i, v in enumerate(votes):
        out[i] = np.bincount(v, minlength=N_CLASS)
    return out


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"), encoding="utf-8-sig")))
    label = {(r["anon_school"], r["anon_class"], r["anon_person"]): LEVEL_IDX[r["t4_anxiety_level"]]
             for r in rows}
    per_mod, subjects = assemble(dict(MODALITIES))
    y = np.array([label[s] for s in subjects])
    groups = np.array([s[1] for s in subjects])
    folds = group_folds(groups, 5)

    S = score_history(subjects)          # 18 dims
    A = per_mod["audio_basic"]           # 216
    V = per_mod["video_clip"]            # 3072

    t3 = np.array([LEVEL_IDX[[r for r in rows
                              if (r['anon_school'], r['anon_class'], r['anon_person']) == s][0]
                            ['t3_anxiety_level']] for s in subjects])

    print(f"n={len(y)}  类别分布={np.bincount(y, minlength=N_CLASS)}")
    print(f"多数类基线: acc={np.mean(y == 0):.4f}")

    # ---- 逐折训练，收集 OOF ----
    res, prob = {}, {}

    def add(name, p):
        prob[name] = p
        res[name] = p.argmax(1)

    P_score = np.zeros((len(y), N_CLASS), np.float32)
    P_sa_add = np.zeros_like(P_score); P_sa_nl = np.zeros_like(P_score)
    P_sav_add = np.zeros_like(P_score); P_sav_nl = np.zeros_like(P_score)
    P_audio = np.zeros_like(P_score); P_video = np.zeros_like(P_score)

    for f in range(5):
        tr, te = folds != f, folds == f
        # 分数：不做 PCA
        Ss_tr, Ss_te = prep(S[tr], S[te], k=S.shape[1])
        Ps = softmax_reg(Ss_tr, y[tr], Ss_te, l2=0.03)
        P_score[te] = Ps

        Aa_tr, Aa_te = prep(A[tr], A[te], k=32)
        Vv_tr, Vv_te = prep(V[tr], V[te], k=32)
        P_audio[te] = softmax_reg(Aa_tr, y[tr], Aa_te)
        P_video[te] = softmax_reg(Vv_tr, y[tr], Vv_te)

        # score + audio
        X1 = np.hstack([Ss_tr, Aa_tr]); X2 = np.hstack([Ss_te, Aa_te])
        P_sa_add[te] = softmax_reg(X1, y[tr], X2, l2=0.05)
        P_sa_nl[te] = mlp(X1, y[tr], X2)

        # score + audio + video
        X1 = np.hstack([Ss_tr, Aa_tr, Vv_tr]); X2 = np.hstack([Ss_te, Aa_te, Vv_te])
        P_sav_add[te] = softmax_reg(X1, y[tr], X2, l2=0.05)
        P_sav_nl[te] = mlp(X1, y[tr], X2)

    add("score_only", P_score)
    add("audio_only", P_audio)
    add("video_only", P_video)
    add("early score+audio (linear)", P_sa_add)
    add("early score+audio (MLP)", P_sa_nl)
    add("early score+audio+video (linear)", P_sav_add)
    add("early score+audio+video (MLP)", P_sav_nl)
    add("late score/audio (soft)", late_from_probs([P_score, P_audio], "soft"))
    add("late score/audio (hard)", late_from_probs([P_score, P_audio], "hard"))
    add("late score/audio/video (soft)", late_from_probs([P_score, P_audio, P_video], "soft"))
    add("late score/audio/video (hard)", late_from_probs([P_score, P_audio, P_video], "hard"))
    res["persistence(t3)"] = t3

    print(f"\n{'model':38s} {'acc':>7s} {'mf1':>7s} {'qwk':>7s} {'mae':>7s}")
    for k, p in res.items():
        s = scores(y, p)
        print(f"{k:38s} {s['acc']:7.3f} {s['mf1']:7.3f} {s['qwk']:7.3f} {s['mae']:7.3f}")

    # ---- 增量检验：配对 bootstrap（按班级整块重采样）----
    print("\n=== 相对 score_only 的 QWK 增量（bootstrap 95% CI，按班级重采样）===")
    rng = np.random.RandomState(0)
    cls_all = np.array(sorted(set(groups)))
    idx_by_cls = {c: np.where(groups == c)[0] for c in cls_all}

    def boot_delta(pA, pB, n_boot=400):
        d = []
        for _ in range(n_boot):
            pick = rng.choice(cls_all, size=len(cls_all), replace=True)
            ii = np.concatenate([idx_by_cls[c] for c in pick])
            d.append(qwk_local(y[ii], pA[ii]) - qwk_local(y[ii], pB[ii]))
        return np.percentile(d, [2.5, 97.5]), np.mean(d)

    from _fusion_probe import qwk as qwk_local
    for k in res:
        if k in ("score_only", "persistence(t3)"):
            continue
        ci, m = boot_delta(res[k], res["score_only"])
        print(f"{k:38s} ΔQWK={m:+.4f}  95%CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]")

    np.savez(os.path.join(HERE, "_probe2_oof.npz"), y=y, folds=folds, t3=t3,
             **{f"P::{k}": v for k, v in prob.items()})


if __name__ == "__main__":
    main()
