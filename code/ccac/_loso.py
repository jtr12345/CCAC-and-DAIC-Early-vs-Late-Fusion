# -*- coding: utf-8 -*-
"""CCAC 正式收口：leave-one-school-out（LOSO），排除 n=1 的 SCH_005。"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fusion_probe import (HERE, N_CLASS, LEVEL_IDX, MODALITIES, qwk,
                           prep, softmax_reg, assemble)
from _capacity_matched import score_feats

sys.stdout.reconfigure(encoding="utf-8")


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"), encoding="utf-8-sig")))
    lab = {(r["anon_school"], r["anon_class"], r["anon_person"]): r for r in rows}
    per_mod, subjects = assemble(dict(MODALITIES))
    y4 = np.array([LEVEL_IDX[lab[s]["t4_anxiety_level"]] for s in subjects])
    y3 = np.array([LEVEL_IDX[lab[s]["t3_anxiety_level"]] for s in subjects])
    schools = np.array([s[0] for s in subjects])
    S, A, V = score_feats(subjects), per_mod["audio_basic"], per_mod["video_clip"]

    keep = [s for s in sorted(set(schools)) if (schools == s).sum() >= 50]
    print(f"纳入学校 {keep}  n={np.isin(schools, keep).sum()}")
    mask_all = np.isin(schools, keep)

    preds = {k: np.zeros(len(y4), int) for k in
             ("score", "audio", "video", "early", "late_equal", "late_weighted")}
    wrec = []
    for sc in keep:
        te = schools == sc
        tr = np.isin(schools, [s for s in keep if s != sc])
        Ss = prep(S[tr], S[te], k=S.shape[1])
        Aa = prep(A[tr], A[te], k=32)
        Vv = prep(V[tr], V[te], k=32)
        ps = softmax_reg(Ss[0], y4[tr], Ss[1], l2=0.05)
        pa = softmax_reg(Aa[0], y4[tr], Aa[1], l2=0.05)
        pv = softmax_reg(Vv[0], y4[tr], Vv[1], l2=0.05)
        pe = softmax_reg(np.hstack([Ss[0], Aa[0], Vv[0]]), y4[tr],
                         np.hstack([Ss[1], Aa[1], Vv[1]]), l2=0.05)
        preds["score"][te] = ps.argmax(1)
        preds["audio"][te] = pa.argmax(1)
        preds["video"][te] = pv.argmax(1)
        preds["early"][te] = pe.argmax(1)
        preds["late_equal"][te] = ((ps + pa + pv) / 3).argmax(1)
        ps_tr = softmax_reg(Ss[0], y4[tr], Ss[0], l2=0.05)
        pa_tr = softmax_reg(Aa[0], y4[tr], Aa[0], l2=0.05)
        pv_tr = softmax_reg(Vv[0], y4[tr], Vv[0], l2=0.05)
        best = (-9, None)
        for w1 in np.arange(0, 1.001, 0.1):
            for w2 in np.arange(0, 1.001 - w1 + 1e-9, 0.1):
                w3 = 1 - w1 - w2
                q = (w1 * ps_tr + w2 * pa_tr + w3 * pv_tr).argmax(1)
                v = qwk(y4[tr], q)
                if v > best[0]:
                    best = (v, (w1, w2, w3))
        w = np.array(best[1]); wrec.append(w)
        preds["late_weighted"][te] = (w[0] * ps + w[1] * pa + w[2] * pv).argmax(1)

    preds["persistence"] = y3

    print(f"\n{'model':16s}" + "".join(f"{s[-3:]:>7s}" for s in keep) + f"{'均值':>8s}{'池化':>8s}")
    for k, p in preds.items():
        vals = [qwk(y4[schools == s], p[schools == s]) for s in keep]
        print(f"{k:16s}" + "".join(f"{v:7.3f}" for v in vals) +
              f"{np.mean(vals):8.3f}{qwk(y4[mask_all], p[mask_all]):8.3f}")
    print(f"\nLOSO 学到的平均权重：score={np.mean([w[0] for w in wrec]):.2f} "
          f"audio={np.mean([w[1] for w in wrec]):.2f} video={np.mean([w[2] for w in wrec]):.2f}")

    # 相对 persistence 的逐校差值
    print(f"\n{'相对 persistence 的逐校 ΔQWK':<30s}" + "".join(f"{s[-3:]:>7s}" for s in keep) + f"{'均值':>8s}")
    for k in ("score", "audio", "video", "early", "late_weighted"):
        d = [qwk(y4[schools == s], preds[k][schools == s]) -
             qwk(y4[schools == s], y3[schools == s]) for s in keep]
        print(f"{k:<30s}" + "".join(f"{v:+7.3f}" for v in d) + f"{np.mean(d):+8.3f}")


if __name__ == "__main__":
    main()
