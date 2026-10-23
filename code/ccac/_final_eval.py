# -*- coding: utf-8 -*-
"""最终可靠性评估：重复 GroupKFold + 置换检验 + 按学校稳健性。

模型全部线性且容量对齐（约 500 参数），避免容量混淆。
目标：t4_anxiety_level（5 级有序）。
"""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fusion_probe import (HERE, N_CLASS, LEVEL_IDX, MODALITIES, scores, qwk,
                           group_folds, prep, softmax_reg, assemble)
from _capacity_matched import score_feats

sys.stdout.reconfigure(encoding="utf-8")
SEEDS = [0, 1, 2, 3, 4]
N_PERM = 1000


def eval_models(y, groups, folds, S, A, V):
    n = len(y)
    P = {k: np.zeros((n, N_CLASS), np.float32) for k in
         ("score", "audio", "video", "early", "late_equal", "late_weighted")}
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        Ss = prep(S[tr], S[te], k=S.shape[1])
        Aa = prep(A[tr], A[te], k=32)
        Vv = prep(V[tr], V[te], k=32)
        ps = softmax_reg(Ss[0], y[tr], Ss[1], l2=0.05)
        pa = softmax_reg(Aa[0], y[tr], Aa[1], l2=0.05)
        pv = softmax_reg(Vv[0], y[tr], Vv[1], l2=0.05)
        pe = softmax_reg(np.hstack([Ss[0], Aa[0], Vv[0]]), y[tr],
                         np.hstack([Ss[1], Aa[1], Vv[1]]), l2=0.05)
        P["score"][te], P["audio"][te], P["video"][te], P["early"][te] = ps, pa, pv, pe
        P["late_equal"][te] = (ps + pa + pv) / 3
        # 训练折内选权重（3 个参数）
        ps_tr = softmax_reg(Ss[0], y[tr], Ss[0], l2=0.05)
        pa_tr = softmax_reg(Aa[0], y[tr], Aa[0], l2=0.05)
        pv_tr = softmax_reg(Vv[0], y[tr], Vv[0], l2=0.05)
        best = (-9, None)
        for w1 in np.arange(0, 1.001, 0.1):
            for w2 in np.arange(0, 1.001 - w1 + 1e-9, 0.1):
                w3 = 1 - w1 - w2
                q = (w1 * ps_tr + w2 * pa_tr + w3 * pv_tr).argmax(1)
                v = qwk(y[tr], q)
                if v > best[0]:
                    best = (v, (w1, w2, w3))
        w = best[1]
        P["late_weighted"][te] = w[0] * ps + w[1] * pa + w[2] * pv
    return {k: v.argmax(1) for k, v in P.items()}, P


def perm_test(y, pred, n_perm=N_PERM, seed=0):
    """置换预测与标签的配对，保留预测的边际分布。"""
    rng = np.random.RandomState(seed)
    obs = qwk(y, pred)
    null = np.array([qwk(rng.permutation(y), pred) for _ in range(n_perm)])
    return obs, float((null >= obs).mean()), float(null.mean()), float(null.std())


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "train", "labels.csv"), encoding="utf-8-sig")))
    lab = {(r["anon_school"], r["anon_class"], r["anon_person"]): r for r in rows}
    per_mod, subjects = assemble(dict(MODALITIES))
    y = np.array([LEVEL_IDX[lab[s]["t4_anxiety_level"]] for s in subjects])
    y3 = np.array([LEVEL_IDX[lab[s]["t3_anxiety_level"]] for s in subjects])
    S, A, V = score_feats(subjects), per_mod["audio_basic"], per_mod["video_clip"]

    for gmode, groups in (
            ("按班级", np.array([s[1] for s in subjects])),
            ("按学校", np.array([s[0] for s in subjects]))):
        print(f"\n{'='*74}\n分组方式：{gmode}   groups={len(set(groups))}\n{'='*74}")
        acc = {k: [] for k in ("persistence", "score", "audio", "video",
                               "early", "late_equal", "late_weighted")}
        pooled = None
        for seed in SEEDS:
            folds = group_folds(groups, 5, seed=seed)
            preds, P = eval_models(y, groups, folds, S, A, V)
            preds["persistence"] = y3
            for k in acc:
                acc[k].append(scores(y, preds[k]))
            if pooled is None:
                pooled = preds

        print(f"{'model':16s} {'QWK mean±sd':>20s} {'acc mean±sd':>20s} {'mf1 mean±sd':>20s}")
        base = np.array([d["qwk"] for d in acc["persistence"]])
        for k, ds in acc.items():
            q = np.array([d["qwk"] for d in ds]); a = np.array([d["acc"] for d in ds])
            m = np.array([d["mf1"] for d in ds])
            print(f"{k:16s} {q.mean():8.4f}±{q.std():.4f}      "
                  f"{a.mean():8.4f}±{a.std():.4f}      {m.mean():8.4f}±{m.std():.4f}")
        print(f"\n{'相对 persistence 的 ΔQWK（逐折配对）':<40s} mean     sd    {len(SEEDS)} 次中为正")
        for k, ds in acc.items():
            if k == "persistence":
                continue
            d = np.array([x["qwk"] for x in ds]) - base
            print(f"{k:40s} {d.mean():+.4f} {d.std():.4f}   {int((d > 0).sum())}/{len(d)}")

        print("\n置换检验（预测 vs 标签配对打乱，1000 次）")
        for k, p in pooled.items():
            obs, pv, mu, sd = perm_test(y, p)
            sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "n.s."))
            print(f"  {k:16s} QWK={obs:+.4f}  零分布均值={mu:+.4f}(sd {sd:.4f})  p={pv:.4f} {sig}")

        if gmode == "按班级":
            print("\n按学校拆分 QWK（池化 OOF）")
            schools = np.array([s[0] for s in subjects])
            for k in ("score", "early", "late_weighted", "persistence"):
                line = []
                for sc in sorted(set(schools)):
                    ii = schools == sc
                    line.append(f"{sc}:{qwk(y[ii], pooled[k][ii]):+.3f}")
                print(f"  {k:16s} " + "  ".join(line))

    json.dump({}, open(os.path.join(HERE, "_final_eval_flag.json"), "w"))


if __name__ == "__main__":
    main()
