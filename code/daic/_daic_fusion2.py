# -*- coding: utf-8 -*-
"""DAIC-WOZ 改进版：嵌套 CV 学权重 + 两个目标 + 10 次重复。

目标1：PHQ 严重度 5 级（QWK）
目标2：PHQ 二分类 >=10（AUC）
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _daic_fusion import (ROOT, C, load, strat_folds, qwk, auc, prep,
                          softmax_reg, mlp, N_CLASS)

sys.stdout.reconfigure(encoding="utf-8")
N_SEEDS = 10


def logistic(Xtr, ytr, Xte, l2=0.05, epochs=400, lr=0.03):
    n, d = Xtr.shape
    w = np.zeros(d, np.float32); b = 0.0
    mw = np.zeros_like(w); vw = np.zeros_like(w); mb = vb = 0.0
    pos = max(ytr.sum(), 1.0)
    cw = np.where(ytr > 0, n / (2 * pos), n / (2 * max(n - pos, 1.0)))
    for t in range(1, epochs + 1):
        p = 1 / (1 + np.exp(-np.clip(Xtr @ w + b, -30, 30)))
        g = (p - ytr) * cw / n
        gw, gb = Xtr.T @ g + l2 * w, g.sum()
        mw = .9 * mw + .1 * gw; vw = .999 * vw + .001 * gw * gw
        w -= lr * (mw / (1 - .9 ** t)) / (np.sqrt(vw / (1 - .999 ** t)) + 1e-8)
        mb = .9 * mb + .1 * gb; vb = .999 * vb + .001 * gb * gb
        b -= lr * (mb / (1 - .9 ** t)) / (np.sqrt(vb / (1 - .999 ** t)) + 1e-8)
    return 1 / (1 + np.exp(-np.clip(Xte @ w + b, -30, 30)))


def make_models(task):
    if task == "severity":
        return {"linear": lambda a, b, c: softmax_reg(a, b, c, l2=0.1),
                "mlp": lambda a, b, c: mlp(a, b, c, h=48, epochs=350)}
    return {"linear": lambda a, b, c: logistic(a, b, c, l2=0.1),
            "mlp": None}


def decide(P):
    return P.argmax(1) if P.ndim == 2 else (P >= 0.5).astype(int)


def metric(task, y, p, score=None):
    if task == "severity":
        return qwk(y, p)
    return auc(y, score if score is not None else p)


def run_seed(task, y, Xa, Xv, folds, model, nested=True):
    n = len(y)
    out = {k: [] for k in ("audio", "video", "early", "late_eq", "late_w", "late_stack")}
    ws = []
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        Aa, Ba = prep(Xa[tr], Xa[te], 32)
        Av, Bv = prep(Xv[tr], Xv[te], 32)
        pa = model(Aa, y[tr], Ba); pv = model(Av, y[tr], Bv)
        pe = model(np.hstack([Aa, Av]), y[tr], np.hstack([Ba, Bv]))
        dec = (lambda z: z.argmax(1)) if task == "severity" else (lambda z: z)
        out["audio"].append((te, dec(pa))); out["video"].append((te, dec(pv)))
        out["early"].append((te, dec(pe)))
        if task == "severity":
            out["late_eq"].append((te, ((pa + pv) / 2).argmax(1)))
        else:
            out["late_eq"].append((te, (pa + pv) / 2))

        # 嵌套：内层 3 折产生 OOF 概率，再选权重
        if nested:
            inner = strat_folds(y[tr], 3, seed=100 + f)
            idx_tr = np.where(tr)[0]
            pa_oof = np.zeros((len(idx_tr), pa.shape[1]) if pa.ndim == 2 else len(idx_tr))
            pv_oof = np.zeros_like(pa_oof)
            for k in range(3):
                itr, ite = inner != k, inner == k
                a1, a2 = prep(Xa[idx_tr[itr]], Xa[idx_tr[ite]], 32)
                v1, v2 = prep(Xv[idx_tr[itr]], Xv[idx_tr[ite]], 32)
                pa_oof[ite] = model(a1, y[idx_tr[itr]], a2)
                pv_oof[ite] = model(v1, y[idx_tr[itr]], v2)
        else:
            pa_oof = model(Aa, y[tr], Aa); pv_oof = model(Av, y[tr], Av)

        best = (-9, None)
        for w in np.arange(0, 1.001, 0.1):
            comb = w * pa_oof + (1 - w) * pv_oof
            p = comb.argmax(1) if task == "severity" else (comb >= 0.5).astype(int)
            v = metric(task, y[tr], p, comb)
            if v > best[0]:
                best = (v, w)
        w = best[1]; ws.append(w)
        comb = w * pa + (1 - w) * pv
        out["late_w"].append((te, comb.argmax(1) if task == "severity" else comb))

        # stacking
        Ftr = np.column_stack([pa_oof, pv_oof]); Fte = np.column_stack([pa, pv])
        if task == "severity":
            ps = softmax_reg(Ftr, y[tr], Fte, l2=0.1)
            out["late_stack"].append((te, ps.argmax(1)))
        else:
            ps = logistic(Ftr, y[tr], Fte, l2=0.1)
            out["late_stack"].append((te, ps))

    # 汇总 OOF
    res = {}
    for k, lst in out.items():
        if task == "severity":
            p = np.zeros(n, int)
            for te, v in lst:
                p[te] = v
            res[k] = qwk(y, p)
        else:
            p = np.zeros(n)
            for te, v in lst:
                p[te] = v
            res[k] = auc(y, p)
    return res, float(np.mean(ws))


def main():
    X, z = load()
    ysev = z["severity"].astype(int)
    ybin = z["binary"].astype(int)
    print(f"n={len(ysev)}  severity={np.bincount(ysev, minlength=5)}  "
          f"binary阳性={ybin.mean()*100:.1f}%")

    pairs = {
        "handcrafted (egemaps+openface)": ("audio_egemaps", "video_openface"),
        "deep (bow+vgg16)": ("audio_bow_egemaps", "video_vgg16"),
    }
    allres = {}
    for task, yy in (("severity", ysev), ("binary", ybin)):
        print(f"\n{'='*66}\n目标：{task}\n{'='*66}")
        for pname, (fa, fv) in pairs.items():
            for mk, model in make_models(task).items():
                if model is None:
                    continue
                rows = []
                ws = []
                for seed in range(N_SEEDS):
                    folds = strat_folds(yy, 5, seed)
                    r, w = run_seed(task, yy, X[fa], X[fv], folds, model)
                    rows.append(r); ws.append(w)
                keys = rows[0].keys()
                print(f"\n--- {pname} [{mk}] ---")
                print(f"  {'model':12s} {'mean':>8s} {'sd':>8s}")
                for k in keys:
                    v = np.array([r[k] for r in rows])
                    print(f"  {k:12s} {v.mean():8.4f} {v.std():8.4f}")
                d = np.array([r["early"] - r["late_w"] for r in rows])
                d2 = np.array([r["early"] - r["late_eq"] for r in rows])
                print(f"  early - late_w : Δ={d.mean():+.4f}  为正 {int((d>0).sum())}/{N_SEEDS}")
                print(f"  early - late_eq: Δ={d2.mean():+.4f}  为正 {int((d2>0).sum())}/{N_SEEDS}")
                print(f"  late 平均权重 audio={np.mean(ws):.2f}")
                allres[f"{task}|{pname}|{mk}"] = {
                    k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))]
                    for k in keys}

    json.dump(allres, open(os.path.join(ROOT, "_daic_fusion2_results.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
