# -*- coding: utf-8 -*-
"""把每个 clip 的 pooled 特征聚合成 subject×timepoint 向量并缓存。

输出 _cache/<feature>.npz:
  X       (n_st, d)   float32
  keys    (n_st, 4)   [school, class, person, timepoint]
  n_clips (n_st,)     参与聚合的 clip 数
"""
import os, sys, json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "train")
OUT = os.path.join(HERE, "_cache")
os.makedirs(OUT, exist_ok=True)

META_KEYS = {"duration_sec", "frame_count"}
FEATS = [d for d in sorted(os.listdir(ROOT))
         if os.path.isdir(os.path.join(ROOT, d)) and d != "transcripts"]


def load_pooled(clip_dir):
    pnpy = os.path.join(clip_dir, "pooled.npy")
    if os.path.exists(pnpy):
        v = np.load(pnpy).astype(np.float32).ravel()
        return v if v.size else None
    pjson = os.path.join(clip_dir, "pooled.json")
    if os.path.exists(pjson):
        d = json.load(open(pjson, encoding="utf-8"))
        ks = [k for k in sorted(d) if k not in META_KEYS and isinstance(d[k], (int, float))]
        return np.array([d[k] for k in ks], dtype=np.float32)
    return None


for f in FEATS:
    base = os.path.join(ROOT, f)
    X, keys, nclips = [], [], []
    for sch in sorted(os.listdir(base)):
        for cls in sorted(os.listdir(os.path.join(base, sch))):
            for per in sorted(os.listdir(os.path.join(base, sch, cls))):
                pd = os.path.join(base, sch, cls, per)
                for t in sorted(os.listdir(pd)):
                    td = os.path.join(pd, t)
                    if not os.path.isdir(td):
                        continue
                    vecs = []
                    for a in sorted(os.listdir(td)):
                        cd = os.path.join(td, a)
                        if not os.path.isdir(cd):
                            continue
                        v = load_pooled(cd)
                        if v is not None:
                            vecs.append(v)
                    if not vecs:
                        continue
                    dims = {v.shape[0] for v in vecs}
                    if len(dims) != 1:
                        print(f"  ! {f} {sch}/{cls}/{per}/{t} 维度不一致 {dims}，跳过")
                        continue
                    X.append(np.mean(vecs, axis=0))
                    keys.append([sch, cls, per, t])
                    nclips.append(len(vecs))
    Xa = np.vstack(X).astype(np.float32)
    ka = np.array(keys, dtype=object)
    np.savez_compressed(os.path.join(OUT, f + ".npz"),
                        X=Xa, keys=ka, n_clips=np.array(nclips, dtype=np.int16))
    print(f"{f:32s} X={Xa.shape}  d={Xa.shape[1]:5d}  clips/ST mean={np.mean(nclips):.2f}")

print("\ndone ->", OUT)
