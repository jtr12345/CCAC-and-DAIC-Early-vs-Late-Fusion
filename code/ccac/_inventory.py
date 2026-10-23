# -*- coding: utf-8 -*-
"""盘点 CCAC_26/train 下各特征目录的格式、维度与可用性。"""
import os, sys, json, collections
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")
FEATS = [d for d in sorted(os.listdir(ROOT))
         if os.path.isdir(os.path.join(ROOT, d)) and d not in ("transcripts",)]

report = []
for f in FEATS:
    base = os.path.join(ROOT, f)
    kinds = collections.Counter()
    dims = collections.Counter()
    n_st = 0          # subject-timepoint 目录数
    n_clip = 0
    missing = 0
    for sch in os.listdir(base):
        for cls in os.listdir(os.path.join(base, sch)):
            for per in os.listdir(os.path.join(base, sch, cls)):
                pd = os.path.join(base, sch, cls, per)
                for t in os.listdir(pd):
                    td = os.path.join(pd, t)
                    if not os.path.isdir(td):
                        continue
                    n_st += 1
                    clips = [a for a in os.listdir(td) if os.path.isdir(os.path.join(td, a))]
                    if not clips:
                        continue
                    n_clip += len(clips)
                    a0 = os.path.join(td, clips[0])
                    names = os.listdir(a0)
                    for nm in names:
                        kinds[os.path.splitext(nm)[1]] += 1
                    if "pooled.npy" in names:
                        dims[np.load(os.path.join(a0, "pooled.npy")).shape] += 1
                    elif "pooled.json" in names:
                        pj = json.load(open(os.path.join(a0, "pooled.json"), encoding="utf-8"))
                        dims[("json", len(pj))] += 1
                    else:
                        missing += 1
    report.append(dict(feat=f, n_st=n_st, n_clip=n_clip,
                       clips_per_st=round(n_clip / max(n_st, 1), 2),
                       exts=dict(kinds), dims={str(k): v for k, v in dims.items()},
                       no_pooled=missing))

print(f"{'feature':32s} {'subj×T':>7s} {'clips':>7s} {'clip/ST':>7s} {'dims':>22s}  exts")
for r in report:
    print(f"{r['feat']:32s} {r['n_st']:7d} {r['n_clip']:7d} {r['clips_per_st']:7.2f} "
          f"{str(list(r['dims'].keys())):>22s}  {list(r['exts'].keys())}")

json.dump(report, open("_inventory.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwrote _inventory.json")
