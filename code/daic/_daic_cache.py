# -*- coding: utf-8 -*-
"""DAIC-WOZ 特征聚合缓存：逐帧特征 → 受试者级 pooled 向量。"""
import os, sys, csv, glob
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "_cache")
os.makedirs(OUT, exist_ok=True)


def detect_delim(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        line = f.readline()
    return ";" if line.count(";") > line.count(",") else ","


def load(path, delim, skip=1, usecols=None):
    if delim == "auto":
        delim = detect_delim(path)
    return np.loadtxt(path, delimiter=delim, skiprows=skip, usecols=usecols, dtype=np.float32)


def pool(X, stats=("mean", "std")):
    out = []
    if "mean" in stats:
        out.append(np.nanmean(X, axis=0))
    if "std" in stats:
        out.append(np.nanstd(X, axis=0))
    return np.concatenate(out)


SPECS = {
    "audio_egemaps":   ("OpenSMILE2.3.0_egemaps.csv", ";", range(2, 25), ("mean", "std")),
    "audio_mfcc":      ("OpenSMILE2.3.0_mfcc.csv", ";", range(2, 41), ("mean", "std")),
    "audio_bow_egemaps": ("BoAW_openSMILE_2.3.0_eGeMAPS.csv", "auto", range(1, 102), ("mean",)),
    "audio_bow_mfcc":  ("BoAW_openSMILE_2.3.0_MFCC.csv", "auto", range(1, 102), ("mean",)),
    "video_openface":  ("OpenFace2.1.0_Pose_gaze_AUs.csv", ",", range(4, 53), ("mean", "std")),
    "video_bovw":      ("BoVW_openFace_2.1.0_Pose_Gaze_AUs.csv", "auto", range(1, 102), ("mean",)),
    "video_vgg16":     ("vgg16.csv", ",", range(2, 4098), ("mean",)),
    "video_densenet":  ("densenet201.csv", ",", range(2, 1922), ("mean",)),
}


def main():
    subs = sorted([d for d in os.listdir(DATA) if d.endswith("_P")])
    print(f"subject dirs: {len(subs)}")
    feats = {k: [] for k in SPECS}
    ids, ok = [], []
    for s in subs:
        fd = os.path.join(DATA, s, "features")
        if not os.path.isdir(fd):
            continue
        vecs = {}
        good = True
        sid = s.replace("_P", "")
        for name, (fn, delim, cols, stats) in SPECS.items():
            fp = os.path.join(fd, f"{sid}_{fn}")
            if not os.path.exists(fp):
                good = False; break
            try:
                X = load(fp, delim, usecols=cols)
                if X.ndim == 1:
                    X = X[None, :]
                vecs[name] = pool(X, stats)
            except Exception as e:
                print(f"  ! {s} {name}: {e}")
                good = False; break
        if not good:
            continue
        ids.append(s.replace("_P", ""))
        ok.append(True)
        for k in SPECS:
            feats[k].append(vecs[k])

    ids = np.array(ids)
    print(f"loaded: {len(ids)}")
    for k in SPECS:
        X = np.vstack(feats[k]).astype(np.float32)
        X[~np.isfinite(X)] = 0.0
        np.save(os.path.join(OUT, f"daic_{k}.npy"), X)
        print(f"{k:20s} {X.shape}")

    # 标签
    rows = list(csv.DictReader(open(os.path.join(ROOT, "labels", "detailed_lables.csv"),
                                    encoding="utf-8-sig")))
    lab = {r["Participant"]: r for r in rows}
    phq, sev, binary, gender, age = [], [], [], [], []
    keep = []
    for i, sid in enumerate(ids):
        if sid not in lab:
            continue
        r = lab[sid]
        items = [float(r[f"PHQ8_{j}_{n}"]) for j, n in [
            (1, "NoInterest"), (2, "Depressed"), (3, "Sleep"), (4, "Tired"),
            (5, "Appetite"), (6, "Failure"), (7, "Concentration"), (8, "Psychomotor")]]
        sc = sum(items)
        phq.append(sc)
        sev.append(0 if sc <= 4 else 1 if sc <= 9 else 2 if sc <= 14 else 3 if sc <= 19 else 4)
        binary.append(int(sc >= 10))
        gender.append(1 if r["gender"] == "male" else 0)
        age.append(float(r["age"]))
        keep.append(i)
    keep = np.array(keep)
    np.savez(os.path.join(OUT, "daic_labels.npz"),
             ids=ids[keep], phq=np.array(phq, np.float32),
             severity=np.array(sev, np.int64), binary=np.array(binary, np.int64),
             gender=np.array(gender, np.int64), age=np.array(age, np.float32),
             row_index=keep)
    print(f"\nlabeled: {len(phq)}  PHQ均值={np.mean(phq):.2f}  "
          f"binary阳性={np.mean(binary)*100:.1f}%  严重度分布={np.bincount(sev, minlength=5)}")


if __name__ == "__main__":
    main()
