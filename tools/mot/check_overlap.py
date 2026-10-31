"""학습 데이터 중복 검사: KITTI_5k(KITTI object) 학습 이미지가 KITTI-MOTS val 시퀀스 frame과 같은 장면인지 확인.

이미지 재인코딩으로 바이트가 달라도 잡히도록, 64x20 흑백 썸네일의 정규화 상관으로 비교합니다.
사용: python check_overlap.py --obj-dir <KITTI_5k 학습 이미지 폴더> --trk-root <KITTI_MOTS image_02 폴더>
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

VAL = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]


def thumb(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    t = cv2.resize(g, (64, 20), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


ap = argparse.ArgumentParser()
ap.add_argument("--obj-dir", required=True)
ap.add_argument("--trk-root", required=True)
ap.add_argument("--thr", type=float, default=0.98)
ap.add_argument("--out", default=os.path.expanduser("~/mot_logs/overlap_kitti5k_vs_motsval.csv"))
a = ap.parse_args()

obj = sorted(glob.glob(os.path.join(a.obj_dir, "*.png")) + glob.glob(os.path.join(a.obj_dir, "*.jpg")))
print(f"KITTI_5k images: {len(obj)}")
O = np.stack([thumb(p) for p in obj])

hits, total = [], 0
for s in VAL:
    frames = sorted(glob.glob(os.path.join(a.trk_root, s, "*.png")))
    total += len(frames)
    T = np.stack([thumb(p) for p in frames])
    sim = T @ O.T                                   # (val frames, object images)
    j = sim.argmax(1)
    m = sim[np.arange(len(frames)), j]
    sel = np.where(m >= a.thr)[0]
    hits += [(s, os.path.basename(frames[i]), os.path.basename(obj[j[i]]), float(m[i])) for i in sel]
    print(f"seq {s}: {len(frames):5d} frames, {len(sel):4d} near-duplicate (corr>={a.thr})")

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", newline="") as f:
    csv.writer(f).writerows([("seq", "frame", "kitti5k_image", "corr")] + hits)
print(f"\ntotal val frames {total}, matched {len(hits)} "
      f"({len(set(h[2] for h in hits))} distinct KITTI_5k images) -> {a.out}")
