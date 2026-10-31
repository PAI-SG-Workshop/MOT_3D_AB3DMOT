#!/usr/bin/env python3
"""
MOT v1.1 (2026-10-31): per-detector AB3DMOT score threshold chosen on the KITTI-MOTS TRAIN sequences, then applied
to val (never tuned on val).

Why per detector: the val sweep (0.5 / 0.7 / 0.9) showed opposite behaviour - DETR-type detectors collapse at high
thresholds (their scores are low), YOLO-type detectors need high thresholds (many duplicate boxes) - so one fixed
threshold is not a fair setting for all of them.
Caveat: the train sequences also overlap with the KITTI_5k training images of the 3D detectors, so their scores
there are optimistic; the chosen thresholds may be higher than ideal for unseen data.

    python tools/mot/dump_dets3d.py --all --seqs <12 train sequences> --out ~/mot_logs/dets3d_train     (GPU)
    python tools/mot/tune_ab3dmot.py                                                                  (CPU)
Output: ~/mot_logs/track3d_trainthr/<det>/ab3dmot.json (val, chosen threshold), ~/mot_logs/ab3dmot_trainthr.txt
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tracker3d as R  # noqa: E402
from common.geometry import bev_iou_3dboxes, iou_3d  # noqa: E402
from common.mot.ab3dmot import AB3DMOT  # noqa: E402

TRAIN = ["0000", "0001", "0003", "0004", "0005", "0009", "0011", "0012", "0015", "0017", "0019", "0020"]
GRID = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
ROOT = Path("~/Physical_AI/datasets/KITTI_tracking/training").expanduser()
MOTS = Path("~/Physical_AI/datasets/KITTI_MOTS/training/image_02").expanduser()


def run(z, seqs, thr):
    gt, pred = {}, {}
    for seq in seqs:
        n = len(z[f"{seq}__ms"])
        gt[seq] = R.load_gt(ROOT, seq, n)
        V2C, R0, P2 = R.read_calib(ROOT / "calib" / f"{seq}.txt", with_p2=True)
        h, w = cv2.imread(str(sorted((MOTS / seq).glob("*.png"))[0])).shape[:2]
        rows = z[seq]
        rows = rows[R.in_camera_fov(rows[:, 1:8], V2C, R0, P2, (w, h))]
        trk = {c: AB3DMOT(det_thr=thr) for c in range(len(R.CLASSES))}
        out = []
        for f in range(n):
            part = []
            for c, t in trk.items():
                o = t.update(rows[(rows[:, 0] == f) & (rows[:, 9] == c)][:, 1:9])
                if len(o):
                    part.append(np.c_[o[:, :7], o[:, 7] + c * 100000, np.full(len(o), c), o[:, 8]])
            out.append(np.concatenate(part) if part else np.zeros((0, 10)))
        pred[seq] = out
    return gt, pred


def one(name):
    zt = np.load(Path("~/mot_logs/dets3d_train").expanduser() / f"{name}.npz", allow_pickle=False)
    train = {}
    for thr in GRID:
        gt, pred = run(zt, TRAIN, thr)
        train[thr] = R.evaluate3d(gt, pred, bev_iou_3dboxes)["mean"]["HOTA"]
    best = max(train, key=train.get)
    zv = np.load(Path("~/mot_logs/dets3d").expanduser() / f"{name}.npz", allow_pickle=False)
    gt, pred = run(zv, R.VAL, best)
    rb, r3 = R.evaluate3d(gt, pred, bev_iou_3dboxes), R.evaluate3d(gt, pred, iou_3d)
    rec = {"tracker": "ab3dmot", "detections": name, "det_thr": best, "threshold_selection": "train BEV HOTA",
           "train_hota_per_thr": train, "camera_fov_filter": True, "bev": rb, "3d": r3,
           "detector_ms": float(np.median(np.concatenate([zv[f"{s}__ms"] for s in R.VAL])))}
    o = Path("~/mot_logs/track3d_trainthr").expanduser() / name
    o.mkdir(parents=True, exist_ok=True)
    (o / "ab3dmot.json").write_text(json.dumps(rec, indent=1))
    m = rb["mean"]
    return (f"{name:<24} thr {best:<4} (train {train[best]:.3f}) | val BEV HOTA {m['HOTA']:.3f} DetA {m['DetA']:.3f} "
            f"AssA {m['AssA']:.3f} MOTA {m['MOTA']:.3f} IDSW {m['IDSW']:>4} | 3D HOTA {r3['mean']['HOTA']:.3f} | "
            f"car {rb['per_class']['Car']['HOTA']:.3f} ped {rb['per_class']['Pedestrian']['HOTA']:.3f}")


def main():
    names = sorted(p.stem for p in Path("~/mot_logs/dets3d_train").expanduser().glob("*.npz"))
    lines = []
    with Pool(8) as p:
        for line in p.imap_unordered(one, names):
            print(line, flush=True)
            lines.append(line)
    Path("~/mot_logs/ab3dmot_trainthr.txt").expanduser().write_text("\n".join(sorted(lines)) + "\n")


if __name__ == "__main__":
    main()
