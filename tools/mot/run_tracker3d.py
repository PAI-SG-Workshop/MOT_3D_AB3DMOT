#!/usr/bin/env python3
"""
MOT v1.1 (2026-10-31): 3D MOT (AB3DMOT) on the LiDAR detections of dump_dets3d.py, KITTI tracking val sequences
(= the 9 KITTI-MOTS val sequences), evaluated with HOTA / CLEAR / Identity in the LiDAR frame.

    python tools/mot/run_tracker3d.py                     # every ~/mot_logs/dets3d/*.npz
    python tools/mot/run_tracker3d.py --dets yolo3d_v5

Ground truth: KITTI tracking label_02 (camera frame) -> LiDAR frame with the sequence calib (R_rect, Tr_velo_cam) and
the framework's own camera_to_lidar_box (the same conversion as the KITTI_5k training labels). Kept: Car and
Pedestrian inside the BEV range (x 0..50 m, |y| < 25 m), as the detectors see it. Van (for Car) and Person_sitting
(for Pedestrian) are ignore regions: an unmatched prediction overlapping one (BEV IoU >= 0.5) is dropped, as the
KITTI tracking benchmark does. Similarity = BEV IoU (primary, fair to Complex-YOLO whose z / h are class constants)
and 3D IoU (secondary).
Output: ~/mot_logs/track3d/<detector>/ab3dmot.json, summary line printed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.geometry import bev_iou_3dboxes, iou_3d  # noqa: E402
from common.kitti import config as cnf  # noqa: E402
from common.kitti.kitti_aug_utils import camera_to_lidar_box  # noqa: E402
from common.mot.ab3dmot import AB3DMOT  # noqa: E402
from common.mot.core import assign  # noqa: E402
from common.mot.evaluate import CLEAR, HOTA, Identity  # noqa: E402

VAL = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]
CLASSES = ("Car", "Pedestrian")
IGNORE = {"Van": 0, "Person_sitting": 1}


def read_calib(p, with_p2=False):
    v = {}
    for line in Path(p).read_text().splitlines():
        if not line.strip():
            continue
        k, *nums = line.replace(":", " ").split()
        v[k] = np.array(nums, float)
    out = (v["Tr_velo_cam"].reshape(3, 4), v["R_rect"].reshape(3, 3))
    return out + (v["P2"].reshape(3, 4),) if with_p2 else out


def in_camera_fov(boxes, V2C, R0, P2, wh):
    """v1.1-b: KITTI labels exist only inside the left camera's field of view, so detections whose box centre
    projects outside the image (or behind the camera) are dropped before tracking, as the KITTI benchmark does."""
    if len(boxes) == 0:
        return np.zeros(0, bool)
    c = np.c_[boxes[:, 0], boxes[:, 1], boxes[:, 2] + boxes[:, 3] / 2, np.ones(len(boxes))]   # centre, LiDAR
    cam = (R0 @ (V2C @ c.T)).T                                                                # rectified camera
    uvw = (P2 @ np.c_[cam, np.ones(len(cam))].T).T
    u, v = uvw[:, 0] / np.maximum(uvw[:, 2], 1e-6), uvw[:, 1] / np.maximum(uvw[:, 2], 1e-6)
    return (cam[:, 2] > 0.5) & (u >= 0) & (u < wh[0]) & (v >= 0) & (v < wh[1])


def load_gt(root, seq, n_frames):
    V2C, R0 = read_calib(root / "calib" / f"{seq}.txt")
    frames = [{"boxes": [], "ids": [], "cls": [], "ign": [], "ign_cls": []} for _ in range(n_frames)]
    for line in (root / "label_02" / f"{seq}.txt").read_text().splitlines():
        p = line.split()
        f, tid, typ = int(p[0]), int(p[1]), p[2]
        if typ not in CLASSES and typ not in IGNORE:
            continue
        h, w, l, x, y, z, ry = map(float, p[10:17])
        b = camera_to_lidar_box(np.array([[x, y, z, h, w, l, ry]]), V2C, R0)[0]
        bc = cnf.boundary
        if not (bc["minX"] < b[0] < bc["maxX"] and bc["minY"] < b[1] < bc["maxY"]) or f >= n_frames:
            continue
        if typ in CLASSES:
            fr = frames[f]
            fr["boxes"].append(b), fr["ids"].append(tid), fr["cls"].append(CLASSES.index(typ))
        else:
            frames[f]["ign"].append(b), frames[f]["ign_cls"].append(IGNORE[typ])
    for fr in frames:
        fr["boxes"] = np.array(fr["boxes"], float).reshape(-1, 7)
        fr["ids"], fr["cls"] = np.array(fr["ids"], int), np.array(fr["cls"], int)
        fr["ign"], fr["ign_cls"] = np.array(fr["ign"], float).reshape(-1, 7), np.array(fr["ign_cls"], int)
    return frames


def sim_matrix(fn, a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return np.stack([fn(x, b) for x in a])


def sequence_data(gt_frames, pred_frames, c, fn):
    gt_ids, tr_ids, sims = [], [], []
    for g, p in zip(gt_frames, pred_frames):
        gm = g["cls"] == c
        gb, gi = g["boxes"][gm], g["ids"][gm]
        p = p[p[:, 8] == c]
        pb, pi = p[:, :7], p[:, 7].astype(int)
        ign = g["ign"][g["ign_cls"] == c]
        if len(pb) and len(ign):
            matched = np.zeros(len(pb), bool)
            s = sim_matrix(bev_iou_3dboxes, gb, pb)
            if s.size:
                m, _, _ = assign(-s)
                ok = s[m[:, 0], m[:, 1]] >= 0.5
                matched[m[ok, 1]] = True
            drop = (~matched) & (sim_matrix(bev_iou_3dboxes, pb, ign).max(1) >= 0.5)
            pb, pi = pb[~drop], pi[~drop]
        gt_ids.append(gi)
        tr_ids.append(pi)
        sims.append(sim_matrix(fn, gb, pb))
    ug = np.unique(np.concatenate(gt_ids)) if any(len(x) for x in gt_ids) else np.zeros(0, int)
    ut = np.unique(np.concatenate(tr_ids)) if any(len(x) for x in tr_ids) else np.zeros(0, int)
    gmap, tmap = {v: i for i, v in enumerate(ug)}, {v: i for i, v in enumerate(ut)}
    return {"num_timesteps": len(gt_ids), "num_gt_ids": len(ug), "num_tracker_ids": len(ut),
            "num_gt_dets": int(sum(len(x) for x in gt_ids)), "num_tracker_dets": int(sum(len(x) for x in tr_ids)),
            "gt_ids": [np.array([gmap[v] for v in x], int) for x in gt_ids],
            "tracker_ids": [np.array([tmap[v] for v in x], int) for x in tr_ids], "similarity_scores": sims}


def evaluate3d(gt, pred, fn):
    metrics = [HOTA(), CLEAR({"PRINT_CONFIG": False}), Identity({"PRINT_CONFIG": False})]
    per = {}
    for c, name in enumerate(CLASSES):
        comb = {}
        for M in metrics:
            comb.update(M.combine_sequences({s: M.eval_sequence(sequence_data(gt[s], pred[s], c, fn)) for s in gt}))
        per[name] = {"HOTA": float(np.mean(comb["HOTA"])), "DetA": float(np.mean(comb["DetA"])),
                     "AssA": float(np.mean(comb["AssA"])), "MOTA": float(comb["MOTA"]), "IDF1": float(comb["IDF1"]),
                     "IDSW": int(comb["IDSW"])}
    mean = {k: float(np.mean([v[k] for v in per.values()])) for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1")}
    mean["IDSW"] = int(sum(v["IDSW"] for v in per.values()))
    return {"per_class": per, "mean": mean}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", nargs="*", default=None)
    ap.add_argument("--src", default="~/mot_logs/dets3d")
    ap.add_argument("--kitti", default="~/Physical_AI/datasets/KITTI_tracking/training")
    ap.add_argument("--out", default="~/mot_logs/track3d")
    ap.add_argument("--det-thr", type=float, default=0.5)
    ap.add_argument("--no-fov", action="store_true", help="v1.1-a behaviour: keep detections outside the camera view")
    ap.add_argument("--seqs", nargs="+", default=VAL)
    ap.add_argument("--mots", default="~/Physical_AI/datasets/KITTI_MOTS/training/image_02")
    a = ap.parse_args()
    src, root = Path(a.src).expanduser(), Path(a.kitti).expanduser()
    names = a.dets or sorted(p.stem for p in src.glob("*.npz"))
    gt_cache = {}
    for name in names:
        z = np.load(src / f"{name}.npz", allow_pickle=False)
        gt, pred, tms = {}, {}, []
        for seq in a.seqs:
            n = len(z[f"{seq}__ms"])
            gt[seq] = gt_cache.setdefault(seq, load_gt(root, seq, n))
            rows = z[seq]
            if not a.no_fov:
                import cv2
                V2C, R0, P2 = read_calib(root / "calib" / f"{seq}.txt", with_p2=True)
                img = sorted((Path(a.mots).expanduser() / seq).glob("*.png"))[0]
                h_img, w_img = cv2.imread(str(img)).shape[:2]
                keep = in_camera_fov(rows[:, 1:8], V2C, R0, P2, (w_img, h_img))
                n_in, n_all = int(keep.sum()), len(rows)
                rows = rows[keep]
            trk = {c: AB3DMOT(det_thr=a.det_thr) for c in range(len(CLASSES))}
            out = []
            for f in range(n):
                part = []
                for c, t in trk.items():
                    d = rows[(rows[:, 0] == f) & (rows[:, 9] == c)][:, 1:9]
                    t0 = time.perf_counter()
                    o = t.update(d)
                    tms.append(1000 * (time.perf_counter() - t0))
                    if len(o):
                        part.append(np.c_[o[:, :7], o[:, 7] + c * 100000, np.full(len(o), c), o[:, 8]])
                out.append(np.concatenate(part) if part else np.zeros((0, 10)))
            pred[seq] = out
        res_bev, res_3d = evaluate3d(gt, pred, bev_iou_3dboxes), evaluate3d(gt, pred, iou_3d)
        det_ms = np.concatenate([z[f"{s}__ms"] for s in a.seqs])
        rec = {"tracker": "ab3dmot", "detections": name, "camera_fov_filter": not a.no_fov, "meta": json.loads(str(z["meta"])), "det_thr": a.det_thr,
               "similarity": {"primary": "BEV IoU", "secondary": "3D IoU"}, "bev": res_bev, "3d": res_3d,
               "tracker_ms": float(np.median(tms)) * len(CLASSES), "detector_ms": float(np.median(det_ms))}
        o = Path(a.out).expanduser() / name
        o.mkdir(parents=True, exist_ok=True)
        (o / "ab3dmot.json").write_text(json.dumps(rec, indent=1))
        m, m3 = res_bev["mean"], res_3d["mean"]
        print(f"{name:<24} BEV HOTA {m['HOTA']:.3f} DetA {m['DetA']:.3f} AssA {m['AssA']:.3f} MOTA {m['MOTA']:.3f} "
              f"IDSW {m['IDSW']:>4} | 3D HOTA {m3['HOTA']:.3f} | car {res_bev['per_class']['Car']['HOTA']:.3f} "
              f"ped {res_bev['per_class']['Pedestrian']['HOTA']:.3f} | det {rec['detector_ms']:.1f} ms",
              flush=True)


if __name__ == "__main__":
    main()
