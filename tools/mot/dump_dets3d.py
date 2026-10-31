#!/usr/bin/env python3
"""
MOT v1.1 (2026-10-31): dump Complex-YOLO / YOLO3D detections on the KITTI tracking val sequences (LiDAR) for 3D MOT.

    python tools/mot/dump_dets3d.py --all                       # every trained complex_* / yolo3d_* checkpoint found
    python tools/mot/dump_dets3d.py --task yolo3d --version v5 --weights <pth>

Input : ~/Physical_AI/datasets/KITTI_tracking/training/velodyne/<seq>/<frame>.bin (the 9 KITTI-MOTS val sequences)
Model : the framework's own build_model + KittiTask.postprocess (same BEV map, NMS and decoding as the KITTI_5k test),
        conf >= --conf (low, the tracker applies its own threshold).
Output: ~/mot_logs/dets3d/<task>_<version>.npz
          <seq>      [N, 10] float32 = frame, x, y, z, h, w, l, yaw, score, class   (LiDAR frame, z = box bottom,
                                        class 0 Car / 1 Pedestrian / 2 Cyclist)
          <seq>__ms  [frames] detector time per frame (BEV map + network + post-process)
          meta       json (task, version, weights, conf, boundary)
Note  : the detectors were trained on KITTI_5k (KITTI object); its frames can come from the same drives as the KITTI
        tracking sequences -> check with check_overlap.py before reading the numbers as unseen-data results.
"""

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.kitti import config as cnf  # noqa: E402
from common.kitti import kitti_bev_utils as bev_utils  # noqa: E402
from common.utils import load_checkpoint  # noqa: E402
from tasks.kitti_task import KittiTask  # noqa: E402

VAL = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]
VERSIONS = ["v2", "v3", "v4", "v5", "v7", "sv4", "yolor", "efficientdet", "retinanet", "faster_rcnn",
            "deformable_detr", "rtdetr"]


def find_weights(task, version, roots):
    best = None
    for root in roots:
        for p in Path(root).expanduser().glob(f"**/{task}_{version}_kitti5k_e*.pth"):
            m = re.search(r"_e(\d+)\.pth$", p.name)
            if m and (best is None or int(m.group(1)) > best[0]):
                best = (int(m.group(1)), p)
    return None if best is None else best[1]


@torch.no_grad()
def dump(task, version, weights, a, device):
    mod = importlib.import_module(f"evaluate_{task}_{version}")
    t = KittiTask(task)
    model = mod.build_model(task, len(t.class_names)).to(device)
    load_checkpoint(weights, model, device)
    model.eval()
    vel = Path(a.kitti).expanduser() / "training" / "velodyne"
    out = {}
    for seq in a.seqs:
        files = sorted((vel / seq).glob("*.bin"))
        rows, ms = [], []
        for f, p in enumerate(files):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
            bev = bev_utils.makeBVFeature(bev_utils.removePoints(pts, cnf.boundary), cnf.DISCRETIZATION, cnf.boundary)
            raw = model(torch.from_numpy(bev).float().unsqueeze(0).to(device))
            det = t.postprocess(raw, bev.shape[-1], a.conf)[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms.append(1000 * (time.perf_counter() - t0))
            for b, s, c in zip(det["boxes"], det["scores"], det["labels"]):
                rows.append([f, *b, s, c])
        out[seq] = np.array(rows, np.float32).reshape(-1, 10)
        out[f"{seq}__ms"] = np.array(ms, np.float32)
        print(f"  {task}_{version} {seq}: {len(files)} frames, {len(out[seq])} boxes, {np.median(ms):.1f} ms", flush=True)
    meta = {"task": task, "version": version, "weights": str(weights), "conf": a.conf, "boundary": cnf.boundary,
            "classes": list(t.class_names), "source": "KITTI tracking velodyne", "split": "KITTI-MOTS val sequences" if a.seqs == VAL else f"sequences {a.seqs}",
            "date": time.strftime("%F %T")}
    dst = Path(a.out).expanduser() / f"{task}_{version}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, meta=np.array(json.dumps(meta)), **out)
    print(f"-> {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--task", choices=["complex", "yolo3d"])
    ap.add_argument("--version")
    ap.add_argument("--weights")
    ap.add_argument("--roots", nargs="+", default=["~/Physical_AI/release_bundle"])
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--kitti", default="~/Physical_AI/datasets/KITTI_tracking")
    ap.add_argument("--out", default="~/mot_logs/dets3d")
    ap.add_argument("--list", action="store_true", help="only show which weights would be used")
    ap.add_argument("--seqs", nargs="+", default=VAL, help="v1.1-b: e.g. the KITTI-MOTS train sequences")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = []
    if a.all:
        for task in ("yolo3d", "complex"):
            for v in VERSIONS:
                w = find_weights(task, v, a.roots)
                print(f"{task:<8} {v:<16} {w if w else 'NOT FOUND'}")
                if w:
                    jobs.append((task, v, w))
    else:
        jobs.append((a.task, a.version, Path(a.weights).expanduser()))
    if a.list:
        return
    for task, v, w in jobs:
        if (Path(a.out).expanduser() / f"{task}_{v}.npz").exists():
            print(f"skip {task}_{v} (exists)")
            continue
        try:
            dump(task, v, w, a, device)
        except Exception as e:                       # one failing model must not stop the others
            print(f"FAILED {task}_{v}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
