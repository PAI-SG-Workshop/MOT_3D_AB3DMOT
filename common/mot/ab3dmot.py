"""
MOT v1.1 (2026-10-31): AB3DMOT - 3D multi-object tracking baseline (Weng et al., IROS 2020, xinshuoweng/AB3DMOT).

Per class, every frame:
  1) predict every track with a 3D Kalman filter, state [x, y, z, yaw, l, w, h, vx, vy, vz] (constant velocity)
  2) 3D IoU between detections and predicted tracks, Hungarian, pairs with IoU < 0.01 unmatched
  3) matched tracks: orientation correction (detection yaw flipped by pi when it differs by more than 90 degrees),
     Kalman update; unmatched detections start tracks
  4) output tracks updated in this frame with hits >= min_hits (or in the first min_hits frames);
     tracks unmatched for more than max_age frames are deleted
Official KITTI values (paper / v1 code): IoU threshold 0.01, min_hits 3, max_age 2, P0 = 10 I (velocity 10000),
Q = I (velocity 0.01), R = I.
[Unified] decisions:
  * frame: the LiDAR frame of the BEV detectors (x forward, y left, z up, z = box bottom), box [x, y, z, h, w, l, yaw];
    the official code works in the camera frame of KITTI - same geometry, different axes.
  * detection score threshold det_thr = 0.5 (the official code takes PointRCNN detections as they are; our
    detectors are dumped at conf >= 0.1).
  * one tracker per class (official: one run per category).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.geometry import iou_3d

DEFAULTS = dict(iou_thr=0.01, min_hits=3, max_age=2, det_thr=0.5)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class _KF:
    def __init__(self, box):                          # box: x y z h w l yaw (LiDAR)
        x, y, z, h, w, l, yaw = box
        self.x = np.array([x, y, z, yaw, l, w, h, 0, 0, 0], float)
        self.F = np.eye(10)
        self.F[0, 7] = self.F[1, 8] = self.F[2, 9] = 1
        self.H = np.eye(7, 10)
        self.P = np.eye(10) * 10.0
        self.P[7:, 7:] *= 1000.0
        self.Q = np.eye(10)
        self.Q[7:, 7:] *= 0.01
        self.R = np.eye(7)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[3] = wrap(self.x[3])

    def update(self, box):
        x, y, z, h, w, l, yaw = box
        yaw = wrap(yaw)
        d = wrap(yaw - self.x[3])
        if abs(d) > np.pi / 2:                        # orientation correction
            self.x[3] = wrap(self.x[3] + np.pi)
        zv = np.array([x, y, z, yaw, l, w, h])
        y_ = zv - self.H @ self.x
        y_[3] = wrap(y_[3])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y_
        self.x[3] = wrap(self.x[3])
        self.P = (np.eye(10) - K @ self.H) @ self.P

    def box(self):
        x, y, z, yaw, l, w, h = self.x[:7]
        return np.array([x, y, z, h, w, l, yaw])


class _Track:
    count = 0

    def __init__(self, box, score):
        _Track.count += 1
        self.id, self.kf, self.score = _Track.count, _KF(box), score
        self.hits, self.time_since_update = 1, 0


class AB3DMOT:
    def __init__(self, **kw):
        p = dict(DEFAULTS, **kw)
        self.iou_thr, self.min_hits, self.max_age, self.det_thr = p["iou_thr"], p["min_hits"], p["max_age"], p["det_thr"]
        self.tracks, self.frame = [], 0

    def update(self, dets):
        """dets [n, 8] = x y z h w l yaw score -> [m, 9] = x y z h w l yaw id score."""
        self.frame += 1
        dets = np.asarray(dets, float).reshape(-1, 8)
        dets = dets[dets[:, 7] >= self.det_thr]
        for t in self.tracks:
            t.kf.predict()
            t.time_since_update += 1
        matched_d = set()
        if len(self.tracks) and len(dets):
            pred = np.stack([t.kf.box() for t in self.tracks])
            iou = np.stack([iou_3d(d[:7], pred) for d in dets])            # [dets, tracks]
            r, c = linear_sum_assignment(-iou)
            for di, ti in zip(r, c):
                if iou[di, ti] >= self.iou_thr:
                    t = self.tracks[ti]
                    t.kf.update(dets[di, :7])
                    t.hits += 1
                    t.time_since_update = 0
                    t.score = dets[di, 7]
                    matched_d.add(di)
        for di in range(len(dets)):
            if di not in matched_d:
                self.tracks.append(_Track(dets[di, :7], dets[di, 7]))
        out = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame <= self.min_hits):
                out.append([*t.kf.box(), t.id, t.score])
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return np.array(out, float).reshape(-1, 9)
