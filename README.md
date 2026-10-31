# MOT_3D_AB3DMOT

**AB3DMOT** - 3D multi-object tracking on LiDAR detections (Complex-YOLO and YOLO3D, 12 detector heads
each, the weights of the *Physical AI Object Detection* series) on the KITTI tracking sequences of KITTI-MOTS val.

- Paper: Weng et al., IROS 2020; official code: `xinshuoweng/AB3DMOT`
- Implementation: `common/mot/ab3dmot.py` (3D Kalman filter, 3D IoU 0.01, min_hits 3, max_age 2 - official KITTI
  values), evaluation `tools/mot/run_tracker3d.py`: KITTI tracking labels converted to the LiDAR frame with the same
  conversion as the detector training labels, camera field of view, Van / Person_sitting as ignore regions, HOTA with
  **BEV IoU** (Complex-YOLO has class-constant z / h) and 3D IoU.
- Detection score threshold: chosen **per detector on the KITTI-MOTS train sequences** (`tools/mot/tune_ab3dmot.py`),
  because the best threshold is opposite for DETR-type (low) and YOLO-type (high) detectors.
- **Caveat - training-data overlap**: 1581 of the 2,981 val frames are near-duplicates (thumbnail correlation >= 0.98) of KITTI_5k images; 846 of the 1,266 matched KITTI_5k images are in its trainval split, i.e. about 28 % of the val frames were seen by the 3D detectors during training. The 3D numbers are therefore optimistic.

### Results (KITTI-MOTS val sequences, LiDAR frame)

| detector | threshold (train) | **BEV HOTA** | DetA | AssA | MOTA | IDSW | 3D HOTA | car | ped | HOTA at 0.5 / 0.7 / 0.9 | detector ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo3d_faster_rcnn | 0.7 | **0.541** | 0.505 | 0.594 | 0.566 | 70 | 0.445 | 0.649 | 0.433 | 0.530 / 0.541 / 0.464 | 45.0 |
| complex_faster_rcnn | 0.8 | **0.535** | 0.515 | 0.574 | 0.561 | 69 | 0.437 | 0.664 | 0.406 | 0.527 / 0.542 / 0.500 | 46.5 |
| complex_rtdetr | 0.3 | **0.482** | 0.439 | 0.559 | 0.201 | 41 | 0.402 | 0.646 | 0.318 | 0.449 / 0.349 / 0.132 | 40.9 |
| yolo3d_rtdetr | 0.3 | **0.464** | 0.441 | 0.510 | 0.216 | 50 | 0.415 | 0.654 | 0.274 | 0.419 / 0.325 / 0.108 | 41.8 |
| complex_deformable_detr | 0.3 | **0.410** | 0.414 | 0.416 | 0.185 | 104 | 0.342 | 0.606 | 0.214 | 0.339 / 0.160 / 0.019 | 75.1 |
| yolo3d_deformable_detr | 0.3 | **0.409** | 0.413 | 0.418 | 0.129 | 126 | 0.360 | 0.605 | 0.213 | 0.298 / 0.130 / 0.007 | 76.7 |
| complex_sv4 | 0.9 | **0.400** | 0.392 | 0.417 | 0.027 | 167 | 0.331 | 0.582 | 0.219 | 0.333 / 0.377 / 0.400 | 41.2 |
| complex_v5 | 0.8 | **0.394** | 0.382 | 0.414 | 0.048 | 210 | 0.325 | 0.576 | 0.211 | 0.329 / 0.382 / 0.377 | 30.9 |
| complex_yolor | 0.9 | **0.393** | 0.382 | 0.418 | 0.111 | 158 | 0.328 | 0.569 | 0.218 | 0.337 / 0.375 / 0.393 | 59.5 |
| complex_v3 | 0.8 | **0.376** | 0.375 | 0.388 | 0.160 | 195 | 0.312 | 0.570 | 0.182 | 0.325 / 0.361 / 0.374 | 44.3 |
| complex_retinanet | 0.9 | **0.374** | 0.371 | 0.387 | 0.142 | 182 | 0.313 | 0.543 | 0.206 | 0.301 / 0.344 / 0.374 | 48.2 |
| complex_v7 | 0.9 | **0.372** | 0.371 | 0.383 | -0.138 | 197 | 0.312 | 0.559 | 0.184 | 0.322 / 0.356 / 0.372 | 37.6 |
| complex_efficientdet | 0.9 | **0.369** | 0.378 | 0.372 | 0.143 | 166 | 0.309 | 0.575 | 0.163 | 0.323 / 0.357 / 0.369 | 56.3 |
| complex_v4 | 0.9 | **0.356** | 0.341 | 0.381 | -0.166 | 186 | 0.299 | 0.500 | 0.211 | 0.259 / 0.311 / 0.356 | 46.3 |
| yolo3d_yolor | 0.9 | **0.326** | 0.287 | 0.388 | 0.090 | 222 | 0.282 | 0.458 | 0.193 | 0.258 / 0.305 / 0.326 | 39.0 |
| yolo3d_v7 | 0.9 | **0.320** | 0.292 | 0.373 | -0.242 | 206 | 0.277 | 0.467 | 0.173 | 0.235 / 0.286 / 0.320 | 37.7 |
| yolo3d_efficientdet | 0.9 | **0.304** | 0.275 | 0.350 | -0.091 | 191 | 0.259 | 0.461 | 0.146 | 0.255 / 0.284 / 0.304 | 31.2 |
| yolo3d_v3 | 0.9 | **0.298** | 0.259 | 0.363 | -0.024 | 180 | 0.254 | 0.449 | 0.146 | 0.247 / 0.275 / 0.298 | 40.6 |
| yolo3d_sv4 | 0.9 | **0.288** | 0.249 | 0.363 | 0.079 | 160 | 0.253 | 0.418 | 0.157 | 0.257 / 0.288 / 0.288 | 38.0 |
| yolo3d_v4 | 0.9 | **0.288** | 0.265 | 0.338 | -0.105 | 218 | 0.250 | 0.423 | 0.152 | 0.202 / 0.255 / 0.288 | 46.4 |
| yolo3d_v5 | 0.8 | **0.287** | 0.275 | 0.317 | -0.191 | 269 | 0.244 | 0.410 | 0.164 | 0.247 / 0.279 / 0.269 | 27.4 |
| complex_v2 | 0.7 | **0.283** | 0.258 | 0.321 | 0.110 | 65 | 0.238 | 0.502 | 0.065 | 0.275 / 0.283 / 0.222 | 33.9 |
| yolo3d_retinanet | 0.9 | **0.277** | 0.269 | 0.295 | -0.072 | 239 | 0.240 | 0.386 | 0.168 | 0.223 / 0.258 / 0.277 | 34.7 |
| yolo3d_v2 | 0.3 | **0.167** | 0.162 | 0.185 | -0.080 | 150 | 0.141 | 0.308 | 0.026 | 0.167 / 0.144 / 0.057 | 46.8 |

Findings: Faster R-CNN heads are best for both detector families (BEV HOTA about 0.54, car about 0.66 - close to the
best 2D car result, as a metric 3D trajectory); YOLO-type heads produce duplicate boxes around real objects (diagnosed:
matched offset 5 cm, so not a conversion error), which keeps their MOTA low; Complex-YOLO is equal or better than
YOLO3D with the same head.

## Usage

```bash
pip install -r requirements.txt
pip install "git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a"
# KITTI tracking velodyne / calib / label_02; 3D detections of the val and train sequences: Release v1.1
python tools/mot/run_tracker3d.py --det-thr 0.5                 # fixed threshold
python tools/mot/tune_ab3dmot.py                                # threshold per detector from train sequences
# dumping detections needs the detector code and weights of the OD_ComplexYOLO_* / OD_YOLO3D_* repositories:
python tools/mot/dump_dets3d.py --all
```

Part of the *Physical AI MOT* comparison of the **Statera Working Group** (v1.1 extension: UCMCTrack, CAMELTrack,
Seg2Track-SAM2, AB3DMOT). The v1.0 comparison (10 trackers x 25 detectors, JDT, E2E) and the v1.1 summary are in
`MOT_Detectors_KITTIMOTS` (`RESULTS.md`).

## Protocol (all MOT_* repositories)

- Online / causal only: every method sees the current frame and its own state; checked by re-running prefixes of
  every sequence (the output of frame t must not change when later frames are removed).
- Input 1216 x 384 (KITTI 1242 x 375 letterboxed), detectors trained from scratch on KITTI-MOTS train (12 sequences),
  30 epochs, Adam 1e-4, effective batch 8, seed 777 - the protocol of the *Physical AI Object Detection* series.
- Metrics: HOTA / DetA / AssA (TrackEval classes, car / pedestrian mean), MOTA, IDF1, IDSW.
- Latency: RTX 4070 Laptop GPU, batch 1, fp32, medians over val; pipeline = detector (forward + post-processing) +
  ReID (appearance trackers) + tracker (CPU); image decoding / letterboxing not included.
- Trackers run with their official default parameters; frame-based values given for 30 fps are scaled to 10 fps.
  A sensitivity table with one common score threshold for every tracker is reported separately (chosen on val, so it
  is analysis, not ranking).

## Data: KITTI-MOTS (box level, val = 9 sequences, 2,981 frames)

KITTI tracking images (21 training sequences) with instance / track annotations; boxes = tight boxes of the masks,
classes car and pedestrian, ignore regions (id 10000) neither rewarded nor penalised. The annotations are taken from
the **KITTI-STEP** panoptic maps (same sequences and split, car / person instances merged from KITTI-MOTS; frames,
track counts and pedestrian masks match the KITTI-MOTS paper, car masks differ by 39 of 26,899).

```bash
bash tools/mot/get_kitti_mots.sh            # KITTI tracking images + annotations (KITTI-STEP fallback)
python tools/mot/kittimots_to_voc.py        # VOC-layout copy with tracks + count report
UNIFIED_IMAGE_DATASET=kittimots python tools/mot/count_loader.py     # counts as the loaders read them
```

**Licence of the data and of every trained weight here: KITTI / KITTI-MOTS / KITTI-STEP are CC BY-NC-SA 3.0
(non-commercial).**

Results of this repository: `results/` (evaluation JSON per detector). Sources and licences: `THIRD_PARTY.md`.

## Global Collaboration

Hills Robotics is currently participating in multiple projects in Korea and internationally.

We are actively interested in building relationships with organizations that can complement our technology with global sales networks, distribution channels, system integration capabilities, manufacturing ecosystems, AI infrastructure, robotics technologies, research capabilities, or strategic investment.

Several technology and business collaborations are already underway. Some partners, projects, system architectures, and commercial programs cannot be publicly disclosed because they are subject to confidentiality and business agreements.

We believe that the emerging Physical AI ecosystem cannot be built by a single company.
It requires a global network connecting robotics companies, AI researchers, semiconductor and computing companies, sensor manufacturers, software developers, universities, system integrators, industrial customers, investors, and global distribution partners.

We have now begun our serious exploration of the Physical AI frontier, and we welcome connections with people and organizations who want to build this ecosystem together.

## Research, Contribution, and Donation

The repositories of the Statera Working Group are intended to contribute to the broader technical discussion around Physical AI.

Researchers, engineers, developers, companies, universities, and independent contributors interested in these subjects are welcome to connect with us.

If you would like to collaborate on Physical AI or autonomous robotics, participate in research or technology development, discuss international business cooperation, propose technology or platform integration, contribute technical knowledge, or support this GitHub knowledge initiative through sponsorship or donation, please contact us.

## Contact

**Richard Minsoo Go**
Head of AI, Hills Robotics
Statera Working Group

Personal / Research Contact:
[gominsoo888@gmail.com](mailto:gominsoo888@gmail.com)

Hills Robotics / Business Contact:
[minsoo.go@hillslogis.com](mailto:minsoo.go@hillslogis.com)

Hills Robotics:
https://hillslogis.com/

For inquiries regarding next-generation outdoor autonomous robotic platforms, Physical AI Spider Grid, Intelligence Cabinet, Edge AI, or On-Premise AI architectures that are not yet publicly presented on the company website, please contact us directly.
