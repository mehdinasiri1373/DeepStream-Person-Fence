#!/usr/bin/env python3
"""Custom virtual fence runner using YOLO detections + foot-point polygon logic.

This is intentionally independent from nvdsanalytics for the fence decision:
  foot = bottom-center of the axis-aligned detector bbox
  inside = point_in_polygon(foot, polygon)
  ENTER/EXIT = transition of inside state for a stable track ID

It writes rendered BGR frames as rawvideo to stdout, so the host run.sh can encode
with an ffmpeg Docker container to MP4 or RTSP/WebRTC.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

Point = Tuple[int, int]
Box = Tuple[float, float, float, float]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def read_polygon(path: str | Path) -> List[Point]:
    pts: List[Point] = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace("[", "").replace("]", "").replace(";", ",")
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            pts.append((int(float(parts[0])), int(float(parts[1]))))
    if len(pts) < 3:
        raise ValueError(f"Polygon file must contain at least 3 points: {path}")
    return pts


def letterbox_to_canvas(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def point_in_poly(pt: Point, poly: List[Point]) -> bool:
    # Ray casting. Boundary counts as inside.
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # On segment check
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        cross = (x - x1) * dy - (y - y1) * dx
        if abs(cross) < 1e-6:
            dot = (x - x1) * (x - x2) + (y - y1) * (y - y2)
            if dot <= 0:
                return True
        if ((y1 > y) != (y2 > y)):
            xinters = (dx * (y - y1) / (dy + 1e-12)) + x1
            if x <= xinters:
                inside = not inside
    return inside


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-9))


def foot_of(box: Box) -> Point:
    x1, y1, x2, y2 = box
    return (int(round((x1 + x2) / 2.0)), int(round(y2)))


def center_of(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Track:
    tid: int
    box: Box
    hits: int = 1
    age: int = 1
    missed: int = 0
    prev_inside: Optional[bool] = None
    last_inside: Optional[bool] = None
    last_seen_frame: int = 0
    smooth_box: Box = field(default_factory=lambda: (0, 0, 0, 0))

    def __post_init__(self):
        self.smooth_box = self.box

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2

    def update(self, box: Box, frame_idx: int, alpha: float = 0.65):
        self.box = box
        # Mild smoothing for display only, keeps box axis-aligned.
        sx1, sy1, sx2, sy2 = self.smooth_box
        x1, y1, x2, y2 = box
        self.smooth_box = (
            alpha * x1 + (1 - alpha) * sx1,
            alpha * y1 + (1 - alpha) * sy1,
            alpha * x2 + (1 - alpha) * sx2,
            alpha * y2 + (1 - alpha) * sy2,
        )
        self.hits += 1
        self.age += 1
        self.missed = 0
        self.last_seen_frame = frame_idx


def match_tracks(tracks: List[Track], detections: List[Box], frame_idx: int) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    candidates: List[Tuple[float, int, int]] = []
    for ti, tr in enumerate(tracks):
        for di, det in enumerate(detections):
            iv = iou(tr.box, det)
            fx1, fy1 = foot_of(tr.box)
            fx2, fy2 = foot_of(det)
            dist = math.hypot(fx1 - fx2, fy1 - fy2)
            dist_score = max(0.0, 1.0 - dist / 260.0)
            # Prefer IoU, but allow foot-point proximity to reconnect after short detector jitter.
            score = 0.65 * iv + 0.35 * dist_score
            if iv > 0.03 or dist < 120:
                candidates.append((score, ti, di))
    candidates.sort(reverse=True, key=lambda x: x[0])
    used_t, used_d = set(), set()
    matches: List[Tuple[int, int]] = []
    for score, ti, di in candidates:
        if score < 0.22:
            continue
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti); used_d.add(di)
        matches.append((ti, di))
    unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_d = [i for i in range(len(detections)) if i not in used_d]
    return matches, unmatched_t, unmatched_d


def draw_overlay(frame: np.ndarray, polygon: List[Point], tracks: List[Track], occupancy: int, enters: int, exits: int, frame_idx: int):
    overlay = frame.copy()
    poly_np = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [poly_np], (0, 180, 0))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [poly_np], True, (0, 255, 0), 3, cv2.LINE_AA)
    for p in polygon:
        cv2.circle(frame, p, 6, (0, 255, 0), -1, cv2.LINE_AA)

    # Compact HUD only. No ENTER_E labels.
    hud = f"IN:{occupancy}  ENTER:{enters}  EXIT:{exits}"
    cv2.rectangle(frame, (18, 86), (18 + 420, 132), (0, 0, 0), -1)
    cv2.putText(frame, hud, (28, 119), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    for tr in tracks:
        if tr.missed != 0 or not tr.confirmed:
            continue
        box = tr.smooth_box
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        foot = foot_of(box)
        inside = bool(tr.last_inside)
        color = (0, 220, 0) if inside else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.circle(frame, foot, 5, (255, 255, 0), -1, cv2.LINE_AA)
        label = f"ID {tr.tid}" + (" IN" if inside else "")
        # Small label, not huge labels on the fence edges.
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ytxt = max(18, y1 - 6)
        cv2.rectangle(frame, (x1, ytxt - th - 6), (x1 + tw + 8, ytxt + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 4, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="/workspace/stage5/models/yolo/yolov8s.pt")
    ap.add_argument("--polygon", default="/workspace/stage5/configs/fence/foot_polygon.txt")
    ap.add_argument("--csv", default="/workspace/stage5/outputs/fence_events.csv")
    ap.add_argument("--status-csv", default="/workspace/stage5/outputs/fence_status.csv")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--max-missed", type=int, default=45)
    args = ap.parse_args()

    polygon = read_polygon(args.polygon)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    events_f = open(args.csv, "w", newline="", encoding="utf-8")
    status_f = open(args.status_csv, "w", newline="", encoding="utf-8")
    events_w = csv.writer(events_f)
    status_w = csv.writer(status_f)
    events_w.writerow(["frame", "time_sec", "track_id", "event", "occupancy", "foot_x", "foot_y"])
    status_w.writerow(["frame", "time_sec", "track_id", "inside", "foot_x", "foot_y", "x1", "y1", "x2", "y2"])

    eprint(f"[fence-custom] Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    try:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    eprint(f"[fence-custom] device={device} video={args.video}")
    eprint(f"[fence-custom] polygon={polygon}")

    tracks: List[Track] = []
    next_id = 1
    enters = 0
    exits = 0
    frame_idx = 0

    while True:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {args.video}")
        while True:
            ok, frame0 = cap.read()
            if not ok:
                break
            frame_idx += 1
            frame = letterbox_to_canvas(frame0, args.width, args.height)

            results = model.predict(frame, imgsz=640, conf=args.conf, classes=[0], device=device, verbose=False)[0]
            detections: List[Box] = []
            if results.boxes is not None and len(results.boxes) > 0:
                xyxy = results.boxes.xyxy.detach().cpu().numpy()
                confs = results.boxes.conf.detach().cpu().numpy()
                # Keep reasonable person boxes, reject tiny false positives.
                for b, c in zip(xyxy, confs):
                    x1, y1, x2, y2 = [float(v) for v in b]
                    if (x2 - x1) < 18 or (y2 - y1) < 35:
                        continue
                    detections.append((x1, y1, x2, y2))

            matches, unmatched_t, unmatched_d = match_tracks(tracks, detections, frame_idx)
            for ti, di in matches:
                tracks[ti].update(detections[di], frame_idx)
            for ti in unmatched_t:
                tracks[ti].missed += 1
                tracks[ti].age += 1
            for di in unmatched_d:
                tr = Track(next_id, detections[di], last_seen_frame=frame_idx)
                next_id += 1
                tracks.append(tr)

            # Drop old lost tracks.
            tracks = [t for t in tracks if t.missed <= args.max_missed]

            occupancy = 0
            active_tracks: List[Track] = []
            for tr in tracks:
                if tr.missed != 0 or not tr.confirmed:
                    continue
                foot = foot_of(tr.smooth_box)
                inside = point_in_poly(foot, polygon)
                prev = tr.last_inside
                if prev is None:
                    tr.prev_inside = inside
                    tr.last_inside = inside
                elif prev != inside:
                    event = "ENTER" if inside else "EXIT"
                    if inside:
                        enters += 1
                    else:
                        exits += 1
                    tr.prev_inside = prev
                    tr.last_inside = inside
                    events_w.writerow([frame_idx, f"{frame_idx/args.fps:.3f}", tr.tid, event, occupancy, foot[0], foot[1]])
                    events_f.flush()
                else:
                    tr.last_inside = inside
                if tr.last_inside:
                    occupancy += 1
                x1, y1, x2, y2 = [int(round(v)) for v in tr.smooth_box]
                status_w.writerow([frame_idx, f"{frame_idx/args.fps:.3f}", tr.tid, int(bool(tr.last_inside)), foot[0], foot[1], x1, y1, x2, y2])
                active_tracks.append(tr)
            status_f.flush()

            draw_overlay(frame, polygon, active_tracks, occupancy, enters, exits, frame_idx)
            sys.stdout.buffer.write(frame.tobytes())
            if frame_idx % 100 == 0:
                sys.stdout.flush()
                eprint(f"[fence-custom] frame={frame_idx} active={len(active_tracks)} occ={occupancy} enter={enters} exit={exits} next_id={next_id}")
        cap.release()
        if not args.loop:
            break
    events_f.close()
    status_f.close()
    eprint(f"[fence-custom] done frames={frame_idx} unique_ids={next_id-1} enter={enters} exit={exits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
