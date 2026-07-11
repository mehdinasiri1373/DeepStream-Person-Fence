#!/usr/bin/env python3
"""Interactive ROI/line picker using OpenCV.

Keys:
- Left click: add point
- r: reset current points
- s: save ROI config using current polygon only
- q/ESC: quit

This is intentionally simple. For exact production config, use write_nvdsanalytics_config.py with known coordinates.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="configs/config_nvdsanalytics_fence.txt")
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read video")
    pts = []
    win = "Pick ROI polygon - click points, s save, r reset, q quit"
    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y))
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, cb)
    while True:
        img = frame.copy()
        for p in pts:
            cv2.circle(img, p, 6, (0,255,255), -1)
        if len(pts) > 1:
            for a,b in zip(pts, pts[1:]): cv2.line(img, a, b, (0,255,255), 2)
        if len(pts) > 2:
            cv2.line(img, pts[-1], pts[0], (0,255,255), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord('q')): break
        if k == ord('r'): pts.clear()
        if k == ord('s'):
            if len(pts) < 3:
                print("Need at least 3 points")
                continue
            roi = " ".join(f"{x},{y}" for x,y in pts)
            cmd = [sys.executable, str(Path(__file__).with_name("write_nvdsanalytics_config.py")), "--roi", roi, "--out", args.out]
            subprocess.check_call(cmd)
            break
    cv2.destroyAllWindows()

if __name__ == "__main__": main()
