#!/usr/bin/env python3
"""Write nvdsanalytics config from coordinates.

Example:
python3 tools/write_nvdsanalytics_config.py \
  --width 1920 --height 1080 \
  --roi "100,100 1800,100 1800,1000 100,1000" \
  --line-in "900,800 1200,800" --dir-in "900,1000 900,700" \
  --out configs/config_nvdsanalytics_fence.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path


def parse_points(s: str) -> str:
    pts = []
    for item in s.strip().split():
        x, y = item.split(",")
        pts.extend([str(int(float(x))), str(int(float(y)))])
    return ";".join(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--roi", required=True, help='Polygon: "x1,y1 x2,y2 x3,y3 ..."')
    ap.add_argument("--line-in", default=None, help='Line: "x1,y1 x2,y2"')
    ap.add_argument("--dir-in", default=None, help='Direction vector: "x1,y1 x2,y2"')
    ap.add_argument("--line-out", default=None)
    ap.add_argument("--dir-out", default=None)
    ap.add_argument("--class-id", default="0")
    ap.add_argument("--out", default="configs/config_nvdsanalytics_fence.txt")
    args = ap.parse_args()

    parts = [
        "[property]",
        "enable=1",
        f"config-width={args.width}",
        f"config-height={args.height}",
        "osd-mode=2",
        "display-font-size=18",
        "",
        "[roi-filtering-stream-0]",
        "enable=1",
        f"class-id={args.class_id}",
        "inverse-roi=0",
        f"roi-FENCE={parse_points(args.roi)}",
        "",
    ]
    if args.line_in and args.dir_in:
        parts += [
            "[line-crossing-stream-0]",
            "enable=1",
            f"class-id={args.class_id}",
            "extended=0",
            "mode=balanced",
            f"line-crossing-IN={parse_points(args.dir_in)};{parse_points(args.line_in)}",
        ]
        if args.line_out and args.dir_out:
            parts.append(f"line-crossing-OUT={parse_points(args.dir_out)};{parse_points(args.line_out)}")
        parts.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote: {out}")

if __name__ == "__main__": main()
