#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path

FPS_PATTERNS = [
    re.compile(r"FPS\s*[:=]\s*(?P<fps>[0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"stream\s*0\s*[:=]\s*(?P<fps>[0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"fps\s*\(\s*0\s*\)\s*[:=]\s*(?P<fps>[0-9]+(?:\.[0-9]+)?)", re.I),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    text = Path(args.log).read_text(errors="ignore", encoding="utf-8")
    fps = []
    for line in text.splitlines():
        if "FPS" not in line and "fps" not in line and "PERF" not in line:
            continue
        for pat in FPS_PATTERNS:
            m = pat.search(line)
            if m:
                fps.append(float(m.group("fps")))
                break
    data = {
        "log": args.log,
        "fps_samples": fps,
        "fps_count": len(fps),
        "fps_avg": sum(fps)/len(fps) if fps else None,
        "fps_max": max(fps) if fps else None,
        "fps_min": min(fps) if fps else None,
    }
    js = json.dumps(data, indent=2)
    print(js)
    if args.out:
        Path(args.out).write_text(js, encoding="utf-8")

if __name__ == "__main__": main()
