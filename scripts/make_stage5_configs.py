#!/usr/bin/env python3
"""
Generate clean DeepStream configs for Stage5.

Baseline modes:
  save: direct local file -> DeepStream -> MP4
  live: MediaMTX cam RTSP -> DeepStream YOLOv8s + NvDCF tracker -> RTSP ds-test

Fence modes add nvdsanalytics ROI overlay/count using configs/fence/main_zone.txt.
Live browser playback is handled outside DeepStream by ffmpeg repacketize/re-encode
and MediaMTX WebRTC path /out/.
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from urllib.parse import quote
from typing import Iterable


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def parse_main_zone(path: Path) -> list[list[int]]:
    """Parse MAIN_ZONE = [[x,y], ...] from the task txt file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"MAIN_ZONE\s*=\s*(\[[\s\S]*?\])\s*(?:#|$)", text)
    if not m:
        # More robust fallback: first list assigned after MAIN_ZONE.
        m = re.search(r"MAIN_ZONE\s*=\s*(\[[\s\S]*?\])", text)
    if not m:
        raise ValueError(f"Could not find MAIN_ZONE in {path}")
    # Regex with nested lists is tricky; use line slicing from assignment until matching bracket count.
    start = text.find("[", text.find("MAIN_ZONE"))
    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError(f"Could not parse MAIN_ZONE bracket block in {path}")
    pts = ast.literal_eval(text[start:end])
    if len(pts) < 3:
        raise ValueError("MAIN_ZONE must contain at least 3 points")
    return [[int(round(float(x))), int(round(float(y)))] for x, y in pts]


def flatten_points(pts: Iterable[Iterable[int]]) -> str:
    vals: list[str] = []
    for x, y in pts:
        vals += [str(int(x)), str(int(y))]
    return ";".join(vals)


def scale_points(pts: list[list[int]], src_w: int, src_h: int, dst_w: int, dst_h: int) -> list[list[int]]:
    """Scale polygon coordinates from the annotation canvas to DeepStream streammux canvas.

    The task txt coordinates were drawn on a 2560x1440 canvas, while this Stage5
    pipeline processes frames at 1920x1080 before tiled-display downscaling.
    Using 2048x1536 made the ROI too wide/right-shifted in the 16:9 output.
    """
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out: list[list[int]] = []
    for x, y in pts:
        xx = max(0, min(dst_w - 1, int(round(x * sx))))
        yy = max(0, min(dst_h - 1, int(round(y * sy))))
        out.append([xx, yy])
    return out


def make_fence_analytics_config(
    root: Path,
    width: int,
    height: int,
    zone_file: Path,
    zone_src_width: int = 2560,
    zone_src_height: int = 1440,
) -> str:
    raw_pts = parse_main_zone(zone_file)

    # v7: user-approved visual polygon. Coordinates are in the DeepStream
    # streammux canvas (1920x1080). This is intentionally NOT scaled from the
    # original task txt because the user corrected the ROI manually from the
    # real output frame.
    base_pts = [[770, 240], [1460, 280], [1100, 720], [135, 620]]
    if width == 1920 and height == 1080:
        pts = base_pts
    else:
        pts = scale_points(base_pts, 1920, 1080, width, height)

    roi = flatten_points(pts)
    raw_roi = flatten_points(raw_pts)

    # Add ENTER/EXIT lines for all four polygon edges. The direction vector for
    # ENTER points from outside of each edge toward the polygon centroid. EXIT is
    # the reverse direction. This approximates: "when the person's foot crosses
    # any fence edge into the polygon => ENTER; when it crosses out => EXIT".
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)

    def clamp_pt(x: float, y: float) -> list[int]:
        return [max(0, min(width - 1, int(round(x)))), max(0, min(height - 1, int(round(y))))]

    line_items: list[str] = []
    for i, (a, b) in enumerate(zip(pts, pts[1:] + pts[:1]), start=1):
        x1, y1 = a
        x2, y2 = b
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        vx = cx - mx
        vy = cy - my
        norm = (vx * vx + vy * vy) ** 0.5 or 1.0
        # Direction arrow length. It only defines crossing direction; the actual
        # crossing line is the polygon edge below.
        arrow = 120.0
        ux, uy = vx / norm, vy / norm
        outside = clamp_pt(mx - ux * arrow / 2.0, my - uy * arrow / 2.0)
        inside = clamp_pt(mx + ux * arrow / 2.0, my + uy * arrow / 2.0)
        edge = flatten_points([a, b])
        enter_dir = flatten_points([outside, inside])
        exit_dir = flatten_points([inside, outside])
        line_items.append(f"line-crossing-ENTER_E{i}={enter_dir};{edge}")
        line_items.append(f"line-crossing-EXIT_E{i}={exit_dir};{edge}")

    line_block = "\n".join(line_items)
    return f"""
[property]
enable=1
config-width={width}
config-height={height}
osd-mode=2
display-font-size=18

# Raw polygon from configs/fence/main_zone.txt was annotated on {zone_src_width}x{zone_src_height}:
# {raw_roi}
# v7 user-corrected polygon used by DeepStream analytics at {width}x{height}:
# {roi}
# Rule target: person is considered inside when foot point is inside this polygon.
# With nvdsanalytics, ROI/line logic is an approximation; exact 65%% bbox-overlap
# and foot-point state machine require a custom pad-probe/app stage.
[roi-filtering-stream-0]
enable=1
class-id=0
inverse-roi=0
roi-MAIN_ZONE={roi}

[line-crossing-stream-0]
enable=1
class-id=0
extended=0
mode=balanced
# Four-edge crossing. ENTER_* means outside -> inside through that edge.
# EXIT_* means inside -> outside through that edge.
{line_block}
"""

def make_infer_config(root: Path, yolo_repo: Path) -> str:
    return f"""
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file={root}/models/yolo/yolov8s.onnx
model-engine-file={root}/models/yolo/model_b1_gpu0_fp16.engine
labelfile-path={yolo_repo}/labels_id.txt
batch-size=1
network-mode=2
num-detected-classes=1
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path={yolo_repo}/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
engine-create-func-name=NvDsInferYoloCudaEngineGet

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.45
topk=50
"""


def app_config(
    *,
    root: Path,
    source_type: int,
    source_uri: str,
    live_source: int,
    sink0_enable: int,
    sink1_enable: int,
    sink0_sync: int,
    sink1_sync: int,
    output_file: str,
    infer_config: Path,
    tracker_config: Path,
    width: int,
    height: int,
    analytics_enable: bool = False,
    analytics_config: Path | None = None,
    tracker_enable: int = 1,
) -> str:
    rtsp_extra = """
select-rtp-protocol=4
latency=0
""" if source_type == 4 else ""

    analytics_block = ""
    if analytics_enable:
        if analytics_config is None:
            raise ValueError("analytics_config is required when analytics_enable=True")
        analytics_block = f"""
[nvds-analytics]
enable=1
config-file={analytics_config}
"""

    return f"""
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5

[tiled-display]
enable=1
rows=1
columns=1
width=1280
height=720
gpu-id=0
nvbuf-memory-type=0

[source0]
enable=1
type={source_type}
uri={source_uri}
num-sources=1
gpu-id=0
cudadec-memtype=0
{rtsp_extra}
[sink0]
enable={sink0_enable}
type=3
codec=1
container=1
sync={sink0_sync}
bitrate=8000000
output-file={output_file}
gpu-id=0
nvbuf-memory-type=0

[osd]
enable=1
gpu-id=0
# Tracker mode: show tracking ID overlay. The class label file is set to "ID" so it will not show "person".
display-text=1
border-width=3
text-size=15
text-color=1;1;1;1;
text-bg-color=0.3;0.3;0.3;1
font=Serif
show-clock=0
clock-x-offset=800
clock-y-offset=820
clock-text-size=12
clock-color=1;0;0;0
nvbuf-memory-type=0

[streammux]
gpu-id=0
live-source={live_source}
batch-size=1
batched-push-timeout=40000
width={width}
height={height}
enable-padding=0
nvbuf-memory-type=0

[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file={infer_config}

[tracker]
enable={tracker_enable}
tracker-width=960
tracker-height=544
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file={tracker_config}
display-tracking-id=1
{analytics_block}
[tests]
file-loop=0

[sink1]
enable={sink1_enable}
type=4
codec=1
sync={sink1_sync}
bitrate=4000000
rtsp-port=8554
udp-port=5400
profile=0
source-id=0
gpu-id=0
nvbuf-memory-type=0
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/workspace/stage5")
    p.add_argument("--video-rel", default="data/Hike Vision.mp4")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fence-zone", default="configs/fence/main_zone.txt")
    p.add_argument("--fence-zone-width", type=int, default=2560)
    p.add_argument("--fence-zone-height", type=int, default=1440)
    args = p.parse_args()

    root = Path(args.root)
    cfg_dir = root / "configs" / "deepstream_yolov8s"
    yolo_repo = root / "third_party" / "DeepStream-Yolo"
    infer_path = cfg_dir / "config_infer_primary_yolov8s_person.txt"
    tracker_path = root / "configs" / "tracker" / "config_tracker_NvDCF_accuracy.yml"
    analytics_path = root / "configs" / "config_nvdsanalytics_fence.txt"
    zone_file = root / args.fence_zone
    save_path = cfg_dir / "deepstream_app_save_file.txt"
    live_path = cfg_dir / "deepstream_app_live_rtsp_cam.txt"
    fence_save_path = cfg_dir / "deepstream_app_fence_save_file.txt"
    fence_live_path = cfg_dir / "deepstream_app_fence_live_rtsp_cam.txt"

    output_file = str(root / "outputs" / "yolov8s_person_output.mp4")
    fence_output_file = str(root / "outputs" / "yolov8s_person_fence_output.mp4")
    # Encode spaces/special chars in file URI so DeepStream/GStreamer can open
    # paths such as data/Hike Vision.mp4 reliably.
    video_file_uri = "file://" + quote(str(root / args.video_rel), safe="/")

    write_text(infer_path, make_infer_config(root, yolo_repo))
    if zone_file.exists():
        write_text(analytics_path, make_fence_analytics_config(root, args.width, args.height, zone_file, args.fence_zone_width, args.fence_zone_height))
    else:
        # Keep a safe placeholder instead of failing normal live/save.
        write_text(analytics_path, make_fence_analytics_config(root, args.width, args.height, root / "configs" / "fence" / "main_zone.txt", args.fence_zone_width, args.fence_zone_height))

    common = dict(root=root, infer_config=infer_path, tracker_config=tracker_path, width=args.width, height=args.height)

    # Baseline save/live configs remain unchanged.
    write_text(save_path, app_config(
        **common, source_type=2, source_uri=video_file_uri, live_source=0,
        sink0_enable=1, sink1_enable=0, sink0_sync=0, sink1_sync=0, output_file=output_file,
    ))
    write_text(live_path, app_config(
        **common, source_type=4, source_uri="rtsp://host.docker.internal:8554/cam", live_source=1,
        sink0_enable=0, sink1_enable=1, sink0_sync=0, sink1_sync=0, output_file=output_file,
    ))

    # Fence configs use the same NvDCF tracker as normal live/save.
    # This keeps stable DeepStream tracking IDs before nvdsanalytics.
    fence_common = dict(common)
    write_text(fence_save_path, app_config(
        **fence_common, source_type=2, source_uri=video_file_uri, live_source=0,
        sink0_enable=1, sink1_enable=0, sink0_sync=0, sink1_sync=0, output_file=fence_output_file,
        analytics_enable=True, analytics_config=analytics_path, tracker_enable=1,
    ))
    write_text(fence_live_path, app_config(
        **fence_common, source_type=4, source_uri="rtsp://host.docker.internal:8554/cam", live_source=1,
        sink0_enable=0, sink1_enable=1, sink0_sync=0, sink1_sync=0, output_file=fence_output_file,
        analytics_enable=True, analytics_config=analytics_path, tracker_enable=1,
    ))

    for path in [infer_path, analytics_path, save_path, live_path, fence_save_path, fence_live_path]:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
