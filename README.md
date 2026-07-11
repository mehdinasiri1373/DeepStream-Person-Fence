# DeepStream Person Fence

> GPU-accelerated person detection, multi-object tracking, polygon zone monitoring, live streaming, and event logging with NVIDIA DeepStream, YOLOv8s, NvDCF, and a custom bottom-center foot-point fence engine.

## Overview

DeepStream Person Fence is a video analytics project for detecting people, assigning stable tracking IDs, and monitoring whether each person is inside or outside a configured polygonal zone.

The project is operated from a single shell entry point: `run.sh`. On a compatible GPU machine, the runner prepares the Docker runtime, builds the YOLO integration for DeepStream, exports the model, creates the TensorRT engine, runs NvDCF tracking, applies the custom fence logic, and produces either a saved annotated MP4 file or a browser-based live stream.

The fence decision is based on the bottom-center point of each tracked person box. This point is used as the person's approximate foot-point on the ground plane, which is more suitable for zone monitoring than the bounding-box center.

## Key features

- YOLOv8s person detection.
- NVIDIA DeepStream accelerated inference.
- NvDCF multi-object tracking.
- Clean sequential display IDs for tracked people.
- Polygon-based inside/outside monitoring.
- Bottom-center foot-point logic for accurate fence decisions.
- ENTER and EXIT event detection.
- Per-frame status logging.
- Annotated MP4 export.
- Browser live streaming through MediaMTX WebRTC.
- Automatic first-run preparation of models, parsers, configs, and runtime assets.

## System architecture

```text
Input video
    |
    v
DeepStream runtime
    |
    +-- Decode
    |
    +-- nvstreammux
    |
    +-- nvinfer
    |     YOLOv8s person detector
    |
    +-- nvtracker
    |     NvDCF tracker
    |
    +-- Python metadata probe
    |     Foot-point extraction
    |     Polygon inside/outside test
    |     ENTER / EXIT state machine
    |     Clean ID mapping
    |     CSV logging
    |     Overlay metadata
    |
    +-- nvdsosd
    |     Final visual rendering
    |
    +-- Output
          |
          +-- Annotated MP4 file
          |
          +-- UDP MPEG-TS -> FFmpeg -> MediaMTX -> WebRTC
```

## Processing pipeline

The analytics pipeline uses DeepStream for GPU inference and tracking, then applies custom Python logic to DeepStream metadata.

```text
source
  -> decode
  -> nvstreammux
  -> nvinfer YOLOv8s
  -> nvtracker NvDCF
  -> Python pyds metadata probe
  -> foot-point fence logic
  -> custom overlay
  -> nvdsosd
  -> encoder / file / stream
```

The project intentionally keeps the fence logic outside `nvdsanalytics`. This provides full control over the point used for zone decisions, the ID mapping, the ENTER/EXIT state machine, and the final visual overlay.

## Fence logic

For every tracked person, the system computes the bottom-center point of the bounding box:

```text
foot_x = bbox_left + bbox_width / 2
foot_y = bbox_top + bbox_height
```

That point is tested against the configured polygon.

The result controls:

- inside/outside state
- ENTER events
- EXIT events
- occupancy count
- box color
- label text
- CSV logging

The visible person box remains a normal rectangle. The foot-point is used only for zone logic.

## Modes

| Mode | Command | Purpose |
|---|---|---|
| `fence-save` | `./run.sh --mode fence-save --video "data/Hike Vision.mp4"` | Runs detection, NvDCF tracking, fence logic, and saves an annotated MP4. |
| `fence-live` | `./run.sh --mode fence-live --video "data/Hike Vision.mp4"` | Runs detection, NvDCF tracking, fence logic, and streams the annotated output to the browser. |

For the full analytics workflow, use `fence-save` and `fence-live`.

## Requirements

### Host machine

A compatible machine should have:

- Linux, or Windows with WSL2.
- Docker Engine or Docker Desktop.
- NVIDIA GPU.
- NVIDIA driver installed on the host.
- Docker GPU support through `--gpus all`.
- Internet access for the first run.
- Enough disk space for Docker images, model files, TensorRT engines, logs, and outputs.

### Recommended setup

- Windows 10/11 with WSL2, or native Linux.
- Docker Desktop with WSL integration enabled.
- Recent NVIDIA driver.
- NVIDIA GPU with enough memory for DeepStream inference.
- A stable internet connection for the first preparation step.

### GPU check

Before running the project, verify that Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this command cannot see the GPU, fix Docker/NVIDIA integration before running the project.

## Project structure

```text
deepstream_person_fence/
├── run.sh
├── README.md
├── configs/
│   ├── deepstream_yolov8s/
│   │   ├── config_infer_primary_yolov8s_person.txt
│   │   ├── deepstream_app_save_file.txt
│   │   ├── deepstream_app_live_rtsp_cam.txt
│   │   ├── deepstream_app_fence_save_file.txt
│   │   └── deepstream_app_fence_live_rtsp_cam.txt
│   ├── fence/
│   │   ├── foot_polygon.txt
│   │   └── main_zone.txt
│   └── tracker/
│       ├── config_tracker_NvDCF_accuracy.yml
│       ├── config_tracker_NvDCF_perf.yml
│       ├── config_tracker_NvSORT.yml
│       ├── config_tracker_NvDeepSORT.yml
│       └── config_tracker_IOU.yml
├── data/
│   └── .gitkeep
├── docs/
│   └── LIVE_WEBRTC_NOTES.md
├── models/
│   └── yolo/
│       └── .gitkeep
├── outputs/
│   └── .gitkeep
├── scripts/
│   ├── container_entry.sh
│   ├── fence_nvdcf_deepstream.py
│   ├── fence_footpoint_yolo.py
│   ├── make_configs.py
│   └── parse_deepstream_perf.py
├── third_party/
│   └── .gitkeep
└── tools/
    ├── pick_roi_line.py
    └── write_nvdsanalytics_config.py
```

## Quick start

Place your input video under `data/`:

```text
data/Hike Vision.mp4
```

Run the save pipeline:

```bash
chmod +x run.sh
./run.sh --mode fence-save --video "data/Hike Vision.mp4"
```

Run the live pipeline:

```bash
chmod +x run.sh
./run.sh --mode fence-live --video "data/Hike Vision.mp4"
```

Open the live stream in a browser:

```text
http://localhost:8889/out/
```

Stop all runtime containers:

```bash
./run.sh --stop
```

View logs:

```bash
./run.sh --logs
```

## First run behavior

On the first run, the project prepares the runtime automatically. Depending on what is already cached, it may:

- Pull the required Docker images.
- Start or reuse the DeepStream container.
- Install required Linux packages inside the container.
- Clone and build the DeepStream YOLO parser.
- Create the Python environment used for model export.
- Download the YOLOv8s model.
- Export the model to a DeepStream-compatible ONNX file.
- Build or load the TensorRT engine.
- Generate DeepStream config files.
- Prepare Python DeepStream bindings.
- Create output directories and logs.

The first run can take several minutes. Later runs are faster because the model, parser, engine, and configs are reused.

## Running on a new machine

When moving the project to another GPU machine, rebuild runtime artifacts for that machine:

```bash
./run.sh --reset --force-rebuild --mode fence-save --video "data/Hike Vision.mp4"
```

or:

```bash
./run.sh --reset --force-rebuild --mode fence-live --video "data/Hike Vision.mp4"
```

This is recommended because TensorRT engines are hardware and driver dependent.

## Output files

The project writes results under `outputs/`.

Common outputs:

```text
outputs/yolov8s_person_fence_output.mp4
outputs/fence_events.csv
outputs/fence_status.csv
outputs/ds_fence_nvdcf_save.log
outputs/ds_fence_nvdcf_live.log
```

### Annotated MP4

`fence-save` writes an annotated video with:

- person bounding boxes
- clean display IDs
- inside/outside labels
- fence polygon
- foot-points
- occupancy/event counters

### Event CSV

`outputs/fence_events.csv` stores zone transition events.

Typical columns:

```text
event_id,frame,time_sec,display_id,raw_track_id,event,foot_x,foot_y
```

Example event values:

```text
ENTER
EXIT
```

### Status CSV

`outputs/fence_status.csv` stores per-frame object state.

Typical columns:

```text
frame,time_sec,display_id,raw_track_id,inside,foot_x,foot_y,bbox_left,bbox_top,bbox_width,bbox_height
```

Use this file for downstream analytics, auditing, plots, and occupancy summaries.

## Fence configuration

The fence polygon is configured here:

```text
configs/fence/foot_polygon.txt
```

The file contains polygon points in image coordinates. The points are interpreted as a closed polygon.

Example:

```text
770,240
1460,280
1100,720
135,620
```

The polygon should be defined in the same coordinate system as the processed video frame.

## Visual behavior

The final overlay uses the following conventions:

- Green box: person is inside the polygon.
- Red box: person is outside the polygon.
- Label: clean display ID and inside/outside state.
- Small point: bottom-center foot-point used for fence decisions.
- Polygon: monitored fence area.

The tracker may produce large internal IDs. The project maps those raw IDs to small sequential display IDs such as `ID 1`, `ID 2`, and `ID 3`. Raw tracker IDs are still preserved in CSV files for debugging and traceability.

## Live streaming

The live path is designed for browser viewing.

```text
DeepStream annotated output
  -> UDP MPEG-TS
  -> FFmpeg bridge
  -> MediaMTX
  -> WebRTC browser stream
```

Open:

```text
http://localhost:8889/out/
```

The live mode is rate-limited to real-time playback for file inputs, so a 25 FPS input video is displayed at approximately 25 FPS instead of being processed as fast as the GPU can run.

## Save mode

Save mode runs the same analytics logic and writes a final MP4 file. It is intended for offline processing, review, reports, and reproducible tests.

```bash
./run.sh --mode fence-save --video "data/Hike Vision.mp4"
```

Check the output:

```bash
ls -lh outputs/yolov8s_person_fence_output.mp4
ls -lh outputs/fence_events.csv outputs/fence_status.csv
```

## Useful commands

Stop runtime containers:

```bash
./run.sh --stop
```

View logs:

```bash
./run.sh --logs
```

Rebuild assets:

```bash
./run.sh --force-rebuild --mode fence-save --video "data/Hike Vision.mp4"
```

Reset runtime state and rebuild:

```bash
./run.sh --reset --force-rebuild --mode fence-save --video "data/Hike Vision.mp4"
```

Run live mode:

```bash
./run.sh --mode fence-live --video "data/Hike Vision.mp4"
```

Run save mode:

```bash
./run.sh --mode fence-save --video "data/Hike Vision.mp4"
```

## Troubleshooting

### Docker cannot see the GPU

Run:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If it fails, fix NVIDIA driver, Docker Desktop GPU integration, or NVIDIA Container Toolkit before running the project.

### Live page opens but no video appears

Check logs:

```bash
./run.sh --logs
```

Then inspect the streaming bridge:

```bash
docker logs --tail 120 ffmpeg-ds-out
docker logs --tail 120 mediamtx
```

Restart cleanly:

```bash
./run.sh --stop
./run.sh --mode fence-live --video "data/Hike Vision.mp4"
```

### Output video is not created

Check the save log:

```bash
tail -n 120 outputs/ds_fence_nvdcf_save.log
```

Also verify that the input video exists:

```bash
ls -lh "data/Hike Vision.mp4"
```

### TensorRT engine fails on another machine

Rebuild the model assets on the target machine:

```bash
./run.sh --reset --force-rebuild --mode fence-save --video "data/Hike Vision.mp4"
```

### Video path contains spaces

Always quote the path:

```bash
./run.sh --mode fence-save --video "data/Hike Vision.mp4"
```

### Audio decoder warning appears

Some input videos contain audio streams that are irrelevant to the analytics pipeline. A warning about an unsupported audio decoder can appear while video processing continues normally.

### Boxes are visible but zone decisions look wrong

Check the polygon file:

```text
configs/fence/foot_polygon.txt
```

The fence is evaluated using the bottom-center foot-point, not the visual center of the bounding box. Adjust polygon points according to the ground-plane area you want to monitor.

## Design notes

### Why NvDCF

NvDCF provides robust DeepStream tracking for people across frames. It is more stable than simple frame-to-frame IOU matching and is suitable for analytics that depend on consistent IDs.

### Why custom ID mapping

DeepStream tracker IDs may be large internal values. The project keeps those raw IDs for traceability but displays compact sequential IDs for readability.

### Why bottom-center foot-point

For people, the bounding-box center is usually around the torso, not the person's contact point with the ground. Zone monitoring is usually about where the person stands. The bottom-center point is a better approximation for this purpose.

### Why custom overlay

The overlay is generated from controlled metadata so person boxes remain clean rectangles, fence colors are consistent, and labels match the custom state machine.

## Development notes

The main implementation files are:

```text
run.sh
scripts/container_entry.sh
scripts/fence_nvdcf_deepstream.py
scripts/make_configs.py
configs/fence/foot_polygon.txt
configs/tracker/config_tracker_NvDCF_accuracy.yml
```

The recommended workflow for changes is:

1. Edit config or source files.
2. Run save mode for deterministic validation.
3. Check the output MP4 and CSV files.
4. Run live mode for browser streaming validation.
5. Use `./run.sh --logs` for debugging.

## Minimal validation checklist

After setup, verify:

- `fence-save` creates an annotated MP4.
- `fence-save` creates `fence_events.csv`.
- `fence-save` creates `fence_status.csv`.
- `fence-live` opens in the browser.
- Boxes are clean rectangles.
- Inside/outside colors are correct.
- ENTER/EXIT events are logged.
- Display IDs are compact and readable.

## Example workflow

```bash
# 1. Stop old runtime containers.
./run.sh --stop

# 2. Run offline analytics and save video.
./run.sh --mode fence-save --video "data/Hike Vision.mp4"

# 3. Review generated outputs.
ls -lh outputs/

# 4. Run browser live mode.
./run.sh --mode fence-live --video "data/Hike Vision.mp4"

# 5. Open the stream.
# http://localhost:8889/out/
```

## Summary

DeepStream Person Fence provides a practical GPU video analytics pipeline for person detection, tracking, polygon zone monitoring, live viewing, and structured event logging. It combines DeepStream performance with custom Python metadata logic so the final behavior is easy to inspect, reproduce, and adapt.
