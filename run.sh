#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPSTREAM_IMAGE="${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:9.0-samples-multiarch}"
FFMPEG_IMAGE="${FFMPEG_IMAGE:-jrottenberg/ffmpeg:6.1-ubuntu}"
MEDIAMTX_IMAGE="${MEDIAMTX_IMAGE:-bluenviron/mediamtx}"
STAGE5_NET="${STAGE5_NET:-stage5-net}"
DS_CONTAINER="${DS_CONTAINER:-stage5-deepstream}"
CONTAINER_PROJECT="/workspace/stage5"
MODE="live"
VIDEO_INPUT="data/Hike Vision.mp4"
FORCE_REBUILD=0
STOP_ONLY=0
STATUS_ONLY=0
LOGS_ONLY=0
RESET_CONTAINERS=0

usage() {
  cat <<'EOF'
Stage5 clean runner

Usage:
  ./run.sh --mode live --video "data/Hike Vision.mp4"
  ./run.sh --mode save --video "data/Hike Vision.mp4"
  ./run.sh --mode fence-live --video "data/Hike Vision.mp4"
  ./run.sh --mode fence-save --video "data/Hike Vision.mp4"
  ./run.sh --stop
  ./run.sh --status
  ./run.sh --logs

Modes:
  live        Continuous browser live stream. Opens through MediaMTX WebRTC path /out/.
  save        Offline DeepStream run. Saves outputs/yolov8s_person_output.mp4.
  fence-live  DeepStream NvDCF tracker + v9-style foot-point fence overlay/live.
  fence-save  DeepStream NvDCF tracker + v9-style foot-point fence overlay/save.

Options:
  --video PATH        Local video path. Relative paths are relative to this project.
  --force-rebuild    Rebuild YOLO parser/ONNX/TensorRT engine.
  --reset            Remove the persistent DeepStream container, then recreate it.
  --stop             Stop runtime containers/processes.
  --status           Show Docker/DeepStream status.
  --logs             Show logs.

Final live URL:
  http://localhost:8889/out/

Notes:
  First run pulls Docker images and installs/builds missing dependencies.
  Later runs reuse the persistent container, model, ONNX, TensorRT engine and images.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --video) VIDEO_INPUT="${2:-}"; shift 2 ;;
    --force-rebuild) FORCE_REBUILD=1; shift ;;
    --reset) RESET_CONTAINERS=1; shift ;;
    --stop) STOP_ONLY=1; shift ;;
    --status) STATUS_ONLY=1; shift ;;
    --logs) LOGS_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) VIDEO_INPUT="$1"; shift ;;
  esac
done

case "$MODE" in
  live|save|fence-live|fence-save) ;;
  *) echo "ERROR: invalid --mode '$MODE'. Use live, save, fence-live, or fence-save."; exit 1 ;;
esac

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: '$1' command not found. Run this inside WSL/Linux with Docker installed."
    exit 1
  fi
}

need_cmd docker
mkdir -p "$PROJECT_DIR"/{data,models/yolo,outputs,third_party,configs/deepstream_yolov8s}

ensure_stage5_network() {
  if ! docker network inspect "$STAGE5_NET" >/dev/null 2>&1; then
    echo "[host] Creating Docker network: $STAGE5_NET"
    docker network create "$STAGE5_NET" >/dev/null
  fi
}

connect_container_to_stage5_network() {
  local name="$1"
  if container_running "$name" || container_exists "$name"; then
    docker network connect "$STAGE5_NET" "$name" 2>/dev/null || true
  fi
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "$1"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -qx "$1"
}

normalize_video() {
  local input="$1"
  if [[ "$input" == rtsp://* || "$input" == rtsps://* || "$input" == http://* || "$input" == https://* ]]; then
    echo "ERROR: this clean runner currently expects a local video file for --video."
    echo "Use a local file under data/ or pass its local path."
    exit 1
  fi

  local abs
  if [[ "$input" = /* ]]; then
    abs="$input"
  else
    if [ -f "$PROJECT_DIR/$input" ]; then
      abs="$PROJECT_DIR/$input"
    else
      abs="$(pwd)/$input"
    fi
  fi
  abs="$(realpath -m "$abs")"

  if [ ! -f "$abs" ]; then
    echo "ERROR: video not found: $abs"
    echo "Put your video in data/ or pass --video /absolute/path/video.mp4"
    exit 1
  fi

  case "$abs" in
    "$PROJECT_DIR"/*)
      VIDEO_REL="${abs#$PROJECT_DIR/}"
      ;;
    *)
      mkdir -p "$PROJECT_DIR/data"
      local base safe_base dest
      base="$(basename "$abs")"
      safe_base="external_${base// /_}"
      dest="$PROJECT_DIR/data/$safe_base"
      if [ ! -f "$dest" ] || [ "$abs" -nt "$dest" ]; then
        echo "[host] Copying external video into project data/: $dest"
        cp -f "$abs" "$dest"
      fi
      VIDEO_REL="data/$safe_base"
      ;;
  esac
}

stop_runtime() {
  echo "[host] Stopping Stage5 runtime..."
  if [ -f "$PROJECT_DIR/outputs/fence_live_pipe.pid" ]; then
    kill "$(cat "$PROJECT_DIR/outputs/fence_live_pipe.pid")" 2>/dev/null || true
    rm -f "$PROJECT_DIR/outputs/fence_live_pipe.pid"
  fi
  docker rm -f ffmpeg-cam ffmpeg-ds-out mediamtx 2>/dev/null || true
  if container_exists "$DS_CONTAINER"; then
    docker exec "$DS_CONTAINER" bash -lc "$CONTAINER_PROJECT/scripts/container_entry.sh stop" 2>/dev/null || true
  fi
  # Old experimental container name from the manual debugging phase. Remove it to free port 8555.
  docker rm -f ds-stage5 2>/dev/null || true
}

show_status() {
  echo "===== DOCKER PS ====="
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  echo
  if container_exists "$DS_CONTAINER"; then
    echo "===== DEEPSTREAM STATUS ====="
    docker exec "$DS_CONTAINER" bash -lc "$CONTAINER_PROJECT/scripts/container_entry.sh status" 2>/dev/null || true
  fi
  echo
  echo "===== MEDIAMTX LOG ====="
  docker logs --tail 80 mediamtx 2>/dev/null || true
  echo
  echo "===== FFMPEG CAM LOG ====="
  docker logs --tail 40 ffmpeg-cam 2>/dev/null || true
  echo
  echo "===== FFMPEG OUT LOG ====="
  docker logs --tail 80 ffmpeg-ds-out 2>/dev/null || true
}

show_logs() {
  if container_exists "$DS_CONTAINER"; then
    echo "===== DEEPSTREAM LIVE LOG ====="
    docker exec "$DS_CONTAINER" bash -lc 'tail -n 160 /workspace/stage5/outputs/ds_live.log 2>/dev/null || true'
    echo "===== DEEPSTREAM SAVE LOG ====="
    docker exec "$DS_CONTAINER" bash -lc 'tail -n 120 /workspace/stage5/outputs/ds_save.log 2>/dev/null || true'
    echo "===== DEEPSTREAM FENCE LIVE LOG ====="
    docker exec "$DS_CONTAINER" bash -lc 'tail -n 160 /workspace/stage5/outputs/ds_fence_live.log 2>/dev/null || true'
    echo "===== DEEPSTREAM FENCE SAVE LOG ====="
    docker exec "$DS_CONTAINER" bash -lc 'tail -n 120 /workspace/stage5/outputs/ds_fence_save.log 2>/dev/null || true'
    echo "===== NVDCF V9 FENCE SAVE LOG ====="
    docker exec "$DS_CONTAINER" bash -lc 'tail -n 160 /workspace/stage5/outputs/ds_fence_nvdcf_save.log 2>/dev/null || true'
    echo "===== NVDCF V9 FENCE LIVE LOG ====="
    docker exec "$DS_CONTAINER" bash -lc 'tail -n 160 /workspace/stage5/outputs/ds_fence_nvdcf_live.log 2>/dev/null || true'
  fi
  echo "===== MEDIAMTX LOG ====="
  docker logs --tail 160 mediamtx 2>/dev/null || true
  echo "===== FFMPEG CAM LOG ====="
  docker logs --tail 100 ffmpeg-cam 2>/dev/null || true
  echo "===== FFMPEG OUT LOG ====="
  docker logs --tail 160 ffmpeg-ds-out 2>/dev/null || true
}

ensure_deepstream_container() {
  ensure_stage5_network
  if [ "$RESET_CONTAINERS" = "1" ] && container_exists "$DS_CONTAINER"; then
    echo "[host] Removing persistent DeepStream container because --reset was used."
    docker rm -f "$DS_CONTAINER" >/dev/null 2>&1 || true
  fi

  # Remove the old manual container if it occupies the same RTSP output port.
  if container_exists ds-stage5; then
    echo "[host] Removing old manual container ds-stage5 to free port 8555."
    docker rm -f ds-stage5 >/dev/null 2>&1 || true
  fi

  if container_exists "$DS_CONTAINER"; then
    if ! container_running "$DS_CONTAINER"; then
      echo "[host] Starting existing DeepStream container: $DS_CONTAINER"
      docker start "$DS_CONTAINER" >/dev/null
    else
      echo "[host] DeepStream container already running: $DS_CONTAINER"
    fi
  else
    echo "[host] Creating persistent DeepStream container: $DS_CONTAINER"
    echo "[host] Image: $DEEPSTREAM_IMAGE"
    docker run -dit \
      --name "$DS_CONTAINER" \
      --gpus all \
      --add-host=host.docker.internal:host-gateway \
      -p 8555:8554/tcp \
      -p 5400:5400/udp \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
      -v "$PROJECT_DIR":"$CONTAINER_PROJECT" \
      "$DEEPSTREAM_IMAGE" \
      bash >/dev/null
  fi
  connect_container_to_stage5_network "$DS_CONTAINER"
}

write_mediamtx_config() {
  cat > /tmp/stage5_mediamtx.yml <<'EOF'
logLevel: info

rtsp: yes
rtspAddress: :8554

webrtc: yes
webrtcAddress: :8889
webrtcLocalUDPAddress: :8189

paths:
  cam:
    source: publisher
  out:
    source: publisher
EOF
}

start_mediamtx() {
  ensure_stage5_network
  docker rm -f mediamtx 2>/dev/null || true
  write_mediamtx_config
  echo "[host] Starting MediaMTX on Docker network $STAGE5_NET..."
  docker run -d --name mediamtx \
    --network "$STAGE5_NET" \
    -p 8554:8554 \
    -p 8888:8888 \
    -p 8889:8889 \
    -p 8189:8189/udp \
    -v /tmp/stage5_mediamtx.yml:/mediamtx.yml \
    "$MEDIAMTX_IMAGE" >/dev/null
  sleep 2
}

start_ffmpeg_cam() {
  local loop_mode="${1:-loop}"
  docker rm -f ffmpeg-cam 2>/dev/null || true

  local loop_args=()
  if [ "$loop_mode" = "loop" ]; then
    echo "[host] Starting ffmpeg input loop -> rtsp://localhost:8554/cam"
    loop_args=(-stream_loop -1)
  else
    echo "[host] Starting ffmpeg one-pass input -> rtsp://localhost:8554/cam"
    loop_args=()
  fi

  docker run -d --name ffmpeg-cam \
    --network "$STAGE5_NET" \
    --add-host=host.docker.internal:host-gateway \
    -v "$PROJECT_DIR":"$CONTAINER_PROJECT":ro \
    "$FFMPEG_IMAGE" \
    -re "${loop_args[@]}" \
    -i "$CONTAINER_PROJECT/$VIDEO_REL" \
    -an \
    -vf "fps=25,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" \
    -c:v libx264 \
    -preset ultrafast \
    -tune zerolatency \
    -profile:v baseline \
    -pix_fmt yuv420p \
    -r 25 \
    -g 25 \
    -payload_size 1200 \
    -f rtsp \
    -rtsp_transport tcp \
    rtsp://mediamtx:8554/cam >/dev/null
  sleep 1
}
wait_for_deepstream_ready() {
  echo "[host] Waiting for DeepStream to build/load TensorRT engine and start RTSP..."
  for i in {1..36}; do
    if docker exec "$DS_CONTAINER" bash -lc "grep -q 'Pipeline running' '$CONTAINER_PROJECT/outputs/ds_live.log' 2>/dev/null || grep -q 'Pipeline running' '$CONTAINER_PROJECT/outputs/ds_fence_live.log' 2>/dev/null"; then
      echo "[host] DeepStream pipeline is running."
      return 0
    fi
    if (( i % 6 == 0 )); then
      echo "[host] still waiting for DeepStream... ($((i*5))s)"
      docker exec "$DS_CONTAINER" bash -lc "(grep -E 'Deserialize engine|Building|Creating|ERROR|WARNING|Pipeline running|Launched RTSP' '$CONTAINER_PROJECT/outputs/ds_live.log' 2>/dev/null; grep -E 'Deserialize engine|Building|Creating|ERROR|WARNING|Pipeline running|Launched RTSP' '$CONTAINER_PROJECT/outputs/ds_fence_live.log' 2>/dev/null) | tail -n 20" || true
    fi
    sleep 5
  done
  echo "WARNING: DeepStream did not report 'Pipeline running' yet. Continuing anyway; check ./run.sh --logs"
}


wait_for_nvdcf_fence_frames() {
  echo "[host] Waiting for DeepStream NvDCF fence to process frames..."
  for i in {1..45}; do
    if docker exec "$DS_CONTAINER" bash -lc "grep -q '\[fence-nvdcf\] frame=' '$CONTAINER_PROJECT/outputs/ds_fence_nvdcf_live.log' 2>/dev/null"; then
      echo "[host] DeepStream NvDCF fence is producing frames."
      return 0
    fi
    if docker exec "$DS_CONTAINER" bash -lc "grep -qE 'ERROR from|Traceback|Unable to set pipeline|Failed to link' '$CONTAINER_PROJECT/outputs/ds_fence_nvdcf_live.log' 2>/dev/null"; then
      echo "ERROR: DeepStream NvDCF fence live pipeline failed before producing frames."
      docker exec "$DS_CONTAINER" bash -lc "tail -n 180 '$CONTAINER_PROJECT/outputs/ds_fence_nvdcf_live.log'" || true
      return 1
    fi
    if (( i % 5 == 0 )); then
      echo "[host] still waiting for first processed frames... (${i}s)"
      docker exec "$DS_CONTAINER" bash -lc "tail -n 30 '$CONTAINER_PROJECT/outputs/ds_fence_nvdcf_live.log' 2>/dev/null" || true
    fi
    sleep 1
  done
  echo "WARNING: DeepStream NvDCF fence did not log processed frames yet. Continuing to republisher."
  docker exec "$DS_CONTAINER" bash -lc "tail -n 120 '$CONTAINER_PROJECT/outputs/ds_fence_nvdcf_live.log' 2>/dev/null" || true
  return 0
}

start_ffmpeg_out() {
  ensure_stage5_network
  connect_container_to_stage5_network "$DS_CONTAINER"
  docker rm -f ffmpeg-ds-out 2>/dev/null || true

  # v22: ffmpeg listens for MPEG-TS over UDP from DeepStream and publishes
  # directly to MediaMTX /out. This removes the fragile GstRtspServer /ds-test
  # hop that returned RTSP 503 in v17-v21.
  local in_url="udp://0.0.0.0:5400?fifo_size=1000000&overrun_nonfatal=1"
  local out_url="rtsp://mediamtx:8554/out"
  echo "[host] Starting ffmpeg UDP MPEG-TS receiver -> MediaMTX /out"
  echo "[host]   input : $in_url"
  echo "[host]   output: $out_url"

  docker run -d --name ffmpeg-ds-out \
    --network "$STAGE5_NET" \
    "$FFMPEG_IMAGE" \
    -fflags nobuffer \
    -flags low_delay \
    -probesize 32768 \
    -analyzeduration 0 \
    -i "$in_url" \
    -an \
    -vf "fps=25" \
    -c:v libx264 \
    -preset ultrafast \
    -tune zerolatency \
    -profile:v baseline \
    -pix_fmt yuv420p \
    -r 25 \
    -g 25 \
    -f rtsp \
    -rtsp_transport tcp \
    "$out_url" >/dev/null
}

wait_for_ffmpeg_out() {
  for attempt in {1..40}; do
    if docker logs mediamtx 2>&1 | grep -q "\[path out\] stream is available and online"; then
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx ffmpeg-ds-out; then
      echo "ERROR: ffmpeg-ds-out exited."
      docker logs --tail 160 ffmpeg-ds-out 2>/dev/null || true
      return 1
    fi
    echo "[host] waiting for MediaMTX /out... retry $attempt/40"
    docker logs --tail 12 ffmpeg-ds-out 2>/dev/null || true
    docker logs --tail 20 mediamtx 2>/dev/null | grep -E "path out|RTSP|publisher|closed|error|ERR|WAR" || true
    sleep 2
  done
  echo "WARNING: ffmpeg-ds-out did not confirm MediaMTX /out. Check ./run.sh --logs"
  return 1
}

wait_for_out() {
  for i in {1..40}; do
    if docker logs mediamtx 2>&1 | grep -q "\[path out\] stream is available and online"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

run_live() {
  normalize_video "$VIDEO_INPUT"
  stop_runtime
  ensure_deepstream_container

  echo "[host] Preparing DeepStream model/parser/configs..."
  local q_video
  q_video="$(printf '%q' "$VIDEO_REL")"
  local prepare_cmd="$CONTAINER_PROJECT/scripts/container_entry.sh prepare --video-rel $q_video"
  if [ "$FORCE_REBUILD" = "1" ]; then prepare_cmd+=" --force-rebuild"; fi
  docker exec "$DS_CONTAINER" bash -lc "$prepare_cmd"

  start_mediamtx
  start_ffmpeg_cam loop

  echo "[host] Starting DeepStream live pipeline..."
  local start_cmd="$CONTAINER_PROJECT/scripts/container_entry.sh start-live --video-rel $q_video"
  docker exec "$DS_CONTAINER" bash -lc "$start_cmd"

  wait_for_deepstream_ready
  start_ffmpeg_out

  echo "[host] Waiting for WebRTC output path..."
  if wait_for_out; then
    echo "============================================================"
    echo "LIVE STREAM READY"
    echo "Open this URL in Chrome/Edge:"
    echo "  http://localhost:8889/out/"
    echo
    echo "Do not use /deepstream/. The final clean path is /out/."
    echo "Stop everything with: ./run.sh --stop"
    echo "Logs: ./run.sh --logs"
    echo "============================================================"
  else
    echo "WARNING: /out/ did not become ready within 20 seconds."
    echo "Run: ./run.sh --logs"
    exit 1
  fi
}

run_save() {
  normalize_video "$VIDEO_INPUT"
  stop_runtime
  ensure_deepstream_container

  local q_video
  q_video="$(printf '%q' "$VIDEO_REL")"
  local prepare_cmd="$CONTAINER_PROJECT/scripts/container_entry.sh prepare --video-rel $q_video"
  if [ "$FORCE_REBUILD" = "1" ]; then prepare_cmd+=" --force-rebuild"; fi
  echo "[host] Preparing DeepStream model/parser/configs..."
  docker exec "$DS_CONTAINER" bash -lc "$prepare_cmd"

  # Save mode must read the local file directly, not RTSP.
  # RTSP is live and can start from the middle if DeepStream connects late.
  # The generated save config uses type=2 + file://... and keeps the same streammux geometry.
  local save_start_cmd="$CONTAINER_PROJECT/scripts/container_entry.sh start-save --video-rel $q_video"
  docker exec "$DS_CONTAINER" bash -lc "$save_start_cmd"

  docker exec "$DS_CONTAINER" bash -lc "$CONTAINER_PROJECT/scripts/container_entry.sh wait-save"

  echo "Saved file on host: $PROJECT_DIR/outputs/yolov8s_person_output.mp4"
}

run_fence_custom_prepare() {
  normalize_video "$VIDEO_INPUT"
  ensure_deepstream_container

  local q_video
  q_video="$(printf '%q' "$VIDEO_REL")"
  local prepare_cmd="$CONTAINER_PROJECT/scripts/container_entry.sh prepare --video-rel $q_video"
  if [ "$FORCE_REBUILD" = "1" ]; then prepare_cmd+=" --force-rebuild"; fi
  echo "[host] Preparing model/parser/venv for custom foot-point fence..."
  docker exec "$DS_CONTAINER" bash -lc "$prepare_cmd"
}

run_fence_save() {
  # v11 hybrid-correct mode:
  #   DeepStream source -> nvinfer YOLOv8s -> nvtracker NvDCF
  #   -> custom v9-style bottom-center foot-point fence overlay -> MP4
  # This uses NvDCF for IDs, but does NOT use nvdsanalytics for fence logic.
  normalize_video "$VIDEO_INPUT"
  stop_runtime
  ensure_deepstream_container

  local q_video
  q_video="$(printf '%q' "$VIDEO_REL")"
  local prepare_cmd="$CONTAINER_PROJECT/scripts/container_entry.sh prepare --video-rel $q_video"
  if [ "$FORCE_REBUILD" = "1" ]; then prepare_cmd+=" --force-rebuild"; fi
  echo "[host] Preparing DeepStream model/parser/configs for NvDCF + v9 foot-point fence save..."
  docker exec "$DS_CONTAINER" bash -lc "$prepare_cmd"

  mkdir -p "$PROJECT_DIR/outputs"
  rm -f "$PROJECT_DIR/outputs/yolov8s_person_fence_output.mp4" \
        "$PROJECT_DIR/outputs/fence_events.csv" \
        "$PROJECT_DIR/outputs/fence_status.csv" \
        "$PROJECT_DIR/outputs/ds_fence_nvdcf_save.log"

  echo "[host] Running DeepStream NvDCF + v9 foot-point fence save..."
  echo "[host] Tracker: configs/tracker/config_tracker_NvDCF_accuracy.yml"
  echo "[host] Logic:   custom bottom-center foot-point state machine, same style as v9"
  echo "[host] Note:    nvdsanalytics is intentionally disabled in this mode"

  local save_cmd="$CONTAINER_PROJECT/scripts/container_entry.sh run-fence-nvdcf-save --video-rel $q_video"
  if ! docker exec "$DS_CONTAINER" bash -lc "$save_cmd"; then
    echo "ERROR: NvDCF v9 fence-save failed."
    echo "----- ds_fence_nvdcf_save.log tail -----"
    tail -n 180 "$PROJECT_DIR/outputs/ds_fence_nvdcf_save.log" 2>/dev/null || true
    echo "----------------------------------------"
    return 1
  fi

  local out_mp4="$PROJECT_DIR/outputs/yolov8s_person_fence_output.mp4"
  local size=0
  if [ -f "$out_mp4" ]; then
    size="$(stat -c%s "$out_mp4" 2>/dev/null || echo 0)"
  fi
  if [ "$size" -lt 10000 ]; then
    echo "ERROR: NvDCF v9 fence-save did not produce a valid MP4."
    echo "MP4 bytes: $size"
    echo "----- ds_fence_nvdcf_save.log tail -----"
    tail -n 180 "$PROJECT_DIR/outputs/ds_fence_nvdcf_save.log" 2>/dev/null || true
    echo "----------------------------------------"
    return 1
  fi

  echo "============================================================"
  echo "DeepStream NvDCF + v9 foot-point fence save finished."
  echo "Video:  $out_mp4"
  echo "Events: $PROJECT_DIR/outputs/fence_events.csv"
  echo "Status: $PROJECT_DIR/outputs/fence_status.csv"
  echo "Log:    $PROJECT_DIR/outputs/ds_fence_nvdcf_save.log"
  echo "============================================================"
}

run_fence_live() {
  # v21 live mode:
  #   DeepStream Python reads local file directly, same proven input path as fence-save
  #   -> nvinfer YOLOv8s -> nvtracker NvDCF -> custom v9 foot-point overlay
  #   -> UDP MPEG-TS -> ffmpeg-ds-out -> MediaMTX /out -> WebRTC browser
  # This removes both unstable hops: MediaMTX /cam input and GstRtspServer /ds-test output.
  normalize_video "$VIDEO_INPUT"
  stop_runtime
  ensure_deepstream_container

  local q_video
  q_video="$(printf '%q' "$VIDEO_REL")"
  local prepare_cmd="$CONTAINER_PROJECT/scripts/container_entry.sh prepare --video-rel $q_video"
  if [ "$FORCE_REBUILD" = "1" ]; then prepare_cmd+=" --force-rebuild"; fi
  echo "[host] Preparing DeepStream model/parser/configs for NvDCF + v9 foot-point fence live..."
  docker exec "$DS_CONTAINER" bash -lc "$prepare_cmd"

  start_mediamtx
  echo "[host] Starting ffmpeg receiver before DeepStream, so UDP MPEG-TS is not missed..."
  start_ffmpeg_out

  echo "[host] Starting DeepStream NvDCF + v9 foot-point fence live pipeline..."
  echo "[host] Input: direct file URI inside DeepStream container, not MediaMTX /cam"
  echo "[host] Tracker: configs/tracker/config_tracker_NvDCF_accuracy.yml"
  echo "[host] Logic:   custom bottom-center foot-point state machine, same style as v9"
  local start_cmd="$CONTAINER_PROJECT/scripts/container_entry.sh start-fence-nvdcf-live --video-rel $q_video"
  docker exec "$DS_CONTAINER" bash -lc "$start_cmd"

  wait_for_nvdcf_fence_frames

  echo "[host] Waiting for ffmpeg -> MediaMTX /out..."
  if wait_for_ffmpeg_out && wait_for_out; then
    echo "============================================================"
    echo "DEEPSTREAM NvDCF + v9 FOOT-POINT FENCE LIVE READY"
    echo "Open this URL in Chrome/Edge:"
    echo "  http://localhost:8889/out/"
    echo
    echo "Tracker: NvDCF via nvtracker/libnvds_nvmultiobjecttracker"
    echo "Fence:   exact bottom-center foot-point state machine like v9"
    echo "Stop:    ./run.sh --stop"
    echo "Logs:    ./run.sh --logs"
    echo "============================================================"
  else
    echo "WARNING: /out/ did not become ready within 20 seconds."
    echo "Run: ./run.sh --logs"
    exit 1
  fi
}

if [ "$STOP_ONLY" = "1" ]; then stop_runtime; exit 0; fi
if [ "$STATUS_ONLY" = "1" ]; then show_status; exit 0; fi
if [ "$LOGS_ONLY" = "1" ]; then show_logs; exit 0; fi

case "$MODE" in
  live) run_live ;;
  save) run_save ;;
  fence-live) run_fence_live ;;
  fence-save) run_fence_save ;;
esac
