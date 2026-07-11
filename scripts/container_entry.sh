#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/stage5"
YOLO_REPO="$ROOT/third_party/DeepStream-Yolo"
YOLO_LIB="$YOLO_REPO/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"
VENV="$ROOT/.venv-yolo"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-/usr/bin/python3}"
VIDEO_REL="data/Hike Vision.mp4"
FORCE_REBUILD=0
WIDTH=1920
HEIGHT=1080

usage() {
  cat <<'EOF'
Container-only helper. Normally call ../run.sh on the host.

Commands:
  prepare --video-rel REL [--force-rebuild]
  start-live --video-rel REL
  start-save --video-rel REL
  wait-save
  start-fence-live --video-rel REL          # legacy deepstream-app + nvdsanalytics
  start-fence-save --video-rel REL          # legacy deepstream-app + nvdsanalytics
  wait-fence-save
  run-fence-nvdcf-save --video-rel REL      # NvDCF + v9 foot-point logic, blocking
  start-fence-nvdcf-live --video-rel REL    # NvDCF + v9 foot-point logic, background RTSP /out
  run-save --video-rel REL [--force-rebuild]   # legacy blocking mode
  stop
  status
EOF
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then usage; exit 1; fi
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-rel) VIDEO_REL="${2:-}"; shift 2 ;;
    --force-rebuild) FORCE_REBUILD=1; shift ;;
    --width) WIDTH="${2:-1920}"; shift 2 ;;
    --height) HEIGHT="${2:-1080}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

cd "$ROOT"
mkdir -p outputs models/yolo third_party configs/deepstream_yolov8s data

file_uri() {
  python3 - "$1" <<'PYURI'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve().as_uri())
PYURI
}

check_deepstream_python_runtime() {
  # IMPORTANT: use the system Python, not .venv-yolo.
  # python3-gi is installed by apt under /usr/lib/python3/dist-packages,
  # while the export venv does not see system dist-packages by default.
  "$SYSTEM_PYTHON" - <<'PY' >/tmp/stage5_pyds_check.log 2>&1
import sys, os, sysconfig
paths = [
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3/dist-packages",
    "/opt/nvidia/deepstream/deepstream/lib",
    "/opt/nvidia/deepstream/deepstream/lib/python3/dist-packages",
    "/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/build",
]
for key in ("purelib", "platlib"):
    p = sysconfig.get_paths().get(key)
    if p:
        paths.append(p)
for p in paths:
    if p and os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
print("python=", sys.executable)
print("sys.path_head=", sys.path[:8])
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pyds
print("OK gi+pyds")
PY
}


ensure_deepstream_python_runtime() {
  # v12 fence mode needs system Python bindings for GStreamer (gi) and DeepStream (pyds).
  # DeepStream 9.0 containers do not always ship these preinstalled, and pyds is no longer
  # distributed as a ready wheel for DS9, so this function installs gi and builds pyds once.
  if check_deepstream_python_runtime; then
    echo "[prepare] DeepStream Python runtime found: gi + pyds."
    return 0
  fi

  echo "[prepare] DeepStream Python runtime missing; installing/building gi + pyds..."
  echo "[prepare] Previous check log:"
  cat /tmp/stage5_pyds_check.log 2>/dev/null || true

  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y \
    python3-gi python3-dev python3-gst-1.0 python-gi-dev gir1.2-gst-rtsp-server-1.0 \
    git meson ninja-build python3 python3-pip python3-venv \
    cmake g++ build-essential libglib2.0-dev libglib2.0-dev-bin \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libtool m4 autoconf automake libgirepository-2.0-dev libcairo2-dev

  export PYTHONPATH="/usr/local/lib/python3/dist-packages:/usr/lib/python3/dist-packages:${PYTHONPATH:-}"

  if check_deepstream_python_runtime; then
    echo "[prepare] DeepStream Python runtime fixed by apt packages."
    return 0
  fi

  local pyds_repo="$ROOT/third_party/deepstream_python_apps"
  local wheel_cache="$ROOT/third_party/pyds_wheels"
  mkdir -p "$wheel_cache"

  # v14: do NOT upgrade Debian/Ubuntu system pip. Ubuntu's apt-installed pip
  # has no pip RECORD file, so `pip install --upgrade pip` fails with:
  #   Cannot uninstall pip 24.0, RECORD file not found.
  # If a pyds wheel was already built by a previous run, install it first.
  if ls "$wheel_cache"/pyds-*.whl >/dev/null 2>&1; then
    echo "[prepare] Installing cached pyds wheel with system Python..."
    "$SYSTEM_PYTHON" -m pip install --break-system-packages --no-deps --force-reinstall "$wheel_cache"/pyds-*.whl
    if check_deepstream_python_runtime; then
      echo "[prepare] DeepStream Python runtime fixed using cached pyds wheel."
      return 0
    fi
  fi

  # Only install the Python build frontend if it is actually missing. Do not
  # upgrade pip itself; use --break-system-packages only for the extra package.
  if ! "$SYSTEM_PYTHON" - <<'PYBUILD' >/dev/null 2>&1
import build
PYBUILD
  then
    echo "[prepare] Installing Python build frontend without upgrading system pip..."
    if ! "$SYSTEM_PYTHON" -m pip install --break-system-packages --no-cache-dir build; then
      echo "[prepare] pip install build failed; falling back to apt python3-build."
      apt-get install -y python3-build python3-wheel python3-setuptools || true
    fi
  fi

  if [ ! -d "$pyds_repo/.git" ]; then
    echo "[prepare] Cloning NVIDIA deepstream_python_apps for pyds build..."
    rm -rf "$pyds_repo"
    git clone --depth 1 https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git "$pyds_repo"
  else
    echo "[prepare] deepstream_python_apps already exists."
  fi

  cd "$pyds_repo"
  git submodule update --init
  "$SYSTEM_PYTHON" bindings/3rdparty/git-partial-submodule/git-partial-submodule.py restore-sparse

  if [ -d bindings/3rdparty/gstreamer/subprojects/gst-python ]; then
    echo "[prepare] Building/installing gst-python helper..."
    cd "$pyds_repo/bindings/3rdparty/gstreamer/subprojects/gst-python"
    if [ ! -d build ]; then
      meson setup build
    fi
    ninja -C build
    ninja -C build install
  fi

  echo "[prepare] Building pyds wheel for this DeepStream container..."
  cd "$pyds_repo/bindings"
  export CMAKE_ARGS="-DDS_PATH=/opt/nvidia/deepstream/deepstream"
  export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
  "$SYSTEM_PYTHON" -m build
  cp -f dist/pyds-*.whl "$wheel_cache"/
  "$SYSTEM_PYTHON" -m pip install --break-system-packages --force-reinstall dist/pyds-*.whl

  if check_deepstream_python_runtime; then
    echo "[prepare] DeepStream Python runtime ready: gi + pyds."
    return 0
  fi

  echo "ERROR: failed to prepare DeepStream Python runtime even after build."
  cat /tmp/stage5_pyds_check.log 2>/dev/null || true
  return 1
}

ensure_cuda_links() {
  if [ -f /usr/local/cuda-13.1/lib64/libcublas.so.13 ]; then
    ln -sf /usr/local/cuda-13.1/lib64/libcublas.so.13 /usr/local/cuda-13.1/lib64/libcublas.so || true
  fi
  if [ -f /usr/local/cuda/lib64/libcublas.so.13 ]; then
    ln -sf /usr/local/cuda/lib64/libcublas.so.13 /usr/local/cuda/lib64/libcublas.so || true
  fi
}

install_deps_if_needed() {
  # If this project already contains the built parser, exported ONNX and Python venv,
  # skip the heavy YOLO/CUDA build dependencies. DeepStream Python runtime is handled
  # separately by ensure_deepstream_python_runtime(), because pyds/gi live in the
  # container's system Python, not in the export venv.
  if [ -f "$YOLO_LIB" ] && [ -f "$ROOT/models/yolo/yolov8s.onnx" ] && [ -x "$VENV/bin/python" ]; then
    echo "[prepare] Prebuilt parser/model/venv found; skipping YOLO/CUDA apt install."
    ensure_cuda_links
    return 0
  fi
  local need=0
  command -v git >/dev/null 2>&1 || need=1
  command -v g++ >/dev/null 2>&1 || need=1
  command -v make >/dev/null 2>&1 || need=1
  command -v python3 >/dev/null 2>&1 || need=1
  command -v pip3 >/dev/null 2>&1 || need=1
  [ -x /usr/local/cuda-13.1/bin/nvcc ] || need=1
  [ -f /usr/local/cuda-13.1/include/cuda_runtime_api.h ] || need=1

  if [ "$need" = "1" ]; then
    echo "[prepare] Installing DeepStream build/export dependencies..."
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git build-essential python3-pip python3-venv \
      cuda-cudart-dev-13-1 cuda-nvcc-13-1
  else
    echo "[prepare] APT/CUDA dependencies already available."
  fi
  ensure_cuda_links
}

clone_deepstream_yolo() {
  if [ ! -d "$YOLO_REPO/.git" ]; then
    echo "[prepare] Cloning DeepStream-Yolo..."
    rm -rf "$YOLO_REPO"
    git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo.git "$YOLO_REPO"
  else
    echo "[prepare] DeepStream-Yolo already exists."
  fi
}

build_yolo_parser() {
  if [ ! -f "$YOLO_LIB" ] || [ "$FORCE_REBUILD" = "1" ]; then
    echo "[prepare] Building DeepStream-Yolo custom parser..."
    CUDA_VER=13.1 make -C "$YOLO_REPO/nvdsinfer_custom_impl_Yolo" clean
    CUDA_VER=13.1 make -C "$YOLO_REPO/nvdsinfer_custom_impl_Yolo"
  else
    echo "[prepare] YOLO custom parser already built."
  fi
}

setup_python() {
  if [ ! -d "$VENV" ]; then
    echo "[prepare] Creating Python venv..."
    python3 -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip >/dev/null
  if ! python - <<'PY' >/tmp/stage5_import_check.log 2>&1
import ultralytics, onnx, onnxscript, onnxruntime
PY
  then
    echo "[prepare] Installing Python export packages..."
    pip install ultralytics onnx onnxslim onnxscript onnxruntime
  else
    echo "[prepare] Python export packages already available."
  fi
}

download_model() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  cd "$ROOT/models/yolo"
  if [ ! -f yolov8s.pt ]; then
    echo "[prepare] Downloading yolov8s.pt..."
    python - <<'PY'
from ultralytics import YOLO
YOLO("yolov8s.pt")
PY
  else
    echo "[prepare] yolov8s.pt already exists."
  fi
}

export_onnx() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  local marker="$ROOT/models/yolo/.exported_with_deepstream_yolo_v8.marker"
  if [ "$FORCE_REBUILD" = "1" ] || [ ! -f "$ROOT/models/yolo/yolov8s.onnx" ] || [ ! -f "$marker" ]; then
    echo "[prepare] Exporting YOLOv8s ONNX with DeepStream-Yolo exporter..."
    rm -f "$ROOT/models/yolo/yolov8s.onnx" \
          "$ROOT/models/yolo/yolov8s.onnx.data" \
          "$ROOT/models/yolo"/*.engine \
          "$ROOT/models/yolo/model_b1_gpu0_fp16.engine" \
          "$YOLO_REPO/model_b1_gpu0_fp16.engine" \
          "$ROOT/model_b1_gpu0_fp16.engine"
    cd "$YOLO_REPO"
    python utils/export_yoloV8.py \
      -w "$ROOT/models/yolo/yolov8s.pt" \
      -s 640 640 \
      --opset 17 \
      --simplify \
      --batch 1
    date > "$marker"
  else
    echo "[prepare] DeepStream-compatible ONNX already exists."
  fi
}

write_project_configs() {
  echo "ID" > "$YOLO_REPO/labels_id.txt"
  python3 "$ROOT/scripts/make_stage5_configs.py" \
    --root "$ROOT" \
    --video-rel "$VIDEO_REL" \
    --width "$WIDTH" \
    --height "$HEIGHT"
}

prepare() {
  echo "============================================================"
  echo "Stage5 prepare inside DeepStream container"
  echo "Root:      $ROOT"
  echo "Video rel: $VIDEO_REL"
  echo "Force:     $FORCE_REBUILD"
  echo "============================================================"
  install_deps_if_needed
  clone_deepstream_yolo
  build_yolo_parser
  setup_python
  download_model
  export_onnx
  write_project_configs
  ensure_deepstream_python_runtime
  echo "[prepare] Done."
}

start_live() {
  write_project_configs
  pkill -x deepstream-app 2>/dev/null || true
  rm -f "$ROOT/outputs/ds_live.log"
  cd "$ROOT/models/yolo"
  nohup deepstream-app -c "$ROOT/configs/deepstream_yolov8s/deepstream_app_live_rtsp_cam.txt" \
    > "$ROOT/outputs/ds_live.log" 2>&1 &
  sleep 8
  echo "===== DEEPSTREAM PROCESS ====="
  pgrep -a deepstream-app || true
  echo "===== DEEPSTREAM LOG ====="
  grep -E "Launched RTSP|Pipeline running|PERF|ERROR|WARNING|EOS" "$ROOT/outputs/ds_live.log" | tail -n 60 || true
}

start_fence_live() {
  write_project_configs
  pkill -x deepstream-app 2>/dev/null || true
  rm -f "$ROOT/outputs/ds_fence_live.log"
  cd "$ROOT/models/yolo"
  nohup deepstream-app -c "$ROOT/configs/deepstream_yolov8s/deepstream_app_fence_live_rtsp_cam.txt"     > "$ROOT/outputs/ds_fence_live.log" 2>&1 &
  sleep 8
  echo "===== DEEPSTREAM FENCE PROCESS ====="
  pgrep -a deepstream-app || true
  echo "===== DEEPSTREAM FENCE LIVE LOG ====="
  grep -E "Launched RTSP|Pipeline running|PERF|ERROR|WARNING|EOS|nvdsanalytics|analytics" "$ROOT/outputs/ds_fence_live.log" | tail -n 80 || true
}

start_fence_save() {
  # Fence save uses direct local file input + nvdsanalytics ROI.
  write_project_configs
  pkill -x deepstream-app 2>/dev/null || true
  rm -f "$ROOT/outputs/yolov8s_person_fence_output.mp4" "$ROOT/outputs/ds_fence_save.log" "$ROOT/outputs/ds_fence_save.pid"
  cd "$ROOT/models/yolo"
  echo "[fence-save] Starting DeepStream fence save from direct local file input..."
  nohup deepstream-app -c "$ROOT/configs/deepstream_yolov8s/deepstream_app_fence_save_file.txt"     > "$ROOT/outputs/ds_fence_save.log" 2>&1 &
  echo $! > "$ROOT/outputs/ds_fence_save.pid"
  sleep 0.5
  echo "===== FENCE SAVE PROCESS ====="
  cat "$ROOT/outputs/ds_fence_save.pid" 2>/dev/null || true
  pgrep -a deepstream-app || true
  echo "===== FENCE SAVE LOG HEAD ====="
  tail -n 80 "$ROOT/outputs/ds_fence_save.log" 2>/dev/null || true
}

start_save() {
  # Save uses direct local file input generated by make_stage5_configs.py.
  # No RTSP publisher is used here, so the saved output starts from frame 0.
  write_project_configs
  pkill -x deepstream-app 2>/dev/null || true
  rm -f "$ROOT/outputs/yolov8s_person_output.mp4" "$ROOT/outputs/ds_save.log" "$ROOT/outputs/ds_save.pid"
  cd "$ROOT/models/yolo"
  echo "[save] Starting DeepStream save from direct local file input..."
  nohup deepstream-app -c "$ROOT/configs/deepstream_yolov8s/deepstream_app_save_file.txt" \
    > "$ROOT/outputs/ds_save.log" 2>&1 &
  echo $! > "$ROOT/outputs/ds_save.pid"
  sleep 0.5
  echo "===== SAVE PROCESS ====="
  cat "$ROOT/outputs/ds_save.pid" 2>/dev/null || true
  pgrep -a deepstream-app || true
  echo "===== SAVE LOG HEAD ====="
  tail -n 60 "$ROOT/outputs/ds_save.log" 2>/dev/null || true
}

wait_save() {
  local pid_file="$ROOT/outputs/ds_save.pid"
  local pid=""
  [ -f "$pid_file" ] && pid="$(cat "$pid_file" 2>/dev/null || true)"
  echo "[save] Waiting for DeepStream save to finish..."
  if [ -n "$pid" ]; then
    while kill -0 "$pid" 2>/dev/null; do
      sleep 3
    done
  else
    while pgrep -x deepstream-app >/dev/null 2>&1; do
      sleep 3
    done
  fi
  echo "============================================================"
  echo "Saved output: $ROOT/outputs/yolov8s_person_output.mp4"
  ls -lah "$ROOT/outputs/yolov8s_person_output.mp4" || true
  echo "Log: $ROOT/outputs/ds_save.log"
  echo "============================================================"
  tail -n 80 "$ROOT/outputs/ds_save.log" 2>/dev/null || true
}

wait_fence_save() {
  local pid_file="$ROOT/outputs/ds_fence_save.pid"
  local pid=""
  [ -f "$pid_file" ] && pid="$(cat "$pid_file" 2>/dev/null || true)"
  echo "[fence-save] Waiting for DeepStream fence save to finish..."
  if [ -n "$pid" ]; then
    while kill -0 "$pid" 2>/dev/null; do
      sleep 3
    done
  else
    while pgrep -x deepstream-app >/dev/null 2>&1; do
      sleep 3
    done
  fi
  echo "============================================================"
  echo "Fence saved output: $ROOT/outputs/yolov8s_person_fence_output.mp4"
  ls -lah "$ROOT/outputs/yolov8s_person_fence_output.mp4" || true
  echo "Fence config: $ROOT/configs/config_nvdsanalytics_fence.txt"
  echo "Log: $ROOT/outputs/ds_fence_save.log"
  echo "============================================================"
  tail -n 100 "$ROOT/outputs/ds_fence_save.log" 2>/dev/null || true
}

run_fence_nvdcf_save() {
  # New v11 mode: DeepStream nvinfer + nvtracker NvDCF, but custom v9-style
  # foot-point fence state machine and clean overlay. No nvdsanalytics.
  write_project_configs
  pkill -x deepstream-app 2>/dev/null || true
  pkill -f fence_nvdcf_deepstream.py 2>/dev/null || true
  rm -f "$ROOT/outputs/yolov8s_person_fence_output.mp4" \
        "$ROOT/outputs/fence_events.csv" \
        "$ROOT/outputs/fence_status.csv" \
        "$ROOT/outputs/ds_fence_nvdcf_save.log"

  if ! check_deepstream_python_runtime; then
    echo "ERROR: pyds/GStreamer Python runtime is missing; cannot run NvDCF v9 fence." | tee "$ROOT/outputs/ds_fence_nvdcf_save.log"
    cat /tmp/stage5_pyds_check.log | tee -a "$ROOT/outputs/ds_fence_nvdcf_save.log"
    return 1
  fi

  local input_uri
  input_uri="$(file_uri "$ROOT/$VIDEO_REL")"
  echo "[fence-nvdcf-save] Starting NvDCF + v9 foot-point fence from direct file input..."
  echo "[fence-nvdcf-save] URI: $input_uri"
  echo "[fence-nvdcf-save] Tracker: $ROOT/configs/tracker/config_tracker_NvDCF_accuracy.yml"
  echo "[fence-nvdcf-save] Logic: custom bottom-center foot-point, not nvdsanalytics"
  PYTHONPATH="/usr/local/lib/python3/dist-packages:/usr/lib/python3/dist-packages:${PYTHONPATH:-}" "$SYSTEM_PYTHON" "$ROOT/scripts/fence_nvdcf_deepstream.py" \
    --uri "$input_uri" \
    --sink file \
    --output "$ROOT/outputs/yolov8s_person_fence_output.mp4" \
    --infer-config "$ROOT/configs/deepstream_yolov8s/config_infer_primary_yolov8s_person.txt" \
    --tracker-config "$ROOT/configs/tracker/config_tracker_NvDCF_accuracy.yml" \
    --polygon "$ROOT/configs/fence/foot_polygon.txt" \
    --events-csv "$ROOT/outputs/fence_events.csv" \
    --status-csv "$ROOT/outputs/fence_status.csv" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fps 25 \
    > "$ROOT/outputs/ds_fence_nvdcf_save.log" 2>&1

  echo "============================================================"
  echo "NvDCF + v9 foot-point fence saved output: $ROOT/outputs/yolov8s_person_fence_output.mp4"
  ls -lah "$ROOT/outputs/yolov8s_person_fence_output.mp4" || true
  echo "Events: $ROOT/outputs/fence_events.csv"
  echo "Status: $ROOT/outputs/fence_status.csv"
  echo "Log:    $ROOT/outputs/ds_fence_nvdcf_save.log"
  echo "============================================================"
  tail -n 100 "$ROOT/outputs/ds_fence_nvdcf_save.log" 2>/dev/null || true
}

start_fence_nvdcf_live() {
  # New v11 live mode: reads RTSP /cam and publishes directly to MediaMTX /out.
  write_project_configs
  pkill -x deepstream-app 2>/dev/null || true
  pkill -f fence_nvdcf_deepstream.py 2>/dev/null || true
  rm -f "$ROOT/outputs/ds_fence_nvdcf_live.log" "$ROOT/outputs/ds_fence_nvdcf_live.pid"

  if ! check_deepstream_python_runtime; then
    echo "ERROR: pyds/GStreamer Python runtime is missing; cannot run NvDCF v9 fence." | tee "$ROOT/outputs/ds_fence_nvdcf_live.log"
    cat /tmp/stage5_pyds_check.log | tee -a "$ROOT/outputs/ds_fence_nvdcf_live.log"
    return 1
  fi

  # v20: for local-video fence-live, read the file directly just like
  # fence-save.  The previous RTSP /cam hop could connect but never delivered
  # frames to the Python/pyds pipeline in WSL/Docker.
  local input_uri
  input_uri="$(file_uri "$ROOT/$VIDEO_REL")"
  local output_uri="rtsp://mediamtx:8554/out"
  local udp_host="${FENCE_LIVE_UDP_HOST:-ffmpeg-ds-out}"
  local udp_port="${FENCE_LIVE_UDP_PORT:-5400}"
  echo "[fence-nvdcf-live] Starting NvDCF + v9 foot-point fence live..."
  echo "[fence-nvdcf-live] Input:  $input_uri"
  echo "[fence-nvdcf-live] Output bridge: UDP MPEG-TS -> ${udp_host}:${udp_port} -> MediaMTX /out"
  nohup env PYTHONPATH="/usr/local/lib/python3/dist-packages:/usr/lib/python3/dist-packages:${PYTHONPATH:-}" "$SYSTEM_PYTHON" "$ROOT/scripts/fence_nvdcf_deepstream.py" \
    --uri "$input_uri" \
    --sink udp-mpegts \
    --rtsp-output "$output_uri" \
    --udp-host "$udp_host" \
    --udp-port "$udp_port" \
    --infer-config "$ROOT/configs/deepstream_yolov8s/config_infer_primary_yolov8s_person.txt" \
    --tracker-config "$ROOT/configs/tracker/config_tracker_NvDCF_accuracy.yml" \
    --polygon "$ROOT/configs/fence/foot_polygon.txt" \
    --events-csv "$ROOT/outputs/fence_events.csv" \
    --status-csv "$ROOT/outputs/fence_status.csv" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fps 25 \
    --live \
    --loop-on-eos \
    --realtime \
    > "$ROOT/outputs/ds_fence_nvdcf_live.log" 2>&1 &
  echo $! > "$ROOT/outputs/ds_fence_nvdcf_live.pid"
  sleep 6
  echo "===== NVDCF V9 FENCE LIVE PROCESS ====="
  cat "$ROOT/outputs/ds_fence_nvdcf_live.pid" 2>/dev/null || true
  pgrep -a -f fence_nvdcf_deepstream.py || true
  echo "===== NVDCF V9 FENCE LIVE LOG ====="
  tail -n 80 "$ROOT/outputs/ds_fence_nvdcf_live.log" 2>/dev/null || true
}

run_save() {
  # Legacy blocking mode for manual container usage only.
  # It prepares before running, but does not start any RTSP publisher by itself.
  prepare
  start_save
  wait_save
}

stop_ds() {
  pkill -x deepstream-app 2>/dev/null || true
  pkill -f fence_nvdcf_deepstream.py 2>/dev/null || true
}

status_ds() {
  echo "===== DEEPSTREAM PROCESS ====="
  pgrep -a deepstream-app || true
  echo "===== LIVE LOG ====="
  tail -n 80 "$ROOT/outputs/ds_live.log" 2>/dev/null || true
  echo "===== SAVE LOG ====="
  tail -n 40 "$ROOT/outputs/ds_save.log" 2>/dev/null || true
  echo "===== FENCE LIVE LOG ====="
  tail -n 60 "$ROOT/outputs/ds_fence_live.log" 2>/dev/null || true
  echo "===== FENCE SAVE LOG ====="
  tail -n 60 "$ROOT/outputs/ds_fence_save.log" 2>/dev/null || true
  echo "===== NVDCF V9 FENCE SAVE LOG ====="
  tail -n 80 "$ROOT/outputs/ds_fence_nvdcf_save.log" 2>/dev/null || true
  echo "===== NVDCF V9 FENCE LIVE LOG ====="
  tail -n 80 "$ROOT/outputs/ds_fence_nvdcf_live.log" 2>/dev/null || true
}

case "$cmd" in
  prepare) prepare ;;
  start-live) start_live ;;
  start-save) start_save ;;
  wait-save) wait_save ;;
  start-fence-live) start_fence_live ;;
  start-fence-save) start_fence_save ;;
  wait-fence-save) wait_fence_save ;;
  run-fence-nvdcf-save) run_fence_nvdcf_save ;;
  start-fence-nvdcf-live) start_fence_nvdcf_live ;;
  run-save) run_save ;;
  stop) stop_ds ;;
  status) status_ds ;;
  *) echo "Unknown command: $cmd"; usage; exit 1 ;;
esac
