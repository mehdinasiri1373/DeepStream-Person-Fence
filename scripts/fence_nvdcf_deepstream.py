#!/usr/bin/env python3
"""DeepStream NvDCF virtual fence runner with v9-style foot-point logic.

Pipeline:
  URI source -> nvstreammux -> nvinfer YOLOv8s -> nvtracker NvDCF
  -> custom pad-probe foot-point state machine -> nvdsosd -> H264 file/RTSP server

This keeps the tracker inside DeepStream/NvDCF, but intentionally does NOT use
nvdsanalytics for fence decisions.  The fence decision is the same idea as v9:
  foot = bottom-center of the axis-aligned bbox
  inside = point_in_polygon(foot, polygon)
  ENTER/EXIT = transition of inside state for a stable NvDCF object_id
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# DeepStream images commonly place pyds in the DeepStream lib directory rather
# than the default Python site-packages.
for _p in (
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3/dist-packages",
    "/opt/nvidia/deepstream/deepstream/lib",
    "/opt/nvidia/deepstream/deepstream/lib/python3/dist-packages",
    "/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/build",
):
    if _p not in sys.path and os.path.exists(_p):
        sys.path.append(_p)

try:
    import gi  # type: ignore
    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst  # type: ignore
    try:
        gi.require_version("GstRtsp", "1.0")
        from gi.repository import GstRtsp  # type: ignore
    except Exception:
        GstRtsp = None  # type: ignore
    try:
        gi.require_version("GstRtspServer", "1.0")
        from gi.repository import GstRtspServer  # type: ignore
    except Exception:
        GstRtspServer = None  # type: ignore
except Exception as exc:  # pragma: no cover - runtime environment check
    raise SystemExit(
        "ERROR: Python GStreamer bindings are missing. Install python3-gi and "
        "python3-gst-1.0 inside the DeepStream container.\n"
        f"Import error: {exc}"
    )

try:
    import pyds  # type: ignore
except Exception as exc:  # pragma: no cover - runtime environment check
    raise SystemExit(
        "ERROR: DeepStream Python binding 'pyds' was not found.\n"
        "Use a DeepStream container/image that includes Python bindings, or install "
        "deepstream_python_apps bindings in the container.\n"
        f"Import error: {exc}"
    )

Point = Tuple[int, int]
Box = Tuple[float, float, float, float]


@dataclass
class TrackState:
    smooth_box: Box
    last_inside: Optional[bool] = None
    hits: int = 0
    last_seen_frame: int = 0


class FenceState:
    def __init__(self, polygon: List[Point], fps: float, events_csv: Path, status_csv: Path, realtime: bool = False):
        self.polygon = polygon
        self.fps = float(fps) if fps > 0 else 25.0
        # When fence-live reads a local file, GStreamer can process it as fast
        # as the GPU allows (150-200 FPS). That is good for save mode, but bad
        # for browser live viewing.  Realtime pacing makes the file behave like
        # a camera stream at self.fps.
        self.realtime = bool(realtime)
        self._pace_start_time: Optional[float] = None
        self._pace_start_frame: Optional[int] = None
        self._pace_last_frame: Optional[int] = None
        self.tracks: Dict[int, TrackState] = {}
        # NvDCF object_id is a 64-bit globally-unique raw tracker ID.  It is
        # stable, but too long for on-screen display.  Keep it internally and
        # expose a short v9-style display ID in first-seen order.
        self.display_id_map: Dict[int, int] = {}
        self.next_display_id = 1
        self.enters = 0
        self.exits = 0
        self.last_log_time = time.monotonic()
        self.last_log_frame = 0
        self.events_f = events_csv.open("w", newline="", encoding="utf-8")
        self.status_f = status_csv.open("w", newline="", encoding="utf-8")
        self.events_w = csv.writer(self.events_f)
        self.status_w = csv.writer(self.status_f)
        self.events_w.writerow(["frame", "time_sec", "display_track_id", "raw_track_id", "event", "occupancy", "foot_x", "foot_y"])
        self.status_w.writerow(["frame", "time_sec", "display_track_id", "raw_track_id", "inside", "foot_x", "foot_y", "x1", "y1", "x2", "y2"])

    def display_id(self, raw_tid: int) -> int:
        short = self.display_id_map.get(raw_tid)
        if short is None:
            short = self.next_display_id
            self.next_display_id += 1
            self.display_id_map[raw_tid] = short
        return short

    def pace(self, frame_num: int) -> None:
        if not self.realtime:
            return
        now = time.monotonic()
        # On loop-on-EOS, DeepStream can reset frame_num. Restart pacing cleanly.
        if (
            self._pace_start_time is None
            or self._pace_start_frame is None
            or self._pace_last_frame is None
            or frame_num < self._pace_last_frame
        ):
            self._pace_start_time = now
            self._pace_start_frame = frame_num
            self._pace_last_frame = frame_num
            return
        target_elapsed = max(0.0, (frame_num - self._pace_start_frame) / max(self.fps, 1e-6))
        target_time = self._pace_start_time + target_elapsed
        sleep_s = target_time - now
        if sleep_s > 0:
            # Keep sleeps small so Ctrl+C/EOS remain responsive.
            time.sleep(min(sleep_s, 0.25))
        self._pace_last_frame = frame_num

    def close(self) -> None:
        try:
            self.events_f.flush(); self.events_f.close()
        finally:
            self.status_f.flush(); self.status_f.close()


def eprint(*args, **kwargs) -> None:
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


def point_in_poly(pt: Point, poly: List[Point]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        cross = (x - x1) * dy - (y - y1) * dx
        if abs(cross) < 1e-6:
            dot = (x - x1) * (x - x2) + (y - y1) * (y - y2)
            if dot <= 0:
                return True
        if (y1 > y) != (y2 > y):
            xinters = (dx * (y - y1) / (dy + 1e-12)) + x1
            if x <= xinters:
                inside = not inside
    return inside


def foot_of(box: Box) -> Point:
    x1, y1, x2, y2 = box
    return (int(round((x1 + x2) / 2.0)), int(round(y2)))


def smooth(prev: Box, cur: Box, alpha: float = 0.65) -> Box:
    px1, py1, px2, py2 = prev
    x1, y1, x2, y2 = cur
    return (
        alpha * x1 + (1.0 - alpha) * px1,
        alpha * y1 + (1.0 - alpha) * py1,
        alpha * x2 + (1.0 - alpha) * px2,
        alpha * y2 + (1.0 - alpha) * py2,
    )


def _bbox_from_obj(obj_meta) -> Box:
    """Prefer detector bbox geometry, but keep NvDCF ID from obj_meta.object_id.

    NvDCF can project/extend boxes during short detector jitter. For visual fence
    counting the user preferred the v9 detector-like geometry, so we draw an
    axis-aligned smoothed detector bbox when it is available. If it is not
    available, fall back to tracker rect_params.
    """
    try:
        det = obj_meta.detector_bbox_info.org_bbox_coords
        if float(det.width) > 1.0 and float(det.height) > 1.0:
            return (float(det.left), float(det.top), float(det.left + det.width), float(det.top + det.height))
    except Exception:
        pass
    rect = obj_meta.rect_params
    return (float(rect.left), float(rect.top), float(rect.left + rect.width), float(rect.top + rect.height))


def _hide_obj_overlay(obj_meta) -> None:
    """Disable native NvOSD object rendering.

    v23 used obj_meta.rect_params for person boxes.  In the live MPEG-TS/WebRTC
    path, some DeepStream/OSD builds render those tracker-updated object rects
    with corrupted/diagonal edges.  Keep object metadata for NvDCF and CSV, but
    draw clean axis-aligned boxes ourselves through frame display_meta instead.
    """
    try:
        obj_meta.rect_params.border_width = 0
        obj_meta.rect_params.has_bg_color = 0
        obj_meta.text_params.display_text = ""
    except Exception:
        pass


def _add_line(display_meta, idx: int, p1: Point, p2: Point, width: int = 3) -> None:
    lp = display_meta.line_params[idx]
    lp.x1 = int(p1[0]); lp.y1 = int(p1[1]); lp.x2 = int(p2[0]); lp.y2 = int(p2[1])
    lp.line_width = width
    lp.line_color.set(0.0, 1.0, 0.0, 1.0)


def _add_filled_rect(display_meta, idx: int, x: int, y: int, w: int, h: int, rgba: Tuple[float, float, float, float]) -> None:
    rp = display_meta.rect_params[idx]
    rp.left = int(x); rp.top = int(y); rp.width = int(w); rp.height = int(h)
    rp.border_width = 0
    rp.has_bg_color = 1
    rp.bg_color.set(*rgba)


def _add_box_rect(display_meta, idx: int, box: Box, inside: bool) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    rp = display_meta.rect_params[idx]
    rp.left = max(0, x1)
    rp.top = max(0, y1)
    rp.width = max(1, x2 - x1)
    rp.height = max(1, y2 - y1)
    rp.border_width = 3
    rp.has_bg_color = 0
    if inside:
        rp.border_color.set(0.0, 0.9, 0.0, 1.0)   # green
    else:
        rp.border_color.set(1.0, 0.0, 0.0, 1.0)   # red


def _set_label(text_param, display_id: int, box: Box, inside: bool) -> None:
    x1, y1, _x2, _y2 = [int(round(v)) for v in box]
    text_param.display_text = f"ID {display_id}" + (" IN" if inside else "")
    text_param.x_offset = max(0, x1)
    text_param.y_offset = max(18, y1 - 8)
    text_param.font_params.font_name = "Serif"
    text_param.font_params.font_size = 15
    text_param.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    text_param.set_bg_clr = 1
    text_param.text_bg_clr.set(0.0, 0.0, 0.0, 0.85)


def _add_display_meta(batch_meta, frame_meta, fence: FenceState, active: List[Tuple[int, Box, Point, bool]], occupancy: int) -> None:
    dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)

    # Fence polygon only.  Do not use line_params for person boxes.
    max_lines = min(len(fence.polygon), 16)
    dm.num_lines = max_lines
    for i in range(max_lines):
        _add_line(dm, i, fence.polygon[i], fence.polygon[(i + 1) % len(fence.polygon)], 3)

    # HUD label + per-person labels are drawn through display_meta, separate
    # from object_meta.  This avoids the slanted/corrupted live bbox overlay.
    dm.num_labels = min(1 + len(active), 16)
    tp = dm.text_params[0]
    tp.display_text = f"IN:{occupancy}  ENTER:{fence.enters}  EXIT:{fence.exits}"
    tp.x_offset = 28
    tp.y_offset = 119
    tp.font_params.font_name = "Serif"
    tp.font_params.font_size = 18
    tp.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    tp.set_bg_clr = 1
    tp.text_bg_clr.set(0.0, 0.0, 0.0, 0.85)

    # Clean axis-aligned person boxes.  NvOSD display_meta rects are stable in
    # the live WebRTC path, while obj_meta rect rendering can look diagonal on
    # some DeepStream/WSL combinations.
    rect_i = 0
    for display_id, box, _foot, inside in active[:16]:
        _add_box_rect(dm, rect_i, box, inside)
        rect_i += 1
    dm.num_rects = rect_i

    for label_i, (display_id, box, _foot, inside) in enumerate(active[:15], start=1):
        _set_label(dm.text_params[label_i], display_id, box, inside)

    pyds.nvds_add_display_meta_to_frame(frame_meta, dm)

def osd_sink_pad_buffer_probe(pad, info, user_data):
    fence: FenceState = user_data
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_num = int(frame_meta.frame_num)
        fence.pace(frame_num)
        active: List[Tuple[int, Box, Point, bool]] = []
        occupancy = 0
        seen_ids = set()

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            try:
                if int(obj_meta.class_id) != 0:
                    l_obj = l_obj.next
                    continue
                tid = int(obj_meta.object_id)
                if tid < 0 or tid >= 0xFFFFFFFFFFFFFFFF:
                    l_obj = l_obj.next
                    continue

                display_tid = fence.display_id(tid)
                cur_box = _bbox_from_obj(obj_meta)
                st = fence.tracks.get(tid)
                if st is None:
                    st = TrackState(smooth_box=cur_box, hits=0, last_seen_frame=frame_num)
                    fence.tracks[tid] = st
                else:
                    st.smooth_box = smooth(st.smooth_box, cur_box)
                st.hits += 1
                st.last_seen_frame = frame_num
                seen_ids.add(tid)

                # Same confirmation idea as v9: avoid drawing/counting one-frame ghosts.
                if st.hits < 2:
                    _hide_obj_overlay(obj_meta)
                    l_obj = l_obj.next
                    continue

                foot = foot_of(st.smooth_box)
                inside = point_in_poly(foot, fence.polygon)
                prev = st.last_inside
                if prev is None:
                    st.last_inside = inside
                elif prev != inside:
                    event = "ENTER" if inside else "EXIT"
                    if inside:
                        fence.enters += 1
                    else:
                        fence.exits += 1
                    st.last_inside = inside
                    fence.events_w.writerow([frame_num, f"{frame_num / fence.fps:.3f}", display_tid, tid, event, occupancy, foot[0], foot[1]])
                    fence.events_f.flush()

                if st.last_inside:
                    occupancy += 1
                x1, y1, x2, y2 = [int(round(v)) for v in st.smooth_box]
                fence.status_w.writerow([frame_num, f"{frame_num / fence.fps:.3f}", display_tid, tid, int(bool(st.last_inside)), foot[0], foot[1], x1, y1, x2, y2])

                _hide_obj_overlay(obj_meta)
                active.append((display_tid, st.smooth_box, foot, bool(st.last_inside)))
            except Exception as exc:
                eprint(f"[probe] object error: {exc}")

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        fence.status_f.flush()
        _add_display_meta(batch_meta, frame_meta, fence, active, occupancy)

        # Drop stale IDs to keep state bounded. NvDCF may re-use only unique IDs, so this is safe.
        stale = [tid for tid, st in fence.tracks.items() if frame_num - st.last_seen_frame > 250]
        for tid in stale:
            fence.tracks.pop(tid, None)

        if frame_num - fence.last_log_frame >= 100:
            now = time.monotonic()
            dt = max(now - fence.last_log_time, 1e-6)
            fps = (frame_num - fence.last_log_frame) / dt
            eprint(f"[fence-nvdcf] frame={frame_num} fps={fps:.1f} active={len(active)} occ={occupancy} enter={fence.enters} exit={fence.exits} ids={len(fence.tracks)}")
            fence.last_log_frame = frame_num
            fence.last_log_time = now

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    source_bin = data
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps(None)
    structure_name = caps.get_structure(0).get_name()
    features = caps.get_features(0)
    if not structure_name.startswith("video"):
        return
    if features and features.contains("memory:NVMM"):
        ghost_pad = source_bin.get_static_pad("src")
        if ghost_pad and not ghost_pad.set_target(decoder_src_pad):
            eprint("ERROR: failed to link decodebin src pad to source bin ghost pad")
    else:
        eprint("ERROR: decodebin did not pick an NVIDIA/NVMM decoder. Check DeepStream image/codecs.")


def decodebin_child_added(child_proxy, obj, name, user_data):
    # Keep RTSP input deterministic.  With MediaMTX inside Docker/WSL,
    # uridecodebin/rtspsrc may try UDP first, wait 5 seconds, then fallback to
    # TCP.  That startup race caused the live republisher to hit /ds-test too
    # early and get 503.  Force TCP from the beginning and keep latency modest.
    lname = (name or "").lower()
    try:
        factory = obj.get_factory()
        factory_name = factory.get_name().lower() if factory else ""
    except Exception:
        factory_name = ""
    is_rtsp_source = "rtspsrc" in factory_name or "rtspsrc" in lname or "source" in lname
    if is_rtsp_source:
        try:
            if obj.find_property("latency"):
                obj.set_property("latency", 100)
        except Exception:
            pass
        try:
            if obj.find_property("drop-on-latency"):
                obj.set_property("drop-on-latency", True)
        except Exception:
            pass
        try:
            if obj.find_property("protocols"):
                tcp_value = GstRtsp.RTSPLowerTrans.TCP if GstRtsp is not None else 4
                obj.set_property("protocols", tcp_value)
                eprint("[fence-nvdcf] RTSP input forced to TCP")
        except Exception as exc:
            eprint(f"[fence-nvdcf] WARNING: could not force RTSP TCP: {exc}")
    if name and "decodebin" in name:
        try:
            obj.connect("child-added", decodebin_child_added, user_data)
        except Exception:
            pass


def rtsp_src_pad_added(rtspsrc, src_pad, depay):
    """Link RTSP RTP pad explicitly into H264 depayloader.

    v18 used uridecodebin for the live RTSP input.  In Docker/WSL this can
    stall during UDP/TCP fallback and no frames reach NvDCF, which leaves the
    output RTSP server returning 503.  The live input path is H264-only from
    our ffmpeg-cam publisher, so use an explicit rtspsrc -> rtph264depay ->
    h264parse -> nvv4l2decoder chain and force TCP from the beginning.
    """
    caps = src_pad.get_current_caps() or src_pad.query_caps(None)
    cap_s = caps.to_string() if caps else ""
    if "application/x-rtp" not in cap_s or "H264" not in cap_s.upper():
        return
    sink_pad = depay.get_static_pad("sink")
    if not sink_pad or sink_pad.is_linked():
        return
    ret = src_pad.link(sink_pad)
    if ret != Gst.PadLinkReturn.OK:
        eprint(f"[fence-nvdcf] ERROR: failed to link RTSP pad to depay: {ret}")
    else:
        eprint("[fence-nvdcf] Explicit RTSP/H264 input linked")


def create_rtsp_h264_source_bin(index: int, uri: str):
    bin_name = f"source-bin-{index:02d}"
    source_bin = Gst.Bin.new(bin_name)
    if not source_bin:
        raise RuntimeError("Unable to create RTSP source bin")

    rtspsrc = make_element("rtspsrc", f"rtsp-source-{index}")
    depay = make_element("rtph264depay", f"rtsp-h264-depay-{index}")
    parser = make_element("h264parse", f"rtsp-h264-parser-{index}")
    decoder = make_element("nvv4l2decoder", f"rtsp-h264-decoder-{index}")

    rtspsrc.set_property("location", uri)
    set_if_property(rtspsrc, "latency", 100)
    set_if_property(rtspsrc, "drop-on-latency", True)
    try:
        if rtspsrc.find_property("protocols"):
            tcp_value = GstRtsp.RTSPLowerTrans.TCP if GstRtsp is not None else 4
            rtspsrc.set_property("protocols", tcp_value)
            eprint("[fence-nvdcf] RTSP input forced to TCP via explicit rtspsrc")
    except Exception as exc:
        eprint(f"[fence-nvdcf] WARNING: could not force explicit RTSP TCP: {exc}")

    for elem in (rtspsrc, depay, parser, decoder):
        source_bin.add(elem)
    if not depay.link(parser):
        raise RuntimeError("Failed to link RTSP depay -> h264parse")
    if not parser.link(decoder):
        raise RuntimeError("Failed to link RTSP h264parse -> nvv4l2decoder")
    rtspsrc.connect("pad-added", rtsp_src_pad_added, depay)

    decoder_src_pad = decoder.get_static_pad("src")
    if not decoder_src_pad:
        raise RuntimeError("Unable to get nvv4l2decoder src pad")
    ghost_pad = Gst.GhostPad.new("src", decoder_src_pad)
    source_bin.add_pad(ghost_pad)
    return source_bin


def create_uridecode_source_bin(index: int, uri: str):
    bin_name = f"source-bin-{index:02d}"
    source_bin = Gst.Bin.new(bin_name)
    if not source_bin:
        raise RuntimeError("Unable to create source bin")
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
    if not uri_decode_bin:
        raise RuntimeError("Unable to create uridecodebin")
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, source_bin)
    uri_decode_bin.connect("child-added", decodebin_child_added, None)
    source_bin.add(uri_decode_bin)
    ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    source_bin.add_pad(ghost_pad)
    return source_bin


def create_source_bin(index: int, uri: str):
    if uri.lower().startswith(("rtsp://", "rtsps://")):
        return create_rtsp_h264_source_bin(index, uri)
    return create_uridecode_source_bin(index, uri)


def make_element(factory: str, name: str):
    elem = Gst.ElementFactory.make(factory, name)
    if not elem:
        raise RuntimeError(f"Unable to create GStreamer element: {factory} ({name})")
    return elem


def set_if_property(element, prop: str, value) -> None:
    try:
        if element.find_property(prop):
            element.set_property(prop, value)
    except Exception:
        pass


def link_many(elements: Iterable) -> None:
    prev = None
    for elem in elements:
        if prev is not None and not prev.link(elem):
            raise RuntimeError(f"Failed to link {prev.get_name()} -> {elem.get_name()}")
        prev = elem


def bus_call(bus, message, ctx):
    """Handle GStreamer bus messages.

    For fence-live from a local video file we intentionally loop on EOS so the
    browser stream behaves like the existing `--mode live` test path.  For save
    mode and normal one-shot inputs we still quit on EOS.
    """
    loop = ctx["loop"]
    pipeline = ctx.get("pipeline")
    loop_on_eos = bool(ctx.get("loop_on_eos"))
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        if loop_on_eos and pipeline is not None:
            eprint("[fence-nvdcf] EOS; looping file source back to start")
            try:
                pipeline.seek_simple(
                    Gst.Format.TIME,
                    Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                    0,
                )
                return True
            except Exception as exc:
                eprint(f"WARNING: loop seek failed: {exc}")
        eprint("[fence-nvdcf] EOS")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        eprint(f"ERROR from {message.src.get_name()}: {err}")
        if debug:
            eprint(f"Debug info: {debug}")
        loop.quit()
    elif msg_type == Gst.MessageType.WARNING:
        warn, debug = message.parse_warning()
        eprint(f"WARNING from {message.src.get_name()}: {warn}")
        if debug:
            eprint(f"Debug info: {debug}")
    return True



def start_local_rtsp_server(args):
    """Serve the encoded RTP stream from a local UDP port as RTSP.

    Direct `rtph264pay -> rtspclientsink` was unreliable in this DS9/GStreamer
    container: it failed with GST_PAD_LINK_NOFORMAT before PLAYING.  This is the
    same robust pattern used by many DeepStream Python examples: the pipeline
    sends RTP/H264 to localhost UDP, and an in-process GstRtspServer publishes
    that UDP RTP stream as `/ds-test`.  The host runner can then re-publish
    rtsp://localhost:8555/ds-test to MediaMTX /out with ffmpeg, just like the
    original low-latency live mode.
    """
    if GstRtspServer is None:
        raise RuntimeError(
            "GstRtspServer Python bindings are missing. Install "
            "gir1.2-gst-rtsp-server-1.0 inside the DeepStream container."
        )

    server = GstRtspServer.RTSPServer.new()
    server.set_address("0.0.0.0")
    server.set_service(str(args.rtsp_server_port))

    factory = GstRtspServer.RTSPMediaFactory.new()
    # `pay0` must be an RTP payloader element for GstRtspServer.  The pipeline
    # sends RTP/H264 to localhost UDP; the server receives, depayloads/parses,
    # and re-payloads it as `pay0`.  This avoids the `GstUDPSrc has no property
    # named pt` warning and makes DESCRIBE succeed once frames are flowing.
    launch = (
        f'( udpsrc port={int(args.udp_port)} buffer-size=1048576 '
        f'caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, '
        f'encoding-name=(string)H264, payload=(int)96" '
        f'! rtpjitterbuffer latency=50 drop-on-latency=true '
        f'! rtph264depay ! h264parse config-interval=1 '
        f'! rtph264pay name=pay0 pt=96 config-interval=1 )'
    )
    factory.set_launch(launch)
    factory.set_shared(True)

    mounts = server.get_mount_points()
    path = args.rtsp_path if str(args.rtsp_path).startswith("/") else f"/{args.rtsp_path}"
    mounts.add_factory(path, factory)
    server.attach(None)
    eprint(f"[fence-nvdcf] RTSP server ready: rtsp://0.0.0.0:{args.rtsp_server_port}{path}")
    eprint(f"[fence-nvdcf] RTP/H264 UDP bridge: 127.0.0.1:{args.udp_port}")
    return server

def build_pipeline(args, fence_state: FenceState):
    Gst.init(None)
    pipeline = Gst.Pipeline.new("stage5-nvdcf-v9-fence")
    if not pipeline:
        raise RuntimeError("Unable to create pipeline")

    source_bin = create_source_bin(0, args.uri)
    streammux = make_element("nvstreammux", "stream-muxer")
    pgie = make_element("nvinfer", "primary-inference")
    tracker = make_element("nvtracker", "tracker")
    queue1 = make_element("queue", "queue-after-tracker")
    nvosd = make_element("nvdsosd", "onscreendisplay")
    nvvidconv = make_element("nvvideoconvert", "convertor")
    capsfilter = make_element("capsfilter", "capsfilter")
    encoder = make_element("nvv4l2h264enc", "h264-encoder")
    parser = make_element("h264parse", "h264-parser")

    # Source/stream geometry matches v9 canvas exactly.
    set_if_property(streammux, "width", args.width)
    set_if_property(streammux, "height", args.height)
    set_if_property(streammux, "batch-size", 1)
    set_if_property(streammux, "batched-push-timeout", 40000)
    set_if_property(streammux, "live-source", 1 if args.live else 0)
    set_if_property(streammux, "enable-padding", 0)
    set_if_property(streammux, "gpu-id", 0)

    pgie.set_property("config-file-path", args.infer_config)

    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", args.tracker_config)
    set_if_property(tracker, "tracker-width", args.tracker_width)
    set_if_property(tracker, "tracker-height", args.tracker_height)
    set_if_property(tracker, "gpu-id", 0)
    set_if_property(tracker, "display-tracking-id", 1)

    set_if_property(nvosd, "display-text", 1)
    set_if_property(nvosd, "display-bbox", 1)
    set_if_property(nvosd, "process-mode", 1)
    set_if_property(nvosd, "gpu-id", 0)

    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
    capsfilter.set_property("caps", caps)
    set_if_property(encoder, "bitrate", args.bitrate)
    set_if_property(encoder, "iframeinterval", int(args.fps))
    set_if_property(encoder, "insert-sps-pps", 1)
    set_if_property(encoder, "bufapi-version", 1)
    set_if_property(encoder, "preset-level", 1)
    set_if_property(encoder, "profile", 0)
    set_if_property(parser, "config-interval", 1)

    for elem in [source_bin, streammux, pgie, tracker, queue1, nvosd, nvvidconv, capsfilter, encoder, parser]:
        pipeline.add(elem)

    srcpad = source_bin.get_static_pad("src")
    sinkpad = streammux.get_request_pad("sink_0")
    if not srcpad or not sinkpad:
        raise RuntimeError("Unable to get source/streammux pads")
    if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Failed to link source bin to streammux")

    link_many([streammux, pgie, tracker, queue1, nvosd, nvvidconv, capsfilter, encoder, parser])

    if args.sink == "file":
        mux = make_element("qtmux", "mp4-muxer")
        sink = make_element("filesink", "file-sink")
        sink.set_property("location", args.output)
        set_if_property(sink, "sync", False)
        set_if_property(sink, "async", False)
        pipeline.add(mux); pipeline.add(sink)
        link_many([parser, mux, sink])
    elif args.sink == "rtsp":
        # Legacy RTSP-server bridge. Kept for debugging, but fence-live v22 uses
        # udp-mpegts because GstRtspServer over udpsrc can return RTSP 503 in
        # Docker/WSL even while the DeepStream pipeline is processing frames.
        rtsp_queue = make_element("queue", "queue-before-rtsp")
        pay = make_element("rtph264pay", "rtp-h264-pay")
        set_if_property(pay, "pt", 96)
        set_if_property(pay, "config-interval", 1)
        sink = make_element("udpsink", "rtp-udp-sink")
        sink.set_property("host", "127.0.0.1")
        sink.set_property("port", int(args.udp_port))
        set_if_property(sink, "async", False)
        set_if_property(sink, "sync", False)
        pipeline.add(rtsp_queue); pipeline.add(pay); pipeline.add(sink)
        link_many([parser, rtsp_queue, pay, sink])
        pipeline._stage5_rtsp_server = start_local_rtsp_server(args)  # keep Python ref alive
    elif args.sink == "udp-mpegts":
        # v22 live fix: avoid the in-process RTSP server completely.
        # Send MPEG-TS over UDP directly to the ffmpeg-ds-out container, which
        # then publishes MediaMTX /out.  MPEG-TS is self-describing enough for
        # ffmpeg to join mid-stream, unlike raw RTP that needs SDP/RTSP.
        udp_queue = make_element("queue", "queue-before-udp-mpegts")
        h264_caps = make_element("capsfilter", "h264-byte-stream-caps")
        h264_caps.set_property("caps", Gst.Caps.from_string("video/x-h264,stream-format=byte-stream,alignment=au"))
        mpegtsmux = make_element("mpegtsmux", "mpegts-muxer")
        sink = make_element("udpsink", "mpegts-udp-sink")
        sink.set_property("host", str(args.udp_host))
        sink.set_property("port", int(args.udp_port))
        set_if_property(sink, "async", False)
        # Keep the live file stream timestamp-paced instead of burst-sending.
        set_if_property(sink, "sync", True if args.realtime else False)
        pipeline.add(udp_queue); pipeline.add(h264_caps); pipeline.add(mpegtsmux); pipeline.add(sink)
        link_many([parser, udp_queue, h264_caps, mpegtsmux, sink])
        eprint(f"[fence-nvdcf] UDP MPEG-TS live output: {args.udp_host}:{args.udp_port}")
    else:
        sink = make_element("fakesink", "fake-sink")
        pipeline.add(sink)
        parser.link(sink)

    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        raise RuntimeError("Unable to get nvdsosd sink pad")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, fence_state)
    return pipeline


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True, help="file://... or rtsp://... input URI")
    ap.add_argument("--sink", choices=["file", "rtsp", "udp-mpegts", "fake"], default="file")
    ap.add_argument("--output", default="/workspace/stage5/outputs/yolov8s_person_fence_output.mp4")
    ap.add_argument("--rtsp-output", default="rtsp://host.docker.internal:8554/out")  # kept for log/backward compatibility
    ap.add_argument("--rtsp-server-port", type=int, default=8554)
    ap.add_argument("--rtsp-path", default="/ds-test")
    ap.add_argument("--udp-port", type=int, default=5400)
    ap.add_argument("--udp-host", default="127.0.0.1", help="UDP destination host for udp-mpegts live output")
    ap.add_argument("--infer-config", default="/workspace/stage5/configs/deepstream_yolov8s/config_infer_primary_yolov8s_person.txt")
    ap.add_argument("--tracker-config", default="/workspace/stage5/configs/tracker/config_tracker_NvDCF_accuracy.yml")
    ap.add_argument("--polygon", default="/workspace/stage5/configs/fence/foot_polygon.txt")
    ap.add_argument("--events-csv", default="/workspace/stage5/outputs/fence_events.csv")
    ap.add_argument("--status-csv", default="/workspace/stage5/outputs/fence_status.csv")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--bitrate", type=int, default=8000000)
    ap.add_argument("--tracker-width", type=int, default=960)
    ap.add_argument("--tracker-height", type=int, default=544)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--loop-on-eos", action="store_true", help="Loop file inputs when streaming live")
    ap.add_argument("--realtime", action="store_true", help="Throttle file-input live mode to --fps instead of processing as fast as possible")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.events_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.status_csv).parent.mkdir(parents=True, exist_ok=True)
    polygon = read_polygon(args.polygon)
    eprint("============================================================")
    eprint("Stage5 NvDCF + v9 foot-point fence")
    eprint(f"URI:      {args.uri}")
    eprint(f"Sink:     {args.sink}")
    if args.sink == "file":
        eprint(f"Output:   {args.output}")
    elif args.sink == "rtsp":
        eprint(f"RTSP server: rtsp://0.0.0.0:{args.rtsp_server_port}{args.rtsp_path}")
        eprint(f"RTP UDP:     127.0.0.1:{args.udp_port}")
        eprint(f"Final /out:  {args.rtsp_output}  # published by host ffmpeg")
    elif args.sink == "udp-mpegts":
        eprint(f"UDP MPEG-TS: {args.udp_host}:{args.udp_port}")
        eprint(f"Final /out:  {args.rtsp_output}  # published by ffmpeg-ds-out -> MediaMTX")
    eprint(f"Infer:    {args.infer_config}")
    eprint(f"Tracker:  {args.tracker_config}")
    eprint(f"Polygon:  {polygon}")
    eprint("Fence:    custom bottom-center foot-point state machine, not nvdsanalytics")
    if args.realtime:
        eprint(f"Realtime: enabled, target_fps={args.fps}")
    eprint("============================================================")

    fence_state = FenceState(polygon, args.fps, Path(args.events_csv), Path(args.status_csv), realtime=args.realtime)
    loop = GLib.MainLoop()
    pipeline = build_pipeline(args, fence_state)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    loop_on_eos = bool(args.loop_on_eos and str(args.uri).startswith("file://"))
    if loop_on_eos:
        eprint("[fence-nvdcf] file live mode: loop-on-EOS enabled")
    bus.connect("message", bus_call, {"loop": loop, "pipeline": pipeline, "loop_on_eos": loop_on_eos})

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        fence_state.close()
        raise RuntimeError("Unable to set pipeline to PLAYING")
    try:
        loop.run()
    except KeyboardInterrupt:
        eprint("[fence-nvdcf] interrupted")
    finally:
        pipeline.set_state(Gst.State.NULL)
        fence_state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
