import threading
import time
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="LifeVision",
    page_icon="🔎",
    layout="wide",
)

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    from camera import LatestFrameProcessor
    from vision import LiveVisionPipeline, ProcessedFrame, YOLOWorldDetector
except ImportError as error:
    st.error(
        "A required computer-vision dependency is unavailable. "
        "Install the project dependencies with "
        "`python -m pip install -r requirements.txt` and restart LifeVision. "
        f"Details: {error}"
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def load_detector() -> YOLOWorldDetector:
    return YOLOWorldDetector()


class LiveMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] = {
            "people_count": 0,
            "tracked_people_count": 0,
            "object_count": 0,
            "tracked_object_count": 0,
            "object_counts": {},
            "processing_fps": 0.0,
            "frame_time_ms": 0.0,
            "last_frame_at": None,
            "error": None,
        }

    def start_stream(self) -> None:
        with self._lock:
            self._snapshot.update(
                people_count=0,
                tracked_people_count=0,
                object_count=0,
                tracked_object_count=0,
                object_counts={},
                processing_fps=0.0,
                frame_time_ms=0.0,
                last_frame_at=None,
                error=None,
            )

    def mark_frame_received(self) -> None:
        with self._lock:
            self._snapshot["last_frame_at"] = time.monotonic()

    def end_stream(self) -> None:
        self.start_stream()

    def update(self, result: ProcessedFrame) -> None:
        with self._lock:
            previous_fps = self._snapshot["processing_fps"]
            smoothed_fps = (
                result.processing_fps
                if previous_fps == 0.0
                else previous_fps * 0.8 + result.processing_fps * 0.2
            )
            self._snapshot.update(
                people_count=result.people_count,
                tracked_people_count=result.tracked_people_count,
                object_count=result.object_count,
                tracked_object_count=result.tracked_object_count,
                object_counts=dict(result.object_counts),
                processing_fps=smoothed_fps,
                frame_time_ms=result.frame_time_ms,
                error=None,
            )

    def set_error(self, error: Exception) -> None:
        with self._lock:
            self._snapshot.update(
                people_count=0,
                tracked_people_count=0,
                object_count=0,
                tracked_object_count=0,
                object_counts={},
                processing_fps=0.0,
                frame_time_ms=0.0,
                error=f"Detection is temporarily unavailable: {error}",
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = dict(self._snapshot)
            snapshot["object_counts"] = dict(self._snapshot["object_counts"])
            return snapshot


if "lifevision_metrics" not in st.session_state:
    st.session_state["lifevision_metrics"] = LiveMetrics()

metrics: LiveMetrics = st.session_state["lifevision_metrics"]
if "lifevision_camera_processor" not in st.session_state:
    st.session_state["lifevision_camera_processor"] = LatestFrameProcessor(
        pipeline_factory=lambda: LiveVisionPipeline(load_detector()),
        on_result=metrics.update,
        on_error=metrics.set_error,
        on_frame_received=metrics.mark_frame_received,
        on_stream_started=metrics.start_stream,
        on_stream_ended=metrics.end_stream,
    )

camera_processor: LatestFrameProcessor = st.session_state[
    "lifevision_camera_processor"
]

st.title("LifeVision")
st.caption("Person detection, object detection, and live tracking")

camera_column, dashboard_column = st.columns([2.2, 1], gap="large")

with camera_column:
    with st.container(border=True):
        st.subheader("Live camera")
        st.caption("Allow camera access in your browser to start detection.")
        webrtc_streamer(
            key="lifevision-camera",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=camera_processor.process_frame,
            on_video_ended=camera_processor.stop_stream,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "frameRate": {"ideal": 30, "max": 30},
                },
                "audio": False,
            },
            async_processing=False,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },
        )


@st.fragment(run_every=0.5)
def render_dashboard(live_metrics: LiveMetrics) -> None:
    snapshot = live_metrics.snapshot()
    last_frame_at = snapshot["last_frame_at"]
    camera_active = (
        last_frame_at is not None
        and time.monotonic() - last_frame_at < 5.0
    )

    with st.container(border=True):
        status = "Camera active" if camera_active else "Waiting for camera"
        st.markdown(f"**Status** · {status}")

        people_column, tracked_people_column = st.columns(2)
        people_column.metric("People", snapshot["people_count"])
        tracked_people_column.metric(
            "Tracked People",
            snapshot["tracked_people_count"],
        )

        objects_column, tracked_objects_column = st.columns(2)
        objects_column.metric("Objects", snapshot["object_count"])
        tracked_objects_column.metric(
            "Tracked Objects",
            snapshot["tracked_object_count"],
        )

        st.markdown("#### Object breakdown")
        object_counts = snapshot["object_counts"]
        if object_counts:
            for name, count in object_counts.items():
                st.write(f"{name.title()} × {count}")
        else:
            st.caption("No recognized objects in the current frame.")

        st.divider()
        performance_column, latency_column = st.columns(2)
        performance_column.metric(
            "Processing FPS",
            f"{snapshot['processing_fps']:.1f}",
        )
        latency_column.metric(
            "Frame time",
            f"{snapshot['frame_time_ms']:.0f} ms",
        )

        if snapshot["error"]:
            st.error(snapshot["error"])


with dashboard_column:
    render_dashboard(metrics)

st.caption(
    "Visible counts describe this frame. Tracked totals keep IDs through "
    "brief detection gaps; IDs are temporary and camera-session local."
)
