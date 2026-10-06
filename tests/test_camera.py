import threading

import av
import numpy as np

from camera import LatestFrameProcessor
from vision import Detection, ProcessedFrame


def make_frame(value: int) -> av.VideoFrame:
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    return av.VideoFrame.from_ndarray(image, format="bgr24")


def make_result(image: np.ndarray) -> ProcessedFrame:
    return ProcessedFrame(
        image=image,
        detections=(),
        people_count=0,
        tracked_people_count=0,
        object_count=0,
        tracked_object_count=0,
        object_counts={},
        processing_fps=10.0,
        frame_time_ms=100.0,
    )


class ControlledPipeline:
    def __init__(
        self,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.started = started
        self.release = release

    def process_frame(self, image: np.ndarray) -> ProcessedFrame:
        value = int(image[0, 0, 0])
        if value == 1 and self.started is not None and self.release is not None:
            self.started.set()
            assert self.release.wait(timeout=2.0)
        return make_result(image)


def test_processor_returns_live_frame_and_discards_stale_pending_frames() -> None:
    inference_started = threading.Event()
    release_inference = threading.Event()
    latest_processed = threading.Event()
    processed_values: list[int] = []

    def handle_result(result: ProcessedFrame) -> None:
        processed_values.append(int(result.image[0, 0, 0]))
        if processed_values[-1] == 3:
            latest_processed.set()

    processor = LatestFrameProcessor(
        pipeline_factory=lambda: ControlledPipeline(
            inference_started,
            release_inference,
        ),
        on_result=handle_result,
        on_error=lambda error: (_ for _ in ()).throw(error),
        on_frame_received=lambda: None,
        on_stream_started=lambda: None,
        on_stream_ended=lambda: None,
        max_inference_rate=60.0,
    )

    first_frame = make_frame(1)
    assert processor.process_frame(first_frame) is first_frame
    assert inference_started.wait(timeout=2.0)

    second_frame = make_frame(2)
    third_frame = make_frame(3)
    assert processor.process_frame(second_frame) is second_frame
    assert processor.process_frame(third_frame) is third_frame

    release_inference.set()
    assert latest_processed.wait(timeout=2.0)
    processor.stop_stream()

    assert processed_values == [1, 3]


def test_processor_stops_and_restarts_with_fresh_tracker_pipeline() -> None:
    factory_calls = 0
    started_count = 0
    ended_count = 0
    result_count = 0
    first_result = threading.Event()
    second_result = threading.Event()
    errors: list[Exception] = []

    def create_pipeline() -> ControlledPipeline:
        nonlocal factory_calls
        factory_calls += 1
        return ControlledPipeline()

    def handle_result(result: ProcessedFrame) -> None:
        assert result.frame_time_ms > 0
        nonlocal result_count
        result_count += 1
        if result_count == 1:
            first_result.set()
        if result_count == 2:
            second_result.set()

    def handle_start() -> None:
        nonlocal started_count
        started_count += 1

    def handle_end() -> None:
        nonlocal ended_count
        ended_count += 1

    processor = LatestFrameProcessor(
        pipeline_factory=create_pipeline,
        on_result=handle_result,
        on_error=errors.append,
        on_frame_received=lambda: None,
        on_stream_started=handle_start,
        on_stream_ended=handle_end,
        max_inference_rate=60.0,
    )

    processor.process_frame(make_frame(4))
    assert first_result.wait(timeout=2.0)
    assert result_count == 1

    processor.stop_stream()
    processor.process_frame(make_frame(5))
    assert second_result.wait(timeout=2.0)
    processor.stop_stream()

    assert factory_calls == 2
    assert started_count == 2
    assert ended_count == 2
    assert errors == []


def test_callback_draws_last_results_on_latest_frame() -> None:
    result_ready = threading.Event()
    detected = Detection(
        x1=2,
        y1=2,
        x2=20,
        y2=20,
        class_name="person",
        confidence=0.9,
        track_id="P1",
    )

    class DetectionPipeline:
        def process_frame(self, image: np.ndarray) -> ProcessedFrame:
            return ProcessedFrame(
                image=image,
                detections=(detected,),
                people_count=1,
                tracked_people_count=1,
                object_count=0,
                tracked_object_count=0,
                object_counts={},
                processing_fps=10.0,
                frame_time_ms=100.0,
            )

    processor = LatestFrameProcessor(
        pipeline_factory=DetectionPipeline,
        on_result=lambda result: (
            result_ready.set() if result.people_count == 1 else None
        ),
        on_error=lambda error: (_ for _ in ()).throw(error),
        on_frame_received=lambda: None,
        on_stream_started=lambda: None,
        on_stream_ended=lambda: None,
    )

    first = make_frame(0)
    assert processor.process_frame(first) is first
    assert result_ready.wait(timeout=2.0)

    latest = make_frame(0)
    rendered = processor.process_frame(latest)
    processor.stop_stream()

    assert rendered is not latest
    assert np.any(
        rendered.to_ndarray(format="bgr24")
        != latest.to_ndarray(format="bgr24")
    )
