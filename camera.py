import threading
import time
from collections.abc import Callable

import av

from vision import Detection, LiveVisionPipeline, ProcessedFrame


class LatestFrameProcessor:
    def __init__(
        self,
        pipeline_factory: Callable[[], LiveVisionPipeline],
        on_result: Callable[[ProcessedFrame], None],
        on_error: Callable[[Exception], None],
        on_frame_received: Callable[[], None],
        on_stream_started: Callable[[], None],
        on_stream_ended: Callable[[], None],
        max_inference_rate: float = 10.0,
        max_overlay_age: float = 1.0,
    ) -> None:
        self._pipeline_factory = pipeline_factory
        self._on_result = on_result
        self._on_error = on_error
        self._on_frame_received = on_frame_received
        self._on_stream_started = on_stream_started
        self._on_stream_ended = on_stream_ended
        if max_inference_rate <= 0:
            raise ValueError("max_inference_rate must be greater than zero.")
        self._inference_interval = 1.0 / max_inference_rate
        self._max_overlay_age = max_overlay_age
        self._condition = threading.Condition()
        self._active = False
        self._generation = 0
        self._pending_frame: av.VideoFrame | None = None
        self._detections: tuple[Detection, ...] = ()
        self._detections_updated_at = 0.0
        self._failed_generation: int | None = None
        self._worker: threading.Thread | None = None

    def process_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        with self._condition:
            if not self._active:
                self._active = True
                self._generation += 1
                self._pending_frame = None
                self._detections = ()
                self._detections_updated_at = 0.0
                self._failed_generation = None
                self._on_stream_started()

            self._on_frame_received()
            if self._failed_generation != self._generation:
                self._pending_frame = frame
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(
                        target=self._run_worker,
                        name="lifevision-inference",
                        daemon=True,
                    )
                    self._worker.start()

            detections = self._detections
            overlay_is_fresh = (
                time.monotonic() - self._detections_updated_at
                <= self._max_overlay_age
            )
            self._condition.notify()

        if not detections or not overlay_is_fresh:
            return frame

        image = frame.to_ndarray(format="bgr24").copy()
        LiveVisionPipeline.render_detections(image, detections)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def stop_stream(self) -> None:
        with self._condition:
            if not self._active:
                return

            self._active = False
            self._generation += 1
            self._pending_frame = None
            self._detections = ()
            self._detections_updated_at = 0.0
            self._failed_generation = None
            self._condition.notify_all()

        self._on_stream_ended()

    def _run_worker(self) -> None:
        pipeline: LiveVisionPipeline | None = None
        pipeline_generation = -1
        last_inference_at = 0.0

        while True:
            with self._condition:
                while (
                    self._active
                    and self._pending_frame is None
                    and self._failed_generation != self._generation
                ):
                    self._condition.wait()
                if not self._active:
                    self._worker = None
                    return
                generation = self._generation
                if self._failed_generation == generation:
                    self._worker = None
                    return

            if pipeline is None or pipeline_generation != generation:
                try:
                    pipeline = self._pipeline_factory()
                except Exception as error:
                    with self._condition:
                        if not self._active or generation != self._generation:
                            pipeline = None
                            continue
                        self._failed_generation = generation
                        self._pending_frame = None
                        self._detections = ()
                        self._worker = None
                        self._condition.notify_all()
                        self._on_error(error)
                    return

                pipeline_generation = generation
                last_inference_at = 0.0

            with self._condition:
                while True:
                    if not self._active:
                        self._worker = None
                        return
                    if generation != self._generation:
                        pipeline = None
                        break
                    if self._failed_generation == generation:
                        self._worker = None
                        return
                    if self._pending_frame is None:
                        self._condition.wait()
                        continue

                    remaining = self._inference_interval - (
                        time.perf_counter() - last_inference_at
                    )
                    if remaining > 0:
                        self._condition.wait(timeout=remaining)
                        continue

                    frame = self._pending_frame
                    self._pending_frame = None
                    last_inference_at = time.perf_counter()
                    break

            if pipeline is None:
                continue

            try:
                result = pipeline.process_frame(
                    frame.to_ndarray(format="bgr24")
                )
            except Exception as error:
                with self._condition:
                    if self._active and generation == self._generation:
                        self._failed_generation = generation
                        self._pending_frame = None
                        self._detections = ()
                        self._detections_updated_at = 0.0
                        self._on_error(error)
                    self._condition.notify_all()
                pipeline = None
                continue

            with self._condition:
                if not self._active or generation != self._generation:
                    pipeline = None
                    continue
                self._detections = result.detections
                self._detections_updated_at = time.monotonic()
                self._on_result(result)
