from unittest.mock import Mock, patch

import numpy as np
from ultralytics.engine.results import Results

from vision import (
    CLASS_PROMPTS,
    INFERENCE_CONFIDENCE,
    IMAGE_SIZE,
    MODEL_NAME,
    NMS_IOU_THRESHOLD,
    LiveVisionPipeline,
    ProcessedFrame,
    YOLOWorldDetector,
)


FRAME = np.zeros((360, 640, 3), dtype=np.uint8)
NAMES = {0: "person", 1: "bottle", 2: "cup"}


def make_result(rows: list[list[float]]) -> Results:
    return Results(
        orig_img=FRAME,
        path="webcam",
        names=NAMES,
        boxes=np.asarray(rows, dtype=np.float32).reshape(-1, 6),
    )


class SequenceDetector:
    def __init__(self, results: list[Results]) -> None:
        self._results = iter(results)

    def infer(self, _frame: np.ndarray) -> Results:
        assert _frame.shape == FRAME.shape
        return next(self._results)


def test_world_detector_uses_real_prompts_and_nms_settings() -> None:
    model = Mock()
    result = make_result([[20, 30, 120, 260, 0.92, 0]])
    model.predict.return_value = [result]

    with patch("vision.YOLOWorld", return_value=model):
        detector = YOLOWorldDetector()

    assert len(CLASS_PROMPTS) > 80
    assert len(CLASS_PROMPTS) == len(set(CLASS_PROMPTS))
    model.set_classes.assert_called_once_with(list(CLASS_PROMPTS))
    assert detector.infer(FRAME) is result
    model.predict.assert_called_once_with(
        source=FRAME,
        imgsz=IMAGE_SIZE,
        conf=INFERENCE_CONFIDENCE,
        iou=NMS_IOU_THRESHOLD,
        max_det=200,
        verbose=False,
    )
    assert MODEL_NAME == "yolov8s-worldv2.pt"


def test_pipeline_keeps_counts_separate_and_tracks_through_a_missed_frame() -> None:
    detector = SequenceDetector(
        [
            make_result(
                [
                    [10, 20, 65, 150, 0.92, 0],
                    [250, 18, 305, 152, 0.88, 0],
                    [105, 100, 145, 160, 0.86, 1],
                    [105, 100, 145, 160, 0.71, 2],
                ]
            ),
            make_result(
                [
                    [255, 18, 310, 152, 0.89, 0],
                    [15, 20, 70, 150, 0.94, 0],
                    [110, 100, 150, 160, 0.87, 1],
                ]
            ),
            make_result([]),
            make_result(
                [
                    [20, 20, 75, 150, 0.93, 0],
                    [115, 100, 155, 160, 0.90, 1],
                ]
            ),
        ]
    )
    pipeline = LiveVisionPipeline(detector)

    first = pipeline.process_frame(FRAME)
    second = pipeline.process_frame(FRAME)
    missed = pipeline.process_frame(FRAME)
    returned = pipeline.process_frame(FRAME)

    assert first.people_count == 2
    assert first.tracked_people_count == 2
    assert first.object_count == 1
    assert first.tracked_object_count == 1
    assert first.object_counts == {"bottle": 1}
    assert second.people_count == 2
    assert second.object_count == 1
    assert person_track_ids(first) == person_track_ids(second)
    assert object_ids(first) == object_ids(second)
    assert missed.people_count == 0
    assert missed.object_count == 0
    assert missed.tracked_people_count == 2
    assert missed.tracked_object_count == 1
    assert returned.people_count == 1
    assert returned.tracked_people_count == 2
    assert returned.object_count == 1
    assert returned.tracked_object_count == 1
    assert returned.object_counts == {"bottle": 1}
    assert returned.image.shape == FRAME.shape
    assert person_track_ids(returned)[0] == person_track_ids(first)[0]
    assert object_ids(returned) == object_ids(first)


def person_track_ids(result: ProcessedFrame) -> list[str | None]:
    people = sorted(
        (item for item in result.detections if item.class_name == "person"),
        key=lambda item: item.x1,
    )
    return [item.track_id for item in people]


def object_ids(result: ProcessedFrame) -> dict[str, str | None]:
    return {
        detection.class_name: detection.track_id
        for detection in result.detections
        if detection.class_name != "person"
    }


def test_track_identity_expires_after_person_leaves() -> None:
    results = [make_result([[10, 20, 65, 150, 0.92, 0]])]
    results.extend(make_result([]) for _ in range(35))
    results.append(make_result([[10, 20, 65, 150, 0.94, 0]]))
    pipeline = LiveVisionPipeline(SequenceDetector(results))

    first = pipeline.process_frame(FRAME)
    for _ in range(35):
        last_empty = pipeline.process_frame(FRAME)

    assert first.detections[0].track_id is not None
    assert last_empty.tracked_people_count == 0
    assert pipeline._track_sets == {}

    reentered = pipeline.process_frame(FRAME)
    assert reentered.tracked_people_count == 1
    assert reentered.detections[0].track_id == "P2"


def test_distinct_overlapping_objects_are_not_collapsed() -> None:
    pipeline = LiveVisionPipeline(
        SequenceDetector(
            [
                make_result(
                    [
                        [100, 100, 200, 200, 0.91, 1],
                        [150, 100, 250, 200, 0.85, 2],
                    ]
                )
            ]
        )
    )

    result = pipeline.process_frame(FRAME)

    assert result.object_count == 2
    assert result.object_counts == {"bottle": 1, "cup": 1}


def test_pipeline_ids_are_isolated_between_camera_sessions() -> None:
    first_session = LiveVisionPipeline(
        SequenceDetector(
            [
                make_result([[10, 20, 65, 150, 0.92, 0]]),
                make_result([[15, 20, 70, 150, 0.94, 0]]),
            ]
        )
    )
    second_session = LiveVisionPipeline(
        SequenceDetector([make_result([[210, 20, 265, 150, 0.91, 0]])])
    )

    first_detection = first_session.process_frame(FRAME).detections[0]
    second_detection = second_session.process_frame(FRAME).detections[0]
    first_returned = first_session.process_frame(FRAME).detections[0]

    assert first_detection.track_id == "P1"
    assert second_detection.track_id == "P1"
    assert first_returned.track_id == first_detection.track_id
