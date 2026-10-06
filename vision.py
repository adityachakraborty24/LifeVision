import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count
from types import SimpleNamespace
from typing import Protocol

import cv2
import numpy as np
from ultralytics import YOLOWorld
from ultralytics.engine.results import Results
from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml


CLASS_PROMPTS = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "shoe",
    "shirt",
    "jacket",
    "coat",
    "dress",
    "pants",
    "skirt",
    "hat",
    "sunglasses",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "plate",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "orange",
    "strawberry",
    "grapes",
    "pear",
    "watermelon",
    "broccoli",
    "carrot",
    "cucumber",
    "bell pepper",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "sandwich",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "lamp",
    "television",
    "laptop",
    "computer monitor",
    "mouse",
    "remote control",
    "keyboard",
    "cell phone",
    "tablet",
    "headphones",
    "microwave",
    "oven",
    "toaster",
    "refrigerator",
    "sink",
    "frying pan",
    "cooking pot",
    "vacuum cleaner",
    "washing machine",
    "book",
    "notebook",
    "pen",
    "pencil",
    "ruler",
    "marker",
    "pencil case",
    "printer",
    "clock",
    "vase",
    "scissors",
    "wallet",
    "keys",
    "toothbrush",
    "teddy bear",
)

MODEL_NAME = "yolov8s-worldv2.pt"
IMAGE_SIZE = 640
INFERENCE_CONFIDENCE = 0.10
NMS_IOU_THRESHOLD = 0.55
PERSON_DISPLAY_CONFIDENCE = 0.25
OBJECT_DISPLAY_CONFIDENCE = 0.25
DUPLICATE_LABEL_IOU = 0.92
TRACK_BUFFER_FRAMES = 30

_BYTE_TRACKER_CONFIG = YAML.load(check_yaml("bytetrack.yaml"))


class Detector(Protocol):
    def infer(self, frame: np.ndarray) -> Results: ...


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    confidence: float
    track_id: str | None = None


@dataclass(frozen=True)
class ProcessedFrame:
    image: np.ndarray
    detections: tuple[Detection, ...]
    people_count: int
    tracked_people_count: int
    object_count: int
    tracked_object_count: int
    object_counts: dict[str, int]
    processing_fps: float
    frame_time_ms: float


def _new_byte_tracker() -> BYTETracker:
    config = dict(_BYTE_TRACKER_CONFIG)
    config["track_buffer"] = TRACK_BUFFER_FRAMES
    tracker = BYTETracker(SimpleNamespace(**config))
    track_ids = count(1)

    class SessionTrack(STrack):
        @staticmethod
        def next_id() -> int:
            return next(track_ids)

    tracker.track_class = SessionTrack
    return tracker


class YOLOWorldDetector:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model = YOLOWorld(model_name)
        self._model.set_classes(list(CLASS_PROMPTS))
        self._model_lock = threading.Lock()

    def infer(self, frame: np.ndarray) -> Results:
        with self._model_lock:
            return self._model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=INFERENCE_CONFIDENCE,
                iou=NMS_IOU_THRESHOLD,
                max_det=200,
                verbose=False,
            )[0]


class _ClassTrackSet:
    def __init__(self, group_name: str, next_id: Iterator[int]) -> None:
        self.group_name = group_name
        self.tracker = _new_byte_tracker()
        self._next_id = next_id
        self._local_ids: dict[int, int] = {}

    def update(
        self,
        boxes: object,
        frame: np.ndarray,
        max_lost_frames: int,
    ) -> tuple[dict[int, int], int]:
        self.tracker.max_frames_lost = max_lost_frames
        rows = self.tracker.update(boxes, frame)
        retained = self.tracker.tracked_stracks + self.tracker.lost_stracks
        retained_ids = {track.track_id for track in retained}

        for track_id in retained_ids:
            if track_id not in self._local_ids:
                self._local_ids[track_id] = next(self._next_id)

        for track_id in self._local_ids.keys() - retained_ids:
            del self._local_ids[track_id]

        detection_ids = {
            int(row[7]): self._local_ids[int(row[4])]
            for row in rows
        }
        return detection_ids, len(retained_ids)


class LiveVisionPipeline:
    def __init__(self, detector: Detector) -> None:
        self._detector = detector
        self._track_sets: dict[str, _ClassTrackSet] = {}
        self._track_id_counters: dict[str, Iterator[int]] = {}
        self._frame_lock = threading.Lock()
        self._estimated_fps = 20.0

    def process_frame(self, frame: np.ndarray) -> ProcessedFrame:
        started_at = time.perf_counter()
        with self._frame_lock:
            result = self._detector.infer(frame)
            raw_detections = self._read_detections(result)
            retained_indices = self._suppress_duplicate_labels(raw_detections)
            box_results = result.boxes
            if box_results is None:
                raise RuntimeError("The detector returned no box results.")

            visible_thresholds = {
                detection_index: (
                    PERSON_DISPLAY_CONFIDENCE
                    if raw_detections[detection_index][4] == "person"
                    else OBJECT_DISPLAY_CONFIDENCE
                )
                for detection_index in retained_indices
            }
            group_names = set(self._track_sets)
            group_names.update(
                raw_detections[index][4] for index in retained_indices
            )
            tracked_ids: dict[int, str] = {}
            tracked_counts: dict[str, int] = {}
            inactive_groups: list[str] = []
            lost_frames = max(
                5,
                min(TRACK_BUFFER_FRAMES, round(self._estimated_fps * 1.5)),
            )

            for group_name in sorted(group_names):
                if group_name not in self._track_sets:
                    if group_name not in self._track_id_counters:
                        self._track_id_counters[group_name] = count(1)
                    self._track_sets[group_name] = _ClassTrackSet(
                        group_name,
                        self._track_id_counters[group_name],
                    )

                indices = [
                    index
                    for index in retained_indices
                    if raw_detections[index][4] == group_name
                ]
                selected_boxes = (
                    box_results[indices] if indices else box_results[:0]
                )
                if hasattr(selected_boxes, "cpu"):
                    selected_boxes = selected_boxes.cpu()
                if hasattr(selected_boxes, "numpy"):
                    selected_boxes = selected_boxes.numpy()
                detection_ids, retained_count = self._track_sets[
                    group_name
                ].update(selected_boxes, frame, lost_frames)
                tracked_counts[group_name] = retained_count
                if retained_count == 0:
                    inactive_groups.append(group_name)

                for local_index, track_id in detection_ids.items():
                    raw_index = indices[local_index]
                    tracked_ids[raw_index] = (
                        f"P{track_id}"
                        if group_name == "person"
                        else f"{group_name.title()} #{track_id}"
                    )

            for group_name in inactive_groups:
                del self._track_sets[group_name]

            visible_detections = []
            for detection_index in retained_indices:
                x1, y1, x2, y2, class_name, confidence = raw_detections[
                    detection_index
                ]
                if confidence < visible_thresholds[detection_index]:
                    continue

                visible_detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        class_name=class_name,
                        confidence=confidence,
                        track_id=tracked_ids.get(detection_index),
                    )
                )

            annotated_image = self.render_detections(
                frame.copy(),
                visible_detections,
            )
            people = [
                detection
                for detection in visible_detections
                if detection.class_name == "person"
            ]
            objects = [
                detection
                for detection in visible_detections
                if detection.class_name != "person"
            ]

            object_counts = dict(
                sorted(Counter(item.class_name for item in objects).items())
            )
            frame_time_ms = (time.perf_counter() - started_at) * 1000
            processing_fps = (
                1000 / frame_time_ms if frame_time_ms > 0 else 0.0
            )
            self._estimated_fps = (
                processing_fps
                if self._estimated_fps == 0.0
                else self._estimated_fps * 0.8 + processing_fps * 0.2
            )

            return ProcessedFrame(
                image=annotated_image,
                detections=tuple(visible_detections),
                people_count=len(people),
                tracked_people_count=tracked_counts.get("person", 0),
                object_count=len(objects),
                tracked_object_count=sum(
                    count
                    for group_name, count in tracked_counts.items()
                    if group_name != "person"
                ),
                object_counts=object_counts,
                processing_fps=processing_fps,
                frame_time_ms=frame_time_ms,
            )

    @classmethod
    def render_detections(
        cls,
        image: np.ndarray,
        detections: tuple[Detection, ...] | list[Detection],
    ) -> np.ndarray:
        for detection in detections:
            if detection.class_name == "person":
                label = (
                    f"Person {detection.track_id or '...'} · "
                    f"{detection.confidence:.0%}"
                )
                color = (67, 220, 190)
            else:
                identity = detection.track_id or detection.class_name.title()
                label = f"{identity} · {detection.confidence:.0%}"
                color = (255, 183, 92)

            cls._draw_detection(image, detection, label, color)

        return image

    @staticmethod
    def _read_detections(
        result: Results,
    ) -> list[tuple[int, int, int, int, str, float]]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names
        rows = boxes.data
        if hasattr(rows, "detach"):
            rows = rows.detach()
        if hasattr(rows, "cpu"):
            rows = rows.cpu()
        if hasattr(rows, "numpy"):
            rows = rows.numpy()
        rows = np.asarray(rows)
        detections = []
        for row in rows:
            class_index = int(row[5])
            if isinstance(names, dict):
                class_name = names.get(class_index, str(class_index))
            else:
                class_name = names[class_index]
            x1, y1, x2, y2 = (int(round(value)) for value in row[:4])
            detections.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    str(class_name).lower(),
                    float(row[4]),
                )
            )
        return detections

    @classmethod
    def _suppress_duplicate_labels(
        cls,
        detections: list[tuple[int, int, int, int, str, float]],
    ) -> list[int]:
        ordered_indices = sorted(
            range(len(detections)),
            key=lambda index: detections[index][5],
            reverse=True,
        )
        kept: list[int] = []
        for index in ordered_indices:
            candidate = detections[index]
            if any(
                candidate[4] != detections[kept_index][4]
                and cls._intersection_over_union(
                    candidate,
                    detections[kept_index],
                )
                >= DUPLICATE_LABEL_IOU
                for kept_index in kept
            ):
                continue
            kept.append(index)
        return kept

    @staticmethod
    def _intersection_over_union(
        first: tuple[int, int, int, int, str, float],
        second: tuple[int, int, int, int, str, float],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(0, first[2] - first[0]) * max(
            0,
            first[3] - first[1],
        )
        second_area = max(0, second[2] - second[0]) * max(
            0,
            second[3] - second[1],
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _draw_detection(
        image: np.ndarray,
        detection: Detection,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        cv2.rectangle(
            image,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2,
            cv2.LINE_AA,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        text_width, text_height = cv2.getTextSize(
            label,
            font,
            scale,
            thickness,
        )[0]
        image_height, image_width = image.shape[:2]
        label_left = max(0, min(detection.x1, image_width - text_width - 12))
        label_top = max(0, detection.y1 - text_height - 10)
        label_bottom = min(image_height - 1, label_top + text_height + 10)
        baseline = min(image_height - 2, label_top + text_height + 5)

        cv2.rectangle(
            image,
            (label_left, label_top),
            (min(image_width - 1, label_left + text_width + 12), label_bottom),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (label_left + 6, baseline),
            font,
            scale,
            (20, 29, 38),
            thickness,
            cv2.LINE_AA,
        )
