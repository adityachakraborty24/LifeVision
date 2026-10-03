import os
import time
import math
import threading
import urllib.request
from dataclasses import dataclass, field
from collections import deque

import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    python = None
    vision = None
    MEDIAPIPE_AVAILABLE = False


APP_VERSION = "15.0"

OBJECT_MODEL = "yolo11n.pt"
POSE_MODEL = "yolo11n-pose.pt"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "lifevision"
)

HAND_MODEL_PATH = os.path.join(
    HAND_CACHE_DIR,
    "hand_landmarker.task"
)

OBJECT_GROUPS = {
    "People": {"person"},
    "Animals": {
        "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe"
    },
    "Vehicles": {
        "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat"
    },
    "Indoor": {
        "chair", "couch", "bed", "dining table", "tv",
        "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear"
    },
    "Food": {
        "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake"
    },
    "Sports": {
        "sports ball", "skateboard", "surfboard",
        "tennis racket", "baseball bat", "baseball glove"
    }
}

POSE_CONNECTIONS = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 5), (0, 6)
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]


@dataclass
class ObjectState:
    label: str
    confidence: float
    bbox: tuple
    track_id: int
    center: tuple
    area: float


@dataclass
class PersonState:
    person_id: int
    bbox: tuple
    center: tuple
    confidence: float
    keypoints: list = field(default_factory=list)
    posture: str = "Unknown"
    movement: str = "Still"
    velocity: float = 0.0
    gestures: list = field(default_factory=list)
    hands: list = field(default_factory=list)
    last_center: tuple = (0, 0)
    history: deque = field(
        default_factory=lambda: deque(maxlen=12)
    )


@dataclass
class HandState:
    hand_id: int
    handedness: str
    landmarks: list
    bbox: tuple
    gesture: str
    confidence: float


@dataclass
class Event:
    timestamp: float
    text: str
    category: str


@dataclass
class Snapshot:
    timestamp: float
    frame_id: int
    objects: list
    people: list
    hands: list
    scene: str
    narrative: str
    fps: float
    ai_fps: float
    hand_available: bool
    runtime_error: str


class Tracker:
    def __init__(self):
        self.next_id = 1
        self.previous = {}

    @staticmethod
    def distance(a, b):
        return math.hypot(
            float(a[0]) - float(b[0]),
            float(a[1]) - float(b[1])
        )

    def assign(self, detections, max_distance=130):
        if not detections:
            self.previous = {}
            return []

        old = list(self.previous.items())
        used = set()
        result = []

        for detection in detections:
            best_id = None
            best_distance = max_distance

            for object_id, previous_center in old:
                if object_id in used:
                    continue

                distance = self.distance(
                    detection["center"],
                    previous_center
                )

                if distance < best_distance:
                    best_distance = distance
                    best_id = object_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            used.add(best_id)
            detection["track_id"] = best_id
            result.append(detection)

        self.previous = {
            d["track_id"]: d["center"]
            for d in result
        }

        return result


class Motion:
    def __init__(self):
        self.previous = {}

    def update(self, person):
        old = self.previous.get(person.person_id)

        if old is None:
            speed = 0.0
        else:
            speed = math.hypot(
                person.center[0] - old[0],
                person.center[1] - old[1]
            )

        person.velocity = speed

        if speed < 3:
            person.movement = "Still"
        elif speed < 12:
            person.movement = "Moving"
        else:
            person.movement = "Fast movement"

        person.history.append(person.center)
        self.previous[person.person_id] = person.center


class Kinematics:
    @staticmethod
    def posture_from_keypoints(points):
        if len(points) < 17:
            return "Unknown"

        try:
            nose = points[0]
            left_shoulder = points[5]
            right_shoulder = points[6]
            left_hip = points[11]
            right_hip = points[12]
            left_knee = points[13]
            right_knee = points[14]
            left_ankle = points[15]
            right_ankle = points[16]

            required = [
                nose,
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip,
                left_knee,
                right_knee,
                left_ankle,
                right_ankle
            ]

            if any(
                len(p) < 3 or p[2] < 0.25
                for p in required
            ):
                return "Unknown"

            shoulder_y = (
                left_shoulder[1] +
                right_shoulder[1]
            ) / 2

            hip_y = (
                left_hip[1] +
                right_hip[1]
            ) / 2

            knee_y = (
                left_knee[1] +
                right_knee[1]
            ) / 2

            ankle_y = (
                left_ankle[1] +
                right_ankle[1]
            ) / 2

            torso = abs(
                hip_y - shoulder_y
            )

            leg = abs(
                ankle_y - knee_y
            )

            shoulder_width = abs(
                left_shoulder[0] -
                right_shoulder[0]
            )

            hip_width = abs(
                left_hip[0] -
                right_hip[0]
            )

            if torso < 35 and leg < 45:
                return "Sitting"

            if torso < 55 and leg < 70:
                return "Sitting"

            if torso > 70 and leg > 55:
                return "Standing"

            if shoulder_width > 0 and hip_width > 0:
                ratio = torso / max(
                    shoulder_width,
                    hip_width,
                    1
                )

                if ratio < 1:
                    return "Lying"

            return "Standing"

        except Exception:
            return "Unknown"


class GestureRecognizer:
    @staticmethod
    def distance(a, b):
        return math.hypot(
            a[0] - b[0],
            a[1] - b[1]
        )

    @staticmethod
    def classify(points):
        if len(points) != 21:
            return "Unknown"

        try:
            wrist = points[0]

            thumb_tip = points[4]
            index_tip = points[8]
            middle_tip = points[12]
            ring_tip = points[16]
            pinky_tip = points[20]

            index_mcp = points[5]
            middle_mcp = points[9]
            ring_mcp = points[13]
            pinky_mcp = points[17]

            index_extended = (
                GestureRecognizer.distance(
                    index_tip,
                    wrist
                )
                >
                GestureRecognizer.distance(
                    index_mcp,
                    wrist
                ) * 1.18
            )

            middle_extended = (
                GestureRecognizer.distance(
                    middle_tip,
                    wrist
                )
                >
                GestureRecognizer.distance(
                    middle_mcp,
                    wrist
                ) * 1.18
            )

            ring_extended = (
                GestureRecognizer.distance(
                    ring_tip,
                    wrist
                )
                >
                GestureRecognizer.distance(
                    ring_mcp,
                    wrist
                ) * 1.12
            )

            pinky_extended = (
                GestureRecognizer.distance(
                    pinky_tip,
                    wrist
                )
                >
                GestureRecognizer.distance(
                    pinky_mcp,
                    wrist
                ) * 1.10
            )

            thumb_extended = (
                GestureRecognizer.distance(
                    thumb_tip,
                    wrist
                )
                >
                GestureRecognizer.distance(
                    points[2],
                    wrist
                ) * 1.12
            )

            extended = [
                index_extended,
                middle_extended,
                ring_extended,
                pinky_extended
            ]

            if all(extended) and thumb_extended:
                return "Open hand"

            if not any(extended) and not thumb_extended:
                return "Fist"

            if (
                thumb_extended
                and not index_extended
                and not middle_extended
                and not ring_extended
                and not pinky_extended
            ):
                return "Thumbs up"

            if (
                index_extended
                and not middle_extended
                and not ring_extended
                and not pinky_extended
            ):
                return "Pointing"

            if (
                middle_extended
                and not index_extended
                and not ring_extended
                and not pinky_extended
            ):
                return "Middle finger"

            if (
                index_extended
                and middle_extended
                and not ring_extended
                and not pinky_extended
            ):
                return "Peace"

            return "Hand gesture"

        except Exception:
            return "Unknown"


class EventEngine:
    def __init__(self):
        self.last_states = {}
        self.last_events = {}

    def emit(self, text, category):
        now = time.time()
        key = f"{category}:{text}"

        if (
            key in self.last_events
            and now - self.last_events[key] < 1.5
        ):
            return None

        self.last_events[key] = now

        return Event(
            timestamp=now,
            text=text,
            category=category
        )

    def update(self, snapshot):
        events = []

        current = {
            "people": len(snapshot.people),
            "scene": snapshot.scene
        }

        for person in snapshot.people:
            current[
                f"person_{person.person_id}_posture"
            ] = person.posture

            current[
                f"person_{person.person_id}_movement"
            ] = person.movement

            current[
                f"person_{person.person_id}_gestures"
            ] = tuple(sorted(person.gestures))

        for key, value in current.items():
            old = self.last_states.get(key)

            if old is None:
                self.last_states[key] = value
                continue

            if old == value:
                continue

            self.last_states[key] = value

            if key == "people":
                if value > old:
                    event = self.emit(
                        f"Person count increased from {old} to {value}.",
                        "People"
                    )
                    if event:
                        events.append(event)

                elif value < old:
                    event = self.emit(
                        f"Person count changed from {old} to {value}.",
                        "People"
                    )
                    if event:
                        events.append(event)

            elif key == "scene":
                event = self.emit(
                    f"Scene changed to {value}.",
                    "Scene"
                )
                if event:
                    events.append(event)

            elif key.endswith("_posture"):
                person_id = key.split("_")[1]

                event = self.emit(
                    f"Person {person_id} is now {value}.",
                    "Body"
                )

                if event:
                    events.append(event)

            elif key.endswith("_movement"):
                person_id = key.split("_")[1]

                if value != "Still":
                    event = self.emit(
                        f"Person {person_id} shows {value.lower()}.",
                        "Motion"
                    )

                    if event:
                        events.append(event)

            elif key.endswith("_gestures"):
                person_id = key.split("_")[1]

                if value:
                    event = self.emit(
                        f"Person {person_id}: {', '.join(value)}.",
                        "Gesture"
                    )

                    if event:
                        events.append(event)

        return events


class HandLandmarker:
    def __init__(self, confidence):
        self.confidence = confidence
        self.landmarker = None
        self.error = ""
        self.load()

    def download(self):
        os.makedirs(
            HAND_CACHE_DIR,
            exist_ok=True
        )

        if os.path.exists(HAND_MODEL_PATH):
            return True

        temporary_path = HAND_MODEL_PATH + ".tmp"

        try:
            urllib.request.urlretrieve(
                HAND_MODEL_URL,
                temporary_path
            )

            os.replace(
                temporary_path,
                HAND_MODEL_PATH
            )

            return True

        except Exception as exc:
            self.error = (
                f"Hand model download failed: {exc}"
            )
            return False

        finally:
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except Exception:
                    pass

    def load(self):
        if not MEDIAPIPE_AVAILABLE:
            self.error = (
                "MediaPipe is unavailable in this environment."
            )
            return

        if not self.download():
            return

        try:
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=HAND_MODEL_PATH
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=8,
                min_hand_detection_confidence=self.confidence,
                min_hand_presence_confidence=self.confidence,
                min_tracking_confidence=self.confidence
            )

            self.landmarker = (
                vision.HandLandmarker
                .create_from_options(options)
            )

        except Exception as exc:
            self.landmarker = None
            self.error = (
                "MediaPipe hand engine unavailable: "
                f"{exc}"
            )

    def detect(self, frame):
        if self.landmarker is None:
            return []

        try:
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = self.landmarker.detect(
                image
            )

            hands = []

            if not result.hand_landmarks:
                return hands

            for index, landmarks in enumerate(
                result.hand_landmarks
            ):
                points = [
                    (
                        float(p.x),
                        float(p.y),
                        float(p.z)
                    )
                    for p in landmarks
                ]

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                handedness = "Unknown"
                confidence = 0.0

                try:
                    category = (
                        result.handedness[index][0]
                    )

                    handedness = (
                        category.category_name
                    )

                    confidence = float(
                        category.score
                    )

                except Exception:
                    pass

                gesture = (
                    GestureRecognizer.classify(
                        points
                    )
                )

                hands.append(
                    HandState(
                        hand_id=index + 1,
                        handedness=handedness,
                        landmarks=points,
                        bbox=(
                            min(xs),
                            min(ys),
                            max(xs),
                            max(ys)
                        ),
                        gesture=gesture,
                        confidence=confidence
                    )
                )

            return hands

        except Exception as exc:
            self.error = (
                f"Hand detection failed: {exc}"
            )
            return []

    def close(self):
        try:
            if self.landmarker is not None:
                self.landmarker.close()
        except Exception:
            pass

        self.landmarker = None


class LifeVisionEngine:
    def __init__(self):
        self.lock = threading.RLock()

        self.object_model = None
        self.pose_model = None
        self.hand_model = None

        self.object_tracker = Tracker()
        self.motion = Motion()
        self.events = EventEngine()

        self.objects = []
        self.people = []
        self.hands = []

        self.frame_id = 0

        self.last_object_time = 0
        self.last_pose_time = 0
        self.last_hand_time = 0
        self.last_process_time = time.time()

        self.ai_fps = 0
        self.camera_fps = 0

        self.input_counter = 0
        self.input_time = time.time()

        self.events_log = deque(
            maxlen=100
        )

        self.runtime_error = ""

        self.hand_available = False
        self.hand_error = ""

        self.current_config = {
            "mode": "Live Scene",
            "object_confidence": 0.38,
            "pose_confidence": 0.38,
            "hand_confidence": 0.45,
            "process_fps": 6,
            "show_boxes": True,
            "show_labels": True,
            "show_body_nodes": True,
            "show_hand_nodes": True,
            "show_gesture_labels": True,
            "mirror": True,
            "groups": tuple(
                OBJECT_GROUPS.keys()
            )
        }

        self.snapshot = Snapshot(
            timestamp=time.time(),
            frame_id=0,
            objects=[],
            people=[],
            hands=[],
            scene="Waiting",
            narrative="Camera waiting for frames.",
            fps=0,
            ai_fps=0,
            hand_available=False,
            runtime_error=""
        )

        self.ensure_object_model()
        self.ensure_pose_model()
        self.ensure_hand_model()

    def ensure_object_model(self):
        if self.object_model is None:
            try:
                self.object_model = YOLO(
                    OBJECT_MODEL
                )
            except Exception as exc:
                self.runtime_error = (
                    f"Object model error: {exc}"
                )

    def ensure_pose_model(self):
        if self.pose_model is None:
            try:
                self.pose_model = YOLO(
                    POSE_MODEL
                )
            except Exception as exc:
                self.runtime_error = (
                    f"Pose model error: {exc}"
                )

    def ensure_hand_model(self):
        if self.hand_model is not None:
            return

        try:
            self.hand_model = HandLandmarker(
                self.current_config[
                    "hand_confidence"
                ]
            )

            self.hand_available = (
                self.hand_model.landmarker
                is not None
            )

            self.hand_error = (
                self.hand_model.error
            )

        except Exception as exc:
            self.hand_available = False
            self.hand_error = str(exc)

    def update_config(
        self,
        mode,
        object_confidence,
        pose_confidence,
        hand_confidence,
        process_fps,
        show_boxes,
        show_labels,
        show_body_nodes,
        show_hand_nodes,
        show_gesture_labels,
        mirror,
        groups
    ):
        with self.lock:
            self.current_config = {
                "mode": mode,
                "object_confidence": float(
                    object_confidence
                ),
                "pose_confidence": float(
                    pose_confidence
                ),
                "hand_confidence": float(
                    hand_confidence
                ),
                "process_fps": int(
                    process_fps
                ),
                "show_boxes": bool(
                    show_boxes
                ),
                "show_labels": bool(
                    show_labels
                ),
                "show_body_nodes": bool(
                    show_body_nodes
                ),
                "show_hand_nodes": bool(
                    show_hand_nodes
                ),
                "show_gesture_labels": bool(
                    show_gesture_labels
                ),
                "mirror": bool(
                    mirror
                ),
                "groups": tuple(
                    groups or ["People"]
                )
            }

    def allowed_labels(self):
        labels = set()

        for group in self.current_config[
            "groups"
        ]:
            labels.update(
                OBJECT_GROUPS.get(
                    group,
                    set()
                )
            )

        return labels

    def detect_objects(
        self,
        frame,
        people_only=False
    ):
        if self.object_model is None:
            return []

        try:
            kwargs = {
                "conf": self.current_config[
                    "object_confidence"
                ],
                "verbose": False,
                "device": "cpu",
                "imgsz": 512,
                "max_det": 30
            }

            if people_only:
                kwargs["classes"] = [0]

            results = self.object_model.predict(
                frame,
                **kwargs
            )

            if not results:
                return []

            result = results[0]

            if result.boxes is None:
                return []

            detections = []

            for box in result.boxes:
                confidence = float(
                    box.conf[0].item()
                )

                cls = int(
                    box.cls[0].item()
                )

                label = str(
                    result.names.get(
                        cls,
                        cls
                    )
                )

                if people_only:
                    if label != "person":
                        continue
                else:
                    if (
                        label != "person"
                        and label not in
                        self.allowed_labels()
                    ):
                        continue

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = [
                    int(v)
                    for v in xyxy
                ]

                center = (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2)
                )

                area = max(
                    1,
                    (x2 - x1) *
                    (y2 - y1)
                )

                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "bbox": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "center": center,
                        "area": area
                    }
                )

            detections = (
                self.object_tracker.assign(
                    detections
                )
            )

            return [
                ObjectState(
                    label=d["label"],
                    confidence=d["confidence"],
                    bbox=d["bbox"],
                    track_id=d["track_id"],
                    center=d["center"],
                    area=d["area"]
                )
                for d in detections
            ]

        except Exception as exc:
            self.runtime_error = (
                f"Object detection: {exc}"
            )
            return self.objects

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(
            0,
            ix2 - ix1
        )

        ih = max(
            0,
            iy2 - iy1
        )

        intersection = iw * ih

        area_a = max(
            1,
            (ax2 - ax1) *
            (ay2 - ay1)
        )

        area_b = max(
            1,
            (bx2 - bx1) *
            (by2 - by1)
        )

        return intersection / (
            area_a +
            area_b -
            intersection +
            1e-6
        )

    def sync_people(self):
        confirmed = [
            obj
            for obj in self.objects
            if obj.label == "person"
        ]

        old = {
            p.person_id: p
            for p in self.people
        }

        new_people = []

        for obj in confirmed:
            previous = old.get(
                obj.track_id
            )

            person = PersonState(
                person_id=obj.track_id,
                bbox=obj.bbox,
                center=obj.center,
                confidence=obj.confidence
            )

            if previous is not None:
                person.keypoints = (
                    previous.keypoints
                )
                person.posture = (
                    previous.posture
                )
                person.movement = (
                    previous.movement
                )
                person.velocity = (
                    previous.velocity
                )
                person.gestures = (
                    previous.gestures
                )
                person.hands = (
                    previous.hands
                )
                person.last_center = (
                    previous.center
                )
                person.history = (
                    previous.history
                )

            new_people.append(
                person
            )

        self.people = new_people

        for person in self.people:
            self.motion.update(
                person
            )

    def detect_pose(self, frame):
        if (
            self.pose_model is None
            or not self.people
        ):
            return

        for person in self.people:
            person.keypoints = []
            person.posture = "Unknown"

        try:
            results = self.pose_model.predict(
                frame,
                conf=self.current_config[
                    "pose_confidence"
                ],
                verbose=False,
                device="cpu",
                imgsz=512,
                max_det=12
            )

            if not results:
                return

            result = results[0]

            if (
                result.boxes is None
                or result.keypoints is None
            ):
                return

            candidates = []

            for index, box in enumerate(
                result.boxes
            ):
                bbox = tuple(
                    int(v)
                    for v in (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )
                )

                try:
                    points = (
                        result
                        .keypoints
                        .data[index]
                        .cpu()
                        .numpy()
                    )

                    points = [
                        (
                            float(p[0]),
                            float(p[1]),
                            float(p[2])
                        )
                        for p in points
                    ]

                except Exception:
                    continue

                candidates.append(
                    (
                        bbox,
                        points
                    )
                )

            used = set()

            for pose_bbox, points in candidates:
                best_person = None
                best_score = 0.10

                for person in self.people:
                    if (
                        person.person_id
                        in used
                    ):
                        continue

                    score = self.iou(
                        person.bbox,
                        pose_bbox
                    )

                    if score > best_score:
                        best_score = score
                        best_person = person

                if best_person is None:
                    continue

                used.add(
                    best_person.person_id
                )

                best_person.keypoints = (
                    points
                )

                best_person.posture = (
                    Kinematics
                    .posture_from_keypoints(
                        points
                    )
                )

        except Exception as exc:
            self.runtime_error = (
                f"Pose detection: {exc}"
            )

    def detect_hands(self, frame):
        if not self.hand_available:
            self.hands = []
            return

        try:
            self.hands = (
                self.hand_model.detect(
                    frame
                )
            )

        except Exception as exc:
            self.hands = []
            self.hand_error = str(exc)

    def attach_hands(
        self,
        width,
        height
    ):
        for person in self.people:
            person.hands = []
            person.gestures = []

        for hand in self.hands:
            x1, y1, x2, y2 = hand.bbox

            hx1 = int(x1 * width)
            hy1 = int(y1 * height)
            hx2 = int(x2 * width)
            hy2 = int(y2 * height)

            hcx = (
                hx1 + hx2
            ) / 2

            hcy = (
                hy1 + hy2
            ) / 2

            best_person = None
            best_distance = float("inf")

            for person in self.people:
                px1, py1, px2, py2 = (
                    person.bbox
                )

                expanded = (
                    px1 - 100,
                    py1 - 100,
                    px2 + 100,
                    py2 + 100
                )

                if (
                    expanded[0] <= hcx <= expanded[2]
                    and
                    expanded[1] <= hcy <= expanded[3]
                ):
                    distance = math.hypot(
                        hcx - person.center[0],
                        hcy - person.center[1]
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_person = person

            if best_person is not None:
                best_person.hands.append(
                    hand
                )

                if (
                    hand.gesture != "Unknown"
                    and
                    hand.gesture
                    not in best_person.gestures
                ):
                    best_person.gestures.append(
                        hand.gesture
                    )

    def build_scene(self):
        people_count = len(
            self.people
        )

        animals = [
            o
            for o in self.objects
            if o.label in
            OBJECT_GROUPS["Animals"]
        ]

        things = [
            o
            for o in self.objects
            if (
                o.label != "person"
                and
                o.label not in
                OBJECT_GROUPS["Animals"]
            )
        ]

        gestures = any(
            person.gestures
            for person in self.people
        )

        if people_count == 0:
            if animals and things:
                return "Animals + Objects"

            if animals:
                return "Animals"

            if things:
                return "Objects"

            return "No human detected"

        base = (
            "Multiple Humans"
            if people_count > 1
            else "Human"
        )

        parts = [base]

        if animals:
            parts.append(
                "Animals"
            )

        if things:
            parts.append(
                "Objects"
            )

        if gestures:
            parts.append(
                "Gestures"
            )

        return " + ".join(parts)

    def build_narrative(self, scene):
        people_count = len(
            self.people
        )

        if people_count == 0:
            labels = sorted(
                set(
                    o.label
                    for o in self.objects
                )
            )

            if labels:
                return (
                    "No human is currently "
                    "confirmed. Visible objects "
                    "include "
                    + ", ".join(labels[:8])
                    + "."
                )

            return (
                "No human is currently "
                "confirmed by the object detector."
            )

        descriptions = []

        for person in self.people:
            text = (
                f"Person {person.person_id} "
                f"is {person.posture.lower()}"
            )

            if person.movement != "Still":
                text += (
                    f" and "
                    f"{person.movement.lower()}"
                )

            if person.gestures:
                text += (
                    ", showing "
                    + ", ".join(
                        person.gestures
                    )
                )

            descriptions.append(
                text
            )

        other_labels = sorted(
            set(
                o.label
                for o in self.objects
                if o.label != "person"
            )
        )

        result = "; ".join(
            descriptions
        )

        if other_labels:
            result += (
                ". Detected nearby items: "
                + ", ".join(
                    other_labels[:8]
                )
                + "."
            )
        else:
            result += "."

        return result

    def draw_object(
        self,
        frame,
        obj
    ):
        x1, y1, x2, y2 = obj.bbox

        if self.current_config[
            "show_boxes"
        ]:
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 220, 80),
                2
            )

        if self.current_config[
            "show_labels"
        ]:
            if obj.label == "person":
                text = (
                    f"Person {obj.track_id} "
                    f"{obj.confidence:.0%}"
                )
            else:
                text = (
                    f"{obj.label} "
                    f"{obj.confidence:.0%}"
                )

            cv2.putText(
                frame,
                text,
                (
                    x1,
                    max(22, y1 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 220, 80),
                2,
                cv2.LINE_AA
            )

    def draw_person(
        self,
        frame,
        person
    ):
        x1, y1, x2, y2 = (
            person.bbox
        )

        if self.current_config[
            "show_boxes"
        ]:
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 180, 0),
                2
            )

        label = (
            f"Person {person.person_id}"
            f" | {person.posture}"
        )

        if person.movement != "Still":
            label += (
                f" | {person.movement}"
            )

        if (
            self.current_config[
                "show_gesture_labels"
            ]
            and person.gestures
        ):
            label += (
                " | "
                + ", ".join(
                    person.gestures
                )
            )

        if self.current_config[
            "show_labels"
        ]:
            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(22, y1 - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 180, 0),
                2,
                cv2.LINE_AA
            )

    def draw_pose(
        self,
        frame,
        person
    ):
        if (
            not self.current_config[
                "show_body_nodes"
            ]
            or not person.keypoints
        ):
            return

        height, width = (
            frame.shape[:2]
        )

        points = []

        for x, y, confidence in (
            person.keypoints
        ):
            if confidence < 0.25:
                points.append(None)
                continue

            px = max(
                0,
                min(width - 1, int(x))
            )

            py = max(
                0,
                min(height - 1, int(y))
            )

            points.append(
                (px, py)
            )

            cv2.circle(
                frame,
                (px, py),
                4,
                (255, 80, 180),
                -1
            )

        for a, b in POSE_CONNECTIONS:
            if (
                a < len(points)
                and b < len(points)
                and points[a] is not None
                and points[b] is not None
            ):
                cv2.line(
                    frame,
                    points[a],
                    points[b],
                    (255, 120, 180),
                    2,
                    cv2.LINE_AA
                )

    def draw_hands(self, frame):
        if not self.hand_available:
            return

        height, width = (
            frame.shape[:2]
        )

        for hand in self.hands:
            points = []

            for x, y, z in (
                hand.landmarks
            ):
                points.append(
                    (
                        int(x * width),
                        int(y * height)
                    )
                )

            if self.current_config[
                "show_hand_nodes"
            ]:
                for a, b in HAND_CONNECTIONS:
                    if (
                        a < len(points)
                        and b < len(points)
                    ):
                        cv2.line(
                            frame,
                            points[a],
                            points[b],
                            (80, 200, 255),
                            2,
                            cv2.LINE_AA
                        )

                for point in points:
                    cv2.circle(
                        frame,
                        point,
                        3,
                        (80, 200, 255),
                        -1
                    )

            x1 = int(
                hand.bbox[0] * width
            )
            y1 = int(
                hand.bbox[1] * height
            )
            x2 = int(
                hand.bbox[2] * width
            )
            y2 = int(
                hand.bbox[3] * height
            )

            if self.current_config[
                "show_boxes"
            ]:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (80, 200, 255),
                    1
                )

            if self.current_config[
                "show_gesture_labels"
            ]:
                text = (
                    f"{hand.handedness}: "
                    f"{hand.gesture}"
                )

                cv2.putText(
                    frame,
                    text,
                    (
                        x1,
                        max(20, y1 - 6)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (80, 200, 255),
                    2,
                    cv2.LINE_AA
                )

    def draw_hud(
        self,
        frame,
        snapshot
    ):
        height, width = (
            frame.shape[:2]
        )

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (width, 72),
            (0, 0, 0),
            -1
        )

        frame[:] = cv2.addWeighted(
            overlay,
            0.58,
            frame,
            0.42,
            0
        )

        cv2.putText(
            frame,
            f"LifeVision {APP_VERSION}",
            (14, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                f"MODE: "
                f"{self.current_config['mode']}"
            ),
            (14, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                f"PEOPLE: {len(snapshot.people)} "
                f"OBJECTS: {len(snapshot.objects)} "
                f"HANDS: {len(snapshot.hands)}"
            ),
            (250, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                f"AI FPS: {snapshot.ai_fps:.1f} "
                f"CAMERA FPS: {snapshot.fps:.1f}"
            ),
            (14, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    def process(
        self,
        frame
    ):
        if frame is None:
            return None, self.info()

        start = time.time()

        with self.lock:
            self.frame_id += 1
            frame_id = self.frame_id

            now = time.time()

            self.input_counter += 1

            if now - self.input_time >= 1:
                self.camera_fps = (
                    self.input_counter /
                    max(
                        0.001,
                        now - self.input_time
                    )
                )

                self.input_counter = 0
                self.input_time = now

            process_interval = max(
                0.10,
                1.0 /
                max(
                    1,
                    self.current_config[
                        "process_fps"
                    ]
                )
            )

            mode = self.current_config[
                "mode"
            ]

            should_process = (
                now -
                self.last_process_time
                >= process_interval
            )

            if should_process:
                self.last_process_time = now

                if mode == (
                    "Gesture & Body Awareness"
                ):
                    self.objects = (
                        self.detect_objects(
                            frame,
                            people_only=True
                        )
                    )
                else:
                    self.objects = (
                        self.detect_objects(
                            frame
                        )
                    )

                self.sync_people()

                if mode in {
                    "Gesture & Body Awareness",
                    "Live Scene"
                }:
                    if (
                        now -
                        self.last_pose_time
                        >= process_interval * 1.25
                    ):
                        self.detect_pose(
                            frame
                        )

                        self.last_pose_time = now

                    if (
                        now -
                        self.last_hand_time
                        >= process_interval * 1.10
                    ):
                        self.detect_hands(
                            frame
                        )

                        self.attach_hands(
                            frame.shape[1],
                            frame.shape[0]
                        )

                        self.last_hand_time = now
                else:
                    self.hands = []

                scene = self.build_scene()

                narrative = (
                    self.build_narrative(
                        scene
                    )
                )

                current = time.time()

                ai_delta = max(
                    0.001,
                    current - start
                )

                instant_ai_fps = (
                    1.0 /
                    ai_delta
                )

                self.ai_fps = (
                    self.ai_fps * 0.8
                    +
                    instant_ai_fps * 0.2
                )

                snapshot = Snapshot(
                    timestamp=current,
                    frame_id=frame_id,
                    objects=list(
                        self.objects
                    ),
                    people=list(
                        self.people
                    ),
                    hands=list(
                        self.hands
                    ),
                    scene=scene,
                    narrative=narrative,
                    fps=self.camera_fps,
                    ai_fps=self.ai_fps,
                    hand_available=(
                        self.hand_available
                    ),
                    runtime_error=(
                        self.runtime_error
                        or self.hand_error
                    )
                )

                new_events = (
                    self.events.update(
                        snapshot
                    )
                )

                for event in new_events:
                    self.events_log.appendleft(
                        event
                    )

                self.snapshot = snapshot

            snapshot = self.snapshot

            output = frame.copy()

            if self.current_config[
                "mirror"
            ]:
                output = cv2.flip(
                    output,
                    1
                )

            if mode in {
                "Object & People Awareness",
                "Live Scene"
            }:
                for obj in snapshot.objects:
                    self.draw_object(
                        output,
                        obj
                    )

            if mode in {
                "Gesture & Body Awareness",
                "Live Scene"
            }:
                for person in snapshot.people:
                    self.draw_person(
                        output,
                        person
                    )

                    self.draw_pose(
                        output,
                        person
                    )

                self.draw_hands(
                    output
                )

            self.draw_hud(
                output,
                snapshot
            )

            return (
                cv2.cvtColor(
                    output,
                    cv2.COLOR_BGR2RGB
                ),
                self.info()
            )

    def info(self):
        with self.lock:
            snapshot = self.snapshot

            people_lines = []

            for person in snapshot.people:
                gestures = (
                    ", ".join(
                        person.gestures
                    )
                    if person.gestures
                    else "None"
                )

                people_lines.append(
                    f"Person {person.person_id} | "
                    f"{person.posture} | "
                    f"{person.movement} | "
                    f"Gesture: {gestures}"
                )

            counts = {}

            for obj in snapshot.objects:
                counts[obj.label] = (
                    counts.get(
                        obj.label,
                        0
                    ) + 1
                )

            object_lines = [
                f"{label}: {count}"
                for label, count
                in sorted(
                    counts.items()
                )
            ]

            event_lines = []

            for event in list(
                self.events_log
            )[:12]:
                timestamp = time.strftime(
                    "%H:%M:%S",
                    time.localtime(
                        event.timestamp
                    )
                )

                event_lines.append(
                    f"{timestamp} · "
                    f"{event.category} · "
                    f"{event.text}"
                )

            hand_status = (
                "Available"
                if self.hand_available
                else "Unavailable"
            )

            runtime = (
                self.runtime_error
                or self.hand_error
                or "None"
            )

            return {
                "people": len(
                    snapshot.people
                ),
                "objects": len(
                    snapshot.objects
                ),
                "hands": len(
                    snapshot.hands
                ),
                "ai_fps": (
                    f"{snapshot.ai_fps:.1f}"
                ),
                "camera_fps": (
                    f"{snapshot.fps:.1f}"
                ),
                "scene": snapshot.scene,
                "narrative": snapshot.narrative,
                "people_text": (
                    "\n".join(
                        people_lines
                    )
                    if people_lines
                    else
                    "No confirmed people."
                ),
                "objects_text": (
                    "\n".join(
                        object_lines
                    )
                    if object_lines
                    else
                    "No selected objects."
                ),
                "events_text": (
                    "\n".join(
                        event_lines
                    )
                    if event_lines
                    else
                    "No events recorded yet."
                ),
                "hand_status": hand_status,
                "runtime": runtime
            }


ENGINE = LifeVisionEngine()


def process_frame(
    frame,
    mode,
    object_confidence,
    pose_confidence,
    hand_confidence,
    process_fps,
    show_boxes,
    show_labels,
    show_body_nodes,
    show_hand_nodes,
    show_gesture_labels,
    mirror,
    groups
):
    ENGINE.update_config(
        mode,
        object_confidence,
        pose_confidence,
        hand_confidence,
        process_fps,
        show_boxes,
        show_labels,
        show_body_nodes,
        show_hand_nodes,
        show_gesture_labels,
        mirror,
        groups
    )

    output, info = ENGINE.process(
        frame
    )

    return (
        output,
        info["people"],
        info["objects"],
        info["hands"],
        info["ai_fps"],
        info["camera_fps"],
        info["scene"],
        info["narrative"],
        info["people_text"],
        info["objects_text"],
        info["events_text"],
        info["hand_status"],
        info["runtime"]
    )


def clear_events():
    with ENGINE.lock:
        ENGINE.events_log.clear()

    return "Events cleared."


CSS = """
#camera_output {
    min-height: 500px;
}

#camera_output img {
    object-fit: contain !important;
}

.status-card {
    border-radius: 12px;
    padding: 12px;
}

.small-note {
    opacity: 0.75;
}
"""


with gr.Blocks(
    title="LifeVision",
    css=CSS,
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
# 👁️ LifeVision

**Real-Time Computer Vision · People · Objects · Body · Hands · Gestures · Scene Understanding**

LifeVision combines object detection, person tracking, body pose analysis,
hand landmarks, gesture recognition and scene interpretation.
"""
    )

    with gr.Row():

        with gr.Column(
            scale=7
        ):
            camera = gr.Image(
                sources=["webcam"],
                type="numpy",
                streaming=True,
                label="LifeVision Camera",
                elem_id="camera_output"
            )

        with gr.Column(
            scale=3
        ):

            gr.Markdown(
                "### Live Status"
            )

            with gr.Row():
                people_metric = gr.Number(
                    label="People",
                    value=0,
                    precision=0
                )

                object_metric = gr.Number(
                    label="Objects",
                    value=0,
                    precision=0
                )

            with gr.Row():
                hand_metric = gr.Number(
                    label="Hands",
                    value=0,
                    precision=0
                )

                ai_fps_metric = gr.Number(
                    label="AI FPS",
                    value=0,
                    precision=1
                )

            camera_fps_metric = gr.Number(
                label="Camera FPS",
                value=0,
                precision=1
            )

            scene_text = gr.Textbox(
                label="Scene",
                value="Waiting",
                interactive=False
            )

            narrative_text = gr.Textbox(
                label="Live Interpretation",
                value="Camera waiting for frames.",
                lines=4,
                interactive=False
            )

    with gr.Accordion(
        "LifeVision Controls",
        open=True
    ):

        with gr.Row():

            with gr.Column():

                mode = gr.Radio(
                    [
                        "Object & People Awareness",
                        "Gesture & Body Awareness",
                        "Live Scene"
                    ],
                    value="Live Scene",
                    label="Vision Mode"
                )

                process_fps = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=6,
                    step=1,
                    label="AI Processing FPS"
                )

            with gr.Column():

                object_confidence = gr.Slider(
                    minimum=0.20,
                    maximum=0.80,
                    value=0.38,
                    step=0.01,
                    label="Object Confidence"
                )

                pose_confidence = gr.Slider(
                    minimum=0.20,
                    maximum=0.80,
                    value=0.38,
                    step=0.01,
                    label="Body Confidence"
                )

                hand_confidence = gr.Slider(
                    minimum=0.20,
                    maximum=0.80,
                    value=0.45,
                    step=0.01,
                    label="Hand Confidence"
                )

        with gr.Row():

            with gr.Column():

                show_boxes = gr.Checkbox(
                    value=True,
                    label="Detection Boxes"
                )

                show_labels = gr.Checkbox(
                    value=True,
                    label="Detection Labels"
                )

                show_body_nodes = gr.Checkbox(
                    value=True,
                    label="Body Skeleton"
                )

            with gr.Column():

                show_hand_nodes = gr.Checkbox(
                    value=True,
                    label="Hand Skeleton"
                )

                show_gesture_labels = gr.Checkbox(
                    value=True,
                    label="Gesture Labels"
                )

                mirror = gr.Checkbox(
                    value=True,
                    label="Mirror Camera"
                )

        groups = gr.CheckboxGroup(
            choices=list(
                OBJECT_GROUPS.keys()
            ),
            value=list(
                OBJECT_GROUPS.keys()
            ),
            label="Object Groups"
        )

    with gr.Row():

        with gr.Column():

            gr.Markdown(
                "### People"
            )

            people_text = gr.Textbox(
                value="No confirmed people.",
                lines=8,
                interactive=False,
                show_label=False
            )

        with gr.Column():

            gr.Markdown(
                "### Objects"
            )

            objects_text = gr.Textbox(
                value="No selected objects.",
                lines=8,
                interactive=False,
                show_label=False
            )

        with gr.Column():

            gr.Markdown(
                "### Events"
            )

            events_text = gr.Textbox(
                value="No events recorded yet.",
                lines=8,
                interactive=False,
                show_label=False
            )

    with gr.Row():

        hand_status = gr.Textbox(
            label="Hand Engine",
            value="Starting...",
            interactive=False
        )

        runtime_status = gr.Textbox(
            label="Runtime Diagnostic",
            value="None",
            interactive=False
        )

    clear_button = gr.Button(
        "Clear Event Log"
    )

    clear_result = gr.Textbox(
        value="",
        show_label=False,
        interactive=False
    )

    gr.Markdown(
        f"""
---

**LifeVision {APP_VERSION}**

Local AI models · No paid API required · YOLO object/pose analysis · Optional MediaPipe hand analysis
"""
    )

    stream_outputs = [
        camera,
        people_metric,
        object_metric,
        hand_metric,
        ai_fps_metric,
        camera_fps_metric,
        scene_text,
        narrative_text,
        people_text,
        objects_text,
        events_text,
        hand_status,
        runtime_status
    ]

    stream_inputs = [
        camera,
        mode,
        object_confidence,
        pose_confidence,
        hand_confidence,
        process_fps,
        show_boxes,
        show_labels,
        show_body_nodes,
        show_hand_nodes,
        show_gesture_labels,
        mirror,
        groups
    ]

    camera.stream(
        fn=process_frame,
        inputs=stream_inputs,
        outputs=stream_outputs,
        stream_every=0.15,
        time_limit=30,
        concurrency_limit=1
    )

    clear_button.click(
        fn=clear_events,
        inputs=[],
        outputs=[clear_result]
    )


if __name__ == "__main__":
    demo.launch()