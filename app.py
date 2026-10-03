import os
import time
import math
import threading
import urllib.request
from dataclasses import dataclass, field
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


st.set_page_config(
    page_title="LifeVision",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


APP_VERSION = "11.0"

OBJECT_MODEL = "yolo11n.pt"
POSE_MODEL = "yolo11n-pose.pt"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)


OBJECT_GROUPS = {
    "People": {
        "person"
    },
    "Animals": {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe"
    },
    "Vehicles": {
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat"
    },
    "Indoor": {
        "chair",
        "couch",
        "bed",
        "dining table",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear"
    },
    "Food": {
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake"
    },
    "Sports": {
        "sports ball",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket"
    }
}


POSE_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16)
]


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17)
]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def center(box):
    return (
        (box[0] + box[2]) / 2,
        (box[1] + box[3]) / 2
    )


def distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def group_for(label):
    for group, labels in OBJECT_GROUPS.items():
        if label in labels:
            return group
    return "Other"


def download_hand_model():
    if os.path.exists(HAND_MODEL_PATH):
        return HAND_MODEL_PATH

    try:
        urllib.request.urlretrieve(
            HAND_MODEL_URL,
            HAND_MODEL_PATH
        )
        return HAND_MODEL_PATH
    except Exception:
        return None


@dataclass
class ObjectState:
    label: str
    confidence: float
    box: tuple
    track_id: int = -1
    group: str = "Other"


@dataclass
class PersonState:
    person_id: int
    confidence: float
    box: tuple
    keypoints: list = field(default_factory=list)
    posture: str = "Unknown"
    movement: str = "Still"
    velocity: float = 0.0
    gestures: list = field(default_factory=list)
    hands: list = field(default_factory=list)


@dataclass
class HandState:
    handedness: str
    confidence: float
    box: tuple
    gesture: str
    landmarks: list = field(default_factory=list)
    person_id: int = -1


@dataclass
class Event:
    timestamp: float
    category: str
    message: str


@dataclass
class Snapshot:
    timestamp: float = 0
    scene: str = "Waiting"
    narrative: str = ""
    camera_fps: float = 0
    ai_fps: float = 0
    latency: float = 0
    objects: list = field(default_factory=list)
    people: list = field(default_factory=list)
    hands: list = field(default_factory=list)
    interactions: list = field(default_factory=list)


@dataclass
class Config:
    mode: str = "Live Scene"
    performance: str = "Fast"
    object_confidence: float = 0.35
    pose_confidence: float = 0.35
    hand_confidence: float = 0.35
    inference_size: int = 320
    mirror: bool = True
    show_hud: bool = True
    show_fps: bool = True
    show_person_boxes: bool = True
    show_object_boxes: bool = True
    show_body_nodes: bool = True
    show_hand_nodes: bool = True
    show_gesture_labels: bool = True
    groups: list = field(
        default_factory=lambda: list(
            OBJECT_GROUPS.keys()
        )
    )


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.snapshot = Snapshot()
        self.events = deque(maxlen=100)

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None, self.frame_id
            return self.frame, self.frame_id

    def set_snapshot(self, snapshot):
        with self.lock:
            self.snapshot = snapshot

    def get_snapshot(self):
        with self.lock:
            return self.snapshot

    def add_event(self, event):
        with self.lock:
            self.events.appendleft(event)

    def get_events(self):
        with self.lock:
            return list(self.events)


class EMA:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.value = None

    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = (
                self.alpha * value
                + (1 - self.alpha) * self.value
            )
        return self.value


class Tracker:
    def __init__(self, threshold=120):
        self.next_id = 1
        self.previous = {}
        self.threshold = threshold

    def update(self, items):
        used = set()
        current = {}

        for item in items:
            best_id = None
            best_distance = float("inf")

            for old_id, old_box in self.previous.items():
                if old_id in used:
                    continue

                d = distance(
                    center(item.box),
                    center(old_box)
                )

                if (
                    d < best_distance
                    and d < self.threshold
                ):
                    best_distance = d
                    best_id = old_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            item.track_id = best_id
            used.add(best_id)
            current[best_id] = item.box

        self.previous = current
        return items


class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.previous = {}
        self.threshold = 160

    def update(self, people):
        used = set()
        current = {}

        for person in people:
            best_id = None
            best_distance = float("inf")

            for old_id, old_box in self.previous.items():
                if old_id in used:
                    continue

                d = distance(
                    center(person.box),
                    center(old_box)
                )

                if (
                    d < best_distance
                    and d < self.threshold
                ):
                    best_distance = d
                    best_id = old_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            person.person_id = best_id
            used.add(best_id)
            current[best_id] = person.box

        self.previous = current

        return people


class Motion:
    def __init__(self):
        self.previous = {}
        self.filters = {}

    def update(self, person_id, box):
        current = center(box)

        if person_id not in self.previous:
            self.previous[person_id] = current
            self.filters[person_id] = EMA(0.3)
            return 0.0, "Still"

        movement = distance(
            current,
            self.previous[person_id]
        )

        self.previous[person_id] = current

        speed = self.filters[
            person_id
        ].update(movement)

        if speed < 2:
            state = "Still"
        elif speed < 10:
            state = "Moving"
        else:
            state = "Moving Quickly"

        return speed, state


class Kinematics:
    @staticmethod
    def angle(a, b, c):
        if not a or not b or not c:
            return 0

        ba = np.array([
            a[0] - b[0],
            a[1] - b[1]
        ])

        bc = np.array([
            c[0] - b[0],
            c[1] - b[1]
        ])

        denominator = (
            np.linalg.norm(ba)
            * np.linalg.norm(bc)
        )

        if denominator == 0:
            return 0

        value = (
            np.dot(ba, bc)
            / denominator
        )

        value = np.clip(
            value,
            -1,
            1
        )

        return float(
            np.degrees(
                np.arccos(value)
            )
        )

    @staticmethod
    def posture(points):
        if len(points) < 17:
            return "Unknown"

        needed = [
            points[5],
            points[6],
            points[11],
            points[12],
            points[13],
            points[14],
            points[15],
            points[16]
        ]

        if any(
            point is None
            for point in needed
        ):
            return "Unknown"

        shoulder_y = (
            points[5][1]
            + points[6][1]
        ) / 2

        hip_y = (
            points[11][1]
            + points[12][1]
        ) / 2

        knee_y = (
            points[13][1]
            + points[14][1]
        ) / 2

        ankle_y = (
            points[15][1]
            + points[16][1]
        ) / 2

        torso = abs(
            hip_y - shoulder_y
        )

        if torso < 45:
            return "Crouching"

        if (
            knee_y > hip_y
            and abs(ankle_y - knee_y) < 50
        ):
            return "Sitting"

        if (
            ankle_y > knee_y
            and knee_y > hip_y
        ):
            return "Standing"

        if (
            torso > 100
            and ankle_y < knee_y
        ):
            return "Reclining"

        return "Standing"


class GestureRecognizer:
    def extended(
        self,
        points,
        tip,
        pip,
        mcp
    ):
        if (
            points[tip] is None
            or points[pip] is None
            or points[mcp] is None
        ):
            return False

        return (
            distance(
                points[tip],
                points[mcp]
            )
            >
            distance(
                points[pip],
                points[mcp]
            ) * 1.05
        )

    def thumb(self, points):
        if (
            points[4] is None
            or points[3] is None
            or points[2] is None
        ):
            return False

        return (
            distance(
                points[4],
                points[2]
            )
            >
            distance(
                points[3],
                points[2]
            ) * 1.08
        )

    def recognize(self, points):
        if len(points) < 21:
            return "Unknown"

        thumb = self.thumb(points)

        index = self.extended(
            points, 8, 6, 5
        )

        middle = self.extended(
            points, 12, 10, 9
        )

        ring = self.extended(
            points, 16, 14, 13
        )

        pinky = self.extended(
            points, 20, 18, 17
        )

        total = sum([
            index,
            middle,
            ring,
            pinky
        ])

        if (
            middle
            and not index
            and not ring
            and not pinky
        ):
            return "Middle Finger"

        if (
            thumb
            and not index
            and not middle
            and not ring
            and not pinky
        ):
            if (
                points[4][1]
                < points[3][1]
            ):
                return "Thumbs Up"

            return "Thumbs Down"

        if (
            index
            and middle
            and not ring
            and not pinky
        ):
            return "Victory / Peace"

        if (
            index
            and not middle
            and not ring
            and not pinky
        ):
            return "Pointing"

        if (
            index
            and middle
            and ring
            and pinky
        ):
            return "Open Palm"

        if total == 3:
            return "Three Fingers"

        if total == 2:
            return "Two Fingers"

        if total == 4:
            return "Four Fingers"

        if total == 0:
            return "Fist"

        return "Unknown"


class EventEngine:
    def __init__(self, shared):
        self.shared = shared
        self.states = {}
        self.last_events = {}

    def update(
        self,
        key,
        value,
        category,
        message
    ):
        if self.states.get(key) == value:
            return

        self.states[key] = value

        now = time.time()

        if (
            now
            - self.last_events.get(key, 0)
            < 1
        ):
            return

        self.last_events[key] = now

        self.shared.add_event(
            Event(
                timestamp=now,
                category=category,
                message=message
            )
        )


class HandLandmarker:
    def __init__(self, confidence):
        self.landmarker = None
        self.available = False

        path = download_hand_model()

        if not path:
            return

        try:
            base = python.BaseOptions(
                model_asset_path=path
            )

            options = (
                vision.HandLandmarkerOptions(
                    base_options=base,
                    running_mode=(
                        vision.RunningMode.IMAGE
                    ),
                    num_hands=4,
                    min_hand_detection_confidence=(
                        confidence
                    ),
                    min_hand_presence_confidence=(
                        confidence
                    ),
                    min_tracking_confidence=(
                        confidence
                    )
                )
            )

            self.landmarker = (
                vision.HandLandmarker
                .create_from_options(
                    options
                )
            )

            self.available = True

        except Exception:
            self.available = False

    def detect(self, frame):
        if not self.available:
            return []

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb
        )

        try:
            result = (
                self.landmarker.detect(
                    image
                )
            )
        except Exception:
            return []

        output = []

        if not result.hand_landmarks:
            return output

        height, width = frame.shape[:2]

        for index, landmarks in enumerate(
            result.hand_landmarks
        ):
            points = []

            xs = []
            ys = []

            for landmark in landmarks:
                x = landmark.x * width
                y = landmark.y * height

                points.append(
                    (x, y)
                )

                xs.append(x)
                ys.append(y)

            if not points:
                continue

            handedness = "Unknown"
            confidence = 0

            if (
                result.handedness
                and index < len(
                    result.handedness
                )
            ):
                category = (
                    result.handedness[index][0]
                )

                handedness = (
                    category.category_name
                    or "Unknown"
                )

                confidence = float(
                    category.score or 0
                )

            box = (
                max(
                    0,
                    int(min(xs) - 10)
                ),
                max(
                    0,
                    int(min(ys) - 10)
                ),
                min(
                    width - 1,
                    int(max(xs) + 10)
                ),
                min(
                    height - 1,
                    int(max(ys) + 10)
                )
            )

            output.append(
                {
                    "handedness":
                        handedness,
                    "confidence":
                        confidence,
                    "box":
                        box,
                    "landmarks":
                        points
                }
            )

        return output


class LifeVisionProcessor:
    def __init__(
        self,
        shared,
        config
    ):
        self.shared = shared
        self.config = config

        self.object_model = None
        self.pose_model = None
        self.hand_model = None

        self.load_models()

        self.object_tracker = Tracker()
        self.person_tracker = PersonTracker()
        self.motion = Motion()
        self.kinematics = Kinematics()
        self.gesture = GestureRecognizer()
        self.events = EventEngine(shared)

        self.stop_event = threading.Event()

        self.latest_output = None
        self.latest_frame_id = -1

        self.objects = []
        self.people = []
        self.hands = []

        self.ai_fps = EMA(0.3)
        self.camera_fps = EMA(0.3)

        self.last_camera_time = (
            time.perf_counter()
        )

        self.last_object_run = 0
        self.last_pose_run = 0
        self.last_hand_run = 0

        self.object_interval = 0.15
        self.pose_interval = 0.20
        self.hand_interval = 0.15

        self.worker = threading.Thread(
            target=self.worker_loop,
            daemon=True
        )

        self.worker.start()

    def load_models(self):
        mode = self.config.mode

        if mode in [
            "Object & People Awareness",
            "Live Scene"
        ]:
            self.object_model = YOLO(
                OBJECT_MODEL
            )

        if mode in [
            "Gesture & Body Awareness",
            "Live Scene"
        ]:
            self.pose_model = YOLO(
                POSE_MODEL
            )

            self.hand_model = (
                HandLandmarker(
                    self.config.hand_confidence
                )
            )

    def inference_size(self):
        if self.config.performance == "Fast":
            return 320

        if self.config.performance == "Balanced":
            return 416

        return 512

    def resize_for_ai(self, frame):
        target = self.inference_size()

        h, w = frame.shape[:2]

        scale = min(
            target / w,
            target / h
        )

        if scale >= 1:
            return frame

        nw = max(
            32,
            int(w * scale)
        )

        nh = max(
            32,
            int(h * scale)
        )

        return cv2.resize(
            frame,
            (nw, nh),
            interpolation=cv2.INTER_AREA
        )

    def worker_loop(self):
        while not self.stop_event.is_set():
            frame, frame_id = (
                self.shared.get_frame()
            )

            if frame is None:
                time.sleep(0.003)
                continue

            if frame_id == self.latest_frame_id:
                time.sleep(0.003)
                continue

            self.latest_frame_id = frame_id

            started = time.perf_counter()

            try:
                self.process(
                    frame
                )
            except Exception:
                pass

            elapsed = (
                time.perf_counter()
                - started
            )

            self.ai_fps.update(
                1 / max(
                    elapsed,
                    0.001
                )
            )

    def process(self, frame):
        ai_frame = (
            self.resize_for_ai(
                frame
            )
        )

        h, w = ai_frame.shape[:2]
        fh, fw = frame.shape[:2]

        sx = fw / w
        sy = fh / h

        now = time.perf_counter()

        if self.config.mode == (
            "Object & People Awareness"
        ):
            self.process_object_mode(
                ai_frame,
                sx,
                sy,
                now
            )

        elif self.config.mode == (
            "Gesture & Body Awareness"
        ):
            self.process_gesture_mode(
                ai_frame,
                sx,
                sy,
                now
            )

        else:
            self.process_live_mode(
                ai_frame,
                sx,
                sy,
                now
            )

        scene = self.get_scene()

        narrative = (
            self.get_narrative(
                scene
            )
        )

        self.generate_events(
            scene
        )

        snapshot = Snapshot(
            timestamp=time.time(),
            scene=scene,
            narrative=narrative,
            camera_fps=(
                self.camera_fps.value
                or 0
            ),
            ai_fps=(
                self.ai_fps.value
                or 0
            ),
            latency=(
                1000 /
                max(
                    self.ai_fps.value or 1,
                    0.1
                )
            ),
            objects=list(
                self.objects
            ),
            people=list(
                self.people
            ),
            hands=list(
                self.hands
            )
        )

        self.shared.set_snapshot(
            snapshot
        )

        self.latest_output = (
            self.draw(
                frame,
                scene
            )
        )

    def process_object_mode(
        self,
        frame,
        sx,
        sy,
        now
    ):
        if (
            self.object_model
            and now - self.last_object_run
            >= self.object_interval
        ):
            self.objects = (
                self.detect_objects(
                    frame,
                    sx,
                    sy
                )
            )

            self.last_object_run = now

        self.people = []

        for obj in self.objects:
            if obj.label == "person":
                self.people.append(
                    PersonState(
                        person_id=obj.track_id,
                        confidence=obj.confidence,
                        box=obj.box
                    )
                )

        self.hands = []

    def process_gesture_mode(
        self,
        frame,
        sx,
        sy,
        now
    ):
        self.objects = []

        if (
            self.pose_model
            and now - self.last_pose_run
            >= self.pose_interval
        ):
            self.people = (
                self.detect_pose(
                    frame,
                    sx,
                    sy
                )
            )

            self.last_pose_run = now

        if (
            self.hand_model
            and self.hand_model.available
            and now - self.last_hand_run
            >= self.hand_interval
        ):
            self.hands = (
                self.detect_hands(
                    frame,
                    sx,
                    sy
                )
            )

            self.last_hand_run = now

        self.attach_hands()

    def process_live_mode(
        self,
        frame,
        sx,
        sy,
        now
    ):
        if (
            self.object_model
            and now - self.last_object_run
            >= self.object_interval
        ):
            self.objects = (
                self.detect_objects(
                    frame,
                    sx,
                    sy
                )
            )

            self.last_object_run = now

        if (
            self.pose_model
            and any(
                obj.label == "person"
                for obj in self.objects
            )
            and now - self.last_pose_run
            >= self.pose_interval
        ):
            self.people = (
                self.detect_pose(
                    frame,
                    sx,
                    sy
                )
            )

            self.last_pose_run = now

        if (
            self.hand_model
            and self.hand_model.available
            and now - self.last_hand_run
            >= self.hand_interval
        ):
            self.hands = (
                self.detect_hands(
                    frame,
                    sx,
                    sy
                )
            )

            self.last_hand_run = now

        self.attach_hands()

    def detect_objects(
        self,
        frame,
        sx,
        sy
    ):
        try:
            results = (
                self.object_model.predict(
                    frame,
                    imgsz=self.inference_size(),
                    conf=self.config.object_confidence,
                    iou=0.5,
                    max_det=30,
                    verbose=False,
                    device="cpu"
                )
            )
        except Exception:
            return self.objects

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        objects = []

        for box in result.boxes:
            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            label = result.names.get(
                class_id,
                str(class_id)
            )

            group = group_for(
                label
            )

            if (
                group
                not in self.config.groups
            ):
                continue

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            objects.append(
                ObjectState(
                    label=label,
                    confidence=confidence,
                    box=(
                        int(x1 * sx),
                        int(y1 * sy),
                        int(x2 * sx),
                        int(y2 * sy)
                    ),
                    group=group
                )
            )

        return self.object_tracker.update(
            objects
        )

    def detect_pose(
        self,
        frame,
        sx,
        sy
    ):
        try:
            results = (
                self.pose_model.predict(
                    frame,
                    imgsz=self.inference_size(),
                    conf=self.config.pose_confidence,
                    iou=0.5,
                    max_det=8,
                    verbose=False,
                    device="cpu"
                )
            )
        except Exception:
            return self.people

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        keypoints = None

        if result.keypoints is not None:
            try:
                keypoints = (
                    result.keypoints.xy
                    .cpu()
                    .numpy()
                )
            except Exception:
                keypoints = None

        people = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box

            points = []

            if (
                keypoints is not None
                and i < len(keypoints)
            ):
                for point in keypoints[i]:
                    x, y = point

                    if x <= 0 or y <= 0:
                        points.append(None)
                    else:
                        points.append(
                            (
                                float(x * sx),
                                float(y * sy)
                            )
                        )

            person = PersonState(
                person_id=-1,
                confidence=0.8,
                box=(
                    int(x1 * sx),
                    int(y1 * sy),
                    int(x2 * sx),
                    int(y2 * sy)
                ),
                keypoints=points
            )

            person.posture = (
                self.kinematics.posture(
                    points
                )
            )

            people.append(
                person
            )

        people = (
            self.person_tracker.update(
                people
            )
        )

        for person in people:
            speed, movement = (
                self.motion.update(
                    person.person_id,
                    person.box
                )
            )

            person.velocity = speed
            person.movement = movement

        return people

    def detect_hands(
        self,
        frame,
        sx,
        sy
    ):
        hands = (
            self.hand_model.detect(
                frame
            )
        )

        output = []

        for hand in hands:
            x1, y1, x2, y2 = (
                hand["box"]
            )

            landmarks = []

            for x, y in hand["landmarks"]:
                landmarks.append(
                    (
                        x * sx,
                        y * sy
                    )
                )

            output.append(
                HandState(
                    handedness=(
                        hand["handedness"]
                    ),
                    confidence=(
                        hand["confidence"]
                    ),
                    box=(
                        int(x1 * sx),
                        int(y1 * sy),
                        int(x2 * sx),
                        int(y2 * sy)
                    ),
                    gesture=(
                        self.gesture.recognize(
                            landmarks
                        )
                    ),
                    landmarks=landmarks
                )
            )

        return output

    def attach_hands(self):
        for person in self.people:
            person.hands = []
            person.gestures = []

        for hand in self.hands:
            hx, hy = center(
                hand.box
            )

            closest = None
            closest_distance = float(
                "inf"
            )

            for person in self.people:
                x1, y1, x2, y2 = (
                    person.box
                )

                if (
                    x1 - 80 <= hx <= x2 + 80
                    and
                    y1 - 80 <= hy <= y2 + 80
                ):
                    d = distance(
                        (hx, hy),
                        center(person.box)
                    )

                    if d < closest_distance:
                        closest_distance = d
                        closest = person

            if closest:
                hand.person_id = (
                    closest.person_id
                )

                closest.hands.append(
                    hand
                )

                if (
                    hand.gesture
                    != "Unknown"
                    and
                    hand.gesture
                    not in closest.gestures
                ):
                    closest.gestures.append(
                        hand.gesture
                    )

    def get_scene(self):
        people = len(
            self.people
        )

        animals = sum(
            1
            for obj in self.objects
            if obj.group == "Animals"
        )

        other_objects = sum(
            1
            for obj in self.objects
            if obj.label != "person"
        )

        gestures = any(
            person.gestures
            for person in self.people
        )

        if self.config.mode == (
            "Object & People Awareness"
        ):
            if people > 1 and other_objects:
                return "Multiple Humans + Objects"

            if people > 1:
                return "Multiple Humans"

            if people == 1 and other_objects:
                return "Human + Objects"

            if people == 1:
                return "Single Human"

            if animals:
                return "Animal Scene"

            if other_objects:
                return "Object Scene"

            return "No Recognized Objects"

        if self.config.mode == (
            "Gesture & Body Awareness"
        ):
            if people > 1 and gestures:
                return "Multiple Humans + Gestures"

            if people > 1:
                return "Multiple Humans"

            if people == 1 and gestures:
                return "Human + Gesture"

            if people == 1:
                return "Human Body"

            return "No Human Detected"

        if people > 1 and animals:
            return "Multiple Humans + Animal"

        if people > 1 and other_objects:
            return "Multiple Humans + Objects"

        if people > 1:
            return "Multiple Humans"

        if people == 1 and animals:
            return "Human + Animal"

        if people == 1 and other_objects:
            return "Human + Objects"

        if people == 1 and gestures:
            return "Human + Gesture"

        if people == 1:
            return "Single Human"

        if animals:
            return "Animal Scene"

        if other_objects:
            return "Object Scene"

        return "No Recognized Scene"

    def get_narrative(self, scene):
        parts = []

        if self.people:
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
                        "; gesture: "
                        + ", ".join(
                            person.gestures
                        )
                    )

                parts.append(text)

        counts = {}

        for obj in self.objects:
            if obj.label == "person":
                continue

            counts[obj.label] = (
                counts.get(
                    obj.label,
                    0
                ) + 1
            )

        if counts:
            values = []

            for label, count in counts.items():
                values.append(
                    f"{count} {label}"
                    if count > 1
                    else label
                )

            parts.append(
                "Objects: "
                + ", ".join(values)
            )

        if not parts:
            return scene

        return " | ".join(parts)

    def generate_events(self, scene):
        self.events.update(
            "scene",
            scene,
            "Scene",
            f"Scene changed to {scene}"
        )

        self.events.update(
            "people",
            len(self.people),
            "People",
            f"Detected {len(self.people)} people"
        )

        for person in self.people:
            self.events.update(
                f"posture_{person.person_id}",
                person.posture,
                "Posture",
                (
                    f"Person "
                    f"{person.person_id}: "
                    f"{person.posture}"
                )
            )

            gestures = tuple(
                sorted(
                    person.gestures
                )
            )

            self.events.update(
                f"gesture_{person.person_id}",
                gestures,
                "Gesture",
                (
                    f"Person "
                    f"{person.person_id}: "
                    f"{', '.join(gestures)}"
                )
            )

    def draw(self, frame, scene):
        output = frame.copy()

        if self.config.mirror:
            output = cv2.flip(
                output,
                1
            )

        height, width = (
            output.shape[:2]
        )

        if (
            self.config.mode
            != "Gesture & Body Awareness"
            and self.config.show_object_boxes
        ):
            for obj in self.objects:
                x1, y1, x2, y2 = obj.box

                if self.config.mirror:
                    x1, x2 = (
                        width - x2,
                        width - x1
                    )

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 210, 255),
                    2
                )

                cv2.putText(
                    output,
                    (
                        f"{obj.label} "
                        f"{obj.confidence:.2f}"
                    ),
                    (
                        x1,
                        max(20, y1 - 7)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 210, 255),
                    2,
                    cv2.LINE_AA
                )

        for person in self.people:
            x1, y1, x2, y2 = (
                person.box
            )

            if self.config.mirror:
                x1, x2 = (
                    width - x2,
                    width - x1
                )

            if self.config.show_person_boxes:
                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 80),
                    2
                )

                cv2.putText(
                    output,
                    (
                        f"Person "
                        f"{person.person_id} "
                        f"| {person.posture}"
                    ),
                    (
                        x1,
                        max(20, y1 - 8)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 80),
                    2,
                    cv2.LINE_AA
                )

            if self.config.show_body_nodes:
                self.draw_pose(
                    output,
                    person.keypoints,
                    width
                )

            if self.config.show_gesture_labels:
                y = y1 + 22

                for gesture in person.gestures:
                    cv2.putText(
                        output,
                        gesture,
                        (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 80, 255),
                        2,
                        cv2.LINE_AA
                    )

                    y += 22

        if self.config.show_hand_nodes:
            for hand in self.hands:
                self.draw_hand(
                    output,
                    hand,
                    width
                )

        if self.config.show_hud:
            cv2.rectangle(
                output,
                (10, 10),
                (
                    min(
                        width - 10,
                        600
                    ),
                    70
                ),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                output,
                "LIFEVISION",
                (22, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                output,
                (
                    f"{self.config.mode} "
                    f"| {scene}"
                ),
                (22, 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (80, 220, 255),
                2,
                cv2.LINE_AA
            )

        if self.config.show_fps:
            cv2.putText(
                output,
                (
                    f"AI "
                    f"{self.ai_fps.value or 0:.1f}"
                ),
                (
                    max(
                        10,
                        width - 90
                    ),
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return output

    def draw_pose(
        self,
        frame,
        points,
        width
    ):
        if len(points) < 17:
            return

        transformed = []

        for point in points:
            if point is None:
                transformed.append(
                    None
                )
                continue

            x, y = point

            if self.config.mirror:
                x = width - x

            transformed.append(
                (
                    int(x),
                    int(y)
                )
            )

        for a, b in POSE_CONNECTIONS:
            if (
                transformed[a] is None
                or transformed[b] is None
            ):
                continue

            cv2.line(
                frame,
                transformed[a],
                transformed[b],
                (70, 190, 255),
                2,
                cv2.LINE_AA
            )

        for point in transformed:
            if point is None:
                continue

            cv2.circle(
                frame,
                point,
                4,
                (255, 255, 255),
                -1,
                cv2.LINE_AA
            )

    def draw_hand(
        self,
        frame,
        hand,
        width
    ):
        points = []

        for x, y in hand.landmarks:
            if self.config.mirror:
                x = width - x

            points.append(
                (
                    int(x),
                    int(y)
                )
            )

        for a, b in HAND_CONNECTIONS:
            if (
                a < len(points)
                and b < len(points)
            ):
                cv2.line(
                    frame,
                    points[a],
                    points[b],
                    (255, 100, 255),
                    2,
                    cv2.LINE_AA
                )

        for point in points:
            cv2.circle(
                frame,
                point,
                3,
                (255, 255, 255),
                -1,
                cv2.LINE_AA
            )

        x1, y1, x2, y2 = (
            hand.box
        )

        if self.config.mirror:
            x1, x2 = (
                width - x2,
                width - x1
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 100, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"{hand.handedness} | "
                f"{hand.gesture}"
            ),
            (
                x1,
                max(20, y1 - 7)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 100, 255),
            2,
            cv2.LINE_AA
        )

    def recv(self, frame):
        image = frame.to_ndarray(
            format="bgr24"
        )

        now = time.perf_counter()

        delta = (
            now
            - self.last_camera_time
        )

        self.last_camera_time = now

        if delta > 0:
            self.camera_fps.update(
                1 / delta
            )

        camera_frame = image

        if self.config.mirror:
            camera_frame = cv2.flip(
                camera_frame,
                1
            )

        self.shared.set_frame(
            camera_frame
        )

        output = (
            self.latest_output
        )

        if output is None:
            output = camera_frame

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )

    def stop(self):
        self.stop_event.set()

        try:
            if (
                self.hand_model
                and self.hand_model.landmarker
            ):
                self.hand_model.landmarker.close()
        except Exception:
            pass


if "lv_shared" not in st.session_state:
    st.session_state.lv_shared = (
        SharedState()
    )

if "lv_config" not in st.session_state:
    st.session_state.lv_config = (
        Config()
    )


shared = st.session_state.lv_shared
config = st.session_state.lv_config


with st.sidebar:
    st.title("👁️ LifeVision")

    new_mode = st.radio(
        "Vision Mode",
        [
            "Object & People Awareness",
            "Gesture & Body Awareness",
            "Live Scene"
        ],
        index=[
            "Object & People Awareness",
            "Gesture & Body Awareness",
            "Live Scene"
        ].index(config.mode)
    )

    config.mode = new_mode

    st.divider()

    config.performance = st.select_slider(
        "Performance",
        options=[
            "Fast",
            "Balanced",
            "Accurate"
        ],
        value=config.performance
    )

    st.caption(
        "Fast is recommended for CPU systems."
    )

    st.divider()

    if config.mode in [
        "Object & People Awareness",
        "Live Scene"
    ]:
        st.subheader("Object Detection")

        config.object_confidence = st.slider(
            "Object confidence",
            0.15,
            0.80,
            config.object_confidence,
            0.05
        )

    if config.mode in [
        "Gesture & Body Awareness",
        "Live Scene"
    ]:
        st.subheader("Body & Gesture")

        config.pose_confidence = st.slider(
            "Pose confidence",
            0.15,
            0.80,
            config.pose_confidence,
            0.05
        )

        config.hand_confidence = st.slider(
            "Hand confidence",
            0.15,
            0.80,
            config.hand_confidence,
            0.05
        )

    st.divider()

    st.subheader("Camera")

    config.mirror = st.checkbox(
        "Mirror camera",
        config.mirror
    )

    config.show_hud = st.checkbox(
        "Show HUD",
        config.show_hud
    )

    config.show_fps = st.checkbox(
        "Show FPS",
        config.show_fps
    )

    st.divider()

    st.subheader("Visual Nodes")

    config.show_person_boxes = st.checkbox(
        "Person boxes",
        config.show_person_boxes
    )

    config.show_object_boxes = st.checkbox(
        "Object boxes",
        config.show_object_boxes
    )

    config.show_body_nodes = st.checkbox(
        "Body nodes",
        config.show_body_nodes
    )

    config.show_hand_nodes = st.checkbox(
        "Hand nodes",
        config.show_hand_nodes
    )

    config.show_gesture_labels = st.checkbox(
        "Gesture labels",
        config.show_gesture_labels
    )

    st.divider()

    if config.mode in [
        "Object & People Awareness",
        "Live Scene"
    ]:
        config.groups = st.multiselect(
            "Object categories",
            list(
                OBJECT_GROUPS.keys()
            ),
            default=config.groups
        )


processor_key = (
    config.mode,
    config.performance,
    config.object_confidence,
    config.pose_confidence,
    config.hand_confidence,
    tuple(config.groups)
)


if (
    "lv_processor_key"
    not in st.session_state
    or
    st.session_state.lv_processor_key
    != processor_key
):
    old_processor = (
        st.session_state.get(
            "lv_processor"
        )
    )

    if old_processor:
        old_processor.stop()

    st.session_state.lv_processor = (
        LifeVisionProcessor(
            shared,
            config
        )
    )

    st.session_state.lv_processor_key = (
        processor_key
    )


processor = (
    st.session_state.lv_processor
)


def processor_factory():
    return processor


webrtc_streamer(
    key="lifevision-camera-v11",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration(
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
    ),
    media_stream_constraints={
        "video": {
            "width": {
                "ideal": 1280
            },
            "height": {
                "ideal": 720
            },
            "frameRate": {
                "ideal": 30,
                "max": 30
            }
        },
        "audio": False
    },
    video_processor_factory=processor_factory,
    async_processing=True
)


st.divider()


@st.fragment(run_every=0.7)
def information_panel():
    snapshot = (
        shared.get_snapshot()
    )

    st.subheader(
        "Live Information"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Mode",
            config.mode
        )

    with c2:
        st.metric(
            "Scene",
            snapshot.scene
        )

    with c3:
        st.metric(
            "People",
            len(snapshot.people)
        )

    with c4:
        st.metric(
            "Objects",
            len(snapshot.objects)
        )

    with c5:
        st.metric(
            "AI FPS",
            f"{snapshot.ai_fps:.1f}"
        )

    st.write(
        snapshot.narrative
        or "Waiting for camera analysis..."
    )

    if config.mode in [
        "Gesture & Body Awareness",
        "Live Scene"
    ]:
        if snapshot.people:
            st.markdown(
                "### Body & Gesture Information"
            )

            rows = []

            for person in snapshot.people:
                rows.append(
                    {
                        "Person":
                            f"Person {person.person_id}",
                        "Posture":
                            person.posture,
                        "Movement":
                            person.movement,
                        "Velocity":
                            f"{person.velocity:.1f}",
                        "Gestures":
                            (
                                ", ".join(
                                    person.gestures
                                )
                                if person.gestures
                                else "None"
                            )
                    }
                )

            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True
            )

        if snapshot.hands:
            st.markdown(
                "### Hand Gestures"
            )

            rows = []

            for index, hand in enumerate(
                snapshot.hands,
                1
            ):
                rows.append(
                    {
                        "Hand":
                            index,
                        "Person":
                            (
                                f"Person "
                                f"{hand.person_id}"
                                if hand.person_id >= 0
                                else "Unassigned"
                            ),
                        "Side":
                            hand.handedness,
                        "Gesture":
                            hand.gesture,
                        "Confidence":
                            f"{hand.confidence:.2f}"
                    }
                )

            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True
            )

    if config.mode in [
        "Object & People Awareness",
        "Live Scene"
    ]:
        if snapshot.objects:
            st.markdown(
                "### Detected Objects"
            )

            rows = []

            for obj in snapshot.objects:
                rows.append(
                    {
                        "Object":
                            obj.label,
                        "Category":
                            obj.group,
                        "Confidence":
                            f"{obj.confidence:.2f}",
                        "Track":
                            obj.track_id
                    }
                )

            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True
            )

    st.markdown(
        "### Event Log"
    )

    events = (
        shared.get_events()
    )

    if events:
        for event in events[:15]:
            stamp = time.strftime(
                "%H:%M:%S",
                time.localtime(
                    event.timestamp
                )
            )

            st.write(
                f"**{stamp} · "
                f"{event.category}** — "
                f"{event.message}"
            )
    else:
        st.caption(
            "No events recorded yet."
        )


information_panel()


st.divider()

st.caption(
    f"LifeVision {APP_VERSION} · "
    "Newest-frame processing · "
    "CPU-optimized vision pipeline"
)