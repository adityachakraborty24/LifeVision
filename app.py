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
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


APP_VERSION = "14.0"

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
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear"
    },
    "Food": {
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
        "skateboard",
        "surfboard",
        "tennis racket",
        "baseball bat",
        "baseball glove"
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

RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302",
                "stun:stun2.l.google.com:19302"
            ]
        },
        {
            "urls": [
                "stun:stun.cloudflare.com:3478"
            ]
        }
    ]
}


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
    history: deque = field(default_factory=lambda: deque(maxlen=12))


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


@dataclass
class Config:
    mode: str = "Live Scene"
    object_confidence: float = 0.38
    pose_confidence: float = 0.38
    hand_confidence: float = 0.45
    process_fps: int = 6
    show_boxes: bool = True
    show_labels: bool = True
    show_body_nodes: bool = True
    show_hand_nodes: bool = True
    show_gesture_labels: bool = True
    mirror: bool = True
    groups: tuple = (
        "People",
        "Animals",
        "Vehicles",
        "Indoor",
        "Food",
        "Sports"
    )


class SharedState:
    def __init__(self):
        self.lock = threading.RLock()
        self.frame = None
        self.frame_id = 0
        self.output = None
        self.snapshot = Snapshot(
            timestamp=time.time(),
            frame_id=0,
            objects=[],
            people=[],
            hands=[],
            scene="Waiting",
            narrative="Camera waiting for frames.",
            fps=0.0,
            ai_fps=0.0
        )
        self.events = deque(maxlen=100)
        self.runtime_error = ""
        self.last_input_time = 0.0

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame
            self.frame_id += 1
            self.last_input_time = time.time()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None, self.frame_id
            return self.frame.copy(), self.frame_id

    def set_output(self, frame):
        with self.lock:
            self.output = frame

    def get_output(self):
        with self.lock:
            if self.output is None:
                return None
            return self.output.copy()

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
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False

    def update(self, value):
        if not self.initialized:
            self.value = value
            self.initialized = True
        else:
            self.value = (
                self.alpha * value
                + (1.0 - self.alpha) * self.value
            )
        return self.value


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

        valid = [
            p for p in points
            if len(p) >= 3 and p[2] > 0.25
        ]

        if len(valid) < 8:
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

            if (
                nose[2] < 0.25
                or left_shoulder[2] < 0.25
                or right_shoulder[2] < 0.25
            ):
                return "Unknown"

            shoulder_y = (
                left_shoulder[1] + right_shoulder[1]
            ) / 2.0

            hip_y = (
                left_hip[1] + right_hip[1]
            ) / 2.0

            knee_y = (
                left_knee[1] + right_knee[1]
            ) / 2.0

            ankle_y = (
                left_ankle[1] + right_ankle[1]
            ) / 2.0

            torso = abs(hip_y - shoulder_y)
            leg = abs(ankle_y - knee_y)

            if torso < 35 and leg < 45:
                return "Sitting"

            if torso < 55 and leg < 70:
                return "Sitting"

            if torso > 70 and leg > 55:
                return "Standing"

            shoulder_width = abs(
                left_shoulder[0] - right_shoulder[0]
            )

            torso_width = abs(
                left_hip[0] - right_hip[0]
            )

            if shoulder_width > 0 and torso_width > 0:
                ratio = torso / max(
                    shoulder_width,
                    torso_width,
                    1
                )

                if ratio < 1.0:
                    return "Lying"

            return "Standing"

        except Exception:
            return "Unknown"


class GestureRecognizer:
    @staticmethod
    def finger_states(points):
        if len(points) != 21:
            return None

        def angle(a, b, c):
            ba = np.array([
                a[0] - b[0],
                a[1] - b[1]
            ])

            bc = np.array([
                c[0] - b[0],
                c[1] - b[1]
            ])

            denom = (
                np.linalg.norm(ba)
                * np.linalg.norm(bc)
                + 1e-6
            )

            value = np.dot(ba, bc) / denom
            value = np.clip(value, -1.0, 1.0)

            return math.degrees(
                math.acos(value)
            )

        fingers = {}

        fingers["index"] = (
            angle(points[5], points[6], points[8]) < 55
        )

        fingers["middle"] = (
            angle(points[9], points[10], points[12]) < 55
        )

        fingers["ring"] = (
            angle(points[13], points[14], points[16]) < 55
        )

        fingers["pinky"] = (
            angle(points[17], points[18], points[20]) < 55
        )

        fingers["thumb"] = (
            angle(points[1], points[2], points[4]) < 55
        )

        return fingers

    @staticmethod
    def classify(points):
        fingers = GestureRecognizer.finger_states(points)

        if fingers is None:
            return "Unknown"

        thumb = fingers["thumb"]
        index = fingers["index"]
        middle = fingers["middle"]
        ring = fingers["ring"]
        pinky = fingers["pinky"]

        if index and middle and ring and pinky and not thumb:
            return "Open hand"

        if not index and not middle and not ring and not pinky:
            return "Fist"

        if thumb and not index and not middle and not ring and not pinky:
            return "Thumbs up"

        if index and not middle and not ring and not pinky:
            return "Pointing"

        if middle and not index and not ring and not pinky:
            return "Middle finger"

        if index and middle and not ring and not pinky:
            return "Peace"

        return "Hand gesture"


class EventEngine:
    def __init__(self, shared):
        self.shared = shared
        self.last_states = {}
        self.last_events = {}

    def emit(self, key, text, category="AI"):
        now = time.time()

        if (
            key in self.last_events
            and now - self.last_events[key] < 1.5
        ):
            return

        self.last_events[key] = now

        self.shared.add_event(
            Event(
                timestamp=now,
                text=text,
                category=category
            )
        )

    def update(self, snapshot):
        current = {}

        current["people"] = len(snapshot.people)
        current["scene"] = snapshot.scene

        if len(snapshot.people) > 0:
            for person in snapshot.people:
                current[
                    f"person_{person.person_id}_posture"
                ] = person.posture

                current[
                    f"person_{person.person_id}_movement"
                ] = person.movement

                gestures = tuple(sorted(person.gestures))

                current[
                    f"person_{person.person_id}_gestures"
                ] = gestures

        if len(snapshot.hands) > 0:
            current["hands"] = len(snapshot.hands)

            for hand in snapshot.hands:
                current[
                    f"hand_{hand.hand_id}"
                ] = hand.gesture

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
                    self.emit(
                        "people_increase",
                        f"Person count increased from {old} to {value}.",
                        "People"
                    )
                elif value < old:
                    self.emit(
                        "people_decrease",
                        f"Person count changed from {old} to {value}.",
                        "People"
                    )

            elif key == "scene":
                self.emit(
                    "scene_change",
                    f"Scene changed to {value}.",
                    "Scene"
                )

            elif key.endswith("_posture"):
                person_id = key.split("_")[1]
                self.emit(
                    key,
                    f"Person {person_id} is now {value}.",
                    "Body"
                )

            elif key.endswith("_movement"):
                person_id = key.split("_")[1]
                if value != "Still":
                    self.emit(
                        key,
                        f"Person {person_id} shows {value.lower()}.",
                        "Motion"
                    )

            elif key.endswith("_gestures"):
                person_id = key.split("_")[1]

                if value:
                    self.emit(
                        key,
                        f"Person {person_id}: {', '.join(value)}.",
                        "Gesture"
                    )

            elif key.startswith("hand_"):
                hand_id = key.split("_")[1]
                self.emit(
                    key,
                    f"Hand {hand_id}: {value}.",
                    "Hand"
                )


class HandLandmarker:
    def __init__(self, confidence):
        self.confidence = confidence
        self.landmarker = None
        self.load()

    def download(self):
        os.makedirs(
            HAND_CACHE_DIR,
            exist_ok=True
        )

        if os.path.exists(HAND_MODEL_PATH):
            return

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

        finally:
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except Exception:
                    pass

    def load(self):
        self.download()

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

        self.landmarker = vision.HandLandmarker.create_from_options(
            options
        )

    def detect(self, frame):
        if self.landmarker is None:
            return []

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.landmarker.detect(image)

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

            try:
                handedness = (
                    result.handedness[index][0].category_name
                )
            except Exception:
                pass

            try:
                confidence = float(
                    result.handedness[index][0].score
                )
            except Exception:
                confidence = 0.0

            gesture = GestureRecognizer.classify(
                points
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

    def close(self):
        try:
            if self.landmarker is not None:
                self.landmarker.close()
        except Exception:
            pass

        self.landmarker = None


class LifeVisionProcessor:
    def __init__(self, shared, config):
        self.shared = shared
        self.config = config

        self.object_model = None
        self.pose_model = None
        self.hand_model = None
        self.hand_confidence_loaded = None

        self.object_tracker = Tracker()
        self.motion = Motion()
        self.events = EventEngine(shared)

        self.objects = []
        self.people = []
        self.hands = []

        self.running = True
        self.worker = None

        self.last_object_time = 0.0
        self.last_pose_time = 0.0
        self.last_hand_time = 0.0

        self.last_processed_frame = -1
        self.last_output = None

        self.camera_fps = EMA(0.15)
        self.ai_fps = EMA(0.15)

        self.last_worker_time = time.time()
        self.input_counter = 0
        self.input_time = time.time()

        self.ensure_models()

        self.worker = threading.Thread(
            target=self.worker_loop,
            daemon=True
        )

        self.worker.start()

    def ensure_models(self):
        mode = self.config.mode

        needs_objects = mode in {
            "Object & People Awareness",
            "Gesture & Body Awareness",
            "Live Scene"
        }

        needs_pose = mode in {
            "Gesture & Body Awareness",
            "Live Scene"
        }

        needs_hands = mode in {
            "Gesture & Body Awareness",
            "Live Scene"
        }

        if needs_objects and self.object_model is None:
            self.object_model = YOLO(
                OBJECT_MODEL
            )

        if needs_pose and self.pose_model is None:
            self.pose_model = YOLO(
                POSE_MODEL
            )

        if needs_hands:
            confidence = float(
                self.config.hand_confidence
            )

            if (
                self.hand_model is None
                or self.hand_confidence_loaded is None
                or abs(
                    self.hand_confidence_loaded
                    - confidence
                ) > 0.001
            ):
                if self.hand_model is not None:
                    self.hand_model.close()

                self.hand_model = HandLandmarker(
                    confidence
                )

                self.hand_confidence_loaded = confidence

    def allowed_labels(self):
        labels = set()

        for group in self.config.groups:
            labels.update(
                OBJECT_GROUPS.get(group, set())
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
                "conf": float(
                    self.config.object_confidence
                ),
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

            detections = []

            if not results:
                return detections

            result = results[0]

            if result.boxes is None:
                return detections

            names = result.names

            for box in result.boxes:
                confidence = float(
                    box.conf[0].item()
                )

                cls = int(
                    box.cls[0].item()
                )

                label = str(
                    names.get(cls, cls)
                )

                if people_only:
                    if label != "person":
                        continue
                else:
                    if (
                        label != "person"
                        and label not in self.allowed_labels()
                    ):
                        continue

                xyxy = box.xyxy[0].cpu().numpy()

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
                    (x2 - x1) * (y2 - y1)
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

            detections = self.object_tracker.assign(
                detections
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
            self.shared.runtime_error = (
                f"Object detection: {exc}"
            )

            if people_only:
                return [
                    obj
                    for obj in self.objects
                    if obj.label == "person"
                ]

            return self.objects

    @staticmethod
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

        area_a = max(
            1,
            (ax2 - ax1) * (ay2 - ay1)
        )

        area_b = max(
            1,
            (bx2 - bx1) * (by2 - by1)
        )

        return intersection / (
            area_a + area_b - intersection + 1e-6
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
                person.keypoints = previous.keypoints
                person.posture = previous.posture
                person.movement = previous.movement
                person.velocity = previous.velocity
                person.gestures = previous.gestures
                person.hands = previous.hands
                person.last_center = previous.center
                person.history = previous.history

            new_people.append(person)

        self.people = new_people

        for person in self.people:
            self.motion.update(person)

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
                conf=float(
                    self.config.pose_confidence
                ),
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
                xyxy = box.xyxy[0].cpu().numpy()

                bbox = tuple(
                    int(v)
                    for v in xyxy
                )

                try:
                    points = (
                        result.keypoints
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

            used_people = set()

            for pose_bbox, points in candidates:
                best_person = None
                best_score = 0.10

                for person in self.people:
                    if person.person_id in used_people:
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

                used_people.add(
                    best_person.person_id
                )

                best_person.keypoints = points
                best_person.posture = (
                    Kinematics.posture_from_keypoints(
                        points
                    )
                )

        except Exception as exc:
            self.shared.runtime_error = (
                f"Pose detection: {exc}"
            )

    def detect_hands(self, frame):
        if self.hand_model is None:
            self.hands = []
            return

        try:
            self.hands = self.hand_model.detect(
                frame
            )

        except Exception as exc:
            self.shared.runtime_error = (
                f"Hand detection: {exc}"
            )

            self.hands = []

    def normalized_hand_bbox(
        self,
        hand,
        width,
        height
    ):
        x1, y1, x2, y2 = hand.bbox

        return (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height)
        )

    def attach_hands(self, width, height):
        for person in self.people:
            person.hands = []
            person.gestures = []

        for hand in self.hands:
            hx1, hy1, hx2, hy2 = (
                self.normalized_hand_bbox(
                    hand,
                    width,
                    height
                )
            )

            hcx = (hx1 + hx2) / 2
            hcy = (hy1 + hy2) / 2

            best_person = None
            best_distance = float("inf")

            for person in self.people:
                px1, py1, px2, py2 = person.bbox

                expanded = (
                    px1 - 80,
                    py1 - 80,
                    px2 + 80,
                    py2 + 80
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
                    and hand.gesture not in best_person.gestures
                ):
                    best_person.gestures.append(
                        hand.gesture
                    )

    def build_scene(self):
        people_count = len(self.people)

        animals = [
            o for o in self.objects
            if o.label in OBJECT_GROUPS["Animals"]
        ]

        things = [
            o for o in self.objects
            if (
                o.label != "person"
                and o.label not in OBJECT_GROUPS["Animals"]
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

        if people_count > 1:
            base = "Multiple Humans"
        else:
            base = "Human"

        parts = [base]

        if animals:
            parts.append("Animals")

        if things:
            parts.append("Objects")

        if gestures:
            parts.append("Gestures")

        return " + ".join(parts)

    def build_narrative(self, scene):
        people_count = len(self.people)

        if people_count == 0:
            if self.objects:
                labels = sorted(
                    set(
                        obj.label
                        for obj in self.objects
                    )
                )

                if labels:
                    return (
                        "No human is currently confirmed. "
                        "Visible objects include "
                        + ", ".join(labels[:8])
                        + "."
                    )

            return (
                "No human is currently confirmed "
                "by the object detector."
            )

        descriptions = []

        for person in self.people:
            text = (
                f"Person {person.person_id} is "
                f"{person.posture.lower()}"
            )

            if person.movement != "Still":
                text += (
                    f" and {person.movement.lower()}"
                )

            if person.gestures:
                text += (
                    ", showing "
                    + ", ".join(person.gestures)
                )

            descriptions.append(
                text
            )

        other_labels = sorted(
            set(
                obj.label
                for obj in self.objects
                if obj.label != "person"
            )
        )

        result = "; ".join(
            descriptions
        )

        if other_labels:
            result += (
                ". Detected nearby items: "
                + ", ".join(other_labels[:8])
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

        if self.config.show_boxes:
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 220, 80),
                2
            )

        if self.config.show_labels:
            text = (
                f"{obj.label} "
                f"{obj.confidence:.0%}"
            )

            if obj.label == "person":
                text = (
                    f"Person {obj.track_id} "
                    f"{obj.confidence:.0%}"
                )

            cv2.putText(
                frame,
                text,
                (x1, max(22, y1 - 8)),
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
        x1, y1, x2, y2 = person.bbox

        if self.config.show_boxes:
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
            self.config.show_gesture_labels
            and person.gestures
        ):
            label += (
                " | "
                + ", ".join(person.gestures)
            )

        if self.config.show_labels:
            cv2.putText(
                frame,
                label,
                (x1, max(22, y1 - 10)),
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
            not self.config.show_body_nodes
            or not person.keypoints
        ):
            return

        height, width = frame.shape[:2]

        points = []

        for x, y, confidence in person.keypoints:
            px = int(x)
            py = int(y)

            if confidence < 0.25:
                points.append(None)
                continue

            px = max(
                0,
                min(width - 1, px)
            )

            py = max(
                0,
                min(height - 1, py)
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

    def draw_hands(
        self,
        frame
    ):
        height, width = frame.shape[:2]

        for hand in self.hands:
            points = []

            for x, y, z in hand.landmarks:
                px = int(x * width)
                py = int(y * height)

                points.append(
                    (px, py)
                )

            if self.config.show_hand_nodes:
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

            x1, y1, x2, y2 = (
                self.normalized_hand_bbox(
                    hand,
                    width,
                    height
                )
            )

            if self.config.show_boxes:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (80, 200, 255),
                    1
                )

            if self.config.show_gesture_labels:
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
        height, width = frame.shape[:2]

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (width, 68),
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

        lines = [
            f"LifeVision {APP_VERSION}",
            f"MODE: {self.config.mode}",
            (
                f"PEOPLE: {len(snapshot.people)}"
                f"  OBJECTS: {len(snapshot.objects)}"
                f"  HANDS: {len(snapshot.hands)}"
            ),
            (
                f"AI FPS: {snapshot.ai_fps:.1f}"
                f"  CAMERA FPS: {snapshot.fps:.1f}"
            )
        ]

        cv2.putText(
            frame,
            lines[0],
            (14, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            lines[1],
            (14, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            lines[2],
            (250, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            lines[3],
            (14, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    def draw(self, frame, snapshot):
        output = frame.copy()

        if self.config.mirror:
            output = cv2.flip(
                output,
                1
            )

        if self.config.mode in {
            "Object & People Awareness",
            "Live Scene"
        }:
            for obj in snapshot.objects:
                self.draw_object(
                    output,
                    obj
                )

        if self.config.mode in {
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

        return output

    def process_frame(self, frame, frame_id):
        now = time.time()

        self.ensure_models()

        mode = self.config.mode

        object_interval = max(
            0.12,
            1.0 / max(
                1,
                self.config.process_fps
            )
        )

        pose_interval = max(
            0.16,
            object_interval * 1.35
        )

        hand_interval = max(
            0.12,
            object_interval * 1.05
        )

        if (
            now - self.last_object_time
            >= object_interval
        ):
            if mode == "Gesture & Body Awareness":
                self.objects = self.detect_objects(
                    frame,
                    people_only=True
                )
            else:
                self.objects = self.detect_objects(
                    frame,
                    people_only=False
                )

            self.last_object_time = now

        self.sync_people()

        if (
            mode in {
                "Gesture & Body Awareness",
                "Live Scene"
            }
            and
            now - self.last_pose_time
            >= pose_interval
        ):
            self.detect_pose(
                frame
            )

            self.last_pose_time = now

        if (
            mode in {
                "Gesture & Body Awareness",
                "Live Scene"
            }
            and
            now - self.last_hand_time
            >= hand_interval
        ):
            self.detect_hands(
                frame
            )

            self.attach_hands(
                frame.shape[1],
                frame.shape[0]
            )

            self.last_hand_time = now

        scene = self.build_scene()

        narrative = self.build_narrative(
            scene
        )

        current_time = time.time()

        delta = max(
            0.001,
            current_time - self.last_worker_time
        )

        current_ai_fps = 1.0 / delta

        ai_fps = self.ai_fps.update(
            current_ai_fps
        )

        self.last_worker_time = current_time

        snapshot = Snapshot(
            timestamp=current_time,
            frame_id=frame_id,
            objects=list(self.objects),
            people=list(self.people),
            hands=list(self.hands),
            scene=scene,
            narrative=narrative,
            fps=self.camera_fps.value,
            ai_fps=ai_fps
        )

        self.events.update(
            snapshot
        )

        output = self.draw(
            frame,
            snapshot
        )

        self.shared.set_snapshot(
            snapshot
        )

        self.shared.set_output(
            output
        )

        self.last_output = output
        self.last_processed_frame = frame_id

    def worker_loop(self):
        while self.running:
            try:
                frame, frame_id = (
                    self.shared.get_frame()
                )

                if frame is None:
                    time.sleep(0.01)
                    continue

                if frame_id == self.last_processed_frame:
                    time.sleep(0.005)
                    continue

                self.process_frame(
                    frame,
                    frame_id
                )

            except Exception as exc:
                self.shared.runtime_error = (
                    f"Processing error: {exc}"
                )

                time.sleep(0.05)

    def recv(self, frame):
        try:
            image = frame.to_ndarray(
                format="bgr24"
            )

            now = time.time()

            self.input_counter += 1

            if now - self.input_time >= 1.0:
                measured = (
                    self.input_counter
                    / max(
                        0.001,
                        now - self.input_time
                    )
                )

                self.camera_fps.update(
                    measured
                )

                self.input_counter = 0
                self.input_time = now

            self.shared.set_frame(
                image
            )

            output = self.shared.get_output()

            if output is None:
                if self.config.mirror:
                    image = cv2.flip(
                        image,
                        1
                    )

                return av.VideoFrame.from_ndarray(
                    image,
                    format="bgr24"
                )

            return av.VideoFrame.from_ndarray(
                output,
                format="bgr24"
            )

        except Exception as exc:
            self.shared.runtime_error = (
                f"Camera callback: {exc}"
            )

            return frame

    def stop(self):
        self.running = False

        if self.hand_model is not None:
            self.hand_model.close()

        self.hand_model = None

        if (
            self.worker is not None
            and self.worker.is_alive()
        ):
            self.worker.join(
                timeout=1.0
            )


def make_processor(shared, config):
    return LifeVisionProcessor(
        shared,
        config
    )


def render_events(shared):
    events = shared.get_events()

    if not events:
        st.caption(
            "No events recorded yet."
        )
        return

    for event in events[:12]:
        timestamp = time.strftime(
            "%H:%M:%S",
            time.localtime(
                event.timestamp
            )
        )

        st.write(
            f"**{timestamp} · {event.category}**  \n"
            f"{event.text}"
        )


def render_people(snapshot):
    if not snapshot.people:
        st.info(
            "No confirmed people detected."
        )
        return

    for person in snapshot.people:
        gestures = (
            ", ".join(person.gestures)
            if person.gestures
            else "None"
        )

        st.write(
            f"**Person {person.person_id}** — "
            f"{person.posture} · "
            f"{person.movement} · "
            f"Gesture: {gestures}"
        )


def render_objects(snapshot):
    if not snapshot.objects:
        st.info(
            "No selected objects detected."
        )
        return

    counts = {}

    for obj in snapshot.objects:
        counts[obj.label] = (
            counts.get(obj.label, 0) + 1
        )

    parts = [
        f"{label}: {count}"
        for label, count
        in sorted(counts.items())
    ]

    st.write(
        " · ".join(parts)
    )


st.set_page_config(
    page_title="LifeVision",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("LifeVision")
st.caption(
    "Real-Time Computer Vision · People · Objects · Body · Hands · Gestures · Scene Understanding"
)

if "lv_shared" not in st.session_state:
    st.session_state.lv_shared = SharedState()

if "lv_config" not in st.session_state:
    st.session_state.lv_config = Config()

if "lv_processor" not in st.session_state:
    st.session_state.lv_processor = None

shared = st.session_state.lv_shared
config = st.session_state.lv_config


with st.sidebar:
    st.header("LifeVision Controls")

    mode = st.radio(
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

    config.mode = mode

    st.subheader("Performance")

    config.process_fps = st.slider(
        "AI processing FPS",
        min_value=2,
        max_value=10,
        value=int(config.process_fps),
        step=1
    )

    st.subheader("Detection")

    config.object_confidence = st.slider(
        "Object confidence",
        min_value=0.20,
        max_value=0.80,
        value=float(config.object_confidence),
        step=0.01
    )

    config.pose_confidence = st.slider(
        "Body confidence",
        min_value=0.20,
        max_value=0.80,
        value=float(config.pose_confidence),
        step=0.01
    )

    config.hand_confidence = st.slider(
        "Hand confidence",
        min_value=0.20,
        max_value=0.80,
        value=float(config.hand_confidence),
        step=0.01
    )

    st.subheader("Visual Nodes")

    config.show_boxes = st.checkbox(
        "Detection boxes",
        value=config.show_boxes
    )

    config.show_labels = st.checkbox(
        "Detection labels",
        value=config.show_labels
    )

    config.show_body_nodes = st.checkbox(
        "Body skeleton",
        value=config.show_body_nodes
    )

    config.show_hand_nodes = st.checkbox(
        "Hand skeleton",
        value=config.show_hand_nodes
    )

    config.show_gesture_labels = st.checkbox(
        "Gesture labels",
        value=config.show_gesture_labels
    )

    config.mirror = st.checkbox(
        "Mirror camera",
        value=config.mirror
    )

    st.subheader("Object Groups")

    selected_groups = []

    for group in OBJECT_GROUPS:
        selected = st.checkbox(
            group,
            value=group in config.groups,
            key=f"group_{group}"
        )

        if selected:
            selected_groups.append(
                group
            )

    if selected_groups:
        config.groups = tuple(
            selected_groups
        )
    else:
        config.groups = ("People",)

    st.divider()

    st.caption(
        f"LifeVision {APP_VERSION}"
    )

    st.caption(
        "Free architecture · Local AI models · "
        "No paid API required"
    )


if st.session_state.lv_processor is None:
    st.session_state.lv_processor = make_processor(
        shared,
        config
    )

processor = st.session_state.lv_processor


st.subheader("Camera")

webrtc_ctx = webrtc_streamer(
    key="lifevision-camera-v14",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {
                "ideal": 640,
                "max": 1280
            },
            "height": {
                "ideal": 480,
                "max": 720
            },
            "frameRate": {
                "ideal": 20,
                "max": 24
            }
        },
        "audio": False
    },
    video_processor_factory=lambda: processor,
    async_processing=True,
    media_toggle_controls=True
)


if not webrtc_ctx.state.playing:
    st.info(
        "Press START above to activate the camera."
    )


@st.fragment(run_every=0.7)
def live_information():
    snapshot = shared.get_snapshot()

    st.divider()

    st.subheader("Live Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "People",
            len(snapshot.people)
        )

    with col2:
        st.metric(
            "Objects",
            len(snapshot.objects)
        )

    with col3:
        st.metric(
            "Hands",
            len(snapshot.hands)
        )

    with col4:
        st.metric(
            "AI FPS",
            f"{snapshot.ai_fps:.1f}"
        )

    st.write(
        f"**Scene:** {snapshot.scene}"
    )

    st.write(
        f"**Live interpretation:** {snapshot.narrative}"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "People",
            "Objects",
            "Events"
        ]
    )

    with tab1:
        render_people(
            snapshot
        )

    with tab2:
        render_objects(
            snapshot
        )

    with tab3:
        render_events(
            shared
        )

    if shared.runtime_error:
        with st.expander(
            "Runtime diagnostic"
        ):
            st.code(
                shared.runtime_error
            )


live_information()