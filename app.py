import os
import time
import math
import threading
import urllib.request
import json
from dataclasses import dataclass, field
from collections import deque, Counter

try:
    import spaces
except Exception:
    class _LocalSpaces:
        @staticmethod
        def GPU(function=None, **kwargs):
            if function is None:
                def wrapper(fn):
                    return fn
                return wrapper
            return function
    spaces = _LocalSpaces()

try:
    import torch
except Exception:
    torch = None

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


APP_VERSION = "26.0"

MODE_FULL = "Full Awareness"
MODE_OBJECTS = MODE_FULL
MODE_GESTURE = MODE_FULL

OBJECT_MODEL = "yolo11n.pt"
POSE_MODEL = "yolo11n-pose.pt"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "lifevision"
)

HAND_MODEL_PATH = os.path.join(
    CACHE_DIR,
    "hand_landmarker.task"
)

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
FACE_MODEL_PATH = os.path.join(
    CACHE_DIR,
    "face_landmarker.task"
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)

OBJECT_GROUPS = {
    "People": [
        "person"
    ],
    "Animals": [
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
    ],
    "Vehicles": [
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat"
    ],
    "Indoor": [
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
    ],
    "Food": [
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
    ],
    "Sports": [
        "sports ball",
        "skateboard",
        "surfboard",
        "tennis racket",
        "baseball bat",
        "baseball glove"
    ]
}

POSE_CONNECTIONS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6)
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
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15)
]

POSE_NAMES = {
    0: "nose",
    5: "left shoulder",
    6: "right shoulder",
    7: "left elbow",
    8: "right elbow",
    9: "left wrist",
    10: "right wrist",
    11: "left hip",
    12: "right hip",
    13: "left knee",
    14: "right knee",
    15: "left ankle",
    16: "right ankle"
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
class HandState:
    hand_id: int
    handedness: str
    landmarks: list
    bbox: tuple
    gesture: str
    confidence: float


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
        default_factory=lambda: deque(maxlen=20)
    )
    last_update: float = 0.0
    body_language: str = "Unknown"
    arm_state: str = "Unknown"
    leg_state: str = "Unknown"
    hand_activity: str = "No hand activity confirmed"
    phone_status: str = "No phone interaction confirmed"
    facial_expression: str = "Face not analyzed"
    face_bbox: tuple = None
    body_history: deque = field(default_factory=lambda: deque(maxlen=5))


@dataclass
class FaceState:
    face_id: int
    bbox: tuple
    expression: str
    confidence: float
    landmarks: list = field(default_factory=list)


@dataclass
class Event:
    timestamp: float
    text: str
    category: str


@dataclass
class Snapshot:
    timestamp: float = 0.0
    frame_id: int = 0
    objects: list = field(default_factory=list)
    people: list = field(default_factory=list)
    hands: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    scene: str = "Waiting for camera..."
    narrative: str = "Start the camera to begin LifeVision."
    fps: float = 0.0
    ai_fps: float = 0.0
    hand_available: bool = False
    runtime_error: str = ""


class Tracker:
    def __init__(self, max_distance=160):
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        if not detections:
            self._expire()
            return []

        used = set()
        output = []
        current_time = time.time()

        for det in detections:
            cx, cy = det["center"]

            best_id = None
            best_distance = self.max_distance

            for track_id, previous in self.tracks.items():
                if track_id in used:
                    continue

                px, py = previous["center"]

                distance = math.hypot(
                    cx - px,
                    cy - py
                )

                if distance < best_distance:
                    best_distance = distance
                    best_id = track_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            used.add(best_id)

            self.tracks[best_id] = {
                "center": (cx, cy),
                "bbox": det["bbox"],
                "timestamp": current_time
            }

            det["track_id"] = best_id
            output.append(det)

        self._expire()

        return output

    def _expire(self):
        current_time = time.time()

        expired = [
            track_id
            for track_id, value in self.tracks.items()
            if current_time - value["timestamp"] > 2.5
        ]

        for track_id in expired:
            self.tracks.pop(
                track_id,
                None
            )


class MotionAnalyzer:
    def calculate(self, person):
        if not person.history:
            return "Still", 0.0

        previous = person.history[-1]

        cx, cy = person.center
        px, py = previous

        distance = math.hypot(
            cx - px,
            cy - py
        )

        if distance < 3:
            movement = "Still"
        elif distance < 12:
            movement = "Moving"
        else:
            movement = "Fast movement"

        return movement, distance


class GestureRecognizer:
    def __init__(self):
        self.finger_data = [
            (8, 6, 5),
            (12, 10, 9),
            (16, 14, 13),
            (20, 18, 17)
        ]

    def distance(self, a, b):
        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2
        )

    def angle(self, a, b, c):
        ba = np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
        bc = np.array(c, dtype=np.float32) - np.array(b, dtype=np.float32)
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom <= 1e-7:
            return 180.0
        value = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return math.degrees(math.acos(value))

    def palm_scale(self, landmarks):
        scale = self.distance(landmarks[0], landmarks[9])
        return max(scale, 0.025)

    def finger_extended(self, landmarks, tip, pip, mcp):
        scale = self.palm_scale(landmarks)
        wrist = landmarks[0]
        tip_point = landmarks[tip]
        pip_point = landmarks[pip]
        mcp_point = landmarks[mcp]
        wrist_tip = self.distance(wrist, tip_point) / scale
        wrist_pip = self.distance(wrist, pip_point) / scale
        bend = self.angle(mcp_point, pip_point, tip_point)
        return wrist_tip > wrist_pip * 1.04 and bend > 145.0

    def thumb_extended(self, landmarks):
        scale = self.palm_scale(landmarks)
        bend = self.angle(landmarks[1], landmarks[2], landmarks[3])
        reach = self.distance(landmarks[0], landmarks[4]) / scale
        return reach > 1.05 and bend > 135.0

    def classify(self, landmarks):
        if len(landmarks) != 21:
            return "Hand detected"

        scale = self.palm_scale(landmarks)
        if scale < 0.025:
            return "Hand detected"

        thumb = self.thumb_extended(landmarks)
        fingers = [
            self.finger_extended(landmarks, tip, pip, mcp)
            for tip, pip, mcp in self.finger_data
        ]
        index, middle, ring, pinky = fingers
        count = sum(fingers) + int(thumb)
        pinch = self.distance(landmarks[4], landmarks[8]) / scale

        if pinch < 0.38 and index and not middle and not ring and not pinky:
            return "Pinch"

        if pinch < 0.48 and middle and ring and pinky and thumb:
            return "OK"

        if not any(fingers) and not thumb:
            return "Fist"

        if index and middle and not ring and not pinky:
            return "Peace"

        if index and not middle and not ring and not pinky:
            return "Pointing"

        if thumb and not index and not middle and not ring and not pinky:
            dy = landmarks[4][1] - landmarks[2][1]
            if dy < -0.07 * scale:
                return "Thumbs up"
            if dy > 0.07 * scale:
                return "Thumbs down"
            return "Thumbs up"

        if count >= 5:
            return "Open hand"

        if index and middle and ring and not pinky:
            return "Three fingers"

        if index and middle and ring and pinky and not thumb:
            return "Four fingers"

        return "Hand detected"


class HandStabilizer:
    def __init__(self):
        self.previous = []
        self.next_id = 1
        self.histories = {}
        self.last_seen = {}

    def center(self, hand):
        return (
            (hand.bbox[0] + hand.bbox[2]) * 0.5,
            (hand.bbox[1] + hand.bbox[3]) * 0.5
        )

    def match(self, hand):
        cx, cy = self.center(hand)

        best_id = None
        best_distance = 0.20

        for previous in self.previous:
            previous_id = previous["id"]

            if previous_id in [
                item["id"]
                for item in self.previous
                if False
            ]:
                continue

            px, py = previous["center"]

            distance = math.hypot(
                cx - px,
                cy - py
            )

            if distance < best_distance:
                if (
                    hand.handedness == "Unknown"
                    or
                    previous["handedness"] == "Unknown"
                    or
                    hand.handedness ==
                    previous["handedness"]
                ):
                    best_distance = distance
                    best_id = previous_id

        if best_id is None:
            best_id = self.next_id
            self.next_id += 1

        return best_id

    def stabilize(self, hands):
        now = time.time()
        current = []
        used_ids = set()

        for hand in hands:
            cx, cy = self.center(
                hand
            )

            candidates = []

            for previous in self.previous:
                previous_id = previous["id"]

                if previous_id in used_ids:
                    continue

                px, py = previous["center"]

                distance = math.hypot(
                    cx - px,
                    cy - py
                )

                if distance > 0.25:
                    continue

                handedness_match = (
                    hand.handedness ==
                    previous["handedness"]
                    or
                    hand.handedness ==
                    "Unknown"
                    or
                    previous["handedness"] ==
                    "Unknown"
                )

                if handedness_match:
                    candidates.append(
                        (
                            distance,
                            previous_id
                        )
                    )

            if candidates:
                candidates.sort(
                    key=lambda item: item[0]
                )
                hand_id = candidates[0][1]
            else:
                hand_id = self.next_id
                self.next_id += 1

            used_ids.add(
                hand_id
            )

            history = self.histories.setdefault(
                hand_id,
                deque(maxlen=5)
            )

            history.append(
                hand.gesture
            )

            valid_gestures = [
                value
                for value in history
                if value
                in {
                    "Open hand",
                    "Fist",
                    "Pointing",
                    "Peace",
                    "Thumbs up",
                    "Thumbs down",
                    "Pinch",
                    "OK",
                    "Three fingers",
                    "Four fingers",
                    "Hand gesture",
                    "Hand detected"
                }
            ]

            if valid_gestures:
                counts = Counter(
                    valid_gestures
                )

                stable_gesture = (
                    counts.most_common(1)[0][0]
                )
            else:
                stable_gesture = "Hand detected"

            if (
                stable_gesture
                in {
                    "Hand gesture",
                    "Hand detected"
                }
                and
                len(history) >= 3
            ):
                recent = list(
                    history
                )[-3:]

                if len(
                    set(recent)
                ) > 1:
                    stable_gesture = (
                        "Hand detected"
                    )

            hand.hand_id = hand_id
            hand.gesture = stable_gesture

            current.append(
                {
                    "id": hand_id,
                    "center": (cx, cy),
                    "handedness": hand.handedness
                }
            )

            self.last_seen[
                hand_id
            ] = now

        self.previous = current

        expired = [
            hand_id
            for hand_id, timestamp
            in self.last_seen.items()
            if now - timestamp > 2.0
        ]

        for hand_id in expired:
            self.last_seen.pop(
                hand_id,
                None
            )
            self.histories.pop(
                hand_id,
                None
            )

        return hands


class PostureAnalyzer:
    def point(
        self,
        keypoints,
        index
    ):
        if index >= len(keypoints):
            return None

        point = keypoints[index]

        if len(point) < 3:
            return None

        x, y, confidence = point

        if confidence < 0.30:
            return None

        return np.array(
            [x, y],
            dtype=np.float32
        )

    def midpoint(self, a, b):
        if a is None or b is None:
            return None

        return (a + b) / 2.0

    def angle_from_horizontal(
        self,
        a,
        b
    ):
        if a is None or b is None:
            return None

        dx = b[0] - a[0]
        dy = b[1] - a[1]

        angle = math.degrees(
            math.atan2(
                dy,
                dx
            )
        )

        while angle > 90:
            angle -= 180

        while angle < -90:
            angle += 180

        return abs(angle)

    def calculate(
        self,
        keypoints,
        bbox
    ):
        if not keypoints:
            return "Unknown"

        nose = self.point(
            keypoints,
            0
        )

        left_shoulder = self.point(
            keypoints,
            5
        )

        right_shoulder = self.point(
            keypoints,
            6
        )

        left_hip = self.point(
            keypoints,
            11
        )

        right_hip = self.point(
            keypoints,
            12
        )

        left_knee = self.point(
            keypoints,
            13
        )

        right_knee = self.point(
            keypoints,
            14
        )

        left_ankle = self.point(
            keypoints,
            15
        )

        right_ankle = self.point(
            keypoints,
            16
        )

        shoulder = self.midpoint(
            left_shoulder,
            right_shoulder
        )

        hip = self.midpoint(
            left_hip,
            right_hip
        )

        knee = self.midpoint(
            left_knee,
            right_knee
        )

        ankle = self.midpoint(
            left_ankle,
            right_ankle
        )

        if (
            shoulder is None
            or
            hip is None
        ):
            return "Unknown"

        x1, y1, x2, y2 = bbox

        width = max(
            1.0,
            x2 - x1
        )

        height = max(
            1.0,
            y2 - y1
        )

        torso_length = np.linalg.norm(
            hip - shoulder
        )

        if torso_length < 10:
            return "Unknown"

        torso_angle = (
            self.angle_from_horizontal(
                shoulder,
                hip
            )
        )

        if (
            torso_angle is not None
            and
            torso_angle < 38
            and
            width > height * 0.85
        ):
            return "Lying"

        if (
            knee is not None
            and
            torso_angle is not None
            and
            torso_angle > 55
        ):
            knee_hip_distance = (
                np.linalg.norm(
                    knee - hip
                )
            )

            if (
                knee_hip_distance
                < torso_length * 1.55
            ):
                return "Sitting"

        if (
            torso_angle is not None
            and
            torso_angle > 62
        ):
            return "Standing"

        if (
            nose is not None
            and
            shoulder is not None
        ):
            head_distance = (
                np.linalg.norm(
                    nose - shoulder
                )
            )

            if (
                torso_length > 0
                and
                head_distance
                < torso_length * 0.85
                and
                torso_angle is not None
                and
                torso_angle > 55
            ):
                return "Standing"

        if width > height * 1.15:
            return "Lying"

        return "Unknown"


class HandLandmarkerEngine:
    def __init__(self):
        self.available = False
        self.error = ""
        self.landmarker = None
        self.lock = threading.Lock()
        self.recognizer = GestureRecognizer()
        self.stabilizer = HandStabilizer()
        self.initialized = False
        self.initializing = False

        if not MEDIAPIPE_AVAILABLE:
            self.error = (
                "MediaPipe is not available."
            )
            self.initialized = True

    def initialize(self):
        with self.lock:
            if self.initialized:
                return self.available

            if self.initializing:
                return False

            self.initializing = True

        try:
            self.ensure_model()
            self.create_landmarker()

            with self.lock:
                self.initialized = True
                self.initializing = False

            return self.available

        except Exception as exc:
            with self.lock:
                self.error = (
                    f"Hand engine unavailable: {exc}"
                )
                self.initialized = True
                self.initializing = False

            return False

    def ensure_model(self):
        if os.path.exists(
            HAND_MODEL_PATH
        ):
            return

        temporary_path = (
            HAND_MODEL_PATH +
            ".download"
        )

        try:
            urllib.request.urlretrieve(
                HAND_MODEL_URL,
                temporary_path
            )

            os.replace(
                temporary_path,
                HAND_MODEL_PATH
            )

        except Exception:
            try:
                if os.path.exists(
                    temporary_path
                ):
                    os.remove(
                        temporary_path
                    )
            except Exception:
                pass

            raise

    def create_landmarker(self):
        base_options = python.BaseOptions(
            model_asset_path=HAND_MODEL_PATH
        )

        options = (
            vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=(
                    vision.RunningMode.IMAGE
                ),
                num_hands=2,
                min_hand_detection_confidence=0.15,
                min_hand_presence_confidence=0.15,
                min_tracking_confidence=0.15
            )
        )

        self.landmarker = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

        self.available = True

    def _detect_image(self, image_bgr):
        if self.landmarker is None:
            return []
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        with self.lock:
            result = self.landmarker.detect(image)
        detected = []
        if not result.hand_landmarks:
            return detected
        for index, landmarks in enumerate(result.hand_landmarks):
            points = [
                (float(point.x), float(point.y), float(point.z))
                for point in landmarks
            ]
            handedness = "Unknown"
            handedness_score = 1.0
            if result.handedness and index < len(result.handedness) and result.handedness[index]:
                item = result.handedness[index][0]
                handedness = item.category_name or "Unknown"
                handedness_score = float(item.score or 0.0)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1 = max(0.0, min(xs))
            y1 = max(0.0, min(ys))
            x2 = min(1.0, max(xs))
            y2 = min(1.0, max(ys))
            if x2 - x1 < 0.004 or y2 - y1 < 0.004:
                continue
            detected.append((points, handedness, handedness_score, (x1, y1, x2, y2)))
        return detected

    def _make_state(self, points, handedness, score, bbox, hand_id):
        gesture = self.recognizer.classify(points)
        return HandState(
            hand_id=hand_id,
            handedness=handedness,
            landmarks=points,
            bbox=bbox,
            gesture=gesture,
            confidence=max(0.0, min(1.0, score))
        )

    def detect(self, frame, confidence=0.25, person_boxes=None):
        if not self.initialized or not self.available or self.landmarker is None:
            return []
        try:
            h, w = frame.shape[:2]
            candidates = []

            full_scale = min(1.0, 1280.0 / max(w, 1))
            full = frame
            if full_scale != 1.0:
                full = cv2.resize(frame, (int(w * full_scale), int(h * full_scale)), interpolation=cv2.INTER_LINEAR)
            for points, handedness, score, bbox in self._detect_image(full):
                mapped = []
                for x, y, z in points:
                    mapped.append((x, y, z))
                candidates.append(self._make_state(mapped, handedness, score, bbox, len(candidates) + 1))

            for pb in (person_boxes or [])[:6]:
                x1, y1, x2, y2 = pb
                px1 = max(0, int(x1 * w))
                py1 = max(0, int(y1 * h))
                px2 = min(w, int(x2 * w))
                py2 = min(h, int(y2 * h))
                bw = max(1, px2 - px1)
                bh = max(1, py2 - py1)
                pad_x = int(bw * 0.18)
                pad_y = int(bh * 0.12)
                cx1 = max(0, px1 - pad_x)
                cy1 = max(0, py1 - pad_y)
                cx2 = min(w, px2 + pad_x)
                cy2 = min(h, py2 + pad_y)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                ch, cw = crop.shape[:2]
                scale = min(1.0, 960.0 / max(cw, ch))
                if scale != 1.0:
                    proc = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))), interpolation=cv2.INTER_LINEAR)
                else:
                    proc = crop
                for points, handedness, score, bbox in self._detect_image(proc):
                    mapped = []
                    for x, y, z in points:
                        ox = (x * proc.shape[1] / max(scale, 1e-6) + cx1) / w
                        oy = (y * proc.shape[0] / max(scale, 1e-6) + cy1) / h
                        mapped.append((float(ox), float(oy), float(z)))
                    xs = [p[0] for p in mapped]
                    ys = [p[1] for p in mapped]
                    mb = (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))
                    candidates.append(self._make_state(mapped, handedness, score, mb, len(candidates) + 1))

            unique = []
            for hand in candidates:
                wrist = hand.landmarks[0]
                duplicate = False
                for existing in unique:
                    ew = existing.landmarks[0]
                    if math.hypot(wrist[0] - ew[0], wrist[1] - ew[1]) < 0.075:
                        duplicate = True
                        if hand.confidence > existing.confidence:
                            unique.remove(existing)
                        else:
                            break
                if not duplicate or not any(
                    math.hypot(wrist[0] - e.landmarks[0][0], wrist[1] - e.landmarks[0][1]) < 0.075 for e in unique
                ):
                    unique.append(hand)

            unique = unique[:2]
            for i, hand in enumerate(unique, 1):
                hand.hand_id = i
            return self.stabilizer.stabilize(unique)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return []


class EventEngine:
    def __init__(self):
        self.events = deque(
            maxlen=80
        )

        self.last_values = {}
        self.last_event_times = {}

    def clear(self):
        self.events.clear()
        self.last_values.clear()
        self.last_event_times.clear()

    def add(
        self,
        category,
        value
    ):
        now = time.time()

        previous = self.last_values.get(
            category
        )

        if previous == value:
            return

        last_time = self.last_event_times.get(
            category,
            0
        )

        if now - last_time < 1.5:
            self.last_values[category] = value
            return

        self.last_values[category] = value
        self.last_event_times[category] = now

        self.events.append(
            Event(
                timestamp=now,
                text=value,
                category=category
            )
        )

    def formatted(self):
        if not self.events:
            return "No events yet."

        lines = []

        for event in reversed(
            self.events
        ):
            stamp = time.strftime(
                "%H:%M:%S",
                time.localtime(
                    event.timestamp
                )
            )

            lines.append(
                f"[{stamp}] {event.text}"
            )

        return "\n".join(lines)


OBJECT_GPU_MODEL = None
POSE_GPU_MODEL = None
MODEL_LOCK = threading.Lock()
MODEL_ERROR = ""


def get_gpu_models():
    global OBJECT_GPU_MODEL
    global POSE_GPU_MODEL
    global MODEL_ERROR

    with MODEL_LOCK:
        try:
            if OBJECT_GPU_MODEL is None:
                OBJECT_GPU_MODEL = YOLO(
                    OBJECT_MODEL
                )

                if torch is not None and torch.cuda.is_available():
                    OBJECT_GPU_MODEL.to("cuda")
                else:
                    OBJECT_GPU_MODEL.to("cpu")

            if POSE_GPU_MODEL is None:
                POSE_GPU_MODEL = YOLO(
                    POSE_MODEL
                )

                if torch is not None and torch.cuda.is_available():
                    POSE_GPU_MODEL.to("cuda")
                else:
                    POSE_GPU_MODEL.to("cpu")

        except Exception as exc:
            MODEL_ERROR = (
                f"{type(exc).__name__}: {exc}"
            )
            raise

    return (
        OBJECT_GPU_MODEL,
        POSE_GPU_MODEL
    )


def run_gpu_inference(
    frame,
    object_confidence,
    pose_confidence,
    mode,
    groups,
    show_pose,
    run_pose
):
    object_model, pose_model = (
        get_gpu_models()
    )

    allowed = set()

    for group in groups:
        allowed.update(
            OBJECT_GROUPS.get(
                group,
                []
            )
        )

    allowed.add("person")
    allowed.add("cell phone")

    object_output = []
    object_error = ""

    try:
        object_result = object_model.predict(
            source=frame,
            conf=float(
                object_confidence
            ),
            iou=0.45,
            imgsz=416,
            max_det=40,
            device=("cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"),
            verbose=False
        )[0]

        if object_result.boxes is not None:
            names = object_model.names

            for box in object_result.boxes:
                cls_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                label = names[cls_id]

                if label not in allowed:
                    continue

                required_confidence = (
                    0.25
                    if label == "person"
                    else max(
                        0.25,
                        float(
                            object_confidence
                        )
                    )
                )

                if confidence < required_confidence:
                    continue

                coordinates = [
                    int(value)
                    for value in box.xyxy[
                        0
                    ].detach().cpu().tolist()
                ]

                x1, y1, x2, y2 = coordinates

                center = (
                    (x1 + x2) // 2,
                    (y1 + y2) // 2
                )

                area = max(
                    1,
                    (x2 - x1) *
                    (y2 - y1)
                )

                object_output.append(
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

    except Exception as exc:
        object_error = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    pose_output = []
    pose_error = ""

    if (
        show_pose
        and
        run_pose
    ):
        try:
            pose_result = pose_model.predict(
                source=frame,
                conf=float(
                    pose_confidence
                ),
                iou=0.45,
                imgsz=416,
                max_det=12,
                device="cpu",
                verbose=False
            )[0]

            if (
                pose_result.keypoints is not None
                and
                pose_result.boxes is not None
            ):
                pose_boxes = (
                    pose_result.boxes.xyxy
                    .detach()
                    .cpu()
                    .tolist()
                )

                pose_keypoints = (
                    pose_result.keypoints.data
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

                for index in range(
                    min(
                        len(pose_boxes),
                        len(pose_keypoints)
                    )
                ):
                    box = [
                        float(value)
                        for value
                        in pose_boxes[index]
                    ]

                    keypoints = [
                        [
                            float(point[0]),
                            float(point[1]),
                            float(point[2])
                        ]
                        for point
                        in pose_keypoints[index]
                    ]

                    pose_output.append(
                        {
                            "bbox": box,
                            "keypoints": keypoints
                        }
                    )

        except Exception as exc:
            pose_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    return (
        object_output,
        pose_output,
        object_error,
        pose_error
    )


class FaceLandmarkerEngine:
    def __init__(self):
        self.available = False
        self.error = ""
        self.landmarker = None
        self.lock = threading.Lock()
        self.initialized = False
        self.initializing = False

    def initialize(self):
        with self.lock:
            if self.initialized:
                return self.available
            if self.initializing:
                return False
            self.initializing = True
        try:
            if not MEDIAPIPE_AVAILABLE:
                raise RuntimeError("MediaPipe is not available")
            if not os.path.exists(FACE_MODEL_PATH):
                temp = FACE_MODEL_PATH + ".download"
                urllib.request.urlretrieve(FACE_MODEL_URL, temp)
                os.replace(temp, FACE_MODEL_PATH)
            base = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=4,
                min_face_detection_confidence=0.25,
                min_face_presence_confidence=0.25,
                min_tracking_confidence=0.15,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            self.available = True
            self.initialized = True
            self.initializing = False
            return True
        except Exception as exc:
            self.error = f"Face engine unavailable: {exc}"
            self.initialized = True
            self.initializing = False
            return False

    def _expression(self, blendshapes):
        scores = {}
        for item in blendshapes or []:
            name = getattr(item, "category_name", "") or ""
            score = float(getattr(item, "score", 0.0) or 0.0)
            scores[name] = score
        smile = (scores.get("mouthSmileLeft",0)+scores.get("mouthSmileRight",0))/2
        frown = (scores.get("mouthFrownLeft",0)+scores.get("mouthFrownRight",0))/2
        brow = scores.get("browInnerUp",0)
        jaw = scores.get("jawOpen",0)
        wide = (scores.get("eyeWideLeft",0)+scores.get("eyeWideRight",0))/2
        squint = (scores.get("eyeSquintLeft",0)+scores.get("eyeSquintRight",0))/2
        if smile > 0.45: return "Smiling"
        if jaw > 0.45 and wide > 0.30: return "Surprised"
        if frown > 0.40 and brow > 0.25: return "Sad / Concerned"
        if brow > 0.50: return "Brows raised"
        if squint > 0.45: return "Eyes narrowed"
        return "Neutral / Unclear"

    def detect(self, frame):
        if not self.initialized or not self.available or self.landmarker is None:
            return []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            with self.lock:
                result = self.landmarker.detect(image)
            output=[]
            for i, landmarks in enumerate(result.face_landmarks or []):
                xs=[float(p.x) for p in landmarks]; ys=[float(p.y) for p in landmarks]
                bbox=(max(0,min(xs)), max(0,min(ys)), min(1,max(xs)), min(1,max(ys)))
                blend=[]
                if result.face_blendshapes and i < len(result.face_blendshapes):
                    blend=result.face_blendshapes[i]
                expression=self._expression(blend)
                output.append(FaceState(i+1,bbox,expression,1.0,[(float(p.x),float(p.y),float(p.z)) for p in landmarks]))
            return output
        except Exception as exc:
            self.error=f"{type(exc).__name__}: {exc}"
            return []


class LifeVisionEngine:
    def __init__(self):
        self.lock = threading.RLock()

        self.latest_output = None
        self.latest_overlay_payload = {
            "width": 640,
            "height": 480,
            "mirror": False,
            "objects": [],
            "people": [],
            "hands": []
        }
        self.latest_snapshot = Snapshot()

        self.latest_frame = None
        self.latest_frame_time = 0.0

        self.frame_id = 0
        self.camera_frames = 0
        self.camera_start = time.time()

        self.process_fps_target = 4.0
        self.last_ai_time = 0.0

        self.ai_times = deque(
            maxlen=20
        )

        self.mode = MODE_FULL

        self.object_confidence = 0.35
        self.pose_confidence = 0.35
        self.hand_confidence = 0.25

        self.show_boxes = True
        self.show_pose = True
        self.show_hands = True
        self.show_labels = True
        self.show_hud = True
        self.mirror = False

        self.groups = [
            "People",
            "Animals",
            "Vehicles",
            "Indoor",
            "Food",
            "Sports"
        ]

        self.object_error = ""
        self.pose_error = ""
        self.gpu_error = ""

        self.object_tracker = Tracker(
            max_distance=150
        )

        self.person_tracker = Tracker(
            max_distance=180
        )

        self.motion = MotionAnalyzer()
        self.posture = PostureAnalyzer()
        self.gesture = GestureRecognizer()
        self.events = EventEngine()
        self.hand_engine = HandLandmarkerEngine()
        self.face_engine = FaceLandmarkerEngine()

        self.people = {}
        self.last_objects = []
        self.last_hands = []

        self.render_objects = []
        self.render_people = []
        self.render_hands = []
        self.render_faces = []

        self.render_scene = (
            "Waiting for camera..."
        )

        self.render_narrative = (
            "Start the camera to begin LifeVision."
        )

        self.processing_lock = threading.Lock()

        self.worker_condition = (
            threading.Condition()
        )

        self.worker_running = False

        self.worker_thread = threading.Thread(
            target=self.ai_worker,
            daemon=True
        )

        self.last_ai_duration = 0.0
        self.ai_cycle = 0
        self.last_pose_cycle = -1
        self.first_frame_received = False

    def update_config(
        self,
        mode,
        process_fps,
        object_confidence,
        pose_confidence,
        hand_confidence,
        show_boxes,
        show_pose,
        show_hands,
        show_labels,
        show_hud,
        mirror,
        groups
    ):
        with self.lock:
            self.mode = (
                mode
                or MODE_OBJECTS
            )

            if self.mode not in {
                MODE_OBJECTS,
                MODE_GESTURE
            }:
                self.mode = MODE_FULL

            self.process_fps_target = max(
                1.0,
                min(
                    10.0,
                    float(
                        process_fps
                        or 4.0
                    )
                )
            )

            self.object_confidence = max(
                0.10,
                min(
                    0.90,
                    float(
                        object_confidence
                        or 0.35
                    )
                )
            )

            self.pose_confidence = max(
                0.10,
                min(
                    0.90,
                    float(
                        pose_confidence
                        or 0.35
                    )
                )
            )

            self.hand_confidence = max(
                0.10,
                min(
                    0.90,
                    float(
                        hand_confidence
                        or 0.45
                    )
                )
            )

            self.show_boxes = bool(
                show_boxes
            )

            self.show_pose = bool(
                show_pose
            )

            self.show_hands = bool(
                show_hands
            )

            self.show_labels = bool(
                show_labels
            )

            self.show_hud = bool(
                show_hud
            )

            self.mirror = bool(
                mirror
            )

            self.groups = list(
                groups
                or ["People"]
            )

    def submit(self, frame):
        if frame is None:
            return

        if not isinstance(
            frame,
            np.ndarray
        ):
            return

        if frame.ndim != 3:
            return

        if frame.shape[2] != 3:
            return

        with self.lock:
            self.camera_frames += 1

            current_frame = (
                np.ascontiguousarray(
                    frame
                )
            )

            current_frame = cv2.cvtColor(
                current_frame,
                cv2.COLOR_RGB2BGR
            )

            if self.mirror:
                current_frame = cv2.flip(
                    current_frame,
                    1
                )

            self.latest_frame = (
                current_frame
            )

            self.latest_frame_time = (
                time.time()
            )

            if not self.first_frame_received:
                self.first_frame_received = True
                self.last_ai_time = 0.0

        with self.worker_condition:
            self.worker_condition.notify()

    def ai_worker(self):
        while self.worker_running:
            try:
                with self.lock:
                    has_frame = (
                        self.latest_frame
                        is not None
                    )

                    target_fps = (
                        self.process_fps_target
                    )

                    last_ai_time = (
                        self.last_ai_time
                    )

                if not has_frame:
                    with self.worker_condition:
                        self.worker_condition.wait(
                            timeout=0.05
                        )
                    continue

                interval = (
                    1.0 /
                    max(
                        1.0,
                        target_fps
                    )
                )

                now = time.time()

                wait_time = (
                    interval -
                    (
                        now -
                        last_ai_time
                    )
                )

                if last_ai_time <= 0:
                    wait_time = 0.0

                if wait_time > 0:
                    with self.worker_condition:
                        self.worker_condition.wait(
                            timeout=min(
                                wait_time,
                                0.025
                            )
                        )
                    continue

                self.run_latest_ai_cycle()

            except Exception as exc:
                with self.lock:
                    self.gpu_error = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                time.sleep(0.05)

    def run_latest_ai_cycle(self):
        with self.lock:
            if self.latest_frame is None:
                return

            frame = (
                self.latest_frame.copy()
            )

            now = time.time()

            interval = (
                1.0 /
                max(
                    1.0,
                    self.process_fps_target
                )
            )

            if (
                self.last_ai_time > 0
                and
                now - self.last_ai_time
                < interval
            ):
                return

            mode = self.mode

            object_confidence = (
                self.object_confidence
            )

            pose_confidence = (
                self.pose_confidence
            )

            groups = list(
                self.groups
            )

            show_pose = self.show_pose

            hand_confidence = (
                self.hand_confidence
            )

            show_hands = (
                self.show_hands
            )

            self.frame_id += 1

            current_frame_id = (
                self.frame_id
            )

            self.ai_cycle += 1

            run_pose = bool(show_pose)

            self.last_ai_time = now

        started = time.time()

        if (
            show_hands
            and
            not self.hand_engine.initialized
        ):
            self.hand_engine.initialize()

        try:
            gpu_result = run_gpu_inference(
                frame,
                object_confidence,
                pose_confidence,
                mode,
                groups,
                show_pose,
                run_pose
            )
        except Exception as exc:
            gpu_result = (
                [],
                [],
                f"{type(exc).__name__}: {exc}",
                f"{type(exc).__name__}: {exc}"
            )

        try:
            output, snapshot, overlay_payload = (
                self.process_ai_result(
                    frame,
                    gpu_result,
                    current_frame_id,
                    show_hands,
                    hand_confidence,
                    run_pose
                )
            )

            duration = max(
                0.0001,
                time.time() - started
            )

            with self.lock:
                self.last_ai_duration = (
                    duration
                )

                self.ai_times.append(
                    duration
                )

                self.latest_snapshot = (
                    snapshot
                )

                self.render_objects = list(
                    snapshot.objects
                )

                self.render_people = list(
                    snapshot.people
                )

                self.render_hands = list(
                    snapshot.hands
                )
                self.render_faces = list(
                    snapshot.faces
                )

                self.render_scene = (
                    snapshot.scene
                )

                self.render_narrative = (
                    snapshot.narrative
                )

                self.latest_output = (
                    output
                )

                self.latest_overlay_payload = overlay_payload

                if run_pose:
                    self.last_pose_cycle = (
                        self.ai_cycle
                    )

                self.gpu_error = ""

        except Exception as exc:
            with self.lock:
                self.gpu_error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    def process_ai_result(
        self,
        frame,
        gpu_result,
        current_frame_id,
        show_hands,
        hand_confidence,
        run_pose
    ):
        if (
            not isinstance(
                gpu_result,
                tuple
            )
            or
            len(gpu_result) != 4
        ):
            gpu_result = (
                [],
                [],
                "Invalid GPU inference result.",
                "Invalid GPU inference result."
            )

        (
            gpu_objects,
            gpu_pose,
            object_error,
            pose_error
        ) = gpu_result

        self.object_error = (
            object_error
            or ""
        )

        if run_pose:
            self.pose_error = (
                pose_error
                or ""
            )

        objects = self.detect_objects(
            gpu_objects
        )

        people = (
            self.synchronize_people(
                objects
            )
        )

        if run_pose:
            self.detect_pose(
                gpu_pose,
                people
            )

        hands = self.detect_hands(
            frame,
            people,
            hand_confidence
        ) if show_hands else []

        if not self.face_engine.initialized:
            self.face_engine.initialize()
        faces = self.detect_faces(frame, people)

        self.analyze_body_language(
            people,
            objects,
            hands,
            frame.shape[1],
            frame.shape[0]
        )

        scene = self.build_scene(
            objects,
            people,
            hands
        )

        narrative = self.build_narrative(
            objects,
            people,
            hands,
            scene
        )

        self.update_events(
            scene,
            people,
            hands
        )

        if self.ai_times:
            average_processing = (
                sum(
                    self.ai_times
                )
                /
                len(
                    self.ai_times
                )
            )

            stable_ai_fps = (
                1.0 /
                max(
                    0.0001,
                    average_processing
                )
            )
        else:
            stable_ai_fps = 0.0

        now = time.time()

        runtime_error = ""

        if self.gpu_error:
            runtime_error = (
                self.gpu_error
            )

        snapshot = Snapshot(
            timestamp=now,
            frame_id=current_frame_id,
            objects=list(objects),
            people=list(people),
            hands=list(hands),
            faces=list(faces),
            scene=scene,
            narrative=narrative,
            fps=self.calculate_camera_fps(),
            ai_fps=stable_ai_fps,
            hand_available=(
                self.hand_engine.available
            ),
            runtime_error=runtime_error
        )

        self.last_objects = list(
            objects
        )

        self.last_hands = list(
            hands
        )

        output = self.render(
            frame,
            objects,
            people,
            hands,
            scene,
            narrative,
            faces
        )

        overlay_payload = self.build_overlay_payload(
            frame,
            objects,
            people,
            hands,
            faces
        )

        return (
            output,
            snapshot,
            overlay_payload
        )

    def get_output(self):
        with self.lock:
            if self.latest_frame is not None:
                output = self.render(
                    self.latest_frame,
                    self.render_objects,
                    self.render_people,
                    self.render_hands,
                    self.render_scene,
                    self.render_narrative,
                    self.render_faces
                )
            elif self.latest_output is not None:
                output = (
                    self.latest_output.copy()
                )
            else:
                output = np.zeros(
                    (
                        480,
                        640,
                        3
                    ),
                    dtype=np.uint8
                )

                cv2.putText(
                    output,
                    "Waiting for camera...",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            return (
                output,
                self.format_snapshot(
                    self.latest_snapshot
                )
            )

    def detect_objects(
        self,
        gpu_objects
    ):
        if isinstance(
            gpu_objects,
            dict
        ):
            self.object_error = (
                gpu_objects.get(
                    "error",
                    "Unknown object inference error."
                )
            )

            return []

        detections = []

        for item in gpu_objects:
            detections.append(
                {
                    "label": item["label"],
                    "confidence": float(
                        item["confidence"]
                    ),
                    "bbox": tuple(
                        item["bbox"]
                    ),
                    "center": tuple(
                        item["center"]
                    ),
                    "area": float(
                        item["area"]
                    )
                }
            )

        tracked = (
            self.object_tracker.update(
                detections
            )
        )

        objects = [
            ObjectState(
                label=item["label"],
                confidence=item[
                    "confidence"
                ],
                bbox=item["bbox"],
                track_id=item[
                    "track_id"
                ],
                center=item["center"],
                area=item["area"]
            )
            for item in tracked
        ]

        return objects

    def synchronize_people(
        self,
        objects
    ):
        person_objects = [
            obj
            for obj in objects
            if obj.label == "person"
        ]

        detections = []

        for obj in person_objects:
            detections.append(
                {
                    "bbox": obj.bbox,
                    "center": obj.center,
                    "confidence": obj.confidence
                }
            )

        tracked = (
            self.person_tracker.update(
                detections
            )
        )

        current = {}

        for item in tracked:
            person_id = item[
                "track_id"
            ]

            previous = self.people.get(
                person_id
            )

            if previous is None:
                person = PersonState(
                    person_id=person_id,
                    bbox=item["bbox"],
                    center=item["center"],
                    confidence=item[
                        "confidence"
                    ],
                    last_center=item[
                        "center"
                    ]
                )
            else:
                person = previous

                person.last_center = (
                    person.center
                )

                person.center = (
                    item["center"]
                )

                person.bbox = (
                    item["bbox"]
                )

                person.confidence = (
                    item["confidence"]
                )

                person.keypoints = []
                person.gestures = []
                person.hands = []

            movement, velocity = (
                self.motion.calculate(
                    person
                )
            )

            person.movement = movement
            person.velocity = velocity

            person.history.append(
                person.center
            )

            person.last_update = (
                time.time()
            )

            current[
                person_id
            ] = person

        self.people = current

        return list(
            current.values()
        )

    def detect_pose(
        self,
        pose_output,
        people
    ):
        if (
            not people
            or
            self.mode == MODE_OBJECTS
            or
            not self.show_pose
        ):
            return

        if isinstance(
            pose_output,
            dict
        ):
            self.pose_error = (
                pose_output.get(
                    "error",
                    "Unknown pose inference error."
                )
            )

            return

        pose_items = []

        for item in pose_output:
            box = item["bbox"]
            keypoints = item[
                "keypoints"
            ]

            pose_items.append(
                (
                    box,
                    keypoints
                )
            )

        used_people = set()

        for box, keypoints in pose_items:
            px1, py1, px2, py2 = box

            best_person = None
            best_iou = 0.0

            for person in people:
                if (
                    person.person_id
                    in used_people
                ):
                    continue

                iou = self.iou(
                    (
                        px1,
                        py1,
                        px2,
                        py2
                    ),
                    person.bbox
                )

                if iou > best_iou:
                    best_iou = iou
                    best_person = person

            if (
                best_person is None
                or
                best_iou < 0.15
            ):
                continue

            used_people.add(
                best_person.person_id
            )

            kp = [
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2])
                )
                for point in keypoints
            ]

            best_person.keypoints = kp

            best_person.posture = (
                self.posture.calculate(
                    kp,
                    best_person.bbox
                )
            )

    def iou(
        self,
        a,
        b
    ):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(
            ax1,
            bx1
        )

        iy1 = max(
            ay1,
            by1
        )

        ix2 = min(
            ax2,
            bx2
        )

        iy2 = min(
            ay2,
            by2
        )

        iw = max(
            0,
            ix2 - ix1
        )

        ih = max(
            0,
            iy2 - iy1
        )

        intersection = (
            iw * ih
        )

        if intersection <= 0:
            return 0.0

        area_a = (
            max(
                0,
                ax2 - ax1
            )
            *
            max(
                0,
                ay2 - ay1
            )
        )

        area_b = (
            max(
                0,
                bx2 - bx1
            )
            *
            max(
                0,
                by2 - by1
            )
        )

        union = (
            area_a
            +
            area_b
            -
            intersection
        )

        if union <= 0:
            return 0.0

        return (
            intersection /
            union
        )

    def detect_hands(
        self,
        frame,
        people,
        confidence
    ):
        if (
            not self.show_hands
        ):
            return []

        person_boxes = [
            person.bbox
            for person in people
        ]

        hands = self.hand_engine.detect(
            frame,
            confidence,
            person_boxes
        )

        if not hands:
            for person in people:
                person.hands = []
                person.gestures = []
            return []

        height, width = (
            frame.shape[:2]
        )

        for person in people:
            person.hands = []
            person.gestures = []

        for hand in hands:
            cx = (
                hand.bbox[0]
                +
                hand.bbox[2]
            ) * 0.5 * width

            cy = (
                hand.bbox[1]
                +
                hand.bbox[3]
            ) * 0.5 * height

            nearest = None
            nearest_distance = (
                float("inf")
            )

            for person in people:
                px, py = (
                    person.center
                )

                expanded_bbox = (
                    person.bbox[0] - 80,
                    person.bbox[1] - 80,
                    person.bbox[2] + 80,
                    person.bbox[3] + 80
                )

                inside = (
                    expanded_bbox[0]
                    <= cx
                    <= expanded_bbox[2]
                    and
                    expanded_bbox[1]
                    <= cy
                    <= expanded_bbox[3]
                )

                distance = math.hypot(
                    cx - px,
                    cy - py
                )

                if inside:
                    if distance < nearest_distance:
                        nearest = person
                        nearest_distance = distance

                elif distance < nearest_distance:
                    nearest = person
                    nearest_distance = distance

            if nearest is not None:
                nearest.hands.append(
                    hand
                )

                if (
                    hand.gesture
                    not in nearest.gestures
                ):
                    nearest.gestures.append(
                        hand.gesture
                    )

        return hands

    def detect_faces(self, frame, people):
        faces = self.face_engine.detect(frame)
        for person in people:
            person.facial_expression = "Face not visible"
            person.face_bbox = None
        h,w=frame.shape[:2]
        for face in faces:
            cx=((face.bbox[0]+face.bbox[2])*0.5)*w
            cy=((face.bbox[1]+face.bbox[3])*0.5)*h
            nearest=None; best=10**9
            for person in people:
                x1,y1,x2,y2=person.bbox
                px,py=person.center
                if x1-50 <= cx <= x2+50 and y1-50 <= cy <= y2+50:
                    d=math.hypot(cx-px,cy-py)
                    if d<best: nearest=person; best=d
            if nearest is not None:
                nearest.facial_expression=face.expression
                nearest.face_bbox=face.bbox
        return faces

    def build_object_scene(
        self,
        objects,
        people
    ):
        animal_count = sum(
            1
            for obj in objects
            if obj.label
            in OBJECT_GROUPS["Animals"]
        )

        vehicle_count = sum(
            1
            for obj in objects
            if obj.label
            in OBJECT_GROUPS["Vehicles"]
        )

        non_people = [
            obj
            for obj in objects
            if obj.label != "person"
        ]

        if (
            not people
            and
            not non_people
        ):
            return "No recognized objects"

        if (
            len(people) > 1
            and
            animal_count > 0
        ):
            return "Multiple people and animals"

        if (
            len(people) > 1
            and
            non_people
        ):
            return "Multiple people and objects"

        if len(people) > 1:
            return "Multiple people"

        if (
            len(people) == 1
            and
            animal_count > 0
        ):
            return "Person and animal"

        if (
            len(people) == 1
            and
            vehicle_count > 0
        ):
            return "Person and vehicle"

        if (
            len(people) == 1
            and
            non_people
        ):
            return "Person and objects"

        if len(people) == 1:
            return "Single person"

        if animal_count > 0:
            return "Animals detected"

        return "Objects detected"

    def build_object_narrative(
        self,
        objects,
        people,
        scene
    ):
        parts = []

        if len(people) == 0:
            parts.append(
                "No person is currently confirmed."
            )
        elif len(people) == 1:
            parts.append(
                "1 person is confirmed."
            )
        else:
            parts.append(
                f"{len(people)} people are confirmed."
            )

        counts = {}

        for obj in objects:
            if obj.label == "person":
                continue

            counts[obj.label] = (
                counts.get(
                    obj.label,
                    0
                )
                +
                1
            )

        if counts:
            object_text = []

            for label, count in sorted(
                counts.items(),
                key=lambda item: item[0]
            ):
                object_text.append(
                    f"{count} {label}"
                    f"{'s' if count != 1 else ''}"
                )

            parts.append(
                "Confirmed objects: "
                +
                ", ".join(
                    object_text
                )
                +
                "."
            )
        else:
            parts.append(
                "No non-person object is currently confirmed."
            )

        parts.append(
            f"Object scene: {scene}."
        )

        return " ".join(parts)

    def analyze_body_language(self, people, objects, hands, width, height):
        phones = [
            obj
            for obj in objects
            if obj.label == "cell phone"
        ]

        def point_from_keypoints(kp, index, minimum=0.30):
            if index >= len(kp) or len(kp[index]) < 3:
                return None
            if float(kp[index][2]) < minimum:
                return None
            return np.array(
                [float(kp[index][0]), float(kp[index][1])],
                dtype=np.float32
            )

        def hand_center(hand):
            return np.array(
                [
                    ((hand.bbox[0] + hand.bbox[2]) * 0.5) * width,
                    ((hand.bbox[1] + hand.bbox[3]) * 0.5) * height
                ],
                dtype=np.float32
            )

        def hand_position(center, nose, shoulder, hip, person_bbox):
            if nose is not None:
                face_distance = float(np.linalg.norm(center - nose))
                face_scale = max(35.0, abs(person_bbox[2] - person_bbox[0]) * 0.18)
                if face_distance < face_scale * 1.8:
                    return "near face"

            if shoulder is not None:
                shoulder_distance = float(np.linalg.norm(center - shoulder))
                shoulder_scale = max(45.0, abs(person_bbox[2] - person_bbox[0]) * 0.28)
                if shoulder_distance < shoulder_scale * 1.8:
                    return "near upper body"

            if hip is not None:
                hip_distance = float(np.linalg.norm(center - hip))
                hip_scale = max(50.0, abs(person_bbox[2] - person_bbox[0]) * 0.30)
                if hip_distance < hip_scale * 1.7:
                    return "near waist"

            if center[1] < person_bbox[1] + abs(person_bbox[3] - person_bbox[1]) * 0.20:
                return "above head"

            if center[0] < person_bbox[0] + abs(person_bbox[2] - person_bbox[0]) * 0.35:
                return "left side"

            if center[0] > person_bbox[0] + abs(person_bbox[2] - person_bbox[0]) * 0.65:
                return "right side"

            return "in front of body"

        for person in people:
            person.arm_state = "Arms not clearly visible"
            person.leg_state = "Legs not clearly visible"
            person.hand_activity = "No hand activity confirmed"
            person.body_language = "Body position unclear"
            person.phone_status = "No phone interaction confirmed"

            kp = person.keypoints
            nose = None
            ls = rs = le = re = lw = rw = lh = rh = lk = rk = la = ra = None

            if len(kp) >= 17:
                nose = point_from_keypoints(kp, 0)
                ls = point_from_keypoints(kp, 5)
                rs = point_from_keypoints(kp, 6)
                le = point_from_keypoints(kp, 7)
                re = point_from_keypoints(kp, 8)
                lw = point_from_keypoints(kp, 9)
                rw = point_from_keypoints(kp, 10)
                lh = point_from_keypoints(kp, 11)
                rh = point_from_keypoints(kp, 12)
                lk = point_from_keypoints(kp, 13)
                rk = point_from_keypoints(kp, 14)
                la = point_from_keypoints(kp, 15)
                ra = point_from_keypoints(kp, 16)

            shoulder = None if ls is None or rs is None else (ls + rs) / 2.0
            hip = None if lh is None or rh is None else (lh + rh) / 2.0
            shoulder_width = 0.0 if ls is None or rs is None else float(np.linalg.norm(ls - rs))
            person_width = max(30.0, abs(person.bbox[2] - person.bbox[0]))
            torso_scale = max(30.0, shoulder_width, person_width * 0.18)

            left_raised = (
                lw is not None
                and ls is not None
                and lw[1] < ls[1] - torso_scale * 0.12
            )
            right_raised = (
                rw is not None
                and rs is not None
                and rw[1] < rs[1] - torso_scale * 0.12
            )

            if left_raised and right_raised:
                person.arm_state = "Both arms raised"
            elif left_raised:
                person.arm_state = "Left arm raised"
            elif right_raised:
                person.arm_state = "Right arm raised"
            elif lw is not None and rw is not None:
                elbow_activity = False
                if le is not None and re is not None and shoulder is not None:
                    elbow_activity = (
                        float(np.linalg.norm(le - shoulder)) > torso_scale * 0.35
                        or
                        float(np.linalg.norm(re - shoulder)) > torso_scale * 0.35
                    )
                person.arm_state = "Arms active" if elbow_activity else "Arms down"
            elif lw is not None or rw is not None:
                person.arm_state = "One arm visible"

            if lk is not None and rk is not None and lh is not None and rh is not None:
                knee_y = (lk[1] + rk[1]) * 0.5
                hip_y = (lh[1] + rh[1]) * 0.5
                leg_span = max(20.0, abs(hip_y - knee_y))
                if knee_y > hip_y + leg_span * 0.20:
                    person.leg_state = "Legs bent"
                else:
                    person.leg_state = "Legs extended"
            elif lk is not None or rk is not None:
                person.leg_state = "One leg visible"

            visible_hands = []
            for hand in person.hands:
                center = hand_center(hand)
                position = hand_position(
                    center,
                    nose,
                    shoulder,
                    hip,
                    person.bbox
                )
                visible_hands.append(
                    (hand, position)
                )

            if visible_hands:
                activity_parts = []
                for hand, position in visible_hands:
                    gesture = hand.gesture or "Hand detected"
                    if gesture in {"Hand detected", "Hand gesture"}:
                        activity_parts.append(
                            f"{hand.handedness} hand {position}"
                        )
                    else:
                        activity_parts.append(
                            f"{gesture} with {hand.handedness.lower()} hand {position}"
                        )
                person.hand_activity = "; ".join(activity_parts)

            phone_hand = False
            phone_near_person = False
            for phone in phones:
                phone_center = np.array(
                    [float(phone.center[0]), float(phone.center[1])],
                    dtype=np.float32
                )
                phone_scale = max(25.0, math.sqrt(max(phone.area, 1)))
                for wrist in [lw, rw]:
                    if wrist is not None:
                        wrist_distance = float(np.linalg.norm(wrist - phone_center))
                        if wrist_distance < max(person_width * 0.20, phone_scale * 2.8):
                            phone_hand = True
                            break
                if phone_hand:
                    break

                x1, y1, x2, y2 = person.bbox
                if (
                    x1 - person_width * 0.15 <= phone.center[0] <= x2 + person_width * 0.15
                    and
                    y1 - person_width * 0.15 <= phone.center[1] <= y2 + person_width * 0.15
                ):
                    phone_near_person = True

            if phone_hand:
                person.phone_status = "Phone likely being held"
            elif phone_near_person:
                person.phone_status = "Phone near person; holding not confirmed"

            body_parts = []
            if person.posture != "Unknown":
                body_parts.append(person.posture)
            if person.arm_state not in {"Arms not clearly visible", "Arms down"}:
                body_parts.append(person.arm_state.lower())
            if person.leg_state == "Legs bent" and person.posture not in {"Sitting", "Lying"}:
                body_parts.append("legs bent")
            if person.movement != "Still":
                body_parts.append(person.movement.lower())
            if visible_hands:
                near_face = any(position == "near face" for _, position in visible_hands)
                if near_face:
                    body_parts.append("using a hand near the face")

            if not body_parts:
                body_parts.append("body visible")

            candidate = ", ".join(dict.fromkeys(body_parts))
            person.body_history.append(candidate)
            counts = Counter(person.body_history)
            person.body_language = counts.most_common(1)[0][0]

    def build_body_scene(self, people, hands):
        if not people and not hands:
            return "No confirmed human body or hands"
        if len(people) > 1:
            if len(hands) > 0:
                return "Multiple people with body and hands visible"
            return "Multiple people with bodies visible"
        if len(people) == 1:
            person = people[0]
            if person.phone_status == "Phone likely being held":
                return "Person likely holding a phone"
            if len(hands) == 2:
                return "Person with two hands detected"
            if len(hands) == 1:
                return "Person with one hand detected"
            if person.posture != "Unknown":
                return f"Person {person.posture.lower()}"
            return "Single person with body visible"
        if len(hands) == 2:
            return "Two hands detected"
        return "One hand detected"

    def build_body_narrative(self, people, hands, scene):
        parts = []

        if not people:
            if len(hands) == 2:
                parts.append("2 hands are detected, but no full person is currently confirmed.")
            elif len(hands) == 1:
                parts.append("1 hand is detected, but no full person is currently confirmed.")
            else:
                parts.append("No person or hand is currently confirmed.")
        else:
            parts.append(
                f"{len(people)} person"
                + (" is" if len(people) == 1 else "s are")
                + " confirmed."
            )

        if hands:
            gesture_counts = Counter(
                hand.gesture
                for hand in hands
                if hand.gesture not in {"Hand detected", "Hand gesture"}
            )
            if gesture_counts:
                parts.append(
                    "Detected hand actions: "
                    + ", ".join(
                        f"{gesture} ({count})" if count > 1 else gesture
                        for gesture, count in gesture_counts.items()
                    )
                    + "."
                )
            else:
                parts.append(
                    "Hands are detected, but no specific gesture is stable enough to identify."
                )

        for person in sorted(people, key=lambda p: p.person_id):
            details = [
                f"posture: {person.posture}",
                f"movement: {person.movement}",
                f"arms: {person.arm_state}",
                f"legs: {person.leg_state}"
            ]

            if person.hands:
                details.append(
                    f"hand activity: {person.hand_activity}"
                )
            else:
                details.append(
                    "hand activity: no hand landmark confirmed"
                )

            if person.phone_status != "No phone interaction confirmed":
                details.append(person.phone_status)

            explanation = []
            if "near face" in person.hand_activity:
                explanation.append(
                    "the hand landmark is close to the detected face/ nose region"
                )
            if person.phone_status == "Phone likely being held":
                explanation.append(
                    "the detected phone is close to a detected wrist"
                )
            elif person.phone_status == "Phone near person; holding not confirmed":
                explanation.append(
                    "the phone is inside or close to the person's body region, but wrist proximity is insufficient to confirm holding"
                )

            if explanation:
                details.append(
                    "reason: " + "; ".join(explanation)
                )

            parts.append(
                f"Person {person.person_id}: "
                + ", ".join(details)
                + "."
            )

        parts.append(
            f"Human scene: {scene}."
        )
        return " ".join(parts).replace("\n", " ")

    def build_body_scene(self, people, hands):
        if not people and not hands:
            return "No confirmed human body"
        if len(people) > 1:
            if len(hands) > 0:
                return "Multiple people with body and hands visible"
            return "Multiple people with bodies visible"
        if len(people) == 1:
            person = people[0]
            if person.phone_status == "Holding phone":
                return "Person holding phone"
            if len(hands) == 2:
                return "Person with two hands visible"
            if len(hands) == 1:
                return "Person with one hand visible"
            if person.posture != "Unknown":
                return f"Person {person.posture.lower()}"
            return "Single person with body visible"
        if len(hands) == 2:
            return "Two hands visible"
        return "One hand visible"

    def build_body_narrative(self, people, hands, scene):
        parts = []
        if not people:
            if len(hands) == 2:
                parts.append("2 hands are detected.")
            elif len(hands) == 1:
                parts.append("1 hand is detected.")
            else:
                parts.append("No person or hand is currently confirmed.")
        else:
            parts.append(f"{len(people)} person" + (" is" if len(people) == 1 else "s are") + " confirmed.")

        if hands:
            parts.append(f"{len(hands)} hand" + (" is" if len(hands) == 1 else "s are") + " visible.")
            gesture_counts = Counter(
                hand.gesture for hand in hands
                if hand.gesture not in {"Hand detected", "Hand gesture"}
            )
            if gesture_counts:
                parts.append("Gestures: " + ", ".join(
                    f"{gesture} ({count})" if count > 1 else gesture
                    for gesture, count in gesture_counts.items()
                ) + ".")
            else:
                parts.append("Hands are detected without a stable specific gesture.")

        for person in sorted(people, key=lambda p: p.person_id):
            details = [
                f"posture: {person.posture}",
                f"movement: {person.movement}",
                f"body language: {person.body_language}",
                f"arms: {person.arm_state}",
                f"legs: {person.leg_state}"
            ]
            if person.phone_status != "No phone interaction confirmed":
                details.append(person.phone_status)
            if person.hands:
                details.append(f"hands visible: {len(person.hands)}")
            parts.append(f"Person {person.person_id}: " + ", ".join(details) + ".")

        parts.append(f"Human scene: {scene}.")
        return " ".join(parts)

    def build_scene(
        self,
        objects,
        people,
        hands
    ):
        animal_count = sum(
            1
            for obj in objects
            if obj.label
            in OBJECT_GROUPS["Animals"]
        )

        vehicle_count = sum(
            1
            for obj in objects
            if obj.label
            in OBJECT_GROUPS["Vehicles"]
        )

        non_people = [
            obj
            for obj in objects
            if obj.label != "person"
        ]

        hand_count = len(
            hands
        )

        if (
            not people
            and
            not non_people
            and
            hand_count == 0
        ):
            return "No recognized subjects"

        if (
            len(people) > 1
            and
            animal_count > 0
        ):
            return "Multiple people and animals"

        if (
            len(people) > 1
            and
            non_people
        ):
            return "Multiple people and objects"

        if (
            len(people) > 1
            and
            hand_count > 0
        ):
            return "Multiple people with hands visible"

        if len(people) > 1:
            return "Multiple people"

        if (
            len(people) == 1
            and
            animal_count > 0
        ):
            return "Person and animal"

        if (
            len(people) == 1
            and
            vehicle_count > 0
        ):
            return "Person and vehicle"

        if (
            len(people) == 1
            and
            non_people
            and
            hand_count > 0
        ):
            return "Person, objects and hands"

        if (
            len(people) == 1
            and
            non_people
        ):
            return "Person and objects"

        if (
            len(people) == 1
            and
            hand_count == 2
        ):
            return "Person with two hands visible"

        if (
            len(people) == 1
            and
            hand_count == 1
        ):
            return "Person with one hand visible"

        if len(people) == 1:
            return "Single person"

        if hand_count == 2:
            return "Two hands visible"

        if hand_count == 1:
            return "One hand visible"

        if animal_count > 0:
            return "Animals detected"

        return "Objects detected"

    def build_narrative(
        self,
        objects,
        people,
        hands,
        scene
    ):
        parts = []

        if len(people) == 0:
            if len(hands) == 2:
                parts.append(
                    "2 hands are visible."
                )
            elif len(hands) == 1:
                parts.append(
                    "1 hand is visible."
                )
            else:
                parts.append(
                    "No person is currently confirmed."
                )
        elif len(people) == 1:
            parts.append(
                "1 person is visible."
            )
        else:
            parts.append(
                f"{len(people)} people are visible."
            )

        if hands:
            if len(hands) == 1:
                parts.append(
                    "1 hand is detected."
                )
            else:
                parts.append(
                    f"{len(hands)} hands are detected."
                )

            gesture_counts = Counter(
                hand.gesture
                for hand in hands
                if hand.gesture
                not in {
                    "Hand detected",
                    "Hand gesture"
                }
            )

            if gesture_counts:
                gesture_text = []

                for gesture, count in (
                    gesture_counts.items()
                ):
                    if count == 1:
                        gesture_text.append(
                            gesture
                        )
                    else:
                        gesture_text.append(
                            f"{gesture} ({count})"
                        )

                parts.append(
                    "Stable hand gestures: "
                    +
                    ", ".join(
                        gesture_text
                    )
                    +
                    "."
                )
            else:
                parts.append(
                    "Hands are detected without a stable specific gesture."
                )

        for person in sorted(
            people,
            key=lambda p: p.person_id
        ):
            details = []

            if (
                person.posture
                != "Unknown"
            ):
                details.append(
                    f"posture: "
                    f"{person.posture.lower()}"
                )

            if (
                person.movement
                != "Still"
            ):
                details.append(
                    f"movement: "
                    f"{person.movement.lower()}"
                )

            if person.hands:
                hand_count = len(
                    person.hands
                )

                details.append(
                    f"{hand_count} hand"
                    f"{'s' if hand_count != 1 else ''}"
                    " visible"
                )

            if details:
                parts.append(
                    f"Person {person.person_id}: "
                    +
                    ", ".join(
                        details
                    )
                    +
                    "."
                )

        counts = {}

        for obj in objects:
            if obj.label == "person":
                continue

            counts[obj.label] = (
                counts.get(
                    obj.label,
                    0
                )
                +
                1
            )

        if counts:
            object_text = []

            for label, count in sorted(
                counts.items(),
                key=lambda item: item[0]
            ):
                object_text.append(
                    f"{count} {label}"
                    f"{'s' if count != 1 else ''}"
                )

            parts.append(
                "Confirmed objects: "
                +
                ", ".join(
                    object_text
                )
                +
                "."
            )

        parts.append(
            f"Overall scene: {scene}."
        )

        return " ".join(parts)

    def update_events(
        self,
        scene,
        people,
        hands
    ):
        self.events.add(
            "scene",
            f"Scene changed to {scene}"
        )

        self.events.add(
            "people",
            f"People count: {len(people)}"
        )

        self.events.add(
            "hands",
            f"Hands count: {len(hands)}"
        )

        for person in people:
            self.events.add(
                f"posture_{person.person_id}",
                f"Person {person.person_id}: "
                f"{person.posture}"
            )

            self.events.add(
                f"movement_{person.person_id}",
                f"Person {person.person_id}: "
                f"{person.movement}"
            )

            if person.gestures:
                gestures = ", ".join(
                    sorted(
                        set(
                            person.gestures
                        )
                    )
                )

                self.events.add(
                    f"gesture_{person.person_id}",
                    f"Person {person.person_id}: "
                    f"{gestures}"
                )

        if hands:
            gestures = sorted(
                set(
                    hand.gesture
                    for hand in hands
                )
            )

            meaningful = [
                gesture
                for gesture in gestures
                if gesture
                not in {
                    "Hand detected",
                    "Hand gesture"
                }
            ]

            if meaningful:
                self.events.add(
                    "global_gesture",
                    "Stable gestures: "
                    +
                    ", ".join(
                        meaningful
                    )
                )

    def build_overlay_payload(
        self,
        frame,
        objects,
        people,
        hands,
        faces=None
    ):
        height, width = frame.shape[:2]
        payload = {
            "width": int(width),
            "height": int(height),
            "mirror": bool(self.mirror),
            "objects": [],
            "people": [],
            "hands": []
        }
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            payload["objects"].append({
                "label": obj.label,
                "confidence": float(obj.confidence),
                "track_id": int(obj.track_id),
                "bbox": [float(x1) / max(1, width), float(y1) / max(1, height), float(x2) / max(1, width), float(y2) / max(1, height)]
            })
        for person in people:
            x1, y1, x2, y2 = person.bbox
            keypoints = []
            for point in person.keypoints or []:
                if len(point) >= 3:
                    keypoints.append([float(point[0]) / max(1, width), float(point[1]) / max(1, height), float(point[2])])
            payload["people"].append({
                "track_id": int(person.person_id),
                "bbox": [float(x1) / max(1, width), float(y1) / max(1, height), float(x2) / max(1, width), float(y2) / max(1, height)],
                "keypoints": keypoints
            })
        for hand in hands:
            landmarks = []
            for point in hand.landmarks:
                if len(point) >= 3:
                    landmarks.append([float(point[0]), float(point[1]), float(point[2])])
            x1, y1, x2, y2 = hand.bbox
            payload["hands"].append({
                "hand_id": int(hand.hand_id),
                "handedness": str(hand.handedness),
                "confidence": float(hand.confidence),
                "gesture": str(hand.gesture),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "landmarks": landmarks
            })
        payload["faces"] = [
            {
                "face_id": int(face.face_id),
                "bbox": [float(v) for v in face.bbox],
                "expression": str(face.expression)
            }
            for face in (faces or [])
        ]
        return payload

    def render(
        self,
        frame,
        objects,
        people,
        hands,
        scene,
        narrative,
        faces=None
    ):
        output = frame.copy()

        if self.show_boxes:
            for obj in objects:
                x1, y1, x2, y2 = (
                    obj.bbox
                )

                thickness = (
                    2
                    if obj.label == "person"
                    else 1
                )

                cv2.rectangle(
                    output,
                    (
                        int(x1),
                        int(y1)
                    ),
                    (
                        int(x2),
                        int(y2)
                    ),
                    (255, 210, 50),
                    thickness
                )

                if self.show_labels:
                    label = (
                        f"{obj.label} "
                        f"{obj.confidence:.0%}"
                    )

                    if (
                        obj.label ==
                        "person"
                    ):
                        label += (
                            f"  P{obj.track_id}"
                        )

                    self.draw_label(
                        output,
                        label,
                        int(x1),
                        max(
                            20,
                            int(y1) - 5
                        )
                    )

        if self.show_pose:
            for person in people:
                self.draw_pose(
                    output,
                    person.keypoints
                )

        if self.show_hands:
            for hand in hands:
                self.draw_hand(
                    output,
                    hand
                )

        for face in (faces or []):
            self.draw_face(output, face)

        if self.show_hud:
            self.draw_hud(
                output,
                objects,
                people,
                hands,
                scene,
                narrative
            )

        return cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )

    def draw_pose(
        self,
        frame,
        keypoints
    ):
        if not keypoints:
            return

        for a, b in POSE_CONNECTIONS:
            if (
                a >= len(keypoints)
                or
                b >= len(keypoints)
            ):
                continue

            p1 = keypoints[a]
            p2 = keypoints[b]

            if (
                p1[2] < 0.35
                or
                p2[2] < 0.35
            ):
                continue

            cv2.line(
                frame,
                (
                    int(p1[0]),
                    int(p1[1])
                ),
                (
                    int(p2[0]),
                    int(p2[1])
                ),
                (50, 220, 120),
                2,
                cv2.LINE_AA
            )

        for point in keypoints:
            if point[2] < 0.35:
                continue

            cv2.circle(
                frame,
                (
                    int(point[0]),
                    int(point[1])
                ),
                4,
                (80, 230, 255),
                -1,
                cv2.LINE_AA
            )

    def draw_hand(
        self,
        frame,
        hand
    ):
        height, width = (
            frame.shape[:2]
        )

        points = []

        for point in hand.landmarks:
            x = int(
                point[0] *
                width
            )

            y = int(
                point[1] *
                height
            )

            points.append(
                (x, y)
            )

        for a, b in HAND_CONNECTIONS:
            if (
                a >= len(points)
                or
                b >= len(points)
            ):
                continue

            cv2.line(
                frame,
                points[a],
                points[b],
                (255, 100, 180),
                2,
                cv2.LINE_AA
            )

        for point in points:
            cv2.circle(
                frame,
                point,
                3,
                (255, 180, 80),
                -1,
                cv2.LINE_AA
            )

        x1 = int(
            hand.bbox[0] *
            width
        )

        y1 = int(
            hand.bbox[1] *
            height
        )

        gesture_text = (
            hand.gesture
        )

        if (
            hand.gesture ==
            "Hand detected"
        ):
            gesture_text = (
                "Hand detected"
            )

        self.draw_label(
            frame,
            (
                f"{hand.handedness}: "
                f"{gesture_text}"
            ),
            x1,
            max(
                20,
                y1 - 5
            )
        )

    def draw_label(
        self,
        frame,
        text,
        x,
        y
    ):
        font = (
            cv2.FONT_HERSHEY_SIMPLEX
        )

        scale = 0.48
        thickness = 1

        size = cv2.getTextSize(
            text,
            font,
            scale,
            thickness
        )[0]

        x = max(
            3,
            min(
                int(x),
                frame.shape[1]
                -
                size[0]
                -
                8
            )
        )

        y = max(
            size[1] + 8,
            min(
                int(y),
                frame.shape[0]
                -
                4
            )
        )

        cv2.rectangle(
            frame,
            (
                x - 3,
                y - size[1] - 6
            ),
            (
                x + size[0] + 4,
                y + 3
            ),
            (15, 15, 15),
            -1
        )

        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA
        )

    def draw_face(self, frame, face):
        h,w=frame.shape[:2]
        x1,y1,x2,y2=face.bbox
        p1=(int(x1*w),int(y1*h)); p2=(int(x2*w),int(y2*h))
        cv2.rectangle(frame,p1,p2,(255,120,220),2)
        self.draw_label(frame,f"Face {face.expression}",p1[0],max(20,p1[1]-5))

    def draw_hud(
        self,
        frame,
        objects,
        people,
        hands,
        scene,
        narrative
    ):
        height, width = (
            frame.shape[:2]
        )

        overlay_height = 86

        overlay = frame[
            0:overlay_height,
            0:width
        ].copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (
                width,
                overlay_height
            ),
            (12, 16, 22),
            -1
        )

        frame[
            0:overlay_height,
            0:width
        ] = cv2.addWeighted(
            overlay,
            0.84,
            frame[
                0:overlay_height,
                0:width
            ],
            0.16,
            0
        )

        fps = (
            self.calculate_camera_fps()
        )

        ai_fps = 0.0

        if self.ai_times:
            average = (
                sum(
                    self.ai_times
                )
                /
                len(
                    self.ai_times
                )
            )

            ai_fps = (
                1.0 /
                max(
                    0.0001,
                    average
                )
            )

        line1 = (
            f"LIFEVISION  |  "
            f"{self.mode}  |  "
            f"Scene: {scene}"
        )

        line2 = (
            f"People: {len(people)}  |  "
            f"Objects: {len(objects)}  |  "
            f"Hands: {len(hands)}  |  "
            f"Camera: {fps:.1f} FPS  |  "
            f"AI: {ai_fps:.1f} FPS"
        )

        cv2.putText(
            frame,
            line1[:130],
            (14, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 240, 240),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            line2[:150],
            (14, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 220, 240),
            1,
            cv2.LINE_AA
        )

        narrative_short = (
            narrative[:115]
        )

        cv2.putText(
            frame,
            narrative_short,
            (14, 71),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (210, 210, 210),
            1,
            cv2.LINE_AA
        )

    def calculate_camera_fps(self):
        elapsed = max(
            0.001,
            time.time()
            -
            self.camera_start
        )

        return (
            self.camera_frames /
            elapsed
        )

    def format_snapshot(
        self,
        snapshot
    ):
        people_lines = []

        for person in sorted(
            snapshot.people,
            key=lambda p: p.person_id
        ):
            line = (
                f"Person {person.person_id} | "
                f"{person.posture} | "
                f"{person.movement} | "
                f"Body: {person.body_language} | "
                f"Arms: {person.arm_state} | "
                f"Legs: {person.leg_state} | "
                f"Face: {person.facial_expression}"
            )

            if person.gestures:
                line += (
                    " | "
                    +
                    ", ".join(
                        sorted(
                            set(
                                person.gestures
                            )
                        )
                    )
                )

            if person.hands:
                line += (
                    f" | Hands: "
                    f"{len(person.hands)}"
                )

            people_lines.append(
                line
            )

        object_counts = {}

        for obj in snapshot.objects:
            if obj.label == "person":
                continue

            object_counts[obj.label] = (
                object_counts.get(
                    obj.label,
                    0
                )
                +
                1
            )

        object_lines = []

        for label, count in sorted(
            object_counts.items()
        ):
            object_lines.append(
                f"{label}: {count}"
            )

        if not object_lines:
            object_text = (
                "No non-person objects recognized."
            )
        else:
            object_text = (
                "\n".join(
                    object_lines
                )
            )

        if not people_lines:
            people_text = (
                "No confirmed people."
            )
        else:
            people_text = (
                "\n".join(
                    people_lines
                )
            )

        events_text = (
            self.events.formatted()
        )

        hand_status = (
            "Available"
            if snapshot.hand_available
            else "Unavailable"
        )

        hand_lines = [
            f"Hand Engine: {hand_status}",
            f"Detected hands: {len(snapshot.hands)}"
        ]

        if len(snapshot.hands) == 1:
            hand_lines.append(
                "Hand count: 1 hand detected"
            )

        elif len(snapshot.hands) == 2:
            hand_lines.append(
                "Hand count: 2 hands detected"
            )

        for hand in snapshot.hands:
            hand_lines.append(
                f"{hand.handedness}: {hand.gesture} | "
                f"21 landmarks | 3D XYZ available | "
                f"confidence {hand.confidence:.0%}"
            )

        runtime_errors = []

        if self.object_error:
            runtime_errors.append(
                "Object: "
                +
                self.object_error
            )

        if self.pose_error:
            runtime_errors.append(
                "Pose: "
                +
                self.pose_error
            )

        if self.gpu_error:
            runtime_errors.append(
                "GPU: "
                +
                self.gpu_error
            )

        if self.hand_engine.error:
            runtime_errors.append(
                "Hands: "
                +
                self.hand_engine.error
            )

        if self.face_engine.error:
            runtime_errors.append(
                "Face: "
                +
                self.face_engine.error
            )

        if snapshot.runtime_error:
            runtime_errors.append(
                "Runtime: "
                +
                snapshot.runtime_error
            )

        if MODEL_ERROR:
            runtime_errors.append(
                "Models: "
                +
                MODEL_ERROR
            )

        if runtime_errors:
            diagnostic = (
                "\n".join(
                    runtime_errors
                )
            )
        else:
            diagnostic = (
                "All active engines are running."
            )

        return {
            "scene": snapshot.scene,
            "narrative": snapshot.narrative,
            "people": people_text,
            "objects": object_text,
            "events": events_text,
            "metrics": (
                f"Camera FPS: "
                f"{snapshot.fps:.1f}\n"
                f"AI FPS: "
                f"{snapshot.ai_fps:.1f}\n"
                f"People: "
                f"{len(snapshot.people)}\n"
                f"Objects: "
                f"{len(snapshot.objects)}\n"
                f"Hands: "
                f"{len(snapshot.hands)}\n"
                f"Mode: "
                f"{self.mode}"
            ),
            "hands": (
                "\n".join(
                    hand_lines
                )
            ),
            "faces": "\n".join(
                f"Face {i+1}: {face.expression}"
                for i, face in enumerate(snapshot.faces)
            ) or "No face detected.",
            "body": "\n".join(
                f"Person {p.person_id}: posture={p.posture} | movement={p.movement} | arms={p.arm_state} | legs={p.leg_state} | body language={p.body_language}"
                for p in snapshot.people
            ) or "No body detected.",
            "diagnostic": diagnostic
        }

    def clear_events(self):
        with self.lock:
            self.events.clear()

    def stop(self):
        self.worker_running = False

        with self.worker_condition:
            self.worker_condition.notify_all()

        if (
            self.worker_thread.is_alive()
        ):
            self.worker_thread.join(
                timeout=1.0
            )


ENGINE = LifeVisionEngine()


@spaces.GPU(
duration=30
)
def process_frame(
    frame,
    process_fps,
    object_confidence,
    pose_confidence,
    hand_confidence,
    show_boxes,
    show_pose,
    show_hands,
    show_labels,
    show_hud,
    mirror,
    groups
):
    ENGINE.update_config(
        MODE_FULL,
        process_fps,
        object_confidence,
        pose_confidence,
        hand_confidence,
        show_boxes,
        show_pose,
        show_hands,
        show_labels,
        show_hud,
        mirror,
        groups
    )

    if frame is None:
        data = ENGINE.format_snapshot(ENGINE.latest_snapshot)
        return (
            data["metrics"], data["scene"], data["narrative"],
            data["people"], data["objects"], data["events"],
            data["hands"], data["faces"], data["body"], data["diagnostic"],
            json.dumps(ENGINE.latest_overlay_payload, separators=(",", ":"))
        )

    current_frame = np.ascontiguousarray(frame)
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
    if mirror:
        current_frame = cv2.flip(current_frame, 1)

    with ENGINE.lock:
        ENGINE.camera_frames += 1
        ENGINE.latest_frame = current_frame.copy()
        ENGINE.latest_frame_time = time.time()
        ENGINE.frame_id += 1
        current_frame_id = ENGINE.frame_id
        ENGINE.ai_cycle += 1

    if show_hands and (not ENGINE.hand_engine.initialized or not ENGINE.hand_engine.available):
        ENGINE.hand_engine.initialized = False
        ENGINE.hand_engine.initializing = False
        ENGINE.hand_engine.initialize()

    started = time.time()
    gpu_result = run_gpu_inference(
        current_frame,
        float(object_confidence),
        float(pose_confidence),
        MODE_FULL,
        list(groups or []),
        bool(show_pose),
        bool(show_pose)
    )

    output, snapshot, overlay_payload = ENGINE.process_ai_result(
        current_frame,
        gpu_result,
        current_frame_id,
        bool(show_hands),
        float(hand_confidence),
        bool(show_pose)
    )

    duration = max(0.0001, time.time() - started)
    with ENGINE.lock:
        ENGINE.last_ai_duration = duration
        ENGINE.ai_times.append(duration)
        ENGINE.latest_snapshot = snapshot
        ENGINE.render_objects = list(snapshot.objects)
        ENGINE.render_people = list(snapshot.people)
        ENGINE.render_hands = list(snapshot.hands)
        ENGINE.render_scene = snapshot.scene
        ENGINE.render_narrative = snapshot.narrative
        ENGINE.latest_output = output
        ENGINE.latest_overlay_payload = overlay_payload
        ENGINE.last_ai_time = time.time()
        ENGINE.gpu_error = ""

    data = ENGINE.format_snapshot(snapshot)
    return (
        data["metrics"],
        data["scene"],
        data["narrative"],
        data["people"],
        data["objects"],
        data["events"],
        data["hands"],
        data["faces"],
        data["body"],
        data["diagnostic"],
        json.dumps(ENGINE.latest_overlay_payload, separators=(",", ":"))
    )


def clear_event_log():
    ENGINE.clear_events()

    return "Event log cleared."


CSS = """
#camera_output {
    min-height: 560px;
}

#camera_output img {
    object-fit: contain !important;
}

#camera_output {
    position: relative !important;
}

#camera_output > div {
    position: relative !important;
}

#camera_overlay_canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    z-index: 20;
    pointer-events: none;
}

.metric-box textarea {
    font-size: 15px !important;
}

.narrative-box textarea {
    font-size: 15px !important;
}

.status-title {
    font-weight: 700;
}

.gradio-container {
    max-width: 1500px !important;
}

#camera_output, .metric-box, .narrative-box {
    border-radius: 14px !important;
}

#camera_output {
    border: 1px solid rgba(128,128,128,0.35) !important;
    overflow: hidden !important;
}

.gradio-textbox {
    border-radius: 12px !important;
}
"""


with gr.Blocks(
    title="LifeVision"
) as demo:

    gr.Markdown(
        """
# LifeVision
### Real-Time Computer Vision, Body & Scene Awareness
Version 25.0
"""
    )

    with gr.Row():

        with gr.Column(
            scale=7,
            min_width=600
        ):
            camera = gr.Image(
                sources=["webcam"],
                type="numpy",
                streaming=True,
                label="Live Camera",
                elem_id="camera_output",
                interactive=True
            )

            overlay_data = gr.Textbox(
                value=json.dumps({"width": 640, "height": 480, "mirror": False, "objects": [], "people": [], "hands": [], "faces": []}, separators=(",", ":")),
                elem_id="overlay_data",
                visible=False,
                interactive=False
            )


        with gr.Column(
            scale=3,
            min_width=320
        ):

            gr.Markdown(
                "### Live Status"
            )

            metrics = gr.Textbox(
                label="System Metrics",
                value="Waiting for camera...",
                lines=6,
                interactive=False,
                elem_classes=[
                    "metric-box"
                ]
            )

            scene = gr.Textbox(
                label="Scene",
                value="Waiting for camera...",
                lines=2,
                interactive=False
            )

            narrative = gr.Textbox(
                label="Live Understanding",
                value=(
                    "Start the camera "
                    "to begin LifeVision."
                ),
                lines=7,
                interactive=False,
                elem_classes=[
                    "narrative-box"
                ]
            )

    with gr.Accordion(
        "LifeVision Controls",
        open=False
    ):

        with gr.Row():

            gr.Markdown("**Mode: Full Awareness** — Objects + People + Hands + Pose + Legs + Body Language + Face/Expression")

            process_fps = gr.Slider(
                minimum=1,
                maximum=10,
                value=4,
                step=1,
                label="AI Processing FPS"
            )

        with gr.Row():

            object_confidence = gr.Slider(
                minimum=0.10,
                maximum=0.90,
                value=0.25,
                step=0.05,
                label="Object Confidence"
            )

            pose_confidence = gr.Slider(
                minimum=0.10,
                maximum=0.90,
                value=0.25,
                step=0.05,
                label="Pose Confidence"
            )

            hand_confidence = gr.Slider(
                minimum=0.10,
                maximum=0.90,
                value=0.25,
                step=0.05,
                label="Hand Confidence"
            )

        with gr.Row():

            show_boxes = gr.Checkbox(
                value=True,
                label="Object Boxes"
            )

            show_pose = gr.Checkbox(
                value=True,
                label="Pose Skeleton"
            )

            show_hands = gr.Checkbox(
                value=True,
                label="Hand Landmarks"
            )

            show_labels = gr.Checkbox(
                value=True,
                label="Labels"
            )

            show_hud = gr.Checkbox(
                value=True,
                label="Camera HUD"
            )

            mirror = gr.Checkbox(
                value=False,
                label="Mirror Camera"
            )

        groups = gr.CheckboxGroup(
            choices=list(
                OBJECT_GROUPS.keys()
            ),
            value=[
                "People",
                "Animals",
                "Vehicles",
                "Indoor",
                "Food",
                "Sports"
            ],
            label="Object Categories"
        )

    with gr.Row():

        with gr.Column():

            people_output = gr.Textbox(
                label="People & Body Awareness",
                value="No confirmed people.",
                lines=8,
                interactive=False
            )

        with gr.Column():

            objects_output = gr.Textbox(
                label="Detected Objects",
                value=(
                    "No non-person "
                    "objects recognized."
                ),
                lines=8,
                interactive=False
            )

    with gr.Row():

        with gr.Column():
            face_output = gr.Textbox(
                label="Facial Expression",
                value="No face analyzed yet.",
                lines=6,
                interactive=False
            )

        with gr.Column():
            body_output = gr.Textbox(
                label="Body Language / Posture / Legs",
                value="No body detected yet.",
                lines=6,
                interactive=False
            )

    with gr.Row():

        with gr.Column():

            events_output = gr.Textbox(
                label="Event Log",
                value="No events yet.",
                lines=12,
                interactive=False
            )

        with gr.Column():

            hands_output = gr.Textbox(
                label="Hand Engine",
                value=(
                    "Hand Engine: Waiting..."
                ),
                lines=7,
                interactive=False
            )

            diagnostic_output = gr.Textbox(
                label="Runtime Diagnostic",
                value=(
                    "Camera ready. "
                    "AI engines initialize "
                    "when processing begins."
                ),
                lines=7,
                interactive=False
            )

    clear_button = gr.Button(
        "Clear Event Log",
        variant="secondary"
    )

    clear_button.click(
        fn=clear_event_log,
        inputs=[],
        outputs=events_output,
        queue=False
    )

    stream_inputs = [
        camera,
        process_fps,
        object_confidence,
        pose_confidence,
        hand_confidence,
        show_boxes,
        show_pose,
        show_hands,
        show_labels,
        show_hud,
        mirror,
        groups
    ]

    stream_outputs = [
        metrics,
        scene,
        narrative,
        people_output,
        objects_output,
        events_output,
        hands_output,
        face_output,
        body_output,
        diagnostic_output,
        overlay_data
    ]

    camera.stream(
        fn=process_frame,
        inputs=stream_inputs,
        outputs=stream_outputs,
        stream_every=0.25,
        concurrency_limit=1,
        queue=True,
        show_progress="hidden"
    )


OVERLAY_JS = r"""
(() => {
    const state = { canvas: null, ctx: null, data: null, timer: null };
    function video() { return document.querySelector("#camera_output video"); }
    function ensure() {
        const root = document.querySelector("#camera_output");
        const v = video();
        if (!root || !v) return false;
        if (!state.canvas || !state.canvas.isConnected) {
            state.canvas = document.createElement("canvas");
            state.canvas.id = "camera_overlay_canvas";
            root.appendChild(state.canvas);
            state.ctx = state.canvas.getContext("2d");
        }
        const w = v.videoWidth || 640, h = v.videoHeight || 480;
        state.canvas.width = w;
        state.canvas.height = h;
        state.canvas.style.width = `${v.clientWidth || w}px`;
        state.canvas.style.height = `${v.clientHeight || h}px`;
        state.canvas.style.left = `${v.offsetLeft || 0}px`;
        state.canvas.style.top = `${v.offsetTop || 0}px`;
        return true;
    }
    function label(ctx, s, x, y, size = 15) {
        ctx.font = `700 ${size}px Arial`;
        const m = ctx.measureText(s);
        ctx.fillStyle = "rgba(0,0,0,0.75)";
        ctx.fillRect(x - 4, y - size - 5, m.width + 8, size + 9);
        ctx.fillStyle = "#fff";
        ctx.fillText(s, x, y);
    }
    function box(ctx, b, w, h, s) {
        const x1=b[0]*w,y1=b[1]*h,x2=b[2]*w,y2=b[3]*h;
        ctx.strokeStyle="#00ff66";ctx.lineWidth=3;ctx.strokeRect(x1,y1,x2-x1,y2-y1);
        label(ctx,s,x1,Math.max(22,y1-6));
    }
    function pose(ctx, people, w, h) {
        const c=[[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];
        for(const person of (people||[])){
            const p=person.keypoints||[];ctx.strokeStyle="#32dc78";ctx.lineWidth=3;
            for(const [a,b] of c){if(!p[a]||!p[b]||p[a][2]<0.35||p[b][2]<0.35)continue;ctx.beginPath();ctx.moveTo(p[a][0]*w,p[a][1]*h);ctx.lineTo(p[b][0]*w,p[b][1]*h);ctx.stroke();}
            for(const q of p){if(!q||q[2]<0.35)continue;ctx.fillStyle="#ffe050";ctx.beginPath();ctx.arc(q[0]*w,q[1]*h,4,0,Math.PI*2);ctx.fill();}
        }
    }
    function hands(ctx,list,w,h) {
        const c=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,17],[17,18],[18,19],[19,20],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[13,17]];
        for(const hand of (list||[])){
            const p=hand.landmarks||[];if(p.length!==21)continue;
            ctx.strokeStyle="#ff3bd4";ctx.lineWidth=3;
            for(const [a,b] of c){ctx.beginPath();ctx.moveTo(p[a][0]*w,p[a][1]*h);ctx.lineTo(p[b][0]*w,p[b][1]*h);ctx.stroke();}
            for(let i=0;i<21;i++){ctx.fillStyle=i===0?"#fff":"#ff3bd4";ctx.beginPath();ctx.arc(p[i][0]*w,p[i][1]*h,5,0,Math.PI*2);ctx.fill();ctx.fillStyle="#fff";ctx.font="600 10px Arial";ctx.fillText(String(i),p[i][0]*w+6,p[i][1]*h-5);}
            box(ctx,hand.bbox,w,h,`${hand.handedness} hand - ${hand.gesture} - 21 points`);
        }
    }
    function draw(){
        if(!ensure()||!state.ctx)return;
        const c=state.ctx,w=state.canvas.width,h=state.canvas.height;c.clearRect(0,0,w,h);
        const d=state.data;if(!d)return;
        for(const o of (d.objects||[])){const id=o.track_id>0?` - P${o.track_id}`:"";box(c,o.bbox,w,h,`${o.label} ${Math.round(o.confidence*100)}%${id}`);}
        pose(c,d.people,w,h);
        hands(c,d.hands,w,h);
        for(const f of (d.faces||[])){box(c,f.bbox,w,h,`Face - ${f.expression}`);}
    }
    function read(){
        const t=document.querySelector("#overlay_data textarea");if(!t)return;
        try{state.data=JSON.parse(t.value||"{}");draw();}catch(e){}
    }
    function boot(){
        read();draw();
        if(state.timer)return;
        state.timer=setInterval(()=>{read();draw();},120);
    }
    setTimeout(boot,1000);
    window.addEventListener("resize",draw);
})();
"""

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
        js=OVERLAY_JS
    )
