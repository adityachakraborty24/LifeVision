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
    import torch
except Exception:
    torch = None

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


APP_VERSION = "16.0"

OBJECT_MODEL = "yolo11n.pt"
POSE_MODEL = "yolo11n-pose.pt"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "lifevision")
HAND_MODEL_PATH = os.path.join(CACHE_DIR, "hand_landmarker.task")

os.makedirs(CACHE_DIR, exist_ok=True)


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
    (14, 15),
    (15, 16)
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
    history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_update: float = 0.0


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
            return []

        used = set()
        output = []

        for det in detections:
            cx, cy = det["center"]
            best_id = None
            best_distance = self.max_distance

            for track_id, previous in self.tracks.items():
                if track_id in used:
                    continue

                px, py = previous["center"]
                distance = math.hypot(cx - px, cy - py)

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
                "timestamp": time.time()
            }

            det["track_id"] = best_id
            output.append(det)

        expired = [
            track_id
            for track_id, value in self.tracks.items()
            if time.time() - value["timestamp"] > 2.5
        ]

        for track_id in expired:
            self.tracks.pop(track_id, None)

        return output


class MotionAnalyzer:
    def calculate(self, person):
        if not person.history:
            return "Still", 0.0

        previous = person.history[-1]
        cx, cy = person.center
        px, py = previous

        distance = math.hypot(cx - px, cy - py)

        if distance < 3:
            movement = "Still"
        elif distance < 12:
            movement = "Moving"
        else:
            movement = "Fast movement"

        return movement, distance


class GestureRecognizer:
    def __init__(self):
        self.finger_indices = {
            "thumb": (4, 3),
            "index": (8, 6),
            "middle": (12, 10),
            "ring": (16, 14),
            "pinky": (20, 18)
        }

    def distance(self, a, b):
        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2
        )

    def angle(self, a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)

        denom = np.linalg.norm(ba) * np.linalg.norm(bc)

        if denom == 0:
            return 180.0

        cosine = np.clip(
            np.dot(ba, bc) / denom,
            -1.0,
            1.0
        )

        return math.degrees(math.acos(cosine))

    def finger_extended(self, landmarks, tip, pip):
        wrist = landmarks[0]
        tip_point = landmarks[tip]
        pip_point = landmarks[pip]

        wrist_distance = self.distance(wrist, tip_point)
        pip_distance = self.distance(wrist, pip_point)

        return wrist_distance > pip_distance * 1.12

    def classify(self, landmarks):
        if len(landmarks) != 21:
            return "Unknown"

        thumb = self.finger_extended(landmarks, 4, 3)
        index = self.finger_extended(landmarks, 8, 6)
        middle = self.finger_extended(landmarks, 12, 10)
        ring = self.finger_extended(landmarks, 16, 14)
        pinky = self.finger_extended(landmarks, 20, 18)

        extended = [thumb, index, middle, ring, pinky]

        if index and middle and ring and pinky and not thumb:
            return "Open hand"

        if not any(extended):
            return "Fist"

        if thumb and not index and not middle and not ring and not pinky:
            return "Thumbs up"

        if index and not middle and not ring and not pinky:
            return "Pointing"

        if middle and not index and not ring and not pinky:
            return "Middle finger"

        if index and middle and not ring and not pinky:
            return "Peace"

        if thumb and index and middle and ring and pinky:
            return "Open hand"

        return "Hand gesture"


class PostureAnalyzer:
    def point(self, keypoints, index):
        if index >= len(keypoints):
            return None

        point = keypoints[index]

        if len(point) < 3:
            return None

        x, y, confidence = point

        if confidence < 0.25:
            return None

        return np.array([x, y], dtype=np.float32)

    def midpoint(self, a, b):
        if a is None or b is None:
            return None

        return (a + b) / 2.0

    def angle_from_horizontal(self, a, b):
        if a is None or b is None:
            return None

        dx = b[0] - a[0]
        dy = b[1] - a[1]

        angle = math.degrees(math.atan2(dy, dx))

        while angle > 90:
            angle -= 180

        while angle < -90:
            angle += 180

        return abs(angle)

    def calculate(self, keypoints, bbox):
        if not keypoints:
            return "Unknown"

        nose = self.point(keypoints, 0)
        left_shoulder = self.point(keypoints, 5)
        right_shoulder = self.point(keypoints, 6)
        left_hip = self.point(keypoints, 11)
        right_hip = self.point(keypoints, 12)
        left_knee = self.point(keypoints, 13)
        right_knee = self.point(keypoints, 14)
        left_ankle = self.point(keypoints, 15)
        right_ankle = self.point(keypoints, 16)

        shoulder = self.midpoint(left_shoulder, right_shoulder)
        hip = self.midpoint(left_hip, right_hip)
        knee = self.midpoint(left_knee, right_knee)
        ankle = self.midpoint(left_ankle, right_ankle)

        if shoulder is None or hip is None:
            return "Unknown"

        x1, y1, x2, y2 = bbox
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        torso_length = np.linalg.norm(hip - shoulder)

        torso_angle = self.angle_from_horizontal(shoulder, hip)

        if torso_angle is not None and torso_angle < 38:
            if width > height * 0.9:
                return "Lying"

        if ankle is not None and knee is not None:
            leg_distance = np.linalg.norm(ankle - knee)
            knee_hip_distance = np.linalg.norm(knee - hip)

            if torso_angle is not None:
                if torso_angle > 63:
                    if knee_hip_distance < torso_length * 1.45:
                        return "Sitting"

        if torso_angle is not None and torso_angle > 62:
            return "Standing"

        if nose is not None and shoulder is not None:
            head_distance = np.linalg.norm(nose - shoulder)

            if torso_length > 0 and head_distance < torso_length * 0.8:
                if torso_angle is not None and torso_angle > 55:
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
        self.last_detection = 0.0

        if not MEDIAPIPE_AVAILABLE:
            self.error = "MediaPipe is not available."
            return

        try:
            self.ensure_model()
            self.create_landmarker()
        except Exception as exc:
            self.error = f"Hand engine unavailable: {exc}"

    def ensure_model(self):
        if os.path.exists(HAND_MODEL_PATH):
            return

        urllib.request.urlretrieve(
            HAND_MODEL_URL,
            HAND_MODEL_PATH
        )

    def create_landmarker(self):
        base_options = python.BaseOptions(
            model_asset_path=HAND_MODEL_PATH
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=8,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45
        )

        self.landmarker = vision.HandLandmarker.create_from_options(
            options
        )

        self.available = True

    def detect(self, frame, confidence=0.45):
        if not self.available or self.landmarker is None:
            return []

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = self.landmarker.detect(image)

            hands = []

            if not result.hand_landmarks:
                return hands

            for index, landmarks in enumerate(result.hand_landmarks):
                points = [
                    (
                        float(point.x),
                        float(point.y),
                        float(point.z)
                    )
                    for point in landmarks
                ]

                handedness = "Unknown"

                if result.handedness:
                    if index < len(result.handedness):
                        if result.handedness[index]:
                            handedness = result.handedness[index][0].category_name

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                x1 = max(0.0, min(xs))
                y1 = max(0.0, min(ys))
                x2 = min(1.0, max(xs))
                y2 = min(1.0, max(ys))

                gesture = GestureRecognizer().classify(points)

                score = 1.0

                if result.handedness and index < len(result.handedness):
                    if result.handedness[index]:
                        score = float(
                            result.handedness[index][0].score
                        )

                if score < confidence:
                    continue

                hands.append(
                    HandState(
                        hand_id=index + 1,
                        handedness=handedness,
                        landmarks=points,
                        bbox=(x1, y1, x2, y2),
                        gesture=gesture,
                        confidence=score
                    )
                )

            return hands

        except Exception as exc:
            self.error = str(exc)
            return []


class EventEngine:
    def __init__(self):
        self.events = deque(maxlen=80)
        self.last_values = {}
        self.last_event_times = {}

    def clear(self):
        self.events.clear()
        self.last_values.clear()
        self.last_event_times.clear()

    def add(self, category, value):
        now = time.time()

        previous = self.last_values.get(category)

        if previous == value:
            return

        last_time = self.last_event_times.get(category, 0)

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

        for event in reversed(self.events):
            stamp = time.strftime(
                "%H:%M:%S",
                time.localtime(event.timestamp)
            )
            lines.append(
                f"[{stamp}] {event.text}"
            )

        return "\n".join(lines)


class LifeVisionEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.input_condition = threading.Condition(self.lock)

        self.latest_input = None
        self.latest_output = None
        self.latest_snapshot = Snapshot()

        self.frame_id = 0
        self.camera_frames = 0
        self.camera_start = time.time()

        self.process_fps_target = 4.0
        self.last_ai_time = 0.0
        self.ai_times = deque(maxlen=20)

        self.running = True

        self.mode = "Live Scene"
        self.object_confidence = 0.35
        self.pose_confidence = 0.35
        self.hand_confidence = 0.45

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

        self.device = "cpu"

        if torch is not None:
            try:
                if torch.cuda.is_available():
                    self.device = "cuda"
            except Exception:
                self.device = "cpu"

        self.object_model = None
        self.pose_model = None

        self.object_error = ""
        self.pose_error = ""

        self.object_tracker = Tracker(max_distance=150)
        self.person_tracker = Tracker(max_distance=180)
        self.motion = MotionAnalyzer()
        self.posture = PostureAnalyzer()
        self.gesture = GestureRecognizer()
        self.events = EventEngine()
        self.hand_engine = HandLandmarkerEngine()

        self.people = {}
        self.last_objects = []
        self.last_hands = []

        self.worker = threading.Thread(
            target=self.worker_loop,
            daemon=True
        )
        self.worker.start()

        self.model_loader = threading.Thread(
            target=self.load_models,
            daemon=True
        )
        self.model_loader.start()

    def load_models(self):
        try:
            self.object_model = YOLO(OBJECT_MODEL)
        except Exception as exc:
            self.object_error = f"Object model error: {exc}"

        try:
            self.pose_model = YOLO(POSE_MODEL)
        except Exception as exc:
            self.pose_error = f"Pose model error: {exc}"

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
            self.mode = mode or "Live Scene"
            self.process_fps_target = max(
                1.0,
                min(10.0, float(process_fps or 4.0))
            )

            self.object_confidence = float(
                object_confidence or 0.35
            )

            self.pose_confidence = float(
                pose_confidence or 0.35
            )

            self.hand_confidence = float(
                hand_confidence or 0.45
            )

            self.show_boxes = bool(show_boxes)
            self.show_pose = bool(show_pose)
            self.show_hands = bool(show_hands)
            self.show_labels = bool(show_labels)
            self.show_hud = bool(show_hud)
            self.mirror = bool(mirror)

            self.groups = list(groups or ["People"])

    def submit(self, frame):
        if frame is None:
            return self.get_output()

        if not isinstance(frame, np.ndarray):
            return self.get_output()

        with self.lock:
            self.camera_frames += 1

            if self.mirror:
                frame = cv2.flip(frame, 1)

            self.latest_input = frame.copy()
            self.frame_id += 1

            self.input_condition.notify()

            output = (
                self.latest_output.copy()
                if self.latest_output is not None
                else frame.copy()
            )

            snapshot = self.latest_snapshot

        return output, self.format_snapshot(snapshot)

    def get_output(self):
        with self.lock:
            if self.latest_output is None:
                output = np.zeros(
                    (480, 640, 3),
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
            else:
                output = self.latest_output.copy()

            return output, self.format_snapshot(
                self.latest_snapshot
            )

    def worker_loop(self):
        while self.running:
            with self.input_condition:
                if self.latest_input is None:
                    self.input_condition.wait(timeout=0.25)

                if self.latest_input is None:
                    continue

                frame = self.latest_input.copy()

                self.latest_input = None

            interval = 1.0 / max(
                1.0,
                self.process_fps_target
            )

            now = time.time()

            elapsed = now - self.last_ai_time

            if elapsed < interval:
                time.sleep(
                    min(
                        interval - elapsed,
                        0.05
                    )
                )

            try:
                output, snapshot = self.process_frame(
                    frame
                )

                with self.lock:
                    self.latest_output = output
                    self.latest_snapshot = snapshot

            except Exception as exc:
                with self.lock:
                    self.latest_snapshot.runtime_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

                    if self.latest_output is None:
                        self.latest_output = frame.copy()

    def detect_objects(self, frame):
        if self.object_model is None:
            return [], []

        if self.mode == "Gesture & Body Awareness":
            allowed = set(["person"])
        else:
            allowed = set()

            for group in self.groups:
                allowed.update(
                    OBJECT_GROUPS.get(group, [])
                )

            allowed.add("person")

        try:
            result = self.object_model.predict(
                source=frame,
                conf=self.object_confidence,
                iou=0.45,
                imgsz=416,
                max_det=40,
                device=self.device,
                verbose=False
            )[0]

            names = self.object_model.names

            detections = []

            if result.boxes is None:
                return [], []

            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = names[cls_id]

                if label not in allowed:
                    continue

                x1, y1, x2, y2 = [
                    int(v)
                    for v in box.xyxy[0].tolist()
                ]

                center = (
                    (x1 + x2) // 2,
                    (y1 + y2) // 2
                )

                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                        "center": center,
                        "area": max(
                            1,
                            (x2 - x1) * (y2 - y1)
                        )
                    }
                )

            tracked = self.object_tracker.update(
                detections
            )

            objects = [
                ObjectState(
                    label=item["label"],
                    confidence=item["confidence"],
                    bbox=item["bbox"],
                    track_id=item["track_id"],
                    center=item["center"],
                    area=item["area"]
                )
                for item in tracked
            ]

            return objects, result

        except Exception as exc:
            self.object_error = str(exc)
            return [], []

    def synchronize_people(self, objects):
        person_objects = [
            obj for obj in objects
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

        tracked = self.person_tracker.update(
            detections
        )

        current = {}

        for item in tracked:
            person_id = item["track_id"]

            previous = self.people.get(person_id)

            if previous is None:
                person = PersonState(
                    person_id=person_id,
                    bbox=item["bbox"],
                    center=item["center"],
                    confidence=item["confidence"],
                    last_center=item["center"]
                )
            else:
                person = previous
                person.last_center = person.center
                person.center = item["center"]
                person.bbox = item["bbox"]
                person.confidence = item["confidence"]

            movement, velocity = self.motion.calculate(
                person
            )

            person.movement = movement
            person.velocity = velocity
            person.history.append(person.center)
            person.last_update = time.time()

            current[person_id] = person

        self.people = current

        return list(current.values())

    def detect_pose(self, frame, people):
        if (
            self.pose_model is None
            or not people
            or self.mode == "Object & People Awareness"
            or not self.show_pose
        ):
            return

        try:
            result = self.pose_model.predict(
                source=frame,
                conf=self.pose_confidence,
                iou=0.45,
                imgsz=416,
                max_det=min(12, len(people) + 3),
                device=self.device,
                verbose=False
            )[0]

            if result.keypoints is None:
                return

            if result.boxes is None:
                return

            pose_items = []

            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i].tolist()

                keypoints = result.keypoints.data[
                    i
                ].cpu().numpy()

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
                    if person.person_id in used_people:
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

                if best_person is None or best_iou < 0.15:
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

        except Exception as exc:
            self.pose_error = str(exc)

    def iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        intersection = iw * ih

        if intersection <= 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(
            0,
            ay2 - ay1
        )

        area_b = max(0, bx2 - bx1) * max(
            0,
            by2 - by1
        )

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def detect_hands(self, frame, people):
        if (
            not self.show_hands
            or self.mode == "Object & People Awareness"
        ):
            return []

        hands = self.hand_engine.detect(
            frame,
            self.hand_confidence
        )

        if not hands:
            return []

        height, width = frame.shape[:2]

        for hand in hands:
            cx = (
                hand.bbox[0] +
                hand.bbox[2]
            ) * 0.5 * width

            cy = (
                hand.bbox[1] +
                hand.bbox[3]
            ) * 0.5 * height

            nearest = None
            nearest_distance = float("inf")

            for person in people:
                px, py = person.center

                if (
                    person.bbox[0] <= cx <= person.bbox[2]
                    and
                    person.bbox[1] <= cy <= person.bbox[3]
                ):
                    nearest = person
                    break

                distance = math.hypot(
                    cx - px,
                    cy - py
                )

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest = person

            if nearest is not None:
                nearest.hands.append(hand)

                if hand.gesture not in nearest.gestures:
                    nearest.gestures.append(
                        hand.gesture
                    )

        return hands

    def build_scene(self, objects, people):
        count = len(people)

        animal_count = sum(
            1
            for obj in objects
            if obj.label in OBJECT_GROUPS["Animals"]
        )

        vehicle_count = sum(
            1
            for obj in objects
            if obj.label in OBJECT_GROUPS["Vehicles"]
        )

        non_people = [
            obj
            for obj in objects
            if obj.label != "person"
        ]

        if count == 0 and animal_count == 0 and not non_people:
            return "No recognized subjects"

        if count > 1 and animal_count > 0:
            return "Multiple humans and animals"

        if count > 1 and non_people:
            return "Multiple humans and objects"

        if count > 1:
            return "Multiple humans"

        if count == 1 and animal_count > 0:
            return "Human and animal"

        if count == 1 and vehicle_count > 0:
            return "Human and vehicle"

        if count == 1 and non_people:
            return "Human and objects"

        if count == 1:
            return "Single human"

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

        for person in sorted(
            people,
            key=lambda p: p.person_id
        ):
            details = []

            if person.posture != "Unknown":
                details.append(
                    person.posture.lower()
                )

            if person.movement != "Still":
                details.append(
                    person.movement.lower()
                )

            if person.hands:
                hand_count = len(person.hands)

                details.append(
                    f"{hand_count} hand"
                    f"{'s' if hand_count != 1 else ''}"
                    " visible"
                )

            gestures = [
                g
                for g in person.gestures
                if g not in ("Unknown",)
            ]

            if gestures:
                details.append(
                    "gesture: " +
                    ", ".join(dict.fromkeys(gestures))
                )

            if details:
                parts.append(
                    f"Person {person.person_id} is "
                    + ", ".join(details)
                    + "."
                )

            if person.keypoints:
                visible = []

                for index, name in POSE_NAMES.items():
                    if index >= len(person.keypoints):
                        continue

                    if person.keypoints[index][2] >= 0.4:
                        visible.append(name)

                if visible:
                    parts.append(
                        f"Person {person.person_id} body landmarks: "
                        + ", ".join(visible[:8])
                        + "."
                    )

        counts = {}

        for obj in objects:
            if obj.label == "person":
                continue

            counts[obj.label] = (
                counts.get(obj.label, 0) + 1
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
                "Visible objects: "
                + ", ".join(object_text)
                + "."
            )

        if hands:
            gesture_counts = {}

            for hand in hands:
                gesture_counts[hand.gesture] = (
                    gesture_counts.get(
                        hand.gesture,
                        0
                    ) + 1
                )

            meaningful = [
                f"{count} {gesture}"
                for gesture, count
                in gesture_counts.items()
                if gesture != "Unknown"
            ]

            if meaningful:
                parts.append(
                    "Hand activity: "
                    + ", ".join(meaningful)
                    + "."
                )

        parts.append(
            f"Scene classification: {scene}."
        )

        return " ".join(parts)

    def update_events(self, scene, people):
        self.events.add(
            "scene",
            f"Scene changed to {scene}"
        )

        self.events.add(
            "people",
            f"People count: {len(people)}"
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
                        set(person.gestures)
                    )
                )

                self.events.add(
                    f"gesture_{person.person_id}",
                    f"Person {person.person_id}: "
                    f"{gestures}"
                )

    def process_frame(self, frame):
        started = time.time()

        objects, _ = self.detect_objects(
            frame
        )

        people = self.synchronize_people(
            objects
        )

        self.detect_pose(
            frame,
            people
        )

        hands = self.detect_hands(
            frame,
            people
        )

        scene = self.build_scene(
            objects,
            people
        )

        narrative = self.build_narrative(
            objects,
            people,
            hands,
            scene
        )

        self.update_events(
            scene,
            people
        )

        output = self.render(
            frame,
            objects,
            people,
            hands,
            scene,
            narrative
        )

        processing_time = max(
            0.0001,
            time.time() - started
        )

        ai_fps = 1.0 / processing_time

        self.ai_times.append(
            processing_time
        )

        average_processing = (
            sum(self.ai_times) /
            max(1, len(self.ai_times))
        )

        stable_ai_fps = (
            1.0 / average_processing
        )

        now = time.time()

        self.last_ai_time = now

        with self.lock:
            self.frame_id += 0

            snapshot = Snapshot(
                timestamp=now,
                frame_id=self.frame_id,
                objects=objects,
                people=people,
                hands=hands,
                scene=scene,
                narrative=narrative,
                fps=self.calculate_camera_fps(),
                ai_fps=stable_ai_fps,
                hand_available=self.hand_engine.available,
                runtime_error=""
            )

        self.last_objects = objects
        self.last_hands = hands

        return output, snapshot

    def calculate_camera_fps(self):
        elapsed = max(
            0.001,
            time.time() - self.camera_start
        )

        return self.camera_frames / elapsed

    def render(
        self,
        frame,
        objects,
        people,
        hands,
        scene,
        narrative
    ):
        output = frame.copy()

        if self.show_boxes:
            for obj in objects:
                x1, y1, x2, y2 = obj.bbox

                if obj.label == "person":
                    thickness = 2
                else:
                    thickness = 1

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (255, 210, 50),
                    thickness
                )

                if self.show_labels:
                    label = (
                        f"{obj.label} "
                        f"{obj.confidence:.0%}"
                    )

                    if obj.label == "person":
                        label += (
                            f"  P{obj.track_id}"
                        )

                    self.draw_label(
                        output,
                        label,
                        x1,
                        max(20, y1 - 5)
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

    def draw_pose(self, frame, keypoints):
        if not keypoints:
            return

        for a, b in POSE_CONNECTIONS:
            if (
                a >= len(keypoints)
                or b >= len(keypoints)
            ):
                continue

            p1 = keypoints[a]
            p2 = keypoints[b]

            if p1[2] < 0.35 or p2[2] < 0.35:
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

    def draw_hand(self, frame, hand):
        height, width = frame.shape[:2]

        points = []

        for point in hand.landmarks:
            x = int(point[0] * width)
            y = int(point[1] * height)

            points.append((x, y))

        for a, b in HAND_CONNECTIONS:
            if a >= len(points) or b >= len(points):
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

        x1 = int(hand.bbox[0] * width)
        y1 = int(hand.bbox[1] * height)

        self.draw_label(
            frame,
            f"{hand.handedness}: {hand.gesture}",
            x1,
            max(20, y1 - 5)
        )

    def draw_label(
        self,
        frame,
        text,
        x,
        y
    ):
        font = cv2.FONT_HERSHEY_SIMPLEX
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
                x,
                frame.shape[1] - size[0] - 8
            )
        )

        y = max(
            size[1] + 8,
            min(
                y,
                frame.shape[0] - 4
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

    def draw_hud(
        self,
        frame,
        objects,
        people,
        hands,
        scene,
        narrative
    ):
        height, width = frame.shape[:2]

        overlay_height = 86

        overlay = frame[
            0:overlay_height,
            0:width
        ].copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (width, overlay_height),
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

        fps = self.calculate_camera_fps()

        line1 = (
            f"LIFEVISION  |  {self.mode}  |  "
            f"Scene: {scene}"
        )

        line2 = (
            f"People: {len(people)}  |  "
            f"Objects: {len(objects)}  |  "
            f"Hands: {len(hands)}  |  "
            f"Camera: {fps:.1f} FPS"
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
            line2,
            (14, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 220, 240),
            1,
            cv2.LINE_AA
        )

        narrative_short = narrative[:115]

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

    def format_snapshot(self, snapshot):
        people_lines = []

        for person in sorted(
            snapshot.people,
            key=lambda p: p.person_id
        ):
            line = (
                f"Person {person.person_id} | "
                f"{person.posture} | "
                f"{person.movement}"
            )

            if person.gestures:
                line += (
                    " | "
                    + ", ".join(
                        sorted(
                            set(person.gestures)
                        )
                    )
                )

            if person.hands:
                line += (
                    f" | Hands: {len(person.hands)}"
                )

            people_lines.append(line)

        object_counts = {}

        for obj in snapshot.objects:
            if obj.label == "person":
                continue

            object_counts[obj.label] = (
                object_counts.get(
                    obj.label,
                    0
                ) + 1
            )

        object_lines = []

        for label, count in sorted(
            object_counts.items()
        ):
            object_lines.append(
                f"{label}: {count}"
            )

        if not object_lines:
            object_text = "No non-person objects recognized."
        else:
            object_text = "\n".join(
                object_lines
            )

        if not people_lines:
            people_text = (
                "No confirmed people."
            )
        else:
            people_text = "\n".join(
                people_lines
            )

        events_text = self.events.formatted()

        hand_status = (
            "Available"
            if snapshot.hand_available
            else "Unavailable"
        )

        runtime_errors = []

        if self.object_error:
            runtime_errors.append(
                "Object: " + self.object_error
            )

        if self.pose_error:
            runtime_errors.append(
                "Pose: " + self.pose_error
            )

        if self.hand_engine.error:
            runtime_errors.append(
                "Hands: " + self.hand_engine.error
            )

        if snapshot.runtime_error:
            runtime_errors.append(
                "Runtime: " + snapshot.runtime_error
            )

        if runtime_errors:
            diagnostic = "\n".join(
                runtime_errors
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
                f"Camera FPS: {snapshot.fps:.1f}\n"
                f"AI FPS: {snapshot.ai_fps:.1f}\n"
                f"People: {len(snapshot.people)}\n"
                f"Objects: {len(snapshot.objects)}\n"
                f"Hands: {len(snapshot.hands)}\n"
                f"Mode: {self.mode}"
            ),
            "hands": (
                f"Hand Engine: {hand_status}\n"
                f"Detected hands: "
                f"{len(snapshot.hands)}"
            ),
            "diagnostic": diagnostic
        }

    def clear_events(self):
        with self.lock:
            self.events.clear()

    def stop(self):
        self.running = False

        with self.input_condition:
            self.input_condition.notify_all()


ENGINE = LifeVisionEngine()


def process_frame(
    frame,
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
    ENGINE.update_config(
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
    )

    output, data = ENGINE.submit(frame)

    return (
        output,
        data["metrics"],
        data["scene"],
        data["narrative"],
        data["people"],
        data["objects"],
        data["events"],
        data["hands"],
        data["diagnostic"]
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

.metric-box textarea {
    font-size: 15px !important;
}

.narrative-box textarea {
    font-size: 15px !important;
}

.status-title {
    font-weight: 700;
}
"""


with gr.Blocks(
    title="LifeVision"
) as demo:

    gr.Markdown(
        """
# LifeVision
### Real-Time Computer Vision & Scene Awareness
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
                label="LifeVision Camera",
                elem_id="camera_output"
            )

        with gr.Column(
            scale=3,
            min_width=320
        ):

            gr.Markdown("### Live Status")

            metrics = gr.Textbox(
                label="System Metrics",
                value="Waiting for camera...",
                lines=6,
                interactive=False
            )

            scene = gr.Textbox(
                label="Scene",
                value="Waiting for camera...",
                lines=2,
                interactive=False
            )

            narrative = gr.Textbox(
                label="Live Understanding",
                value="Start the camera to begin LifeVision.",
                lines=7,
                interactive=False,
                elem_classes=["narrative-box"]
            )

    with gr.Accordion(
        "LifeVision Controls",
        open=False
    ):

        with gr.Row():

            mode = gr.Radio(
                choices=[
                    "Object & People Awareness",
                    "Gesture & Body Awareness",
                    "Live Scene"
                ],
                value="Live Scene",
                label="Mode"
            )

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
                value=0.35,
                step=0.05,
                label="Object Confidence"
            )

            pose_confidence = gr.Slider(
                minimum=0.10,
                maximum=0.90,
                value=0.35,
                step=0.05,
                label="Pose Confidence"
            )

            hand_confidence = gr.Slider(
                minimum=0.10,
                maximum=0.90,
                value=0.45,
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
                value="No non-person objects recognized.",
                lines=8,
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
                value="Hand Engine: Loading...",
                lines=5,
                interactive=False
            )

            diagnostic_output = gr.Textbox(
                label="Runtime Diagnostic",
                value="Loading AI engines...",
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
        outputs=events_output
    )

    stream_inputs = [
        camera,
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
    ]

    stream_outputs = [
        camera,
        metrics,
        scene,
        narrative,
        people_output,
        objects_output,
        events_output,
        hands_output,
        diagnostic_output
    ]

    camera.stream(
        fn=process_frame,
        inputs=stream_inputs,
        outputs=stream_outputs,
        stream_every=0.08,
        concurrency_limit=1
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS
    )