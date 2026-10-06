# LifeVision

LifeVision is a local webcam app with two focused capabilities:

1. Person detection, visible-person counting, and temporary ID tracking.
2. Broad prompted everyday-object detection, visible counting, and per-class ID
   tracking.

It uses one open-vocabulary YOLO-World detector for both capabilities and
separate ByteTrack association for people and each object class. Detection
weights are loaded once per app process; tracker state is isolated per browser
session.

## Object vocabulary and limitations

The detector is prompted with the explicit class list in [vision.py](./vision.py).
It covers common COCO objects and additional prompted categories for produce,
clothing, stationery, appliances, electronics, furniture, vehicles, and
animals. YOLO-World is an open-vocabulary model: these prompts are not custom
training, and reliability varies by category, lighting, object size, camera
quality, and occlusion. The application does not label categories outside its
configured prompt list. A confidence score is model confidence, not a guarantee
of correctness.

The detector uses a 640-pixel inference size, 0.10 model confidence floor,
0.55 IoU NMS threshold, and a 0.25 display threshold. A stricter cross-label
overlap check suppresses near-identical boxes assigned competing class names.
The ByteTrack buffer retains recently missed IDs for up to roughly 1.5 seconds
at the current processing rate (bounded to 5–30 frames). Visible counts remain
per-frame counts; tracked counts include recently lost tracks during that
grace period. IDs are temporary and local to one camera session, not
cross-session identities or cumulative entry counts. Object counts exclude
people.

YOLO-World small is larger and slower on CPU than a nano closed-set model; on
the development machine a warm 640-pixel synthetic-frame run was about 6 FPS.
The UI reports measured processing FPS and frame time. Actual throughput
depends on the computer, camera, and available accelerator. First startup
downloads the approximately 25 MB detector and OpenAI CLIP text-encoder
weights (about 354 MB), and may take a few minutes. Both are cached outside
source control; internet access is needed for the initial downloads.

The webcam requests up to 1280×720 at 30 FPS. Video callbacks return promptly
and a per-session worker runs inference separately, capped at 10 starts per
second. The worker keeps only one pending camera frame and replaces it with
new arrivals, avoiding a growing queue of stale frames. The newest detections
are overlaid on subsequent live frames; overlays older than one second are
hidden rather than left at stale positions. Stopping the camera clears the
session tracker state; restarting creates fresh tracks while reusing the
process-cached detector.

## Run locally on Windows

From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit and allow camera access in the browser.
The browser must support WebRTC. Public deployments need HTTPS and a network
configuration that permits WebRTC connections.

## Test

```powershell
python -m pip install pytest
python -m pytest
```

Tests exercise prompt configuration, confidence/NMS settings, duplicate label
suppression, per-frame counts, separate person/object trackers, identity
retention across a missed frame, and track expiry after a person leaves.
