import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".thaidancescoring_mpl"))
import sys

# Redirect stdout/stderr to log file when running as bundled app
# ใช้ ~/Documents เพื่อแก้ปัญหา Read-only บน macOS (App Translocation)
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "Documents", "ThaiDanceScoring")
try:
    os.makedirs(USER_DATA_DIR, exist_ok=True)
except Exception:
    pass

LOG_FILE = os.path.join(USER_DATA_DIR, "debug.log")
if getattr(sys, 'frozen', False):
    try:
        log_handle = open(LOG_FILE, 'w', buffering=1)  # line buffered
        sys.stdout = log_handle
        sys.stderr = log_handle
        print(f"[LOG] App started at {os.getcwd()}")
        print(f"[LOG] sys.executable = {sys.executable}")
        print(f"[LOG] sys._MEIPASS = {getattr(sys, '_MEIPASS', 'N/A')}")
        print(f"[LOG] USER_DATA_DIR = {USER_DATA_DIR}")
    except Exception as e:
        pass  # Ignore if can't write log
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
import math
import time
import json
import urllib.request
import urllib.parse
import threading
import mimetypes
import webbrowser
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import mediapipe as mp
import subprocess
import tempfile

from PySide6.QtCore import Qt, QTimer, QSize, QObject, Signal, Slot, QThread, QUrl
from PySide6.QtGui import QImage, QPixmap, QFont, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QGridLayout, QGroupBox, QProgressBar,
    QSpinBox, QDoubleSpinBox, QMessageBox, QStackedWidget, QScrollArea, QFrame,
    QCheckBox, QLineEdit, QComboBox, QSizePolicy, QSplitter, QSlider
)

# Qt Multimedia for audio playback
QT_MULTIMEDIA_OK = True
try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
except ImportError:
    QT_MULTIMEDIA_OK = False
    QMediaPlayer = None
    QAudioOutput = None

# =========================
# Google Sheet Web App URL
# =========================
GSHEET_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbxolQRtIY9uWyYBX94TYyZ8--Ppc43OvNOxWxy7LLfln_xe5hBNnPK_EDEbo61LPBkbDg/exec"

# =========================
# Teacher video dropdown dir
# =========================
# ใช้ resource_path() สำหรับ bundled app
# ส่วน recordings และ token ใช้ USER_DATA_DIR เพื่อเลี่ยงปัญหา Read-only
if getattr(sys, 'frozen', False):
    # Running as bundled app - use executable directory
    APP_DIR = os.path.dirname(sys.executable)
    # For macOS .app bundle, go up from MacOS folder to the app bundle root
    if APP_DIR.endswith('/Contents/MacOS'):
        APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(APP_DIR)))
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"[DEBUG] APP_DIR = {APP_DIR}")
TEACHER_VIDEO_DIR = resource_path("teacher_videos")
print(f"[DEBUG] TEACHER_VIDEO_DIR = {TEACHER_VIDEO_DIR}")

# Use user writable dir for recordings
RECORDINGS_DIR = os.path.join(USER_DATA_DIR, "recordings")
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")

# =========================
# Google Drive Upload (OAuth)
# =========================
DRIVE_OK = True
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except Exception:
    DRIVE_OK = False

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]  # upload files created/selected by this app
DRIVE_CREDENTIALS_FILE = resource_path("credentials.json")
DRIVE_TOKEN_FILE = os.path.join(USER_DATA_DIR, "token.json")

# YOLO model path
YOLO_MODEL_PATH = resource_path("yolov8n.pt")

# Put a folder ID here to upload into a specific folder (empty = My Drive root)
DRIVE_FOLDER_ID = ""


def ensure_teacher_dir():
    try:
        os.makedirs(TEACHER_VIDEO_DIR, exist_ok=True)
    except Exception:
        pass


def ensure_recordings_dir():
    try:
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
    except Exception:
        pass


def list_teacher_videos_in_dir() -> List[str]:
    ensure_teacher_dir()
    out = []
    try:
        for fn in sorted(os.listdir(TEACHER_VIDEO_DIR)):
            if fn.lower().endswith(VIDEO_EXTS):
                out.append(os.path.join(TEACHER_VIDEO_DIR, fn))
    except Exception:
        pass
    return out


def probe_video(path: str) -> Tuple[float, int, float]:
    """return fps, frame_count, duration_sec (best effort)"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 30.0, 0, 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1e-3:
        fps = 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = (frames / fps) if (frames > 0 and fps > 1e-6) else 0.0
    try:
        cap.release()
    except Exception:
        pass
    return float(fps), int(frames), float(dur)


def post_score_to_gsheet(payload: Dict):
    base = (GSHEET_WEBAPP_URL or "").strip()
    if not base or (not base.startswith("https://script.google.com/macros/s/")) or (not base.endswith("/exec")):
        print("[WARN] GSHEET_WEBAPP_URL not set or invalid, skip posting.")
        return
    try:
        payload_json = json.dumps(payload, ensure_ascii=False)
        payload_q = urllib.parse.quote(payload_json, safe="")
        url = f"{base}?payload={payload_q}"
        if len(url) > 15000:
            print("[WARN] Payload too large for GET URL, skip posting.")
            return
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=25) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            print("[INFO] Google Sheet response:", body[:300])
    except Exception as e:
        print("[WARN] Google Sheet logging failed:", e)


def _get_drive_service():
    if not DRIVE_OK:
        raise RuntimeError("Drive libs not installed. Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlab")
    if not os.path.exists(DRIVE_CREDENTIALS_FILE):
        raise RuntimeError(f"Missing credentials.json at: {DRIVE_CREDENTIALS_FILE}")

    creds = None
    if os.path.exists(DRIVE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(DRIVE_TOKEN_FILE, DRIVE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"[WARN] Token refresh failed: {e}. Clearing creds to re-login.")
                creds = None
        
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(DRIVE_CREDENTIALS_FILE, DRIVE_SCOPES)
            # Use run_local_server with open_browser=True to ensure browser opens
            # If that fails, manually open browser
            try:
                # First try with open_browser=True
                creds = flow.run_local_server(port=0, open_browser=True)
            except Exception as e:
                print(f"[WARN] run_local_server failed: {e}")
                # Try manual browser open approach
                auth_url, _ = flow.authorization_url(access_type='offline')
                print(f"[INFO] Opening browser for login: {auth_url}")
                # Try to open browser manually
                try:
                    webbrowser.open(auth_url)
                except Exception as e2:
                    print(f"[WARN] Could not open browser: {e2}")
                # Show a message to user about the login URL
                raise RuntimeError(f"กรุณาเปิด Browser แล้วไปที่ URL นี้เพื่อ Login:\n\n{auth_url}")
        
        # Try to save token, ignore if fails (at least we are logged in for this session)
        try:
            with open(DRIVE_TOKEN_FILE, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            print(f"[INFO] Token saved to: {DRIVE_TOKEN_FILE}")
        except Exception as e:
            print(f"[ERR] Could not save token.json: {e}")

    return build("drive", "v3", credentials=creds)


def upload_video_to_drive(file_path: str, folder_id: str = "", make_shared: bool = False) -> dict:
    """
    Upload video via Google Drive Official API (resumable)
    return dict: {id, name, webViewLink} or {ok, fileId, webViewLink}
    """
    if not file_path or (not os.path.exists(file_path)):
        raise RuntimeError(f"File not found: {file_path}")

    # Authenticate
    service = _get_drive_service()
    
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    print(f"[INFO] Uploading: {file_name} ({file_size / (1024*1024):.2f} MB)")

    # Prepare metadata
    file_metadata = {'name': file_name}
    if folder_id:
        file_metadata['parents'] = [folder_id]

    # Chunk size for resumable upload (must be multiple of 256KB)
    CHUNK_SIZE = 5 * 1024 * 1024 # 5MB chunks

    media = MediaFileUpload(
        file_path, 
        mimetype=mimetypes.guess_type(file_path)[0] or 'application/octet-stream',
        resumable=True,
        chunksize=CHUNK_SIZE
    )

    request = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, name, webViewLink'
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"[INFO] Uploaded {int(status.progress() * 100)}%")
    
    # If shared requested
    if make_shared:
        try:
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            service.permissions().create(
                fileId=response.get('id'),
                body=permission,
                fields='id',
            ).execute()
        except Exception as e:
            print(f"[WARN] Failed to make file shared: {e}")

    return {
        "id": response.get('id'),
        "name": response.get('name'),
        "webViewLink": response.get('webViewLink')
    }


class DriveUploadThread(QThread):
    done = Signal(object)  # dict {ok, info|error}

    def __init__(self, file_path: str, folder_id: str = "", make_shared: bool = False):
        super().__init__()
        self.file_path = file_path
        self.folder_id = folder_id
        self.make_shared = bool(make_shared)

    def run(self):
        try:
            info = upload_video_to_drive(self.file_path, folder_id=self.folder_id, make_shared=self.make_shared)
            self.done.emit({"ok": True, "info": info})
        except Exception as e:
            self.done.emit({"ok": False, "error": str(e)})


# -----------------------------
# Utilities
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def resize_limit(frame, max_w=860):
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    s = max_w / float(w)
    nh = int(round(h * s))
    return cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_AREA)


def make_canvas_fit(frame_bgr, canvas_w: int, canvas_h: int):
    canvas_w = max(2, int(canvas_w))
    canvas_h = max(2, int(canvas_h))

    h, w = frame_bgr.shape[:2]
    if h < 2 or w < 2:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        return canvas, 1.0, 0, 0

    s = min(canvas_w / float(w), canvas_h / float(h))
    nw = max(2, int(round(w * s)))
    nh = max(2, int(round(h * s)))

    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    x0 = (canvas_w - nw) // 2
    y0 = (canvas_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, s, x0, y0


def bgr_canvas_to_qpixmap(canvas_bgr):
    rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def sec_to_mmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


# -----------------------------
# Pose scoring
# -----------------------------
def angle_3pts(a, b, c):
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return None
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = clamp(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def robust_scale(pts, left_shoulder, right_shoulder, left_hip, right_hip):
    def dist(i, j):
        if pts[i] is None or pts[j] is None:
            return None
        return float(np.linalg.norm(pts[i] - pts[j]))

    d1 = dist(left_shoulder, right_shoulder)
    d2 = dist(left_hip, right_hip)
    if d1 and d1 > 1e-6:
        return d1
    if d2 and d2 > 1e-6:
        return d2

    if (pts[left_shoulder] is not None and pts[right_shoulder] is not None and
            pts[left_hip] is not None and pts[right_hip] is not None):
        mid_sh = (pts[left_shoulder] + pts[right_shoulder]) * 0.5
        mid_hp = (pts[left_hip] + pts[right_hip]) * 0.5
        d3 = float(np.linalg.norm(mid_sh - mid_hp))
        if d3 > 1e-6:
            return d3
    return None


def normalize_pose_points(points, idx_left_sh, idx_right_sh, idx_left_hip, idx_right_hip):
    pts = points
    origin = None

    if pts[idx_left_hip] is not None and pts[idx_right_hip] is not None:
        origin = (pts[idx_left_hip] + pts[idx_right_hip]) * 0.5
    elif pts[idx_left_sh] is not None and pts[idx_right_sh] is not None:
        origin = (pts[idx_left_sh] + pts[idx_right_sh]) * 0.5
    else:
        for p in pts:
            if p is not None:
                origin = p.copy()
                break

    if origin is None:
        return None

    scale = robust_scale(pts, idx_left_sh, idx_right_sh, idx_left_hip, idx_right_hip)
    if scale is None or scale < 1e-6:
        return None

    out = []
    for p in pts:
        out.append(None if p is None else (p - origin) / scale)
    return out


def build_pose_features(norm_pts, lm):
    PL = mp.solutions.pose.PoseLandmark

    def pt(i):
        return norm_pts[i] if norm_pts is not None else None

    L_SH = PL.LEFT_SHOULDER.value
    R_SH = PL.RIGHT_SHOULDER.value
    L_EL = PL.LEFT_ELBOW.value
    R_EL = PL.RIGHT_ELBOW.value
    L_WR = PL.LEFT_WRIST.value
    R_WR = PL.RIGHT_WRIST.value
    L_HI = PL.LEFT_HIP.value
    R_HI = PL.RIGHT_HIP.value
    L_KN = PL.LEFT_KNEE.value
    R_KN = PL.RIGHT_KNEE.value
    L_AN = PL.LEFT_ANKLE.value
    R_AN = PL.RIGHT_ANKLE.value
    NOSE = PL.NOSE.value

    L_TH = PL.LEFT_THUMB.value
    R_TH = PL.RIGHT_THUMB.value
    L_IN = PL.LEFT_INDEX.value
    R_IN = PL.RIGHT_INDEX.value
    L_PI = PL.LEFT_PINKY.value
    R_PI = PL.RIGHT_PINKY.value

    def ang(a, b, c):
        pa, pb, pc = pt(a), pt(b), pt(c)
        if pa is None or pb is None or pc is None:
            return None
        return angle_3pts(pa, pb, pc)

    angles = {
        "L_ELBOW": ang(L_SH, L_EL, L_WR),
        "R_ELBOW": ang(R_SH, R_EL, R_WR),
        "L_SHOULDER": ang(L_EL, L_SH, L_HI),
        "R_SHOULDER": ang(R_EL, R_SH, R_HI),
        "L_KNEE": ang(L_HI, L_KN, L_AN),
        "R_KNEE": ang(R_HI, R_KN, R_AN),
    }

    vecs: Dict[str, Optional[np.ndarray]] = {}

    def unit_vec(a, b, name):
        pa, pb = pt(a), pt(b)
        if pa is None or pb is None:
            vecs[name] = None
            return
        v = pb - pa
        n = np.linalg.norm(v)
        vecs[name] = None if n < 1e-6 else (v / n)

    unit_vec(L_SH, L_EL, "L_UPPER_ARM")
    unit_vec(R_SH, R_EL, "R_UPPER_ARM")
    unit_vec(L_EL, L_WR, "L_FOREARM")
    unit_vec(R_EL, R_WR, "R_FOREARM")

    if pt(L_SH) is not None and pt(R_SH) is not None and pt(L_HI) is not None and pt(R_HI) is not None:
        mid_sh = (pt(L_SH) + pt(R_SH)) * 0.5
        mid_hp = (pt(L_HI) + pt(R_HI)) * 0.5
        v = mid_hp - mid_sh
        n = np.linalg.norm(v)
        vecs["TORSO"] = None if n < 1e-6 else (v / n)
    else:
        vecs["TORSO"] = None

    if pt(L_SH) is not None and pt(R_SH) is not None and pt(NOSE) is not None:
        mid_sh = (pt(L_SH) + pt(R_SH)) * 0.5
        v = pt(NOSE) - mid_sh
        n = np.linalg.norm(v)
        vecs["HEAD"] = None if n < 1e-6 else (v / n)
    else:
        vecs["HEAD"] = None

    unit_vec(L_WR, L_IN, "L_HAND_INDEX")
    unit_vec(L_WR, L_PI, "L_HAND_PINKY")
    unit_vec(L_WR, L_TH, "L_HAND_THUMB")
    unit_vec(R_WR, R_IN, "R_HAND_INDEX")
    unit_vec(R_WR, R_PI, "R_HAND_PINKY")
    unit_vec(R_WR, R_TH, "R_HAND_THUMB")

    conf = 0.0
    if lm is not None:
        ids = [
            L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HI, R_HI, L_KN, R_KN, L_AN, R_AN,
            L_TH, R_TH, L_IN, R_IN, L_PI, R_PI
        ]
        conf = float(np.mean([float(lm[i].visibility) for i in ids]))
    return {"angles": angles, "vecs": vecs, "conf": conf}


def compare_pose_features(f_user, f_teacher,
                          dead_zone_deg: float = 8.0,
                          vec_dead_cos: float = 0.985):
    if f_user is None or f_teacher is None:
        return 0.0, {"arms": 0.0, "hands": 0.0, "torso": 0.0, "legs": 0.0}

    angle_weights = {
        "L_ELBOW": 1.2, "R_ELBOW": 1.2,
        "L_SHOULDER": 1.6, "R_SHOULDER": 1.6,
        "L_KNEE": 0.8, "R_KNEE": 0.8
    }
    ANG_MAX = 60.0

    def angle_component(k):
        au = f_user["angles"].get(k)
        at = f_teacher["angles"].get(k)
        if au is None or at is None:
            return None
        d = abs(au - at)
        if d <= dead_zone_deg:
            return 1.0
        d = min(d, ANG_MAX)
        return clamp(1.0 - (d / ANG_MAX), 0.0, 1.0)

    vec_weights = {
        "L_UPPER_ARM": 1.3, "R_UPPER_ARM": 1.3,
        "L_FOREARM": 1.1, "R_FOREARM": 1.1,
        "TORSO": 0.9, "HEAD": 0.4
    }

    def vec_component(k):
        vu = f_user["vecs"].get(k)
        vt = f_teacher["vecs"].get(k)
        if vu is None or vt is None:
            return None
        cos = float(np.dot(vu, vt))
        if cos >= vec_dead_cos:
            return 1.0
        return clamp((cos + 1.0) * 0.5, 0.0, 1.0)

    def weighted_mean(keys, fn, wmap):
        s = 0.0
        w = 0.0
        for k in keys:
            comp = fn(k)
            if comp is None:
                continue
            wk = float(wmap.get(k, 1.0))
            s += comp * wk
            w += wk
        return (s / w) if w > 1e-6 else None

    arms_keys_ang = ["L_ELBOW", "R_ELBOW", "L_SHOULDER", "R_SHOULDER"]
    legs_keys_ang = ["L_KNEE", "R_KNEE"]
    arms_keys_vec = ["L_UPPER_ARM", "R_UPPER_ARM", "L_FOREARM", "R_FOREARM"]
    torso_keys_vec = ["TORSO", "HEAD"]

    arms_ang = weighted_mean(arms_keys_ang, angle_component, angle_weights)
    legs_ang = weighted_mean(legs_keys_ang, angle_component, angle_weights)
    arms_vec = weighted_mean(arms_keys_vec, vec_component, vec_weights)
    torso_vec = weighted_mean(torso_keys_vec, vec_component, vec_weights)

    if arms_ang is not None and arms_vec is not None:
        arms = 0.55 * arms_ang + 0.45 * arms_vec
    else:
        arms = arms_ang if arms_ang is not None else (arms_vec if arms_vec is not None else 0.0)

    legs = legs_ang if legs_ang is not None else 0.0
    torso = torso_vec if torso_vec is not None else 0.0

    hand_keys = [
        "L_HAND_INDEX", "L_HAND_PINKY", "L_HAND_THUMB",
        "R_HAND_INDEX", "R_HAND_PINKY", "R_HAND_THUMB"
    ]

    def hand_component(k):
        vu = f_user["vecs"].get(k)
        vt = f_teacher["vecs"].get(k)
        if vu is None or vt is None:
            return None
        cos = float(np.dot(vu, vt))
        if cos >= vec_dead_cos:
            return 1.0
        return clamp((cos + 1.0) * 0.5, 0.0, 1.0)

    hands = weighted_mean(hand_keys, hand_component, {k: 1.0 for k in hand_keys})
    hands = hands if hands is not None else 0.0

    parts = {"arms": arms, "hands": hands, "torso": torso, "legs": legs}
    overall = 0.55 * arms + 0.20 * hands + 0.15 * torso + 0.10 * legs

    conf = min(f_user.get("conf", 0.0), f_teacher.get("conf", 0.0))
    conf = clamp(conf, 0.2, 1.0)
    overall = overall * conf + (1.0 - conf) * 0.35

    return overall * 100.0, {k: v * 100.0 for k, v in parts.items()}


# -----------------------------
# MediaPipe Pose
# -----------------------------
class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb)


def landmarks_to_points_pose(results, w, h):
    if not results.pose_landmarks:
        return None, None
    lms = results.pose_landmarks.landmark
    pts = []
    for lm in lms:
        if lm.visibility < 0.4:
            pts.append(None)
        else:
            pts.append(np.array([lm.x * w, lm.y * h], dtype=np.float32))
    return pts, lms


# -----------------------------
# Multi-person detection (YOLO)
# -----------------------------
YOLO_OK = True
try:
    from ultralytics import YOLO
    print("[DEBUG] ultralytics import OK")
except Exception as e:
    print(f"[DEBUG] ultralytics import FAILED: {e}")
    YOLO_OK = False
    YOLO = None


@dataclass
class PersonDet:
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = int(clamp(x1, 0, w - 1))
    x2 = int(clamp(x2, 0, w - 1))
    y1 = int(clamp(y1, 0, h - 1))
    y2 = int(clamp(y2, 0, h - 1))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    return (x1, y1, x2, y2)


class PersonDetectorYOLO:
    def __init__(self, model_name="yolov8n.pt", conf=0.35):
        if not YOLO_OK:
            raise RuntimeError("ultralytics not installed")
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame_bgr, max_people=2) -> List[PersonDet]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # save=False prevents trying to write to read-only filesystem when launched from Finder
        # project=tempfile.gettempdir() ensures writable directory even if save=False
        import tempfile
        results = self.model.predict(
            rgb, 
            verbose=False, 
            conf=self.conf, 
            save=False,
            project=tempfile.gettempdir()
        )

        dets: List[PersonDet] = []
        if not results:
            return dets

        r = results[0]
        if r.boxes is None:
            return dets

        for b in r.boxes:
            cls = int(b.cls[0]) if hasattr(b.cls, "__len__") else int(b.cls)
            if cls != 0:
                continue
            conf = float(b.conf[0]) if hasattr(b.conf, "__len__") else float(b.conf)
            if conf < self.conf:
                continue
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            box = clamp_box(x1, y1, x2, y2, w, h)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            dets.append(PersonDet(bbox=(x1, y1, x2, y2), center=(cx, cy)))

        dets.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
        return dets[:max_people]


# -----------------------------
# Simple ID tracker (locks Person 1/2)
# -----------------------------
@dataclass
class Track:
    tid: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    missed: int = 0


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter))


class SimpleTracker:
    def __init__(self, max_tracks=2, max_missed=12):
        self.max_tracks = max_tracks
        self.max_missed = max_missed
        self.tracks: List[Track] = []
        self.next_id = 1

    def reset(self):
        self.tracks = []
        self.next_id = 1

    def set_max_tracks(self, n: int):
        n = int(max(1, n))
        if n != self.max_tracks:
            self.max_tracks = n
            self.reset()
            # Reset next_id to prevent IDs exceeding max_tracks
            self.next_id = 1

    def update(self, dets: List[PersonDet]) -> List[Track]:
        if len(self.tracks) == 0:
            dets_sorted = sorted(dets, key=lambda d: d.center[0])
            self.tracks = []
            for d in dets_sorted[:self.max_tracks]:
                self.tracks.append(Track(tid=self.next_id, bbox=d.bbox, center=d.center, missed=0))
                self.next_id += 1
            return sorted(self.tracks, key=lambda t: t.tid)

        used_det = set()
        assigned = {}

        for ti, tr in enumerate(self.tracks):
            best_j = None
            best_cost = 1e18
            tx1, ty1, tx2, ty2 = tr.bbox
            diag = math.sqrt(max(1, (tx2 - tx1) ** 2 + (ty2 - ty1) ** 2))

            for j, d in enumerate(dets):
                if j in used_det:
                    continue
                ov = iou(tr.bbox, d.bbox)
                dx = tr.center[0] - d.center[0]
                dy = tr.center[1] - d.center[1]
                dist = math.sqrt(dx * dx + dy * dy) / diag
                cost = (1.0 - ov) * 0.7 + dist * 0.3
                if ov < 0.01 and dist > 1.2:
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_j = j

            if best_j is not None:
                assigned[ti] = best_j
                used_det.add(best_j)

        for ti, tr in enumerate(self.tracks):
            if ti in assigned:
                d = dets[assigned[ti]]
                tr.bbox = d.bbox
                tr.center = d.center
                tr.missed = 0
            else:
                tr.missed += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        if len(self.tracks) < self.max_tracks:
            remaining = [d for idx, d in enumerate(dets) if idx not in used_det]
            remaining = sorted(remaining, key=lambda d: d.center[0])
            for d in remaining:
                if len(self.tracks) >= self.max_tracks:
                    break
                # Only assign IDs 1 to max_tracks to prevent "Teacher 3" issue
                new_id = self.next_id
                if new_id > self.max_tracks:
                    # Find available ID
                    used_ids = {t.tid for t in self.tracks}
                    for candidate in range(1, self.max_tracks + 1):
                        if candidate not in used_ids:
                            new_id = candidate
                            break
                    else:
                        break  # No available ID
                self.tracks.append(Track(tid=new_id, bbox=d.bbox, center=d.center, missed=0))
                self.next_id = new_id + 1

        return sorted(self.tracks, key=lambda t: t.tid)


# -----------------------------
# Shared frame bus (thread-safe)
# -----------------------------
class FrameBus:
    def __init__(self):
        self.lock = threading.Lock()
        self.teacher_frame = None
        self.teacher_idx = -1
        self.teacher_ts = 0.0

        self.user_frame = None
        self.user_idx = -1
        self.user_ts = 0.0

        self.teacher_fps = 30.0
        self.user_fps = 30.0

    def update_teacher(self, frame, idx, ts):
        with self.lock:
            self.teacher_frame = frame
            self.teacher_idx = int(idx)
            self.teacher_ts = float(ts)

    def update_user(self, frame, idx, ts):
        with self.lock:
            self.user_frame = frame
            self.user_idx = int(idx)
            self.user_ts = float(ts)

    def snapshot(self):
        with self.lock:
            tf = None if self.teacher_frame is None else self.teacher_frame.copy()
            uf = None if self.user_frame is None else self.user_frame.copy()
            return tf, int(self.teacher_idx), uf, int(self.user_idx)


# -----------------------------
# Smooth video reader thread (no UI blocking)
# + NEW: optional recording for camera
# -----------------------------
class VideoReaderThread(QThread):
    frameReady = Signal(object, int, float)   # frame_bgr, frame_idx, ts_wall
    metaReady = Signal(float)                 # fps

    def __init__(self, source, *, is_camera=False, mirror=False,
                 max_w=860, emit_fps=30.0, start_wall=None, start_frame=0,
                 record_path: Optional[str] = None, record_fps: Optional[float] = None):
        super().__init__()
        self.source = source
        self.is_camera = is_camera
        self.mirror = bool(mirror)
        self.max_w = int(max_w)
        self.emit_fps = float(max(5.0, min(60.0, emit_fps)))
        self.start_wall = start_wall
        self.start_frame = int(start_frame)

        self.record_path = record_path
        self.record_fps = float(record_fps) if record_fps else None

        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            return

        if self.is_camera:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1e-3:
            fps = 30.0

        self.metaReady.emit(float(fps))

        native_fps = float(fps)
        emit_dt = 1.0 / float(self.emit_fps)

        if self.start_wall is None:
            self.start_wall = time.perf_counter()
        start_wall = float(self.start_wall)
        start_frame = int(self.start_frame)

        next_emit = time.perf_counter()
        cam_idx = 0

        writer = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        rec_fps = float(self.record_fps if self.record_fps else self.emit_fps)

        while not self._stop:
            now = time.perf_counter()

            if now < next_emit:
                time.sleep(min(0.005, next_emit - now))
                continue

            if self.is_camera:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                # Keep original for recording (HD), resize for UI
                frame_small = resize_limit(frame, self.max_w)

                if self.record_path:
                    if writer is None:
                        try:
                            os.makedirs(os.path.dirname(self.record_path), exist_ok=True)
                        except Exception:
                            pass
                        # Use full resolution for recording
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(self.record_path, fourcc, rec_fps, (w, h))
                    if writer is not None:
                        writer.write(frame)

                cam_idx += 1
                self.frameReady.emit(frame_small, cam_idx, time.time())
                next_emit += emit_dt
                continue

            elapsed = max(0.0, now - start_wall)
            exp_idx = start_frame + int(elapsed * native_fps)

            cur_next = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)

            if abs(exp_idx - cur_next) > 90:
                cap.set(cv2.CAP_PROP_POS_FRAMES, exp_idx)
                cur_next = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or exp_idx)

            if exp_idx > cur_next:
                skip = exp_idx - cur_next
                if skip > 240:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, exp_idx)
                else:
                    for _ in range(skip):
                        if not cap.grab():
                            break

            ok = cap.grab()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                start_wall = time.perf_counter()
                start_frame = 0
                continue

            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue

            frame = resize_limit(frame, self.max_w)
            pos_next = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or (exp_idx + 1))
            idx = pos_next - 1

            self.frameReady.emit(frame, idx, time.time())
            next_emit += emit_dt

        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass

        try:
            cap.release()
        except Exception:
            pass


# -----------------------------
# Worker settings snapshot
# -----------------------------
@dataclass
class WorkerSettings:
    proc_interval_sec: float = 0.5
    n_students: int = 1
    win_frames: int = 8
    hands_enabled: bool = True
    teacher_map: List[int] = None  # e.g. [1,2]
    spacing_weight: float = 0.12   # applied only when 2 students + spacing available


# -----------------------------
# Scoring worker thread (heavy processing, no UI freeze)
# -----------------------------
class ScoringWorker(QThread):
    resultReady = Signal(object)

    def __init__(self, frame_bus: FrameBus):
        super().__init__()
        self.bus = frame_bus
        self._stop = False

        self._lock = threading.Lock()
        self._running = False
        self._settings = WorkerSettings(teacher_map=[1, 2])

        self._reset_flag = False

        self.pose_est = PoseEstimator()
        self.mp_pose = mp.solutions.pose
        PL = self.mp_pose.PoseLandmark
        self.idx_LSH = PL.LEFT_SHOULDER.value
        self.idx_RSH = PL.RIGHT_SHOULDER.value
        self.idx_LHI = PL.LEFT_HIP.value
        self.idx_RHI = PL.RIGHT_HIP.value

        self.detector = None
        if YOLO_OK:
            try:
                print(f"[DEBUG] Loading YOLO from: {YOLO_MODEL_PATH}")
                print(f"[DEBUG] YOLO file exists: {os.path.exists(YOLO_MODEL_PATH)}")
                self.detector = PersonDetectorYOLO(YOLO_MODEL_PATH, conf=0.35)
                print("[DEBUG] YOLO loaded successfully!")
            except Exception as e:
                print(f"[ERROR] YOLO load failed: {e}")
                self.detector = None
        else:
            print("[DEBUG] YOLO_OK is False - ultralytics not installed")

        self.tracker_teacher = SimpleTracker(max_tracks=2, max_missed=12)
        self.tracker_user = SimpleTracker(max_tracks=1, max_missed=12)

        self.teacher_buf: List[deque] = [deque(maxlen=240), deque(maxlen=240)]
        self.score_hist: List[deque] = [deque(maxlen=30), deque(maxlen=30)]

        self.last_t_idx = -1
        self.last_u_idx = -1

        self.last_u_tracks: Dict[int, Tuple[int, int, int, int]] = {}
        self.last_t_tracks: Dict[int, Tuple[int, int, int, int]] = {}

        self.last_u_skel: Dict[int, List[Optional[Tuple[int, int]]]] = {}
        self.last_t_skel: Dict[int, List[Optional[Tuple[int, int]]]] = {}

        self.feedback_interval_sec = 1.0
        self.last_feedback_wall = [0.0, 0.0]
        self.feedback_text = ["", ""]
        self.feedback_color = [(255, 255, 255), (255, 255, 255)]  # BGR

    def stop(self):
        self._stop = True

    def set_running(self, running: bool):
        with self._lock:
            self._running = bool(running)
            if self._running:
                now = time.time()
                self.last_feedback_wall = [now, now]

    def update_settings(self, settings: WorkerSettings):
        with self._lock:
            self._settings = settings
            self.tracker_user.set_max_tracks(int(max(1, settings.n_students)))
            while len(self.score_hist) < settings.n_students:
                self.score_hist.append(deque(maxlen=30))
            self.score_hist = self.score_hist[:settings.n_students]

    def request_reset(self):
        with self._lock:
            self._reset_flag = True

    @staticmethod
    def score_to_feedback(score: float) -> Tuple[str, Tuple[int, int, int]]:
        if score >= 92:
            return "Perfect ", (80, 255, 120)
        if score >= 85:
            return "Very Good ", (80, 255, 120)
        if score >= 70:
            return "Good ", (0, 220, 255)
        if score >= 55:
            return "Almost ", (0, 165, 255)
        return "Try Again ", (60, 60, 255)

    def detect_people(self, frame, max_people):
        h, w = frame.shape[:2]
        if self.detector is not None:
            return self.detector.detect(frame, max_people=max_people)
        return [PersonDet(bbox=(0, 0, w - 1, h - 1), center=(w * 0.5, h * 0.5))]

    def crop_pose_features(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        x1 = int(clamp(x1, 0, frame.shape[1] - 1))
        x2 = int(clamp(x2, 0, frame.shape[1] - 1))
        y1 = int(clamp(y1, 0, frame.shape[0] - 1))
        y2 = int(clamp(y2, 0, frame.shape[0] - 1))
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            return None, None

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None, None

        res_pose = self.pose_est.process(crop)
        ch, cw = crop.shape[:2]
        pts, lm = landmarks_to_points_pose(res_pose, cw, ch)

        feat_pose = None
        skel_full = None

        if pts is not None:
            npts = normalize_pose_points(pts, self.idx_LSH, self.idx_RSH, self.idx_LHI, self.idx_RHI)
            feat_pose = build_pose_features(npts, lm)

            skel_full = []
            for p in pts:
                if p is None:
                    skel_full.append(None)
                else:
                    skel_full.append((int(x1 + p[0]), int(y1 + p[1])))

        return feat_pose, skel_full

    def reset_for_loop(self):
        self.teacher_buf = [deque(maxlen=240), deque(maxlen=240)]
        self.tracker_teacher.reset()
        self.tracker_user.reset()
        self.last_t_tracks = {}
        self.last_u_tracks = {}
        self.last_t_skel = {}
        self.last_u_skel = {}
        self.last_t_idx = -1
        self.last_u_idx = -1

    @staticmethod
    def _center_and_scale_from_bbox(bbox):
        # bbox: (x1,y1,x2,y2)
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            return (0.0, 0.0), 1.0
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        h = max(1.0, float(y2 - y1))
        w = max(1.0, float(x2 - x1))
        scale = 0.5 * (h + w)  # robust-ish
        return (cx, cy), scale

    @staticmethod
    def _spacing_score(t_bbox1, t_bbox2, u_bbox1, u_bbox2):
        # normalized distance (center distance / avg scale)
        tc1, ts1 = ScoringWorker._center_and_scale_from_bbox(t_bbox1)
        tc2, ts2 = ScoringWorker._center_and_scale_from_bbox(t_bbox2)
        uc1, us1 = ScoringWorker._center_and_scale_from_bbox(u_bbox1)
        uc2, us2 = ScoringWorker._center_and_scale_from_bbox(u_bbox2)

        t_scale = max(1.0, 0.5 * (ts1 + ts2))
        u_scale = max(1.0, 0.5 * (us1 + us2))

        t_dist = math.hypot(float(tc1[0]) - float(tc2[0]), float(tc1[1]) - float(tc2[1])) / t_scale
        u_dist = math.hypot(float(uc1[0]) - float(uc2[0]), float(uc1[1]) - float(uc2[1])) / u_scale

        diff = abs(u_dist - t_dist)

        dead = 0.05
        tol = 0.25
        if diff <= dead:
            score = 100.0
        else:
            score = clamp(1.0 - (diff - dead) / max(1e-6, (tol - dead)), 0.0, 1.0) * 100.0

        return float(score), float(t_dist), float(u_dist), float(diff)

    def run(self):
        next_proc = time.time()

        while not self._stop:
            with self._lock:
                running = self._running
                settings = self._settings
                do_reset = self._reset_flag
                if do_reset:
                    self._reset_flag = False

            if do_reset:
                self.reset_for_loop()

            if not running:
                time.sleep(0.01)
                continue

            now = time.time()
            if now < next_proc:
                time.sleep(min(0.01, next_proc - now))
                continue

            next_proc = now + float(settings.proc_interval_sec)

            tf, t_idx, uf, u_idx = self.bus.snapshot()
            if tf is None or uf is None:
                continue
            
            print(f"[WORKER] Processing frame t_idx={t_idx}, u_idx={u_idx}")

            # detect loop (video restarted)
            if self.last_t_idx >= 0 and t_idx >= 0 and (t_idx + 10 < self.last_t_idx):
                self.reset_for_loop()
            if self.last_u_idx >= 0 and u_idx >= 0 and (u_idx + 10 < self.last_u_idx):
                self.tracker_user.reset()
                self.last_u_tracks = {}
                self.last_u_skel = {}

            self.last_t_idx = t_idx
            self.last_u_idx = u_idx

            n_students = int(max(1, settings.n_students))
            teacher_map = settings.teacher_map or [1, 2]
            if len(teacher_map) < n_students:
                teacher_map = (teacher_map + [1] * n_students)[:n_students]

            dets_u = self.detect_people(uf, max_people=n_students)
            tracks_u = self.tracker_user.update(dets_u)
            self.last_u_tracks = {tr.tid: tr.bbox for tr in tracks_u}

            dets_t = self.detect_people(tf, max_people=2)
            tracks_t = self.tracker_teacher.update(dets_t)
            self.last_t_tracks = {tr.tid: tr.bbox for tr in tracks_t}

            group_spacing_score = None
            group_t_norm = None
            group_u_norm = None
            group_diff = None

            if n_students >= 2:
                if (1 in self.last_t_tracks) and (2 in self.last_t_tracks) and (1 in self.last_u_tracks) and (2 in self.last_u_tracks):
                    group_spacing_score, group_t_norm, group_u_norm, group_diff = self._spacing_score(
                        self.last_t_tracks[1], self.last_t_tracks[2],
                        self.last_u_tracks[1], self.last_u_tracks[2]
                    )

            t_by_id = {tr.tid: tr for tr in tracks_t}
            self.last_t_skel = {}
            for teacher_id in (1, 2):
                if teacher_id in t_by_id:
                    ft_pose, t_skel = self.crop_pose_features(tf, t_by_id[teacher_id].bbox)
                    if t_skel is not None:
                        self.last_t_skel[teacher_id] = t_skel
                    if ft_pose is not None:
                        self.teacher_buf[teacher_id - 1].append((int(t_idx), ft_pose))

            u_by_id = {tr.tid: tr for tr in tracks_u}
            self.last_u_skel = {}

            out_scores = []
            now_wall = time.time()
            spacing_w = float(clamp(settings.spacing_weight, 0.0, 0.3))
            spacing_available = (group_spacing_score is not None)

            for student_id in range(1, n_students + 1):
                fu_pose = None
                if student_id in u_by_id:
                    fu_pose, u_skel = self.crop_pose_features(uf, u_by_id[student_id].bbox)
                    if u_skel is not None:
                        self.last_u_skel[student_id] = u_skel

                teacher_ref = int(teacher_map[student_id - 1]) if (student_id - 1) < len(teacher_map) else 1
                teacher_ref = 1 if teacher_ref != 2 else 2

                best_score = 0.0
                best_parts = {"arms": 0.0, "hands": 0.0, "torso": 0.0, "legs": 0.0}

                buf = self.teacher_buf[teacher_ref - 1]
                if fu_pose is not None and len(buf) > 0:
                    cand = [(fi, fp) for (fi, fp) in buf if abs(int(fi) - int(t_idx)) <= (int(settings.win_frames) * 2)]
                    if not cand:
                        cand = list(buf)

                    for _, ft_pose_c in cand:
                        score, parts = compare_pose_features(
                            fu_pose, ft_pose_c,
                            dead_zone_deg=8.0,
                            vec_dead_cos=0.985
                        )

                        if not bool(settings.hands_enabled):
                            parts["hands"] = 0.0
                            score = 0.65 * parts["arms"] + 0.20 * parts["torso"] + 0.15 * parts["legs"]

                        if score > best_score:
                            best_score = score
                            best_parts = parts

                while len(self.score_hist) < n_students:
                    self.score_hist.append(deque(maxlen=30))
                self.score_hist = self.score_hist[:n_students]

                self.score_hist[student_id - 1].append(best_score)
                smooth_pose = float(np.mean(self.score_hist[student_id - 1])) if self.score_hist[student_id - 1] else best_score

                if spacing_available and n_students >= 2:
                    smooth = (1.0 - spacing_w) * smooth_pose + spacing_w * float(group_spacing_score)
                else:
                    smooth = smooth_pose

                if now_wall - self.last_feedback_wall[student_id - 1] >= self.feedback_interval_sec:
                    txt, col = self.score_to_feedback(smooth)
                    self.feedback_text[student_id - 1] = txt
                    self.feedback_color[student_id - 1] = col
                    self.last_feedback_wall[student_id - 1] = now_wall

                out_scores.append({
                    "student_id": student_id,
                    "overall": smooth,
                    "overall_pose_only": smooth_pose,
                    "parts": best_parts,
                    "feedback_text": self.feedback_text[student_id - 1],
                    "feedback_color": self.feedback_color[student_id - 1],
                })

            self.resultReady.emit({
                "t_tracks": self.last_t_tracks,
                "u_tracks": self.last_u_tracks,
                "t_skel": self.last_t_skel,
                "u_skel": self.last_u_skel,
                "scores": out_scores,

                "group_spacing_score": group_spacing_score,
                "group_t_norm": group_t_norm,
                "group_u_norm": group_u_norm,
                "group_diff": group_diff
            })


# -----------------------------
# UI widgets
# -----------------------------
class PersonCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("PersonCard")

        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight:900; font-size:14px;")

        self.pb_overall = QProgressBar(); self.pb_overall.setRange(0, 100)
        self.pb_arms = QProgressBar(); self.pb_arms.setRange(0, 100)
        self.pb_hands = QProgressBar(); self.pb_hands.setRange(0, 100)
        self.pb_torso = QProgressBar(); self.pb_torso.setRange(0, 100)
        self.pb_legs = QProgressBar(); self.pb_legs.setRange(0, 100)

        for pb in [self.pb_overall, self.pb_arms, self.pb_hands, self.pb_torso, self.pb_legs]:
            pb.setValue(0)
            pb.setTextVisible(True)

        grid = QGridLayout()
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.addWidget(self.title, 0, 0, 1, 2)
        grid.addWidget(QLabel("รวม (Overall)"), 1, 0); grid.addWidget(self.pb_overall, 1, 1)
        grid.addWidget(QLabel("แขน (Arms)"),    2, 0); grid.addWidget(self.pb_arms,    2, 1)
        grid.addWidget(QLabel("มือ (Hands)"),   3, 0); grid.addWidget(self.pb_hands,   3, 1)
        grid.addWidget(QLabel("ลำตัว (Torso)"),   4, 0); grid.addWidget(self.pb_torso,   4, 1)
        grid.addWidget(QLabel("ขา (Legs)"),    5, 0); grid.addWidget(self.pb_legs,    5, 1)
        self.setLayout(grid)

    def set_scores(self, overall: float, parts: dict):
        self.pb_overall.setValue(int(round(overall)))
        self.pb_arms.setValue(int(round(parts.get("arms", 0.0))))
        self.pb_hands.setValue(int(round(parts.get("hands", 0.0))))
        self.pb_torso.setValue(int(round(parts.get("torso", 0.0))))
        self.pb_legs.setValue(int(round(parts.get("legs", 0.0))))


# -----------------------------
# Teacher Audio Player (sync audio from video)
# -----------------------------
class TeacherAudioPlayer(QObject):
    """Play audio from teacher video using Qt Multimedia, synced with video playback."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path: Optional[str] = None
        self.is_playing = False
        self._volume = 0.7
        
        self.player: Optional[QMediaPlayer] = None
        self.audio_output: Optional[QAudioOutput] = None
        
        if QT_MULTIMEDIA_OK:
            try:
                self.player = QMediaPlayer()
                self.audio_output = QAudioOutput()
                self.player.setAudioOutput(self.audio_output)
                self.audio_output.setVolume(self._volume)
            except Exception as e:
                print(f"Qt Multimedia init failed: {e}")
                self.player = None
                self.audio_output = None
    
    def set_source(self, video_path: str) -> bool:
        """Set the video file as audio source."""
        if not QT_MULTIMEDIA_OK or not self.player:
            return False
        
        self.stop()
        self.video_path = video_path
        
        try:
            url = QUrl.fromLocalFile(video_path)
            self.player.setSource(url)
            return True
        except Exception as e:
            print(f"Set audio source failed: {e}")
            return False
    
    def play(self, start_sec: float = 0.0):
        """Start playing audio from specified position."""
        if not QT_MULTIMEDIA_OK or not self.player:
            return
        
        try:
            self.player.setPosition(int(start_sec * 1000))  # position in ms
            self.player.play()
            self.is_playing = True
        except Exception as e:
            print(f"Audio play failed: {e}")
            self.is_playing = False
    
    def pause(self):
        """Pause audio playback."""
        if not QT_MULTIMEDIA_OK or not self.player:
            return
        try:
            self.player.pause()
            self.is_playing = False
        except Exception:
            pass
    
    def unpause(self):
        """Resume audio playback."""
        if not QT_MULTIMEDIA_OK or not self.player:
            return
        try:
            self.player.play()
            self.is_playing = True
        except Exception:
            pass
    
    def stop(self):
        """Stop audio playback."""
        self.is_playing = False
        if not QT_MULTIMEDIA_OK or not self.player:
            return
        try:
            self.player.stop()
        except Exception:
            pass
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)."""
        self._volume = clamp(volume, 0.0, 1.0)
        if not QT_MULTIMEDIA_OK or not self.audio_output:
            return
        try:
            self.audio_output.setVolume(self._volume)
        except Exception:
            pass
    
    def seek(self, sec: float):
        """Seek to position in seconds."""
        if not QT_MULTIMEDIA_OK or not self.player:
            return
        try:
            self.player.setPosition(int(sec * 1000))
        except Exception:
            pass
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if self.player:
            try:
                self.player.setSource(QUrl())
            except Exception:
                pass
        self.video_path = None


# -----------------------------
# Background Widget (with paintEvent)
# -----------------------------
class BackgroundWidget(QWidget):
    """Widget that draws a background image."""
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.background_pixmap = None
        if os.path.exists(self.image_path):
            self.background_pixmap = QPixmap(self.image_path)
    
    def paintEvent(self, event):
        """Draw background image."""
        if self.background_pixmap and not self.background_pixmap.isNull():
            from PySide6.QtGui import QPainter
            painter = QPainter(self)
            # Scale to fill entire widget
            scaled_pixmap = self.background_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )
            # Center the image
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)
        super().paintEvent(event)


# -----------------------------
# Start Screen Widget
# -----------------------------
class StartScreen(QWidget):
    """Start screen that shows start.JPG and waits for mouse click."""
    clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_image_path = resource_path("start.JPG")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        
        # Load and display start image
        if os.path.exists(self.start_image_path):
            pixmap = QPixmap(self.start_image_path)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("start.JPG not found")
            self.image_label.setStyleSheet("font-size: 50px; color: #888;")
        
        layout.addWidget(self.image_label)
        self.setLayout(layout)
    
    def mousePressEvent(self, event):
        """Emit clicked signal on any mouse button press."""
        self.clicked.emit()
        event.accept()
    
    def resizeEvent(self, event):
        """Scale image to fit window while maintaining aspect ratio."""
        super().resizeEvent(event)
        if os.path.exists(self.start_image_path):
            pixmap = QPixmap(self.start_image_path)
            scaled_pixmap = pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)


# -----------------------------
# Main Window
# -----------------------------
class MenuScreen(BackgroundWidget):
    def __init__(self, image_path: str, parent=None):
        super().__init__(image_path, parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignCenter)

        # Button Style
        btn_style = """
            QPushButton {
                background-color: white;
                color: black;
                font-size: 24px;
                font-weight: bold;
                border-radius: 30px;
                padding: 20px 40px;
                min-width: 400px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """

        # Button 1
        self.btn_manual = QPushButton("1. คู่มือการใช้งานโปรแกรม")
        self.btn_manual.setStyleSheet(btn_style)
        self.btn_manual.clicked.connect(self.open_manual)

        # Button 2
        self.btn_learning = QPushButton("2. หน่วยการเรียนรู้ การแสดงนาฏศิลป์เป็นคู่ เป็นหมู่และหลักวิจารณ์การแสดง")
        self.btn_learning.setStyleSheet(btn_style)
        self.btn_learning.clicked.connect(self.open_learning)

        # Button 3
        self.btn_practice = QPushButton("3. โปรแกรมฝึกปฏิบัติทักษะนาฏศิลป์ไทยประเภทระบำมาตรฐาน")
        self.btn_practice.setStyleSheet(btn_style)
        # Connected in MainWindow

        layout.addStretch()
        layout.addWidget(self.btn_manual, 0, Qt.AlignCenter)
        layout.addWidget(self.btn_learning, 0, Qt.AlignCenter)
        layout.addWidget(self.btn_practice, 0, Qt.AlignCenter)
        layout.addStretch()

        self.setLayout(layout)

    def open_manual(self):
        url = QUrl("https://thep-bantherng.my.canva.site/?fbclid=IwY2xjawPh0xFleHRuA2FlbQIxMABicmlkETFDanRCcjdFem05OFY4TmF5c3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHu1eq7ohjOsqFTeBDvGDE0dv7xd3z0csXrYoouAwZ2u_BzNf3wkdYMND16lL_aem_plfCPz3-wd_bhvzvoZ9PIw")
        QDesktopServices.openUrl(url)

    def open_learning(self):
        url = QUrl("https://thep-bantherng.my.canva.site/dag-yp6nhvu")
        QDesktopServices.openUrl(url)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thai Dance Scoring (Smooth)")
        self.setMinimumSize(QSize(1100, 700))
        self.resize(1500, 900)
        
        # Background image path
        self.background_image_path = resource_path("background.JPG")
        self.background2_image_path = resource_path("background2.JPG")

        self.mode = "realtime"  # or "video"

        self.teacher_path: Optional[str] = None
        self.user_video_path: Optional[str] = None

        self.teacher_fps_file = 30.0
        self.teacher_frames_total = 0
        self.teacher_dur_sec = 0.0

        self.user_fps_file = 30.0
        self.user_frames_total = 0
        self.user_dur_sec = 0.0

        self.seek_sec = 0.0
        self.seek_max_sec = 0.0

        self.teacher_fps = 30.0
        self.teacher_dt = 1.0 / 30.0
        self.win_frames = 8

        self.display_fps = 30
        self.bus = FrameBus()

        self.th_teacher: Optional[VideoReaderThread] = None
        self.th_user: Optional[VideoReaderThread] = None

        self.worker = ScoringWorker(self.bus)
        self.worker.resultReady.connect(self.on_worker_result)
        self.worker.start()

        self.last_t_tracks = {}
        self.last_u_tracks = {}
        self.last_t_skel = {}
        self.last_u_skel = {}
        self.last_feedback_text = ["", ""]
        self.last_feedback_color = [(255, 255, 255), (255, 255, 255)]

        self.last_group_spacing_score = None
        self.last_group_t_norm = None
        self.last_group_u_norm = None
        self.last_group_diff = None

        self.acc = []
        self.acc_group = {"count": 0, "sum_spacing": 0.0}
        self.session_started_at = ""
        self.running = False
        
        # Countdown settings
        self.countdown_sec = 5  # Default 5 seconds countdown
        self.countdown_remaining = 0
        self.countdown_timer: Optional[QTimer] = None

        self.realtime_record_path: Optional[str] = None
        self._pending_payloads: Optional[List[dict]] = None
        self._drive_upload_thread: Optional[DriveUploadThread] = None

        # Teacher audio player
        self.teacher_audio = TeacherAudioPlayer()
        self.teacher_audio_extracted = False

        self._build_ui()
        self._apply_styles()
        self._apply_background()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(int(1000 / self.display_fps))

        self.refresh_teacher_dropdown()

    # ---------- Feedback badge drawing ----------
    @staticmethod
    def draw_feedback_badge(frame, x, y, text, color_bgr):
        if not text:
            return
        h, w = frame.shape[:2]
        x = int(clamp(x, 8, w - 8))
        y = int(clamp(y, 30, h - 8))

        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.0
        thick = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

        pad_x, pad_y = 10, 8
        x1 = int(clamp(x, 0, w - 1))
        y1 = int(clamp(y - th - pad_y * 2, 0, h - 1))
        x2 = int(clamp(x + tw + pad_x * 2, 0, w - 1))
        y2 = int(clamp(y, 0, h - 1))

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(frame, text, (x1 + pad_x, y2 - pad_y), font, scale, color_bgr, thick, cv2.LINE_AA)

    @staticmethod
    def draw_skeleton(frame, skel_pts: List[Optional[Tuple[int, int]]], color=(180, 180, 255)):
        if not skel_pts or len(skel_pts) < 33:
            return
        for a, b in mp.solutions.pose.POSE_CONNECTIONS:
            pa = skel_pts[a]
            pb = skel_pts[b]
            if pa is None or pb is None:
                continue
            cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
        for p in skel_pts:
            if p is None:
                continue
            cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)

    def get_student_name(self, student_id_1based: int) -> str:
        i = student_id_1based - 1
        if 0 <= i < len(self.name_edits):
            name = (self.name_edits[i].text() or "").strip()
            if name:
                return name
        return f"Student {student_id_1based}"

    def get_teacher_ref_for_student(self, student_id_1based: int) -> int:
        i = student_id_1based - 1
        if 0 <= i < len(self.combo_teacher_for_student):
            return int(self.combo_teacher_for_student[i].currentIndex()) + 1
        return 1

    def _update_student_controls_enabled(self):
        n = int(self.spin_students.value())
        self.name_edits[0].setEnabled(True)
        self.combo_teacher_for_student[0].setEnabled(True)
        self.name_edits[1].setEnabled(n >= 2)
        self.combo_teacher_for_student[1].setEnabled(n >= 2)

    def _init_session_accumulators(self):
        n = int(self.spin_students.value())
        self.acc = []
        for _ in range(n):
            self.acc.append({
                "count": 0,
                "sum_overall": 0.0,
                "sum_arms": 0.0,
                "sum_hands": 0.0,
                "sum_torso": 0.0,
                "sum_legs": 0.0
            })
        self.acc_group = {"count": 0, "sum_spacing": 0.0}

    def _recalc_win_frames(self):
        self.teacher_dt = 1.0 / float(self.teacher_fps if self.teacher_fps > 1e-6 else 30.0)
        self.win_frames = max(1, int(round(float(self.spin_window.value()) / self.teacher_dt)))

    # ---------- Teacher dropdown ----------
    def refresh_teacher_dropdown(self):
        ensure_teacher_dir()
        cur_path = self.teacher_path
        self.teacher_combo.blockSignals(True)
        self.teacher_combo.clear()

        items = []
        for p in list_teacher_videos_in_dir():
            items.append(p)

        for p in items:
            self.teacher_combo.addItem(os.path.basename(p), userData=p)

        if cur_path:
            found = False
            for i in range(self.teacher_combo.count()):
                if self.teacher_combo.itemData(i) == cur_path:
                    self.teacher_combo.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self.teacher_combo.insertItem(0, os.path.basename(cur_path), userData=cur_path)
                self.teacher_combo.setCurrentIndex(0)

        self.teacher_combo.blockSignals(False)
        self.on_teacher_combo_changed(self.teacher_combo.currentIndex())

    def add_teacher_video_external(self):
        path, _ = QFileDialog.getOpenFileName(self, "เพิ่มวิดีโอครู (Add to dropdown)", "", "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)")
        if not path:
            return
        exists = False
        for i in range(self.teacher_combo.count()):
            if self.teacher_combo.itemData(i) == path:
                self.teacher_combo.setCurrentIndex(i)
                exists = True
                break
        if not exists:
            self.teacher_combo.insertItem(0, os.path.basename(path), userData=path)
            self.teacher_combo.setCurrentIndex(0)
        self.on_teacher_combo_changed(self.teacher_combo.currentIndex())

    @Slot(int)
    def on_teacher_combo_changed(self, idx: int):
        if idx < 0:
            return
        path = self.teacher_combo.itemData(idx)
        if not path:
            return
        self.set_teacher_path(path)

    def set_teacher_path(self, path: str):
        self.teacher_path = path
        self.lbl_teacher_path.setText(path)
        self.lbl_teacher_path_v.setText(path)

        fps, frames, dur = probe_video(path)
        self.teacher_fps_file = fps
        self.teacher_frames_total = frames
        self.teacher_dur_sec = dur

        self.update_seek_range()
        self._recalc_win_frames()
        
        # Set audio source from teacher video
        self.teacher_audio_extracted = False
        if QT_MULTIMEDIA_OK:
            try:
                self.teacher_audio_extracted = self.teacher_audio.set_source(path)
                if self.teacher_audio_extracted:
                    self.teacher_audio.set_volume(self.slider_volume.value() / 100.0)
            except Exception as e:
                print(f"Audio setup failed: {e}")

    # ---------- Seek ----------
    def update_seek_range(self):
        max_sec = 0.0
        if self.teacher_path and self.teacher_dur_sec > 0:
            max_sec = self.teacher_dur_sec
        if self.mode == "video" and self.user_video_path and self.user_dur_sec > 0 and max_sec > 0:
            max_sec = min(max_sec, self.user_dur_sec)

        self.seek_max_sec = float(max(0.0, max_sec))
        if self.seek_max_sec <= 0.0:
            self.seek_slider.setEnabled(False)
            self.seek_slider.setRange(0, 0)
            self.seek_label.setText("Seek: --:-- / --:--")
            self.seek_sec = 0.0
            return

        self.seek_slider.setEnabled(True)
        max_ms = int(round(self.seek_max_sec * 1000.0))
        max_ms = max(0, max_ms)
        self.seek_slider.setRange(0, max_ms)

        self.seek_sec = float(clamp(self.seek_sec, 0.0, max(0.0, self.seek_max_sec - 0.05)))
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(int(round(self.seek_sec * 1000.0)))
        self.seek_slider.blockSignals(False)
        self.seek_label.setText(f"Seek: {sec_to_mmss(self.seek_sec)} / {sec_to_mmss(self.seek_max_sec)}")

    @Slot(int)
    def on_seek_changed(self, value_ms: int):
        if self.seek_max_sec <= 0:
            return
        self.seek_sec = float(value_ms) / 1000.0
        self.seek_sec = float(clamp(self.seek_sec, 0.0, max(0.0, self.seek_max_sec - 0.05)))
        self.seek_label.setText(f"Seek: {sec_to_mmss(self.seek_sec)} / {sec_to_mmss(self.seek_max_sec)}")

    @Slot()
    def on_seek_released(self):
        if self.seek_max_sec <= 0:
            return
        if self.running:
            self.seek_to_sec(self.seek_sec)

    def seek_to_sec(self, sec: float):
        sec = float(clamp(sec, 0.0, max(0.0, self.seek_max_sec - 0.05)))
        self.seek_sec = sec
        self.update_seek_range()

        if not self.teacher_path:
            return

        self.worker.request_reset()
        self._stop_threads()

        start_wall = time.perf_counter()
        fps_t = float(self.teacher_fps_file if self.teacher_fps_file > 1e-6 else 30.0)
        start_frame_t = int(round(sec * fps_t))

        self.th_teacher = VideoReaderThread(
            self.teacher_path,
            is_camera=False,
            mirror=False,
            max_w=860,
            emit_fps=30.0,
            start_wall=start_wall,
            start_frame=start_frame_t
        )
        self.th_teacher.frameReady.connect(lambda f, idx, ts: self.bus.update_teacher(f, idx, ts))
        self.th_teacher.metaReady.connect(self._on_teacher_meta)
        self.th_teacher.start()

        if self.mode == "realtime":
            cam = int(self.spin_cam.value())
            mirror = int(self.spin_mirror.value()) == 1
            self.th_user = VideoReaderThread(
                cam,
                is_camera=True,
                mirror=mirror,
                max_w=860,
                emit_fps=30.0,
                record_path=self.realtime_record_path,
                record_fps=30.0
            )
            self.th_user.metaReady.connect(self._on_user_meta)
        else:
            if not self.user_video_path:
                return
            fps_u = float(self.user_fps_file if self.user_fps_file > 1e-6 else 30.0)
            start_frame_u = int(round(sec * fps_u))
            self.th_user = VideoReaderThread(
                self.user_video_path,
                is_camera=False,
                mirror=False,
                max_w=860,
                emit_fps=30.0,
                start_wall=start_wall,
                start_frame=start_frame_u
            )
            self.th_user.metaReady.connect(self._on_user_meta)

        self.th_user.frameReady.connect(lambda f, idx, ts: self.bus.update_user(f, idx, ts))
        self.th_user.start()

        self._recalc_win_frames()
        self._push_worker_settings()
        
        # Sync audio with seek position if running and audio enabled
        if self.running and self.cb_teacher_audio.isChecked() and self.teacher_audio_extracted:
            self.teacher_audio.play(sec)

    # ---------- UI ----------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        self.stack = QStackedWidget()
        
        # Page 0: Start Screen
        self.page_start = StartScreen()
        self.page_start.clicked.connect(self.on_start_screen_clicked)

        # Page 1: Main Menu (New)
        self.page_main_menu = MenuScreen(self.background_image_path)
        self.page_main_menu.btn_practice.clicked.connect(lambda: self.stack.setCurrentIndex(2))

        # Page 2: Mode Selection Menu (with background)
        page_menu = BackgroundWidget(self.background_image_path)
        ml = QVBoxLayout()
        ml.setContentsMargins(50, 50, 50, 50)
        ml.setSpacing(20)
        ml.setAlignment(Qt.AlignCenter)

        # Button Style (Same as MenuScreen)
        btn_style = """
            QPushButton {
                background-color: white;
                color: black;
                font-size: 24px;
                font-weight: bold;
                border-radius: 30px;
                padding: 20px 40px;
                min-width: 400px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """

        title = QLabel("โปรแกรมฝึกปฏิบัติทักษะนาฏศิลป์ไทยประเภทระบำมาตรฐาน")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:32px; font-weight:900; color: white;")

        subtitle = QLabel("เลือกโหมดการใช้งาน")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size:24px; font-weight:bold; color: white; margin-bottom: 20px;")

        btn_rt = QPushButton("1. โหมดเปิดกล้อง (Realtime)")
        btn_rt.setStyleSheet(btn_style)
        
        btn_vv = QPushButton("2. โหมดวิดีโอเทียบวิดีโอ (Teacher vs Your Video)")
        btn_vv.setStyleSheet(btn_style)

        btn_rt.clicked.connect(lambda: self.open_mode("realtime"))
        btn_vv.clicked.connect(lambda: self.open_mode("video"))
        
        btn_back_to_main = QPushButton("← ย้อนกลับ")
        btn_back_to_main.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        btn_back_to_main.setStyleSheet("background: transparent; border: none; font-size: 16px; color: white;")

        ml.addStretch()
        ml.addWidget(title)
        ml.addWidget(subtitle)
        ml.addSpacing(10)
        ml.addWidget(btn_rt, 0, Qt.AlignCenter)
        ml.addWidget(btn_vv, 0, Qt.AlignCenter)
        ml.addSpacing(20)
        ml.addWidget(btn_back_to_main, 0, Qt.AlignCenter)
        ml.addStretch()
        page_menu.setLayout(ml)

        # Page 3: Main App (with background)
        page_app = BackgroundWidget(self.background2_image_path)
        app_layout = QHBoxLayout()
        app_layout.setContentsMargins(10, 10, 10, 10)
        app_layout.setSpacing(10)

        self.lbl_teacher = QLabel("ครูต้นแบบ (Teacher)")
        self.lbl_teacher.setAlignment(Qt.AlignCenter)
        self.lbl_teacher.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_teacher.setMinimumSize(320, 240)

        self.lbl_user = QLabel("ผู้เรียน (User)")
        self.lbl_user.setAlignment(Qt.AlignCenter)
        self.lbl_user.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_user.setMinimumSize(320, 240)

        vids = QHBoxLayout()
        vids.setContentsMargins(0, 0, 0, 0)
        vids.setSpacing(10)
        vids.addWidget(self.lbl_teacher, 1)
        vids.addWidget(self.lbl_user, 1)

        # Seek Bar (Moved here)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setEnabled(False)
        self.seek_slider.valueChanged.connect(self.on_seek_changed)
        self.seek_slider.sliderReleased.connect(self.on_seek_released)
        self.seek_label = QLabel("เวลา: --:-- / --:--")
        self.seek_label.setStyleSheet("color:#93c5fd; font-weight:900;")

        vids_layout = QVBoxLayout()
        vids_layout.addLayout(vids)
        vids_layout.addWidget(self.seek_label)
        vids_layout.addWidget(self.seek_slider)

        vids_wrap = QWidget()
        vids_wrap.setLayout(vids_layout)

        self.btn_back = QPushButton("← กลับเมนู")

        # ----- Teacher Dropdown (shared) -----
        self.gb_teacher_select = QGroupBox("เลือกวิดีโอครูต้นแบบ")
        gt = QGridLayout()
        gt.setContentsMargins(20, 24, 20, 20)
        gt.setHorizontalSpacing(10)
        gt.setVerticalSpacing(8)

        self.teacher_combo = QComboBox()
        self.btn_refresh_teachers = QPushButton("รีเฟรช (Refresh)")
        self.btn_add_teacher = QPushButton("เพิ่มไฟล์วิดีโอ...")

        self.lbl_teacher_path = QLabel("ยังไม่เลือกวิดีโอครู")
        self.lbl_teacher_path.setStyleSheet("color:#cbd5e1;")

        self.lbl_teacher_path_v = QLabel("ยังไม่เลือกวิดีโอครู")
        self.lbl_teacher_path_v.setStyleSheet("color:#cbd5e1;")

        self.teacher_combo.currentIndexChanged.connect(self.on_teacher_combo_changed)
        self.btn_refresh_teachers.clicked.connect(self.refresh_teacher_dropdown)
        self.btn_add_teacher.clicked.connect(self.add_teacher_video_external)

        gt.addWidget(QLabel("รายการวิดีโอ:"), 0, 0)
        gt.addWidget(self.teacher_combo, 0, 1, 1, 2)
        gt.addWidget(self.btn_refresh_teachers, 1, 1)
        gt.addWidget(self.btn_add_teacher, 1, 2)
        gt.addWidget(QLabel("วิดีโอที่เลือก:"), 2, 0)
        gt.addWidget(self.lbl_teacher_path, 2, 1, 1, 2)
        self.gb_teacher_select.setLayout(gt)
        
        # Set light blue background for teacher selection box
        self.gb_teacher_select.setStyleSheet("""
            QGroupBox {
                background: rgba(173, 216, 230, 0.8);
                border: 2px solid rgba(173, 216, 230, 0.6);
                border-radius: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #ffffff;
                font-size: 50px;
                font-weight: bold;
            }
        """)

        # ----- Realtime Controls -----
        self.gb_realtime = QGroupBox("ตั้งค่ากล้อง (Realtime Controls)")
        gr = QGridLayout()
        gr.setContentsMargins(10, 12, 10, 10)
        gr.setHorizontalSpacing(10)
        gr.setVerticalSpacing(8)

        self.spin_cam = QSpinBox(); self.spin_cam.setRange(0, 10); self.spin_cam.setValue(0)
        self.spin_mirror = QSpinBox(); self.spin_mirror.setRange(0, 1); self.spin_mirror.setValue(1)

        gr.addWidget(QLabel("เลือกกล้อง (Index)"), 0, 0); gr.addWidget(self.spin_cam, 0, 1)
        gr.addWidget(QLabel("กลับด้านภาพ (0=ปกติ, 1=กลับด้าน)"), 1, 0); gr.addWidget(self.spin_mirror, 1, 1)
        self.gb_realtime.setLayout(gr)
        
        # Set light blue background for realtime controls box
        self.gb_realtime.setStyleSheet("""
            QGroupBox {
                background: rgba(173, 216, 230, 0.8);
                border: 2px solid rgba(173, 216, 230, 0.6);
                border-radius: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #ffffff;
                font-size: 50px;
                font-weight: bold;
            }
        """)

        # ----- Video vs Video Controls -----
        self.gb_video = QGroupBox("ตั้งค่าโหมดวิดีโอเทียบวิดีโอ")
        gv = QGridLayout()
        gv.setContentsMargins(10, 12, 10, 10)
        gv.setHorizontalSpacing(10)
        gv.setVerticalSpacing(8)

        self.btn_pick_user_video = QPushButton("เลือกวิดีโอผู้เรียน")
        self.lbl_user_path = QLabel("ยังไม่เลือกวิดีโอผู้เรียน")
        self.lbl_user_path.setStyleSheet("color:#cbd5e1;")

        gv.addWidget(self.btn_pick_user_video, 0, 0, 1, 2)
        gv.addWidget(QLabel("ไฟล์ผู้เรียน:"), 1, 0); gv.addWidget(self.lbl_user_path, 1, 1)
        self.gb_video.setLayout(gv)
        
        # Set light blue background for video controls box
        self.gb_video.setStyleSheet("""
            QGroupBox {
                background: rgba(173, 216, 230, 0.8);
                border: 2px solid rgba(173, 216, 230, 0.6);
                border-radius: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #ffffff;
                font-size: 50px;
                font-weight: bold;
            }
        """)

        # ----- Shared Controls -----
        self.gb_shared = QGroupBox("การตั้งค่าการให้คะแนน (Shared Controls)")
        gs = QGridLayout()
        gs.setContentsMargins(10, 12, 10, 10)
        gs.setHorizontalSpacing(10)
        gs.setVerticalSpacing(8)

        self.spin_window = QDoubleSpinBox()
        self.spin_window.setRange(0.05, 1.50)
        self.spin_window.setSingleStep(0.05)
        self.spin_window.setValue(0.25)
        self.spin_window.valueChanged.connect(lambda _: self._recalc_win_frames())

        self.spin_students = QSpinBox()
        self.spin_students.setRange(1, 2)
        self.spin_students.setValue(2)
        self.spin_students.valueChanged.connect(lambda _: self._on_students_changed())

        self.cb_hands = QCheckBox("เปิดระบบตรวจสอบนิ้วมือ")
        self.cb_hands.setChecked(True)

        self.cb_show_teacher_video = QCheckBox("แสดงวิดีโอครู (เพื่อดูภาพประกอบ)")
        self.cb_show_teacher_video.setChecked(True)

        self.cb_show_pose_lines = QCheckBox("แสดงเส้นท่าทาง (Pose Lines)")
        self.cb_show_pose_lines.setChecked(True)

        self.cb_draw_teacher = QCheckBox("แสดงเส้นท่าทางฝั่งครู (Teacher Lines)")
        self.cb_draw_teacher.setChecked(True)

        # NEW: Teacher audio toggle and volume
        self.cb_teacher_audio = QCheckBox("🔊 เปิดเสียงวิดีโอครู")
        self.cb_teacher_audio.setChecked(True)  # Auto-play audio by default
        self.cb_teacher_audio.stateChanged.connect(self._on_teacher_audio_toggled)
        
        # Countdown spinner
        self.spin_countdown = QSpinBox()
        self.spin_countdown.setRange(0, 30)
        self.spin_countdown.setValue(5)
        self.spin_countdown.setToolTip("จำนวนวินาทีนับถอยหลังก่อนเริ่มเกม (0 = ไม่นับถอยหลัง)")
        
        self.slider_volume = QSlider(Qt.Horizontal)
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(70)
        self.slider_volume.setMaximumWidth(120)
        self.slider_volume.valueChanged.connect(self._on_volume_changed)
        self.lbl_volume = QLabel("ความดัง: 70%")
        self.lbl_volume.setStyleSheet("color:#93c5fd;")

        # NEW: Drive upload toggles
        self.cb_upload_drive = QCheckBox("อัปโหลดวิดีโอนักเรียนไป Google Drive ตอนกดหยุด")
        self.cb_upload_drive.setChecked(True)
        self.cb_drive_share = QCheckBox("แชร์ลิงก์วิดีโอ (Anyone with link) - เปิดสาธารณะ")
        self.cb_drive_share.setChecked(False)
        self.lbl_drive_status = QLabel("สถานะ Drive: -")
        self.lbl_drive_status.setStyleSheet("color:#93c5fd; font-weight:900;")

        self.name_edits = [QLineEdit(), QLineEdit()]
        self.name_edits[0].setPlaceholderText("ชื่อผู้เรียนคนที่ 1")
        self.name_edits[1].setPlaceholderText("ชื่อผู้เรียนคนที่ 2")
        self.name_edits[0].setText("ผู้เรียน 1")
        self.name_edits[1].setText("ผู้เรียน 2")

        self.combo_teacher_for_student = [QComboBox(), QComboBox()]
        for cb in self.combo_teacher_for_student:
            cb.addItems(["ครู 1", "ครู 2"])
        self.combo_teacher_for_student[0].setCurrentIndex(0)
        self.combo_teacher_for_student[1].setCurrentIndex(1)

        self.btn_start = QPushButton("เริ่ม")
        self.btn_stop_save = QPushButton("หยุด (บันทึกผล + อัปโหลด)")
        self.btn_reset = QPushButton("รีเซ็ตค่าทั้งหมด")



        self.gb_spacing = QGroupBox("คะแนนระยะห่าง (สำหรับ 2 คน)")
        gsp = QGridLayout()
        gsp.setContentsMargins(10, 12, 10, 10)
        gsp.setHorizontalSpacing(10)
        gsp.setVerticalSpacing(8)
        self.pb_spacing = QProgressBar()
        self.pb_spacing.setRange(0, 100)
        self.pb_spacing.setValue(0)
        self.lbl_spacing_detail = QLabel("ต้องตรวจพบครู 2 คน และนักเรียน 2 คน")
        self.lbl_spacing_detail.setStyleSheet("color:black;")
        gsp.addWidget(QLabel("คะแนนระยะห่าง"), 0, 0)
        gsp.addWidget(self.pb_spacing, 0, 1)
        gsp.addWidget(self.lbl_spacing_detail, 1, 0, 1, 2)
        self.gb_spacing.setLayout(gsp)
        
        # Set light blue background for spacing box
        self.gb_spacing.setStyleSheet("""
            QGroupBox {
                background: rgba(173, 216, 230, 0.8);
                border: 2px solid rgba(173, 216, 230, 0.6);
                border-radius: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #ffffff;
                font-size: 50px;
                font-weight: bold;
            }
        """)

        # Add tooltips for settings explanation
        self.spin_window.setToolTip("ช่วงเวลาที่ใช้เปรียบเทียบท่าทาง (วินาที) - ค่าน้อย=เข้มงวด, ค่ามาก=ยืดหยุ่น")
        self.spin_students.setToolTip("จำนวนนักเรียนที่จะตรวจจับ (1 หรือ 2 คน)")
        self.cb_hands.setToolTip("เปิด/ปิด การให้คะแนนมือ - บางครั้งการตรวจจับมืออาจไม่แม่นยำ")
        self.cb_show_teacher_video.setToolTip("แสดงหรือซ่อนวิดีโอครูทางซ้าย")
        self.cb_show_pose_lines.setToolTip("แสดงเส้นท่าทาง (skeleton) บนร่างกายนักเรียน")
        self.cb_draw_teacher.setToolTip("แสดงเส้นท่าทางบนร่างกายครูด้วย")
        
        gs.addWidget(QLabel("ช่วงเวลาเทียบท่า (วินาที)"), 0, 0); gs.addWidget(self.spin_window, 0, 1)
        gs.addWidget(QLabel("จำนวนผู้เรียน (1-2 คน)"), 0, 2); gs.addWidget(self.spin_students, 0, 3)

        gs.addWidget(self.cb_hands, 1, 0, 1, 2)
        gs.addWidget(self.cb_show_teacher_video, 1, 2, 1, 2)

        gs.addWidget(self.cb_show_pose_lines, 2, 0, 1, 2)
        gs.addWidget(self.cb_draw_teacher, 2, 2, 1, 2)

        gs.addWidget(QLabel("ชื่อคนที่ 1"), 3, 0); gs.addWidget(self.name_edits[0], 3, 1)
        gs.addWidget(QLabel("เทียบกับครู ->"), 3, 2); gs.addWidget(self.combo_teacher_for_student[0], 3, 3)

        gs.addWidget(QLabel("ชื่อคนที่ 2"), 4, 0); gs.addWidget(self.name_edits[1], 4, 1)
        gs.addWidget(QLabel("เทียบกับครู ->"), 4, 2); gs.addWidget(self.combo_teacher_for_student[1], 4, 3)



        gs.addWidget(self.cb_upload_drive, 6, 0, 1, 2)
        gs.addWidget(self.cb_drive_share, 6, 2, 1, 2)
        
        # Audio controls row
        gs.addWidget(self.cb_teacher_audio, 7, 0, 1, 2)
        audio_vol_widget = QWidget()
        audio_vol_layout = QHBoxLayout()
        audio_vol_layout.setContentsMargins(0, 0, 0, 0)
        audio_vol_layout.setSpacing(5)
        audio_vol_layout.addWidget(self.slider_volume)
        audio_vol_layout.addWidget(self.lbl_volume)
        audio_vol_widget.setLayout(audio_vol_layout)
        gs.addWidget(audio_vol_widget, 7, 2, 1, 2)
        
        # Countdown control row
        gs.addWidget(QLabel("⏱️ Countdown (วินาที)"), 8, 0); gs.addWidget(self.spin_countdown, 8, 1)
        gs.addWidget(self.lbl_drive_status, 8, 2, 1, 2)
        
        # Countdown display label
        self.lbl_countdown = QLabel("")
        self.lbl_countdown.setStyleSheet("color:#fbbf24; font-size:18px; font-weight:900;")
        self.lbl_countdown.setAlignment(Qt.AlignCenter)

        gs.addWidget(self.btn_stop_save, 9, 0, 1, 1)
        gs.addWidget(self.btn_start, 9, 1, 1, 1)
        gs.addWidget(self.btn_reset, 9, 2, 1, 2)
        
        # Set light blue background for shared controls box
        self.gb_shared.setStyleSheet("""
            QGroupBox {
                background: rgba(173, 216, 230, 0.8);
                border: 2px solid rgba(173, 216, 230, 0.6);
                border-radius: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #ffffff;
                font-size: 50px;
                font-weight: bold;
            }
        """)

        self.gb_shared.setLayout(gs)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.score_container = QWidget()
        self.score_layout = QVBoxLayout()
        self.score_layout.setContentsMargins(8, 8, 8, 8)
        self.score_layout.setSpacing(10)
        self.score_container.setLayout(self.score_layout)
        self.scroll.setWidget(self.score_container)

        # === TOP CONTROL BAR (easy access without scrolling) ===
        self.top_control_bar = QWidget()
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(10, 8, 10, 8)
        top_bar_layout.setSpacing(10)
        
        top_bar_layout.addWidget(self.btn_back)
        top_bar_layout.addWidget(self.btn_start)
        top_bar_layout.addWidget(self.btn_reset)
        top_bar_layout.addWidget(self.btn_stop_save)
        top_bar_layout.addWidget(self.lbl_countdown)
        top_bar_layout.addStretch(1)
        
        # Toggle settings button
        self.btn_toggle_settings = QPushButton("⚙️ ซ่อน/แสดงตั้งค่า")
        self.btn_toggle_settings.setCheckable(True)
        self.btn_toggle_settings.setChecked(False)
        self.btn_toggle_settings.clicked.connect(self._toggle_settings)
        top_bar_layout.addWidget(self.btn_toggle_settings)
        
        self.top_control_bar.setLayout(top_bar_layout)

        # === Settings panel (collapsible) ===
        self.settings_panel = QWidget()
        settings_layout = QVBoxLayout()
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(8)
        settings_layout.addWidget(self.gb_teacher_select)
        settings_layout.addWidget(self.gb_realtime)
        settings_layout.addWidget(self.gb_video)
        settings_layout.addWidget(self.gb_shared)
        settings_layout.addWidget(self.gb_spacing)
        self.settings_panel.setLayout(settings_layout)

        left_panel = QWidget()
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.setSpacing(8)

        left_panel_layout.addWidget(vids_wrap, 3)
        left_panel_layout.addWidget(self.settings_panel, 0)
        left_panel_layout.addStretch(1)
        left_panel.setLayout(left_panel_layout)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setWidget(left_panel)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)

        # Main app layout with top bar
        main_app_layout = QVBoxLayout()
        main_app_layout.setContentsMargins(0, 0, 0, 0)
        main_app_layout.setSpacing(0)
        main_app_layout.addWidget(self.top_control_bar, 0)
        main_app_layout.addWidget(splitter, 1)
        
        app_layout.addLayout(main_app_layout)
        page_app.setLayout(app_layout)

        self.stack.addWidget(self.page_start)
        self.stack.addWidget(self.page_main_menu)
        self.stack.addWidget(page_menu)
        self.stack.addWidget(page_app)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.stack)
        root.setLayout(root_layout)

        self.btn_back.clicked.connect(self.go_menu)
        self.btn_pick_user_video.clicked.connect(self.pick_user_video)

        self.btn_start.clicked.connect(self.start)
        self.btn_stop_save.clicked.connect(self.stop_and_save)
        self.btn_reset.clicked.connect(self.reset_all)

        self._update_student_controls_enabled()
        self._ensure_student_cards(int(self.spin_students.value()))

        self.gb_realtime.setVisible(True)
        self.gb_video.setVisible(False)

        self._recalc_win_frames()
        self.update_seek_range()
    
    def _apply_background(self):
        """Apply background image to all pages - now using BackgroundWidget with paintEvent."""
        # Background is now handled via BackgroundWidget class with paintEvent
        # This method is kept for compatibility but no longer needed
        pass
    
    def on_start_screen_clicked(self):
        """Handle start screen click - go to mode selection menu."""
        self.stack.setCurrentIndex(1)  # Go to mode selection menu (page_menu)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background: transparent; }
            QLabel { color: #000000; }
            QCheckBox { 
                color: #000000; 
                font-weight: 800; 
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                background: rgba(255, 255, 255, 0.9);
                border: 2px solid rgba(0, 0, 0, 0.5);
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background: rgba(135, 206, 250, 0.9);
                border: 2px solid rgba(0, 0, 0, 0.7);
                image: url(checkbox_x.png);
            }
            QCheckBox::indicator:checked:hover {
                background: rgba(135, 206, 250, 1.0);
                image: url(checkbox_x.png);
            }

            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: rgba(255,255,255,0.9);
                border: 1px solid rgba(0,0,0,0.25);
                border-radius: 10px;
                padding: 8px 10px;
                color: #000000;
                font-weight: 800;
            }
            QLineEdit:disabled, QComboBox:disabled {
                color: rgba(0,0,0,0.35);
                background: rgba(200,200,200,0.5);
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
                selection-background-color: #d1d5db;
                selection-color: black;
            }
            QGroupBox {
                color: #000000;
                border: 2px solid rgba(173, 216, 230, 0.6);
                border-radius: 12px;
                margin-top: 12px;
                padding: 10px;
                background: rgba(173, 216, 230, 0.8);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #ffffff;
                font-size: 50px;
                font-weight: bold;
            }

            QPushButton {
                background: rgba(173, 216, 230, 0.15);
                color: #e5e7eb;
                border: 2px solid rgba(135, 206, 250, 0.8);
                border-radius: 10px;
                padding: 12px 12px;
                font-weight: 900;
            }
            QPushButton:hover { 
                background: rgba(173, 216, 230, 0.7); 
                border-color: rgba(135, 206, 250, 1.0);
                color: #0f172a;
            }
            QPushButton:pressed { 
                background: rgba(173, 216, 230, 0.9);
                color: #0f172a;
            }

            QProgressBar {
                background: rgba(15,23,42,0.7);
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 9px;
                height: 18px;
                text-align: center;
                color: #f9fafb;
                font-weight: 900;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #22c55e, stop:0.5 #f59e0b, stop:1 #ef4444);
                border-radius: 9px;
            }
            QFrame#PersonCard {
                border: 2px solid rgba(125, 211, 252, 0.9);
                border-radius: 14px;
                padding: 10px;
                background: rgba(125, 211, 252, 0.5);
            }
            
            /* Make scroll areas and containers transparent */
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget {
                background: transparent;
            }
            QScrollArea QWidget#qt_scrollarea_viewport {
                background: transparent;
            }
            
            QWidget {
                background: transparent;
            }
            
            QSplitter {
                background: transparent;
            }
            
            QFrame {
                background: transparent;
            }
        """)
        self.setFont(QFont("Arial", 10))

    def go_menu(self):
        # Full reset - clear everything like starting fresh
        self.stop_playback_only()
        
        # Stop any countdown in progress
        if self.countdown_timer:
            self.countdown_timer.stop()
            self.countdown_timer = None
        self.countdown_remaining = 0
        self.lbl_countdown.setText("")
        
        # Clear frames
        self.bus.update_teacher(None, -1, 0.0)
        self.bus.update_user(None, -1, 0.0)
        
        # Clear tracking and skeleton data
        self.last_t_tracks = {}
        self.last_u_tracks = {}
        self.last_t_skel = {}
        self.last_u_skel = {}
        self.last_feedback_text = ["", ""]
        self.last_feedback_color = [(255, 255, 255), (255, 255, 255)]
        self.last_group_spacing_score = None
        self.last_group_t_norm = None
        self.last_group_u_norm = None
        self.last_group_diff = None
        
        # Reset scores
        for c in getattr(self, "person_cards", []):
            c.set_scores(0.0, {"arms": 0.0, "hands": 0.0, "torso": 0.0, "legs": 0.0})
        self.pb_spacing.setValue(0)
        self.lbl_spacing_detail.setText("ต้องตรวจพบครู 2 คน และนักเรียน 2 คน")
        self._init_session_accumulators()
        
        # Reset worker state
        self.worker.request_reset()
        
        # Clear video labels
        self.lbl_teacher.clear()
        self.lbl_user.clear()
        
        # Reset drive status
        self.lbl_drive_status.setText("Drive: -")
        self._pending_payloads = None
        
        # Reset seek to beginning
        self.seek_sec = 0.0
        self.update_seek_range()
        
        self.stack.setCurrentIndex(2)  # Go to mode selection menu (page_menu)

    def open_mode(self, mode: str):
        self.mode = mode
        self.stack.setCurrentIndex(3)  # Go to main app (page_app)

        if self.mode == "realtime":
            self.gb_realtime.setVisible(True)
            self.gb_video.setVisible(False)
            self.btn_start.setText("เริ่มกล้องเช็คแบบเรียลไทม์")
        else:
            self.gb_realtime.setVisible(False)
            self.gb_video.setVisible(True)
            self.btn_start.setText("เริ่มประมวลผลวิดีโอเทียบวิดีโอ")

        self.update_seek_range()

    def _toggle_settings(self):
        """Toggle visibility of settings panel."""
        is_hidden = self.btn_toggle_settings.isChecked()
        self.settings_panel.setVisible(not is_hidden)
        if is_hidden:
            self.btn_toggle_settings.setText("⚙️ แสดงตั้งค่า")
        else:
            self.btn_toggle_settings.setText("⚙️ ซ่อนตั้งค่า")

    def _on_students_changed(self):
        self._update_student_controls_enabled()
        self._ensure_student_cards(int(self.spin_students.value()))
        self.reset_all()

    def _on_teacher_audio_toggled(self, state):
        """Handle teacher audio checkbox toggle."""
        if state == Qt.CheckState.Checked.value or state == 2:  # Checked
            # If running and audio is set up, start playing
            if self.running and self.teacher_audio_extracted:
                self.teacher_audio.set_volume(self.slider_volume.value() / 100.0)
                self.teacher_audio.play(self.seek_sec)
        else:
            # Stop audio
            self.teacher_audio.stop()

    def _on_volume_changed(self, value):
        """Handle volume slider change."""
        self.lbl_volume.setText(f"Vol: {value}%")
        self.teacher_audio.set_volume(value / 100.0)

    def _clear_student_cards(self):
        while self.score_layout.count():
            item = self.score_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        self.person_cards = []

    def _ensure_student_cards(self, n_students: int):
        n_students = int(clamp(n_students, 1, 2))
        self._clear_student_cards()

        self.person_cards = []
        for i in range(n_students):
            card = PersonCard(self.get_student_name(i + 1))
            self.score_layout.addWidget(card)
            self.person_cards.append(card)
        self.score_layout.addStretch(1)

        self._init_session_accumulators()

    def pick_user_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกวิดีโอตัวเอง", "", "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)")
        if not path:
            return
        self.user_video_path = path
        self.lbl_user_path.setText(path)

        fps, frames, dur = probe_video(path)
        self.user_fps_file = fps
        self.user_frames_total = frames
        self.user_dur_sec = dur

        self.update_seek_range()

    @Slot(object)
    def on_worker_result(self, result: dict):
        self.last_t_tracks = result.get("t_tracks", {}) or {}
        self.last_u_tracks = result.get("u_tracks", {}) or {}
        self.last_t_skel = result.get("t_skel", {}) or {}
        self.last_u_skel = result.get("u_skel", {}) or {}

        scores = result.get("scores", []) or []

        self.last_group_spacing_score = result.get("group_spacing_score", None)
        self.last_group_t_norm = result.get("group_t_norm", None)
        self.last_group_u_norm = result.get("group_u_norm", None)
        self.last_group_diff = result.get("group_diff", None)

        if self.last_group_spacing_score is None:
            self.pb_spacing.setValue(0)
            self.lbl_spacing_detail.setText("Need 2 teachers + 2 students detected")
        else:
            self.pb_spacing.setValue(int(round(self.last_group_spacing_score)))
            self.lbl_spacing_detail.setText(
                f"TeacherDist(norm)={self.last_group_t_norm:.2f} | StudentDist(norm)={self.last_group_u_norm:.2f} | diff={self.last_group_diff:.2f}"
            )
            if self.running:
                self.acc_group["count"] += 1
                self.acc_group["sum_spacing"] += float(self.last_group_spacing_score)

        for item in scores:
            sid = int(item.get("student_id", 1))
            overall = float(item.get("overall", 0.0))
            parts = item.get("parts", {}) or {}
            ftxt = item.get("feedback_text", "")
            fcol = item.get("feedback_color", (255, 255, 255))

            if 1 <= sid <= len(self.person_cards):
                self.person_cards[sid - 1].title.setText(self.get_student_name(sid))
                self.person_cards[sid - 1].set_scores(overall, parts)

            if sid - 1 < len(self.last_feedback_text):
                self.last_feedback_text[sid - 1] = ftxt
                self.last_feedback_color[sid - 1] = fcol

            if self.running and (sid - 1) < len(self.acc):
                a = self.acc[sid - 1]
                a["count"] += 1
                a["sum_overall"] += float(overall)
                a["sum_arms"] += float(parts.get("arms", 0.0))
                a["sum_hands"] += float(parts.get("hands", 0.0))
                a["sum_torso"] += float(parts.get("torso", 0.0))
                a["sum_legs"] += float(parts.get("legs", 0.0))

    def _stop_threads(self):
        for th in [self.th_teacher, self.th_user]:
            if th is not None:
                try:
                    th.stop()
                except Exception:
                    pass
        for th in [self.th_teacher, self.th_user]:
            if th is not None:
                try:
                    th.wait(2500)
                except Exception:
                    pass
        self.th_teacher = None
        self.th_user = None

    def _push_worker_settings(self):
        n = int(self.spin_students.value())
        teacher_map = [self.get_teacher_ref_for_student(i + 1) for i in range(n)]
        st = WorkerSettings(
            proc_interval_sec=0.5,
            n_students=n,
            win_frames=int(self.win_frames),
            hands_enabled=bool(self.cb_hands.isChecked()),
            teacher_map=teacher_map,
            spacing_weight=0.12
        )
        self.worker.update_settings(st)

    def _set_controls_enabled(self, enabled: bool):
        for w in [
            self.btn_start, self.btn_stop_save, self.btn_reset, self.btn_back,
            self.btn_pick_user_video, self.teacher_combo, self.btn_refresh_teachers, self.btn_add_teacher
        ]:
            try:
                w.setEnabled(enabled)
            except Exception:
                pass

    def start(self):
        if not self.teacher_path:
            QMessageBox.warning(self, "ยังไม่พร้อม", "กรุณาเลือกวิดีโอครูก่อน (จาก Dropdown หรือ Add Teacher Video…)")
            return
        if self.mode == "video" and not self.user_video_path:
            QMessageBox.warning(self, "ยังไม่พร้อม", "กรุณาเลือกวิดีโอตัวเอง (โหมด Video vs Video)")
            return

        if (not YOLO_OK) or (self.worker.detector is None):
            QMessageBox.information(
                self, "YOLO ไม่พร้อม",
                "ระบบจะ fallback เป็น 1 คนเต็มเฟรม\n"
                "ถ้าต้องการแยก 2 คนให้ดี: pip install ultralytics"
            )

        # Start countdown if enabled
        countdown_sec = int(self.spin_countdown.value())
        if countdown_sec > 0:
            self.countdown_remaining = countdown_sec
            self.lbl_countdown.setText(f"⏱️ เตรียมตัว {self.countdown_remaining} วินาที...")
            self.btn_start.setEnabled(False)
            self.countdown_timer = QTimer(self)
            self.countdown_timer.timeout.connect(self._on_countdown_tick)
            self.countdown_timer.start(1000)
            return
        
        # No countdown - start immediately
        self._do_start()

    def _on_countdown_tick(self):
        """Handle countdown timer tick."""
        self.countdown_remaining -= 1
        if self.countdown_remaining <= 0:
            # Countdown finished - start the game
            if self.countdown_timer:
                self.countdown_timer.stop()
                self.countdown_timer = None
            self.lbl_countdown.setText("🎬 เริ่ม!")
            self.btn_start.setEnabled(True)
            # Clear countdown label after a short delay
            QTimer.singleShot(1000, lambda: self.lbl_countdown.setText(""))
            self._do_start()
        else:
            self.lbl_countdown.setText(f"⏱️ เตรียมตัว {self.countdown_remaining} วินาที...")

    def _do_start(self):
        """Actually start the video playback and scoring."""
        self.lbl_drive_status.setText("Drive: -")
        self._pending_payloads = None

        self.worker.request_reset()
        self._stop_threads()

        self._init_session_accumulators()
        self.session_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        start_wall = time.perf_counter()

        self.update_seek_range()
        sec = float(clamp(self.seek_sec, 0.0, max(0.0, self.seek_max_sec - 0.05)))

        fps_t = float(self.teacher_fps_file if self.teacher_fps_file > 1e-6 else 30.0)
        start_frame_t = int(round(sec * fps_t))

        self.th_teacher = VideoReaderThread(
            self.teacher_path,
            is_camera=False,
            mirror=False,
            max_w=860,
            emit_fps=30.0,
            start_wall=start_wall,
            start_frame=start_frame_t
        )
        self.th_teacher.frameReady.connect(lambda f, idx, ts: self.bus.update_teacher(f, idx, ts))
        self.th_teacher.metaReady.connect(self._on_teacher_meta)
        self.th_teacher.start()

        if self.mode == "realtime":
            cam = int(self.spin_cam.value())
            mirror = int(self.spin_mirror.value()) == 1

            self.realtime_record_path = None
            if self.cb_upload_drive.isChecked():
                ensure_recordings_dir()
                self.realtime_record_path = os.path.join(RECORDINGS_DIR, f"realtime_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

            self.th_user = VideoReaderThread(
                cam,
                is_camera=True,
                mirror=mirror,
                max_w=860,
                emit_fps=30.0,
                record_path=self.realtime_record_path,
                record_fps=30.0
            )
            self.th_user.metaReady.connect(self._on_user_meta)
        else:
            fps_u = float(self.user_fps_file if self.user_fps_file > 1e-6 else 30.0)
            start_frame_u = int(round(sec * fps_u))
            self.th_user = VideoReaderThread(
                self.user_video_path,
                is_camera=False,
                mirror=False,
                max_w=860,
                emit_fps=30.0,
                start_wall=start_wall,
                start_frame=start_frame_u
            )
            self.th_user.metaReady.connect(self._on_user_meta)

        self.th_user.frameReady.connect(lambda f, idx, ts: self.bus.update_user(f, idx, ts))
        self.th_user.start()

        self._recalc_win_frames()
        self._push_worker_settings()

        self.running = True
        self.worker.set_running(True)
        
        # Start teacher audio if enabled
        if self.cb_teacher_audio.isChecked() and self.teacher_audio_extracted:
            self.teacher_audio.set_volume(self.slider_volume.value() / 100.0)
            self.teacher_audio.play(sec)

    @Slot(float)
    def _on_teacher_meta(self, fps: float):
        self.teacher_fps = float(fps if fps and fps > 1e-6 else 30.0)
        self._recalc_win_frames()
        self._push_worker_settings()

    @Slot(float)
    def _on_user_meta(self, fps: float):
        self.bus.user_fps = float(fps if fps and fps > 1e-6 else 30.0)

    # ---- stop only (no save/upload) ----
    def stop_playback_only(self):
        self.running = False
        self.worker.set_running(False)
        self._stop_threads()
        # Stop teacher audio
        self.teacher_audio.stop()

    # ---- stop + save to sheet + optional drive upload ----
    def stop_and_save(self):
        was_running = self.running
        self.running = False
        self.worker.set_running(False)
        self._stop_threads()
        # Stop teacher audio
        self.teacher_audio.stop()

        if not was_running:
            return

        ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        n = int(self.spin_students.value())

        group_spacing_avg = None
        if self.acc_group["count"] > 0:
            group_spacing_avg = float(self.acc_group["sum_spacing"] / float(self.acc_group["count"]))

        payloads: List[dict] = []
        for student_id in range(1, n + 1):
            a = self.acc[student_id - 1] if (student_id - 1) < len(self.acc) else None
            if not a or a["count"] <= 0:
                continue

            cnt = float(a["count"])
            avg_overall = a["sum_overall"] / cnt
            avg_parts = {
                "arms": a["sum_arms"] / cnt,
                "hands": a["sum_hands"] / cnt,
                "torso": a["sum_torso"] / cnt,
                "legs": a["sum_legs"] / cnt,
            }

            fb, _ = self.worker.score_to_feedback(avg_overall)

            payload = {
                "timestamp": ended_at,
                "mode": self.mode,
                "teacher_video": self.teacher_path or "",
                "user_video": self.user_video_path or "",
                "student_id": student_id,
                "student_name": self.get_student_name(student_id),
                "teacher_ref": self.get_teacher_ref_for_student(student_id),
                "overall": round(float(avg_overall), 2),
                "arms": round(float(avg_parts["arms"]), 2),
                "hands": round(float(avg_parts["hands"]), 2),
                "torso": round(float(avg_parts["torso"]), 2),
                "legs": round(float(avg_parts["legs"]), 2),
                "group_spacing_avg": (None if group_spacing_avg is None else round(float(group_spacing_avg), 2)),
                "feedback": fb,
                # will be filled if drive upload ok:
                "student_video_drive_file_id": "",
                "student_video_drive_link": "",
                "student_video_drive_name": "",
            }
            payloads.append(payload)

        if not payloads:
            QMessageBox.information(self, "ไม่มีข้อมูล", "ยังไม่มีคะแนนสะสมพอที่จะบันทึก (ลองกดเริ่มแล้วรอสักครู่)")
            return

        do_upload = self.cb_upload_drive.isChecked()
        upload_path = None
        if do_upload:
            if self.mode == "realtime":
                upload_path = self.realtime_record_path
            else:
                upload_path = self.user_video_path

        if do_upload and upload_path and os.path.exists(upload_path):
            if not DRIVE_OK:
                QMessageBox.warning(self, "Drive ไม่พร้อม", "ยังไม่ได้ติดตั้งไลบรารี Drive\nรัน: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
                self._post_payloads_with_drive(payloads, None)
                return
            if not os.path.exists(DRIVE_CREDENTIALS_FILE):
                QMessageBox.warning(self, "Drive ไม่พร้อม", f"ไม่พบ credentials.json ที่:\n{DRIVE_CREDENTIALS_FILE}")
                self._post_payloads_with_drive(payloads, None)
                return

            # upload in background thread
            self._pending_payloads = payloads
            self.lbl_drive_status.setText("Drive: uploading...")
            self._set_controls_enabled(False)

            self._drive_upload_thread = DriveUploadThread(
                upload_path,
                folder_id=DRIVE_FOLDER_ID,
                make_shared=bool(self.cb_drive_share.isChecked())
            )
            self._drive_upload_thread.done.connect(self._on_drive_upload_done)
            self._drive_upload_thread.start()
            return

        # no upload
        self._post_payloads_with_drive(payloads, None)

    @Slot(object)
    def _on_drive_upload_done(self, result: dict):
        self._set_controls_enabled(True)

        payloads = self._pending_payloads or []
        self._pending_payloads = None

        if not payloads:
            self.lbl_drive_status.setText("Drive: -")
            return

        if not result.get("ok", False):
            err = result.get("error", "unknown error")
            self.lbl_drive_status.setText(f"Drive: failed - {err}")
            QMessageBox.warning(self, "อัปโหลด Drive ไม่สำเร็จ", str(err))
            self._post_payloads_with_drive(payloads, None)
            return

        info = result.get("info", {}) or {}
        link = info.get("webViewLink", "")
        name = info.get("name", "")
        self.lbl_drive_status.setText(f"Drive: uploaded - {name}")

        self._post_payloads_with_drive(payloads, info)

    def _post_payloads_with_drive(self, payloads: List[dict], drive_info: Optional[dict]):
        if drive_info:
            fid = drive_info.get("id", "")
            link = drive_info.get("webViewLink", "")
            name = drive_info.get("name", "")
            for p in payloads:
                p["student_video_drive_file_id"] = fid
                p["student_video_drive_link"] = link
                p["student_video_drive_name"] = name

        posted_any = False
        for p in payloads:
            print("[INFO] Posting payload:", {k: p.get(k) for k in ["timestamp", "student_id", "student_name", "overall", "student_video_drive_link"]})
            post_score_to_gsheet(p)
            posted_any = True

        if posted_any:
            QMessageBox.information(self, "บันทึกเรียบร้อย", "บันทึกค่าเฉลี่ยลง Google Sheet แล้ว (แนบลิงก์วิดีโอนักเรียนถ้ามี)")
        else:
            QMessageBox.information(self, "ไม่มีข้อมูล", "ยังไม่มีข้อมูลเพียงพอที่จะบันทึก")

    def reset_all(self):
        self.stop_playback_only()
        for c in getattr(self, "person_cards", []):
            c.set_scores(0.0, {"arms": 0.0, "hands": 0.0, "torso": 0.0, "legs": 0.0})
        self.pb_spacing.setValue(0)
        self.lbl_spacing_detail.setText("Need 2 teachers + 2 students detected")
        self._init_session_accumulators()
        self.last_t_tracks = {}
        self.last_u_tracks = {}
        self.last_t_skel = {}
        self.last_u_skel = {}
        self.last_group_spacing_score = None
        self.lbl_drive_status.setText("Drive: -")

    def on_tick(self):
        if self.stack.currentIndex() != 3:  # Check if on main app page (page_app)
            return

        self._push_worker_settings()

        tf, t_idx, uf, u_idx = self.bus.snapshot()

        if tf is None:
            tf = np.zeros((460, 620, 3), dtype=np.uint8)
            cv2.putText(tf, "Select teacher video", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        if uf is None:
            uf = np.zeros((460, 620, 3), dtype=np.uint8)
            msg = "Camera (Realtime)" if self.mode == "realtime" else "Select your video"
            cv2.putText(uf, msg, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        show_teacher = self.cb_show_teacher_video.isChecked()

        if not show_teacher:
            tf_show = np.zeros_like(tf)
            cv2.putText(tf_show, "Teacher video hidden", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        else:
            tf_show = tf.copy()

        uf_show = uf.copy()

        show_pose_lines = self.cb_show_pose_lines.isChecked()
        show_teacher_lines = show_pose_lines and self.cb_draw_teacher.isChecked()

        if show_teacher:
            for tid, bbox in (self.last_t_tracks or {}).items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(tf_show, (x1, y1), (x2, y2), (120, 255, 120), 2)
                cv2.putText(tf_show, f"Teacher {tid}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2)
                if show_teacher_lines and tid in (self.last_t_skel or {}):
                    self.draw_skeleton(tf_show, self.last_t_skel[tid], color=(160, 200, 255))

        for tid, bbox in (self.last_u_tracks or {}).items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(uf_show, (x1, y1), (x2, y2), (120, 255, 120), 2)
            name = self.get_student_name(tid)
            cv2.putText(uf_show, name, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2)
            if show_pose_lines and tid in (self.last_u_skel or {}):
                self.draw_skeleton(uf_show, self.last_u_skel[tid], color=(255, 180, 180))

        n_students = int(self.spin_students.value())
        for sid in range(1, n_students + 1):
            txt = self.last_feedback_text[sid - 1] if (sid - 1) < len(self.last_feedback_text) else ""
            col = self.last_feedback_color[sid - 1] if (sid - 1) < len(self.last_feedback_color) else (255, 255, 255)
            if not txt:
                continue
            if sid in (self.last_u_tracks or {}):
                x1, y1, x2, y2 = self.last_u_tracks[sid]
                badge_x = x1 + 10
                badge_y = y2 + 38
                if badge_y > uf_show.shape[0] - 10:
                    badge_y = y1 - 8
                self.draw_feedback_badge(uf_show, badge_x, badge_y, txt, col)
            else:
                self.draw_feedback_badge(uf_show, 20, 60 + (sid - 1) * 60, f"Student {sid}: {txt}", col)

        if n_students >= 2 and self.last_group_spacing_score is not None:
            txt = f"Spacing {int(round(self.last_group_spacing_score))}%"
            col = (80, 255, 120) if self.last_group_spacing_score >= 85 else ((0, 220, 255) if self.last_group_spacing_score >= 70 else (60, 60, 255))
            self.draw_feedback_badge(uf_show, 20, 40, txt, col)

        t_canvas, _, _, _ = make_canvas_fit(tf_show, self.lbl_teacher.width(), self.lbl_teacher.height())
        u_canvas, _, _, _ = make_canvas_fit(uf_show, self.lbl_user.width(), self.lbl_user.height())

        self.lbl_teacher.setPixmap(bgr_canvas_to_qpixmap(t_canvas))
        self.lbl_user.setPixmap(bgr_canvas_to_qpixmap(u_canvas))

    def closeEvent(self, event):
        try:
            self.stop_playback_only()
            self.worker.stop()
            self.worker.wait(1200)
            # Cleanup audio resources
            self.teacher_audio.cleanup()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)

    # ❗ บังคับ style ให้เหมือนกันทุกเครื่อง (สำคัญมาก)
    app.setStyle("Fusion")

    # โหลด QSS จากไฟล์ (ถ้ามี)
    qss_path = resource_path("style.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            # ❌ อย่า setStyleSheet ระดับ app
            pass
    else:
        # fallback กันเครื่องอื่น UI เพี้ยน
        app.setStyleSheet("""
            QWidget {
                background-color: #f2f2f2;
                color: #111;
            }
            QPushButton {
                background: #1f2937;
                color: white;
                padding: 6px 12px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #374151;
            }
        """)

    win = MainWindow()
    win.show()

    sys.exit(app.exec())



if __name__ == "__main__":
    main()
