# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
import os
import sys

block_cipher = None

# เก็บไฟล์ data ของ mediapipe (สำคัญมากสำหรับ solutions)
mp_datas = collect_data_files(
    "mediapipe",
    includes=[
        "**/*.binarypb",
        "**/*.tflite",
        "**/*.pbtxt",
        "**/*.txt",
        "**/*.json",
        "**/*.npy",
    ],
)

# เก็บ Qt plugins สำหรับ PySide6 (แก้ปัญหา "no Qt platform plugin could be initialized")
pyside6_datas = collect_data_files("PySide6", includes=["plugins/**/*"])

excludes = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebEngine",
    "PySide6.QtWebView",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
]

# ไฟล์ data เพิ่มเติมของโปรเจกต์
app_datas = [
    ("style.qss", "."),                      # style sheet
    ("credentials.json", "."),               # Google credentials
    ("yolov8n.pt", "."),                     # YOLO model สำหรับ 2-person detection
    ("teacher_videos", "teacher_videos"),    # โฟลเดอร์วิดีโอครู
    ("start.JPG", "."),
    ("background.JPG", "."),
    ("background2.JPG", "."),
    ("checkbox_x.png", "."),
]

# Hidden imports สำหรับ ultralytics (YOLO)
ultralytics_imports = [
    "ultralytics",
    "ultralytics.engine",
    "ultralytics.engine.model",
    "ultralytics.engine.predictor",
    "ultralytics.engine.results",
    "ultralytics.models",
    "ultralytics.models.yolo",
    "ultralytics.models.yolo.detect",
    "ultralytics.nn",
    "ultralytics.nn.tasks",
    "ultralytics.utils",
    "ultralytics.utils.ops",
    "ultralytics.utils.plotting",
    "ultralytics.data",
]

# Hidden imports สำหรับ Google Drive API
google_imports = [
    "googleapiclient",
    "googleapiclient.discovery",
    "googleapiclient.http",
    "googleapiclient.errors",
    "google.auth",
    "google.auth.transport",
    "google.auth.transport.requests",
    "google.oauth2",
    "google.oauth2.credentials",
    "google_auth_oauthlib",
    "google_auth_oauthlib.flow",
    "httplib2",
]

# Hidden imports สำหรับ PySide6 (แก้ปัญหา No module named 'PySide6.QtGui')
pyside6_imports = [
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNetwork",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
]

# Hidden imports สำหรับ MediaPipe (สำคัญมาก - แก้ปัญหา 'mediapipe has no attribute solutions')
mediapipe_imports = [
    "mediapipe",
    "mediapipe.python",
    "mediapipe.python.solutions",
    "mediapipe.python.solutions.pose",
    "mediapipe.python.solutions.drawing_utils",
    "mediapipe.python.solutions.drawing_styles",
    "mediapipe.python.solutions.hands",
    "mediapipe.python.solutions.face_mesh",
    "mediapipe.python.solutions.holistic",
    "mediapipe.modules",
    "mediapipe.modules.pose_detection",
    "mediapipe.modules.pose_landmark",
    "mediapipe.calculators",
    "mediapipe.framework",
    "mediapipe.tasks",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=mp_datas + pyside6_datas + app_datas,   # รวม mediapipe + Qt plugins + app data files
    hiddenimports=ultralytics_imports + google_imports + pyside6_imports + mediapipe_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ทำ EXE แบบ onedir (exclude_binaries=True) แล้วค่อย COLLECT
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ThaiDanceScoring",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="ThaiDanceScoring",
)

app = BUNDLE(
    coll,
    name="ThaiDanceScoring.app",
    icon=None,
    bundle_identifier="com.rumthai.thaidancescoring",
    info_plist={
        "NSCameraUsageDescription": "This app requires camera access for dance pose comparison.",
        "NSMicrophoneUsageDescription": "This app may record audio with video.",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
    },
)
