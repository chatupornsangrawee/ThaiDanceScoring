# rthook_mediapipe.py
# Runtime hook to ensure mediapipe.solutions is available on Windows
import sys

try:
    # Try the standard import first
    import mediapipe
    if not hasattr(mediapipe, 'solutions'):
        try:
            # Try importing from mediapipe.python.solutions
            import mediapipe.python.solutions as solutions
            mediapipe.solutions = solutions
        except ImportError:
            try:
                # Alternative: try to import solutions directly
                from mediapipe import solutions
            except ImportError:
                # If all else fails, just pass - the app will handle it
                pass
except ImportError:
    pass
