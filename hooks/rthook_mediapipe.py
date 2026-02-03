# rthook_mediapipe.py - Runtime hook for MediaPipe on Windows
# This hook ensures mediapipe.solutions is properly accessible

try:
    import mediapipe
    # Try to access solutions directly first
    if not hasattr(mediapipe, 'solutions'):
        try:
            import mediapipe.python.solutions as solutions
            mediapipe.solutions = solutions
        except ImportError:
            # If mediapipe.python.solutions doesn't exist, try direct import
            try:
                from mediapipe import solutions
            except ImportError:
                pass  # Solutions might be loaded differently
except ImportError:
    pass  # MediaPipe not available, skip hook
