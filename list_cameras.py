
import cv2

def list_cameras(max_check=5):
    available = []
    print(f"Checking cameras 0 to {max_check-1}...")
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i}: OK (Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
                available.append(i)
            else:
                print(f"Camera {i}: Opened but failed to read frame")
            cap.release()
        else:
            print(f"Camera {i}: Not found")
    return available

if __name__ == "__main__":
    cams = list_cameras()
    print(f"Available camera indices: {cams}")
