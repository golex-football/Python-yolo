import cv2
import time
import numpy as np
from ultralytics import YOLO

# ------------- CONFIG -------------

# Try 0 first. If it doesn't show OBS feed, change this to 1 and run again.
CAMERA_INDEX = 1  # you already found 1 works, keep it on 1

# Use a smaller, faster segmentation model
MODEL_NAME = "yolov8s-seg.pt"  # smaller than m, much faster

CONF_THRES = 0.5

# Inference resolution (width). YOLO will internally resize to something like this.
# Smaller = faster but less detail. We start with 960.
INFER_WIDTH = 960

# ----------------------------------

def build_binary_mask(results, frame_shape):
    """
    Build a single-channel binary mask (uint8, 0 or 255) where:
    - white (255) = all persons (players, referees, etc.)
    - black (0)   = everything else
    """
    h, w = frame_shape[:2]

    if results.masks is None:
        return np.zeros((h, w), dtype=np.uint8)

    masks = results.masks.data.cpu().numpy()         # [N, mask_h, mask_w]
    classes = results.boxes.cls.cpu().numpy().astype(int)

    combined = np.zeros((h, w), dtype=np.uint8)

    for i, m in enumerate(masks):
        cls_id = classes[i]

        # COCO person class is usually 0
        if cls_id != 0:
            continue

        # resize mask to frame size
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        m_bin = (m_resized > 0.5).astype(np.uint8) * 255
        combined = np.maximum(combined, m_bin)

        # Smooth edges a bit and remove tiny noise
    combined = cv2.GaussianBlur(combined, (5, 5), 0)
    _, combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)

    # Morphological closing to fill small gaps and smooth contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    return combined


def main():
    print("Loading YOLO model:", MODEL_NAME)
    model = YOLO(MODEL_NAME)  # this will download weights on first run

    # --- DEBUG: show what device YOLO is using ---
    try:
        import torch
        print("Torch CUDA available:", torch.cuda.is_available())
        print("Model first param device:", next(model.model.parameters()).device)
    except Exception as e:
        print("Could not inspect model device:", e)


    print(f"Opening camera index {CAMERA_INDEX} (OBS Virtual Camera)")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        print("Try changing CAMERA_INDEX to 1 or another value and run again.")
        return

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("WARNING: Empty frame from camera.")
            time.sleep(0.1)
            continue

                # frame from OpenCV is BGR, YOLO accepts BGR just fine

        # --- Resize frame for faster inference ---
        orig_h, orig_w = frame.shape[:2]
        scale = INFER_WIDTH / float(orig_w)
        new_w = INFER_WIDTH
        new_h = int(orig_h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        t0 = time.time()
        results_list = model.predict(
            frame_resized,
            conf=CONF_THRES,
            imgsz=INFER_WIDTH,
            device=0,          # force GPU 0
            verbose=False
        )
        infer_time = (time.time() - t0) * 1000.0  # ms

        results = results_list[0]

        # Build binary mask at resized resolution
        mask_resized = build_binary_mask(results, frame_resized.shape)

        # Upscale mask back to original frame size for display
        mask = cv2.resize(mask_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


        # For visualization: show original + mask side by side
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((frame, mask_bgr))

        cv2.imshow("Original (left) | Mask (right)", stacked)

        # FPS calculation
        frame_count += 1
        now = time.time()
        elapsed = now - prev_time
        if elapsed > 1.0:
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f} | Inference: {infer_time:.1f} ms | "
                  f"Detections: {len(results.boxes)}")
            prev_time = now
            frame_count = 0

        # ESC to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()
