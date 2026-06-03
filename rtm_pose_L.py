import cv2
import threading
from rtmlib import Wholebody
import time
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import os
import uuid

DROIDCAM_URL = "http://192.168.1.X:4747/video"
CALIBRATION_DIR = "reference_images"
GALLERY_PATH = "ppe_gallery.npy"

# --- Load pose model ---
pose = Wholebody(mode="lightweight", backend="onnxruntime", device="cpu")

# --- Load ResNet18 feature extractor ---
print("Loading ResNet18...")
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.eval()
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- Load gallery if it exists ---
gallery_embeddings = None
gallery_labels = None
if os.path.exists(GALLERY_PATH):
    gallery = np.load(GALLERY_PATH, allow_pickle=True).item()
    gallery_embeddings = gallery["embeddings"]
    gallery_labels = gallery["labels"]
    print(f"Gallery loaded: {len(gallery_labels)} embeddings, classes: {set(gallery_labels)}")
else:
    print("⚠️ No gallery found. Calibrate first using 'p' and 'n' keys, then rebuild.")

ROI_TO_CLASSES = {
    "goggles":  ("glasses", "no_glasses"),
}

K = 5

def get_embedding(crop):
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        emb = feature_extractor(tensor).squeeze().numpy()
        emb = emb / np.linalg.norm(emb)
    return emb

def check_ppe(crop, positive_class, negative_class):
    if gallery_embeddings is None:
        return False, 0.0, 0.0

    emb = get_embedding(crop)
    similarities = gallery_embeddings @ emb

    # Get average similarity for each class
    pos_sims = [similarities[i] for i, l in enumerate(gallery_labels) if l == positive_class]
    neg_sims = [similarities[i] for i, l in enumerate(gallery_labels) if l == negative_class]

    avg_pos = np.mean(pos_sims) if pos_sims else 0.0
    avg_neg = np.mean(neg_sims) if neg_sims else 0.0

    present = avg_pos > avg_neg
    return present, avg_pos, avg_neg

def save_calibration(rois, label_prefix):
    for roi_name, crop in rois.items():
        if crop is None:
            continue
        if "goggles" in roi_name:
            folder = f"{label_prefix}glasses"
        else:
            continue

        save_dir = os.path.join(CALIBRATION_DIR, folder)
        os.makedirs(save_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        path = os.path.join(save_dir, f"cal_{unique_id}.jpg")
        cv2.imwrite(path, crop)
        print(f"💾 Saved {path}")

def get_ppe_rois(frame, keypoints, scores, min_score=0.5):
    h, w = frame.shape[:2]
    rois = {}
    boxes = {}

    def safe_crop(x1, y1, x2, y2):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None, None
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    kp = keypoints
    sc = scores

    eye_ids = [1, 2]
    eye_pts = [(kp[i], sc[i]) for i in eye_ids if sc[i] > min_score]
    if len(eye_pts) == 2:
        xs = [p[0][0] for p in eye_pts]
        ys = [p[0][1] for p in eye_pts]
        eye_width = abs(xs[1] - xs[0])
        pad = eye_width * 0.6
        crop, box = safe_crop(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
        if crop is not None:
            rois["goggles"] = crop
            boxes["goggles"] = box

    return rois, boxes

# --- Shared state ---
latest_frame = None
latest_keypoints = None
latest_scores = None
lock = threading.Lock()
running = True

def capture_thread():
    global latest_frame, running
    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            with lock:
                latest_frame = frame
    cap.release()

def pose_thread():
    global latest_keypoints, latest_scores, running
    while running:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            continue
        small = cv2.resize(frame, (320, 240))
        kpts, scores = pose(small)
        if kpts is not None:
            kpts[:, :, 0] *= 2
            kpts[:, :, 1] *= 2
        with lock:
            latest_keypoints = kpts
            latest_scores = scores

# --- Start threads ---
t1 = threading.Thread(target=capture_thread, daemon=True)
t2 = threading.Thread(target=pose_thread, daemon=True)
t1.start()
t2.start()

prev_time = time.time()
print("\n🎮 Controls:")
print("  'p' = capture POSITIVE samples (wear the equipment)")
print("  'n' = capture NEGATIVE samples (no equipment)")
print("  'q' = quit")
print("After calibrating, quit, run build_gallery.py, then run this again.\n")

opened_windows = set()

while True:
    with lock:
        frame = latest_frame.copy() if latest_frame is not None else None
        kpts = latest_keypoints
        scrs = latest_scores

    if frame is None:
        continue

    displayed_windows = set()

    if kpts is not None and len(kpts) > 0:
        all_rois = {}
        all_boxes = {}
        for person_id in range(len(kpts)):
            rois, boxes = get_ppe_rois(frame, kpts[person_id], scrs[person_id])
            for name, crop in rois.items():
                all_rois[f"p{person_id}_{name}"] = crop
            for name, box in boxes.items():
                all_boxes[f"p{person_id}_{name}"] = box
        rois = all_rois
        boxes = all_boxes

        for name, (x1, y1, x2, y2) in boxes.items():
            base_name = name.split("_", 1)[1]
            color = (255, 255, 255)
            label = base_name

            if base_name in ROI_TO_CLASSES and name in rois and rois[name] is not None:
                pos_class, neg_class = ROI_TO_CLASSES[base_name]
                present, avg_pos, avg_neg = check_ppe(rois[name], pos_class, neg_class)
                if present:
                    color = (0, 255, 0)
                    label = f"{base_name} OK ({avg_pos:.2f}>{avg_neg:.2f})"
                else:
                    color = (0, 0, 255)
                    label = f"{base_name} MISSING ({avg_neg:.2f}>{avg_pos:.2f})"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the separate ROI crop in its own window
            if name in rois and rois[name] is not None and rois[name].size > 0:
                cv2.imshow(name, rois[name])
                displayed_windows.add(name)
                opened_windows.add(name)

    # Close separate windows that are no longer active or detected
    for name in list(opened_windows):
        if name not in displayed_windows:
            try:
                cv2.destroyWindow(name)
            except Exception:
                pass
            opened_windows.remove(name)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-9)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("PPE Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        running = False
        break
    elif key == ord('p'):
        if kpts is not None and len(kpts) > 0:
            for person_id in range(len(kpts)):
                rois_cal, _ = get_ppe_rois(frame, kpts[person_id], scrs[person_id])
                save_calibration(rois_cal, "")
            print("✅ Positive samples saved!")
    elif key == ord('n'):
        if kpts is not None and len(kpts) > 0:
            for person_id in range(len(kpts)):
                rois_cal, _ = get_ppe_rois(frame, kpts[person_id], scrs[person_id])
                save_calibration(rois_cal, "no_")
            print("❌ Negative samples saved!")

cv2.destroyAllWindows()