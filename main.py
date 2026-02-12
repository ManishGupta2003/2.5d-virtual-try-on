# main.py (stability-focused smoothing)
import cv2
import mediapipe as mp
import os
import sys
import time
from threading import Thread
from tkinter import Tk, filedialog
from collections import deque
from utils.overlay_utils import overlay_clothes, overlay_clothes_fast, clear_overlay_cache
import numpy as np

# ---------- Config ----------
CAM_INDEX = int(os.environ.get("CAM_INDEX", 0))
TARGET_W, TARGET_H = 640, 480
POSE_SCALE = 0.5
PROCESS_EVERY_N = 2
MIN_FPS_FOR_FULL_WARP = 14

# Smoothing parameters
ALPHA = 0.35        # exponential smoothing factor: 0 < ALPHA <= 1 (higher = more responsive, lower = smoother)
MAX_JUMP = 0.10     # max allowed normalized jump per frame (as fraction of frame width/height)
# ----------------------------

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

class VideoGrabber:
    def __init__(self, src=0, target_w=TARGET_W, target_h=TARGET_H):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
    def read(self):
        if self.frame is None:
            return False, None
        return self.ret, self.frame.copy()
    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()

def upload_clothing():
    root = Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
    root.update(); root.destroy()
    if not file_path:
        return None, None
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if img.ndim == 3 and img.shape[2] == 3:
        bgr = img
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
        alpha = 255 - mask
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        img[:, :, 3] = alpha
    elif img.ndim == 2:
        bgra = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA); bgra[:,:,3]=255; img = bgra
    return img, os.path.basename(file_path)

# load clothing
ASSETS_DIR = "assets"
DEFAULT_ASSETS = ["tshirt.png","jacket.png","hoodie.png","tshirt2.png","tshirt3.png","shirt.png","shirt2.png"]
clothing_options, clothing_names = [], []
for fname in DEFAULT_ASSETS:
    p = os.path.join(ASSETS_DIR, fname)
    if os.path.exists(p):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is not None:
            clothing_options.append(img); clothing_names.append(os.path.splitext(fname)[0])
current_index = 0
draw_landmarks = False

# mediapipe pose (lighter model for speed)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

vg = VideoGrabber(CAM_INDEX, TARGET_W, TARGET_H)
frame_count = 0
last_time = time.time()
fps_smooth = 0.0

# For smoothing: maintain smoothed normalized positions for key indices
# We'll track these landmarks: 11,12,13,14,15,16,23,24  (ls,rs,le,re,lw,rw,lh,rh)
KEY_IDS = [11,12,13,14,15,16,23,24]
smoothed = {k: None for k in KEY_IDS}

def clamp_jump(prev, curr, max_jump):
    """Clamp curr so it doesn't jump more than max_jump (in normalized coords) from prev."""
    if prev is None:
        return curr
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    dist = np.sqrt(dx*dx + dy*dy)
    if dist <= max_jump:
        return curr
    # scale down the jump
    ratio = max_jump / (dist + 1e-9)
    return (prev[0] + dx * ratio, prev[1] + dy * ratio)

print("Controls: q-quit, c-next, x-prev, u-upload, s-save, d-toggle landmarks, h-help")

try:
    while True:
        ret, frame = vg.read()
        if not ret or frame is None:
            time.sleep(0.01); continue

        frame = cv2.flip(frame, 1)
        frame_count += 1

        now = time.time()
        dt = now - last_time if now != last_time else 0.001
        fps_inst = 1.0 / dt if dt > 0 else 0.0
        last_time = now
        fps_smooth = 0.92 * fps_smooth + 0.08 * fps_inst

        do_process = (frame_count % PROCESS_EVERY_N) == 0
        results = None
        if do_process:
            small = cv2.resize(frame, (0,0), fx=POSE_SCALE, fy=POSE_SCALE, interpolation=cv2.INTER_LINEAR)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_small)

            if results.pose_landmarks:
                # Update smoothed positions for each tracked key
                fh, fw = frame.shape[:2]
                for kid in KEY_IDS:
                    kp = results.pose_landmarks.landmark[kid]
                    # normalized coords from small frame are compatible with original normalization; keep normalized
                    curr = (kp.x, kp.y)
                    prev = smoothed.get(kid)
                    # clamp sudden jumps (normalized)
                    curr_clamped = clamp_jump(prev, curr, MAX_JUMP)
                    if prev is None:
                        smoothed[kid] = curr_clamped
                    else:
                        # exponential smoothing
                        sx = ALPHA * curr_clamped[0] + (1.0 - ALPHA) * prev[0]
                        sy = ALPHA * curr_clamped[1] + (1.0 - ALPHA) * prev[1]
                        smoothed[kid] = (sx, sy)
            # else: do not update smoothed (keep previous)

        # Build a simple landmark-like list using smoothed positions for overlay (create light objects with x,y)
        landmark_list = None
        if any(smoothed[k] is not None for k in KEY_IDS):
            class _KP:
                def __init__(self,x,y): self.x=x; self.y=y; self.z=0; self.visibility=1.0
            # create length-33 dummy list filled with zeros but we will set required ids
            landmark_list = [type('L', (), {'x':0,'y':0,'z':0,'visibility':0})() for _ in range(33)]
            for k in KEY_IDS:
                sx, sy = smoothed[k] if smoothed[k] is not None else (0,0)
                landmark_list[k] = _KP(sx, sy)

        used_fast = False
        if landmark_list and clothing_options:
            if fps_smooth < MIN_FPS_FOR_FULL_WARP:
                used_fast = True
                frame = overlay_clothes_fast(frame, clothing_options[current_index], landmark_list)
            else:
                frame = overlay_clothes(frame, clothing_options[current_index], landmark_list)

        # draw landmarks if desired (draw original results if available)
        if draw_landmarks and results is not None and results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # UI
        label = clothing_names[current_index] if clothing_names else "None"
        cv2.putText(frame, f"Cloth: {label}  FPS: {int(fps_smooth)} {'FAST' if used_fast else ''}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Virtual Try-On (stable)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            current_index = (current_index + 1) % len(clothing_options) if clothing_options else 0
            clear_overlay_cache()
            print("[+] Switched to:", clothing_names[current_index] if clothing_names else "None")
        elif key == ord('x'):
            current_index = (current_index - 1) % len(clothing_options) if clothing_options else 0
            clear_overlay_cache()
            print("[+] Switched to:", clothing_names[current_index] if clothing_names else "None")
        elif key == ord('u'):
            img, name = upload_clothing()
            if img is not None:
                clothing_options.append(img); clothing_names.append(name); current_index = len(clothing_options)-1
                clear_overlay_cache(); print("[+] Uploaded and switched to:", name)
        elif key == ord('s'):
            out = f"tryon_snapshot_{int(time.time())}.png"; cv2.imwrite(out, frame); print("[+] Saved", out)
        elif key == ord('d'):
            draw_landmarks = not draw_landmarks; print("[+] Draw landmarks:", draw_landmarks)
        elif key == ord('h'):
            print("Help: q-quit, c-next, x-prev, u-upload, s-save, d-toggle landmarks, h-help")

finally:
    vg.release()
    cv2.destroyAllWindows()
    pose.close()
