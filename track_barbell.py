import os
import shutil
import glob
import tkinter as tk
from tkinter import filedialog
import cv2
import torch
import numpy as np
import pandas as pd
from sam2.build_sam import build_sam2_video_predictor

# --- STEP 0: FILE SELECTION DIALOG ---
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

INPUT_VIDEO_PATH = filedialog.askopenfilename(
    title="Select Lifting Video to Analyze",
    filetypes=[
        ("Video Files", "*.mov *.mp4 *.avi *.mkv *.MOV *.MP4 *.m4v"),
        ("All Files", "*.*")
    ]
)
root.destroy()

if not INPUT_VIDEO_PATH:
    print("No file selected. Exiting.")
    exit()

# --- CONFIGURATION & PATH SETUP ---
CHECKPOINT = "checkpoints/sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

OUTPUT_BASE_DIR = "outputs"
FRAMES_DIR = "temp_frames"
TARGET_WIDTH = 720
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Automatically mirror the video's parent folder name (e.g., "aug16_2026") under outputs/
parent_dir_name = os.path.basename(os.path.dirname(INPUT_VIDEO_PATH))
TARGET_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, parent_dir_name) if parent_dir_name else OUTPUT_BASE_DIR
os.makedirs(TARGET_OUTPUT_DIR, exist_ok=True)

# 1. PURGE & RECREATE temp_frames (Fixes dimension collision)
if os.path.exists(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)
os.makedirs(FRAMES_DIR, exist_ok=True)

input_file_name = os.path.basename(INPUT_VIDEO_PATH)
base_name, _ = os.path.splitext(input_file_name)
OUTPUT_VIDEO_PATH = os.path.join(TARGET_OUTPUT_DIR, f"{base_name}_OUT.mp4")
OUTPUT_CSV_PATH = os.path.join(TARGET_OUTPUT_DIR, f"{base_name}_OUT.csv")

# --- STEP 1: EXTRACT & DOWNSCALE FRAMES ---
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

frame_idx = 0
print(f"Extracting frames from {INPUT_VIDEO_PATH}...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    target_height = int(h * (TARGET_WIDTH / w))
    resized = cv2.resize(frame, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(os.path.join(FRAMES_DIR, f"{frame_idx:05d}.jpg"), resized)
    frame_idx += 1

cap.release()
print(f"Extracted {frame_idx} frames at {fps:.2f} FPS.")

# --- STEP 2: INTERACTIVE POINT SELECTION ON FRAME 0 ---
first_frame_path = os.path.join(FRAMES_DIR, "00000.jpg")
first_frame = cv2.imread(first_frame_path)
clicks = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        cv2.circle(first_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Click the Barbell Plate (Press SPACE when done)", first_frame)

cv2.imshow("Click the Barbell Plate (Press SPACE when done)", first_frame)
cv2.setMouseCallback("Click the Barbell Plate (Press SPACE when done)", mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()

if not clicks:
    raise RuntimeError("No points were selected. Exiting.")

# --- STEP 3: INITIALIZE SAM 2 VIDEO PREDICTOR ---
print("Initializing SAM 2...")
predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=DEVICE)
inference_state = predictor.init_state(video_path=FRAMES_DIR)

point_coords = np.array(clicks, dtype=np.float32)
point_labels = np.ones(len(clicks), dtype=np.int32)  # 1 = foreground positive click

# Add prompt on frame 0
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=point_coords,
    labels=point_labels,
)

# 1. Extract 2D binary mask on Frame 0
mask_frame0 = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8)

# 2. Alpha Blending Preview
overlay = first_frame.copy()
overlay[mask_frame0 == 1] = [0, 255, 0]
vis_frame = cv2.addWeighted(first_frame, 0.65, overlay, 0.35, 0)

# 3. Calculate and display centroid on Frame 0
y_idx, x_idx = np.where(mask_frame0)
if len(x_idx) > 0:
    cx0 = int(np.mean(x_idx))
    cy0 = int(np.mean(y_idx))
    cv2.circle(vis_frame, (cx0, cy0), 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(vis_frame, f"Centroid: ({cx0}, {cy0})", (cx0 + 10, cy0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow("Frame 0 Mask & Centroid Preview (Press any key to propagate)", vis_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- STEP 4: PROPAGATE TRACKING & COMPUTE CENTROIDS ---
print("Propagating tracking across frames...")
trajectory = []

with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.float16):
    for f_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
        
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0:
            cx = float(np.mean(x_indices))
            cy = float(np.mean(y_indices))
        else:
            cx, cy = None, None
            
        trajectory.append({
            "frame": f_idx, 
            "time_s": f_idx / fps, 
            "x_px": cx, 
            "y_px": cy,
            "fps": fps
        })

df_traj = pd.DataFrame(trajectory)
df_traj.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Saved trajectory data to {OUTPUT_CSV_PATH}")

# --- STEP 5: RENDER VIDEO WITH MOTION TRAIL ---
print("Rendering tracked video...")
frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
sample = cv2.imread(frame_files[0])
h_out, w_out = sample.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w_out, h_out))

for i, f_path in enumerate(frame_files):
    f_img = cv2.imread(f_path)
    
    # Draw motion trail
    for k in range(1, i + 1):
        p1 = trajectory[k - 1]
        p2 = trajectory[k]
        if p1["x_px"] is not None and p2["x_px"] is not None:
            pt1 = (int(p1["x_px"]), int(p1["y_px"]))
            pt2 = (int(p2["x_px"]), int(p2["y_px"]))
            cv2.line(f_img, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)
            
    # Draw current centroid dot
    curr = trajectory[i]
    if curr["x_px"] is not None:
        center = (int(curr["x_px"]), int(curr["y_px"]))
        cv2.circle(f_img, center, 6, (0, 0, 255), -1, cv2.LINE_AA)
        
    writer.write(f_img)

writer.release()
print(f"Rendered tracked video saved to {OUTPUT_VIDEO_PATH}")