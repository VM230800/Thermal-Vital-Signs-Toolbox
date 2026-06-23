"""Quick test with only 200 frames."""
import numpy as np
import yaml
from data.bp4d_loader import BP4DDataset
from utils.yolo_processing import process_with_yolo
from preprocessing.roi_extraction import compute_rois
from methods.thermal_mean import ThermalMeanMethod

# Load config
with open("configs/run_config.yaml") as f:
    config = yaml.safe_load(f)

# Load dataset
with open("configs/bp4d.yaml") as f:
    ds_config = yaml.safe_load(f)

dataset = BP4DDataset(
    root_dir=ds_config["root_dir"],
    subjects=["F001"],
    tasks=["T1"],
    fps=25,
)

sample = dataset[0]

# ── NUR 200 Frames nehmen ──
frames = sample["frames"][:200]
print(f"Frames: {frames.shape}, {frames.dtype}")
print(f"RAM: ~{frames.nbytes / 1e6:.0f} MB")

# YOLO
det = config["detection"]
cropped, keypoints = process_with_yolo(
    frames,
    model_path=det["model_path"],
    target_size=tuple(det["target_size"]),
    padding=det.get("padding", 50),
)

# ROIs
rois_per_frame = []
for i in range(len(keypoints)):
    kp = keypoints[i]
    if np.isnan(kp).all():
        rois_per_frame.append(None)
    else:
        rois_per_frame.append(compute_rois(kp))

n_ok = sum(1 for r in rois_per_frame if r is not None)
print(f"YOLO: {n_ok}/{len(keypoints)} frames with keypoints")

# Thermal Mean
method = ThermalMeanMethod(
    config["methods"]["thermal_mean"],
    config["signal"],
)
result = method.estimate(cropped, rois_per_frame, sample["fps"])
print(f"HR: {result['hr_bpm']:.1f} BPM")
print(f"Ground Truth HR: {sample['hr_bpm']:.1f} BPM")
print("FERTIG!")
