"""Quick test with NPZ dataset – 50 frames."""
import numpy as np
import yaml
from data.npz_loader import NPZDataset
from utils.yolo_processing import process_with_yolo
from preprocessing.roi_extraction import compute_rois
from methods.thermal_mean import ThermalMeanMethod
from methods.ica import ICAMethod
from methods.garbey import GarbeyMethod

# ── Load config ──
with open("configs/run_config.yaml") as f:
    config = yaml.safe_load(f)

with open("configs/npz.yaml") as f:
    ds_config = yaml.safe_load(f)

# ── Load dataset ──
dataset = NPZDataset(
    root_dir=ds_config["root_dir"],
    subjects=["005"],
    warmup_seconds=ds_config.get("warmup_seconds", 10),
    fps=ds_config.get("fps", 30),
)

print(f"Anzahl Recordings: {len(dataset)}")

if len(dataset) == 0:
    print("FEHLER: Keine Recordings gefunden!")
    exit()

sample = dataset[0]

# ── NUR 50 Frames (NPZ ist groß: 768x1024) ──
frames = sample["frames"][:50]
fps = sample["fps"]
recording_id = sample.get("recording_id", "005_rec_0")

print(f"Recording: {recording_id}")
print(f"Frames: {frames.shape}, {frames.dtype}")
print(f"RAM: ~{frames.nbytes / 1e6:.0f} MB")
print(f"Dauer: {len(frames)/fps:.1f}s bei {fps:.1f} FPS")

# ── NPZ Frames sind (N, H, W) → YOLO braucht (N, H, W, 3) ──
if frames.ndim == 3:
    print("Konvertiere Grayscale → 3-Kanal...")
    frames = np.stack([frames, frames, frames], axis=-1)

print(f"Frames nach Konvertierung: {frames.shape}")

# ── YOLO ──
det = config["detection"]
cropped, keypoints = process_with_yolo(
    frames,
    model_path=det["model_path"],
    target_size=tuple(det["target_size"]),
    padding=det.get("padding", 50),
)

# ── ROIs ──
rois_per_frame = []
for i in range(len(keypoints)):
    kp = keypoints[i]
    if np.isnan(kp).all():
        rois_per_frame.append(None)
    else:
        rois_per_frame.append(compute_rois(kp))

n_ok = sum(1 for r in rois_per_frame if r is not None)
print(f"YOLO: {n_ok}/{len(keypoints)} frames with keypoints")

if n_ok == 0:
    print("FEHLER: YOLO hat kein Gesicht erkannt!")
    exit()

# ── Methode 1: Thermal Mean ──
print("\n1. Thermal Mean:")
tm = ThermalMeanMethod(
    config["methods"]["thermal_mean"],
    config["signal"],
)
r1 = tm.estimate(cropped, rois_per_frame, fps)
print(f"   HR: {r1['hr_bpm']:.1f} BPM")

# ── Methode 2: ICA ──
print("\n2. ICA:")
ica = ICAMethod(
    config["methods"]["ica"],
    config["signal"],
)
r2 = ica.estimate(cropped, rois_per_frame, fps)
print(f"   HR: {r2['hr_bpm']:.1f} BPM")
print(f"   RR: {r2['rr_bpm']:.1f} BPM")

# ── Methode 3: Garbey ──
print("\n3. Garbey:")
garbey = GarbeyMethod(
    config["methods"]["garbey"],
    config["signal"],
)
r3 = garbey.estimate(cropped, keypoints, fps)
print(f"   HR: {r3['hr_bpm']:.1f} BPM")
print(f"   RR: {r3['rr_bpm']:.1f} BPM")

# ── Zusammenfassung ──
print(f"\n{'═' * 50}")
print(f"  ZUSAMMENFASSUNG – {recording_id}")
print(f"{'═' * 50}")
print(f"  {'Methode':<15} {'HR':>8} {'RR':>8}")
print(f"  {'─' * 35}")
print(f"  {'Thermal Mean':<15} {r1['hr_bpm']:>7.1f} {r1['rr_bpm']:>8.1f}")
print(f"  {'ICA':<15} {r2['hr_bpm']:>7.1f} {r2['rr_bpm']:>8.1f}")
print(f"  {'Garbey':<15} {r3['hr_bpm']:>7.1f} {r3['rr_bpm']:>8.1f}")
print(f"\n  Hinweis: NPZ hat kein Ground Truth BPM.")
print(f"  HR sollte ca. 60-100 BPM sein.")
print(f"  RR sollte ca. 12-20 BPM sein.")
print(f"\nFERTIG!")
