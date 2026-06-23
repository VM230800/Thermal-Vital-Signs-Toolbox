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

# ── NUR 50 Frames ──
frames_raw = sample["frames"][:50]
fps = sample["fps"]

# ── FPS Fix: Falls 0 oder unrealistisch ──
if fps < 1:
    print(f"  FPS aus Timestamps ungültig ({fps:.3f}), nutze Fallback: 30")
    fps = 30.0

recording_id = sample.get("recording_id", "005_rec_0")

print(f"Recording: {recording_id}")
print(f"Frames: {frames_raw.shape}, {frames_raw.dtype}")
print(f"Temperatur-Range: {frames_raw.min():.1f}°C – {frames_raw.max():.1f}°C")
print(f"RAM: ~{frames_raw.nbytes / 1e6:.0f} MB")
print(f"Dauer: {len(frames_raw)/fps:.1f}s bei {fps:.1f} FPS")

# ══════════════════════════════════════════════════════
# WICHTIG: Thermal → Bild konvertieren für YOLO
# NPZ Frames sind Temperaturwerte (z.B. 25.0-37.0°C)
# YOLO braucht Pixelwerte (0-255)
# ══════════════════════════════════════════════════════
print("\nKonvertiere Thermal → Bildformat für YOLO...")

# Normalisieren: min-max auf 0-255
vmin = np.percentile(frames_raw, 1)   # Robust gegen Ausreißer
vmax = np.percentile(frames_raw, 99)
frames_norm = (frames_raw - vmin) / (vmax - vmin)
frames_norm = np.clip(frames_norm, 0, 1)
frames_uint8 = (frames_norm * 255).astype(np.uint8)

# Grayscale → 3-Kanal (YOLO braucht RGB)
frames_yolo = np.stack([frames_uint8, frames_uint8, frames_uint8],
                       axis=-1).astype(np.float32)

print(f"Frames für YOLO: {frames_yolo.shape}, "
      f"Range: {frames_yolo.min():.0f}-{frames_yolo.max():.0f}")

# ── YOLO ──
det = config["detection"]
cropped, keypoints = process_with_yolo(
    frames_yolo,
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
    print("\nFEHLER: YOLO hat kein Gesicht erkannt!")
    print("Mögliche Ursachen:")
    print("  - Thermalbilder zu unscharf")
    print("  - Person nicht im Bild")
    print("  - Anderes Recording versuchen")
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

# ── Ground Truth ──
gt_hr = sample["hr_bpm"]
gt_rr = sample["rr_bpm"]
print(f"\nGround Truth (aus Rohsignal berechnet):")
print(f"  HR: {gt_hr:.1f} BPM")
print(f"  RR: {gt_rr:.1f} BPM")
print(f"{'─' * 50}")


# ── Zusammenfassung ──
print(f"\n{'═' * 50}")
print(f"  ZUSAMMENFASSUNG – {recording_id}")
print(f"{'═' * 50}")
print(f"  {'Methode':<15} {'HR est':>8} {'HR GT':>8} {'Error':>8} {'RR est':>8} {'RR GT':>8}")
print(f"  {'─' * 58}")
print(f"  {'Thermal Mean':<15} {r1['hr_bpm']:>7.1f} {gt_hr:>8.1f} {abs(r1['hr_bpm']-gt_hr):>7.1f} {r1['rr_bpm']:>8.1f} {gt_rr:>8.1f}")
print(f"  {'ICA':<15} {r2['hr_bpm']:>7.1f} {gt_hr:>8.1f} {abs(r2['hr_bpm']-gt_hr):>7.1f} {r2['rr_bpm']:>8.1f} {gt_rr:>8.1f}")
print(f"  {'Garbey':<15} {r3['hr_bpm']:>7.1f} {gt_hr:>8.1f} {abs(r3['hr_bpm']-gt_hr):>7.1f} {r3['rr_bpm']:>8.1f} {gt_rr:>8.1f}")
print(f"\nFERTIG!")
