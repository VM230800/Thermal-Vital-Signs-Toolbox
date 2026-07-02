# Thermal Vital Signs Toolbox

Contactless estimation of heart rate (HR) and respiration rate (RR)
from thermal video using facial region analysis.

Developed as part of a university project, inspired by the architecture 
of rPPG-Toolbox (Liu et al., NeurIPS 2023), adapted to the thermal 
domain.

The toolbox loads recordings from multiple datasets, locates 54 anatomical
keypoints with a YOLO model, extracts vital-sign estimates using independent
methods and evaluates them against ground-truth measurements with standard 
statistics (MAE, Bland-Altman, ...) 

---

## Quick Start

```bash
1. Install dependencies
pip install -r requirements.txt

2. Place YOLO model
  Copy YOLOv11_TFL_252.pt into models/

3. Configure dataset path
 Edit configs/bp4d.yaml or configs/npz.yaml

4. Run
python main.py
```

---

## Repository Structure

```text
Thermal Vital Signs Toolbox
|
|-- main.py                      <- Pipeline orchestration
|
|-- configs/
|   |-- bp4d.yaml                <- BP4D+ dataset paths
|   |-- npz.yaml                 <- NPZ dataset paths
|   +-- run_config.yaml          <- Pipeline settings
|
|-- data/
|   |-- base_loader.py           <- Abstract BaseLoader with shared logic
|   |-- bp4d_loader.py           <- BP4D+ loader (.wmv + .txt)
|   +-- npz_loader.py            <- NPZ loader (.npz files)
|
|-- evaluation/
|   |-- bland_altman.py          <- Bland-Altman and scatter plots
|   |-- metrics.py               <- MAE, RMSE, Pearson
|   +-- results_table.py         <- Comparison table (CSV + PDF)
|
|-- methods/
|   |-- garbey.py                <- line-FFT method (Garbey et al. 2007)
|   |-- ica.py                   <- ICA-based source separation
|   +-- thermal_mean.py          <- Baseline: mean ROI temperature
|
|-- models/
|   |-- YOLOv11_TFL_252.pt
|   +-- run_yolo.py
|
|-- preprocessing/
|   |-- peak_extraction.py       <- Bandpass filtering + BPM estimation
|   |-- roi_extraction.py        <- Keypoints to ROI boxes
|   |-- signal_extraction.py     <- Frames + ROIs to temperature signal
|
|-- results/
|
+-- utils/
    |-- visualization.py         <- ROI overlays
    |-- yolo_keypoints.py        <- keypoint names and regions
    +-- yolo_processing.py       <- YOLO batch processing
```

--- 

## Configuration
```text
...
```

---

## Supported Datasets
```text
1. BP4D+
   Format:.wmv videos, colour-mapped thermal
   Frame rate: 25 fps
   Ground truth: Pulse Rate_BPM.txt, Respiration Rate_BPM.txt
2. NPZ
   Format: .npz, raw temperature in degrees C
   Frame rate: 30 fps
   Ground truth: Raw pulse/respiration waveforms

Both datasets are accessed through a common BaseLoader interface, so a new
dataset only needs to implement:
_discover_samples()
_load_frames()
_load_single_frame()
_get_total_frames()
_get_fps()
_load_ground_truth()
_get_subject()
_get_task()
```

--- 

## Face Detection
```text
A already trained YOLO model detects the face and outputs 54 anatomical
keypoints per frame:
- Facial contour
- Forehead
- Nose
- Eyes
- Mouth

From these keypoints, preprocessing/roi_extraction.py derives five
ROIs used by thermal_mean and ica:
Forehead, left and right cheek, nose, philtrum
methods/garbey.py does not use these ROIs as it defines its own
line-segment geometry between two keypoints.
```

---

## Implemented Methods
```text
...
```

---

## Signal Processing
```text
...
```

---

## Evaluation
```text
...
```

---

## Open Items
```text
...
```

---

## References
```text
...
```
