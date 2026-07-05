# Thermal Vital Signs Toolbox

Contactless estimation of heart rate (HR) and respiration rate (RR) from thermal video using facial region analysis.

Developed as part of a university project, inspired by the architecture of rPPG-Toolbox (Liu et al., NeurIPS 2023), adapted to the thermal domain.

The toolbox loads recordings from multiple datasets, locates 54 anatomical keypoints with a YOLO model, extracts vital-sign estimates using independent methods and evaluates them against ground-truth measurements with standard statistics (MAE, Bland-Altman, ...) 

---

## Quick Start

```text
1. Install dependencies
pip install -r requirements.txt

2. Place the trained YOLO model at the path referenced in
configs/run_config.yaml

3. Configure dataset path
Set the dataset root directories in configs/bp4d.yaml and configs/npz.yaml

4. Run the pipeline
python main.py

The results are written to the directory configured under output.save_dir in run_config.yaml (default: results/).
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
|   |-- hrv_analysis.py          <- Inter-Beat Interval (IBI) extraction & HRV time-domain metrics
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
Set in configs/bp4d.yaml and configs/npz.yaml:
- dataset root path
- subject/task filters
- fallback FPS
- warm-up seconds

Set in configs/run_configs.yaml:
- which dataset should be processed
- which methods are enabled + their parameters.
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

Both datasets are accessed through a common BaseLoader interface, so a new dataset only needs to implement:
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
A already trained YOLO model detects the face and outputs 54 anatomical keypoints per frame:
- Facial contour
- Forehead
- Nose
- Eyes
- Mouth

From these keypoints, preprocessing/roi_extraction.py derives five ROIs used by thermal_mean and ica:
Forehead, left and right cheek, nose, philtrum
methods/garbey.py does not use these ROIs as it defines its own line-segment geometry between two keypoints.
```

---

## Implemented Methods

### Thermal Mean - methods/thermal_mean.py
```text
Mean temperature per ROI per frame
↓
bandpass filter
↓
FFT
↓
dominant frequency
↓
BPM
```

### ICA - methods/ica.py
```text
Extracts multi-ROI temperature signals
↓
detrends (Savitzky-Golay)
↓
bandpass-filters
↓
z-score normalises; then applies FastICA to separate mixed signals into independent components. The component with the strongest concentrated energy in the target frequency band is selected as the HR/RR signal.
```

### Garbey (2007) - methods/garbey.py
```text
Define a keypoint line
↓
sample points along the line
↓
mirror signal
↓
compute its FFT power spectrum
↓
average power spectrum
↓
find dominant frequency
↓
subharmonic correction
```

---

## Signal Processing
```text
Configured under "signal:" in run_config.yaml:
 
  Parameter        | HR                  | RR
  Bandpass low     | 0.7 Hz (~42 BPM)    | 0.1 Hz (~6 BPM)
  Bandpass high    | 4.0 Hz (~240 BPM)   | 0.5 Hz (~30 BPM)
  Filter order     | 4 (Butterworth)     | 4 (Butterworth)
```

---

## Evaluation
```text
evaluation/metrics.py computes, per method/dataset/target:
  - MAE (+ standard error)
  - RMSE
  - MAPE
  - Pearson r
  - n (valid sample count)
 
evaluation/bland_altman.py produces:
  - Scatter plots
  - Classic Bland-Altman difference plots
 
evaluation/results_table.py aggregates everything into:
  - results/summary/results.csv
  - results/summary/results_table.pdf (best MAE per Dataset x
    Target highlighted)
  - results/summary/per_sample_results.csv (raw per-recording
    results, useful for outlier investigation)
```

---

## References
```text
 - Garbey, M., Sun, N., Merla, A., & Pavlidis, I. (2007).
    Contact-Free Measurement of Cardiac Pulse Based on the
    Analysis of Thermal Imagery. IEEE Transactions on Biomedical
    Engineering, 54(8), 1418-1426.
 
  - Gioia, F., Pura, F., Greco, A., Piga, D., Merla, A., &
    Forgione, M. (2025). Contactless Estimation of Respiratory
    Frequency Using 3D-CNN on Thermal Images. IEEE Journal of
    Biomedical and Health Informatics, 29(10), 7387-7396.
 
  - Tarmizi, S. S. A., Suriani, N. S., & Nor Rashid, F. A. (2022).
    A Review of Facial Thermography Assessment for Vital Signs
    Estimation. IEEE Access, 10, 115583-115602.
 
  - Liu, X. et al. (2023). rPPG-Toolbox: Deep Remote PPG Toolbox.
    NeurIPS 2023.
```
