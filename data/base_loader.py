"""
data/base_loader.py
===================
Abstract base class for all dataset loaders.

Handles all SHARED logic:
    - __len__, __getitem__
    - Warmup removal
    - iter_frames (RAM-friendly streaming)
    - get_metadata (no frames loaded)
    - Cache system (save/load preprocessed data)
    - Standardised return dict

Child classes only implement DATASET-SPECIFIC logic:
    - _discover_samples()    → find recordings on disk
    - _load_frames()         → load all frames from one recording
    - _load_single_frame()   → load one frame by index
    - _get_total_frames()    → number of frames in recording
    - _get_fps()             → framerate
    - _load_ground_truth()   → HR/RR ground truth
    - _get_subject()         → subject ID
    - _get_task()            → task/recording ID
"""



import os
import numpy as np
from abc import ABC, abstractmethod


class BaseLoader(ABC):

    def __init__(self, root_dir, subjects=None,
                 warmup_seconds=0, fps=30.0,
                 cache_dir="cache", force_preprocess=False):
        """
        Args:
            root_dir:          Path to dataset root folder.
            subjects:          List of subject IDs, or None for all.
            warmup_seconds:    Seconds to skip at recording start.
            fps:               Fallback FPS if metadata unavailable.
            cache_dir:         Folder for cached preprocessed data.
            force_preprocess:  If True, ignore existing cache.
        """
        self.root_dir = root_dir
        self.warmup_seconds = warmup_seconds
        self.default_fps = fps
        self.cache_dir = cache_dir
        self.force_preprocess = force_preprocess

        os.makedirs(self.cache_dir, exist_ok=True)

        # Child class fills this via _discover_samples()
        self.samples = self._discover_samples(subjects)

        print(f"{self.__class__.__name__}: "
              f"{len(self.samples)} recordings found")

    # ─────────────────────────────────────────────────────
    # SHARED: Standard Python methods
    # ─────────────────────────────────────────────────────

    def __len__(self):
        """Total number of recordings."""
        return len(self.samples)

    def __repr__(self):
        return (f"{self.__class__.__name__}: "
                f"{len(self.samples)} recordings")

    # ─────────────────────────────────────────────────────
    # SHARED: __getitem__ (loads everything)
    # ─────────────────────────────────────────────────────

    def __getitem__(self, idx):
        """
        Load one recording with ALL frames.

        Returns standardised dict:
            {
                "frames":       np.ndarray (N, H, W, ...) float32,
                "fps":          float,
                "subject":      str,
                "task":         str,
                "recording_id": str,
                "hr_bpm":       float,
                "rr_bpm":       float,
                ... (dataset-specific extra fields)
            }
        """
        sample_info = self.samples[idx]

        # ── Dataset-specific loading ──
        fps = self._get_fps(sample_info)
        frames = self._load_frames(sample_info)
        gt = self._load_ground_truth(sample_info, fps)

        subject = self._get_subject(sample_info)
        task = self._get_task(sample_info)

        # ── Shared: Remove warmup ──
        warmup_frames = int(self.warmup_seconds * fps)
        if warmup_frames > 0 and warmup_frames < len(frames):
            frames = frames[warmup_frames:]

        # ── Build standard result dict ──
        result = {
            "frames":       frames,
            "fps":          fps,
            "subject":      subject,
            "task":         task,
            "recording_id": f"{subject}_{task}",
            "hr_bpm":       gt.get("hr_bpm", float("nan")),
            "rr_bpm":       gt.get("rr_bpm", float("nan")),
        }

        # Add any dataset-specific extra fields
        for key, value in gt.items():
            if key not in result:
                result[key] = value

        return result

    # ─────────────────────────────────────────────────────
    # SHARED: iter_frames (RAM-friendly streaming)
    # ─────────────────────────────────────────────────────

    def iter_frames(self, idx, max_frames=None):
        """
        Yield frames one-by-one. Only ONE frame in RAM.

        Handles warmup removal automatically.

        Args:
            idx:        Sample index
            max_frames: Max frames to yield (None = all)

        Yields:
            np.ndarray – one frame at a time
        """
        sample_info = self.samples[idx]
        fps = self._get_fps(sample_info)
        warmup_frames = int(self.warmup_seconds * fps)
        total = self._get_total_frames(sample_info)

        start = min(warmup_frames, total)

        count = 0
        for i in range(start, total):
            if max_frames is not None and count >= max_frames:
                break

            frame = self._load_single_frame(sample_info, i)
            yield frame
            count += 1

    # ─────────────────────────────────────────────────────
    # SHARED: get_metadata (no frames loaded)
    # ─────────────────────────────────────────────────────

    def get_metadata(self, idx):
        """
        Get recording metadata WITHOUT loading any frames.

        Returns:
            dict with fps, subject, task, recording_id,
                 hr_bpm, rr_bpm, total_frames
        """
        sample_info = self.samples[idx]
        fps = self._get_fps(sample_info)
        gt = self._load_ground_truth(sample_info, fps)

        subject = self._get_subject(sample_info)
        task = self._get_task(sample_info)

        warmup_frames = int(self.warmup_seconds * fps)
        total = self._get_total_frames(sample_info)
        usable_frames = max(0, total - warmup_frames)

        result = {
            "fps":          fps,
            "subject":      subject,
            "task":         task,
            "recording_id": f"{subject}_{task}",
            "hr_bpm":       gt.get("hr_bpm", float("nan")),
            "rr_bpm":       gt.get("rr_bpm", float("nan")),
            "total_frames": usable_frames,
        }

        # Add extra fields from ground truth
        for key, value in gt.items():
            if key not in result:
                result[key] = value

        return result

    # ─────────────────────────────────────────────────────
    # SHARED: Cache system
    # ─────────────────────────────────────────────────────

    def _get_cache_path(self, recording_id):
        """Full path for cached file."""
        return os.path.join(self.cache_dir,
                            recording_id + ".npz")

    def is_cached(self, recording_id):
        """Check if processed data exists in cache."""
        return os.path.exists(
            self._get_cache_path(recording_id))

    def save_to_cache(self, recording_id, data_dict):
        """Save processed data to cache."""
        path = self._get_cache_path(recording_id)
        np.savez_compressed(path, **data_dict)

    def load_from_cache(self, recording_id):
        """Load previously cached data."""
        path = self._get_cache_path(recording_id)
        return dict(np.load(path, allow_pickle=True))

    # ─────────────────────────────────────────────────────
    # ABSTRACT: Child classes MUST implement these
    # ─────────────────────────────────────────────────────

    @abstractmethod
    def _discover_samples(self, subjects):
        """
        Find all recordings on disk.

        Returns:
            list of sample_info (tuples, dicts, etc.)
        """
        pass

    @abstractmethod
    def _load_frames(self, sample_info):
        """
        Load ALL frames from one recording.

        Returns:
            np.ndarray (N, H, W) or (N, H, W, 3), float32
        """
        pass

    @abstractmethod
    def _load_single_frame(self, sample_info, frame_idx):
        """
        Load ONE frame by index (for streaming).

        Returns:
            np.ndarray (H, W) or (H, W, 3), float32
        """
        pass

    @abstractmethod
    def _get_total_frames(self, sample_info):
        """Return total number of frames in recording."""
        pass

    @abstractmethod
    def _get_fps(self, sample_info):
        """Return framerate of recording."""
        pass

    @abstractmethod
    def _load_ground_truth(self, sample_info, fps):
        """
        Load ground truth.

        Returns:
            dict with at least "hr_bpm" and "rr_bpm"
                  (can include extra fields like
                   "pulse_rate", "resp_rate", etc.)
        """
        pass

    @abstractmethod
    def _get_subject(self, sample_info):
        """Return subject ID as string."""
        pass

    @abstractmethod
    def _get_task(self, sample_info):
        """Return task/recording ID as string."""
        pass
