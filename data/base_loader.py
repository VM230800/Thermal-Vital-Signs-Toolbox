"""
 BaseLoader for all datasets.
 Handles:
    - sample discovery
    - caching / preprocessing
    - loading processed files
"""


import os
import numpy as np


class BaseLoader:

    def __init__(self, root_dir, cache_dir="cache", force_preprocess=False):
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.force_preprocess = force_preprocess

        # Create cache folder if it does not exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # List of all dataset samples (filled by child class)
        self.samples = self._collect_samples()

    # ─────────────────────────────────────────────────────────────────
    # Methods that must be implemented in child class
    # ─────────────────────────────────────────────────────────────────

    def _collect_samples(self):
        raise NotImplementedError("Implement this method in the child class")

    def _load_raw(self, idx):
        raise NotImplementedError("Implement this method in the child class")

    def _preprocess(self, raw_data):
        raise NotImplementedError("Implement this method in the child class")

    def _get_sample_id(self, idx):
        raise NotImplementedError("Implement this method in the child class")

    # ─────────────────────────────────────────────────────────────────
    # Cache helper functions
    # ─────────────────────────────────────────────────────────────────

    def _get_cache_path(self, sample_id):
        """Returns full path of cached file for one sample"""
        return os.path.join(self.cache_dir, sample_id + ".npz")

    def _is_cached(self, sample_id):
        """Check if processed file already exists"""
        return os.path.exists(self._get_cache_path(sample_id))

    def _save_cache(self, sample_id, data):
        """Save processed data as compressed npz file"""
        np.savez_compressed(self._get_cache_path(sample_id), **data)

    def _load_cache(self, sample_id):
        """Load cached processed data"""
        return np.load(self._get_cache_path(sample_id), allow_pickle=True)

    # ─────────────────────────────────────────────────────────────────
    # Main logic
    # ─────────────────────────────────────────────────────────────────

    def _get_processed_sample(self, idx):
        """
        Returns a processed sample.
        If cache exists -> load it.
        If not -> run preprocessing and save it.
        """

        sample_id = self._get_sample_id(idx)

        # If cache does not exist or we force recomputation
        if (not self._is_cached(sample_id)) or self.force_preprocess:

            # Step 1: load raw data
            raw_data = self._load_raw(idx)

            # Step 2: run preprocessing (YOLO, ROI extraction, etc.)
            processed_data = self._preprocess(raw_data)

            # Step 3: save to cache
            self._save_cache(sample_id, processed_data)

        # Always return cached version
        return self._load_cache(sample_id)

    # ─────────────────────────────────────────────────────────────────
    # Python standard methods
    # ─────────────────────────────────────────────────────────────────

    def __len__(self):
        """Return number of samples in dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get one processed sample"""
        return self._get_processed_sample(idx)