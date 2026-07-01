"""
# ======================================
# FEATURE EXTRACTOR
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- Extracts appearance embeddings using OSNet.
- Converts person crops into feature vectors.
- Used for ReID identity matching.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torchreid


# ======================================
# FEATURE EXTRACTOR
# ======================================

class FeatureExtractor:

    # ======================================
    # INIT
    # ======================================

    def __init__(self):

        # ======================================
        # DEVICE
        # ======================================

        self.device = (

            "cuda"

            if torch.cuda.is_available()

            else "cpu"
        )

        print(
            f"[ReID] Using device: {self.device}"
        )

        # ======================================
        # OSNET MODEL
        # ======================================

        self.extractor = (
            torchreid.utils.FeatureExtractor(

                model_name="osnet_x0_25",

                model_path="",

                device=self.device
            )
        )

        print(
            "[ReID] OSNet initialized"
        )

    # ======================================
    # EXTRACT FEATURES
    # ======================================

    def extract(self, person_crop):

        try:

            # ======================================
            # VALIDATION
            # ======================================

            if person_crop is None:
                return None

            if person_crop.size == 0:
                return None

            # ======================================
            # RESIZE
            # ======================================

            person_crop = cv2.resize(
                person_crop,
                (128, 256)
            )

            # ======================================
            # BGR -> RGB
            # ======================================

            person_crop = cv2.cvtColor(
                person_crop,
                cv2.COLOR_BGR2RGB
            )

            # ======================================
            # EXTRACT FEATURES
            # ======================================

            features = self.extractor(
                person_crop
            )

            # ======================================
            # TORCH TENSOR
            # ======================================

            if torch.is_tensor(features):

                features = (
                    features
                    .detach()
                    .cpu()
                    .numpy()
                )

            # ======================================
            # NUMPY ARRAY
            # ======================================

            if isinstance(
                features,
                np.ndarray
            ):

                features = features.flatten()

                return features

            print(
                f"[ReID ERROR] Unsupported feature type: {type(features)}"
            )

            return None

        except Exception as e:

            print(
                f"[FeatureExtractor ERROR] {e}"
            )

            return None