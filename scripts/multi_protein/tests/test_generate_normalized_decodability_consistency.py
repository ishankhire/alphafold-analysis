from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_normalized_decodability_consistency import (  # noqa: E402
    normalize_r2_by_model,
)


class NormalizeR2Tests(unittest.TestCase):
    def test_normalization_is_independent_and_layer_aligned(self) -> None:
        frame = pd.DataFrame(
            {
                "analysis_name": [
                    "model_b",
                    "model_a",
                    "model_a",
                    "model_b",
                    "model_a",
                    "model_b",
                ],
                "layer": [2, 2, 0, 0, 1, 1],
                "raw_mean_per_protein_r2": [0.8, 0.5, 0.1, 0.4, 0.3, 0.6],
            }
        )

        result = normalize_r2_by_model(frame)

        model_a = result[result["analysis_name"].eq("model_a")].sort_values("layer")
        model_b = result[result["analysis_name"].eq("model_b")].sort_values("layer")
        self.assertEqual(model_a["layer"].tolist(), [0, 1, 2])
        self.assertEqual(model_b["layer"].tolist(), [0, 1, 2])
        np.testing.assert_allclose(model_a["normalized_r2"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(model_b["normalized_r2"], [0.0, 0.5, 1.0])
        self.assertEqual(set(model_a["model_r2_min"]), {0.1})
        self.assertEqual(set(model_a["model_r2_max"]), {0.5})
        self.assertEqual(set(model_b["model_r2_min"]), {0.4})
        self.assertEqual(set(model_b["model_r2_max"]), {0.8})

    def test_zero_span_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "analysis_name": ["constant", "constant"],
                "layer": [0, 1],
                "raw_mean_per_protein_r2": [0.7, 0.7],
            }
        )

        with self.assertRaisesRegex(ValueError, "zero aggregated R2 span"):
            normalize_r2_by_model(frame)

    def test_duplicate_layers_are_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "analysis_name": ["duplicate", "duplicate"],
                "layer": [0, 0],
                "raw_mean_per_protein_r2": [0.2, 0.4],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate model-layer"):
            normalize_r2_by_model(frame)


if __name__ == "__main__":
    unittest.main()
