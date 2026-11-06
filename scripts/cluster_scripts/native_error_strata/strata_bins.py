"""Fine bins that nest exactly inside the canonical sequence and distance bins."""
import os
from pathlib import Path

import numpy as np

OUTPUT_ENV = "NATIVE_ERROR_STRATA_OUTPUT_ROOT"
SEQ_EDGES = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 200, np.inf])
SEQ_LABELS = ("1", "2", "3", "4", "5-6", "7-9", "10-14", "15-19", "20-29", "30-49", "50-99", "100-199", "200+")
DIST_EDGES = np.array([0, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, np.inf])
DIST_LABELS = ("0-4", "4-5", "5-6", "6-7", "7-8", "8-10", "10-12", "12-15", "15-20", "20-25", "25-30",
               "30-40", "40-50", "50-60", "60-80", "80-100", "100+")
CANONICAL_SEQ = {"1": (0,), "2-4": (1, 2, 3), "5-9": (4, 5), "10-19": (6, 7), "20-49": (8, 9), "50+": (10, 11, 12)}
CANONICAL_DIST = {"0-5": (0, 1), "5-10": (2, 3, 4, 5), "10-20": (6, 7, 8), "20-40": (9, 10, 11),
                  "40+": (12, 13, 14, 15, 16)}
N_S, N_D = len(SEQ_LABELS), len(DIST_LABELS)
SCOPES = ("excluding_7stz_C", "all_552")


def output_root() -> Path:
    value = os.environ.get(OUTPUT_ENV)
    if value:
        return Path(value).expanduser().resolve()
    return Path.home() / "diagnostic_experiments" / "native_error_strata_v1"
