"""Fine bins that nest exactly inside the canonical sequence and distance bins."""
import os
from pathlib import Path

import numpy as np

OUTPUT_ENV = "NATIVE_ERROR_STRATA_OUTPUT_ROOT"
PROBE_ENV = "NATIVE_ERROR_STRATA_PROBE"
COHORT_ENV = "NATIVE_ERROR_STRATA_COHORT"
OLS_PROBE = "shared_ols_base256"
XGBOOST_PROBE = "shared_xgboost_base256_fixed_s4"
PROBE_LABELS = {OLS_PROBE: "", XGBOOST_PROBE: "XGBoost "}
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
    return ols_output_root()


def ols_output_root() -> Path:
    return Path.home() / "diagnostic_experiments" / "native_error_strata_v1"


def probe_name() -> str:
    value = os.environ.get(PROBE_ENV, OLS_PROBE)
    if value not in PROBE_LABELS:
        raise ValueError(f"Unknown {PROBE_ENV} {value!r}; expected one of {sorted(PROBE_LABELS)}")
    return value


def cohort_config() -> tuple[str, tuple[str, str], dict[str, str]]:
    """Return the strata table, its cohort scopes, and the figure note for each scope."""
    label = PROBE_LABELS[probe_name()]
    cohort = os.environ.get(COHORT_ENV, "full")
    if cohort == "full":
        return "native_error_strata.csv", SCOPES, {
            "excluding_7stz_C": f"Real data: native {label}probes, 551 proteins, OOF. Excludes 7stz_C.",
            "all_552": f"Real data: native {label}probes, all 552 proteins, OOF. Includes 7stz_C.",
        }
    if cohort == "cameo":
        return "native_error_strata_cameo.csv", ("excluding_7stz_C", "all_cameo"), {
            "excluding_7stz_C": f"Real data: native {label}probes, 153 CAMEO proteins, OOF. Excludes 7stz_C.",
            "all_cameo": f"Real data: native {label}probes, all 154 CAMEO proteins, OOF. Includes 7stz_C.",
        }
    raise ValueError(f"Unknown {COHORT_ENV} {cohort!r}; expected 'full' or 'cameo'")
