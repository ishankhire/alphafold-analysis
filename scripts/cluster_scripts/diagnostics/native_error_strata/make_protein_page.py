#!/usr/bin/env python3
"""Combine the page template and the packed data into one self-contained HTML file."""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from strata_bins import PROBE_LABELS, output_root, probe_name  # noqa: E402

WORK = output_root()
template = (SCRIPT_DIR / "protein_page_template.html").read_text()
payload = (WORK / "data/protein_page_payload_cameo.json").read_text()
check = json.loads((WORK / "data/protein_strata_cameo_check.json").read_text())
whole = check["checks"]["stored_whole_protein"]
canonical = check["checks"].get("v1_canonical_subgroups") or check["checks"]["published_subgroup_mae"]
if not check["passed"]:
    raise SystemExit("Refusing to build the page: the per-protein checks did not pass")
worst = max(value for key, value in canonical.items() if key.startswith("max_abs_diff_") and key != "max_abs_diff_n")
line = (f"Fine bins reproduce {whole['rows_compared']:,} of {whole['expected_rows']:,} stored protein-layer rows and "
        f"{canonical['rows_compared']:,} of {canonical['expected_rows']:,} published per-protein subgroup rows "
        f"(within {worst:.0e} Å). Values on this page are rounded to 0.05 Å.")
target = WORK / "protein_error_strata_page.html"
label = PROBE_LABELS[probe_name()]
heading = f"{label}protein error strata" if label else "Protein error strata"
target.write_text(template.replace("__PROBE_LABEL__", label).replace("__PROBE_HEADING__", heading)
                  .replace("__CHECK_LINE__", line).replace("__PAYLOAD__", payload))
print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
