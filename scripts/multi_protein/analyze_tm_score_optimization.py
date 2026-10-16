#!/usr/bin/env python3
"""Compare saved and optimized TM-scores over the complete frozen OOF cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = ROOT / "delta_files/regression_5fold_v1/tm_score_optimized_v1"
OUTPUT = ROOT / "delta_files/regression_5fold_v1/analysis/tm_score_optimization_comparison_v1"
MODELS = {
    "shared_ols_base256": ("Shared OLS", "#377eb8"),
    "per_protein_ols_base256": ("Per-protein OLS", "#e68613"),
    "sequence_gated_ols_base256": ("Sequence-gated OLS", "#41956b"),
    "shared_xgboost_base256_fixed_s4": ("Shared XGBoost", "#9b61a8"),
}
TOLERANCE = 1e-6
SUMMARY_STATISTICS = ("mean", "std", "median", "q25", "q75")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def index_rows(rows, fields):
    indexed = {tuple(row[field] for field in fields): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate geometry identities: {fields}")
    return indexed


def csv_bytes(path, columns, rows):
    # Preserve the canonical newline convention as well as column/row order.
    stream = io.StringIO(newline="")
    ending = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator=ending)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def compare_columns(family, table, columns, pairs, *, rounding=False):
    results = []
    for column in columns:
        mismatches, maximum = 0, 0.0
        for old, new in pairs:
            left, right = old[column], new[column]
            equal = left == right
            try:
                a, b = float(left), float(right)
                if np.isfinite(a) and np.isfinite(b):
                    maximum = max(maximum, abs(a - b))
                    equal = (bool(np.isclose(a, b, rtol=1e-12, atol=1e-12))
                             if rounding else a == b)
            except ValueError:
                pass
            mismatches += not equal
        results.append(dict(analysis_name=family, table=table, column=column,
                            comparison_count=len(pairs), mismatch_count=int(mismatches),
                            max_abs_difference=maximum, passed=mismatches == 0))
    return results


def prepare_geometry_refresh(regression_root, optimized_root, ledger, manifest, proteins, layers):
    """Validate every family before returning any canonical replacement bytes."""
    checks, updates, provenance = [], {}, {}
    manifest_hash = sha256(optimized_root / "run_manifest.json")
    for family in MODELS:
        directory = regression_root / family
        family_manifest_path = directory / "run_manifest.json"
        family_manifest = json.loads(family_manifest_path.read_text())
        prior = family_manifest.get("tm_score_refresh", {})
        outputs = {}
        for name in ("geometry_metrics.csv", "geometry_summary.csv", "run_manifest.json"):
            relative = f"{family}/{name}"
            if sha256(optimized_root / relative) != manifest["output_sha256"][relative]:
                raise ValueError(f"Replacement checksum mismatch: {relative}")
        scoring_manifest = json.loads((optimized_root / family / "run_manifest.json").read_text())
        if (scoring_manifest["run_status"] != "complete"
                or scoring_manifest["identity_digest"] != manifest["identity_digest"]
                or scoring_manifest["source_geometry_sha256"] !=
                manifest["identity"]["source_geometry_sha256"][family]):
            raise ValueError(f"Replacement provenance mismatch: {family}")
        original_hash = manifest["identity"]["source_geometry_sha256"][family]
        metrics_path = directory / "geometry_metrics.csv"
        if sha256(metrics_path) != original_hash:
            if (prior.get("source_manifest_sha256") != manifest_hash
                    or sha256(metrics_path) != prior.get("output_sha256", {}).get("geometry_metrics.csv")):
                raise ValueError(f"Canonical geometry differs from scoring source: {family}")
        for name in ("geometry_metrics.csv", "geometry_summary.csv"):
            path = directory / name
            columns, original = read_csv(path)
            incoming_columns, incoming = read_csv(optimized_root / family / name)
            if columns != incoming_columns:
                raise ValueError(f"Geometry schema mismatch: {family}/{name}")
            is_metrics = name == "geometry_metrics.csv"
            keys = ("analysis_name", "layer", "protein_id") if is_metrics else (
                "analysis_name", "layer", "geometry_metric")
            old_index, new_index = index_rows(original, keys), index_rows(incoming, keys)
            if set(old_index) != set(new_index):
                raise ValueError(f"Geometry coverage mismatch: {family}/{name}")
            if is_metrics:
                expected = {(family, str(layer), pid) for layer in layers for pid in proteins}
                if set(old_index) != expected:
                    raise ValueError(f"Wrong frozen cohort/layer coverage: {family}")
                pairs = [(row, new_index[key]) for key, row in old_index.items()]
                checks.extend(compare_columns(family, name, [c for c in columns if c != "tm_score"], pairs))
                for key, row in old_index.items():
                    score = ledger[family, key[2], int(key[1])]
                    replacement = new_index[key]
                    if (row["geometry_status"] != "complete"
                            or row["evaluation_scope"] != "oof_all_pairs"
                            or float(replacement["tm_score"]) != float(score["tm_score"])
                            or float(row["tm_score"]) not in (
                                float(score["tm_score_saved"]), float(score["tm_score"]))
                            or any(row[c] != score[c] for c in score.keys() & row.keys()
                                   if c != "tm_score")):
                        raise ValueError(f"Geometry/ledger identity or score mismatch: {key}")
                refreshed = [{**row, "tm_score": new_index[key]["tm_score"]}
                             for key, row in old_index.items()]
            else:
                expected = {(family, str(layer), metric) for layer in layers
                            for metric in {row["geometry_metric"] for row in original}}
                if set(old_index) != expected or not any(k[2] == "tm_score" for k in expected):
                    raise ValueError(f"Summary layer/metric coverage mismatch: {family}")
                unchanged = [(row, new_index[key]) for key, row in old_index.items() if key[2] != "tm_score"]
                checks.extend(compare_columns(family, name, columns, unchanged, rounding=True))
                tm_pairs = [(row, new_index[key]) for key, row in old_index.items() if key[2] == "tm_score"]
                checks.extend(compare_columns(family, name + ":tm_identity",
                                              [c for c in columns if c not in SUMMARY_STATISTICS], tm_pairs))
                for key, row in new_index.items():
                    if key[2] == "tm_score":
                        values = np.array([float(ledger[family, pid, int(key[1])]["tm_score"])
                                           for pid in proteins])
                        expected_stats = (values.mean(), values.std(), np.median(values),
                                          np.quantile(values, .25), np.quantile(values, .75))
                        if not np.allclose([float(row[c]) for c in SUMMARY_STATISTICS], expected_stats,
                                           rtol=1e-12, atol=1e-12):
                            raise ValueError(f"TM summary differs from score ledger: {key}")
                refreshed = [{**row, **{c: new_index[key][c] for c in SUMMARY_STATISTICS}}
                             if key[2] == "tm_score" else row for key, row in old_index.items()]
            contents = csv_bytes(path, columns, refreshed)
            outputs[name] = hashlib.sha256(contents).hexdigest()
            updates[path] = contents
        provenance[family] = dict(
            source_manifest=str((optimized_root / "run_manifest.json").relative_to(regression_root)),
            source_manifest_sha256=manifest_hash, source_geometry_sha256=original_hash,
            tm_score_method=manifest["tm_score_method"], validation=manifest["validation"],
            output_sha256=outputs,
        )
        if prior != provenance[family]:
            family_manifest["tm_score_refresh"] = provenance[family]
            updates[family_manifest_path] = (json.dumps(family_manifest, indent=2) + "\n").encode()
    if any(not row["passed"] for row in checks):
        failed = [(r["analysis_name"], r["table"], r["column"]) for r in checks if not r["passed"]]
        raise ValueError(f"Non-TM geometry changed: {failed}")
    return checks, updates, provenance


def apply_geometry_refresh(updates):
    """Write only validated changes; repeated application leaves files untouched."""
    for path, contents in updates.items():
        if path.read_bytes() != contents:
            temporary = path.with_name(path.name + ".tm-refresh.tmp")
            temporary.write_bytes(contents)
            temporary.replace(path)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def statistics(old, new):
    gain = new - old
    return {
        "tm_score_saved_mean": float(old.mean()),
        "tm_score_optimized_mean": float(new.mean()),
        "mean_gain": float(gain.mean()),
        "median_gain": float(np.median(gain)),
        "q90_gain": float(np.quantile(gain, 0.9)),
        "fraction_improved": float(np.mean(gain > TOLERANCE)),
        "fraction_decreased": float(np.mean(gain < -TOLERANCE)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-geometry", action="store_true",
                        help="After validating all families, replace canonical TM-score cells in place")
    args = parser.parse_args()
    manifest = json.loads((INPUT / "run_manifest.json").read_text())
    checksum = hashlib.sha256((INPUT / "tm_scores.csv").read_bytes()).hexdigest()
    if manifest["run_status"] != "complete" or checksum != manifest["output_sha256"]["tm_scores.csv"]:
        raise ValueError("Score ledger is incomplete or differs from its completion manifest")
    with (INPUT / "tm_scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    _, cohort = read_csv(INPUT.parent / "metadata/regression_outer_folds_v1.csv")
    proteins = sorted(row["protein_id"] for row in cohort)
    if len(set(proteins)) != len(proteins):
        raise ValueError("Duplicate frozen cohort IDs")
    expected = {(model, pid, layer) for model in MODELS for pid in proteins for layer in range(48)}
    indexed = {(row["analysis_name"], row["protein_id"], int(row["layer"])): row for row in rows}
    if len(proteins) != 552 or len(rows) != 105984 or len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Expected 552 proteins, four probes, and all 48 layers without duplicate rows")
    if any(row["status"] != "complete" or row["evaluation_scope"] != "oof_all_pairs" for row in rows):
        raise ValueError("Incomplete scores or inconsistent evaluation scope")

    if (manifest["n_scores"] != 105984 or manifest["n_pdbs"] != 105984
            or set(manifest["layer_manifest_sha256"]) != {str(i) for i in range(48)}):
        raise ValueError("Incomplete optimized run manifest")
    checks, updates, provenance = prepare_geometry_refresh(
        INPUT.parent, INPUT, indexed, manifest, proteins, range(48))
    model_rows, layer_rows, threshold_rows = [], [], []
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for axis, (model, (label, color)) in zip(axes.flat, MODELS.items()):
        data = [[indexed[model, pid, layer] for layer in range(48)] for pid in proteins]
        old = np.array([[float(row["tm_score_saved"]) for row in protein] for protein in data])
        new = np.array([[float(row["tm_score"]) for row in protein] for protein in data])
        baseline = np.array([[float(row["tm_score_kabsch"]) for row in protein] for protein in data])
        if (not np.all(np.isfinite([old, new, baseline])) or np.any(new < baseline - 1e-12)
                or np.any(new < 0) or np.any(new > 1) or np.any(old < 0) or np.any(old > 1)):
            raise ValueError(f"Invalid score range or alignment floor: {model}")
        summary = dict(analysis_name=model, n_proteins=552, n_layers=48, n_scores=old.size,
                       **statistics(old, new),
                       layer_0_mean_gain=float((new[:, 0] - old[:, 0]).mean()),
                       layer_47_saved_mean=float(old[:, 47].mean()),
                       layer_47_optimized_mean=float(new[:, 47].mean()),
                       layer_47_mean_gain=float((new[:, 47] - old[:, 47]).mean()),
                       mean_kabsch_saved_difference=float((baseline - old).mean()),
                       max_abs_kabsch_saved_difference=float(np.max(np.abs(baseline - old))))
        model_rows.append(summary)
        for threshold in (.4, .5, .7):
            old_crossings = np.flatnonzero(old.mean(axis=0) > threshold)
            new_crossings = np.flatnonzero(new.mean(axis=0) > threshold)
            threshold_rows.append(dict(analysis_name=model, threshold=threshold, n_proteins=len(proteins),
                                       saved_first_layer=int(old_crossings[0]) if old_crossings.size else "",
                                       optimized_first_layer=int(new_crossings[0]) if new_crossings.size else ""))
        for layer in range(48):
            layer_rows.append(dict(analysis_name=model, layer=layer, n_proteins=552,
                                   **statistics(old[:, layer], new[:, layer])))
        axis.plot(range(48), old.mean(axis=0), color="#777777", linestyle="--", label="Saved (Kabsch)")
        axis.plot(range(48), new.mean(axis=0), color=color, linewidth=2, label="Optimized TM-score")
        axis.set_title(label)
        axis.set_xlim(0, 47)
        axis.set_ylim(0.35, 1)
        axis.set_xticks([0, 10, 20, 30, 40, 47])
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.supxlabel("Evoformer layer")
    figure.supylabel("Mean TM-score across 552 proteins")
    figure.suptitle("TM-score optimization: same predicted distances, improved superposition")
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.apply_geometry:
        apply_geometry_refresh(updates)
    write_csv(OUTPUT / "geometry_validation.csv", checks)
    write_csv(OUTPUT / "threshold_crossings.csv", threshold_rows)
    write_csv(OUTPUT / "model_summary.csv", model_rows)
    write_csv(OUTPUT / "layer_summary.csv", layer_rows)
    figure.savefig(OUTPUT / "tm_score_comparison.png", dpi=180)
    figure.savefig(OUTPUT / "tm_score_comparison.pdf")
    plt.close(figure)
    summary = {
        "source_ledger": str((INPUT / "tm_scores.csv").relative_to(ROOT)),
        "source_sha256": checksum,
        "source_validation": manifest.get("validation"),
        "geometry_refresh": {
            "applied": args.apply_geometry,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "all_non_tm_checks_passed": all(row["passed"] for row in checks),
            "non_tm_column_checks": len(checks),
            "families": provenance,
        },
        "aggregation": "equal proteins within each layer; equal layers for the overall descriptive mean",
        "improvement_tolerance": TOLERANCE,
        "notes": ["Same regression predictions; the alignment objective changed.",
                  "All-layer means describe trajectories, not a designated primary layer.",
                  "No confidence intervals or hypothesis tests; improvement is favored by the scoring procedure."],
        "models": model_rows,
    }
    (OUTPUT / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
