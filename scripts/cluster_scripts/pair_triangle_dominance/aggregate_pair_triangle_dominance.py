#!/usr/bin/env python3
from pathlib import Path
import csv, os, statistics
def main():
 root=Path(os.environ.get("TRIANGLE_OUTPUT_DIR",Path(os.environ.get("CSV_ROOT","delta_files"))/"pair_representation_triangle_dominance")); sh=root/"layer_shards"; rows=[]; statuses=[]
 for layer in range(48):
  p=sh/f"per_protein_metrics_layer_{layer:02d}.csv"; s=sh/f"layer_status_{layer:02d}.csv"
  if not p.exists() or not s.exists(): raise RuntimeError(f"missing layer shard {layer}")
  with p.open() as f: rows.extend(csv.DictReader(f))
  with s.open() as f: statuses.extend(csv.DictReader(f))
 fields=list(rows[0]) if rows else []
 with (root/"per_protein_metrics.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 layerrows=[]
 for layer in range(48):
  for variant in {r["representation_variant"] for r in rows}:
   for metric in {r["metric"] for r in rows}:
    x=[float(r["mean"]) for r in rows if int(r["layer"])==layer and r["representation_variant"]==variant and r["metric"]==metric and r["status"]=="complete"]
    if x: layerrows.append(dict(layer=layer,representation_variant=variant,metric=metric,protein_equal_mean=statistics.mean(x),protein_std=statistics.pstdev(x),n_proteins_finite=len(x),n_proteins_attempted=154))
 with (root/"layer_summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(layerrows[0])); w.writeheader(); w.writerows(layerrows)
 with (root/"layer_status.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(statuses[0])); w.writeheader(); w.writerows(statuses)
if __name__=="__main__": main()
