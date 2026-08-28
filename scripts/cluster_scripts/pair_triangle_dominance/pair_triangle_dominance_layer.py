#!/usr/bin/env python3
from pathlib import Path
import csv, os, sys, time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent)); from pair_triangle_dominance_common import *
def main():
 layer=int(os.environ.get("SLURM_ARRAY_TASK_ID",os.environ.get("LAYER","0"))); root=Path(os.environ.get("TRIANGLE_OUTPUT_DIR",Path(os.environ.get("CSV_ROOT","delta_files"))/"pair_representation_triangle_dominance")); meta=root/"metadata"; shard=root/"layer_shards"; shard.mkdir(parents=True,exist_ok=True)
 sample=np.load(meta/"triangle_samples.npz",allow_pickle=False); ids=sample["protein_ids"].astype(str); off=sample["offsets"]; rows=[]; channel_mean=np.full((len(ids),4,128,4),np.nan); channel_pos=np.full((len(ids),4,128),np.nan); channel_zero=np.full((len(ids),4,128),np.nan); status=[]
 for pidx,pid in enumerate(ids):
  t=np.column_stack((sample["tri_i"][off[pidx]:off[pidx+1]],sample["tri_j"][off[pidx]:off[pidx+1]],sample["tri_k"][off[pidx]:off[pidx+1]])).astype(int); path=Path(os.environ.get("PAIR_BLOCK_ROOT","."))/pid/f"{pid}_pair_block_{layer}.npy"; source="cameo"; z=None; reason=""
  try:
   z=np.load(path)
   if z.shape!=(int(z.shape[0]),int(z.shape[1]),128) or z.shape[0]!=z.shape[1] or not np.isfinite(z).all(): raise ValueError("invalid tensor")
   if not validate_samples(t,z.shape[0]): raise ValueError("invalid samples")
   measurements=[]; zeros=[]
   for sides in transform_sides(z,t): measurements.append(channel_measurements(sides)[0]); zeros.append(channel_measurements(sides)[1])
   for vi,mset in enumerate(measurements):
    tri=np.mean(mset,axis=2); pos=np.mean(mset[0]>0,axis=1); zero=zeros[vi].mean(axis=1)
    for mi,name in enumerate(PRIMARY_METRICS):
     vals=summarize(tri[:,mi]); rows.append(dict(analysis_name="pair_representation_triangle_dominance",schema_version="1.0",layer=layer,protein_id=pid,source=source,representation_variant=VARIANTS[vi],metric=name,n_residues=z.shape[0],n_population_triangles=int(__import__('math').comb(z.shape[0],3)),n_sampled_triangles=len(t),sample_sha256=sha256_bytes(t.tobytes()),mean=vals[0],std=vals[1],median=vals[2],p90=vals[3],p99=vals[4],n_finite_triangles=vals[5],zero_scale_channel_fraction=float(np.mean(zeros[vi])),status="complete",failure_reason=""))
     channel_mean[pidx,vi,:,mi]=np.mean(mset,axis=0); channel_pos[pidx,vi]=np.mean(mset[0]>0,axis=0); channel_zero[pidx,vi]=np.mean(zeros[vi],axis=0)
   status.append(dict(layer=layer,protein_id=pid,status="complete",failure_reason=""))
  except Exception as exc: status.append(dict(layer=layer,protein_id=pid,status="failed",failure_reason=str(exc)))
 with (shard/f"per_protein_metrics_layer_{layer:02d}.csv").open("w",newline="") as f:
  fields=list(rows[0]) if rows else ["layer","protein_id","status"]; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 atomic_save_npz(shard/f"per_protein_channel_metrics_layer_{layer:02d}.npz",channel_metric_mean=channel_mean,channel_positive_fraction=channel_pos,channel_zero_scale_fraction=channel_zero,protein_status_code=np.array([int(x["status"]=="complete") for x in status]))
 with (shard/f"layer_status_{layer:02d}.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=["layer","protein_id","status","failure_reason"]); w.writeheader(); w.writerows(status)
if __name__=="__main__": main()
