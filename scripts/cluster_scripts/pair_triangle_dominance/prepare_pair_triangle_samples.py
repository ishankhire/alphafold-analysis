#!/usr/bin/env python3
from pathlib import Path
import csv, json, os, sys, math
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pair_triangle_dominance.pair_triangle_dominance_common import *

IDS = """6tf4_A 6wmk_A 6wqc_A 6zpp_A 7ab3_E 7ar0_B 7atr_A 7au7_A 7awk_A 7b0d_A 7b1k_B 7b1w_F 7b26_C 7b28_F 7b29_A 7b2a_A 7b2o_A 7b3a_A 7b4q_A 7b7t_A 7bbz_A 7bcz_A 7bew_A 7bhy_A 7cjs_B 7dcm_A 7dfe_A 7djy_A 7dk9_A 7dkk_A 7dko_A 7dmf_A 7dms_A 7dnu_A 7don_B 7dq9_A 7dru_C 7dtp_A 7dup_A 7dut_A 7dvn_A 7e0m_A 7e2v_A 7e37_A 7e3z_A 7ebt_A 7ee3_C 7ef6_A 7eqx_A 7esx_A 7et8_A 7ewu_B 7ezg_A 7f0h_A 7f6e_B 7f8a_A 7fbh_A 7fbp_B 7fe3_A 7ffa_A 7fh3_A 7fip_A 7jiz_A 7jt9_A 7kdx_B 7kik_A 7kiu_A 7kos_A 7kqv_D 7ksp_A 7kua_A 7kzh_A 7l6j_A 7l6y_A 7l8n_A 7lbu_A 7lc5_A 7lgr_A 7ljh_A 7lnu_B 7lqm_D 7lqn_A 7ls0_B 7lvz_D 7m4n_A 7mcc_A 7mfi_A 7mfw_B 7mpz_A 7mrk_D 7mro_A 7mu9_A 7mwr_A 7ndr_A 7nl4_A 7ntn_A 7ny6_A 7o49_F 7obm_A 7ofn_A 7on9_A 7ool_A 7ouq_A 7p3b_B 7p3t_B 7p82_C 7pab_A 7pbk_A 7plb_B 7poh_B 7ppp_A 7pq7_A 7pqf_A 7prd_A 7puo_A 7q03_A 7q1b_A 7q47_A 7r84_A 7rbw_A 7rdt_A 7re4_A 7s94_C 7sf6_A 7sir_A 7spo_C 7spp_C 7stt_A 7stz_C 7swh_A 7swk_B 7sy9_A 7t24_A 7t5f_B 7t71_A 7t9w_B 7ta9_A 7tav_B 7tbs_A 7tcb_B 7thw_A 7tj1_D 7tkv_A 7v1v_A 7v5y_B 7vdy_A 7vmu_A 7vnb_A 7vpf_A 7vw0_B 7w83_A 7wbr_A 7wgk_A""".split()
def main():
 out=Path(os.environ.get("TRIANGLE_OUTPUT_DIR", Path(os.environ.get("CSV_ROOT", "delta_files"))/"pair_representation_triangle_dominance")); meta=out/"metadata"; meta.mkdir(parents=True,exist_ok=True)
 ids=np.array(IDS); offsets=[0]; ranks=[]; ts=[]; rows=[]
 for pid in ids:
  n=int(os.environ.get("N_RESIDUES_"+pid, "0")); t,r,seed=sample_triples(pid,n); ts.append(t); ranks.append(r); offsets.append(offsets[-1]+len(t)); rows.append(dict(sample_version=SAMPLE_VERSION,protein_id=pid,n_residues=n,n_population_triangles=math.comb(n,3) if n>=3 else 0,n_sampled_triangles=len(t),base_seed=20260827,derived_seed=seed,rank_order="ascending",unranking_algorithm="colex_v1",rng_algorithm="PCG64",sample_offset_start=offsets[-2],sample_offset_stop=offsets[-1],sample_sha256=sha256_bytes(t.tobytes()),status="complete" if n>=3 else "insufficient_residues"))
 np.savez_compressed(meta/"triangle_samples.npz",protein_ids=ids,offsets=np.array(offsets),sampled_ranks=np.concatenate(ranks) if ranks else np.empty(0,np.int64),tri_i=np.concatenate([x[:,0] for x in ts]) if ts else np.empty(0,np.int32),tri_j=np.concatenate([x[:,1] for x in ts]) if ts else np.empty(0,np.int32),tri_k=np.concatenate([x[:,2] for x in ts]) if ts else np.empty(0,np.int32))
 with (meta/"sample_manifest.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 json.dump({"sample_version":SAMPLE_VERSION,"protein_count":len(ids),"sample_cap":SAMPLE_CAP,"base_seed":20260827},(meta/"sample_provenance.json").open("w"),indent=2)
if __name__=="__main__": main()
