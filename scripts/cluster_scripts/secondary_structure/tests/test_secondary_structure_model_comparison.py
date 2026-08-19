from __future__ import annotations

import csv
import hashlib
import math
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from scripts.cluster_scripts.secondary_structure import secondary_structure_model_comparison as m


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)


class ModelComparisonTests(unittest.TestCase):
    def test_shared_ols_loader_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"variant":"general_base256_combined","layer":0,"feature_index":-1,"coefficient_raw_scale":2.0}]
            rows += [{"variant":"general_base256_combined","layer":0,"feature_index":i,"coefficient_raw_scale":i} for i in range(256)]
            write_csv(root/"coefficients_0-23.csv", rows)
            intercept, coef, _ = m.load_shared_ols(root, 0)
            self.assertEqual(intercept, 2.0); self.assertEqual(coef.shape, (256,)); self.assertEqual(coef[255], 255)

    def test_shared_ols_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=[{"variant":"general_base256_combined","layer":0,"feature_index":-1,"coefficient_raw_scale":0}]
            rows += [{"variant":"general_base256_combined","layer":0,"feature_index":i if i<255 else 254,"coefficient_raw_scale":0} for i in range(256)]
            write_csv(root/"coefficients_0-23.csv",rows)
            with self.assertRaises(ValueError): m.load_shared_ols(root,0)

    def make_indexed(self, root, *, gated=True):
        shard=root/"model_artifacts/coefficient_shards/layer.npz"; shard.parent.mkdir(parents=True)
        n=6 if gated else 1
        np.savez(shard,coefficient_raw_scale=np.arange(n*256,dtype=float).reshape(n,256),intercept_raw_scale=np.arange(n,dtype=float),protein_id=np.asarray(["" for _ in range(n)] if gated else ["p"]))
        digest=hashlib.sha256(shard.read_bytes()).hexdigest(); rows=[]
        for i in range(n):
            rows.append({"layer":0,"protein_id":"" if gated else "p","gate_aa_pair":"all" if gated else "","gate_sequence_bin":m.SEQUENCE_BIN_LABELS[i] if gated else "","shard_path":"model_artifacts/coefficient_shards/layer.npz","array_row":i,"fit_status":"fitted","file_sha256":digest})
        write_csv(root/"model_artifacts/coefficient_index.csv",rows)
        return shard

    def test_sequence_index_and_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.make_indexed(root)
            models=m.load_sequence_gated(root,0); sep=np.asarray([1,2,4,5,9,10,19,20,49,50]); hi=sep; lo=np.zeros_like(sep); x=np.zeros((len(sep),256))
            prediction=m.route_sequence_predictions(x,hi,lo,np.asarray(["A"]*51),models)
            self.assertEqual(prediction.tolist(),[0,1,1,2,2,3,3,4,4,5])

    def test_index_checksum_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); shard=self.make_indexed(root); shard.write_bytes(shard.read_bytes()+b"bad")
            with self.assertRaises(ValueError): m.load_sequence_gated(root,0)

    def test_per_protein_index_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.make_indexed(root,gated=False)
            loaded=m.load_per_protein(root,0,["p"])
            self.assertEqual(set(loaded),{"p"})

    def test_xgboost_dependency_error_is_explicit(self):
        real_import = __import__
        def blocked(name, *args, **kwargs):
            if name == "xgboost": raise ImportError("synthetic missing dependency")
            return real_import(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=blocked):
            with self.assertRaisesRegex(RuntimeError, "xgboost is required"):
                m.load_xgboost(Path("unused"), 0)

    def test_final_xgboost_complete_status_is_accepted(self):
        rows = [{"layer":layer,"fit_status":"complete","selected_configuration_id":"S4","final_boosting_rounds":"700","training_row_cap":"5000"} for layer in range(48)]
        m.validate_xgboost_model_status(rows)

    def test_final_xgboost_status_rejects_incomplete_layer(self):
        rows = [{"layer":layer,"fit_status":"complete","selected_configuration_id":"S4","final_boosting_rounds":"700","training_row_cap":"5000"} for layer in range(47)]
        with self.assertRaisesRegex(ValueError, "rows=47"):
            m.validate_xgboost_model_status(rows)

    def test_canonical_feature_direction(self):
        block=np.zeros((3,3,128),dtype=np.float32); block[2,0]=1; block[0,2]=2
        x,hi,lo=m.sparse_features(block,[(0,2)])
        self.assertTrue(np.all(x[0,:128]==1)); self.assertTrue(np.all(x[0,128:]==2)); self.assertEqual((hi[0],lo[0]),(2,0))

    def test_pair_position_matches_lower_triangle(self):
        for n in range(2,12):
            ii,jj=np.tril_indices(n,k=-1)
            for pos,(i,j) in enumerate(zip(ii,jj)): self.assertEqual(m.canonical_pair_position(int(i),int(j)),pos)

    def test_selected_union_is_deduplicated(self):
        cache={"dssp_codes":np.asarray(list("HHHHH")),"chains":np.asarray(["A"]*5),"bridge_i":np.asarray([0]),"bridge_j":np.asarray([2]),"bridge_main_eligible":np.asarray([1]),"control_i":np.asarray([0]),"control_j":np.asarray([2])}
        pairs=m.selected_pair_union(cache)
        self.assertEqual(len(pairs),len(set(pairs))); self.assertIn((0,2),pairs)

    def test_chain_boundaries_stop_segments(self):
        ids,segments=m.geometry.build_segments(np.asarray(list("HHHH")),np.asarray(["A","A","B","B"]),"H")
        self.assertEqual([s["length"] for s in segments],[2,2]); self.assertNotEqual(ids[1],ids[2])

    def test_dssp_bridge_orientation_parser(self):
        line=list(" "*80); line[0:5]=f"{1:5d}"; line[5:10]=f"{10:5d}"; line[11]="A"; line[16]="E"; line[23]="a"; line[25:29]=f"{2:4d}"
        rows=m.geometry.parse_dssp_bridge_rows("".join(line))
        self.assertEqual(rows[0]["partners"][0][1],"parallel")

    def test_matched_random_is_deterministic_and_matched(self):
        residues=[{"chain":"A"} for _ in range(20)]; codes=np.asarray(["E"]*20); seg=np.zeros(20,dtype=int); bridges=[{"bridge_id":0,"i":2,"j":8,"main_eligible":1}]
        first=m.geometry.choose_bridge_controls("p",residues,codes,seg,bridges); second=m.geometry.choose_bridge_controls("p",residues,codes,seg,bridges)
        self.assertEqual(first,second)
        self.assertTrue(all(abs(r["i"]-r["j"])==6 for r in first if r["type"]=="matched_random"))

    def test_calibration_score_and_both_gates(self):
        c={"zscore_mean":[0,0,0],"zscore_scale":[1,1,1],"mu_helix_z":[1,1,1],"mu_strand_z":[3,3,3]}
        score,dh,de=m.geometry.score_triples(np.asarray([[1,1,1.]]),c)
        self.assertGreater(score[0],0); self.assertEqual(dh[0],0); self.assertGreater(de[0],0)
        self.assertFalse(bool((score[0]>0) and (10<=1)))

    def test_heldout_helix_requires_all_three(self):
        held={(0,2):True,(0,3):True,(0,4):False}
        self.assertFalse(all(held[p] for p in ((0,2),(0,3),(0,4))))

    def test_pooled_auc_differs_from_paired_correct(self):
        partners=np.asarray([1.,10.]); controls=np.asarray([2.,3.,11.,12.])
        auc=m.safe_auc(np.r_[np.ones(2),np.zeros(4)],-np.r_[partners,controls])
        paired=np.mean([1<2.5,10<11.5])
        self.assertNotEqual(auc,paired)

    def test_trajectory_fields_complete(self):
        rows=[]
        visible=[0]*4+[1]*3+[0]+[1]*40
        for layer in range(48): rows.append({"model_family":"m","evaluation_scope":"s","protein_id":"p","segment_id":"1","layer":str(layer),"visible":str(visible[layer]),"mean_score":str(layer)})
        m.trajectory_fields(rows,"mean_score")
        self.assertEqual(rows[0]["first_visible_layer"],4); self.assertEqual(rows[0]["crossed_then_reverted"],1); self.assertEqual(rows[0]["n_visibility_transitions"],3); self.assertEqual(rows[0]["longest_visible_run"],40); self.assertAlmostEqual(rows[0]["visibility_auc"],43/48)

    def test_support_zero_summary_is_retained(self):
        row=m.summarize("shared_global_ols",0,m.SCOPES[0],[{"n_decoded_structure_pairs":0,"n_negative_predictions":0}],[],[],[])
        self.assertEqual(row["n_helix_residues"],0); self.assertTrue(math.isnan(row["helix_strand_auroc"])); self.assertEqual(row["n_helix_segments_eligible"],0)

    def test_shallow_flat_deployment_paths_do_not_index_missing_parents(self):
        flat = Path("/u/user/openfold_interpretability_canonical")
        with mock.patch.object(m, "HERE", flat), mock.patch.object(m, "build_dataset_protein_manifest", return_value=[]), mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(m.project_or_flat_root(), flat)
            self.assertEqual(m.output_root(), flat/"delta_files"/m.ANALYSIS_NAME)
            self.assertEqual(m.input_roots()["shared_global_ols"], flat/"delta_files/general_base256_combined")
            self.assertEqual(m._dataset_records(), {})

    def test_pairwise_differences_use_both_model_rows(self):
        rows=[]
        for layer in range(48):
            for scope in m.SCOPES:
                for model_index,model in enumerate(m.MODELS):
                    rows.append({"analysis_name":m.ANALYSIS_NAME,"model_family":model,"layer":str(layer),"evaluation_scope":scope,"split":"test","n_helix_residues":"10","helix_strand_auroc":str(0.5+0.1*model_index)})
        differences=m._pairwise(rows)
        expected=48*len(m.SCOPES)*math.comb(len(m.MODELS),2)
        self.assertEqual(len(differences),expected)
        first=differences[0]
        self.assertEqual((first["model_a"],first["model_b"]),(m.MODELS[0],m.MODELS[1]))
        self.assertAlmostEqual(first["difference_b_minus_a"],0.1)


if __name__ == "__main__": unittest.main()
