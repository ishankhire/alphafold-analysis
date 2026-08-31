#!/usr/bin/env python3
"""Verify the flat protocol-2.0 cross-validation deployment on Delta."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


# SHA-256 values from the repository deployment inventory on 2026-08-31.
DEPLOYMENT_FILES = {
    "__init__.py": "46a9c0ea5ad6a7db3caddc16707aa2f4ea8e8af8a7e5b7fde9041a2beb12cfff",
    "aggregate_per_protein_ols_base256.py": "4d931d54175283a6ffbb18ba4146dcc98a48d98a23234192b647695fd0b6948a",
    "aggregate_per_protein_ols_base256.sbatch": "564bda5d6ab10cb3b2b6e0948f7154d0085f9da14d27b64cd20308015d384fcf",
    "aggregate_sequence_gated_ols_base256.py": "7554bcbb5478b7f2d8a6e16c88b57a1fe7ca331bed9184e8c5dd2c7f10b4807b",
    "aggregate_sequence_gated_ols_base256.sbatch": "46c88a5bf959f362472d0f6498e382073ed5345f83f18c25de65e20006675f29",
    "aggregate_shared_ols_base256.py": "8793b8ab0e6c489155748397389a8d92fd67400ce46321c63d1c27d165aa44fa",
    "aggregate_shared_ols_base256.sbatch": "81416c6503743a21782684e9d8921dc0cf308aa936d036aab1825d93f95146d8",
    "aggregate_shared_xgboost_base256_fixed_s4.py": "8d6a24cd77759796d3e34ad8e5d630ed6b16dd739bfd7595d17d3866671135f4",
    "aggregate_shared_xgboost_base256_fixed_s4.sbatch": "0f538d63266ae622ae75c3a8bc5b3e8f861ca009b11ffcd08071239a8dd4b190",
    "cluster_regression_helpers.py": "52aaf788044f1ee7434bfffcf6c85469b531cb6d7b773eed840bfd84f81377a7",
    "cross_validation_common.py": "b1c686252d7508ad9a681e5d152a47b33ef57efaa156f1f684ef93147aee94eb",
    "per_protein_ols_base256_common.py": "eca523acb510900f90a73b62e52f356524bc13e93e4e5aba353ec3c85fdc1d60",
    "per_protein_ols_base256_layer.py": "f65de917e64cd989322224d9476301017a443407df160c7f0d12ae0c69ef1131",
    "per_protein_ols_base256_layer.sbatch": "578588ca63645feee4e1d76030796cd11a648ac0c667e4c9f94c006467553f6d",
    "regression_output_helpers.py": "5bf381690d84e9ecf1777d2a3e2f95741100b6f8aac40158c23af44638d3159c",
    "regression_protocol.py": "88c460f77fc590f30891e8e2d9e9f3d7dcf36dbf96781300ef4c1a83572c664e",
    "regression_protocol_data/regression_master_split_v1.csv": "16cbb0b976bd76aabfa67ce032e009416b095117016225403fc3348d58f2cdf3",
    "sequence_gated_ols_base256_common.py": "928781e14bbd0b06318abdd66324a6ec6208502cc81f57636212007889faf52e",
    "sequence_gated_ols_base256_fold_layer.py": "757ffb6406a5626a6e00ff7fed088513341432600980e91d77c7360641108511",
    "sequence_gated_ols_base256_fold_layer.sbatch": "8f4ebd395f16a91ca3413ba6ab02d0341cc7a8403cefbe894d629d9f8fadbe0b",
    "shared_ols_base256_common.py": "5a1b9600c0b0de327a68162b13c35fff8c9b19a6a2a05f2c333c1499e03c86fc",
    "shared_ols_base256_fold_layer.py": "e42eaad83a8ddf10fdaa3d42db3170ce53ba1511ed3eba10dc8a83e3d97e8997",
    "shared_ols_base256_fold_layer.sbatch": "90338fe503c75945dcde70727b569fdaf63d4e7e6b1cd2bbcb173c77d51a5994",
    "shared_xgboost_base256_fixed_s4_common.py": "c85b867db230512fa0ec0411cbfc4f19ad760b92e86f9522acff3ba18aa8e549",
    "shared_xgboost_base256_fixed_s4_config.json": "e8e77b8a3c88eba4ff428ce5fa0f1edc4215a66e7e422416a45b9b99801d88f5",
    "shared_xgboost_base256_fixed_s4_fold_layer.py": "23134561a826d3318726ac3631a19746efeef339827051a0b0941f5d00a0a063",
    "shared_xgboost_base256_fixed_s4_fold_layer.sbatch": "5dfb470523c35891fa7e37997244ac3b249e640eb28d97a72b09607d704043f7",
}

TEST_FILES = {
    "tests/__init__.py": "b0351f20db5155cefd3da9fa94d77a3107fb45999d0c67247b20cae8ac7881f3",
    "tests/test_adversarial_cv_wiring.py": "1cc5b15fa3e55484925a9490d6d97155dd30dfa1822400c22422f450341428ff",
    "tests/test_cross_validation_common.py": "715e47668394467ace9ef67580be3b6f4ecb23257712d0c161704dc1e266c813",
    "tests/test_geometry_independent_oracles.py": "04ac51c3f65649e466d06f8bd040bf9cd854e95e757fe4a0e4980545d6a8d41d",
    "tests/test_per_protein_ols_base256.py": "985a0f3bf7a5a37f75cb9e46c60d4f2574588efebf0e00fbd92d8ff0a2b42aab",
    "tests/test_sequence_gated_ols_base256.py": "839c3b341b3010e4a23d0c4a323636b34ccd3c41a1f8d438d55870450b6f1f29",
    "tests/test_shared_ols_base256.py": "bfdbf7289d00c96ba1f26596fc79c6df15ef39571b5170b95e00fd45e36387be",
    "tests/test_shared_xgboost_base256_fixed_s4.py": "2bf861fc83c287f499f0ca647daa3a8e8043c41ddedb31b699c4ed80ac438f25",
    "triangle_violation_subgroup_common.py": "f0c8671cc6b1ed7be96d24ebb26f6749ceb4c16dba15a237568c3b13d441b491",
}

METADATA_FILES = {
    "fold_digests.json": "da4746eb5f2b30458ba05cc7c04852bf97714a551a165aad201de552dcc19318",
    "per_protein_pair_folds_v1.csv": "7e5166d70f3a5fe1fe7502352d0ce987396c5bb7830366f72a9d0db1614c4011",
    "regression_outer_folds_v1.csv": "173ad746e627705046f6acac1e1c19c3dcae1e4f6b36b8d87f401590fc42ae98",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_group(label: str, root: Path, expected: dict[str, str], hashes: bool) -> int:
    failures = 0
    print(f"\n{label}: {root}")
    for relative, expected_hash in sorted(expected.items()):
        path = root / relative
        if not path.is_file():
            print(f"  MISSING  {relative}")
            failures += 1
        elif hashes and sha256_file(path) != expected_hash:
            print(f"  DIFFERENT {relative}")
            failures += 1
        else:
            print(f"  OK       {relative}")
    return failures


def parse_args() -> argparse.Namespace:
    default_csv_root = Path(os.environ.get("CSV_ROOT", Path.home() / "csv_files"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment-root", type=Path, default=Path(__file__).resolve().parent,
        help="Flat script directory (default: directory containing this verifier).",
    )
    parser.add_argument(
        "--csv-root", type=Path, default=default_csv_root,
        help="CSV output root (default: CSV_ROOT or ~/csv_files).",
    )
    parser.add_argument(
        "--existence-only", action="store_true",
        help="Check paths but do not compare them with the embedded current hashes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deployment_root = args.deployment_root.expanduser().resolve()
    metadata_root = (
        args.csv_root.expanduser().resolve() / "regression_5fold_v1" / "metadata"
    )
    check_hashes = not args.existence_only

    failures = 0
    failures += check_group("Production deployment", deployment_root, DEPLOYMENT_FILES, check_hashes)
    failures += check_group("Focused-test support", deployment_root, TEST_FILES, check_hashes)
    failures += check_group("Frozen fold metadata", metadata_root, METADATA_FILES, check_hashes)

    mode = "presence and current hashes" if check_hashes else "presence only"
    if failures:
        print(f"\nFAILED: {failures} required file(s) missing or different ({mode}).")
        return 1
    print(f"\nPASS: all {len(DEPLOYMENT_FILES) + len(TEST_FILES) + len(METADATA_FILES)} files match ({mode}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
