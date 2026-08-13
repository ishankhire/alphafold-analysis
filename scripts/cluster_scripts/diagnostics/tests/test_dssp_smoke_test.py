import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / "dssp_smoke_test.py"
spec = importlib.util.spec_from_file_location("dssp_smoke_test", MODULE)
smoke_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke_test)


def test_make_dssp_input_removes_openfold_parent_records(tmp_path):
    source = tmp_path / "openfold.pdb"
    source.write_text("PARENT N/A\nATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \nEND\n")
    prepared = smoke_test.make_dssp_input(source)
    try:
        prepared_text = prepared.read_text()
    finally:
        prepared.unlink(missing_ok=True)
    assert prepared_text.startswith("CRYST1")
    assert "PARENT" not in prepared_text
    assert "ATOM      1  CA  ALA" in prepared_text
