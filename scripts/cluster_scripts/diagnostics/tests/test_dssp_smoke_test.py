import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / "dssp_smoke_test.py"
spec = importlib.util.spec_from_file_location("dssp_smoke_test", MODULE)
smoke_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke_test)


def test_make_dssp_input_removes_openfold_metadata(tmp_path):
    source = tmp_path / "openfold.pdb"
    source.write_text(
        "REMARK no_recycling=3, max_templates=4\n"
        "PARENT N/A\n"
        "MODEL     1\n"
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ENDMDL\nEND\n"
    )
    prepared = smoke_test.make_dssp_input(source)
    try:
        prepared_text = prepared.read_text()
    finally:
        prepared.unlink(missing_ok=True)
    assert prepared_text.startswith("CRYST1")
    assert "REMARK" not in prepared_text
    assert "PARENT" not in prepared_text
    assert "MODEL     1" in prepared_text
    assert "ATOM      1  CA  ALA" in prepared_text


def test_make_dssp_input_rejects_metadata_only_file(tmp_path):
    source = tmp_path / "metadata_only.pdb"
    source.write_text("REMARK no_recycling=3\nPARENT N/A\n")
    try:
        smoke_test.make_dssp_input(source)
    except ValueError as error:
        assert "No ATOM or HETATM records" in str(error)
    else:
        raise AssertionError("metadata-only PDB should be rejected")
