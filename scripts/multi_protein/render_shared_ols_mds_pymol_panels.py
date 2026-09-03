#!/usr/bin/env python3
"""Render publication-ready PyMOL panels for shared-OLS MDS structures.

Run this file from an active PyMOL session:

    PyMOL> run /absolute/path/to/render_shared_ols_mds_pymol_panels.py

The renderer writes one full-atom Structure Module panel and five CA-only MDS
panels for each protein in the completed export manifest.  It computes a
separate union frame for each protein and reuses that exact frame across the
six panels, so scale is comparable within a protein but not forced across
different proteins.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_EXECUTING_FRAME = inspect.currentframe()
if _EXECUTING_FRAME is None:
    raise RuntimeError("Could not resolve the executing PyMOL script path")
SCRIPT_PATH = Path(_EXECUTING_FRAME.f_code.co_filename).resolve()
del _EXECUTING_FRAME

PROJECT_ROOT = SCRIPT_PATH.parents[2]
EXPORT_ROOT = PROJECT_ROOT / "delta_files" / "shared_ols_mds_pdb_exports"
MANIFEST_PATH = EXPORT_ROOT / "run_manifest.json"
PDB_ROOT = EXPORT_ROOT / "pdb_files"
ORIGINAL_FULL_ROOT = (
    EXPORT_ROOT
    / "aug-16-archive"
    / "reconstructed_pdb_files"
    / "general_base256_avg_symmetric"
)
OUTPUT_ROOT = EXPORT_ROOT / "pymol_renders"

LAYERS = (0, 10, 20, 30, 47)
EXPECTED_PROTEIN_COUNT = 10
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 1200
IMAGE_DPI = 300
VIEW_BUFFER_ANGSTROM = 8.0
VIEW_TOLERANCE = 1e-4

ORIGINAL_RGB = (0.00, 0.55, 0.05)
MDS_RGB = (0.82, 0.00, 0.00)
ORIGINAL_COLOR_NAME = "shared_ols_mds_original_green"
MDS_COLOR_NAME = "shared_ols_mds_reconstruction_red"
MDS_TUBE_RADIUS = 0.45

RENDER_OBJECT_PREFIX = "shared_ols_mds_panel_render"
AUTO_RUN_IN_PYMOL = True

GLOBAL_SCENE_SETTINGS: dict[str, Any] = {
    "opaque_background": 1,
    "ray_opaque_background": 1,
    "auto_zoom": 0,
    "orthoscopic": 1,
    "antialias": 2,
    "ray_shadow": 0,
    "ray_shadows": 0,
    "ambient": 0.55,
    "direct": 0.45,
    "specular": 0.15,
    "depth_cue": 0,
    "ray_trace_fog": 0,
}


@dataclass(frozen=True)
class PdbSummary:
    atom_count: int
    ca_count: int
    conect_count: int


@dataclass(frozen=True)
class RenderItem:
    protein_id: str
    label: str
    layer: int | None
    pdb_path: Path
    output_path: Path
    object_name: str
    is_original_full: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_pdb(path: Path) -> PdbSummary:
    atom_count = 0
    ca_count = 0
    conect_count = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith("ATOM  "):
                atom_count += 1
                if line[12:16].strip() == "CA":
                    ca_count += 1
            elif line.startswith("CONECT"):
                conect_count += 1
    return PdbSummary(
        atom_count=atom_count,
        ca_count=ca_count,
        conect_count=conect_count,
    )


def load_completed_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {path}")

    with path.open() as handle:
        manifest = json.load(handle)

    if manifest.get("run_status") != "complete":
        raise ValueError(
            f"Expected a complete export manifest, found: {manifest.get('run_status')!r}"
        )

    proteins = manifest.get("protein_ids")
    if not isinstance(proteins, list) or len(proteins) != EXPECTED_PROTEIN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PROTEIN_COUNT} manifest proteins, found: {proteins!r}"
        )
    if len(set(proteins)) != len(proteins) or not all(
        isinstance(protein_id, str) and protein_id for protein_id in proteins
    ):
        raise ValueError("Manifest protein_ids must be unique nonempty strings")

    manifest_layers = manifest.get("layers")
    if not isinstance(manifest_layers, list) or not set(LAYERS).issubset(manifest_layers):
        raise ValueError(
            f"Manifest layers {manifest_layers!r} do not contain requested layers {LAYERS}"
        )

    return manifest


def original_full_path(protein_id: str) -> Path:
    return (
        ORIGINAL_FULL_ROOT
        / protein_id
        / "original_full"
        / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    )


def reference_ca_path(protein_id: str) -> Path:
    return PDB_ROOT / protein_id / f"{protein_id}_reference_ca.pdb"


def reconstructed_path(protein_id: str, layer: int) -> Path:
    return (
        PDB_ROOT
        / protein_id
        / f"{protein_id}_shared_ols_mds_layer_{layer:02d}.pdb"
    )


def render_output_path(protein_id: str, layer: int | None) -> Path:
    protein_output_dir = OUTPUT_ROOT / protein_id
    if layer is None:
        return protein_output_dir / f"{protein_id}_original_full.png"
    return (
        protein_output_dir
        / f"{protein_id}_shared_ols_mds_layer_{layer:02d}.png"
    )


def object_name(protein_id: str, label: str) -> str:
    return f"{RENDER_OBJECT_PREFIX}_{protein_id}_{label}".replace("-", "_")


def collect_and_validate_render_items() -> dict[str, list[RenderItem]]:
    manifest = load_completed_manifest()
    errors: list[str] = []
    items_by_protein: dict[str, list[RenderItem]] = {}

    for protein_id in manifest["protein_ids"]:
        original_path = original_full_path(protein_id)
        reference_path = reference_ca_path(protein_id)
        required_paths = [
            original_path,
            reference_path,
            *(reconstructed_path(protein_id, layer) for layer in LAYERS),
        ]
        missing = [path for path in required_paths if not path.is_file()]
        if missing:
            errors.extend(f"Missing input: {path}" for path in missing)
            continue

        reference_summary = summarize_pdb(reference_path)
        original_summary = summarize_pdb(original_path)
        if reference_summary.atom_count == 0:
            errors.append(f"Reference PDB has no ATOM records: {reference_path}")
        if reference_summary.atom_count != reference_summary.ca_count:
            errors.append(f"Reference PDB is not CA-only: {reference_path}")
        if original_summary.ca_count != reference_summary.ca_count:
            errors.append(
                f"CA-count mismatch for {protein_id}: original="
                f"{original_summary.ca_count}, reference={reference_summary.ca_count}"
            )
        if original_summary.atom_count <= original_summary.ca_count:
            errors.append(f"Original PDB is not full-atom: {original_path}")

        protein_items = [
            RenderItem(
                protein_id=protein_id,
                label="original_full",
                layer=None,
                pdb_path=original_path,
                output_path=render_output_path(protein_id, None),
                object_name=object_name(protein_id, "original_full"),
                is_original_full=True,
            )
        ]

        for layer in LAYERS:
            pdb_path = reconstructed_path(protein_id, layer)
            summary = summarize_pdb(pdb_path)
            if summary.atom_count != summary.ca_count:
                errors.append(f"Reconstructed PDB is not CA-only: {pdb_path}")
            if summary.ca_count != reference_summary.ca_count:
                errors.append(
                    f"CA-count mismatch for {protein_id} layer {layer}: "
                    f"reconstruction={summary.ca_count}, "
                    f"reference={reference_summary.ca_count}"
                )
            expected_connections = max(0, summary.ca_count - 1)
            if summary.conect_count < expected_connections:
                errors.append(
                    f"Insufficient CONECT records for {pdb_path}: "
                    f"found={summary.conect_count}, expected>={expected_connections}"
                )

            protein_items.append(
                RenderItem(
                    protein_id=protein_id,
                    label=f"layer_{layer:02d}",
                    layer=layer,
                    pdb_path=pdb_path,
                    output_path=render_output_path(protein_id, layer),
                    object_name=object_name(protein_id, f"layer_{layer:02d}"),
                    is_original_full=False,
                )
            )

        items_by_protein[protein_id] = protein_items

    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(errors))

    all_items = [item for items in items_by_protein.values() for item in items]
    if len(all_items) != EXPECTED_PROTEIN_COUNT * (len(LAYERS) + 1):
        raise AssertionError(f"Expected 60 render items, found {len(all_items)}")

    expected_outputs = {item.output_path.resolve() for item in all_items}
    existing_outputs = (
        {path.resolve() for path in OUTPUT_ROOT.rglob("*.png")}
        if OUTPUT_ROOT.exists()
        else set()
    )
    unexpected_outputs = sorted(existing_outputs - expected_outputs)
    if unexpected_outputs:
        unexpected_text = "\n".join(str(path) for path in unexpected_outputs)
        raise ValueError(
            "Unexpected existing PNGs under the output root; move them before "
            f"rendering:\n{unexpected_text}"
        )

    return items_by_protein


def set_if_available(cmd, name: str, value: Any) -> bool:
    try:
        cmd.set(name, value)
    except Exception:
        print(f"Warning: PyMOL setting {name!r} is unavailable in this build")
        return False
    return True


def snapshot_available_settings(cmd, names: Iterable[str]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for name in names:
        try:
            snapshot[name] = cmd.get(name)
        except Exception:
            print(f"Warning: could not snapshot PyMOL setting {name!r}")
    return snapshot


def restore_settings(cmd, snapshot: dict[str, str]) -> None:
    for name, value in snapshot.items():
        try:
            cmd.set(name, value)
        except Exception as exc:
            print(f"Warning: could not restore PyMOL setting {name!r}: {exc}")


def configure_scene(cmd) -> None:
    cmd.bg_color("white")
    cmd.set_color(ORIGINAL_COLOR_NAME, ORIGINAL_RGB)
    cmd.set_color(MDS_COLOR_NAME, MDS_RGB)
    for name, value in GLOBAL_SCENE_SETTINGS.items():
        set_if_available(cmd, name, value)


def apply_original_style(cmd, item: RenderItem) -> None:
    cmd.hide("everything", item.object_name)
    cmd.set("cartoon_trace_atoms", 0, item.object_name)
    cmd.dss(item.object_name)
    cmd.cartoon("automatic", item.object_name)
    cmd.show("cartoon", item.object_name)
    cmd.color(ORIGINAL_COLOR_NAME, item.object_name)
    cmd.rebuild(item.object_name)


def apply_mds_style(cmd, item: RenderItem) -> None:
    cmd.hide("everything", item.object_name)
    cmd.set("cartoon_trace_atoms", 1, item.object_name)
    cmd.set("cartoon_tube_radius", MDS_TUBE_RADIUS, item.object_name)
    cmd.cartoon("tube", item.object_name)
    cmd.show("cartoon", item.object_name)
    cmd.color(MDS_COLOR_NAME, item.object_name)
    cmd.rebuild(item.object_name)


def cleanup_render_objects(cmd) -> None:
    for name in cmd.get_names("objects"):
        if name.startswith(RENDER_OBJECT_PREFIX):
            cmd.delete(name)


def restore_enabled_objects(cmd, enabled_objects: list[str]) -> None:
    existing_objects = set(cmd.get_names("objects"))
    cmd.disable("all")
    for name in enabled_objects:
        if name in existing_objects:
            cmd.enable(name)


def load_and_style_protein(cmd, items: list[RenderItem]) -> None:
    for item in items:
        cmd.load(str(item.pdb_path), item.object_name)
        if item.is_original_full:
            apply_original_style(cmd, item)
        else:
            apply_mds_style(cmd, item)
        cmd.enable(item.object_name)


def derive_union_view(cmd, items: list[RenderItem]) -> tuple[float, ...]:
    original = next(item for item in items if item.is_original_full)
    union_selection = " or ".join(f"({item.object_name})" for item in items)

    cmd.orient(original.object_name)
    cmd.zoom(union_selection, buffer=VIEW_BUFFER_ANGSTROM, complete=1)
    cmd.refresh()
    view = tuple(float(value) for value in cmd.get_view())
    if len(view) != 18 or not all(math.isfinite(value) for value in view):
        raise RuntimeError(f"PyMOL returned an invalid view for {original.protein_id}")
    return view


def assert_view_matches(cmd, expected_view: tuple[float, ...], protein_id: str) -> None:
    actual_view = tuple(float(value) for value in cmd.get_view())
    if len(actual_view) != len(expected_view) or any(
        abs(actual - expected) > VIEW_TOLERANCE
        for actual, expected in zip(actual_view, expected_view)
    ):
        raise RuntimeError(f"PyMOL view changed unexpectedly while rendering {protein_id}")


def render_item(cmd, item: RenderItem, shared_view: tuple[float, ...]) -> None:
    cmd.disable("all")
    cmd.enable(item.object_name)
    cmd.set_view(shared_view)
    cmd.rebuild(item.object_name)
    cmd.refresh()
    assert_view_matches(cmd, shared_view, item.protein_id)

    item.output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Rendering {item.protein_id} {item.label} -> {item.output_path} "
        f"(input_sha256={sha256_file(item.pdb_path)[:12]})"
    )
    cmd.png(
        str(item.output_path),
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        dpi=IMAGE_DPI,
        ray=1,
        quiet=0,
    )


def paeth_predictor(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def png_content_bbox(path: Path) -> tuple[tuple[int, int], tuple[int, int, int, int] | None]:
    """Return dimensions and non-white bounds for a simple 8-bit PNG.

    PyMOL's ray renderer writes non-interlaced RGB or RGBA PNGs.  Decoding that
    small, fixed subset here avoids requiring Pillow in the PyMOL environment.
    """
    png_signature = b"\x89PNG\r\n\x1a\n"
    with path.open("rb") as handle:
        if handle.read(len(png_signature)) != png_signature:
            raise ValueError("invalid PNG signature")

        ihdr: bytes | None = None
        compressed_parts: list[bytes] = []
        while True:
            length_bytes = handle.read(4)
            if not length_bytes:
                break
            if len(length_bytes) != 4:
                raise ValueError("truncated PNG chunk length")
            chunk_length = struct.unpack(">I", length_bytes)[0]
            chunk_type = handle.read(4)
            chunk_data = handle.read(chunk_length)
            chunk_crc = handle.read(4)
            if len(chunk_type) != 4 or len(chunk_data) != chunk_length or len(chunk_crc) != 4:
                raise ValueError("truncated PNG chunk")
            if chunk_type == b"IHDR":
                ihdr = chunk_data
            elif chunk_type == b"IDAT":
                compressed_parts.append(chunk_data)
            elif chunk_type == b"IEND":
                break

    if ihdr is None or len(ihdr) != 13 or not compressed_parts:
        raise ValueError("PNG is missing IHDR or IDAT data")

    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", ihdr)
    )
    if bit_depth != 8 or color_type not in {0, 2, 4, 6}:
        raise ValueError(
            f"unsupported PNG encoding: bit_depth={bit_depth}, color_type={color_type}"
        )
    if compression != 0 or filtering != 0 or interlace != 0:
        raise ValueError(
            "unsupported compressed, filtered, or interlaced PNG encoding: "
            f"compression={compression}, filtering={filtering}, interlace={interlace}"
        )

    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    row_bytes = width * channels
    raw = zlib.decompress(b"".join(compressed_parts))
    expected_raw_length = height * (row_bytes + 1)
    if len(raw) != expected_raw_length:
        raise ValueError(
            f"unexpected decompressed length: {len(raw)}, expected {expected_raw_length}"
        )

    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    previous = bytearray(row_bytes)
    offset = 0

    for y in range(height):
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + row_bytes])
        offset += row_bytes

        for index in range(row_bytes):
            left = row[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = paeth_predictor(left, above, upper_left)
            elif filter_type == 0:
                predictor = 0
            else:
                raise ValueError(f"unsupported PNG row filter: {filter_type}")
            row[index] = (row[index] + predictor) & 0xFF

        for x in range(width):
            pixel_offset = x * channels
            if color_type == 0:
                red = green = blue = row[pixel_offset]
                alpha = 255
            elif color_type == 2:
                red, green, blue = row[pixel_offset : pixel_offset + 3]
                alpha = 255
            elif color_type == 4:
                red = green = blue = row[pixel_offset]
                alpha = row[pixel_offset + 1]
            else:
                red, green, blue, alpha = row[pixel_offset : pixel_offset + 4]

            if alpha > 4 and (red < 251 or green < 251 or blue < 251):
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        previous = row

    bbox = None if max_x < 0 else (min_x, min_y, max_x + 1, max_y + 1)
    return (width, height), bbox


def validate_png_bundle(items_by_protein: dict[str, list[RenderItem]]) -> None:
    all_items = [item for items in items_by_protein.values() for item in items]
    expected_paths = {item.output_path.resolve() for item in all_items}
    actual_paths = {path.resolve() for path in OUTPUT_ROOT.rglob("*.png")}
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        details = [*(f"Missing PNG: {path}" for path in missing)]
        details.extend(f"Unexpected PNG: {path}" for path in unexpected)
        raise RuntimeError("PNG inventory validation failed:\n" + "\n".join(details))

    errors: list[str] = []
    for path in sorted(expected_paths):
        try:
            dimensions, bbox = png_content_bbox(path)
            if dimensions != (IMAGE_WIDTH, IMAGE_HEIGHT):
                errors.append(
                    f"Wrong dimensions for {path}: {dimensions}, expected "
                    f"{(IMAGE_WIDTH, IMAGE_HEIGHT)}"
                )
                continue
            if bbox is None:
                errors.append(f"Blank PNG: {path}")
                continue
            left, top, right, bottom = bbox
            if left <= 0 or top <= 0 or right >= IMAGE_WIDTH or bottom >= IMAGE_HEIGHT:
                errors.append(f"Rendered content touches an image boundary: {path}")
        except Exception as exc:
            errors.append(f"Could not validate {path}: {exc}")

    if errors:
        raise RuntimeError("PNG pixel validation failed:\n" + "\n".join(errors))

    print(f"Validated {len(expected_paths)} nonblank {IMAGE_WIDTH}x{IMAGE_HEIGHT} PNGs")


def render_shared_ols_mds_pymol_panels() -> None:
    """Validate all inputs, render all panels, and restore the PyMOL session."""
    items_by_protein = collect_and_validate_render_items()

    try:
        from pymol import cmd
    except ImportError as exc:
        raise SystemExit(
            "This renderer must run inside PyMOL. In the PyMOL command line, use: "
            f"run {SCRIPT_PATH}"
        ) from exc

    started = time.monotonic()
    original_view = tuple(cmd.get_view())
    enabled_objects = cmd.get_names("objects", enabled_only=1)
    setting_snapshot = snapshot_available_settings(
        cmd,
        ["bg_rgb", *GLOBAL_SCENE_SETTINGS],
    )

    try:
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Output root: {OUTPUT_ROOT}")
        configure_scene(cmd)
        cleanup_render_objects(cmd)

        for protein_id, items in items_by_protein.items():
            print(f"Preparing shared union frame for {protein_id}")
            cmd.disable("all")
            load_and_style_protein(cmd, items)
            shared_view = derive_union_view(cmd, items)
            for item in items:
                render_item(cmd, item, shared_view)
            cleanup_render_objects(cmd)

        validate_png_bundle(items_by_protein)
        elapsed = time.monotonic() - started
        print(f"Finished 60 PyMOL panel renders in {elapsed:.1f} seconds")
    finally:
        cleanup_render_objects(cmd)
        restore_settings(cmd, setting_snapshot)
        restore_enabled_objects(cmd, enabled_objects)
        cmd.set_view(original_view)
        cmd.refresh()


try:
    from pymol import cmd as _pymol_cmd
except ImportError:
    _pymol_cmd = None
else:
    _pymol_cmd.extend(
        "render_shared_ols_mds_pymol_panels",
        render_shared_ols_mds_pymol_panels,
    )


if (_pymol_cmd is not None and AUTO_RUN_IN_PYMOL) or __name__ == "__main__":
    render_shared_ols_mds_pymol_panels()
