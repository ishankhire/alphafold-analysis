#!/usr/bin/env python3
"""
Batch-render the 7b3a_A full original structure and selected CA reconstructions.

Usage from an active PyMOL session with the named view already stored:

    PyMOL> run chimera_files/render_original_full_layers.py

For command-line use, load a session that contains the saved view first:

    pymol -cq saved_views.pse chimera_files/render_original_full_layers.py

The active PyMOL session must contain a saved view named "7b3a_A_view".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT_ENV_VAR = "ALPHAFOLD_PROTEIN_ROOT"
FALLBACK_PROJECT_ROOT = (
    Path.home() / "Desktop" / "ai_safety" / "AlphaFoldResearch" / "protein"
)


def candidate_roots_from(start: Path) -> list[Path]:
    resolved = start.expanduser().resolve()
    if resolved.is_file():
        resolved = resolved.parent
    return [resolved, *resolved.parents]


def looks_like_project_root(path: Path) -> bool:
    return (path / "reconstructed_pdb_files").is_dir() and (path / "chimera_files").is_dir()


def find_project_root() -> Path:
    starts: list[Path] = []
    env_root = os.environ.get(PROJECT_ROOT_ENV_VAR)
    if env_root:
        starts.append(Path(env_root))

    starts.append(Path.cwd())
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    starts.append(FALLBACK_PROJECT_ROOT)

    checked: set[Path] = set()
    for start in starts:
        for candidate in candidate_roots_from(start):
            if candidate in checked:
                continue
            checked.add(candidate)
            if looks_like_project_root(candidate):
                return candidate

    raise FileNotFoundError(
        "Could not find the protein project root. Start PyMOL from the repo, "
        f"cd into the repo/chimera_files directory, or set {PROJECT_ROOT_ENV_VAR}."
    )


PROJECT_ROOT = find_project_root()
SCRIPT_DIR = PROJECT_ROOT / "chimera_files"

# Input/output configuration.
INPUT_DIR = PROJECT_ROOT / "reconstructed_pdb_files"
OUTPUT_DIR = SCRIPT_DIR
PROTEIN_ID = "7f6e_B"
ORIGINAL_FULL_PDB_PATH_TEMPLATE = (
    "{protein_id}/original_full/{protein_id}_model_1_ptm_unrelaxed.pdb"
)
RECONSTRUCTED_PDB_PATH_TEMPLATE = (
    "{protein_id}/reconstructed_ca_layers/{protein_id}_reconstruct_L{layer:02d}.pdb"
)

# Render subset.
LAYERS = (10, 15, 22, 47)
USE_SAVED_VIEWS = True

# Image settings.
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 1200
RAY_TRACE = True
ENABLE_RAY_SHADOWS = True
ENABLE_DEPTH_FADING = True
DEPTH_FADING_START = 0.4
FORCE_ORTHOSCOPIC = True
RENDER_CLIP_BUFFER = 100.0
RENDER_CLIP_SLAB = 1000.0
BACKGROUND_COLOR = "white"

# Colors.
ORIGINAL_FULL_COLOR = "original_full_green"
ORIGINAL_FULL_RGB = (0.00, 0.55, 0.05)
RECONSTRUCTED_COLOR = "reconstructed_red"
RECONSTRUCTED_RGB = (0.82, 0.00, 0.00)

# Cartoon settings.
CA_TUBE_RADIUS = 0.45
RENDER_OBJECT_PREFIX = "original_full_layers_render"

# Make `run render_original_full_layers.py` render immediately in PyMOL.
AUTO_RUN_IN_PYMOL = True


@dataclass(frozen=True)
class RenderItem:
    protein_id: str
    label: str
    pdb_path: Path
    output_path: Path
    is_original_full: bool


def original_full_pdb_path(protein_id: str) -> Path:
    return INPUT_DIR / ORIGINAL_FULL_PDB_PATH_TEMPLATE.format(protein_id=protein_id)


def reconstructed_pdb_path(protein_id: str, layer: int) -> Path:
    return INPUT_DIR / RECONSTRUCTED_PDB_PATH_TEMPLATE.format(
        protein_id=protein_id,
        layer=layer,
    )


def collect_render_items() -> list[RenderItem]:
    protein_output_dir = OUTPUT_DIR / PROTEIN_ID
    items = [
        RenderItem(
            protein_id=PROTEIN_ID,
            label="original_full",
            pdb_path=original_full_pdb_path(PROTEIN_ID),
            output_path=protein_output_dir / f"{PROTEIN_ID}_original_full.png",
            is_original_full=True,
        )
    ]

    for layer in LAYERS:
        items.append(
            RenderItem(
                protein_id=PROTEIN_ID,
                label=f"layer{layer:02d}",
                pdb_path=reconstructed_pdb_path(PROTEIN_ID, layer),
                output_path=protein_output_dir / f"{PROTEIN_ID}_layer{layer:02d}.png",
                is_original_full=False,
            )
        )

    missing = [str(item.pdb_path) for item in items if not item.pdb_path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing PDB inputs for {PROTEIN_ID}:\n" + "\n".join(missing)
        )

    return items


def object_name_for(item: RenderItem) -> str:
    safe_label = item.label.replace("-", "_")
    safe_protein_id = item.protein_id.replace("-", "_")
    return f"{RENDER_OBJECT_PREFIX}_{safe_protein_id}_{safe_label}"


def required_view_name(protein_id: str) -> str:
    return f"{protein_id}_view"


def recall_required_view(cmd, view_name: str) -> None:
    try:
        cmd.view(view_name, "recall")
    except Exception as exc:
        raise RuntimeError(
            f"Required PyMOL view '{view_name}' could not be recalled. "
            "Store it in the active session before running this script."
        ) from exc


def prepare_view(cmd, protein_id: str, view_name: str) -> None:
    if USE_SAVED_VIEWS:
        recall_required_view(cmd, view_name)
        return

    raise RuntimeError(
        "This renderer expects a fixed saved PyMOL view. Set "
        f"USE_SAVED_VIEWS=True and store '{view_name}' before running."
    )


def set_if_available(cmd, name: str, value) -> None:
    try:
        cmd.set(name, value)
    except Exception:
        print(f"Warning: PyMOL setting '{name}' is not available in this build.")


def apply_scene_settings(cmd) -> None:
    cmd.bg_color(BACKGROUND_COLOR)
    cmd.set("opaque_background", 1)
    cmd.set("ray_opaque_background", 1)
    set_if_available(cmd, "auto_zoom", 0)
    set_if_available(cmd, "orthoscopic", 1 if FORCE_ORTHOSCOPIC else 0)
    cmd.set_color(ORIGINAL_FULL_COLOR, ORIGINAL_FULL_RGB)
    cmd.set_color(RECONSTRUCTED_COLOR, RECONSTRUCTED_RGB)

    if RAY_TRACE:
        set_if_available(cmd, "ray_shadow", 1 if ENABLE_RAY_SHADOWS else 0)
        set_if_available(cmd, "ray_shadows", 1 if ENABLE_RAY_SHADOWS else 0)
        set_if_available(cmd, "ambient", 0.50)
        set_if_available(cmd, "direct", 0.50)
        set_if_available(cmd, "specular", 0.15)
        set_if_available(cmd, "depth_cue", 1 if ENABLE_DEPTH_FADING else 0)
        set_if_available(cmd, "ray_trace_fog", 1 if ENABLE_DEPTH_FADING else 0)
        set_if_available(cmd, "fog_start", DEPTH_FADING_START)


def reset_render_clipping(cmd, object_name: str) -> None:
    try:
        cmd.clip("atoms", RENDER_CLIP_BUFFER, object_name)
    except Exception:
        cmd.clip("slab", RENDER_CLIP_SLAB)
    cmd.clip("slab", RENDER_CLIP_SLAB)


def apply_original_full_style(cmd, object_name: str) -> None:
    cmd.hide("everything", object_name)
    cmd.set("cartoon_trace_atoms", 0, object_name)
    cmd.dss(object_name)
    cmd.cartoon("automatic", object_name)
    cmd.show("cartoon", object_name)
    cmd.color(ORIGINAL_FULL_COLOR, object_name)
    cmd.rebuild(object_name)


def apply_ca_trace_style(cmd, object_name: str) -> None:
    cmd.hide("everything", object_name)
    cmd.set("cartoon_trace_atoms", 1, object_name)
    cmd.set("cartoon_tube_radius", CA_TUBE_RADIUS, object_name)
    cmd.cartoon("tube", object_name)
    cmd.show("cartoon", object_name)
    cmd.color(RECONSTRUCTED_COLOR, object_name)
    cmd.rebuild(object_name)


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


def render_item(cmd, item: RenderItem, view_name: str) -> None:
    object_name = object_name_for(item)

    print(f"Rendering {item.protein_id} {item.label} -> {item.output_path}")
    cmd.delete(object_name)
    cmd.load(str(item.pdb_path), object_name)
    cmd.disable("all")
    cmd.enable(object_name)

    if item.is_original_full:
        apply_original_full_style(cmd, object_name)
    else:
        apply_ca_trace_style(cmd, object_name)

    recall_required_view(cmd, view_name)
    reset_render_clipping(cmd, object_name)
    cmd.rebuild(object_name)
    cmd.refresh()
    item.output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd.png(
        str(item.output_path),
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        ray=1 if RAY_TRACE else 0,
    )
    cmd.delete(object_name)


def render_original_full_layers() -> None:
    try:
        from pymol import cmd
    except ImportError as exc:
        raise SystemExit(
            "This script must be run inside PyMOL, for example: "
            "run chimera_files/render_original_full_layers.py"
        ) from exc

    items = collect_render_items()
    enabled_objects = cmd.get_names("objects", enabled_only=1)
    view_name = required_view_name(PROTEIN_ID)

    try:
        print(f"Using project root: {PROJECT_ROOT}")
        print(f"Writing PNGs under: {OUTPUT_DIR / PROTEIN_ID}")
        apply_scene_settings(cmd)
        cleanup_render_objects(cmd)
        prepare_view(cmd, PROTEIN_ID, view_name)
        for item in items:
            render_item(cmd, item, view_name)
        print(f"Finished rendering {len(items)} PNGs.")
    finally:
        cleanup_render_objects(cmd)
        restore_enabled_objects(cmd, enabled_objects)


try:
    from pymol import cmd as _pymol_cmd
except ImportError:
    _pymol_cmd = None
else:
    _pymol_cmd.extend("render_original_full_layers", render_original_full_layers)


if (_pymol_cmd is not None and AUTO_RUN_IN_PYMOL) or __name__ == "__main__":
    render_original_full_layers()
