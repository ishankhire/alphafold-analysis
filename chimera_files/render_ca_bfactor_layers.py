#!/usr/bin/env python3
"""
Batch-render CA-only reconstructed structures in PyMOL.

Usage from an active PyMOL session with named views already stored:

    PyMOL> run chimera_files/render_ca_bfactor_layers.py

In full-batch mode, each protein should have a saved view named
"{protein_id}_view" before this script runs. In the current test mode, the
script creates that view once from the reference structure so camera/framing
problems can be isolated from rendering problems.
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
REFERENCE_PDB_PATH_TEMPLATE = (
    "{protein_id}/reference_ca/{protein_id}_reference_ca.pdb"
)
RECONSTRUCTED_PDB_PATH_TEMPLATE = (
    "{protein_id}/reconstructed_ca_layers/{protein_id}_reconstruct_L{layer:02d}.pdb"
)

# Render subset.
# Testing mode: render only 7b3a_A reference plus layer 47.
#
# For the broader batch later, restore:
#     MAX_PROTEINS_TO_RENDER = 10
#     PROTEIN_IDS = None
#     LAYERS = (0, 10, 15, 22, 30, 40, 47)
#     USE_SAVED_VIEWS = True
#     RAY_TRACE = False
MAX_PROTEINS_TO_RENDER = 10
PROTEIN_IDS: tuple[str, ...] | None = ("7b3a_A",)
LAYERS = (47,)
USE_SAVED_VIEWS = False

# Image settings.
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 1200
RAY_TRACE = True
BACKGROUND_COLOR = "white"

# Distance-to-final B-factor thresholds.
LOW_DISTANCE_THRESHOLD = 2.0
HIGH_DISTANCE_THRESHOLD = 4.0
SELECTION_OFFSET = 0.001

# Colors.
REFERENCE_COLOR = "blue"
CLOSE_COLOR = "red"
MIDDLE_COLOR = "orange"
FAR_COLOR = "tv_yellow"

# CA-trace cartoon settings.
CARTOON_TUBE_RADIUS = 0.45
STICK_TRACE_RADIUS = 0.30
REFERENCE_VIEW_BUFFER = 10.0
REFERENCE_VIEW_SLAB = 250.0
RENDER_OBJECT_PREFIX = "ca_bfactor_render"

# Make `run render_ca_bfactor_layers.py` render immediately in PyMOL.
AUTO_RUN_IN_PYMOL = True


@dataclass(frozen=True)
class RenderItem:
    protein_id: str
    label: str
    pdb_path: Path
    output_path: Path
    is_reference: bool


def discover_proteins() -> list[str]:
    if PROTEIN_IDS is not None:
        return list(PROTEIN_IDS)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    proteins = sorted(path.name for path in INPUT_DIR.iterdir() if path.is_dir())
    selected = proteins[:MAX_PROTEINS_TO_RENDER]
    if len(selected) < MAX_PROTEINS_TO_RENDER:
        raise FileNotFoundError(
            f"Expected at least {MAX_PROTEINS_TO_RENDER} protein directories in "
            f"{INPUT_DIR}, found {len(selected)}."
        )
    return selected


def reference_pdb_path(protein_id: str) -> Path:
    return INPUT_DIR / REFERENCE_PDB_PATH_TEMPLATE.format(protein_id=protein_id)


def reconstructed_pdb_path(protein_id: str, layer: int) -> Path:
    return INPUT_DIR / RECONSTRUCTED_PDB_PATH_TEMPLATE.format(
        protein_id=protein_id,
        layer=layer,
    )


def collect_render_items() -> dict[str, list[RenderItem]]:
    items_by_protein: dict[str, list[RenderItem]] = {}

    for protein_id in discover_proteins():
        protein_output_dir = OUTPUT_DIR / protein_id
        items = [
            RenderItem(
                protein_id=protein_id,
                label="reference_ca",
                pdb_path=reference_pdb_path(protein_id),
                output_path=protein_output_dir / f"{protein_id}_reference_ca.png",
                is_reference=True,
            )
        ]

        for layer in LAYERS:
            items.append(
                RenderItem(
                    protein_id=protein_id,
                    label=f"layer{layer:02d}",
                    pdb_path=reconstructed_pdb_path(protein_id, layer),
                    output_path=protein_output_dir / f"{protein_id}_layer{layer:02d}.png",
                    is_reference=False,
                )
            )

        missing = [str(item.pdb_path) for item in items if not item.pdb_path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing PDB inputs for {protein_id}:\n" + "\n".join(missing)
            )

        items_by_protein[protein_id] = items

    return items_by_protein


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


def create_view_from_reference(cmd, protein_id: str, view_name: str) -> None:
    object_name = f"{RENDER_OBJECT_PREFIX}_{protein_id}_view_reference"
    pdb_path = reference_pdb_path(protein_id)

    print(f"Creating {view_name} from reference: {pdb_path}")
    cmd.delete(object_name)
    cmd.load(str(pdb_path), object_name)
    cmd.disable("all")
    cmd.enable(object_name)
    apply_ca_trace_style(cmd, object_name)
    cmd.color(REFERENCE_COLOR, object_name)
    cmd.orient(object_name)
    cmd.zoom(object_name, buffer=REFERENCE_VIEW_BUFFER, complete=1)
    cmd.clip("slab", REFERENCE_VIEW_SLAB)
    cmd.view(view_name, "store")
    cmd.delete(object_name)


def prepare_view(cmd, protein_id: str, view_name: str) -> None:
    if USE_SAVED_VIEWS:
        recall_required_view(cmd, view_name)
    else:
        create_view_from_reference(cmd, protein_id, view_name)


def apply_scene_settings(cmd) -> None:
    cmd.bg_color(BACKGROUND_COLOR)
    cmd.set("opaque_background", 1)
    cmd.set("ray_opaque_background", 1)
    cmd.set("cartoon_trace_atoms", 1)
    cmd.set("cartoon_tube_radius", CARTOON_TUBE_RADIUS)
    cmd.set("stick_radius", STICK_TRACE_RADIUS)


def apply_ca_trace_style(cmd, object_name: str) -> None:
    cmd.hide("everything", object_name)
    cmd.set("cartoon_trace_atoms", 1, object_name)
    cmd.set("cartoon_tube_radius", CARTOON_TUBE_RADIUS, object_name)
    cmd.set("stick_radius", STICK_TRACE_RADIUS, object_name)
    cmd.cartoon("tube", object_name)
    cmd.show("cartoon", object_name)
    cmd.show("sticks", object_name)
    cmd.rebuild(object_name)


def color_reconstructed_by_bfactor(cmd, object_name: str) -> None:
    low_overlap = LOW_DISTANCE_THRESHOLD - SELECTION_OFFSET
    high_overlap = HIGH_DISTANCE_THRESHOLD - SELECTION_OFFSET

    far_selection = f"{object_name} and b > {high_overlap:.3f}"
    middle_selection = (
        f"{object_name} and b > {low_overlap:.3f} "
        f"and b < {HIGH_DISTANCE_THRESHOLD:.1f}"
    )
    close_selection = f"{object_name} and b < {LOW_DISTANCE_THRESHOLD:.1f}"

    cmd.color(FAR_COLOR, far_selection)
    cmd.color(MIDDLE_COLOR, middle_selection)
    cmd.color(CLOSE_COLOR, close_selection)


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

    apply_ca_trace_style(cmd, object_name)
    if item.is_reference:
        cmd.color(REFERENCE_COLOR, object_name)
    else:
        color_reconstructed_by_bfactor(cmd, object_name)

    recall_required_view(cmd, view_name)
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


def render_ca_bfactor_layers() -> None:
    try:
        from pymol import cmd
    except ImportError as exc:
        raise SystemExit(
            "This script must be run inside PyMOL, for example: "
            "run chimera_files/render_ca_bfactor_layers.py"
        ) from exc

    items_by_protein = collect_render_items()
    enabled_objects = cmd.get_names("objects", enabled_only=1)

    try:
        print(f"Using project root: {PROJECT_ROOT}")
        print(f"Writing PNGs under: {OUTPUT_DIR}")
        apply_scene_settings(cmd)
        cleanup_render_objects(cmd)
        for protein_id, items in items_by_protein.items():
            view_name = required_view_name(protein_id)
            prepare_view(cmd, protein_id, view_name)
            for item in items:
                render_item(cmd, item, view_name)
        print(f"Finished rendering {sum(len(items) for items in items_by_protein.values())} PNGs.")
    finally:
        cleanup_render_objects(cmd)
        restore_enabled_objects(cmd, enabled_objects)


try:
    from pymol import cmd as _pymol_cmd
except ImportError:
    _pymol_cmd = None
else:
    _pymol_cmd.extend("render_ca_bfactor_layers", render_ca_bfactor_layers)


if (_pymol_cmd is not None and AUTO_RUN_IN_PYMOL) or __name__ == "__main__":
    render_ca_bfactor_layers()
