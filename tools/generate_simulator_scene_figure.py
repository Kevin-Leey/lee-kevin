"""Replay frozen trajectories and render the simulator-scene paper figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER_FIGURES = ROOT / "paper" / "figures"
DOC_FIGURES = ROOT / "docs" / "pipeline_figures"
HIGHWAY_ROOT = ROOT / "results" / "highway_result" / "formal_run" / "2026-07-19"
METADRIVE_ROOT = ROOT / "results" / "metadrive_result" / "formal_run" / "2026-07-19"


# Each panel comes from a frozen RGD evaluation trace. The selected frames make
# road geometry and the local traffic configuration readable at publication size.
SCENE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "kind": "highway",
        "scenario": "highway",
        "seed": 300,
        "frame": 20,
        "label": "(a) Highway",
    },
    {
        "kind": "highway",
        "scenario": "merge",
        "seed": 300,
        "frame": 7,
        "label": "(b) Merge",
    },
    {
        "kind": "highway",
        "scenario": "roundabout",
        "seed": 300,
        "frame": 8,
        "label": "(c) Roundabout",
    },
    {
        "kind": "highway",
        "scenario": "intersection",
        "seed": 300,
        "frame": 7,
        "label": "(d) Intersection",
    },
    {
        "kind": "metadrive",
        "scenario": "highway",
        "seed": 300,
        "frame": 35,
        "label": "(e) Highway",
    },
    {
        "kind": "metadrive",
        "scenario": "merge",
        "seed": 300,
        "frame": 60,
        "label": "(f) Merge",
        # Keep the frozen state while centering the actual in-ramp junction.
        "view": {
            "camera_position": (95.0, 0.0),
            "scaling": 10.0,
            "target_agent_heading_up": False,
        },
    },
    {
        "kind": "metadrive",
        "scenario": "roundabout",
        "seed": 300,
        "frame": 470,
        "label": "(g) Roundabout",
        # A road-centric view retains the complete circular geometry and ego state.
        "view": {
            "camera_position": (88.0, 8.0),
            "scaling": 7.0,
            "target_agent_heading_up": False,
        },
    },
    {
        "kind": "metadrive",
        "scenario": "intersection",
        "seed": 300,
        "frame": 44,
        "label": "(h) Intersection",
    },
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_replay_state(
    env: Any,
    record: Dict[str, Any],
    context: str,
    tolerance: float = 1e-5,
) -> Dict[str, float]:
    vehicle = env.unwrapped.vehicle
    expected_position = np.array(
        [float(record["position_x"]), float(record["position_y"])], dtype=float
    )
    observed_position = np.asarray(vehicle.position, dtype=float)[:2]
    position_error = float(np.linalg.norm(observed_position - expected_position))
    speed_error = abs(float(vehicle.speed) - float(record["speed"]))
    if position_error > tolerance or speed_error > tolerance:
        raise RuntimeError(
            f"{context} replay drift: position={position_error:.3e} m, "
            f"speed={speed_error:.3e} m/s"
        )
    return {
        "position_error_m": position_error,
        "speed_error_mps": speed_error,
        "tolerance": tolerance,
    }


def _highway_run(scenario: str, seed: int) -> Path:
    return (
        HIGHWAY_ROOT
        / f"cross-zero-v2-qwen3-hw-{scenario}"
        / "rgd_fixed_policy"
        / scenario
        / f"seed_{seed}"
    )


def _metadrive_run(scenario: str, seed: int) -> Path:
    return (
        METADRIVE_ROOT
        / f"cross-zero-v2-qwen3-md-{scenario}"
        / "rgd_fixed_policy"
        / f"metadrive_{scenario}"
        / f"seed_{seed}"
    )


def _scene_worker_name(spec: Dict[str, Any]) -> str:
    return f"{spec['kind']}-{spec['scenario']}"


def _scene_spec(worker: str) -> Dict[str, Any]:
    for spec in SCENE_SPECS:
        if _scene_worker_name(spec) == worker:
            return dict(spec)
    raise ValueError(f"unknown scene worker: {worker}")


def _highway_frame(
    *, scenario: str, seed: int, frame: int, scratch: Path
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from dilu.runtime_episode_setup import create_episode_env

    run = _highway_run(scenario, seed)
    snapshot_path = run / "experiment_snapshot.json"
    trace_path = run / f"ep_{seed}" / f"{scenario}_{seed}_physical_frames.json"
    cfg = dict(_load_json(snapshot_path)["config"])
    # Historical manifests encode an unspecified spacing as null. Removing the
    # key restores the environment default used by the frozen run.
    if cfg.get("ego_spacing") is None:
        cfg.pop("ego_spacing", None)
    cfg.update(
        {
            "render_mode": "rgb_array",
            "enable_physical_metrics": False,
            "enable_reasoning_recording": False,
        }
    )
    trace = list(_load_json(trace_path)["frames"])
    if not 0 <= frame < len(trace):
        raise ValueError(
            f"highway-env {scenario} frame {frame} is outside trace length {len(trace)}"
        )

    env, _, _, _, _, close_env = create_episode_env(
        seed, cfg, str(scratch / scenario), [seed]
    )
    try:
        env.unwrapped.config.update(
            {
                "screen_width": 1600,
                "screen_height": 650,
                "scaling": 14,
                "centering_position": [0.34, 0.5],
                "show_trajectories": False,
            }
        )
        for index in range(frame):
            env.step(int(trace[index]["action_id"]))
        replay_error = _assert_replay_state(
            env, trace[frame], f"highway-env {scenario} frame {frame}"
        )
        image = np.asarray(env.render()).copy()
    finally:
        if close_env:
            env.close()

    return image, {
        "simulator": "highway-env",
        "scenario": scenario,
        "seed": seed,
        "frame": frame,
        "snapshot": str(snapshot_path.relative_to(ROOT)),
        "trace": str(trace_path.relative_to(ROOT)),
        "snapshot_sha256": _sha256(snapshot_path),
        "trace_sha256": _sha256(trace_path),
        "replay_error": replay_error,
    }


def _metadrive_frame(
    *,
    scenario: str,
    seed: int,
    frame: int,
    scratch: Path,
    view: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from dilu.runtime_episode_setup import create_episode_env

    run = _metadrive_run(scenario, seed)
    env_name = f"metadrive_{scenario}"
    snapshot_path = run / "experiment_snapshot.json"
    trace_path = run / f"ep_{seed}" / f"{env_name}_{seed}_physical_frames.json"
    cfg = dict(_load_json(snapshot_path)["config"])
    cfg.update(
        {
            "render_mode": "none",
            "enable_physical_metrics": False,
            "enable_reasoning_recording": False,
        }
    )
    trace = list(_load_json(trace_path)["frames"])
    if not 0 <= frame < len(trace):
        raise ValueError(
            f"MetaDrive {scenario} frame {frame} is outside trace length {len(trace)}"
        )

    view_config = dict(view or {})
    camera_scaling = float(view_config.get("scaling", 20.0))
    target_agent_heading_up = bool(
        view_config.get("target_agent_heading_up", True)
    )
    camera_position_value = view_config.get("camera_position")
    camera_position = None
    if camera_position_value is not None:
        if len(camera_position_value) != 2:
            raise ValueError("MetaDrive camera_position must contain two coordinates")
        camera_position = tuple(float(value) for value in camera_position_value)

    env, _, _, _, _, _ = create_episode_env(seed, cfg, str(scratch / scenario), [seed])
    try:
        for index in range(frame):
            env.step(int(trace[index]["action_id"]))
        record = trace[frame]
        pre_position = np.asarray(env.vehicle.position, dtype=float)[:2]
        target_position = np.asarray(
            [record["position_x"], record["position_y"]], dtype=float
        )
        pre_reconstruction_error = {
            "position_error_m": float(np.linalg.norm(pre_position - target_position)),
            "speed_error_mps": abs(float(env.vehicle.speed) - float(record["speed"])),
        }

        traffic_manager = getattr(env.engine, "traffic_manager", None)
        traffic = [
            vehicle
            for vehicle in list(
                getattr(traffic_manager, "vehicles", [])
                if traffic_manager is not None
                else []
            )
            if vehicle is not env.vehicle
        ]
        snapshots = list(record.get("neighbor_snapshots", []) or [])
        if len(traffic) < len(snapshots):
            raise RuntimeError(
                f"MetaDrive {scenario} has {len(traffic)} renderable traffic vehicles "
                f"for {len(snapshots)} recorded neighbors"
            )

        ego_heading = float(getattr(env.vehicle, "heading_theta", 0.0) or 0.0)
        env.vehicle.set_position(target_position.tolist())
        env.vehicle.set_heading_theta(ego_heading)
        env.vehicle.set_velocity(
            [math.cos(ego_heading), math.sin(ego_heading)],
            float(record["speed"]),
            in_local_frame=False,
        )
        for vehicle, snapshot in zip(traffic, snapshots):
            heading = float(snapshot["heading"])
            position = target_position + np.asarray(
                [snapshot["rel_x"], snapshot["rel_y"]], dtype=float
            )
            vehicle.set_position(position.tolist())
            vehicle.set_heading_theta(heading)
            vehicle.set_velocity(
                [math.cos(heading), math.sin(heading)],
                float(snapshot["speed"]),
                in_local_frame=False,
            )

        # The recording stores all task-relevant neighbors. Extra regenerated
        # actors are moved outside the rendering window without modifying them.
        for offset, vehicle in enumerate(traffic[len(snapshots) :], start=1):
            vehicle.set_position(
                (target_position + np.asarray([500.0 + 5.0 * offset, 500.0])).tolist()
            )

        # Panda3D quantizes the velocity setter at the fifth decimal place.
        # A 1e-4 tolerance remains far below the displayed-state resolution.
        replay_error = _assert_replay_state(
            env,
            record,
            f"MetaDrive {scenario} reconstructed state",
            tolerance=1e-4,
        )
        replay_error["pre_reconstruction_position_error_m"] = pre_reconstruction_error[
            "position_error_m"
        ]
        replay_error["pre_reconstruction_speed_error_mps"] = pre_reconstruction_error[
            "speed_error_mps"
        ]
        replay_error["recorded_neighbors_rendered"] = len(snapshots)
        replay_error["render_mode"] = "recorded_state_reconstruction"

        observed_distances = [
            float(
                np.linalg.norm(
                    np.asarray(vehicle.position, dtype=float)[:2]
                    - np.asarray(env.vehicle.position, dtype=float)[:2]
                )
            )
            for vehicle in traffic
            if vehicle is not env.vehicle and hasattr(vehicle, "position")
        ]
        expected_min_distance = float(record["min_distance_all"])
        if observed_distances and np.isfinite(expected_min_distance):
            min_distance_error = abs(min(observed_distances) - expected_min_distance)
            if min_distance_error > 1e-5:
                raise RuntimeError(
                    f"MetaDrive {scenario} traffic replay drift: "
                    f"nearest-distance error={min_distance_error:.3e} m"
                )
            replay_error["min_distance_error_m"] = min_distance_error

        render_options: Dict[str, Any] = {
            "mode": "top_down",
            "window": False,
            "screen_size": (1600, 1000),
            "film_size": (4000, 3000),
            "scaling": camera_scaling,
            "target_agent_heading_up": target_agent_heading_up,
            "num_stack": 1,
            "draw_contour": True,
        }
        if camera_position is not None:
            render_options["camera_position"] = camera_position
        image = np.asarray(env.render(**render_options)).copy()
    finally:
        env.close()

    camera_metadata: Dict[str, Any] = {
        "scaling": camera_scaling,
        "target_agent_heading_up": target_agent_heading_up,
    }
    if camera_position is not None:
        camera_metadata["camera_position"] = list(camera_position)

    return image, {
        "simulator": "MetaDrive",
        "scenario": scenario,
        "seed": seed,
        "frame": frame,
        "snapshot": str(snapshot_path.relative_to(ROOT)),
        "trace": str(trace_path.relative_to(ROOT)),
        "snapshot_sha256": _sha256(snapshot_path),
        "trace_sha256": _sha256(trace_path),
        "replay_error": replay_error,
        "camera": camera_metadata,
    }


def _prepare_panel(image: np.ndarray, *, highway: bool) -> np.ndarray:
    """Crop to a common printed aspect while preserving native geometry."""
    rgb = np.asarray(image)[..., :3]
    target_aspect = 1.42
    height, width = rgb.shape[:2]
    if highway:
        crop_width = min(width, int(round(height * target_aspect)))
        red = rgb[..., 0].astype(int)
        green = rgb[..., 1].astype(int)
        blue = rgb[..., 2].astype(int)
        ego_mask = (green > 180) & (green > red + 80) & (green > blue + 80)
        ego_columns = np.where(ego_mask)[1]
        ego_x = int(np.median(ego_columns)) if len(ego_columns) else width // 2
        x0 = ego_x - int(round(0.20 * crop_width))
        x0 = max(0, min(width - crop_width, x0))
        return rgb[:, x0 : x0 + crop_width]

    current_aspect = width / max(height, 1)
    if current_aspect > target_aspect:
        crop_width = int(round(height * target_aspect))
        x0 = max(0, (width - crop_width) // 2)
        return rgb[:, x0 : x0 + crop_width]
    crop_height = int(round(width / target_aspect))
    y0 = max(0, (height - crop_height) // 2)
    return rgb[y0 : y0 + crop_height, :]


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.linewidth": 0.55,
            "svg.fonttype": "none",
            "svg.image_inline": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(
    images: Iterable[np.ndarray], specs: Iterable[Dict[str, Any]], output_stem: Path
) -> None:
    _configure_plotting()
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 3.45))
    for axis, raw_image, spec in zip(axes.flat, images, specs):
        image = _prepare_panel(raw_image, highway=spec["kind"] == "highway")
        axis.imshow(image, interpolation="none")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#4c4c4c")
            spine.set_linewidth(0.55)
        axis.text(
            0.5,
            -0.105,
            spec["label"],
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.3,
            color="#202020",
        )
    fig.subplots_adjust(
        left=0.012,
        right=0.988,
        top=0.99,
        bottom=0.08,
        wspace=0.025,
        hspace=0.34,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    png_path = output_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Matplotlib's PDF backend can misplace independently cropped raster panels.
    # Use the inspected high-resolution composite as the paper companion instead.
    with Image.open(png_path) as image:
        image.convert("RGB").save(
            output_stem.with_suffix(".pdf"), "PDF", resolution=600.0
        )


def _run_worker(worker: str, output: Path, env: Dict[str, str], attempts: int) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        worker,
        "--worker-output",
        str(output),
    ]
    last_stderr = ""
    for _ in range(attempts):
        output.unlink(missing_ok=True)
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0 and output.is_file():
            return
        last_stderr = result.stderr
    raise RuntimeError(
        f"{worker} replay did not match its frozen trace after {attempts} attempts:\n"
        f"{last_stderr}"
    )


def _render_worker(spec: Dict[str, Any], output: Path) -> None:
    if spec["kind"] == "highway":
        image, metadata = _highway_frame(
            scenario=str(spec["scenario"]),
            seed=int(spec["seed"]),
            frame=int(spec["frame"]),
            scratch=output.parent,
        )
    else:
        image, metadata = _metadrive_frame(
            scenario=str(spec["scenario"]),
            seed=int(spec["seed"]),
            frame=int(spec["frame"]),
            scratch=output.parent,
            view=spec.get("view"),
        )
    np.savez_compressed(
        output,
        image=image,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=True)),
    )


def main() -> None:
    worker_names = tuple(_scene_worker_name(spec) for spec in SCENE_SPECS)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-stem", type=Path, default=PAPER_FIGURES / "fig_simulator_scenes"
    )
    parser.add_argument("--worker", choices=worker_names)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument(
        "--frame",
        type=int,
        help="Override a worker's representative frame for visual review.",
    )
    args = parser.parse_args()

    if args.worker:
        if args.worker_output is None:
            raise ValueError("--worker requires --worker-output")
        worker_output = args.worker_output.resolve()
        worker_output.parent.mkdir(parents=True, exist_ok=True)
        spec = _scene_spec(args.worker)
        if args.frame is not None:
            spec["frame"] = int(args.frame)
        _render_worker(spec, worker_output)
        return

    output_stem = args.output_stem.resolve()
    (ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sim_scene_", dir=ROOT / "tmp") as scratch_name:
        scratch = Path(scratch_name)
        bundles: List[Path] = []
        for spec in SCENE_SPECS:
            worker = _scene_worker_name(spec)
            bundle = scratch / f"{worker}_frame.npz"
            worker_env = dict(os.environ)
            worker_env["SDL_VIDEODRIVER"] = "windows" if spec["kind"] == "highway" else "dummy"
            _run_worker(
                worker,
                bundle,
                worker_env,
                attempts=4 if spec["kind"] == "highway" else 10,
            )
            bundles.append(bundle)

        images: List[np.ndarray] = []
        metadata: List[Dict[str, Any]] = []
        for bundle in bundles:
            with np.load(bundle, allow_pickle=False) as data:
                images.append(data["image"])
                metadata.append(json.loads(str(data["metadata"])))
        _save_figure(images, SCENE_SPECS, output_stem)

    DOC_FIGURES.mkdir(parents=True, exist_ok=True)
    doc_svg = DOC_FIGURES / output_stem.with_suffix(".svg").name
    shutil.copy2(output_stem.with_suffix(".svg"), doc_svg)
    manifest = {
        "figure": str(output_stem.relative_to(ROOT)),
        "svg_copy": str(doc_svg.relative_to(ROOT)),
        "panels": metadata,
    }
    manifest_path = output_stem.with_name(output_stem.name + "_manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
