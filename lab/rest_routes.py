"""FastAPI REST routes for the lab server."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from simulations.zebrafish.environment import AquaticArenaEnvironment
from simulations.zebrafish.neuron_mapping import ZebrafishNervousSystem

from lab.introspect.body_introspect import build_body_view
from lab.introspect.connectome_introspect import build_connectome_view
from lab.parameters import ParameterRegistry
from lab.parameters.applicators import apply_patches
from lab.sim_runtime import LabSimRuntime
from lab.video_pipeline import BackendVideoPipeline, safe_upload_name


class TransportAction(BaseModel):
    action: Literal["play", "pause", "step"]


class PacingBody(BaseModel):
    """Optional wall-clock pacing for the lab sim thread (milliseconds).

    ``0`` means no added delay for that axis (run as fast as the CPU allows).
    """

    real_ms_per_physics_step: float | None = None
    real_ms_per_neural_tick: float | None = None


class Patch(BaseModel):
    path: str
    value: Any


class PatchBody(BaseModel):
    patches: list[Patch] = Field(default_factory=list)


class NeuronParamPatch(BaseModel):
    """Patch neuron parameters, runtime state, or synaptic terminals.

    * **Params scalars** — ``field`` is ``r_base``, ``c``, … (see
      ``_NEURON_SCALAR_FIELDS``); ``index`` / ``subfield`` ignored.
    * **Params vectors** — ``field`` is ``gamma`` / ``w_r`` / ``w_b`` /
      ``w_tref``: either replace the whole vector (``value`` is a list,
      ``index`` is None) or set one slot (``index`` = element, ``value`` =
      number).
    * **Runtime** — ``field`` is ``S``, ``O``, ``r``, ``b``, ``t_ref``,
      ``F_avg``, ``t_last_fire``; scalar ``value`` on the live neuron.
    * **``M_vector``** — same pattern as params vectors but on the neuron
      object (length = ``num_neuromodulators``).
    * **Postsynaptic** — ``field`` = ``postsynaptic``, ``index`` = synapse
      slot id; ``subfield`` one of ``info``, ``plast``, ``potential``,
      ``adapt`` (``adapt`` requires ``vec_index``).
    * **Presynaptic** — ``field`` = ``presynaptic``, ``index`` = terminal id;
      ``subfield`` ``u_o_info``, ``u_i_retro``, or ``mod`` (``mod`` requires
      ``vec_index``).
    """

    field: str
    value: Any
    index: int | None = None
    subfield: str | None = None
    vec_index: int | None = None


class NeuronPatchBody(BaseModel):
    patches: list[NeuronParamPatch] = Field(default_factory=list)


class BodyPatch(BaseModel):
    """One live mjModel field edit.

    ``target`` selects the MuJoCo table (``joint``, ``actuator``, ``body``,
    ``pair`` or ``opt``); ``id`` indexes into that table and is omitted for
    ``opt``. ``field`` names the logical property; ``index`` is used when the
    property is a vector (forcerange, gear, inertia, gravity, friction).
    """

    target: Literal["joint", "actuator", "body", "pair", "opt"]
    field: str
    value: Any
    id: int | None = None
    index: int | None = None


class BodyPatchBody(BaseModel):
    patches: list[BodyPatch] = Field(default_factory=list)


class VideoStimulusConfigBody(BaseModel):
    enabled: bool | None = None
    gain: float | None = None
    file_name: str | None = None


class VideoStimulusFrameBody(BaseModel):
    file_name: str = ""
    frame_index: int = 0
    video_time_s: float = 0.0
    sample_hz: float = 15.0
    visual_left: float = 0.0
    visual_right: float = 0.0
    optic_flow_left: float = 0.0
    optic_flow_right: float = 0.0
    lateral_line_left: float = 0.0
    lateral_line_right: float = 0.0
    visual_up: float = 0.0
    visual_down: float = 0.0
    light_level: float = 0.0
    startle: float = 0.0
    motion_energy: float = 0.0
    asymmetry: float = 0.0
    action_kick: float = 0.0
    action_force: float = 0.0
    action_side_score: float = 0.0
    action_kick_score: float = 0.0
    action_confidence: float = 0.0
    action_source: str = ""
    action_branch: str = ""
    action_bout_type: str = "none"
    tail_frequency_hz: float = 0.0
    tail_amplitude: float = 0.0
    tail_targets: list[float] = Field(default_factory=list)
    zapbench_row: int = 0
    enabled: bool | None = None


class BackendVideoFrameBody(BaseModel):
    file_name: str = ""
    frame_index: int = 0
    video_time_s: float = 0.0
    sample_hz: float = 15.0
    enabled: bool | None = None


class CalciumStimulusConfigBody(BaseModel):
    enabled: bool | None = None
    gain: float | None = None
    source: str | None = None


class CalciumActionFrameBody(BaseModel):
    source: str = ""
    row: int = 0
    frame_index: int = 0
    calcium_time_s: float = 0.0
    kick: float = 0.0
    side_score: float = 0.0
    force: float = 0.0
    kick_score: float = 0.0
    confidence: float = 0.0
    tail_phase: float = 0.0
    enabled: bool | None = None


class CalciumReplayBody(BaseModel):
    enabled: bool | None = None
    gain: float | None = None
    condition: str | None = None
    loop: bool | None = None
    replay_path: str | None = None


_NEURON_SCALAR_FIELDS: dict[str, type] = {
    "r_base": float,
    "b_base": float,
    "c": int,
    "lambda_param": float,
    "p": float,
    "eta_post": float,
    "eta_retro": float,
    "delta_decay": float,
    "beta_avg": float,
}
_NEURON_VECTOR_FIELDS = ("gamma", "w_r", "w_b", "w_tref")
_NEURON_RUNTIME_FIELDS: dict[str, type] = {
    "S": float,
    "O": float,
    "r": float,
    "b": float,
    "t_ref": float,
    "F_avg": float,
    "t_last_fire": float,
}

_ZAPBENCH_CONDITION_NAMES = (
    "gain",
    "dots",
    "flash",
    "taxis",
    "turning",
    "position",
    "open loop",
    "rotation",
    "dark",
)
_ZAPBENCH_CONDITION_OFFSETS = (0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879)
_ZAPBENCH_STIMULUS_COLUMNS: tuple[dict[str, Any], ...] = (
    {
        "index": 0,
        "name": "gain_value",
        "group": "gain",
        "description": "Low/high optomotor gain value.",
    },
    {
        "index": 1,
        "name": "gain_active",
        "group": "gain",
        "description": "Gain condition indicator.",
    },
    {
        "index": 2,
        "name": "dot_motion",
        "group": "dots",
        "description": "Dot-field orientation/coherence command.",
    },
    {
        "index": 3,
        "name": "dots_active",
        "group": "dots",
        "description": "Moving dots condition indicator.",
    },
    {
        "index": 4,
        "name": "flash_polarity",
        "group": "flash",
        "description": "Dark/bright flash polarity.",
    },
    {
        "index": 5,
        "name": "flash_active",
        "group": "flash",
        "description": "Flash condition indicator.",
    },
    {
        "index": 6,
        "name": "taxis_left",
        "group": "taxis",
        "description": "Left hemifield taxis stimulus.",
    },
    {
        "index": 7,
        "name": "taxis_right",
        "group": "taxis",
        "description": "Right hemifield taxis stimulus.",
    },
    {
        "index": 8,
        "name": "taxis_active",
        "group": "taxis",
        "description": "Phototaxis condition indicator.",
    },
    {
        "index": 9,
        "name": "turning_velocity",
        "group": "turning",
        "description": "Turning stimulus velocity magnitude.",
    },
    {
        "index": 10,
        "name": "turning_sin",
        "group": "turning",
        "description": "Sine component of turning direction.",
    },
    {
        "index": 11,
        "name": "turning_cos",
        "group": "turning",
        "description": "Cosine component of turning direction.",
    },
    {
        "index": 12,
        "name": "turning_active",
        "group": "turning",
        "description": "Turning condition indicator.",
    },
    {
        "index": 13,
        "name": "position_grating_a",
        "group": "position",
        "description": "Position grating type one-hot component.",
    },
    {
        "index": 14,
        "name": "position_grating_b",
        "group": "position",
        "description": "Position grating type one-hot component.",
    },
    {
        "index": 15,
        "name": "position_grating_c",
        "group": "position",
        "description": "Position grating type one-hot component.",
    },
    {
        "index": 16,
        "name": "position_delay",
        "group": "position",
        "description": "Position stimulus delay covariate.",
    },
    {
        "index": 17,
        "name": "position_active",
        "group": "position",
        "description": "Position condition indicator.",
    },
    {
        "index": 18,
        "name": "open_loop_active",
        "group": "open loop",
        "description": "Open-loop condition indicator.",
    },
    {
        "index": 19,
        "name": "rotation_direction",
        "group": "rotation",
        "description": "Rotation direction, rightward negative and leftward positive.",
    },
    {
        "index": 20,
        "name": "rotation_active",
        "group": "rotation",
        "description": "Rotation condition indicator.",
    },
    {
        "index": 21,
        "name": "dark_active",
        "group": "dark",
        "description": "Dark condition indicator.",
    },
    {
        "index": 22,
        "name": "specimen_covariate_0",
        "group": "specimen",
        "description": "Released specimen-level covariate.",
    },
    {
        "index": 23,
        "name": "specimen_covariate_1",
        "group": "specimen",
        "description": "Released specimen-level covariate.",
    },
    {
        "index": 24,
        "name": "specimen_covariate_2",
        "group": "specimen",
        "description": "Released specimen-level covariate.",
    },
    {
        "index": 25,
        "name": "specimen_covariate_3",
        "group": "specimen",
        "description": "Released specimen-level covariate.",
    },
)


class AppContext:
    """Container wired into FastAPI state at startup."""

    def __init__(self, runtime: LabSimRuntime, registry: ParameterRegistry) -> None:
        self.runtime = runtime
        self.registry = registry
        self.pending_patches: dict[str, Any] = {}
        cache_root = _repo_root() / "analysis" / "cache"
        self.video_pipeline = BackendVideoPipeline(
            cache_root=cache_root,
            upload_root=cache_root / "video_stimuli" / "uploads",
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
    return _repo_root().parent


def _video_sample_dir() -> Path:
    return _repo_root() / "analysis" / "cache" / "video_stimuli" / "clips"


def _video_upload_dir() -> Path:
    return _repo_root() / "analysis" / "cache" / "video_stimuli" / "uploads"


def _video_sample_path(slug: str) -> Path:
    safe_slug = "".join(ch for ch in slug if ch.isalnum() or ch in {"_", "-"})
    if safe_slug != slug or not safe_slug:
        raise HTTPException(status_code=404, detail="unknown video sample")
    path = (_video_sample_dir() / f"{safe_slug}.mp4").resolve()
    root = _video_sample_dir().resolve()
    if root not in path.parents:
        raise HTTPException(status_code=404, detail="unknown video sample")
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown video sample")
    return path


def _video_upload_path(file_name: str) -> Path:
    safe_name = Path(file_name).name
    if safe_name != file_name or not safe_name:
        raise HTTPException(status_code=404, detail="unknown uploaded video")
    path = (_video_upload_dir() / safe_name).resolve()
    root = _video_upload_dir().resolve()
    if root not in path.parents:
        raise HTTPException(status_code=404, detail="unknown uploaded video")
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown uploaded video")
    return path


def _video_source_path(file_name: str) -> Path:
    safe_name = Path(file_name).name
    if not safe_name:
        raise HTTPException(status_code=404, detail="unknown video source")
    sample = (_video_sample_dir() / safe_name).resolve()
    sample_root = _video_sample_dir().resolve()
    if sample.exists() and sample_root in sample.parents:
        return sample
    upload = (_video_upload_dir() / safe_name).resolve()
    upload_root = _video_upload_dir().resolve()
    if upload.exists() and upload_root in upload.parents:
        return upload
    raise HTTPException(status_code=404, detail="unknown video source")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _decoder_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            thresholds = json.loads(str(data["thresholds_json"]))
            return {
                "path": str(path),
                "exists": True,
                "context": int(data["context"]),
                "neurons": int(data["trace_mean"].shape[0]),
                "projection_components": int(data["projection"].shape[1]),
                "metadata": metadata,
                "thresholds": thresholds,
            }
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "exists": True, "error": str(exc)}


def _label_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            force = np.asarray(data["force"], dtype=float)
            kick = np.asarray(data["kick"], dtype=bool)
            side = np.asarray(data["side_class"], dtype=int)
            rows = np.asarray(data["rows"], dtype=int)
            stride = max(1, int(math.ceil(rows.shape[0] / 160)))
            samples = [
                {
                    "row": int(rows[i]),
                    "force": float(force[i]),
                    "kick": bool(kick[i]),
                    "side": int(side[i]),
                }
                for i in range(0, rows.shape[0], stride)
            ]
            return {
                "path": str(path),
                "exists": True,
                "frames": int(rows.shape[0]),
                "kick_rate": float(np.mean(kick)),
                "force_mean": float(np.mean(force)),
                "force_p95": float(np.quantile(force, 0.95)),
                "side_counts": {
                    "none": int(np.sum(side == 0)),
                    "left": int(np.sum(side == 1)),
                    "right": int(np.sum(side == 2)),
                },
                "samples": samples,
            }
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "exists": True, "error": str(exc)}


@lru_cache(maxsize=1)
def _zapbench_summary() -> dict[str, Any]:
    repo = _repo_root()
    workspace = _workspace_root()
    active = workspace / "active-inference"
    out = repo / "analysis" / "out"
    direct_dir = out / "zapbench_ephys_action_decoder"
    nonlinear_dir = out / "zapbench_ephys_action_decoder_sklearn"
    research = active / "simulations" / "zebrafish" / "research"
    return {
        "sources": [
            {
                "label": "ZAPBench landing",
                "url": "https://zapbench-release.storage.googleapis.com/landing.html",
            },
            {
                "label": "Dataset README",
                "url": "https://zapbench-release.storage.googleapis.com/volumes/README.html",
            },
            {
                "label": "GCS bucket browser",
                "url": "https://storage.googleapis.com/zapbench-release/browse.html",
            },
            {
                "label": "GitHub repository",
                "url": "https://github.com/google-research/zapbench",
            },
            {
                "label": "ICLR 2025 paper",
                "url": "https://openreview.net/forum?id=oCHsDpyawq",
            },
        ],
        "bucket": {
            "trace_tensorstore": "gs://zapbench-release/volumes/20240930/traces/",
            "stimulus_features": "gs://zapbench-release/volumes/20240930/stimuli_features/",
            "raw_ephys": "gs://zapbench-release/volumes/20240930/stimuli_raw/stimuli_and_ephys.10chFlt",
            "position_embedding": "gs://zapbench-release/volumes/20240930/position_embedding/",
            "segmentation_dataframe": "gs://zapbench-release/volumes/20240930/segmentation/dataframe.json",
            "benchmark_results": "gs://zapbench-release/dataframes/20250131/combined.json",
            "neuroglancer_states": "gs://zapbench-release/neuroglancer/20250131/",
            "fluroglancer_results": "gs://zapbench-release/figures/20250131/fluroglancer/",
        },
        "condition_names": [
            "gain",
            "dots",
            "flash",
            "taxis",
            "turning",
            "position",
            "open loop",
            "rotation",
            "dark",
        ],
        "condition_offsets": [0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879],
        "connectome_public": False,
        "connectome_note": (
            "ZAPBench states that synaptic-level anatomical mapping is in progress; "
            "no public connectome object was found in the release bucket."
        ),
        "linear_direct": {
            "metrics": _read_json(direct_dir / "direct_ephys_metrics.json"),
            "artifact": _decoder_artifact(
                active
                / "simulations"
                / "zebrafish"
                / "data"
                / "zapbench_ephys_tail_action_decoder.npz"
            ),
            "lag0_artifact": _decoder_artifact(
                active
                / "simulations"
                / "zebrafish"
                / "data"
                / "zapbench_ephys_tail_action_decoder_lag0.npz"
            ),
            "labels": _label_summary(direct_dir / "direct_ephys_labels.npz"),
            "report_path": str(direct_dir / "direct_ephys_report.md"),
        },
        "stimulus_fallback": {
            "metrics": _read_json(out / "zapbench_action_decoder" / "metrics.json"),
            "artifact": _decoder_artifact(
                active
                / "simulations"
                / "zebrafish"
                / "data"
                / "zapbench_tail_action_decoder.npz"
            ),
        },
        "nonlinear_experimental": {
            "metrics": _read_json(nonlinear_dir / "metrics.json"),
            "artifact_path": str(nonlinear_dir / "sklearn_tail_action_decoder.joblib"),
        },
        "research_notes": [
            str(research / "zapbench_source_notes_2026-05-13.md"),
            str(research / "zapbench_bucket_inventory_2026-05-13.md"),
        ],
    }


def _zapbench_cache_dir() -> Path:
    return _repo_root() / "analysis" / "cache" / "zapbench"


def _zapbench_direct_dir() -> Path:
    return _repo_root() / "analysis" / "out" / "zapbench_ephys_action_decoder"


def _zapbench_ephys_artifact_path() -> Path:
    return (
        _workspace_root()
        / "active-inference"
        / "simulations"
        / "zebrafish"
        / "data"
        / "zapbench_ephys_tail_action_decoder.npz"
    )


def _zapbench_advanced_metrics_path() -> Path:
    return _repo_root() / "analysis" / "out" / "zapbench_motor_bridge_lag0_stimulus" / "advanced_motor_bridge_metrics.json"


def _zapbench_projection_cache_path() -> Path:
    return _zapbench_cache_dir() / "zapbench_all_neuron_projection_512_seed17.npz"


@lru_cache(maxsize=16)
def _load_npz(path: str) -> dict[str, Any]:
    import numpy as np

    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(str(src))
    with np.load(src, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _decode_npz_scalar(value: Any) -> Any:
    try:
        if getattr(value, "shape", None) == ():
            return value.item()
    except Exception:  # noqa: BLE001
        return value
    return value


def _side_name(side_class: int) -> str:
    if side_class == 1:
        return "left"
    if side_class == 2:
        return "right"
    return "none"


def _zapbench_condition_index(row: int) -> int:
    for idx in range(len(_ZAPBENCH_CONDITION_OFFSETS) - 1):
        if _ZAPBENCH_CONDITION_OFFSETS[idx] <= row < _ZAPBENCH_CONDITION_OFFSETS[idx + 1]:
            return idx
    return max(0, len(_ZAPBENCH_CONDITION_NAMES) - 1)


def _zapbench_condition(row: int) -> dict[str, Any]:
    idx = _zapbench_condition_index(row)
    start = _ZAPBENCH_CONDITION_OFFSETS[idx]
    end = _ZAPBENCH_CONDITION_OFFSETS[idx + 1]
    phase = (row - start) / max(1, end - start - 1)
    return {
        "id": idx,
        "name": _ZAPBENCH_CONDITION_NAMES[idx],
        "start_row": start,
        "end_row": end - 1,
        "length": end - start,
        "phase": max(0.0, min(1.0, float(phase))),
    }


def _zapbench_condition_timeline(row: int) -> list[dict[str, Any]]:
    current = _zapbench_condition_index(row)
    timeline = []
    for idx, name in enumerate(_ZAPBENCH_CONDITION_NAMES):
        start = _ZAPBENCH_CONDITION_OFFSETS[idx]
        end = _ZAPBENCH_CONDITION_OFFSETS[idx + 1]
        timeline.append(
            {
                "id": idx,
                "name": name,
                "start_row": start,
                "end_row": end - 1,
                "length": end - start,
                "current": idx == current,
            }
        )
    return timeline


def _row_index(rows: Any, row: int) -> int | None:
    import numpy as np

    arr = np.asarray(rows, dtype=np.int64)
    if arr.size == 0:
        return None
    if int(arr[0]) == 0 and int(arr[-1]) == arr.size - 1:
        return row if 0 <= row < arr.size else None
    idx = int(np.searchsorted(arr, row))
    if idx < arr.size and int(arr[idx]) == row:
        return idx
    return None


def _zapbench_ephys_truth(row: int) -> dict[str, Any]:
    path = _zapbench_direct_dir() / "direct_ephys_labels.npz"
    try:
        data = _load_npz(str(path))
        idx = _row_index(data["rows"], row)
        if idx is None:
            return {"available": False, "path": str(path), "reason": "row outside direct-label artifact"}
        side_class = int(data["side_class"][idx])
        return {
            "available": True,
            "path": str(path),
            "row": int(data["rows"][idx]),
            "left_power": float(data["left_power"][idx]),
            "right_power": float(data["right_power"][idx]),
            "total_power": float(data["total_power"][idx]),
            "kick": bool(float(data["kick"][idx]) >= 0.5),
            "side": float(data["side"][idx]),
            "side_class": side_class,
            "side_name": _side_name(side_class),
            "force": float(data["force"][idx]),
            "raw_start_sample": int(data["starts"][idx]),
            "raw_end_sample": int(data["ends"][idx]),
            "raw_window_samples": int(data["ends"][idx] - data["starts"][idx]),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": str(path), "error": str(exc)}


def _zapbench_replay_action(row: int) -> dict[str, Any]:
    path = _zapbench_cache_dir() / "zapbench_calcium_action_replay.npz"
    try:
        data = _load_npz(str(path))
        idx = _row_index(data["rows"], row)
        if idx is None:
            return {"available": False, "path": str(path), "reason": "row outside replay artifact"}
        side_class = int(data["side_class"][idx])
        return {
            "available": True,
            "path": str(path),
            "row": int(data["rows"][idx]),
            "calcium_time_s": float(data["calcium_time_s"][idx]),
            "kick": float(data["kick"][idx]),
            "side_score": float(data["side_score"][idx]),
            "side_class": side_class,
            "side_name": _side_name(side_class),
            "force": float(data["force"][idx]),
            "kick_score": float(data["kick_score"][idx]),
            "confidence": float(data["confidence"][idx]),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": str(path), "error": str(exc)}


def _zapbench_decoder_prediction(row: int) -> dict[str, Any]:
    replay_prediction = _zapbench_replay_decoder_prediction(row)
    if replay_prediction.get("available"):
        return replay_prediction

    scores_path = _zapbench_direct_dir() / "linear_scores_for_thresholds.npz"
    artifact_path = _zapbench_ephys_artifact_path()
    try:
        scores = _load_npz(str(scores_path))
        artifact = _load_npz(str(artifact_path))
        thresholds = json.loads(str(_decode_npz_scalar(artifact["thresholds_json"])))
        metadata = json.loads(str(_decode_npz_scalar(artifact["metadata_json"])))
        idx = _row_index(scores["target_rows"], row)
        if idx is None:
            return {
                "available": False,
                "path": str(scores_path),
                "artifact_path": str(artifact_path),
                "reason": "row outside decoder score artifact",
                "thresholds": thresholds,
            }
        kick_score, side_score, force = [float(v) for v in scores["scores"][idx]]
        kick = kick_score >= float(thresholds.get("kick", 0.5))
        side_abs = float(thresholds.get("side_abs", 0.15))
        force_none = float(thresholds.get("force_none", 0.0))
        if force <= force_none or abs(side_score) < side_abs:
            side_class = 0
        else:
            side_class = 1 if side_score < 0 else 2
        return {
            "available": True,
            "path": str(scores_path),
            "artifact_path": str(artifact_path),
            "row": int(scores["target_rows"][idx]),
            "kick_score": kick_score,
            "side_score": side_score,
            "force": force,
            "kick": bool(kick),
            "side_class": side_class,
            "side_name": _side_name(side_class),
            "is_train_row": bool(scores["train"][idx]),
            "is_test_row": bool(scores["test"][idx]),
            "thresholds": thresholds,
            "metadata": {
                "label_lag": int(metadata.get("label_lag", 0)),
                "label_type": metadata.get("label_type"),
                "side_convention": metadata.get("side_convention"),
                "left_channel": metadata.get("left_channel"),
                "right_channel": metadata.get("right_channel"),
                "estimated_sample_rate_hz": metadata.get("estimated_sample_rate_hz"),
                "window_samples_10ms": metadata.get("window_samples_10ms"),
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "path": str(scores_path),
            "artifact_path": str(artifact_path),
            "error": str(exc),
        }


def _zapbench_replay_decoder_prediction(row: int) -> dict[str, Any]:
    path = _zapbench_cache_dir() / "zapbench_calcium_action_replay.npz"
    try:
        data = _load_npz(str(path))
        idx = _row_index(data["rows"], row)
        if idx is None:
            return {"available": False, "path": str(path), "reason": "row outside current replay artifact"}
        metadata = json.loads(str(_decode_npz_scalar(data.get("metadata_json", "{}"))))
        thresholds = metadata.get("thresholds", {})
        side_class = int(data["side_class"][idx])
        return {
            "available": True,
            "path": str(path),
            "artifact_path": str(_zapbench_advanced_metrics_path()),
            "row": int(data["rows"][idx]),
            "source_feature_row": int(data["source_feature_rows"][idx])
            if "source_feature_rows" in data
            else int(data["rows"][idx]),
            "kick_score": float(data["kick_score"][idx]),
            "side_score": float(data["side_score"][idx]),
            "force": float(data["force"][idx]),
            "kick": bool(float(data["kick"][idx]) >= 0.5),
            "side_class": side_class,
            "side_name": _side_name(side_class),
            "confidence": float(data["confidence"][idx]) if "confidence" in data else None,
            "is_train_row": False,
            "is_test_row": False,
            "thresholds": thresholds,
            "metadata": {
                "label_lag": int(metadata.get("lag", 0)),
                "label_type": "raw_tail_ephys_current_replay",
                "include_stimulus": bool(metadata.get("include_stimulus", False)),
                "row_semantics": metadata.get("row_semantics"),
                "frame_hz": metadata.get("frame_hz"),
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": str(path), "error": str(exc)}


def _zapbench_stimulus_covariates(row: int) -> dict[str, Any]:
    path = _zapbench_cache_dir() / "zapbench_stimulus_features.npz"
    try:
        data = _load_npz(str(path))
        features = data["features"]
        if row < 0 or row >= features.shape[0]:
            return {"available": False, "path": str(path), "reason": "row outside stimulus cache"}
        values = features[row]
        columns = []
        active = []
        for spec in _ZAPBENCH_STIMULUS_COLUMNS:
            value = float(values[int(spec["index"])])
            item = {**spec, "value": value, "active": abs(value) > 1e-6}
            columns.append(item)
            if item["active"]:
                active.append(item)
        return {
            "available": True,
            "path": str(path),
            "source": str(_decode_npz_scalar(data.get("source", ""))),
            "row": row,
            "columns": columns,
            "active_columns": active,
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": str(path), "error": str(exc)}


def _zapbench_brain_preview(row: int) -> dict[str, Any]:
    path = _zapbench_cache_dir() / "zapbench_trace_preview.npz"
    try:
        import numpy as np

        data = _load_npz(str(path))
        traces = np.asarray(data["traces"], dtype=np.float32)
        neuron_ids = np.asarray(data["neuron_ids"], dtype=np.int32)
        if row < 0 or row >= traces.shape[0]:
            return {"available": False, "path": str(path), "reason": "row outside trace cache"}
        start = max(0, row - 12)
        end = min(traces.shape[0], row + 13)
        neuron_step = max(1, int(math.ceil(traces.shape[1] / 48)))
        selected = np.arange(0, traces.shape[1], neuron_step, dtype=np.int32)[:48]
        window = traces[start:end, selected]
        finite = window[np.isfinite(window)]
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = float(np.max(finite)) if finite.size else 0.0
        return {
            "available": True,
            "path": str(path),
            "source": str(_decode_npz_scalar(data.get("source", ""))),
            "artifact_path": str(_decode_npz_scalar(data.get("artifact_path", ""))),
            "row": row,
            "start_row": start,
            "end_row": end - 1,
            "current_column": row - start,
            "neuron_ids": [int(v) for v in neuron_ids[selected]],
            "rows": [int(v) for v in range(start, end)],
            "values": np.round(window, 4).astype(float).tolist(),
            "min": vmin,
            "max": vmax,
            "shape": [int(window.shape[0]), int(window.shape[1])],
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": str(path), "error": str(exc)}


def _zapbench_model_card(row: int) -> dict[str, Any]:
    advanced = _read_json(_zapbench_advanced_metrics_path()) or {}
    best = advanced.get("best") or {}
    metrics = best.get("test_metrics") or {}
    projection = _projection_summary()
    per_condition = metrics.get("per_condition", {})
    condition_name = _zapbench_condition(row)["name"]
    return {
        "label": "advanced all-neuron calcium-plus-stimulus motor bridge",
        "metrics": {
            "kick_accuracy": metrics.get("kick_accuracy"),
            "kick_f1": metrics.get("kick_f1"),
            "side_accuracy": metrics.get("side_accuracy"),
            "side_active_accuracy": metrics.get("side_active_accuracy"),
            "side_macro_accuracy": metrics.get("side_macro_accuracy"),
            "force_mae": metrics.get("force_mae"),
            "force_r2": metrics.get("force_r2"),
        },
        "condition_metrics": per_condition.get(condition_name),
        "nonlinear_experimental": {},
        "artifact": {
            "path": advanced.get("model_path"),
            "exists": bool(advanced.get("model_path")),
            "neurons": projection.get("neurons"),
            "projection_components": projection.get("projection_components"),
            "projection": projection.get("projection"),
            "include_stimulus": best.get("include_stimulus"),
            "lag": best.get("lag"),
        },
        "label_artifact": _label_summary(_zapbench_direct_dir() / "direct_ephys_labels.npz"),
        "decoder_note": (
            "Current lab replay scores are generated by the advanced all-neuron ZAPBench bridge "
            "trained against raw left/right tail motor-nerve ephys and optional official stimulus covariates."
        ),
    }


def _projection_summary() -> dict[str, Any]:
    path = _zapbench_projection_cache_path()
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        data = _load_npz(str(path))
        metadata = json.loads(str(_decode_npz_scalar(data.get("metadata_json", "{}"))))
        return {
            "path": str(path),
            "exists": True,
            "neurons": int(metadata.get("neurons", 0)),
            "projection_components": int(metadata.get("projection_components", data["projected"].shape[1])),
            "projection": metadata.get("projection"),
            "frames": int(metadata.get("frames", data["projected"].shape[0])),
        }
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "exists": True, "error": str(exc)}


def _zapbench_replay_detail(row: int) -> dict[str, Any]:
    bounded_row = int(max(0, min(_ZAPBENCH_CONDITION_OFFSETS[-1] - 1, row)))
    return {
        "row": bounded_row,
        "condition": _zapbench_condition(bounded_row),
        "condition_timeline": _zapbench_condition_timeline(bounded_row),
        "ephys_truth": _zapbench_ephys_truth(bounded_row),
        "replay_action": _zapbench_replay_action(bounded_row),
        "decoder_prediction": _zapbench_decoder_prediction(bounded_row),
        "stimulus_covariates": _zapbench_stimulus_covariates(bounded_row),
        "brain_preview": _zapbench_brain_preview(bounded_row),
        "model_card": _zapbench_model_card(bounded_row),
    }


def build_rest_router(app_ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api")

    def _video_env() -> AquaticArenaEnvironment:
        env = app_ctx.runtime.engine.environment
        if not isinstance(env, AquaticArenaEnvironment):
            raise HTTPException(status_code=500, detail="zebrafish environment missing")
        return env

    @router.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "tick": app_ctx.runtime.transport_snapshot()["tick"]}

    @router.get("/sim/transport")
    def get_transport() -> dict[str, Any]:
        return app_ctx.runtime.transport_snapshot()

    @router.post("/sim/transport")
    def set_transport(body: TransportAction) -> dict[str, Any]:
        if body.action == "play":
            return app_ctx.runtime.play()
        if body.action == "pause":
            return app_ctx.runtime.pause()
        if body.action == "step":
            return app_ctx.runtime.step_once()
        raise HTTPException(status_code=400, detail=f"unknown action: {body.action}")

    @router.get("/sim/pacing")
    def get_pacing() -> dict[str, Any]:
        return app_ctx.runtime.pacing_snapshot()

    @router.post("/sim/pacing")
    def set_pacing(body: PacingBody) -> dict[str, Any]:
        return app_ctx.runtime.set_pacing(
            real_ms_per_physics_step=body.real_ms_per_physics_step,
            real_ms_per_neural_tick=body.real_ms_per_neural_tick,
        )

    @router.get("/schema")
    def schema() -> dict[str, Any]:
        specs: list[dict[str, Any]] = []
        snap = app_ctx.registry.snapshot(app_ctx)
        for spec in app_ctx.registry.all():
            specs.append(
                {
                    "path": spec.path,
                    "label": spec.label,
                    "group": spec.group,
                    "kind": spec.kind,
                    "apply": spec.apply,
                    "min": spec.min,
                    "max": spec.max,
                    "step": spec.step,
                    "enum": list(spec.enum) if spec.enum else None,
                    "help": spec.help,
                    "value": snap.get(spec.path),
                }
            )
        return {"specs": specs, "pending": dict(app_ctx.pending_patches)}

    @router.get("/connectome")
    def connectome() -> dict[str, Any]:
        ns = app_ctx.runtime.engine.nervous_system
        if not isinstance(ns, ZebrafishNervousSystem):
            raise HTTPException(status_code=500, detail="nervous system missing")
        return build_connectome_view(ns._connectome, ns.get_neuron_names_paula_order())  # type: ignore[attr-defined]

    @router.get("/body")
    def body_view() -> dict[str, Any]:
        body = app_ctx.runtime.engine.body
        return build_body_view(body)  # type: ignore[arg-type]

    @router.get("/zapbench")
    def zapbench() -> dict[str, Any]:
        return _zapbench_summary()

    @router.get("/zapbench/replay-detail")
    def zapbench_replay_detail(row: int | None = None) -> dict[str, Any]:
        resolved_row = row
        if resolved_row is None:
            with app_ctx.runtime.sim_lock:
                env_state = _video_env().calcium_stimulus_state()
                if env_state.get("has_frame"):
                    resolved_row = int(env_state.get("row") or 0)
                else:
                    replay = app_ctx.runtime.calcium_replay_snapshot()
                    resolved_row = int(replay.get("row") or 0)
        return _zapbench_replay_detail(int(resolved_row or 0))

    @router.get("/video-stimulus")
    def get_video_stimulus() -> dict[str, Any]:
        with app_ctx.runtime.sim_lock:
            return _video_env().video_stimulus_state()

    @router.get("/video-stimulus/samples")
    def get_video_stimulus_samples() -> dict[str, Any]:
        root = _video_sample_dir()
        samples = []
        if root.exists():
            for path in sorted(root.glob("*.mp4")):
                samples.append(
                    {
                        "slug": path.stem,
                        "file_name": path.name,
                        "url": f"/api/video-stimulus/samples/{path.stem}",
                        "bytes": int(path.stat().st_size),
                    }
                )
        return {"samples": samples}

    @router.get("/video-stimulus/samples/{slug}")
    def get_video_stimulus_sample_file(slug: str) -> FileResponse:
        return FileResponse(_video_sample_path(slug), media_type="video/mp4")

    @router.get("/video-stimulus/uploads/{file_name}")
    def get_video_stimulus_upload_file(file_name: str) -> FileResponse:
        return FileResponse(_video_upload_path(file_name), media_type="video/mp4")

    @router.post("/video-stimulus/upload")
    async def upload_video_stimulus(request: Request, file_name: str) -> dict[str, Any]:
        safe_name = safe_upload_name(file_name)
        root = _video_upload_dir()
        root.mkdir(parents=True, exist_ok=True)
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty upload")
        path = (root / safe_name).resolve()
        if root.resolve() not in path.parents:
            raise HTTPException(status_code=400, detail="invalid upload path")
        path.write_bytes(body)
        return {
            "slug": path.stem,
            "file_name": path.name,
            "url": f"/api/video-stimulus/uploads/{path.name}",
            "bytes": int(path.stat().st_size),
            "uploaded": True,
        }

    @router.post("/video-stimulus/config")
    def set_video_stimulus(body: VideoStimulusConfigBody) -> dict[str, Any]:
        try:
            return app_ctx.runtime.configure_video_stimulus(
                enabled=body.enabled,
                gain=body.gain,
                file_name=body.file_name,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post("/video-stimulus/frame")
    def push_video_stimulus_frame(body: VideoStimulusFrameBody) -> dict[str, Any]:
        payload = body.model_dump()
        sample_hz = float(payload.pop("sample_hz", 15.0))
        try:
            return app_ctx.runtime.push_video_stimulus_frame_and_step(payload, sample_hz=sample_hz)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post("/video-stimulus/backend-frame")
    def push_backend_video_stimulus_frame(body: BackendVideoFrameBody) -> dict[str, Any]:
        try:
            path = _video_source_path(body.file_name)
            extracted = app_ctx.video_pipeline.extract(
                path=path,
                file_name=Path(body.file_name).name,
                frame_index=body.frame_index,
                video_time_s=body.video_time_s,
                sample_hz=body.sample_hz,
            )
            payload = dict(extracted.features)
            if body.enabled is not None:
                payload["enabled"] = body.enabled
            state = app_ctx.runtime.push_video_stimulus_frame_and_step(
                payload,
                sample_hz=body.sample_hz,
            )
            return {
                "state": state,
                "features": extracted.features,
                "diagnostics": extracted.diagnostics,
            }
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post("/video-stimulus/clear")
    def clear_video_stimulus() -> dict[str, Any]:
        try:
            return app_ctx.runtime.clear_video_stimulus()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get("/calcium-stimulus")
    def get_calcium_stimulus() -> dict[str, Any]:
        with app_ctx.runtime.sim_lock:
            env_state = _video_env().calcium_stimulus_state()
            return {
                **env_state,
                "replay": app_ctx.runtime.calcium_replay_snapshot(),
            }

    @router.post("/calcium-stimulus/config")
    def set_calcium_stimulus(body: CalciumStimulusConfigBody) -> dict[str, Any]:
        with app_ctx.runtime.sim_lock:
            env = _video_env()
            env.set_calcium_stimulus(
                enabled=body.enabled,
                gain=body.gain,
                source=body.source,
            )
            return env.calcium_stimulus_state()

    @router.post("/calcium-stimulus/frame")
    def push_calcium_action_frame(body: CalciumActionFrameBody) -> dict[str, Any]:
        with app_ctx.runtime.sim_lock:
            return _video_env().push_calcium_action_frame(body.model_dump())

    @router.post("/calcium-stimulus/replay")
    def set_calcium_replay(body: CalciumReplayBody) -> dict[str, Any]:
        if body.enabled is True:
            app_ctx.runtime.clear_video_stimulus()
        return app_ctx.runtime.configure_calcium_replay(
            enabled=body.enabled,
            gain=body.gain,
            condition=body.condition,
            loop=body.loop,
            replay_path=body.replay_path,
        )

    @router.post("/calcium-stimulus/clear")
    def clear_calcium_stimulus() -> dict[str, Any]:
        return app_ctx.runtime.clear_calcium_replay()

    @router.get("/neurons/{name}")
    def neuron(name: str) -> dict[str, Any]:
        ns = app_ctx.runtime.engine.nervous_system
        if not isinstance(ns, ZebrafishNervousSystem):
            raise HTTPException(status_code=500, detail="nervous system missing")
        neuron = ns.get_neuron_by_name(name)
        if neuron is None:
            raise HTTPException(status_code=404, detail=f"unknown neuron: {name}")
        params = neuron.params
        id_to_name = {nid: nm for nm, nid in ns.name_to_id.items()}
        postsynaptic: list[dict[str, Any]] = []
        for sid in sorted(neuron.postsynaptic_points.keys()):
            pt = neuron.postsynaptic_points[sid]
            src = neuron.synapse_sources.get(sid)
            pre_name = id_to_name.get(src[0]) if src else None
            postsynaptic.append(
                {
                    "id": int(sid),
                    "pre_paula_id": int(src[0]) if src else None,
                    "pre_terminal": int(src[1]) if src else None,
                    "pre_name": pre_name,
                    "info": float(pt.u_i.info),
                    "plast": float(pt.u_i.plast),
                    "adapt": [float(x) for x in pt.u_i.adapt],
                    "potential": float(pt.potential),
                }
            )
        presynaptic: list[dict[str, Any]] = []
        for tid in sorted(neuron.presynaptic_points.keys()):
            pr = neuron.presynaptic_points[tid]
            presynaptic.append(
                {
                    "id": int(tid),
                    "u_o_info": float(pr.u_o.info),
                    "u_o_mod": [float(x) for x in pr.u_o.mod],
                    "u_i_retro": float(pr.u_i_retro),
                }
            )
        t_last_fire = float(neuron.t_last_fire)
        return {
            "name": name,
            "paula_id": int(neuron.id),
            "S": float(neuron.S),
            "O": float(neuron.O),
            "r": float(neuron.r),
            "b": float(neuron.b),
            "t_ref": float(neuron.t_ref),
            "F_avg": float(neuron.F_avg),
            "t_last_fire": t_last_fire if math.isfinite(t_last_fire) else None,
            "M_vector": [float(x) for x in neuron.M_vector],
            "pq_len": int(len(neuron.propagation_queue)),
            "postsynaptic": postsynaptic,
            "presynaptic": presynaptic,
            "params": {
                "r_base": float(params.r_base),
                "b_base": float(params.b_base),
                "c": int(params.c),
                "lambda_param": float(params.lambda_param),
                "p": float(params.p),
                "eta_post": float(params.eta_post),
                "eta_retro": float(params.eta_retro),
                "delta_decay": float(params.delta_decay),
                "beta_avg": float(params.beta_avg),
                "gamma": [float(x) for x in params.gamma],
                "w_r": [float(x) for x in params.w_r],
                "w_b": [float(x) for x in params.w_b],
                "w_tref": [float(x) for x in params.w_tref],
                "num_neuromodulators": int(params.num_neuromodulators),
                "num_inputs": int(params.num_inputs),
            },
        }

    @router.post("/neurons/{name}/patch")
    def patch_neuron(name: str, body: NeuronPatchBody) -> dict[str, Any]:
        """Hot-patch one neuron's :class:`NeuronParameters` live.

        All edits are applied under ``sim_lock`` so a tick cannot run between
        a field update and the next simulation step.
        """
        ns = app_ctx.runtime.engine.nervous_system
        if not isinstance(ns, ZebrafishNervousSystem):
            raise HTTPException(status_code=500, detail="nervous system missing")
        neuron = ns.get_neuron_by_name(name)
        if neuron is None:
            raise HTTPException(status_code=404, detail=f"unknown neuron: {name}")

        import numpy as np

        from neuron.neuron import MAX_SYNAPTIC_WEIGHT, MIN_SYNAPTIC_WEIGHT

        applied: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        with app_ctx.runtime.sim_lock:
            params = neuron.params
            for patch in body.patches:
                try:
                    if patch.field in _NEURON_RUNTIME_FIELDS:
                        caster = _NEURON_RUNTIME_FIELDS[patch.field]
                        setattr(neuron, patch.field, caster(patch.value))
                        applied.append({"field": patch.field})
                    elif patch.field == "M_vector":
                        arr = neuron.M_vector
                        if patch.index is None:
                            new_arr = np.asarray(patch.value, dtype=float).reshape(
                                -1,
                            )
                            if new_arr.shape[0] != arr.shape[0]:
                                raise ValueError(
                                    f"M_vector length {new_arr.shape[0]} != {arr.shape[0]}"
                                )
                            neuron.M_vector = new_arr.astype(float, copy=True)
                            applied.append({"field": patch.field})
                        else:
                            if not (0 <= patch.index < arr.shape[0]):
                                raise IndexError(
                                    f"M_vector[{patch.index}] out of range"
                                )
                            arr[patch.index] = float(patch.value)
                            applied.append(
                                {"field": patch.field, "index": patch.index}
                            )
                    elif patch.field == "postsynaptic":
                        if patch.index is None:
                            raise ValueError("postsynaptic patch requires index (slot id)")
                        pt = neuron.postsynaptic_points.get(patch.index)
                        if pt is None:
                            raise KeyError(f"unknown postsynaptic slot {patch.index}")
                        sf = (patch.subfield or "info").lower().replace("u_i.", "")
                        if sf == "info":
                            v = float(patch.value)
                            pt.u_i.info = float(
                                np.clip(v, MIN_SYNAPTIC_WEIGHT, MAX_SYNAPTIC_WEIGHT)
                            )
                        elif sf == "plast":
                            pt.u_i.plast = float(patch.value)
                        elif sf == "potential":
                            pt.potential = float(patch.value)
                        elif sf == "adapt":
                            if patch.vec_index is None:
                                raise ValueError("adapt patch requires vec_index")
                            if not (0 <= patch.vec_index < pt.u_i.adapt.shape[0]):
                                raise IndexError("adapt vec_index out of range")
                            pt.u_i.adapt[patch.vec_index] = float(patch.value)
                        else:
                            raise KeyError(f"unknown postsynaptic subfield: {patch.subfield}")
                        applied.append(
                            {
                                "field": patch.field,
                                "index": patch.index,
                                "subfield": patch.subfield,
                                "vec_index": patch.vec_index,
                            }
                        )
                    elif patch.field == "presynaptic":
                        if patch.index is None:
                            raise ValueError("presynaptic patch requires index (terminal id)")
                        pr = neuron.presynaptic_points.get(patch.index)
                        if pr is None:
                            raise KeyError(f"unknown presynaptic terminal {patch.index}")
                        sf = (patch.subfield or "u_o_info").lower()
                        if sf in ("u_o_info", "info", "u_o.info"):
                            v = float(patch.value)
                            pr.u_o.info = float(
                                np.clip(v, MIN_SYNAPTIC_WEIGHT, MAX_SYNAPTIC_WEIGHT)
                            )
                        elif sf in ("u_i_retro", "retro"):
                            pr.u_i_retro = float(patch.value)
                        elif sf in ("mod", "u_o.mod"):
                            if patch.vec_index is None:
                                raise ValueError("mod patch requires vec_index")
                            if not (0 <= patch.vec_index < pr.u_o.mod.shape[0]):
                                raise IndexError("mod vec_index out of range")
                            pr.u_o.mod[patch.vec_index] = float(patch.value)
                        else:
                            raise KeyError(f"unknown presynaptic subfield: {patch.subfield}")
                        applied.append(
                            {
                                "field": patch.field,
                                "index": patch.index,
                                "subfield": patch.subfield,
                                "vec_index": patch.vec_index,
                            }
                        )
                    elif patch.field in _NEURON_SCALAR_FIELDS:
                        caster = _NEURON_SCALAR_FIELDS[patch.field]
                        setattr(params, patch.field, caster(patch.value))
                        applied.append({"field": patch.field})
                    elif patch.field in _NEURON_VECTOR_FIELDS:
                        arr = getattr(params, patch.field)
                        if patch.index is None:
                            new_arr = np.asarray(patch.value, dtype=float).reshape(
                                arr.shape
                            )
                            setattr(params, patch.field, new_arr)
                            applied.append({"field": patch.field})
                        else:
                            if not (0 <= patch.index < arr.shape[0]):
                                raise IndexError(
                                    f"{patch.field}[{patch.index}] out of range"
                                )
                            arr[patch.index] = float(patch.value)
                            applied.append(
                                {"field": patch.field, "index": patch.index}
                            )
                    else:
                        raise KeyError(f"unknown neuron field: {patch.field}")
                except Exception as exc:  # noqa: BLE001
                    failed.append({"field": patch.field, "error": str(exc)})
        return {"applied": applied, "failed": failed}

    @router.get("/muscles/{muscle_name}")
    def muscle(muscle_name: str) -> dict[str, Any]:
        body = app_ctx.runtime.engine.body
        import mujoco

        aid = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle_name)
        if aid < 0:
            raise HTTPException(status_code=404, detail=f"unknown muscle: {muscle_name}")
        mj = body.model
        mj_data = body.data
        force_max = float(mj.actuator_forcerange[aid, 1]) or 1.0
        return {
            "name": muscle_name,
            "id": int(aid),
            "ctrl": float(mj_data.ctrl[aid]),
            "activation": float(mj_data.ctrl[aid]) / force_max if force_max else 0.0,
            "forcerange": [
                float(mj.actuator_forcerange[aid, 0]),
                float(mj.actuator_forcerange[aid, 1]),
            ],
            "gear": [float(x) for x in mj.actuator_gear[aid]],
            "target_joint_id": int(mj.actuator_trnid[aid, 0]),
        }

    @router.post("/body/patch")
    def patch_body(body: BodyPatchBody) -> dict[str, Any]:
        """Hot-patch mjModel fields live under ``sim_lock``.

        Only scalar-ish fields that MuJoCo re-reads each step are accepted
        (damping, armature, actuator forcerange/gear, body mass, contact pair
        friction/solref/solimp, opt gravity/viscosity/density).
        """
        import mujoco

        mj = app_ctx.runtime.engine.body.model
        applied: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        with app_ctx.runtime.sim_lock:
            for patch in body.patches:
                try:
                    if patch.target == "joint":
                        assert patch.id is not None
                        dof = int(mj.jnt_dofadr[patch.id])
                        if patch.field == "damping":
                            mj.dof_damping[dof] = float(patch.value)
                        elif patch.field == "armature":
                            mj.dof_armature[dof] = float(patch.value)
                        else:
                            raise KeyError(patch.field)
                    elif patch.target == "actuator":
                        assert patch.id is not None
                        if patch.field == "forcerange":
                            idx = patch.index if patch.index is not None else 1
                            mj.actuator_forcerange[patch.id, idx] = float(patch.value)
                        elif patch.field == "gear":
                            idx = patch.index if patch.index is not None else 0
                            mj.actuator_gear[patch.id, idx] = float(patch.value)
                        else:
                            raise KeyError(patch.field)
                    elif patch.target == "body":
                        assert patch.id is not None
                        if patch.field == "mass":
                            mj.body_mass[patch.id] = float(patch.value)
                        elif patch.field == "inertia":
                            assert patch.index is not None
                            mj.body_inertia[patch.id, patch.index] = float(patch.value)
                        else:
                            raise KeyError(patch.field)
                    elif patch.target == "pair":
                        assert patch.id is not None
                        if patch.field == "friction":
                            assert patch.index is not None
                            mj.pair_friction[patch.id, patch.index] = float(patch.value)
                        elif patch.field == "solref":
                            assert patch.index is not None
                            mj.pair_solref[patch.id, patch.index] = float(patch.value)
                        elif patch.field == "solimp":
                            assert patch.index is not None
                            mj.pair_solimp[patch.id, patch.index] = float(patch.value)
                        else:
                            raise KeyError(patch.field)
                    elif patch.target == "opt":
                        # Whitelist every writable mjOption field (mirrors
                        # lab.parameters.mujoco_engine_params).
                        _opt_float = {
                            "timestep",
                            "impratio",
                            "tolerance",
                            "noslip_tolerance",
                            "ls_tolerance",
                            "ccd_tolerance",
                            "sleep_tolerance",
                            "density",
                            "viscosity",
                            "o_margin",
                        }
                        _opt_int = {
                            "integrator",
                            "cone",
                            "jacobian",
                            "solver",
                            "iterations",
                            "noslip_iterations",
                            "sdf_iterations",
                            "ccd_iterations",
                            "ls_iterations",
                            "sdf_initpoints",
                            "disableflags",
                            "enableflags",
                            "disableactuator",
                            "enableactuator",
                        }
                        _opt_vec = ("gravity", "wind", "magnetic", "o_solref", "o_solimp", "o_friction")
                        if patch.field in _opt_float:
                            setattr(mj.opt, patch.field, float(patch.value))
                        elif patch.field in _opt_int:
                            setattr(mj.opt, patch.field, int(patch.value))
                        elif patch.field in _opt_vec:
                            assert patch.index is not None
                            getattr(mj.opt, patch.field)[patch.index] = float(
                                patch.value
                            )
                        else:
                            raise KeyError(patch.field)
                    else:
                        raise KeyError(f"unknown target: {patch.target}")

                    applied.append(
                        {
                            "target": patch.target,
                            "id": patch.id,
                            "field": patch.field,
                            "index": patch.index,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    failed.append(
                        {
                            "target": patch.target,
                            "id": patch.id,
                            "field": patch.field,
                            "index": patch.index,
                            "error": str(exc),
                        }
                    )

            # Body mass changes invalidate composite rigid-body inertias; cheap to recompute.
            if any(p.target == "body" and p.field == "mass" for p in body.patches):
                mujoco.mj_setTotalmass(mj, float(mj.body_mass.sum()))

        return {"applied": applied, "failed": failed}

    @router.get("/pending")
    def get_pending() -> dict[str, Any]:
        return {"pending": dict(app_ctx.pending_patches)}

    @router.post("/patch")
    def patch(body: PatchBody) -> dict[str, Any]:
        result = apply_patches(
            app_ctx.registry,
            app_ctx,
            [{"path": p.path, "value": p.value} for p in body.patches],
            enqueue_live=app_ctx.runtime.enqueue_patch,
        )
        return {
            "applied": result.applied,
            "pending": result.pending,
            "failed": result.failed,
        }

    @router.post("/apply-pending")
    def apply_pending() -> dict[str, Any]:
        """Commit every pending rebuild patch, then reset the simulation."""
        stash = dict(app_ctx.pending_patches)
        applied: list[str] = []
        failed: list[dict[str, Any]] = []
        with app_ctx.runtime.sim_lock:
            for path, value in stash.items():
                try:
                    spec = app_ctx.registry.get(path)
                except Exception as exc:  # noqa: BLE001
                    failed.append({"path": path, "error": str(exc)})
                    continue
                try:
                    spec.setter(app_ctx, value)
                    applied.append(path)
                except Exception as exc:  # noqa: BLE001
                    failed.append({"path": path, "error": str(exc)})
            if applied:
                app_ctx.runtime.loop.reset()
        for k in applied:
            app_ctx.pending_patches.pop(k, None)
        return {"applied": applied, "failed": failed}

    @router.post("/reset")
    def reset_sim() -> dict[str, Any]:
        with app_ctx.runtime.sim_lock:
            app_ctx.runtime.loop.reset()
        return {"ok": True}

    return router
