"""Generate and validate calcium-imaging-driven zebrafish behavior.

This script converts public ZAPBench calcium traces into a compact replay file
that the lab can use directly.  Each replay row is decoded with the existing
direct-ephys-trained calcium-to-tail-action artifact:

``active-inference/simulations/zebrafish/data/zapbench_ephys_tail_action_decoder.npz``

The output is intentionally small enough to load in the lab server: row id,
condition id, decoded kick/side/force, and decoder scores.  The raw calcium
traces stay in the public TensorStore bucket and are only read during artifact
generation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ACTIVE_INFERENCE = REPO_ROOT / "active-inference"
if str(ACTIVE_INFERENCE) not in sys.path:
    sys.path.insert(0, str(ACTIVE_INFERENCE))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from simulations.zebrafish.action_decoder import LinearTailActionDecoder  # noqa: E402
from zapbench_action_decoder import CONDITION_NAMES, CONDITION_OFFSETS, TRACE_SPEC, _read_tensorstore  # noqa: E402


ZAPBENCH_FRAME_HZ = 1.093
PHYSICS_DT_S = 0.005


def _ticks_for_zapbench_frame(frame_index: int) -> int:
    """No-drift conversion from one ~1 Hz ZAPBench action frame to physics ticks."""
    idx = max(0, int(frame_index))
    ticks_per_frame = 1.0 / (ZAPBENCH_FRAME_HZ * PHYSICS_DT_S)
    start_tick = int(round(idx * ticks_per_frame))
    end_tick = int(round((idx + 1) * ticks_per_frame))
    return max(1, end_tick - start_tick)


def _condition_for_row(row: int) -> int:
    for idx in range(len(CONDITION_OFFSETS) - 1):
        if CONDITION_OFFSETS[idx] <= row < CONDITION_OFFSETS[idx + 1]:
            return idx
    return len(CONDITION_NAMES) - 1


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _load_artifact_ranges(artifact_path: Path) -> np.ndarray:
    with np.load(artifact_path, allow_pickle=False) as data:
        return np.asarray(data["block_ranges"], dtype=np.int32)


def _load_selected_trace_blocks(block_ranges: np.ndarray) -> np.ndarray:
    trace_ds = _read_tensorstore(TRACE_SPEC)
    arrays: list[np.ndarray] = []
    for start, stop in block_ranges:
        arrays.append(np.asarray(trace_ds[:, int(start) : int(stop)].read().result(), dtype=np.float32))
    return np.concatenate(arrays, axis=1)


def generate_replay(artifact_path: Path, output_path: Path) -> dict[str, Any]:
    decoder = LinearTailActionDecoder.from_npz(artifact_path)
    block_ranges = _load_artifact_ranges(artifact_path)
    traces = _load_selected_trace_blocks(block_ranges)
    rows = np.arange(decoder.context - 1, traces.shape[0], dtype=np.int32)
    n = rows.shape[0]

    kick = np.zeros(n, dtype=np.float32)
    side_score = np.zeros(n, dtype=np.float32)
    force = np.zeros(n, dtype=np.float32)
    kick_score = np.zeros(n, dtype=np.float32)
    side_class = np.zeros(n, dtype=np.int32)
    confidence = np.zeros(n, dtype=np.float32)
    for i, row in enumerate(rows):
        action = decoder.decode(traces[int(row) - decoder.context + 1 : int(row) + 1])
        kick[i] = 1.0 if action.kick else 0.0
        side_score[i] = float(np.clip(action.side_score, -1.0, 1.0))
        force[i] = float(np.clip(action.force, 0.0, 1.0))
        kick_score[i] = float(np.clip(action.kick_score, 0.0, 1.0))
        confidence[i] = float(np.clip(max(action.kick_score, action.force, abs(action.side_score)), 0.0, 1.0))
        if action.side == "left":
            side_class[i] = 1
        elif action.side == "right":
            side_class[i] = 2

    condition_id = np.asarray([_condition_for_row(int(row)) for row in rows], dtype=np.int32)
    calcium_time_s = rows.astype(np.float32) / np.float32(ZAPBENCH_FRAME_HZ)
    metadata = {
        "source": "ZAPBench calcium traces decoded with direct-ephys tail-action decoder",
        "trace_tensorstore": TRACE_SPEC["kvstore"],
        "artifact_path": str(artifact_path),
        "frame_hz": ZAPBENCH_FRAME_HZ,
        "condition_names": CONDITION_NAMES,
        "condition_offsets": CONDITION_OFFSETS,
        "side_class": {"0": "none", "1": "left", "2": "right"},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        rows=rows,
        calcium_time_s=calcium_time_s,
        condition_id=condition_id,
        kick=kick,
        side_score=side_score,
        side_class=side_class,
        force=force,
        kick_score=kick_score,
        confidence=confidence,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = summarize_replay(output_path)
    report_path = output_path.with_suffix(".md")
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return {**summary, "output_path": str(output_path), "report_path": str(report_path)}


def summarize_replay(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        rows = np.asarray(data["rows"], dtype=np.int32)
        kick = np.asarray(data["kick"], dtype=np.float32)
        side_score = np.asarray(data["side_score"], dtype=np.float32)
        side_class = np.asarray(data["side_class"], dtype=np.int32)
        force = np.asarray(data["force"], dtype=np.float32)
        kick_score = np.asarray(data["kick_score"], dtype=np.float32)
        confidence = np.asarray(data["confidence"], dtype=np.float32)
        condition_id = np.asarray(data["condition_id"], dtype=np.int32)
        metadata = json.loads(str(data["metadata_json"]))

    per_condition: dict[str, Any] = {}
    for i, name in enumerate(metadata["condition_names"]):
        mask = condition_id == i
        if not np.any(mask):
            continue
        per_condition[name] = {
            "frames": int(np.sum(mask)),
            "kick_rate": float(np.mean(kick[mask] > 0.5)),
            "side_active_rate": float(np.mean(side_class[mask] != 0)),
            "force": _stats(force[mask]),
            "kick_score": _stats(kick_score[mask]),
        }
    return {
        "path": str(path),
        "frames": int(rows.shape[0]),
        "duration_s": float(rows.shape[0] / ZAPBENCH_FRAME_HZ),
        "kick_rate": float(np.mean(kick > 0.5)),
        "side_active_rate": float(np.mean(side_class != 0)),
        "side_left_rate": float(np.mean(side_class == 1)),
        "side_right_rate": float(np.mean(side_class == 2)),
        "force": _stats(force),
        "kick_score": _stats(kick_score),
        "confidence": _stats(confidence),
        "side_score": _stats(side_score),
        "metadata": metadata,
        "per_condition": per_condition,
    }


def simulate_replay(replay_path: Path, seconds: float, condition: str, seed: int) -> dict[str, Any]:
    from simulations.zebrafish.simulation import build_zebrafish_simulation

    with np.load(replay_path, allow_pickle=False) as data:
        rows = np.asarray(data["rows"], dtype=np.int32)
        calcium_time_s = np.asarray(data["calcium_time_s"], dtype=np.float32)
        condition_id = np.asarray(data["condition_id"], dtype=np.int32)
        kick = np.asarray(data["kick"], dtype=np.float32)
        side_score = np.asarray(data["side_score"], dtype=np.float32)
        side_class = np.asarray(data["side_class"], dtype=np.int32)
        force = np.asarray(data["force"], dtype=np.float32)
        kick_score = np.asarray(data["kick_score"], dtype=np.float32)
        confidence = np.asarray(data["confidence"], dtype=np.float32)
        metadata = json.loads(str(data["metadata_json"]))

    if condition == "all":
        indices = np.arange(rows.shape[0], dtype=np.int32)
    else:
        try:
            cond_id = list(metadata["condition_names"]).index(condition)
        except ValueError as exc:
            raise SystemExit(f"unknown condition {condition!r}; choose one of {metadata['condition_names']}") from exc
        indices = np.flatnonzero(condition_id == cond_id).astype(np.int32)
    max_frames = min(indices.shape[0], max(1, int(math.ceil(seconds * ZAPBENCH_FRAME_HZ))))
    indices = indices[:max_frames]

    engine, loop = build_zebrafish_simulation(
        food_positions=[],
        log_level="ERROR",
        record_neural_states=False,
        max_history=8,
        suppress_connectome_summary=True,
        seed=seed,
    )
    engine.reset(nervous_rebuild=False)
    env = engine.environment
    env.set_calcium_stimulus(enabled=True, gain=1.0, source=f"zapbench:{condition}")

    speed: list[float] = []
    turn: list[float] = []
    swim: list[float] = []
    sim_force: list[float] = []
    fe: list[float] = []
    for frame_index, idx in enumerate(indices):
        env.push_calcium_action_frame(
            {
                "enabled": True,
                "source": f"zapbench:{condition}",
                "row": int(rows[idx]),
                "frame_index": int(frame_index),
                "calcium_time_s": float(calcium_time_s[idx]),
                "kick": float(kick[idx]),
                "side_score": float(side_score[idx]),
                "force": float(force[idx]),
                "kick_score": float(kick_score[idx]),
                "confidence": float(confidence[idx]),
            }
        )
        for _ in range(_ticks_for_zapbench_frame(frame_index)):
            step = engine.step()
            extra = step.body_state.extra or {}
            speed.append(float(extra.get("speed_m_s", 0.0)) * 1000.0)
            state = engine.nervous_system.behavior_state
            swim.append(float(state.get("swim_drive", 0.0)))
            turn.append(float(state.get("turn_bias", 0.0)))
            sim_force.append(float(force[idx]))
            if loop.log_free_energy and loop.free_energy_trace.prediction_error:
                fe.append(float(loop.free_energy_trace.prediction_error[-1]))
    swim_arr = np.asarray(swim, dtype=np.float32)
    turn_arr = np.asarray(turn, dtype=np.float32)
    speed_arr = np.asarray(speed, dtype=np.float32)
    force_arr = np.asarray(sim_force, dtype=np.float32)
    return {
        "condition": condition,
        "frames": int(indices.shape[0]),
        "ticks": int(swim_arr.shape[0]),
        "seconds": float(swim_arr.shape[0] * PHYSICS_DT_S),
        "ticks_per_frame": float(1.0 / (ZAPBENCH_FRAME_HZ * PHYSICS_DT_S)),
        "decoded_kick_rate": float(np.mean(kick[indices] > 0.5)) if indices.size else 0.0,
        "simulated_swim_rate": float(np.mean(swim_arr > 0.12)) if swim_arr.size else 0.0,
        "decoded_side_active_rate": float(np.mean(side_class[indices] != 0)) if indices.size else 0.0,
        "simulated_side_active_rate": float(np.mean(np.abs(turn_arr) > 0.065)) if turn_arr.size else 0.0,
        "decoded_force": _stats(force_arr),
        "speed_mm_s": _stats(speed_arr),
        "turn_bias": _stats(turn_arr),
        "swim_drive": _stats(swim_arr),
        "free_energy": _stats(np.asarray(fe, dtype=np.float32)),
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# ZAPBench Calcium Action Replay",
        "",
        "This artifact decodes public ZAPBench calcium trace windows into tail-action commands.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps({k: v for k, v in summary.items() if k != "per_condition"}, indent=2, sort_keys=True),
        "```",
        "",
        "## Per Condition",
        "",
        "| condition | frames | kick rate | side-active rate | force mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in summary["per_condition"].items():
        lines.append(
            f"| {name} | {item['frames']} | {item['kick_rate']:.3f} | "
            f"{item['side_active_rate']:.3f} | {item['force']['mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=ACTIVE_INFERENCE / "simulations" / "zebrafish" / "data" / "zapbench_ephys_tail_action_decoder.npz",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=SCRIPT_DIR / "cache" / "zapbench" / "zapbench_calcium_action_replay.npz",
    )
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--simulate-seconds", type=float, default=0.0)
    parser.add_argument("--simulate-condition", default="turning")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.skip_generate:
        summary = summarize_replay(args.output_path)
    else:
        summary = generate_replay(args.artifact_path, args.output_path)
    if args.simulate_seconds > 0:
        summary["simulation"] = simulate_replay(
            args.output_path,
            seconds=args.simulate_seconds,
            condition=args.simulate_condition,
            seed=args.seed,
        )
        sim_report = args.output_path.with_name(args.output_path.stem + "_simulation.json")
        sim_report.write_text(json.dumps(summary["simulation"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["simulation_path"] = str(sim_report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
