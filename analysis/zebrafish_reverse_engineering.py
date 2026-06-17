"""Reverse-engineering metrics for the zebrafish live simulation.

This script is intentionally independent of the UI. It can analyze either the
live lab WebSocket stream or a fresh local simulation run, then prints compact
JSON metrics that expose body mechanics, motor drive, neural activity, and
behavioral state.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


JOINT_SCALE = 1e4
MUSCLE_SCALE = 1e4
NEURAL_SCALE = 1e4
MM_SCALE = 1e6


@dataclass
class Frame:
    tick: int
    segments_mm: np.ndarray
    com_mm: np.ndarray
    tail_yaw: np.ndarray
    tail_pitch: np.ndarray
    joint_angles: np.ndarray
    joint_velocities: np.ndarray
    muscle_activations: np.ndarray
    neuron_s: np.ndarray
    neuron_r: np.ndarray
    fired: np.ndarray
    neuromod: np.ndarray
    free_energy: float
    behavior: dict[str, float] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)


def _decode_bits(b64: str, n: int) -> np.ndarray:
    if not b64 or n <= 0:
        return np.zeros(n, dtype=np.float64)
    raw = base64.b64decode(b64)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out[i] = 1.0 if raw[i >> 3] & (1 << (i & 7)) else 0.0
    return out


def _scaled(values: Any, scale: float) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(values, dtype=np.float64) / scale


def _segments_from_wire(raw: Any) -> np.ndarray:
    arr = np.asarray(raw or [], dtype=np.float64)
    if arr.size < 3:
        return np.zeros((0, 3), dtype=np.float64)
    return arr.reshape((-1, 3)) / MM_SCALE


def _frame_from_wire(payload: dict[str, Any], names: list[str]) -> Frame:
    n_neurons = len(names)
    return Frame(
        tick=int(payload.get("k", 0)),
        segments_mm=_segments_from_wire(payload.get("sm")),
        com_mm=np.asarray(payload.get("cm", [0, 0, 0]), dtype=np.float64) / MM_SCALE,
        tail_yaw=_scaled(payload.get("ta"), JOINT_SCALE),
        tail_pitch=_scaled(payload.get("tpa"), JOINT_SCALE),
        joint_angles=_scaled(payload.get("ja"), JOINT_SCALE),
        joint_velocities=_scaled(payload.get("jv"), JOINT_SCALE),
        muscle_activations=_scaled(payload.get("ma"), MUSCLE_SCALE),
        neuron_s=_scaled(payload.get("Si"), NEURAL_SCALE),
        neuron_r=_scaled(payload.get("Ri"), NEURAL_SCALE),
        fired=_decode_bits(str(payload.get("Fb", "")), n_neurons),
        neuromod=np.asarray(payload.get("nm01", [0.0, 0.0]), dtype=np.float64),
        free_energy=float(payload.get("fe", 0.0)),
        names=names,
    )


async def capture_live(url: str, frames_target: int, timeout_s: float) -> list[Frame]:
    import websockets

    names: list[str] = []
    frames: list[Frame] = []
    async with websockets.connect(url, open_timeout=timeout_s) as ws:
        while len(frames) < frames_target:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            payload = json.loads(raw)
            if payload.get("t") == "h":
                names = list((payload.get("L") or {}).get("nm") or [])
                continue
            if payload.get("t") == "s":
                frames.append(_frame_from_wire(payload, names))
    return frames


def run_offline(steps: int, seed: int | None, no_cpg: bool, sample_every: int = 1) -> list[Frame]:
    from simulations.zebrafish.simulation import build_zebrafish_simulation

    engine, _loop = build_zebrafish_simulation(
        food_positions=[],
        log_level="ERROR",
        record_neural_states=False,
        max_history=16,
        suppress_connectome_summary=True,
        seed=seed,
    )
    ns = engine.nervous_system
    if no_cpg:
        ns.tail_cpg_enabled = False
        ns.spontaneous_bout_prob = 0.0
        ns.baseline_drive = 0.0
    names = ns.get_neuron_names_paula_order()
    frames: list[Frame] = []
    sample_every = max(1, int(sample_every))
    for i in range(1, steps + 1):
        step = engine.step()
        if i % sample_every != 0:
            continue
        state = step.body_state
        extra = state.extra or {}
        segments = np.asarray(extra.get("tail_points", []), dtype=np.float64) * 1000.0
        if segments.size == 0:
            segments = engine.body.get_body_shape() * 1000.0
        joint_names = engine.body.joint_names
        muscle_names = engine.body.muscle_names
        neuron_s = np.zeros(len(names), dtype=np.float64)
        fired = np.zeros(len(names), dtype=np.float64)
        neuron_r = np.zeros(len(names), dtype=np.float64)
        if ns._network is not None:  # type: ignore[attr-defined]
            neurons = ns._network.network.neurons  # type: ignore[attr-defined]
            for nid, neuron in neurons.items():
                idx = int(nid)
                if idx < len(neuron_s):
                    neuron_s[idx] = float(neuron.S)
                    fired[idx] = 1.0 if float(neuron.O) > 0 else 0.0
                    neuron_r[idx] = float(neuron.r)
        frames.append(
            Frame(
                tick=int(step.tick),
                segments_mm=np.asarray(segments, dtype=np.float64).reshape((-1, 3)),
                com_mm=np.asarray(state.position, dtype=np.float64) * 1000.0,
                tail_yaw=np.asarray(extra.get("tail_angles", []), dtype=np.float64),
                tail_pitch=np.asarray(extra.get("tail_pitch_angles", []), dtype=np.float64),
                joint_angles=np.asarray(
                    [state.joint_angles.get(n, 0.0) for n in joint_names],
                    dtype=np.float64,
                ),
                joint_velocities=np.asarray(
                    [state.joint_velocities.get(n, 0.0) for n in joint_names],
                    dtype=np.float64,
                ),
                muscle_activations=np.asarray(
                    [step.motor_outputs.get(n, 0.0) for n in muscle_names],
                    dtype=np.float64,
                ),
                neuron_s=neuron_s,
                neuron_r=neuron_r,
                fired=fired,
                neuromod=np.asarray(ns.neuromod_levels, dtype=np.float64),
                free_energy=0.0,
                behavior={k: float(v) for k, v in ns.behavior_state.items()},
                names=names,
            )
        )
    return frames


def _path_metrics(points: np.ndarray) -> dict[str, float]:
    if points.shape[0] < 3:
        return {}
    diffs = np.diff(points, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    path = float(np.sum(lengths))
    chord = float(np.linalg.norm(points[0] - points[-1]))
    tangents = diffs[:, :2] / np.maximum(np.linalg.norm(diffs[:, :2], axis=1, keepdims=True), 1e-12)
    angle_deltas: list[float] = []
    for a, b in zip(tangents[:-1], tangents[1:]):
        cross = float(a[0] * b[1] - a[1] * b[0])
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        angle_deltas.append(math.atan2(cross, dot))
    local_abs = np.abs(np.asarray(angle_deltas, dtype=np.float64))
    tangent3 = diffs / np.maximum(lengths[:, None], 1e-12)
    local3: list[float] = []
    for a, b in zip(tangent3[:-1], tangent3[1:]):
        local3.append(float(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))))
    local3_abs = np.asarray(local3, dtype=np.float64)
    min_non_neighbor = math.inf
    for i in range(points.shape[0]):
        for j in range(i + 3, points.shape[0]):
            min_non_neighbor = min(min_non_neighbor, float(np.linalg.norm(points[i] - points[j])))
    axis = points[0] - points[-1]
    xy_chord = float(np.linalg.norm(axis[:2]))
    body_pitch = math.atan2(float(axis[2]), max(xy_chord, 1e-12))
    return {
        "path_mm": path,
        "chord_mm": chord,
        "xy_chord_mm": xy_chord,
        "straightness": chord / path if path > 1e-12 else 0.0,
        "signed_curvature_rad": float(np.sum(angle_deltas)),
        "abs_curvature_rad": float(np.sum(local_abs)),
        "max_local_bend_rad": float(np.max(local_abs)) if local_abs.size else 0.0,
        "abs_curvature_3d_rad": float(np.sum(local3_abs)),
        "max_local_bend_3d_rad": float(np.max(local3_abs)) if local3_abs.size else 0.0,
        "min_non_neighbor_distance_mm": min_non_neighbor if math.isfinite(min_non_neighbor) else 0.0,
        "z_span_mm": float(np.ptp(points[:, 2])),
        "body_pitch_rad": float(body_pitch),
    }


def _stats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _muscle_groups(ma: np.ndarray) -> dict[str, np.ndarray]:
    if ma.size == 0:
        z = np.zeros(0, dtype=np.float64)
        return {"left": z, "right": z, "dorsal": z, "ventral": z}
    trimmed = ma[: (ma.size // 4) * 4].reshape((-1, 4))
    return {
        "left": trimmed[:, 0],
        "right": trimmed[:, 1],
        "dorsal": trimmed[:, 2],
        "ventral": trimmed[:, 3],
    }


def summarize(frames: list[Frame]) -> dict[str, Any]:
    if not frames:
        return {"frames": 0}
    path_rows = [_path_metrics(f.segments_mm) for f in frames if f.segments_mm.size]
    yaw_abs = [float(np.max(np.abs(f.tail_yaw))) for f in frames if f.tail_yaw.size]
    pitch_abs = [float(np.max(np.abs(f.tail_pitch))) for f in frames if f.tail_pitch.size]
    yaw_sum_abs = [float(np.sum(np.abs(f.tail_yaw))) for f in frames if f.tail_yaw.size]
    yaw_limit_frac = [
        float(np.mean(np.abs(f.tail_yaw) > 0.54))
        for f in frames
        if f.tail_yaw.size
    ]
    com = np.asarray([f.com_mm for f in frames], dtype=np.float64)
    tick_span = max(1, frames[-1].tick - frames[0].tick)
    speeds = np.linalg.norm(np.diff(com[:, :2], axis=0), axis=1) / np.maximum(np.diff([f.tick for f in frames]), 1) / 0.005
    group_accum: dict[str, list[float]] = defaultdict(list)
    signed_front: list[float] = []
    drive: list[float] = []
    for f in frames:
        groups = _muscle_groups(f.muscle_activations)
        for key, arr in groups.items():
            if arr.size:
                group_accum[key].append(float(np.mean(arr)))
        if groups["left"].size and groups["right"].size:
            signed = groups["right"] - groups["left"]
            signed_front.append(float(np.mean(signed[: max(3, signed.size // 3)])))
            drive.append(float(np.mean(groups["left"] + groups["right"])))
    fired_counts = [float(np.sum(f.fired)) for f in frames if f.fired.size]
    s_means = [float(np.mean(f.neuron_s)) for f in frames if f.neuron_s.size]
    s_max = [float(np.max(f.neuron_s)) for f in frames if f.neuron_s.size]
    top_fired: list[dict[str, Any]] = []
    if frames[0].names:
        counts = np.zeros(len(frames[0].names), dtype=np.float64)
        mean_s = np.zeros(len(frames[0].names), dtype=np.float64)
        seen = 0
        for f in frames:
            if f.fired.size == counts.size:
                counts += f.fired
            if f.neuron_s.size == mean_s.size:
                mean_s += f.neuron_s
                seen += 1
        if seen:
            mean_s /= seen
        idxs = np.argsort(-(counts + 0.01 * mean_s))[:12]
        top_fired = [
            {
                "name": frames[0].names[int(i)],
                "fire_fraction": float(counts[int(i)] / max(1, len(frames))),
                "mean_S": float(mean_s[int(i)]),
            }
            for i in idxs
        ]
    behavior = {
        "xy_speed_mm_s": _stats(speeds),
        "depth_mm": _stats(com[:, 2]),
        "travel_mm": float(np.linalg.norm(com[-1, :2] - com[0, :2])),
        "m0": _stats([f.neuromod[0] for f in frames if f.neuromod.size >= 2]),
        "m1": _stats([f.neuromod[1] for f in frames if f.neuromod.size >= 2]),
        "free_energy": _stats([f.free_energy for f in frames]),
    }
    if any(f.behavior for f in frames):
        behavior.update(
            {
                "swim_drive_state": _stats([f.behavior.get("swim_drive", 0.0) for f in frames if f.behavior]),
                "turn_bias_state": _stats([f.behavior.get("turn_bias", 0.0) for f in frames if f.behavior]),
                "pitch_bias_state": _stats([f.behavior.get("pitch_bias", 0.0) for f in frames if f.behavior]),
            }
        )
    return {
        "frames": len(frames),
        "tick_start": frames[0].tick,
        "tick_end": frames[-1].tick,
        "tick_span": tick_span,
        "body": {
            key: _stats([row[key] for row in path_rows if key in row])
            for key in (
                "path_mm",
                "chord_mm",
                "xy_chord_mm",
                "straightness",
                "signed_curvature_rad",
                "abs_curvature_rad",
                "max_local_bend_rad",
                "abs_curvature_3d_rad",
                "max_local_bend_3d_rad",
                "min_non_neighbor_distance_mm",
                "z_span_mm",
                "body_pitch_rad",
            )
        },
        "joints": {
            "max_abs_tail_yaw_rad": _stats(yaw_abs),
            "max_abs_tail_pitch_rad": _stats(pitch_abs),
            "sum_abs_tail_yaw_rad": _stats(yaw_sum_abs),
            "fraction_yaw_joints_above_0p54_rad": _stats(yaw_limit_frac),
        },
        "behavior": behavior,
        "motors": {
            "mean_left": _stats(group_accum["left"]),
            "mean_right": _stats(group_accum["right"]),
            "mean_dorsal": _stats(group_accum["dorsal"]),
            "mean_ventral": _stats(group_accum["ventral"]),
            "front_right_minus_left": _stats(signed_front),
            "axial_drive": _stats(drive),
        },
        "neural": {
            "fired_count": _stats(fired_counts),
            "mean_S": _stats(s_means),
            "max_S": _stats(s_max),
            "top_fired_or_active": top_fired,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["live", "offline"], default="live")
    parser.add_argument("--url", default="ws://127.0.0.1:8811/ws/state")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--offline-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-cpg", action="store_true")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="For offline runs, keep every Nth frame. Useful for long stability traces.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.source == "live":
        frames = asyncio.run(capture_live(args.url, args.frames, args.timeout_s))
    else:
        frames = run_offline(args.offline_steps, args.seed, args.no_cpg, args.sample_every)
    summary = summarize(frames)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
