"""High-fidelity long-run recorder for the zebrafish digital twin.

The output is a compressed NumPy archive containing per-tick MuJoCo state,
body geometry, muscle activations, PAULA state vectors, and derived behavioral
metrics. A companion JSON summary is written next to the archive.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from zebrafish_reverse_engineering import _path_metrics, _stats


DT_S = 0.005


def _heading_from_points(points_mm: np.ndarray) -> float:
    if points_mm.shape[0] < 2:
        return 0.0
    axis = points_mm[0] - points_mm[-1]
    return float(math.atan2(float(axis[1]), float(axis[0])))


def _angle_wrap(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _contiguous_events(mask: np.ndarray) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if bool(value) and start is None:
            start = i
        elif not bool(value) and start is not None:
            events.append((start, i))
            start = None
    if start is not None:
        events.append((start, int(mask.size)))
    return events


def _event_stats(events: list[tuple[int, int]], ticks: np.ndarray) -> dict[str, Any]:
    if not events:
        return {
            "count": 0,
            "rate_hz": 0.0,
            "duration_ms": _stats([]),
            "inter_event_interval_ms": _stats([]),
        }
    durations = np.asarray([(b - a) * DT_S * 1000.0 for a, b in events], dtype=np.float64)
    starts = np.asarray([ticks[a] for a, _ in events], dtype=np.float64)
    ibis = np.diff(starts) * DT_S * 1000.0
    total_s = max(1e-12, (float(ticks[-1]) - float(ticks[0]) + 1.0) * DT_S)
    return {
        "count": int(len(events)),
        "rate_hz": float(len(events) / total_s),
        "duration_ms": _stats(durations),
        "inter_event_interval_ms": _stats(ibis),
    }


def _motif_summary(
    ticks: np.ndarray,
    swim_drive: np.ndarray,
    turn_bias: np.ndarray,
    pitch_bias: np.ndarray,
    speed_mm_s: np.ndarray,
    body_pitch_rad: np.ndarray,
    max_tail_yaw_rad: np.ndarray,
) -> dict[str, Any]:
    bout_events = _contiguous_events(swim_drive > 0.12)
    turn_events = _contiguous_events((swim_drive > 0.12) & (np.abs(turn_bias) > 0.18))
    climb_events = _contiguous_events((swim_drive > 0.12) & (pitch_bias > 0.12))
    dive_events = _contiguous_events((swim_drive > 0.12) & (pitch_bias < -0.12))
    c_or_o_events = _contiguous_events((swim_drive > 0.20) & (max_tail_yaw_rad > 0.18))
    vertical_instability = _contiguous_events(np.abs(body_pitch_rad) > 0.65)

    out: dict[str, Any] = {
        "swim_bouts": _event_stats(bout_events, ticks),
        "turn_bouts": _event_stats(turn_events, ticks),
        "climb_bouts": _event_stats(climb_events, ticks),
        "dive_bouts": _event_stats(dive_events, ticks),
        "high_bend_c_or_o_like_bouts": _event_stats(c_or_o_events, ticks),
        "vertical_instability_events": _event_stats(vertical_instability, ticks),
    }
    if bout_events:
        bout_peak_speeds = [float(np.max(speed_mm_s[a:b])) for a, b in bout_events if b > a]
        bout_peak_bends = [float(np.max(max_tail_yaw_rad[a:b])) for a, b in bout_events if b > a]
        out["bout_peak_speed_mm_s"] = _stats(bout_peak_speeds)
        out["bout_peak_tail_yaw_rad"] = _stats(bout_peak_bends)
    else:
        out["bout_peak_speed_mm_s"] = _stats([])
        out["bout_peak_tail_yaw_rad"] = _stats([])
    return out


def _apply_protocol(engine: Any, tick_index: int, protocol: str) -> tuple[int, bool, tuple[float, float], bool]:
    """Apply deterministic no-food stimulus protocols.

    Returns a compact stimulus record: phase code, startle command, water flow,
    and whether a depth perturbation was applied on this tick.
    """

    if protocol == "baseline":
        return 0, False, (0.0, 0.0), False
    if protocol != "sensorimotor_battery":
        raise ValueError(f"unknown protocol: {protocol}")

    env = engine.environment
    body = engine.body
    phase = tick_index % 40_000
    code = 0
    startle = False
    flow = (0.0, 0.0)
    depth_perturb = False

    if 5_000 <= phase < 10_000:
        code = 1
        if phase in {5_000, 6_500, 8_000}:
            env.trigger_startle(duration_ticks=24)
            startle = True
    elif 10_000 <= phase < 18_000:
        code = 2
        flow_x = 0.012 if phase < 14_000 else -0.012
        flow_y = 0.004 if (phase // 500) % 2 == 0 else -0.004
        env.set_water_flow(flow_x, flow_y)
        flow = (flow_x, flow_y)
    elif 18_000 <= phase < 28_000:
        code = 3
        if phase in {18_000, 23_000}:
            qpos, qvel = body.export_mujoco_state()
            qpos[2] = -0.00055 if phase == 18_000 else -0.0062
            qvel[2] = 0.0
            body.import_mujoco_state(qpos, qvel)
            depth_perturb = True
    elif 28_000 <= phase < 34_000:
        code = 4
        flow_y = 0.010 if phase < 31_000 else -0.010
        env.set_water_flow(0.0, flow_y)
        flow = (0.0, flow_y)
    else:
        env.set_water_flow(0.0, 0.0)

    return code, startle, flow, depth_perturb


def record_run(
    ticks: int,
    seed: int,
    out_dir: Path,
    label: str,
    log_every: int,
    protocol: str,
) -> dict[str, Any]:
    from simulations.zebrafish.simulation import build_zebrafish_simulation

    out_dir.mkdir(parents=True, exist_ok=True)
    engine, _loop = build_zebrafish_simulation(
        food_positions=[],
        log_level="ERROR",
        record_neural_states=False,
        max_history=8,
        suppress_connectome_summary=True,
        seed=seed,
    )
    engine.reset(nervous_rebuild=False)
    ns = engine.nervous_system
    body = engine.body
    model = body.model
    data = body.data

    neuron_names = ns.get_neuron_names_paula_order()
    muscle_names = list(body.muscle_names)
    joint_names = list(body.joint_names)
    n_body_points = 1 + len([name for name in joint_names if name.startswith("tail_yaw_")])

    ticks_arr = np.zeros(ticks, dtype=np.int32)
    qpos = np.zeros((ticks, model.nq), dtype=np.float32)
    qvel = np.zeros((ticks, model.nv), dtype=np.float32)
    com_mm = np.zeros((ticks, 3), dtype=np.float32)
    body_points_mm = np.zeros((ticks, n_body_points, 3), dtype=np.float32)
    tail_yaw_rad = np.zeros((ticks, n_body_points - 1), dtype=np.float32)
    tail_pitch_rad = np.zeros((ticks, n_body_points - 1), dtype=np.float32)
    muscles = np.zeros((ticks, len(muscle_names)), dtype=np.float32)
    neuron_s = np.zeros((ticks, len(neuron_names)), dtype=np.float32)
    neuron_r = np.zeros((ticks, len(neuron_names)), dtype=np.float32)
    neuron_o = np.zeros((ticks, len(neuron_names)), dtype=np.float32)
    neuron_fired = np.zeros((ticks, len(neuron_names)), dtype=np.bool_)
    neuromod = np.zeros((ticks, 2), dtype=np.float32)
    swim_drive = np.zeros(ticks, dtype=np.float32)
    turn_bias = np.zeros(ticks, dtype=np.float32)
    pitch_bias = np.zeros(ticks, dtype=np.float32)
    speed_mm_s = np.zeros(ticks, dtype=np.float32)
    vertical_speed_mm_s = np.zeros(ticks, dtype=np.float32)
    heading_rad = np.zeros(ticks, dtype=np.float32)
    yaw_rate_rad_s = np.zeros(ticks, dtype=np.float32)
    body_pitch_rad = np.zeros(ticks, dtype=np.float32)
    z_span_mm = np.zeros(ticks, dtype=np.float32)
    straightness = np.zeros(ticks, dtype=np.float32)
    abs_curvature_3d_rad = np.zeros(ticks, dtype=np.float32)
    max_local_bend_3d_rad = np.zeros(ticks, dtype=np.float32)
    stimulus_code = np.zeros(ticks, dtype=np.int16)
    startle_command = np.zeros(ticks, dtype=np.bool_)
    water_flow_m_s = np.zeros((ticks, 2), dtype=np.float32)
    depth_perturbation = np.zeros(ticks, dtype=np.bool_)

    t0 = time.perf_counter()
    last_heading: float | None = None
    for i in range(ticks):
        code, startle, flow, depth_perturb = _apply_protocol(engine, i, protocol)
        step = engine.step()
        st = step.body_state
        extra = st.extra or {}
        points = np.asarray(extra.get("tail_points", body.get_body_shape()), dtype=np.float64) * 1000.0
        if points.shape != (n_body_points, 3):
            fixed = np.zeros((n_body_points, 3), dtype=np.float64)
            fixed[: min(n_body_points, points.shape[0])] = points[:n_body_points]
            points = fixed
        metrics = _path_metrics(points)
        heading = _heading_from_points(points)
        if last_heading is None:
            yaw_rate = 0.0
        else:
            yaw_rate = float(_angle_wrap(np.asarray([heading - last_heading]))[0] / DT_S)
        last_heading = heading

        ticks_arr[i] = int(step.tick)
        qpos[i] = np.asarray(data.qpos, dtype=np.float32)
        qvel[i] = np.asarray(data.qvel, dtype=np.float32)
        com_mm[i] = np.asarray(st.position, dtype=np.float64) * 1000.0
        body_points_mm[i] = points
        tail_yaw_rad[i] = np.asarray(extra.get("tail_angles", []), dtype=np.float32)
        tail_pitch_rad[i] = np.asarray(extra.get("tail_pitch_angles", []), dtype=np.float32)
        muscles[i] = np.asarray([step.motor_outputs.get(name, 0.0) for name in muscle_names], dtype=np.float32)
        neuromod[i] = np.asarray(ns.neuromod_levels, dtype=np.float32)
        swim_drive[i] = float(extra.get("swim_drive", ns.behavior_state.get("swim_drive", 0.0)))
        turn_bias[i] = float(extra.get("turn_bias", ns.behavior_state.get("turn_bias", 0.0)))
        pitch_bias[i] = float(extra.get("pitch_bias", ns.behavior_state.get("pitch_bias", 0.0)))
        speed_mm_s[i] = float(extra.get("speed_m_s", 0.0)) * 1000.0
        vertical_speed_mm_s[i] = float(extra.get("vertical_velocity_m_s", 0.0)) * 1000.0
        heading_rad[i] = heading
        yaw_rate_rad_s[i] = yaw_rate
        body_pitch_rad[i] = float(metrics.get("body_pitch_rad", 0.0))
        z_span_mm[i] = float(metrics.get("z_span_mm", 0.0))
        straightness[i] = float(metrics.get("straightness", 0.0))
        abs_curvature_3d_rad[i] = float(metrics.get("abs_curvature_3d_rad", 0.0))
        max_local_bend_3d_rad[i] = float(metrics.get("max_local_bend_3d_rad", 0.0))
        stimulus_code[i] = int(code)
        startle_command[i] = bool(startle)
        water_flow_m_s[i] = np.asarray(flow, dtype=np.float32)
        depth_perturbation[i] = bool(depth_perturb)

        if ns._network is not None:  # type: ignore[attr-defined]
            neurons = ns._network.network.neurons  # type: ignore[attr-defined]
            for nid, neuron in neurons.items():
                j = int(nid)
                if j >= len(neuron_names):
                    continue
                neuron_s[i, j] = float(neuron.S)
                neuron_r[i, j] = float(neuron.r)
                neuron_o[i, j] = float(neuron.O)
                neuron_fired[i, j] = float(neuron.O) > 0.0

        if log_every > 0 and (i + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            print(
                json.dumps(
                    {
                        "label": label,
                        "seed": seed,
                        "protocol": protocol,
                        "tick": int(step.tick),
                        "elapsed_s": round(elapsed, 3),
                        "speed_mm_s": round(float(speed_mm_s[i]), 3),
                        "body_pitch_rad": round(float(body_pitch_rad[i]), 4),
                        "z_span_mm": round(float(z_span_mm[i]), 4),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    wall_s = time.perf_counter() - t0
    max_tail_yaw_rad = np.max(np.abs(tail_yaw_rad), axis=1) if tail_yaw_rad.size else np.zeros(ticks)
    summary: dict[str, Any] = {
        "label": label,
        "protocol": protocol,
        "seed": int(seed),
        "ticks": int(ticks),
        "dt_s": DT_S,
        "simulated_seconds": float(ticks * DT_S),
        "wall_seconds": float(wall_s),
        "wall_ticks_per_second": float(ticks / max(1e-12, wall_s)),
        "neuron_names": neuron_names,
        "muscle_names": muscle_names,
        "joint_names": joint_names,
        "kinematics": {
            "speed_mm_s": _stats(speed_mm_s),
            "vertical_speed_mm_s": _stats(vertical_speed_mm_s),
            "yaw_rate_rad_s": _stats(yaw_rate_rad_s),
            "body_pitch_rad": _stats(body_pitch_rad),
            "z_span_mm": _stats(z_span_mm),
            "straightness": _stats(straightness),
            "abs_curvature_3d_rad": _stats(abs_curvature_3d_rad),
            "max_local_bend_3d_rad": _stats(max_local_bend_3d_rad),
            "max_abs_tail_yaw_rad": _stats(max_tail_yaw_rad),
            "depth_mm": _stats(com_mm[:, 2]),
            "travel_mm": float(np.linalg.norm(com_mm[-1, :2] - com_mm[0, :2])),
        },
        "control": {
            "swim_drive": _stats(swim_drive),
            "turn_bias": _stats(turn_bias),
            "pitch_bias": _stats(pitch_bias),
            "neuromod_m0": _stats(neuromod[:, 0]),
            "neuromod_m1": _stats(neuromod[:, 1]),
        },
        "stimulus": {
            "startle_commands": int(np.sum(startle_command)),
            "depth_perturbations": int(np.sum(depth_perturbation)),
            "flow_x_m_s": _stats(water_flow_m_s[:, 0]),
            "flow_y_m_s": _stats(water_flow_m_s[:, 1]),
            "phase_code_counts": {
                str(int(code)): int(np.sum(stimulus_code == code))
                for code in np.unique(stimulus_code)
            },
        },
        "neural": {
            "mean_S": _stats(np.mean(neuron_s, axis=1)),
            "max_S": _stats(np.max(neuron_s, axis=1)),
            "fired_count": _stats(np.sum(neuron_fired, axis=1)),
            "mean_r": _stats(np.mean(neuron_r, axis=1)),
            "top_active": [],
        },
        "motifs": _motif_summary(
            ticks_arr,
            swim_drive,
            turn_bias,
            pitch_bias,
            speed_mm_s,
            body_pitch_rad,
            max_tail_yaw_rad,
        ),
    }
    if neuron_names:
        fire_fraction = np.mean(neuron_fired, axis=0)
        mean_s = np.mean(neuron_s, axis=0)
        order = np.argsort(-(fire_fraction + 0.001 * mean_s))[:16]
        summary["neural"]["top_active"] = [
            {
                "name": neuron_names[int(i)],
                "fire_fraction": float(fire_fraction[int(i)]),
                "mean_S": float(mean_s[int(i)]),
            }
            for i in order
        ]

    archive_path = out_dir / f"{label}_seed{seed}_{ticks}ticks.npz"
    summary_path = out_dir / f"{label}_seed{seed}_{ticks}ticks_summary.json"
    np.savez_compressed(
        archive_path,
        ticks=ticks_arr,
        qpos=qpos,
        qvel=qvel,
        com_mm=com_mm,
        body_points_mm=body_points_mm,
        tail_yaw_rad=tail_yaw_rad,
        tail_pitch_rad=tail_pitch_rad,
        muscles=muscles,
        neuron_s=neuron_s,
        neuron_r=neuron_r,
        neuron_o=neuron_o,
        neuron_fired=neuron_fired,
        neuromod=neuromod,
        swim_drive=swim_drive,
        turn_bias=turn_bias,
        pitch_bias=pitch_bias,
        speed_mm_s=speed_mm_s,
        vertical_speed_mm_s=vertical_speed_mm_s,
        heading_rad=heading_rad,
        yaw_rate_rad_s=yaw_rate_rad_s,
        body_pitch_rad=body_pitch_rad,
        z_span_mm=z_span_mm,
        straightness=straightness,
        abs_curvature_3d_rad=abs_curvature_3d_rad,
        max_local_bend_3d_rad=max_local_bend_3d_rad,
        stimulus_code=stimulus_code,
        startle_command=startle_command,
        water_flow_m_s=water_flow_m_s,
        depth_perturbation=depth_perturbation,
        neuron_names=np.asarray(neuron_names, dtype=object),
        muscle_names=np.asarray(muscle_names, dtype=object),
        joint_names=np.asarray(joint_names, dtype=object),
    )
    summary["archive_path"] = str(archive_path)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=200_000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", default="long")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/recordings"))
    parser.add_argument("--log-every", type=int, default=25_000)
    parser.add_argument(
        "--protocol",
        choices=["baseline", "sensorimotor_battery"],
        default="baseline",
    )
    args = parser.parse_args()

    summary = record_run(args.ticks, args.seed, args.out_dir, args.label, args.log_every, args.protocol)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
