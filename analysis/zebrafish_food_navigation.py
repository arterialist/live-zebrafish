"""Quantify zebrafish v2 food navigation and PAULA/procedural ablations."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger


FOOD_POSITIONS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("east", (0.018, 0.000, 0.0)),
    ("north", (0.000, 0.018, 0.0)),
    ("southwest", (-0.014, -0.014, 0.0)),
    ("southeast", (0.014, -0.018, 0.0)),
)

SENSORY_NEURONS = ("RETINA_L", "RETINA_R", "OLFACTORY_L", "OLFACTORY_R")
MOTOR_NEURONS = (
    "SWIM_GATE",
    "RETICULOSPINAL_L",
    "RETICULOSPINAL_R",
    "SPINAL_CPG_L",
    "SPINAL_CPG_R",
)


@dataclass(frozen=True)
class Mode:
    name: str
    description: str
    configure: Callable[[Any], None]


def _mode_normal(ns: Any) -> None:
    return None


def _mode_paula_silenced(ns: Any) -> None:
    ns.sensory_input_gain = 0.0
    ns.paula_activity_gain = 0.0
    ns.set_neuromodulator_enabled(0, False)
    ns.set_neuromodulator_enabled(1, False)


def _mode_food_gains_zero(ns: Any) -> None:
    ns.visual_approach_gain = 0.0
    ns.odor_approach_gain = 0.0
    ns.visual_turn_gain = 0.0
    ns.odor_turn_gain = 0.0


MODES: tuple[Mode, ...] = (
    Mode(
        "normal",
        "Default zebrafish v2 pathway: PAULA receives food cues and procedural food gains are active.",
        _mode_normal,
    ),
    Mode(
        "paula_silenced",
        "No sensory injection into PAULA, no behavior-to-PAULA nudges, and M0/M1 disabled; procedural food gains remain active.",
        _mode_paula_silenced,
    ),
    Mode(
        "food_gains_zero",
        "PAULA still receives food cues and neuromodulation, but explicit visual/odor drive and turn gains are zeroed.",
        _mode_food_gains_zero,
    ),
)


def _stats(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": float(statistics.fmean(finite)),
        "median": float(statistics.median(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
        "std": float(statistics.pstdev(finite)) if len(finite) > 1 else 0.0,
    }


def _safe_corr(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(a) != len(b):
        return 0.0
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _neuron_vectors(ns: Any, names: list[str]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    s_by_name: dict[str, float] = {}
    m0_by_name: dict[str, float] = {}
    m1_by_name: dict[str, float] = {}
    network = getattr(ns, "_network", None)
    if network is None:
        return s_by_name, m0_by_name, m1_by_name
    neurons = network.network.neurons
    for nid, neuron in neurons.items():
        idx = int(nid)
        if idx >= len(names):
            continue
        name = names[idx]
        s_by_name[name] = float(neuron.S)
        m_vec = np.asarray(getattr(neuron, "M_vector", []), dtype=float)
        m0_by_name[name] = float(m_vec[0]) if m_vec.size > 0 else 0.0
        m1_by_name[name] = float(m_vec[1]) if m_vec.size > 1 else 0.0
    return s_by_name, m0_by_name, m1_by_name


def run_trial(
    *,
    mode: Mode,
    food_label: str,
    food: tuple[float, float, float],
    seed: int,
    ticks: int,
    sample_every: int,
) -> dict[str, Any]:
    from simulations.zebrafish.simulation import build_zebrafish_simulation

    engine, _loop = build_zebrafish_simulation(
        food_positions=[food],
        log_level="ERROR",
        record_neural_states=False,
        max_history=8,
        suppress_connectome_summary=True,
        seed=seed,
    )
    ns = engine.nervous_system
    mode.configure(ns)
    engine.reset(nervous_rebuild=False)

    names = ns.get_neuron_names_paula_order()
    food_xy = np.asarray(food[:2], dtype=np.float64)
    distances: list[float] = []
    progress_rates: list[float] = []
    drive: list[float] = []
    turn_abs: list[float] = []
    neuromod_m0: list[float] = []
    neuromod_m1: list[float] = []
    paula_food_s: list[float] = []
    paula_motor_s: list[float] = []
    paula_food_m0: list[float] = []
    paula_food_m1: list[float] = []
    retinal_odor_inputs: list[float] = []
    alignment_samples: list[float] = []
    last_distance: float | None = None
    first_head: list[float] | None = None
    last_head: list[float] | None = None
    consumed_tick: int | None = None

    t0 = time.perf_counter()
    for i in range(ticks):
        step = engine.step()
        st = step.body_state
        head = np.asarray(st.head_position[:2], dtype=np.float64)
        last_head = [float(head[0]), float(head[1])]
        if first_head is None:
            first_head = list(last_head)
        dist = float(np.linalg.norm(food_xy - head))
        if last_distance is not None:
            progress_rates.append(last_distance - dist)
        last_distance = dist
        if consumed_tick is None and len(engine.environment.get_active_food_positions()) == 0:
            consumed_tick = int(step.tick)

        if i % sample_every != 0 and i != ticks - 1:
            continue

        obs_extra = step.observation.extra or {}
        sensory_peak = max(
            float(obs_extra.get("visual_left", 0.0)),
            float(obs_extra.get("visual_right", 0.0)),
            float(step.observation.chemicals.get("food_odor", 0.0)),
        )
        extra = st.extra or {}
        velocity = np.asarray(extra.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float64)[:2]
        target = food_xy - head
        speed = float(np.linalg.norm(velocity))
        target_norm = float(np.linalg.norm(target))
        if speed > 1e-12 and target_norm > 1e-12:
            alignment_samples.append(float(np.dot(velocity, target) / (speed * target_norm)))

        s_by_name, m0_by_name, m1_by_name = _neuron_vectors(ns, names)
        distances.append(dist)
        drive.append(float(ns.behavior_state.get("swim_drive", 0.0)))
        turn_abs.append(abs(float(ns.behavior_state.get("turn_bias", 0.0))))
        m0, m1 = ns.neuromod_levels
        neuromod_m0.append(float(m0))
        neuromod_m1.append(float(m1))
        paula_food_s.append(float(np.mean([s_by_name.get(n, 0.0) for n in SENSORY_NEURONS])))
        paula_motor_s.append(float(np.mean([s_by_name.get(n, 0.0) for n in MOTOR_NEURONS])))
        paula_food_m0.append(float(np.mean([m0_by_name.get(n, 0.0) for n in SENSORY_NEURONS])))
        paula_food_m1.append(float(np.mean([m1_by_name.get(n, 0.0) for n in SENSORY_NEURONS])))
        retinal_odor_inputs.append(sensory_peak)

    initial = distances[0]
    final = distances[-1]
    progress = initial - final
    return {
        "mode": mode.name,
        "food_label": food_label,
        "food_xyz_m": list(food),
        "seed": int(seed),
        "ticks": int(ticks),
        "sample_every": int(sample_every),
        "wall_seconds": float(time.perf_counter() - t0),
        "initial_distance_m": float(initial),
        "final_distance_m": float(final),
        "min_distance_m": float(min(distances)),
        "progress_m": float(progress),
        "progress_fraction": float(progress / initial) if initial > 0 else 0.0,
        "approached": bool(progress > 0.001),
        "consumed_tick": consumed_tick,
        "first_head_xy_m": first_head,
        "last_head_xy_m": last_head,
        "mean_step_progress_m": float(statistics.fmean(progress_rates)) if progress_rates else 0.0,
        "mean_velocity_alignment_to_food": float(statistics.fmean(alignment_samples)) if alignment_samples else 0.0,
        "behavior": {
            "swim_drive": _stats(drive),
            "abs_turn_bias": _stats(turn_abs),
            "neuromod_proxy_m0": _stats(neuromod_m0),
            "neuromod_proxy_m1": _stats(neuromod_m1),
        },
        "paula": {
            "food_input_peak": _stats(retinal_odor_inputs),
            "food_sensory_S": _stats(paula_food_s),
            "motor_gate_S": _stats(paula_motor_s),
            "food_sensory_M0": _stats(paula_food_m0),
            "food_sensory_M1": _stats(paula_food_m1),
            "food_input_vs_food_sensory_S_corr": _safe_corr(retinal_odor_inputs, paula_food_s),
            "food_input_vs_motor_gate_S_corr": _safe_corr(retinal_odor_inputs, paula_motor_s),
        },
    }


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        by_mode.setdefault(str(trial["mode"]), []).append(trial)
    mode_summary = {}
    for mode, rows in by_mode.items():
        mode_summary[mode] = {
            "n_trials": len(rows),
            "approach_rate": float(sum(1 for r in rows if r["approached"]) / max(1, len(rows))),
            "consumed_count": int(sum(1 for r in rows if r["consumed_tick"] is not None)),
            "progress_m": _stats([float(r["progress_m"]) for r in rows]),
            "progress_fraction": _stats([float(r["progress_fraction"]) for r in rows]),
            "final_distance_m": _stats([float(r["final_distance_m"]) for r in rows]),
            "min_distance_m": _stats([float(r["min_distance_m"]) for r in rows]),
            "velocity_alignment": _stats([float(r["mean_velocity_alignment_to_food"]) for r in rows]),
            "neuromod_proxy_m1_mean": _stats(
                [float(r["behavior"]["neuromod_proxy_m1"]["mean"]) for r in rows]
            ),
            "food_input_peak_mean": _stats(
                [float(r["paula"]["food_input_peak"]["mean"]) for r in rows]
            ),
            "paula_food_sensory_S_mean": _stats(
                [float(r["paula"]["food_sensory_S"]["mean"]) for r in rows]
            ),
            "paula_motor_gate_S_mean": _stats(
                [float(r["paula"]["motor_gate_S"]["mean"]) for r in rows]
            ),
        }
    return {"mode_summary": mode_summary}


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]["mode_summary"]
    normal = summary.get("normal", {})
    silenced = summary.get("paula_silenced", {})
    zero = summary.get("food_gains_zero", {})
    lines = [
        "# Zebrafish v2 Food Navigation Verification",
        "",
        f"- Date: {payload['date']}",
        f"- Ticks per trial: {payload['ticks']} at 5 ms/tick ({payload['ticks'] * 0.005:.1f} simulated s)",
        f"- Seeds: {payload['seeds']}",
        f"- Food positions: {payload['food_positions']}",
        f"- Command: `{payload['command']}`",
        "",
        "## Quantitative Summary",
        "",
        "| mode | n | approach rate | consumed | mean progress (m) | mean final distance (m) | mean velocity alignment | mean food cue | mean M1 proxy | mean PAULA food S | mean PAULA motor S |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode_name in [m.name for m in MODES]:
        row = summary[mode_name]
        lines.append(
            "| {mode} | {n} | {rate:.2f} | {consumed} | {progress:.6f} | {final:.6f} | {align:.3f} | {cue:.3f} | {m1:.3f} | {food_s:.3f} | {motor_s:.3f} |".format(
                mode=mode_name,
                n=row["n_trials"],
                rate=row["approach_rate"],
                consumed=row["consumed_count"],
                progress=row["progress_m"]["mean"],
                final=row["final_distance_m"]["mean"],
                align=row["velocity_alignment"]["mean"],
                cue=row["food_input_peak_mean"]["mean"],
                m1=row["neuromod_proxy_m1_mean"]["mean"],
                food_s=row["paula_food_sensory_S_mean"]["mean"],
                motor_s=row["paula_motor_gate_S_mean"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- Default trials showed approach if mean progress is positive. "
                f"Here normal mean progress was {normal.get('progress_m', {}).get('mean', 0.0):.6f} m."
            ),
            (
                "- Silencing PAULA sensory injection, behavior-to-PAULA nudges, and M0/M1 did not remove the procedural food path if progress remains comparable. "
                f"Here paula_silenced mean progress was {silenced.get('progress_m', {}).get('mean', 0.0):.6f} m."
            ),
            (
                "- Zeroing the explicit visual/odor approach and turn gains leaves PAULA food cues active but removes the direct procedural food controller. "
                f"Here food_gains_zero mean progress was {zero.get('progress_m', {}).get('mean', 0.0):.6f} m."
            ),
            (
                "- The sampled raw food cue was nonzero in every mode, but the measured PAULA food sensory S stayed at "
                f"{normal.get('paula_food_sensory_S_mean', {}).get('mean', 0.0):.3f} in normal trials; the PAULA motor-gate S is mostly behavior-state nudging."
            ),
            "",
            "Conclusion: current zebrafish v2 food approach is not PAULA/neuromodulation-driven in the C. elegans sense if the normal and PAULA-silenced trials behave similarly while food-gain-zero trials lose approach.",
            "",
            "## Code-Path Evidence",
            "",
            "- `ZebrafishNervousSystem.tick()` injects sensory inputs into PAULA, runs the network, then calls `_update_behavior_state(sensory_inputs, ...)` and `_decode_tail_motors(...)`.",
            "- `_update_behavior_state()` computes appetitive drive and turn bias directly from raw `RETINA_*` and `OLFACTORY_*` sensory values.",
            "- `_decode_tail_motors()` converts those procedural `swim_drive` and `turn_bias` values into tail motor outputs; PAULA states are nudged for visibility after behavior is already computed.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['json_path']}`",
            f"- Report: `{path}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=3_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 41])
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/out"))
    parser.add_argument("--date", default="2026-05-13")
    args = parser.parse_args()

    logger.remove()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []
    start = time.perf_counter()
    for mode in MODES:
        for food_label, food in FOOD_POSITIONS:
            for seed in args.seeds:
                trial = run_trial(
                    mode=mode,
                    food_label=food_label,
                    food=food,
                    seed=seed,
                    ticks=args.ticks,
                    sample_every=args.sample_every,
                )
                trials.append(trial)
                print(
                    json.dumps(
                        {
                            "mode": mode.name,
                            "food": food_label,
                            "seed": seed,
                            "progress_m": round(trial["progress_m"], 6),
                            "final_distance_m": round(trial["final_distance_m"], 6),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    payload: dict[str, Any] = {
        "date": args.date,
        "ticks": int(args.ticks),
        "seeds": [int(s) for s in args.seeds],
        "sample_every": int(args.sample_every),
        "command": " ".join(["uv", "run", "python", *sys.argv]),
        "food_positions": {label: list(pos) for label, pos in FOOD_POSITIONS},
        "modes": {mode.name: mode.description for mode in MODES},
        "wall_seconds_total": float(time.perf_counter() - start),
        "trials": trials,
        "summary": summarize(trials),
    }
    json_path = args.out_dir / f"zebrafish_food_navigation_{args.date}.json"
    report_path = args.out_dir / f"zebrafish_food_navigation_report_{args.date}.md"
    payload["json_path"] = str(json_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_path, payload)
    print(json.dumps({"json_path": str(json_path), "report_path": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
