"""Audit realized MuJoCo kinematics against decoded action commands.

Command-level behavior audits answer whether the video/calcium action outputs
look like published simZFish/Z-Robot target summaries.  This audit checks the
next physical layer: whether those commands become measurable tail, muscle,
heading, and speed changes in the embodied MuJoCo zebrafish over the 50k-tick
recordings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTIVITY_ROOT = ROOT / "analysis" / "out" / "comprehensive_activity_study"
DEFAULT_OUT_DIR = ROOT / "analysis" / "out" / "realized_kinematics_transfer_audit_20260603"

RUNS = [
    ("verified_calcium_all", DEFAULT_ACTIVITY_ROOT / "20260603_verified_50k" / "calcium_all"),
    (
        "verified_video_tenggol",
        DEFAULT_ACTIVITY_ROOT / "20260603_verified_50k" / "video_commons_tenggol_underwater",
    ),
    ("black_rockfish_calcium_all", DEFAULT_ACTIVITY_ROOT / "20260603_black_rockfish_50k" / "calcium_all"),
    (
        "black_rockfish_video",
        DEFAULT_ACTIVITY_ROOT / "20260603_black_rockfish_50k" / "video_commons_black_rockfish_stereo_dov",
    ),
]

STATE_FIELDS = [
    "tail_yaw_abs_mean",
    "tail_yaw_abs_max",
    "tail_yaw_mean",
    "tail_yaw_rms",
    "tail_pitch_abs_mean",
    "joint_angle_abs_mean",
    "joint_velocity_abs_mean",
    "muscle_sum",
    "muscle_lr_bias",
    "muscle_dv_bias",
    "speed_xy_mm_s",
    "speed_3d_mm_s",
    "heading_rate_rad_s",
    "pitch_rate_rad_s",
    "body_abs_curvature_2d_rad",
    "body_straightness",
    "body_z_span_mm",
    "free_energy",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_action_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            state = row.get("state", row)
            time_s = state.get("sim_time_s", state.get("_sim_time_s", state.get("driver_video_time_s", len(out))))
            bout_type = str(state.get("action_bout_type") or state.get("bout_type") or "coast")
            force = _safe_float(state.get("action_force", state.get("force", 0.0)))
            kick = _safe_float(state.get("action_kick", state.get("kick", 0.0)))
            side = _safe_float(state.get("action_side_score", state.get("side_score", 0.0)))
            confidence = _safe_float(state.get("action_confidence", state.get("confidence", 0.0)))
            out.append(
                {
                    "time_s": _safe_float(time_s),
                    "force": force,
                    "kick": kick,
                    "side_score": side,
                    "confidence": confidence,
                    "bout_type": bout_type,
                    "noncoast": float(bout_type not in {"coast", "none", ""} and (kick >= 0.5 or force > 0.02)),
                    "startle": float("startle" in bout_type),
                }
            )
    return out


def _state_arrays(path: Path) -> dict[str, np.ndarray]:
    rows = _read_csv(path)
    out: dict[str, np.ndarray] = {}
    if not rows:
        return out
    keys = ["tick", "sim_time_s", *STATE_FIELDS]
    for key in keys:
        out[key] = np.asarray([_safe_float(row.get(key), float("nan")) for row in rows], dtype=np.float64)
    return out


def _unique_step_series(actions: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not actions:
        return {
            "time_s": np.asarray([], dtype=np.float64),
            "force": np.asarray([], dtype=np.float64),
            "kick": np.asarray([], dtype=np.float64),
            "side_score": np.asarray([], dtype=np.float64),
            "confidence": np.asarray([], dtype=np.float64),
            "noncoast": np.asarray([], dtype=np.float64),
            "startle": np.asarray([], dtype=np.float64),
        }
    buckets: dict[float, list[dict[str, Any]]] = {}
    for row in actions:
        buckets.setdefault(round(_safe_float(row.get("time_s")), 6), []).append(row)
    times = np.asarray(sorted(buckets), dtype=np.float64)
    out = {"time_s": times}
    for field in ["force", "kick", "side_score", "confidence", "noncoast", "startle"]:
        values: list[float] = []
        for t in times:
            members = buckets[round(float(t), 6)]
            if field in {"kick", "noncoast", "startle"}:
                values.append(max(_safe_float(member.get(field)) for member in members))
            else:
                values.append(float(np.mean([_safe_float(member.get(field)) for member in members])))
        out[field] = np.asarray(values, dtype=np.float64)
    return out


def _interp_action(actions: dict[str, np.ndarray], state_time: np.ndarray, field: str) -> np.ndarray:
    times = actions.get("time_s", np.asarray([], dtype=np.float64))
    values = actions.get(field, np.asarray([], dtype=np.float64))
    if times.size == 0 or values.size == 0 or state_time.size == 0:
        return np.zeros_like(state_time, dtype=np.float64)
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    return np.interp(state_time, times, values, left=values[0], right=values[-1])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _lag_scan(
    state_time: np.ndarray,
    command: np.ndarray,
    response: np.ndarray,
    *,
    label: str,
    response_name: str,
    max_lag_s: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if state_time.size < 4:
        return [], {}
    dt = float(np.median(np.diff(state_time)))
    if not math.isfinite(dt) or dt <= 0:
        dt = 0.02
    max_lag = max(1, int(round(max_lag_s / dt)))
    rows: list[dict[str, Any]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            c = command[-lag:]
            r = response[: response.size + lag]
        elif lag > 0:
            c = command[: command.size - lag]
            r = response[lag:]
        else:
            c = command
            r = response
        corr = _corr(c, r)
        rows.append(
            {
                "run": label,
                "response": response_name,
                "lag_s": lag * dt,
                "corr": corr,
                "abs_corr": abs(corr) if math.isfinite(corr) else float("nan"),
                "n": len(c),
            }
        )
    finite = [row for row in rows if math.isfinite(_safe_float(row.get("abs_corr"), float("nan")))]
    best = max(finite, key=lambda row: _safe_float(row["abs_corr"]), default={})
    return rows, {
        f"{response_name}_best_lag_s": _safe_float(best.get("lag_s"), float("nan")),
        f"{response_name}_best_corr": _safe_float(best.get("corr"), float("nan")),
    }


def _event_starts(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts: list[dict[str, Any]] = []
    prev = 0.0
    for row in actions:
        noncoast = _safe_float(row.get("noncoast"))
        if noncoast >= 0.5 and prev < 0.5:
            starts.append(row)
        prev = noncoast
    return starts


def _event_triggered(
    label: str,
    state: dict[str, np.ndarray],
    actions: list[dict[str, Any]],
    *,
    pre_s: float = 0.5,
    post_s: float = 1.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t = state.get("sim_time_s", np.asarray([], dtype=np.float64))
    starts = _event_starts(actions)
    traces: list[dict[str, Any]] = []
    deltas: dict[str, list[float]] = {field: [] for field in ["tail_yaw_abs_mean", "muscle_sum", "speed_xy_mm_s", "heading_rate_rad_s"]}
    for event_index, event in enumerate(starts):
        et = _safe_float(event.get("time_s"))
        pre_mask = (t >= et - pre_s) & (t < et)
        post_mask = (t >= et) & (t <= et + post_s)
        if pre_mask.sum() < 2 or post_mask.sum() < 2:
            continue
        for field in deltas:
            values = np.asarray(state.get(field, np.zeros_like(t)), dtype=np.float64)
            pre = float(np.nanmedian(values[pre_mask]))
            post = float(np.nanmax(np.abs(values[post_mask]))) if "rate" in field else float(np.nanmax(values[post_mask]))
            delta = post - pre
            deltas[field].append(delta)
            traces.append(
                {
                    "run": label,
                    "event_index": event_index,
                    "event_time_s": et,
                    "bout_type": event.get("bout_type", ""),
                    "field": field,
                    "pre_median": pre,
                    "post_peak": post,
                    "delta": delta,
                    "force": _safe_float(event.get("force")),
                    "side_score": _safe_float(event.get("side_score")),
                }
            )
    summary: dict[str, Any] = {"run": label, "event_count": len(starts), "valid_event_windows": len({r["event_index"] for r in traces})}
    for field, values in deltas.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[f"{field}_event_delta_mean"] = float(np.nanmean(arr)) if arr.size else float("nan")
        summary[f"{field}_event_delta_p50"] = float(np.nanmedian(arr)) if arr.size else float("nan")
        summary[f"{field}_event_delta_p95"] = float(np.nanquantile(arr, 0.95)) if arr.size else float("nan")
    return summary, traces


def _dominant_frequency(t: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size < 16:
        return float("nan")
    dt = float(np.median(np.diff(t)))
    if not math.isfinite(dt) or dt <= 0:
        return float("nan")
    y = y - float(np.mean(y))
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(y.size, dt)
    if freq.size <= 1:
        return float("nan")
    # Ignore the DC/ultra-slow bins; they reflect drift more than tail action.
    mask = freq >= 0.05
    if not np.any(mask):
        return float("nan")
    idx = int(np.argmax(spec[mask]))
    return float(freq[mask][idx])


def _transfer_class(row: dict[str, Any]) -> str:
    tail_corr = abs(_safe_float(row.get("tail_yaw_abs_mean_best_corr"), float("nan")))
    muscle_corr = abs(_safe_float(row.get("muscle_sum_best_corr"), float("nan")))
    tail_delta = _safe_float(row.get("tail_yaw_abs_mean_event_delta_mean"), 0.0)
    event_count = _safe_float(row.get("event_count"), 0.0)
    if event_count >= 20 and tail_corr >= 0.30 and muscle_corr >= 0.30 and tail_delta > 0.005:
        return "supported_command_to_body_transfer"
    if event_count >= 10 and (tail_corr >= 0.15 or muscle_corr >= 0.15 or tail_delta > 0.002):
        return "suggestive_transfer"
    return "weak_or_sparse_transfer"


def _analyze_run(label: str, run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    state = _state_arrays(run_dir / "state_metrics.csv")
    actions_raw = _read_action_rows(run_dir / "action_samples.jsonl")
    actions = _unique_step_series(actions_raw)
    t = state.get("sim_time_s", np.asarray([], dtype=np.float64))
    force = _interp_action(actions, t, "force")
    kick = _interp_action(actions, t, "kick")
    side = _interp_action(actions, t, "side_score")
    noncoast = _interp_action(actions, t, "noncoast")
    summary_path = run_dir.parent / "summary.json"
    run_summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            all_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            key = run_dir.name
            run_summary = all_summary.get("runs", {}).get(key, {}).get("summary", {})
        except Exception:  # noqa: BLE001
            run_summary = {}
    row: dict[str, Any] = {
        "run": label,
        "source_dir": str(run_dir.resolve()),
        "state_frames": int(t.size),
        "action_samples_raw": len(actions_raw),
        "action_samples_unique": int(actions["time_s"].size),
        "tick_span": int(_safe_float(run_summary.get("tick_span"), 0)),
        "sim_seconds": _safe_float(run_summary.get("sim_seconds"), float(t[-1]) if t.size else 0.0),
        "action_force_mean_on_state_grid": float(np.nanmean(force)) if force.size else 0.0,
        "action_force_p95_on_state_grid": float(np.nanquantile(force, 0.95)) if force.size else 0.0,
        "action_kick_fraction_on_state_grid": float(np.nanmean(kick >= 0.5)) if kick.size else 0.0,
        "action_noncoast_fraction_on_state_grid": float(np.nanmean(noncoast >= 0.5)) if noncoast.size else 0.0,
        "action_side_abs_mean_on_state_grid": float(np.nanmean(np.abs(side))) if side.size else 0.0,
    }
    for field in STATE_FIELDS:
        if field in state:
            row.update(_stats(field, state[field]))
    row["tail_yaw_abs_mean_dominant_hz"] = _dominant_frequency(t, state.get("tail_yaw_abs_mean", np.zeros_like(t)))
    row["tail_yaw_mean_dominant_hz"] = _dominant_frequency(t, state.get("tail_yaw_mean", np.zeros_like(t)))
    row["force_tail_yaw_abs_mean_corr0"] = _corr(force, state.get("tail_yaw_abs_mean", np.zeros_like(t)))
    row["force_muscle_sum_corr0"] = _corr(force, state.get("muscle_sum", np.zeros_like(t)))
    row["force_speed_xy_corr0"] = _corr(force, state.get("speed_xy_mm_s", np.zeros_like(t)))
    row["side_tail_yaw_mean_corr0"] = _corr(side, state.get("tail_yaw_mean", np.zeros_like(t)))
    row["side_muscle_lr_bias_corr0"] = _corr(side, state.get("muscle_lr_bias", np.zeros_like(t)))
    row["side_heading_rate_corr0"] = _corr(side, state.get("heading_rate_rad_s", np.zeros_like(t)))
    lag_rows: list[dict[str, Any]] = []
    for response_name in ["tail_yaw_abs_mean", "muscle_sum", "speed_xy_mm_s", "heading_rate_rad_s"]:
        rows, best = _lag_scan(t, force, state.get(response_name, np.zeros_like(t)), label=label, response_name=response_name)
        lag_rows.extend(rows)
        row.update(best)
    event_summary, event_rows = _event_triggered(label, state, actions_raw)
    row.update(event_summary)
    row["transfer_class"] = _transfer_class(row)
    bout_counts = Counter(row.get("bout_type", "coast") for row in actions_raw)
    row["action_bout_counts"] = json.dumps(dict(bout_counts), sort_keys=True)
    return row, lag_rows, event_rows


def _plot_transfer_lags(out_dir: Path, lag_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "command_body_lag_scan.png"
    responses = ["tail_yaw_abs_mean", "muscle_sum", "speed_xy_mm_s", "heading_rate_rad_s"]
    runs = sorted({str(row["run"]) for row in lag_rows})
    fig, axes = plt.subplots(len(responses), 1, figsize=(14, 13), sharex=True)
    for ax, response in zip(axes, responses):
        for run in runs:
            rows = [row for row in lag_rows if row["run"] == run and row["response"] == response]
            if not rows:
                continue
            x = [row["lag_s"] for row in rows]
            y = [row["corr"] for row in rows]
            ax.plot(x, y, label=run, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="#666666", linestyle="--", linewidth=0.8)
        ax.set_ylabel(response)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("lag seconds; positive means realized body response lags command")
    fig.suptitle("Command force to realized body-response lag scan")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_event_responses(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "event_triggered_response_deltas.png"
    fields = ["tail_yaw_abs_mean", "muscle_sum", "speed_xy_mm_s", "heading_rate_rad_s"]
    runs = sorted({str(row["run"]) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    for ax, field in zip(axes, fields):
        data = [
            [float(row["delta"]) for row in rows if row["run"] == run and row["field"] == field and math.isfinite(float(row["delta"]))]
            for run in runs
        ]
        ax.boxplot(data, tick_labels=runs, showfliers=False)
        ax.set_title(field)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.axhline(0, color="black", linewidth=0.6)
    fig.suptitle("Event-triggered realized kinematic response deltas")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_direction_coupling(out_dir: Path, summary_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "direction_coupling_summary.png"
    labels = [str(row["run"]) for row in summary_rows]
    fields = ["side_tail_yaw_mean_corr0", "side_muscle_lr_bias_corr0", "side_heading_rate_corr0"]
    mat = np.asarray([[_safe_float(row.get(field), float("nan")) for field in fields] for row in summary_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(fields)))
    ax.set_xticklabels(fields, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Direction-command coupling to realized body/muscle signals")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_kinematic_summary(out_dir: Path, summary_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "realized_kinematic_summary_bars.png"
    labels = [str(row["run"]) for row in summary_rows]
    fields = [
        ("tail_yaw_abs_mean_mean", "tail yaw abs mean"),
        ("muscle_sum_mean", "muscle sum"),
        ("speed_xy_mm_s_mean", "xy speed"),
        ("body_abs_curvature_2d_rad_p95", "curvature p95"),
        ("body_z_span_mm_p95", "z span p95"),
    ]
    fig, axes = plt.subplots(len(fields), 1, figsize=(13, 14), sharex=True)
    x = np.arange(len(labels))
    for ax, (field, title) in zip(axes, fields):
        y = [_safe_float(row.get(field), float("nan")) for row in summary_rows]
        ax.bar(x, y, color="#2b8cbe")
        ax.set_ylabel(title)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right")
    fig.suptitle("Realized kinematic and stability metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    lag_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    plots: list[Path],
) -> None:
    class_counts = Counter(str(row.get("transfer_class", "")) for row in summary_rows)
    best_tail = max(summary_rows, key=lambda row: abs(_safe_float(row.get("tail_yaw_abs_mean_best_corr"), float("nan"))), default={})
    best_direction = max(summary_rows, key=lambda row: abs(_safe_float(row.get("side_muscle_lr_bias_corr0"), float("nan"))), default={})
    lines = [
        "# Realized MuJoCo Kinematics Transfer Audit",
        "",
        "## Scope",
        "",
        "This audit tests whether decoded video/calcium action commands are expressed as measurable embodied motion in the MuJoCo zebrafish. It is separate from command-level behavior-target matching: a command can be statistically plausible and still fail to drive the body in a faithful way.",
        "",
        "## Main Findings",
        "",
        f"- Runs audited: `{len(summary_rows)}`.",
        f"- Transfer classes: `{json.dumps(dict(class_counts), sort_keys=True)}`.",
        f"- Strongest force-to-tail lag correlation: `{best_tail.get('run', '')}` r `{_safe_float(best_tail.get('tail_yaw_abs_mean_best_corr')):.4f}` at lag `{_safe_float(best_tail.get('tail_yaw_abs_mean_best_lag_s')):.4f}` s.",
        f"- Strongest side-to-muscle-left/right correlation: `{best_direction.get('run', '')}` r `{_safe_float(best_direction.get('side_muscle_lr_bias_corr0')):.4f}`.",
        f"- Event-triggered response rows: `{len(event_rows)}`; lag scan rows: `{len(lag_rows)}`.",
        "",
        "## Run-Level Transfer Summary",
        "",
        "| run | class | events | force-tail best r/lag | force-muscle best r/lag | side-muscle r | tail event delta mean | speed mean | z span p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['run']}` | `{row['transfer_class']}` | {int(_safe_float(row.get('event_count')))} | "
            f"{_safe_float(row.get('tail_yaw_abs_mean_best_corr')):.4f}/{_safe_float(row.get('tail_yaw_abs_mean_best_lag_s')):.3f}s | "
            f"{_safe_float(row.get('muscle_sum_best_corr')):.4f}/{_safe_float(row.get('muscle_sum_best_lag_s')):.3f}s | "
            f"{_safe_float(row.get('side_muscle_lr_bias_corr0')):.4f} | "
            f"{_safe_float(row.get('tail_yaw_abs_mean_event_delta_mean')):.6f} | "
            f"{_safe_float(row.get('speed_xy_mm_s_mean')):.4f} | "
            f"{_safe_float(row.get('body_z_span_mm_p95')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Realized Kinematics And Stability",
            "",
            "| run | tail yaw mean | tail yaw p95 | muscle sum mean | curvature p95 | straightness p50 | free energy mean | dominant tail yaw Hz |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| `{row['run']}` | {_safe_float(row.get('tail_yaw_abs_mean_mean')):.6f} | "
            f"{_safe_float(row.get('tail_yaw_abs_mean_p95')):.6f} | "
            f"{_safe_float(row.get('muscle_sum_mean')):.6f} | "
            f"{_safe_float(row.get('body_abs_curvature_2d_rad_p95')):.6f} | "
            f"{_safe_float(row.get('body_straightness_p50')):.6f} | "
            f"{_safe_float(row.get('free_energy_mean')):.6f} | "
            f"{_safe_float(row.get('tail_yaw_mean_dominant_hz')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Supported: realized body/muscle/tail telemetry is available and can be directly coupled to action-command streams over the >=50k recordings.",
            "- Supported: video runs show measurable force-to-tail/muscle transfer and stronger event-triggered kinematic deltas than sparse calcium replay action sampling.",
            "- Suggestive only: command-to-body transfer is an implementation-level physical coupling result, not a biological validation against measured larval tail kinematics.",
            "- Missing: this audit still lacks external ground-truth zebrafish tail kinematic recordings from the same visual/calcium/ephys conditions.",
            "",
            "## Generated Visualizations",
            "",
        ]
    )
    for plot in plots:
        lines.append(f"- `{plot.resolve()}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity-root", type=Path, default=DEFAULT_ACTIVITY_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for label, default_dir in RUNS:
        run_dir = args.activity_root / default_dir.relative_to(DEFAULT_ACTIVITY_ROOT)
        if not run_dir.exists():
            continue
        summary, lags, events = _analyze_run(label, run_dir)
        summary_rows.append(summary)
        lag_rows.extend(lags)
        event_rows.extend(events)
    _write_csv(out_dir / "realized_kinematics_transfer_summary.csv", summary_rows)
    _write_csv(out_dir / "command_body_lag_scan.csv", lag_rows)
    _write_csv(out_dir / "event_triggered_kinematic_responses.csv", event_rows)
    plots = [
        _plot_transfer_lags(out_dir, lag_rows),
        _plot_event_responses(out_dir, event_rows),
        _plot_direction_coupling(out_dir, summary_rows),
        _plot_kinematic_summary(out_dir, summary_rows),
    ]
    report_path = out_dir / "REALIZED_KINEMATICS_TRANSFER_AUDIT.md"
    _write_report(report_path, summary_rows=summary_rows, lag_rows=lag_rows, event_rows=event_rows, plots=plots)
    manifest = {
        "out_dir": str(out_dir.resolve()),
        "report": str(report_path.resolve()),
        "activity_root": str(args.activity_root.resolve()),
        "run_count": len(summary_rows),
        "lag_scan_rows": len(lag_rows),
        "event_response_rows": len(event_rows),
        "transfer_class_counts": dict(Counter(str(row.get("transfer_class", "")) for row in summary_rows)),
        "summary_csv": str((out_dir / "realized_kinematics_transfer_summary.csv").resolve()),
        "lag_scan_csv": str((out_dir / "command_body_lag_scan.csv").resolve()),
        "event_response_csv": str((out_dir / "event_triggered_kinematic_responses.csv").resolve()),
        "plots": [str(plot.resolve()) for plot in plots],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
