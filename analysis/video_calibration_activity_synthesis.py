"""Focused 50k-tick synthesis for zebrafish video/calcium replay runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_DIR = ROOT / "analysis" / "out" / "video_replay_calibration_20260603"
DT_S = 0.005


FRIENDLY = {
    "baseline_video_tenggol": "video baseline",
    "calcium_all_current": "ZAPBench calcium",
    "calibrated_mild_video_tenggol": "video mild gate",
    "calibrated_strong_video_tenggol": "video strong gate",
}
ORDER = [
    "baseline_video_tenggol",
    "calcium_all_current",
    "calibrated_mild_video_tenggol",
    "calibrated_strong_video_tenggol",
]


@dataclass
class RunData:
    label: str
    friendly: str
    archive: Path
    summary_path: Path
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]
    audit_row: dict[str, str]

    @property
    def seconds(self) -> float:
        return float(self.summary.get("simulated_seconds", 0.0))

    @property
    def time_s(self) -> np.ndarray:
        n = int(np.asarray(self.arrays["ticks"]).shape[0])
        return np.arange(n, dtype=np.float64) * float(self.summary.get("dt_s", DT_S))


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _nested(data: dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _stats(values: Any) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(aa.size, bb.size)
    if n < 16:
        return 0.0
    aa = aa[:n]
    bb = bb[:n]
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 16:
        return 0.0
    aa = aa[mask] - float(np.mean(aa[mask]))
    bb = bb[mask] - float(np.mean(bb[mask]))
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    return float(np.sum(aa * bb) / denom) if denom > 1e-12 else 0.0


def _events(mask: np.ndarray, dt_s: float) -> dict[str, float]:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    starts: list[int] = []
    ends: list[int] = []
    start: int | None = None
    for i, value in enumerate(arr):
        if value and start is None:
            start = i
        elif not value and start is not None:
            starts.append(start)
            ends.append(i)
            start = None
    if start is not None:
        starts.append(start)
        ends.append(arr.size)
    durations = np.asarray([(e - s) * dt_s * 1000.0 for s, e in zip(starts, ends)], dtype=np.float64)
    return {
        "count": float(len(starts)),
        "rate_hz": float(len(starts) / max(1e-12, arr.size * dt_s)),
        "duration_ms_p50": float(np.quantile(durations, 0.50)) if durations.size else 0.0,
        "duration_ms_p95": float(np.quantile(durations, 0.95)) if durations.size else 0.0,
    }


def _load_runs(study_dir: Path) -> list[RunData]:
    run_rows = {
        Path(row["archive"]).name: row
        for row in _read_csv(study_dir / "high_rate_run_summary.csv")
    }
    out: list[RunData] = []
    for path in sorted((study_dir / "recordings").glob("*_50000ticks_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        label = str(summary["label"])
        archive = (path.parent / Path(str(summary["archive_path"])).name).resolve()
        arrays: dict[str, np.ndarray] = {}
        with np.load(archive, allow_pickle=True) as npz:
            for key in npz.files:
                arrays[key] = np.asarray(npz[key])
        out.append(
            RunData(
                label=label,
                friendly=FRIENDLY.get(label, label),
                archive=archive,
                summary_path=path.resolve(),
                summary=summary,
                arrays=arrays,
                audit_row=run_rows.get(archive.name, {}),
            )
        )
    by_label = {run.label: run for run in out}
    ordered = [by_label[label] for label in ORDER if label in by_label]
    ordered.extend(run for run in out if run.label not in ORDER)
    return ordered


def _run_metrics(run: RunData) -> dict[str, Any]:
    arrays = run.arrays
    summary = run.summary
    body_pitch = np.asarray(arrays.get("body_pitch_rad", []), dtype=np.float64)
    z_span = np.asarray(arrays.get("z_span_mm", []), dtype=np.float64)
    speed = np.asarray(arrays.get("speed_mm_s", []), dtype=np.float64)
    action_force = np.asarray(arrays.get("action_force", []), dtype=np.float64)
    action_conf = np.asarray(arrays.get("action_confidence", []), dtype=np.float64)
    tail_yaw = np.asarray(arrays.get("tail_yaw_rad", np.zeros((0, 0))), dtype=np.float64)
    muscle = np.asarray(arrays.get("muscles", np.zeros((0, 0))), dtype=np.float64)
    neuron_s = np.asarray(arrays.get("neuron_s", np.zeros((0, 0))), dtype=np.float64)
    fired = np.asarray(arrays.get("neuron_fired", np.zeros((0, 0))), dtype=np.float64)
    gate = np.asarray(arrays.get("video_calibration_gate", np.zeros(speed.shape[0])), dtype=np.float64)
    raw_force = np.asarray(arrays.get("video_calibration_raw_action_force", np.zeros(speed.shape[0])), dtype=np.float64)
    source_frames = np.asarray(arrays.get("source_frame_index", []), dtype=np.int64)
    finite_sources = source_frames[source_frames >= 0]
    tail_yaw_abs_max = np.max(np.abs(tail_yaw), axis=1) if tail_yaw.ndim == 2 and tail_yaw.size else np.zeros(speed.size)
    muscle_sum = np.sum(muscle, axis=1) if muscle.ndim == 2 and muscle.size else np.zeros(speed.size)
    neuron_abs = np.mean(np.abs(neuron_s), axis=1) if neuron_s.ndim == 2 and neuron_s.size else np.zeros(speed.size)
    fired_count = np.sum(fired, axis=1) if fired.ndim == 2 and fired.size else np.zeros(speed.size)
    vertical_events = _nested(summary, "motifs", "vertical_instability_events", "count", default=0)
    high_bend_events = _nested(summary, "motifs", "high_bend_c_or_o_like_bouts", "count", default=0)
    swim_events = _nested(summary, "motifs", "swim_bouts", "count", default=0)
    turn_events = _nested(summary, "motifs", "turn_bouts", "count", default=0)
    return {
        "label": run.label,
        "run": run.friendly,
        "mode": summary.get("mode", ""),
        "ticks": int(summary.get("ticks", 0)),
        "simulated_s": float(summary.get("simulated_seconds", 0.0)),
        "source_type": _nested(summary, "source", "type", default=""),
        "source_file": _nested(summary, "source", "file_name", default=""),
        "source_frames_unique": int(np.unique(finite_sources).size) if finite_sources.size else 0,
        "source_loop_index_max": int(np.max(arrays.get("source_loop_index", np.zeros(1)))) if arrays else 0,
        "video_calibration": json.dumps(_nested(summary, "source", "calibration", default={}), sort_keys=True),
        "swim_bouts": int(swim_events),
        "turn_bouts": int(turn_events),
        "high_bend_events": int(high_bend_events),
        "vertical_instability_events": int(vertical_events),
        "event_frequency_hz": _safe_float(run.audit_row.get("event_frequency_hz")),
        "realized_tbf_hz_median": _safe_float(run.audit_row.get("realized_tail_frequency_tbf_band_hz_median")),
        "speed_mean_mm_s": _stats(speed)["mean"],
        "speed_p95_mm_s": _stats(speed)["p95"],
        "speed_max_mm_s": _stats(speed)["max"],
        "travel_path_mm": _safe_float(run.audit_row.get("travel_path_mm")),
        "body_pitch_abs_p95_rad": _stats(np.abs(body_pitch))["p95"],
        "body_pitch_abs_max_rad": _stats(np.abs(body_pitch))["max"],
        "z_span_p95_mm": _stats(z_span)["p95"],
        "z_span_max_mm": _stats(z_span)["max"],
        "tail_yaw_abs_p95_rad": _stats(tail_yaw_abs_max)["p95"],
        "action_force_mean": _stats(action_force)["mean"],
        "action_force_p95": _stats(action_force)["p95"],
        "action_confidence_mean": _stats(action_conf)["mean"],
        "video_calibration_gate_mean": _stats(gate)["mean"],
        "video_raw_action_force_mean": _stats(raw_force)["mean"],
        "muscle_sum_mean": _stats(muscle_sum)["mean"],
        "muscle_sum_p95": _stats(muscle_sum)["p95"],
        "paula_s_abs_mean": _stats(neuron_abs)["mean"],
        "paula_fired_count_mean": _stats(fired_count)["mean"],
        "archive": str(run.archive),
        "summary_path": str(run.summary_path),
    }


def _lag_rows(run: RunData) -> list[dict[str, Any]]:
    arrays = run.arrays
    force = np.asarray(arrays.get("action_force", []), dtype=np.float64)
    confidence = np.asarray(arrays.get("action_confidence", []), dtype=np.float64)
    speed = np.asarray(arrays.get("speed_mm_s", []), dtype=np.float64)
    pitch_abs = np.abs(np.asarray(arrays.get("body_pitch_rad", []), dtype=np.float64))
    z_span = np.asarray(arrays.get("z_span_mm", []), dtype=np.float64)
    muscle = np.asarray(arrays.get("muscles", np.zeros((0, 0))), dtype=np.float64)
    muscle_sum = np.sum(muscle, axis=1) if muscle.ndim == 2 and muscle.size else np.zeros(speed.size)
    rows: list[dict[str, Any]] = []
    for driver_name, driver in {"action_force": force, "action_confidence": confidence}.items():
        for response_name, response in {
            "speed_mm_s": speed,
            "abs_body_pitch_rad": pitch_abs,
            "z_span_mm": z_span,
            "muscle_sum": muscle_sum,
        }.items():
            for lag_ms in (-2000, -1000, -500, -250, 0, 250, 500, 1000, 2000, 5000):
                lag = int(round((lag_ms / 1000.0) / DT_S))
                if lag > 0:
                    a = driver[:-lag]
                    b = response[lag:]
                elif lag < 0:
                    a = driver[-lag:]
                    b = response[:lag]
                else:
                    a = driver
                    b = response
                rows.append(
                    {
                        "run": run.friendly,
                        "label": run.label,
                        "driver": driver_name,
                        "response": response_name,
                        "lag_ms_response_after_driver": lag_ms,
                        "pearson_r": _corr(a, b),
                    }
                )
    return rows


def _plot_dashboard(study_dir: Path, metrics: list[dict[str, Any]]) -> Path:
    labels = [m["run"] for m in metrics]
    colors = ["#8aa6ff", "#78c2a4", "#e0b85d", "#de7b7b"]
    panels = [
        ("vertical_instability_events", "vertical events", None),
        ("swim_bouts", "swim bouts", None),
        ("high_bend_events", "high-bend events", None),
        ("event_frequency_hz", "bout/event Hz", None),
        ("speed_mean_mm_s", "speed mean mm/s", "speed_p95_mm_s"),
        ("body_pitch_abs_p95_rad", "|pitch| p95 rad", "body_pitch_abs_max_rad"),
        ("z_span_p95_mm", "z-span p95 mm", "z_span_max_mm"),
        ("tail_yaw_abs_p95_rad", "|tail yaw|max p95 rad", None),
        ("action_force_mean", "action force mean", "action_force_p95"),
        ("action_confidence_mean", "action confidence mean", None),
        ("muscle_sum_mean", "muscle sum mean", "muscle_sum_p95"),
        ("paula_s_abs_mean", "PAULA |S| mean", "paula_fired_count_mean"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, (metric, title, overlay) in zip(axes.reshape(-1), panels):
        values = [float(m.get(metric, 0.0)) for m in metrics]
        ax.bar(labels, values, color=colors[: len(labels)], alpha=0.86)
        if overlay:
            overlay_values = [float(m.get(overlay, 0.0)) for m in metrics]
            ax.plot(labels, overlay_values, color="#202020", marker="o", linewidth=1.8, label=overlay)
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("50k-tick zebrafish replay activity: stability, motion, action, muscle, PAULA", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = study_dir / "focused_50k_activity_dashboard.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _plot_phase_space(study_dir: Path, runs: list[RunData]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, run in zip(axes.reshape(-1), runs):
        speed = np.asarray(run.arrays.get("speed_mm_s", []), dtype=np.float64)
        pitch = np.abs(np.asarray(run.arrays.get("body_pitch_rad", []), dtype=np.float64))
        force = np.asarray(run.arrays.get("action_force", []), dtype=np.float64)
        step = max(1, speed.size // 1800)
        sc = ax.scatter(speed[::step], pitch[::step], c=force[::step], s=6, cmap="viridis", alpha=0.65)
        ax.axhline(0.65, color="#b33", linestyle="--", linewidth=1.0)
        ax.set_title(run.friendly)
        ax.set_xlabel("speed mm/s")
        ax.set_ylabel("|body pitch| rad")
        ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.72, label="action force")
    fig.suptitle("Body phase space: action force vs speed and pitch", fontsize=15)
    out = study_dir / "focused_body_phase_space_by_run.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _plot_lags(study_dir: Path, lag_rows: list[dict[str, Any]]) -> Path:
    runs = list(dict.fromkeys(str(row["run"]) for row in lag_rows))
    responses = ["speed_mm_s", "abs_body_pitch_rad", "z_span_mm", "muscle_sum"]
    fig, axes = plt.subplots(len(responses), 1, figsize=(13, 11), sharex=True)
    for ax, response in zip(axes, responses):
        for run in runs:
            rows = [
                row
                for row in lag_rows
                if row["run"] == run and row["driver"] == "action_force" and row["response"] == response
            ]
            rows.sort(key=lambda row: float(row["lag_ms_response_after_driver"]))
            ax.plot(
                [float(row["lag_ms_response_after_driver"]) for row in rows],
                [float(row["pearson_r"]) for row in rows],
                marker="o",
                linewidth=1.2,
                label=run,
            )
        ax.axhline(0.0, color="#222", linewidth=0.8)
        ax.set_title(f"action force -> {response}")
        ax.set_ylabel("Pearson r")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("response lag after driver (ms)")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Lagged coupling from action command to realized body/muscle state", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = study_dir / "focused_action_lag_coupling.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _plot_event_bands(study_dir: Path, runs: list[RunData]) -> Path:
    fig, axes = plt.subplots(len(runs), 1, figsize=(16, 9), sharex=True)
    for ax, run in zip(axes, runs):
        t = run.time_s
        swim = np.asarray(run.arrays.get("swim_drive", []), dtype=np.float64) > 0.12
        pitch = np.abs(np.asarray(run.arrays.get("body_pitch_rad", []), dtype=np.float64)) > 0.65
        tail = np.asarray(run.arrays.get("tail_yaw_rad", np.zeros((0, 0))), dtype=np.float64)
        high_bend = np.max(np.abs(tail), axis=1) > 0.35 if tail.ndim == 2 and tail.size else np.zeros(t.size, bool)
        force = np.asarray(run.arrays.get("action_force", []), dtype=np.float64)
        ax.plot(t[: force.size], force, color="#3d7dd8", linewidth=0.65, alpha=0.7, label="action force")
        for y, mask, color, label in [
            (1.15, swim, "#64c17b", "swim"),
            (1.32, high_bend, "#e0a13b", "high bend"),
            (1.49, pitch, "#cf4f4f", "vertical"),
        ]:
            idx = np.flatnonzero(mask)
            if idx.size:
                ax.scatter(t[idx], np.full(idx.size, y), s=1.2, color=color, label=label)
        ax.set_ylim(-0.05, 1.62)
        ax.set_ylabel(run.friendly, rotation=0, labelpad=54, va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.18)
    axes[0].legend(ncol=4, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("simulation time (s)")
    fig.suptitle("Event bands over the full 250 s / 50k-tick recording", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = study_dir / "focused_event_bands_250s.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _report(study_dir: Path, metrics: list[dict[str, Any]], plots: list[Path]) -> Path:
    lines: list[str] = []
    lines.append("# Focused 50k-Tick Zebrafish Activity Analysis")
    lines.append("")
    lines.append("All runs below are exact 200 Hz MuJoCo/PAULA recordings with 50,000 physics ticks each.")
    lines.append("The video runs use the same 2500 sampled backend frames from `commons_tenggol_underwater.mp4`; the calcium run uses 274 ZAPBench replay frames observed during the same 250 s interval.")
    lines.append("")
    lines.append("## Headline Result")
    baseline = next((m for m in metrics if m["label"] == "baseline_video_tenggol"), None)
    mild = next((m for m in metrics if m["label"] == "calibrated_mild_video_tenggol"), None)
    strong = next((m for m in metrics if m["label"] == "calibrated_strong_video_tenggol"), None)
    calcium = next((m for m in metrics if m["label"] == "calcium_all_current"), None)
    if baseline and mild and strong:
        lines.append(
            f"- Baseline video is the failure mode: {baseline['vertical_instability_events']} vertical-instability events, "
            f"{baseline['high_bend_events']} high-bend events, |pitch| p95 {baseline['body_pitch_abs_p95_rad']:.3f} rad, "
            f"z-span p95 {baseline['z_span_p95_mm']:.3f} mm."
        )
        lines.append(
            f"- Mild video calibration is the best tradeoff in this sweep: {mild['vertical_instability_events']} vertical events, "
            f"{mild['swim_bouts']} swim bouts, speed mean/p95 {mild['speed_mean_mm_s']:.3f}/{mild['speed_p95_mm_s']:.3f} mm/s, "
            f"TBF median {mild['realized_tbf_hz_median']:.3f} Hz."
        )
        lines.append(
            f"- Strong video calibration removes vertical events ({strong['vertical_instability_events']}) but suppresses behavior "
            f"to {strong['swim_bouts']} swim bouts and {strong['speed_mean_mm_s']:.3f} mm/s mean speed."
        )
    if calcium:
        lines.append(
            f"- ZAPBench calcium replay is physically stable with {calcium['vertical_instability_events']} vertical events and "
            f"{calcium['swim_bouts']} swim bouts, but remains lower-energy than the video branch: speed mean/p95 "
            f"{calcium['speed_mean_mm_s']:.3f}/{calcium['speed_p95_mm_s']:.3f} mm/s."
        )
    lines.append("")
    lines.append("## Run Metrics")
    headers = [
        "run",
        "ticks",
        "source_frames_unique",
        "swim_bouts",
        "vertical_instability_events",
        "high_bend_events",
        "event_frequency_hz",
        "realized_tbf_hz_median",
        "speed_mean_mm_s",
        "speed_p95_mm_s",
        "body_pitch_abs_p95_rad",
        "z_span_p95_mm",
        "action_force_mean",
        "action_confidence_mean",
        "muscle_sum_mean",
        "paula_s_abs_mean",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for m in metrics:
        row = []
        for key in headers:
            value = m[key]
            if isinstance(value, float):
                row.append(f"{value:.4g}")
            else:
                row.append(str(value))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Calibration Settings")
    for m in metrics:
        lines.append(f"- {m['run']}: `{m['video_calibration']}`")
    lines.append("")
    lines.append("## Visual Evidence")
    for plot in plots:
        lines.append(f"![{plot.stem}]({plot.resolve()})")
        lines.append("")
    lines.append("## Full Artifact Inventory")
    lines.append(f"- Main high-rate audit: `{(study_dir / 'HIGH_RATE_TAIL_VALIDATION_AUDIT.md').resolve()}`")
    for name in [
        "high_rate_run_summary.csv",
        "high_rate_scalar_stats.csv",
        "high_rate_matrix_channel_stats.csv",
        "high_rate_bout_events.csv",
        "high_rate_window_summary_10s.csv",
        "high_rate_window_summary_30s.csv",
        "high_rate_tail_segment_spectra.csv",
        "high_rate_tail_window_spectra.csv",
        "focused_50k_run_metrics.csv",
        "focused_action_lag_correlations.csv",
    ]:
        lines.append(f"- `{(study_dir / name).resolve()}`")
    lines.append("")
    lines.append("## Interpretation Boundary")
    lines.append("This study validates what the current implementation actually does over 50k ticks: command conversion, PAULA state nudging, muscle activation, body kinematics, event structure, and stability. It does not prove that arbitrary underwater video is experimentally equivalent to ZAPBench projector stimuli. The mild calibration is an engineering improvement over the baseline video branch; it is not a closed scientific validation of video-to-calcium-to-ephys-to-motion.")
    out = study_dir / "FOCUSED_50K_ACTIVITY_ANALYSIS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    study_dir = DEFAULT_STUDY_DIR
    runs = _load_runs(study_dir)
    metrics = [_run_metrics(run) for run in runs]
    lag_rows: list[dict[str, Any]] = []
    for run in runs:
        lag_rows.extend(_lag_rows(run))
    _write_csv(study_dir / "focused_50k_run_metrics.csv", metrics)
    _write_csv(study_dir / "focused_action_lag_correlations.csv", lag_rows)
    plots = [
        _plot_dashboard(study_dir, metrics),
        _plot_phase_space(study_dir, runs),
        _plot_lags(study_dir, lag_rows),
        _plot_event_bands(study_dir, runs),
    ]
    report = _report(study_dir, metrics, plots)
    print(
        json.dumps(
            {
                "report": str(report.resolve()),
                "metrics": str((study_dir / "focused_50k_run_metrics.csv").resolve()),
                "lag_correlations": str((study_dir / "focused_action_lag_correlations.csv").resolve()),
                "plots": [str(path.resolve()) for path in plots],
                "runs": [run.friendly for run in runs],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
