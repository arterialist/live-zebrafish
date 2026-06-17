"""Cross-video 50k-tick embodied zebrafish activity synthesis."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CROSS_DIR = ROOT / "analysis" / "out" / "cross_video_embodied_20260603"
CALIBRATION_DIR = ROOT / "analysis" / "out" / "video_replay_calibration_20260603"
BACKEND_SUMMARY = (
    ROOT
    / "analysis"
    / "out"
    / "backend_video_robustness"
    / "20260603_all_selected_60s"
    / "backend_video_robustness_summary.csv"
)
DT_S = 0.005


FRIENDLY_LABELS = {
    "baseline_video_tenggol": "Tenggol raw video",
    "calibrated_mild_video_tenggol": "Tenggol mild gate",
    "calibrated_strong_video_tenggol": "Tenggol strong gate",
    "calcium_all_current": "ZAPBench calcium",
    "cross_mild_brycon": "Brycon high-motion video",
    "cross_mild_gopro_crabbing": "GoPro crabbing video",
    "cross_mild_noaa_batfish": "NOAA batfish ROV video",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _stats(values: Any) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _corr(a: Any, b: Any) -> float:
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


def _summary_rows(study_dir: Path, cohort: str) -> list[dict[str, Any]]:
    rows = _read_csv(study_dir / "high_rate_run_summary.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        summary_path = Path(row.get("summary_path", ""))
        summary = _read_json(summary_path)
        label = str(summary.get("label") or row.get("run") or summary_path.stem)
        source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
        motifs = summary.get("motifs") if isinstance(summary.get("motifs"), dict) else {}
        branch = str(summary.get("mode") or "video")
        clip = str(source.get("file_name") or source.get("condition") or "")
        out.append(
            {
                "cohort": cohort,
                "label": label,
                "run": FRIENDLY_LABELS.get(label, label),
                "branch": branch,
                "clip": clip,
                "ticks": int(_num(row.get("ticks"))),
                "simulated_s": _num(row.get("simulated_s")),
                "sample_hz": _num(row.get("sample_hz")),
                "source_frames_unique": int(_num(row.get("source_frames_unique"))),
                "source_loop_index_max": int(_num(row.get("source_loop_index_max"))),
                "swim_bouts": int(_nested(motifs, "swim_bouts", "count", default=_num(row.get("bout_count")))),
                "turn_bouts": int(_nested(motifs, "turn_bouts", "count", default=0)),
                "high_bend_events": int(_nested(motifs, "high_bend_c_or_o_like_bouts", "count", default=0)),
                "vertical_instability_events": int(_nested(motifs, "vertical_instability_events", "count", default=0)),
                "event_frequency_hz": _num(row.get("event_frequency_hz")),
                "realized_tbf_hz_median": _num(row.get("realized_tail_frequency_tbf_band_hz_median")),
                "bout_duration_ms_p50": _num(row.get("bout_duration_ms_p50")),
                "interbout_ms_p50": _num(row.get("interbout_ms_p50")),
                "speed_mean_mm_s": _num(row.get("speed_xy_mm_s_mean")),
                "speed_p95_mm_s": _num(row.get("speed_xy_mm_s_p95")),
                "speed_max_mm_s": _num(row.get("speed_xy_mm_s_max")),
                "travel_path_mm": _num(row.get("travel_path_mm")),
                "net_displacement_mm": _num(row.get("net_displacement_mm")),
                "body_pitch_abs_p95_rad": _num(row.get("body_pitch_abs_p95_rad")),
                "z_span_p95_mm": _num(row.get("body_z_span_mm_p95")),
                "tail_yaw_abs_max_deg_p95": _num(row.get("tail_yaw_abs_max_deg_p95")),
                "action_force_mean": _num(row.get("action_force_mean")),
                "action_force_p95": _num(row.get("action_force_p95")),
                "action_confidence_mean": _num(row.get("action_confidence_mean")),
                "video_motion_energy_mean": _num(row.get("video_motion_energy_mean")),
                "video_flow_reliability_mean": _num(row.get("video_flow_reliability_mean")),
                "video_camera_shake_mean": _num(row.get("video_camera_shake_mean")),
                "video_calibration_gate_mean": _num(row.get("video_calibration_gate_mean")),
                "video_raw_action_force_mean": _num(row.get("video_calibration_raw_action_force_mean")),
                "video_raw_startle_mean": _num(row.get("video_calibration_raw_startle_mean")),
                "muscle_sum_mean": _num(row.get("muscle_sum_mean")),
                "swim_drive_mean": _num(row.get("swim_drive_mean")),
                "neuron_s_abs_mean": _num(row.get("neuron_s_abs_mean")),
                "neuron_fired_count_mean": _num(row.get("neuron_fired_count_mean")),
                "calcium_pulse_mean": _num(row.get("calcium_pulse_mean")),
                "archive": str(Path(row.get("archive", "")).resolve()),
                "summary_path": str(summary_path.resolve()),
                "source_path": str(source.get("path") or ""),
                "source_calibration": json.dumps(source.get("calibration") or {}, sort_keys=True),
            }
        )
    return out


def _load_selected_arrays(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    out: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    for row in rows:
        path = Path(str(row["archive"]))
        if not path.exists():
            continue
        arrays: dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=True) as npz:
            for key in npz.files:
                arrays[key] = np.asarray(npz[key])
        out.append((row, arrays))
    return out


def _source_body_correlations(cross_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta, arrays in _load_selected_arrays(cross_rows):
        speed = np.asarray(arrays.get("speed_mm_s", []), dtype=np.float64)
        pitch_abs = np.abs(np.asarray(arrays.get("body_pitch_rad", []), dtype=np.float64))
        z_span = np.asarray(arrays.get("z_span_mm", []), dtype=np.float64)
        tail = np.asarray(arrays.get("tail_yaw_rad", np.zeros((0, 0))), dtype=np.float64)
        tail_abs = np.max(np.abs(tail), axis=1) if tail.ndim == 2 and tail.size else np.zeros(speed.size)
        muscles = np.asarray(arrays.get("muscles", np.zeros((0, 0))), dtype=np.float64)
        muscle_sum = np.sum(muscles, axis=1) if muscles.ndim == 2 and muscles.size else np.zeros(speed.size)
        neuron_s = np.asarray(arrays.get("neuron_s", np.zeros((0, 0))), dtype=np.float64)
        paula_abs = np.mean(np.abs(neuron_s), axis=1) if neuron_s.ndim == 2 and neuron_s.size else np.zeros(speed.size)
        drivers = {
            "video_motion_energy": arrays.get("video_motion_energy", []),
            "video_flow_reliability": arrays.get("video_flow_reliability", []),
            "video_camera_shake": arrays.get("video_camera_shake", []),
            "video_compression_noise": arrays.get("video_compression_noise", []),
            "video_zapbench_distance": arrays.get("video_zapbench_distance", []),
            "video_calibration_gate": arrays.get("video_calibration_gate", []),
            "action_force": arrays.get("action_force", []),
            "action_confidence": arrays.get("action_confidence", []),
        }
        responses = {
            "speed_mm_s": speed,
            "tail_yaw_abs_max_rad": tail_abs,
            "abs_body_pitch_rad": pitch_abs,
            "z_span_mm": z_span,
            "muscle_sum": muscle_sum,
            "paula_s_abs_mean": paula_abs,
        }
        for driver_name, driver in drivers.items():
            driver_arr = np.asarray(driver, dtype=np.float64)
            for response_name, response in responses.items():
                response_arr = np.asarray(response, dtype=np.float64)
                for lag_ms in (0, 250, 500, 1000, 2000):
                    lag = int(round((lag_ms / 1000.0) / DT_S))
                    if lag > 0:
                        a = driver_arr[:-lag]
                        b = response_arr[lag:]
                    else:
                        a = driver_arr
                        b = response_arr
                    rows.append(
                        {
                            "run": meta["run"],
                            "label": meta["label"],
                            "clip": meta["clip"],
                            "driver": driver_name,
                            "response": response_name,
                            "lag_ms_response_after_driver": lag_ms,
                            "pearson_r": _corr(a, b),
                        }
                    )
    return rows


def _window_extremes(study_dir: Path, metric_names: set[str]) -> list[dict[str, Any]]:
    rows = _read_csv(study_dir / "high_rate_window_summary_10s.csv")
    out: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        metric = row.get("metric", "")
        if metric in metric_names:
            grouped.setdefault((row.get("run", ""), metric), []).append(row)
    for (run, metric), items in grouped.items():
        for field in ("mean", "p95", "max"):
            best = max(items, key=lambda r: _num(r.get(field)))
            worst = min(items, key=lambda r: _num(r.get(field)))
            out.append(
                {
                    "run": run,
                    "metric": metric,
                    "field": field,
                    "max_window_index": best.get("window_index"),
                    "max_window_start_s": best.get("start_s"),
                    "max_window_end_s": best.get("end_s"),
                    "max_window_value": _num(best.get(field)),
                    "min_window_index": worst.get("window_index"),
                    "min_window_start_s": worst.get("start_s"),
                    "min_window_end_s": worst.get("end_s"),
                    "min_window_value": _num(worst.get(field)),
                }
            )
    return out


def _plot_integrated_dashboard(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    labels = [str(r["run"]) for r in rows]
    colors = ["#b95d5d" if r["branch"] == "calcium" else "#4477aa" for r in rows]
    panels = [
        ("swim_bouts", "swim bouts"),
        ("turn_bouts", "turn bouts"),
        ("vertical_instability_events", "vertical events"),
        ("realized_tbf_hz_median", "TBF median Hz"),
        ("speed_mean_mm_s", "speed mean mm/s"),
        ("speed_p95_mm_s", "speed p95 mm/s"),
        ("body_pitch_abs_p95_rad", "|pitch| p95 rad"),
        ("z_span_p95_mm", "z-span p95 mm"),
        ("action_force_mean", "action force mean"),
        ("muscle_sum_mean", "muscle sum mean"),
        ("neuron_s_abs_mean", "PAULA |S| mean"),
        ("neuron_fired_count_mean", "PAULA fired count mean"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    for ax, (key, title) in zip(axes.reshape(-1), panels):
        vals = [_num(r.get(key)) for r in rows]
        ax.bar(labels, vals, color=colors, alpha=0.86)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.24)
        ax.tick_params(axis="x", labelrotation=38, labelsize=8)
    fig.suptitle("Integrated 50k-tick zebrafish replay metrics: calibration, calcium, and cross-video runs", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / "integrated_50k_activity_dashboard.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _plot_source_quality(out_dir: Path, cross_rows: list[dict[str, Any]]) -> Path:
    backend = _read_csv(BACKEND_SUMMARY)
    selected = {str(r["clip"]).removesuffix(".mp4"): r for r in cross_rows}
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax = axes[0, 0]
    for row in backend:
        clip = row["clip"]
        is_selected = clip in selected
        color = "#d65f5f" if is_selected else "#8a8a8a"
        ax.scatter(
            _num(row.get("camera_shake_mean")),
            _num(row.get("action_force_mean")),
            s=40 + 160 * _num(row.get("compression_noise_mean")),
            color=color,
            alpha=0.82,
            edgecolor="white" if is_selected else "none",
            linewidth=0.8,
        )
        if is_selected:
            ax.annotate(clip.replace("commons_", "").replace("noaa_", ""), (_num(row.get("camera_shake_mean")), _num(row.get("action_force_mean"))), fontsize=8)
    ax.set_xlabel("backend camera shake mean")
    ax.set_ylabel("backend raw action force mean")
    ax.set_title("Backend-only source stress map; marker size = compression noise")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for row in cross_rows:
        clip_key = str(row["clip"]).removesuffix(".mp4")
        b = next((item for item in backend if item["clip"] == clip_key), {})
        ax.scatter(_num(b.get("action_force_mean")), _num(row.get("speed_mean_mm_s")), s=110, color="#4477aa")
        ax.annotate(str(row["run"]), (_num(b.get("action_force_mean")), _num(row.get("speed_mean_mm_s"))), fontsize=8)
    ax.set_xlabel("backend raw action force mean")
    ax.set_ylabel("embodied speed mean mm/s")
    ax.set_title("Selected videos: decoded drive vs realized speed")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for row in cross_rows:
        clip_key = str(row["clip"]).removesuffix(".mp4")
        b = next((item for item in backend if item["clip"] == clip_key), {})
        ax.scatter(_num(b.get("camera_shake_mean")), _num(row.get("body_pitch_abs_p95_rad")), s=100, color="#c7772f", label="pitch p95" if row is cross_rows[0] else None)
        ax.scatter(_num(b.get("camera_shake_mean")), _num(row.get("z_span_p95_mm")), s=100, color="#44aa88", marker="s", label="z-span p95" if row is cross_rows[0] else None)
        ax.annotate(str(row["run"]).split(" ")[0], (_num(b.get("camera_shake_mean")), _num(row.get("z_span_p95_mm"))), fontsize=8)
    ax.set_xlabel("backend camera shake mean")
    ax.set_ylabel("posture/depth metric")
    ax.set_title("Selected videos: source shake vs body posture/depth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    backend_selected = [row for row in backend if row["clip"] in selected]
    xs = [_num(row.get("flow_reliability_mean")) for row in backend_selected]
    ys = [_num(selected[row["clip"]].get("swim_bouts")) for row in backend_selected]
    ax.scatter(xs, ys, s=120, color="#8866aa")
    for row in backend_selected:
        ax.annotate(row["clip"].replace("commons_", "").replace("noaa_", ""), (_num(row.get("flow_reliability_mean")), _num(selected[row["clip"]].get("swim_bouts"))), fontsize=8)
    ax.set_xlabel("backend flow reliability mean")
    ax.set_ylabel("embodied swim bouts")
    ax.set_title("Selected videos: reliability vs bout production")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = out_dir / "cross_video_source_quality_vs_body.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _plot_windowed_timescales(out_dir: Path, cross_rows: list[dict[str, Any]]) -> Path:
    windows = _read_csv(CROSS_DIR / "high_rate_window_summary_10s.csv")
    run_names = [str(r["summary_path"]) for r in cross_rows]
    summary_to_run: dict[str, str] = {}
    audit_to_friendly: dict[str, str] = {}
    for row in cross_rows:
        summary_to_run[str(row["summary_path"])] = str(row["run"])
    for audit_row in _read_csv(CROSS_DIR / "high_rate_run_summary.csv"):
        summary = _read_json(Path(audit_row["summary_path"]))
        audit_to_friendly[audit_row["run"]] = FRIENDLY_LABELS.get(str(summary.get("label")), str(summary.get("label")))
    metrics = [
        ("speed_mm_s", "mean", "speed mean mm/s"),
        ("swim_drive", "mean", "swim-drive mean"),
        ("action_force", "mean", "action-force mean"),
        ("z_span_mm", "p95", "z-span p95 mm"),
        ("body_pitch_rad", "p95", "body pitch p95 rad"),
        ("video_camera_shake", "mean", "camera-shake mean"),
        ("video_flow_reliability", "mean", "flow reliability mean"),
        ("video_calibration_gate", "mean", "calibration gate mean"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 18), sharex=True)
    for ax, (metric, field, title) in zip(axes, metrics):
        for audit_run, friendly in audit_to_friendly.items():
            subset = [
                row
                for row in windows
                if row.get("run") == audit_run and row.get("metric") == metric
            ]
            subset.sort(key=lambda row: _num(row.get("start_s")))
            if not subset:
                continue
            ax.plot(
                [_num(row.get("start_s")) for row in subset],
                [_num(row.get(field)) for row in subset],
                marker="o",
                markersize=2,
                linewidth=1.2,
                label=friendly,
            )
        ax.set_ylabel(title)
        ax.grid(alpha=0.22)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("simulation time, 10 s windows")
    fig.suptitle("Cross-video 50k-tick timescale structure", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = out_dir / "cross_video_windowed_timescale_traces.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _plot_coupling(out_dir: Path, corr_rows: list[dict[str, Any]]) -> Path:
    drivers = [
        "video_motion_energy",
        "video_camera_shake",
        "video_flow_reliability",
        "video_zapbench_distance",
        "video_calibration_gate",
        "action_force",
    ]
    responses = ["speed_mm_s", "tail_yaw_abs_max_rad", "z_span_mm", "abs_body_pitch_rad", "muscle_sum", "paula_s_abs_mean"]
    runs = list(dict.fromkeys(row["run"] for row in corr_rows))
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 6), squeeze=False)
    for ax, run in zip(axes.reshape(-1), runs):
        matrix = np.zeros((len(drivers), len(responses)), dtype=np.float64)
        for i, driver in enumerate(drivers):
            for j, response in enumerate(responses):
                values = [
                    _num(row.get("pearson_r"))
                    for row in corr_rows
                    if row["run"] == run
                    and row["driver"] == driver
                    and row["response"] == response
                    and str(row["lag_ms_response_after_driver"]) == "500"
                ]
                matrix[i, j] = values[0] if values else 0.0
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax.set_title(f"{run}\n500 ms response lag")
        ax.set_xticks(np.arange(len(responses)), responses, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(drivers)), drivers, fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Pearson r")
    fig.suptitle("Video/source/action coupling into realized body, muscle, and PAULA state", fontsize=15)
    out = out_dir / "cross_video_source_body_coupling.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_negative_control(out_dir: Path, cross_rows: list[dict[str, Any]]) -> Path:
    noaa = next((row for row in cross_rows if "noaa" in str(row["label"])), None)
    if noaa is None:
        return out_dir / "cross_video_negative_control_detail.png"
    with np.load(str(noaa["archive"]), allow_pickle=True) as npz:
        t = np.arange(np.asarray(npz["ticks"]).shape[0], dtype=np.float64) * DT_S
        action_force = np.asarray(npz["action_force"], dtype=np.float64)
        speed = np.asarray(npz["speed_mm_s"], dtype=np.float64)
        z_span = np.asarray(npz["z_span_mm"], dtype=np.float64)
        pitch = np.abs(np.asarray(npz["body_pitch_rad"], dtype=np.float64))
        tail = np.asarray(npz["tail_yaw_rad"], dtype=np.float64)
        tail_abs = np.max(np.abs(tail), axis=1)
        gate = np.asarray(npz["video_calibration_gate"], dtype=np.float64)
        shake = np.asarray(npz["video_camera_shake"], dtype=np.float64)
    fig, axes = plt.subplots(6, 1, figsize=(16, 13), sharex=True)
    series = [
        (action_force, "action force", "#4477aa"),
        (speed, "speed mm/s", "#44aa88"),
        (tail_abs, "max |tail yaw| rad", "#aa7733"),
        (pitch, "|body pitch| rad", "#cc6677"),
        (z_span, "z-span mm", "#778833"),
        (gate, "calibration gate", "#8866aa"),
    ]
    for ax, (values, label, color) in zip(axes, series):
        ax.plot(t, values, color=color, linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[3].axhline(0.65, color="#7a1f1f", linestyle="--", linewidth=1, label="vertical threshold")
    axes[5].plot(t, shake, color="#555", linewidth=0.7, alpha=0.5, label="camera shake")
    axes[5].legend(fontsize=8)
    axes[-1].set_xlabel("simulation time (s)")
    fig.suptitle("NOAA batfish ROV negative-control detail: stable low-action replay with posture/depth drift", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / "cross_video_negative_control_detail.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _report(
    out_dir: Path,
    combined_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    plots: list[Path],
) -> Path:
    lines: list[str] = []
    lines.append("# Cross-Video 50k-Tick Embodied Zebrafish Activity Synthesis")
    lines.append("")
    lines.append("This report combines the completed 50,000-physics-tick embodied recordings with the prior Tenggol calibration sweep and ZAPBench calcium replay. All embodied rows are 200 Hz MuJoCo/PAULA recordings over 250 s.")
    lines.append("")
    lines.append("## What Was Tested")
    lines.append("")
    lines.append("- Calibration comparator: raw Tenggol video, mild-gated Tenggol video, strong-gated Tenggol video, and ZAPBench calcium replay.")
    lines.append("- Cross-video generalization: Brycon high-motion underwater footage, GoPro crabbing footage, and NOAA batfish ROV footage, all using the same mild calibrated backend video action settings.")
    lines.append("- Large-timescale coverage: each embodied run records 50k physics ticks, 250 s simulated time, per-tick qpos/qvel, COM/body points, tail yaw/pitch, muscle activations, PAULA state variables, neuromodulation, action commands, video feature channels, source frame indices, and motif summaries.")
    lines.append("")
    lines.append("## Headline Findings")
    lines.append("")
    brycon = next((r for r in cross_rows if "brycon" in str(r["label"])), None)
    gopro = next((r for r in cross_rows if "gopro" in str(r["label"])), None)
    noaa = next((r for r in cross_rows if "noaa" in str(r["label"])), None)
    calcium = next((r for r in combined_rows if r["label"] == "calcium_all_current"), None)
    mild = next((r for r in combined_rows if r["label"] == "calibrated_mild_video_tenggol"), None)
    baseline = next((r for r in combined_rows if r["label"] == "baseline_video_tenggol"), None)
    if baseline and mild:
        lines.append(
            f"- The calibration sweep still identifies raw Tenggol as the explicit failure mode: {baseline['vertical_instability_events']} vertical events and z-span p95 {baseline['z_span_p95_mm']:.3f} mm. The mild gate reduces that to {mild['vertical_instability_events']} vertical events with {mild['swim_bouts']} swim bouts."
        )
    if brycon:
        lines.append(
            f"- The hard high-motion Brycon clip produced strong but bounded motion: {brycon['swim_bouts']} swim bouts, {brycon['turn_bouts']} turns, mean/p95 speed {brycon['speed_mean_mm_s']:.3f}/{brycon['speed_p95_mm_s']:.3f} mm/s, 0 vertical events, z-span p95 {brycon['z_span_p95_mm']:.3f} mm."
        )
    if gopro:
        lines.append(
            f"- The lower-drive GoPro clip stayed low-energy: {gopro['swim_bouts']} swim bouts, mean speed {gopro['speed_mean_mm_s']:.3f} mm/s, 0 vertical events. That supports stimulus-dependent amplitude rather than always-on swimming."
        )
    if noaa:
        lines.append(
            f"- The NOAA ROV negative-control clip produced no segmented swim bouts and very low speed ({noaa['speed_mean_mm_s']:.3f} mm/s mean), but it still had posture/depth excursions: |pitch| p95 {noaa['body_pitch_abs_p95_rad']:.3f} rad and z-span p95 {noaa['z_span_p95_mm']:.3f} mm. This is not a catastrophic spin, but it is a residual passive-depth/posture issue."
        )
    if calcium:
        lines.append(
            f"- The ZAPBench calcium branch remains physically stable over 50k ticks: {calcium['swim_bouts']} swim bouts, 0 vertical events, mean/p95 speed {calcium['speed_mean_mm_s']:.3f}/{calcium['speed_p95_mm_s']:.3f} mm/s. It is separate from the selected-video branch and should not be described as frame-exact video-to-calcium replay."
        )
    lines.append("")
    lines.append("## Integrated Run Metrics")
    headers = [
        "cohort",
        "run",
        "branch",
        "clip",
        "swim_bouts",
        "turn_bouts",
        "vertical_instability_events",
        "event_frequency_hz",
        "realized_tbf_hz_median",
        "speed_mean_mm_s",
        "speed_p95_mm_s",
        "body_pitch_abs_p95_rad",
        "z_span_p95_mm",
        "action_force_mean",
        "muscle_sum_mean",
        "neuron_s_abs_mean",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in combined_rows:
        vals: list[str] = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Coupling Summary")
    lines.append("")
    interesting = [
        row
        for row in corr_rows
        if row["driver"] in {"action_force", "video_camera_shake", "video_motion_energy", "video_calibration_gate"}
        and row["response"] in {"speed_mm_s", "z_span_mm", "abs_body_pitch_rad", "muscle_sum"}
        and row["lag_ms_response_after_driver"] == 500
    ]
    interesting.sort(key=lambda row: abs(_num(row["pearson_r"])), reverse=True)
    lines.append("Top absolute 500 ms driver->response correlations in the cross-video recordings:")
    for row in interesting[:12]:
        lines.append(
            f"- {row['run']}: {row['driver']} -> {row['response']} r={_num(row['pearson_r']):.3f}"
        )
    lines.append("")
    lines.append("## Visual Evidence")
    for plot in plots:
        lines.append("")
        lines.append(f"![{plot.stem}]({plot.resolve()})")
    lines.append("")
    lines.append("## Artifact Index")
    for name in [
        "integrated_50k_activity_metrics.csv",
        "cross_video_source_body_correlations.csv",
        "cross_video_10s_window_extremes.csv",
        "high_rate_run_summary.csv",
        "high_rate_scalar_stats.csv",
        "high_rate_matrix_channel_stats.csv",
        "high_rate_bout_events.csv",
        "high_rate_window_summary_10s.csv",
        "high_rate_window_summary_30s.csv",
        "HIGH_RATE_TAIL_VALIDATION_AUDIT.md",
    ]:
        path = out_dir / name
        if path.exists():
            lines.append(f"- `{path.resolve()}`")
    lines.append(f"- Calibration comparator report: `{(CALIBRATION_DIR / 'FOCUSED_50K_ACTIVITY_ANALYSIS.md').resolve()}`")
    lines.append(f"- DANDI OMR neural validation: `{(ROOT / 'analysis' / 'out' / 'dandi_omr_neural_validation_20260603' / 'DANDI_OMR_NEURAL_VALIDATION.md').resolve()}`")
    lines.append(f"- DANDI selected-video alignment: `{(ROOT / 'analysis' / 'out' / 'dandi_video_omr_alignment_20260603' / 'DANDI_VIDEO_OMR_ALIGNMENT.md').resolve()}`")
    lines.append(f"- DANDI projector semantic bridge audit: `{(ROOT / 'analysis' / 'out' / 'dandi_projector_semantic_bridge_audit_20260603' / 'DANDI_PROJECTOR_SEMANTIC_BRIDGE_AUDIT.md').resolve()}`")
    lines.append("")
    lines.append("## Scientific Boundary")
    lines.append("")
    lines.append("Supported by these artifacts: the current mild-calibrated backend video branch can drive stable embodied motion across a high-motion underwater clip, a low-drive natural clip, and an ROV-like low-action clip without reproducing the old vertical-spin failure. The calcium branch can drive stable ZAPBench-derived action replay over the same 50k-tick horizon.")
    lines.append("")
    lines.append("Not yet supported: a precise frame-exact natural-video -> whole-brain calcium -> ephys/muscle -> free-swimming kinematics reproduction. The public ZAPBench material available in the current project provides calcium/action replay and ephys-style motor proxy structure, while the Z-Robot/DANDI resources provide OMR stimulus-class calcium motifs. The selected underwater videos are bridged through engineered backend features and class-level plausibility checks, not synchronized experimental projector movies.")
    lines.append("")
    lines.append("Main remaining issue: the NOAA negative-control case does not spin or swim, but it exposes passive depth/posture drift under very weak action drive. That should be treated as the next physics/controller calibration target before making stronger claims about low-stimulus resting posture.")
    out = out_dir / "CROSS_VIDEO_EMBODIED_GENERALIZATION.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    out_dir = CROSS_DIR
    cross_rows = _summary_rows(CROSS_DIR, "cross_video_mild_gate")
    calibration_rows = _summary_rows(CALIBRATION_DIR, "tenggol_calibration_and_calcium")
    wanted_calibration = {
        "baseline_video_tenggol",
        "calcium_all_current",
        "calibrated_mild_video_tenggol",
        "calibrated_strong_video_tenggol",
    }
    calibration_rows = [row for row in calibration_rows if row["label"] in wanted_calibration]
    combined_rows = calibration_rows + cross_rows
    _write_csv(out_dir / "integrated_50k_activity_metrics.csv", combined_rows)
    corr_rows = _source_body_correlations(cross_rows)
    _write_csv(out_dir / "cross_video_source_body_correlations.csv", corr_rows)
    extremes = _window_extremes(
        CROSS_DIR,
        {
            "speed_mm_s",
            "swim_drive",
            "action_force",
            "body_pitch_rad",
            "z_span_mm",
            "video_camera_shake",
            "video_flow_reliability",
            "video_calibration_gate",
            "muscle_sum",
        },
    )
    _write_csv(out_dir / "cross_video_10s_window_extremes.csv", extremes)
    plots = [
        _plot_integrated_dashboard(out_dir, combined_rows),
        _plot_source_quality(out_dir, cross_rows),
        _plot_windowed_timescales(out_dir, cross_rows),
        _plot_coupling(out_dir, corr_rows),
        _plot_negative_control(out_dir, cross_rows),
    ]
    report = _report(out_dir, combined_rows, cross_rows, corr_rows, plots)
    manifest = {
        "report": str(report.resolve()),
        "metrics": str((out_dir / "integrated_50k_activity_metrics.csv").resolve()),
        "source_body_correlations": str((out_dir / "cross_video_source_body_correlations.csv").resolve()),
        "window_extremes": str((out_dir / "cross_video_10s_window_extremes.csv").resolve()),
        "plots": [str(path.resolve()) for path in plots],
        "runs": [row["run"] for row in combined_rows],
    }
    (out_dir / "cross_video_embodied_synthesis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
