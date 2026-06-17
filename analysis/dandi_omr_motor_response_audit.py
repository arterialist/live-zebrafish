"""Run DANDI OMR stimulus classes through the current simZFish motor adapter.

The DANDI/Z-Robot NWB files expose class-level OMR calcium responses.  The
selected-video audit shows arbitrary underwater videos mostly collapse to weak
mixed OMR classes.  This script asks a stricter implementation question:

If we give the current simZFish-inspired action adapter calibrated, projector-
like two-eye OMR stimuli for the 20 DANDI stimulus classes, does its motor
output align with the DANDI calcium response target surface?

The answer is a runtime audit of the current controller path, not a parameter
fit and not a biological validation by itself.
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

from simulations.zebrafish.simzfish_omr import SimZFishOMRActionAdapter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DANDI_SUMMARY = ROOT / "analysis" / "out" / "dandi_omr_neural_validation_20260603" / "stimulus_response_summary.csv"
DEFAULT_OUT_DIR = ROOT / "analysis" / "out" / "dandi_omr_motor_response_audit_20260603"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_stimuli(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        out.append(
            {
                "dandi_rank": rank,
                "stimulus": row["stimulus"],
                "primary_axis": row.get("primary_axis", ""),
                "forward_drive": _safe_float(row.get("forward_drive")),
                "side_drive": _safe_float(row.get("side_drive")),
                "expansion_drive": _safe_float(row.get("expansion_drive")),
                "dandi_population_response_mean": _safe_float(row.get("population_response_mean")),
                "dandi_response_sem": _safe_float(row.get("population_response_sem_across_files")),
                "dandi_active_roi_fraction": _safe_float(row.get("selective_active_roi_fraction_mean")),
                "dandi_positive_roi_fraction": _safe_float(row.get("positive_roi_fraction_mean")),
            }
        )
    return out


def _base_frames(
    *,
    width: int,
    height: int,
    active: bool,
) -> tuple[np.ndarray, np.ndarray]:
    previous = np.full((height, width), 0.55, dtype=np.float32)
    current = previous.copy()
    if active:
        row0 = height // 2 + max(2, int(round(25.0 * height / 240.0)))
        previous[row0:, :] = 0.72
        current[row0:, :] = 0.34
    return current, previous


def _synthetic_flow_and_features(
    stim: dict[str, Any],
    *,
    width: int,
    height: int,
    frame_index: int,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    fwd = float(np.clip(stim["forward_drive"], -1.0, 1.0))
    side = float(np.clip(stim["side_drive"], -1.0, 1.0))
    exp = float(np.clip(stim["expansion_drive"], -1.0, 1.0))
    drive = float(np.linalg.norm([fwd, side, exp]))
    active = drive > 1e-6
    flow = np.zeros((height, width, 2), dtype=np.float32)
    if active:
        norm_u = max(1.0, width * 0.035)
        norm_v = max(1.0, height * 0.035)
        phase = 2.0 * math.pi * frame_index / 12.0
        flicker = 0.88 + 0.12 * math.sin(phase)

        # Projector-like convention:
        # - forward: opposing horizontal flow across the two eyes
        # - side: common horizontal flow plus visual asymmetry
        # - expansion/contraction: radial horizontal + lower-field vertical flow
        left_u = (-0.90 * fwd + 0.45 * side - 0.55 * exp) * flicker
        right_u = (0.90 * fwd + 0.45 * side + 0.55 * exp) * flicker
        v = (0.45 * exp - 0.20 * fwd) * flicker
        flow[:, : width // 2, 0] = left_u * norm_u
        flow[:, width // 2 :, 0] = right_u * norm_u
        flow[height // 2 :, :, 1] = v * norm_v

    visual_left = float(np.clip(0.42 + 0.18 * max(0.0, -side) + 0.06 * abs(exp), 0.0, 1.0))
    visual_right = float(np.clip(0.42 + 0.18 * max(0.0, side) + 0.06 * abs(exp), 0.0, 1.0))
    visual_up = float(np.clip(0.42 + 0.18 * max(0.0, fwd), 0.0, 1.0))
    visual_down = float(np.clip(0.42 + 0.18 * max(0.0, -fwd), 0.0, 1.0))
    motion_energy = float(np.clip(0.04 + 0.54 * min(1.0, drive / 1.5), 0.0, 1.0))
    startle = float(np.clip(0.12 * max(0.0, abs(exp) - 0.5) + 0.08 * max(0.0, drive - 1.2), 0.0, 1.0))
    asymmetry = float(np.clip(0.65 * side + 0.15 * fwd * side, -1.0, 1.0))
    features = {
        "visual_left": visual_left,
        "visual_right": visual_right,
        "visual_up": visual_up,
        "visual_down": visual_down,
        "optic_flow_left": float(np.clip(abs(-0.90 * fwd + 0.45 * side - 0.55 * exp), 0.0, 1.0)),
        "optic_flow_right": float(np.clip(abs(0.90 * fwd + 0.45 * side + 0.55 * exp), 0.0, 1.0)),
        "lateral_line_left": float(np.clip(max(0.0, -side) + 0.2 * abs(exp), 0.0, 1.0)),
        "lateral_line_right": float(np.clip(max(0.0, side) + 0.2 * abs(exp), 0.0, 1.0)),
        "light_level": 0.45,
        "startle": startle,
        "motion_energy": motion_energy,
        "asymmetry": asymmetry,
    }
    diagnostics = {
        "true_optical_flow": True,
        "camera_stabilized": True,
        "flow_coherence": 0.96 if active else 1.0,
        "residual_flow_coherence": 0.96 if active else 1.0,
        "camera_motion": 0.0,
        "camera_shake": 0.0,
        "compression_noise": 0.0,
        "contrast": 0.6 if active else 0.1,
        "flow_reliability": 0.96 if active else 1.0,
        "global_horizontal_flow": float(np.clip(0.45 * side, -1.0, 1.0)),
        "global_vertical_flow": float(np.clip(-0.25 * fwd + 0.25 * exp, -1.0, 1.0)),
        "global_rotation": 0.0,
        "affine_inlier_ratio": 1.0,
    }
    return flow, features, diagnostics


def _corr(x: list[float], y: list[float]) -> float:
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(ax) & np.isfinite(ay)
    if np.sum(keep) < 3:
        return 0.0
    ax = ax[keep]
    ay = ay[keep]
    if float(np.std(ax)) <= 1e-12 or float(np.std(ay)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(ax, ay)[0, 1])


def _rank(values: list[float], *, reverse: bool = True) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)
    ranks = [0 for _ in values]
    for rank, i in enumerate(order, start=1):
        ranks[i] = rank
    return ranks


def _simulate_stimulus(
    stim: dict[str, Any],
    *,
    sample_hz: float,
    seconds: float,
    width: int,
    height: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapter = SimZFishOMRActionAdapter()
    n_frames = int(round(sample_hz * seconds))
    current, previous = _base_frames(
        width=width,
        height=height,
        active=float(np.linalg.norm([stim["forward_drive"], stim["side_drive"], stim["expansion_drive"]])) > 1e-6,
    )
    frame_rows: list[dict[str, Any]] = []
    prev_noncoast = False
    transitions = 0
    bout_counter: Counter[str] = Counter()
    for frame_i in range(n_frames):
        flow, features, diagnostics = _synthetic_flow_and_features(stim, width=width, height=height, frame_index=frame_i)
        latent, diag = adapter.action_from_frame(
            current_gray=current,
            previous_gray=previous,
            residual_flow=flow,
            features=features,
            diagnostics=diagnostics,
            file_name=f"dandi_omr_synthetic:{stim['stimulus']}",
            frame_index=frame_i,
            video_time_s=frame_i / sample_hz,
            sample_hz=sample_hz,
        )
        state = latent.to_state_dict()
        noncoast = float(state["kick"]) >= 0.5 and float(state["force"]) > 0.02
        if noncoast and not prev_noncoast:
            transitions += 1
        prev_noncoast = noncoast
        bout_counter[str(state["bout_type"])] += 1
        tail_targets = np.asarray(state["tail_targets"], dtype=np.float64)
        frame_rows.append(
            {
                "stimulus": stim["stimulus"],
                "frame_index": frame_i,
                "time_s": frame_i / sample_hz,
                "kick": float(state["kick"]),
                "kick_score": float(state["kick_score"]),
                "force": float(state["force"]),
                "side_score": float(state["side_score"]),
                "confidence": float(state["confidence"]),
                "bout_type": state["bout_type"],
                "tail_frequency_hz": float(state["tail_frequency_hz"]),
                "tail_abs_max": float(np.max(np.abs(tail_targets))) if tail_targets.size else 0.0,
                "tail_rms": float(np.sqrt(np.mean(tail_targets * tail_targets))) if tail_targets.size else 0.0,
                "simzfish_left_pt": float(diag.get("simzfish_left_pt", 0.0)),
                "simzfish_right_pt": float(diag.get("simzfish_right_pt", 0.0)),
                "simzfish_turn_state": float(diag.get("simzfish_turn_state", 0.0)),
                "simzfish_ss_mlf": float(diag.get("simzfish_ss_mlf", 0.0)),
                "simzfish_candidate_force": float(diag.get("simzfish_candidate_force", 0.0)),
            }
        )
    force = np.asarray([row["force"] for row in frame_rows], dtype=np.float64)
    kick = np.asarray([row["kick"] for row in frame_rows], dtype=np.float64)
    confidence = np.asarray([row["confidence"] for row in frame_rows], dtype=np.float64)
    side_score = np.asarray([row["side_score"] for row in frame_rows], dtype=np.float64)
    tail_abs = np.asarray([row["tail_abs_max"] for row in frame_rows], dtype=np.float64)
    candidate_force = np.asarray([row["simzfish_candidate_force"] for row in frame_rows], dtype=np.float64)
    dominant_bout, dominant_count = bout_counter.most_common(1)[0]
    summary = {
        **stim,
        "frames": n_frames,
        "duration_s": seconds,
        "kick_fraction": float(np.mean(kick > 0.5)),
        "bout_transition_frequency_hz": float(transitions / seconds),
        "mean_force": float(np.mean(force)),
        "p95_force": float(np.quantile(force, 0.95)),
        "mean_confidence": float(np.mean(confidence)),
        "mean_abs_side_score": float(np.mean(np.abs(side_score))),
        "mean_side_score": float(np.mean(side_score)),
        "mean_tail_abs_max": float(np.mean(tail_abs)),
        "p95_tail_abs_max": float(np.quantile(tail_abs, 0.95)),
        "mean_candidate_force": float(np.mean(candidate_force)),
        "dominant_bout_type": dominant_bout,
        "dominant_bout_fraction": float(dominant_count / max(1, n_frames)),
        "motor_drive_index": float(np.mean(force * kick * confidence)),
    }
    return summary, frame_rows


def _save_rank_comparison(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "dandi_calcium_vs_motor_rank.png"
    labels = [row["stimulus"] for row in rows]
    dandi = np.asarray([row["dandi_population_response_mean"] for row in rows], dtype=np.float64)
    motor = np.asarray([row["motor_drive_index"] for row in rows], dtype=np.float64)
    x = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(x - 0.18, dandi, width=0.36, label="DANDI calcium response", color="#3182bd")
    ax1.set_ylabel("DANDI response")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, motor, width=0.36, label="current motor drive", color="#e6550d")
    ax2.set_ylabel("motor drive index")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_title("DANDI OMR calcium response rank versus current motor-adapter response")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_scatter(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "dandi_response_vs_motor_drive.png"
    x = np.asarray([row["dandi_population_response_mean"] for row in rows], dtype=np.float64)
    y = np.asarray([row["motor_drive_index"] for row in rows], dtype=np.float64)
    c = np.asarray([row["mean_abs_side_score"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(x, y, c=c, s=70, cmap="viridis")
    for row in rows:
        ax.text(row["dandi_population_response_mean"], row["motor_drive_index"], row["stimulus"], fontsize=8)
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlabel("DANDI population calcium response")
    ax.set_ylabel("current motor drive index")
    ax.set_title(f"Calcium/motor alignment across DANDI OMR classes (r={_corr(x.tolist(), y.tolist()):.3f})")
    fig.colorbar(sc, ax=ax, label="mean |side score|")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_heatmap(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "stimulus_motor_metric_heatmap.png"
    labels = [row["stimulus"] for row in rows]
    fields = [
        "dandi_population_response_mean",
        "kick_fraction",
        "bout_transition_frequency_hz",
        "mean_force",
        "mean_abs_side_score",
        "mean_tail_abs_max",
        "motor_drive_index",
    ]
    mat = np.asarray([[row[field] for field in fields] for row in rows], dtype=np.float64)
    z = (mat - np.mean(mat, axis=0, keepdims=True)) / np.maximum(1e-9, np.std(mat, axis=0, keepdims=True))
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(z, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(len(fields)))
    ax.set_xticklabels(fields, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("DANDI OMR class motor-response metric fingerprint")
    fig.colorbar(im, ax=ax, label="z-score across stimulus classes")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_timelines(out_dir: Path, frame_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "top_dandi_stimulus_motor_timelines.png"
    top_stims = [row["stimulus"] for row in sorted(summary_rows, key=lambda r: r["dandi_rank"])[:6]]
    fig, axes = plt.subplots(len(top_stims), 1, figsize=(13, max(4, 2.2 * len(top_stims))), sharex=True)
    if len(top_stims) == 1:
        axes = [axes]
    by_stim: dict[str, list[dict[str, Any]]] = {stim: [] for stim in top_stims}
    for row in frame_rows:
        stim = str(row["stimulus"])
        if stim in by_stim:
            by_stim[stim].append(row)
    for ax, stim in zip(axes, top_stims, strict=False):
        rows = by_stim[stim]
        t = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
        force = np.asarray([row["force"] for row in rows], dtype=np.float64)
        side = np.asarray([row["side_score"] for row in rows], dtype=np.float64)
        tail = np.asarray([row["tail_abs_max"] for row in rows], dtype=np.float64)
        ax.plot(t, force, label="force", color="#e6550d", linewidth=1.1)
        ax.plot(t, np.abs(side), label="|side|", color="#756bb1", linewidth=0.9)
        ax.plot(t, tail, label="tail abs max", color="#31a354", linewidth=0.9)
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel(stim, fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=3, loc="upper right")
    axes[-1].set_xlabel("stimulus time (s)")
    fig.suptitle("Current motor-adapter timelines for top DANDI calcium-response OMR classes", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_axis_plot(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "dandi_axis_motor_response.png"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fields = ["forward_drive", "side_drive", "expansion_drive"]
    y = np.asarray([row["motor_drive_index"] for row in rows], dtype=np.float64)
    for ax, field in zip(axes, fields, strict=True):
        x = np.asarray([row[field] for row in rows], dtype=np.float64)
        ax.scatter(x, y, s=65, color="#2b8cbe")
        for row in rows:
            ax.text(row[field], row["motor_drive_index"], row["stimulus"], fontsize=7)
        ax.set_xlabel(field)
        ax.set_ylabel("motor drive index")
        ax.set_title(f"{field} vs motor drive")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    plots: list[Path],
    sample_hz: float,
    seconds: float,
) -> None:
    dandi_values = [row["dandi_population_response_mean"] for row in summary_rows]
    motor_values = [row["motor_drive_index"] for row in summary_rows]
    dandi_ranks = [row["dandi_rank"] for row in summary_rows]
    motor_ranks = _rank(motor_values)
    rank_corr = _corr([float(x) for x in dandi_ranks], [float(x) for x in motor_ranks])
    value_corr = _corr(dandi_values, motor_values)
    best_motor = max(summary_rows, key=lambda row: row["motor_drive_index"])
    best_dandi = min(summary_rows, key=lambda row: row["dandi_rank"])
    lines = [
        "# DANDI OMR Stimulus-To-Motor Response Audit",
        "",
        "## Scope",
        "",
        "This audit feeds calibrated, projector-like synthetic versions of the 20 DANDI OMR stimulus classes into the current simZFish-inspired motor adapter. It measures whether the existing video-to-motion controller path naturally ranks motor output in a way that resembles the public DANDI calcium response atlas.",
        "",
        "No parameter fitting is performed. A mismatch is therefore evidence about the current implementation, not evidence against DANDI or Z-Robot.",
        "",
        "## Coverage",
        "",
        f"- Stimulus classes: `{len(summary_rows)}`",
        f"- Frames per stimulus: `{int(round(sample_hz * seconds))}`",
        f"- Total synthetic frames: `{len(frame_rows)}`",
        f"- Sample rate: `{sample_hz:.3f}` Hz",
        f"- Duration per stimulus: `{seconds:.3f}` s",
        "",
        "## Main Result",
        "",
        f"- DANDI response versus motor-drive value correlation: `{value_corr:.4f}`.",
        f"- DANDI rank versus motor-drive rank correlation: `{rank_corr:.4f}`.",
        f"- Highest DANDI calcium stimulus: `{best_dandi['stimulus']}` with response `{best_dandi['dandi_population_response_mean']:.6g}` and motor-drive rank `{motor_ranks[summary_rows.index(best_dandi)]}`.",
        f"- Highest motor-drive stimulus: `{best_motor['stimulus']}` with motor index `{best_motor['motor_drive_index']:.6g}` and DANDI rank `{best_motor['dandi_rank']}`.",
        "",
        "| DANDI rank | stimulus | axis | DANDI response | motor rank | motor drive | kick fraction | event Hz | mean force | dominant bout |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    row_to_motor_rank = {row["stimulus"]: rank for row, rank in zip(summary_rows, motor_ranks, strict=True)}
    for row in sorted(summary_rows, key=lambda r: r["dandi_rank"]):
        lines.append(
            f"| {int(row['dandi_rank'])} | `{row['stimulus']}` | `{row['primary_axis']}` | "
            f"{row['dandi_population_response_mean']:.6g} | {row_to_motor_rank[row['stimulus']]} | "
            f"{row['motor_drive_index']:.6g} | {row['kick_fraction']:.4f} | "
            f"{row['bout_transition_frequency_hz']:.4f} | {row['mean_force']:.4f} | `{row['dominant_bout_type']}` |"
        )
    lines.extend(
        [
            "",
            "## Generated Visualizations",
            "",
        ]
    )
    for plot in plots:
        lines.append(f"- `{plot.resolve()}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If calcium/motor correlations are weak or negative, the current motor adapter should not be claimed to be DANDI/Z-Robot neural-response aligned even when it can produce stable motion.",
            "- This audit is stricter than selected-video alignment because it removes underwater video artifacts and directly uses the known DANDI stimulus classes.",
            "- It still does not validate free-swimming kinematics, muscle force, PAULA neuron identity, or exact simZFish C/Webots equivalence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dandi-summary", type=Path, default=DEFAULT_DANDI_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stimuli = _load_stimuli(args.dandi_summary)
    summary_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for stim in stimuli:
        summary, frames = _simulate_stimulus(
            stim,
            sample_hz=args.sample_hz,
            seconds=args.seconds,
            width=args.width,
            height=args.height,
        )
        summary_rows.append(summary)
        frame_rows.extend(frames)
    motor_ranks = _rank([row["motor_drive_index"] for row in summary_rows])
    for row, motor_rank in zip(summary_rows, motor_ranks, strict=True):
        row["motor_drive_rank"] = motor_rank
        row["rank_delta_motor_minus_dandi"] = motor_rank - int(row["dandi_rank"])

    summary_path = out_dir / "stimulus_motor_response_summary.csv"
    frame_path = out_dir / "stimulus_motor_response_frames.csv"
    _write_csv(summary_path, summary_rows)
    _write_csv(frame_path, frame_rows)
    plots = [
        _save_rank_comparison(out_dir, sorted(summary_rows, key=lambda row: row["dandi_rank"])),
        _save_scatter(out_dir, summary_rows),
        _save_heatmap(out_dir, sorted(summary_rows, key=lambda row: row["dandi_rank"])),
        _save_timelines(out_dir, frame_rows, summary_rows),
        _save_axis_plot(out_dir, summary_rows),
    ]
    report_path = out_dir / "DANDI_OMR_MOTOR_RESPONSE_AUDIT.md"
    _write_report(
        report_path,
        summary_rows=summary_rows,
        frame_rows=frame_rows,
        plots=plots,
        sample_hz=args.sample_hz,
        seconds=args.seconds,
    )
    dandi_values = [row["dandi_population_response_mean"] for row in summary_rows]
    motor_values = [row["motor_drive_index"] for row in summary_rows]
    manifest = {
        "out_dir": str(out_dir.resolve()),
        "report": str(report_path.resolve()),
        "dandi_summary": str(args.dandi_summary.resolve()),
        "stimulus_count": len(summary_rows),
        "frames": len(frame_rows),
        "sample_hz": args.sample_hz,
        "seconds": args.seconds,
        "dandi_response_motor_drive_corr": _corr(dandi_values, motor_values),
        "dandi_rank_motor_rank_corr": _corr(
            [float(row["dandi_rank"]) for row in summary_rows],
            [float(row["motor_drive_rank"]) for row in summary_rows],
        ),
        "top_dandi_stimulus": min(summary_rows, key=lambda row: row["dandi_rank"])["stimulus"],
        "top_motor_stimulus": max(summary_rows, key=lambda row: row["motor_drive_index"])["stimulus"],
        "csv": {
            "stimulus_motor_response_summary": str(summary_path.resolve()),
            "stimulus_motor_response_frames": str(frame_path.resolve()),
        },
        "plots": [str(plot.resolve()) for plot in plots],
        "limitations": [
            "Synthetic projector-like stimuli, not actual DANDI visual movies.",
            "No fitting of current adapter parameters to DANDI calcium responses.",
            "Motor output is adapter-level, not full MuJoCo body kinematics.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
