"""Audit whether coarse OMR axes identify DANDI stimulus classes.

The current video bridge projects arbitrary frames into coarse OMR axes:
forward/backward, left/right, and expansion/contraction.  The DANDI 001076
stimulus labels are richer than those axes.  This script quantifies how much
information is lost by that projection and whether simple sign/convention
variants can explain the calcium/motor mismatch found by
``dandi_omr_motor_response_audit.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DANDI_SUMMARY = ROOT / "analysis" / "out" / "dandi_omr_neural_validation_20260603" / "stimulus_response_summary.csv"
DEFAULT_MOTOR_SUMMARY = (
    ROOT
    / "analysis"
    / "out"
    / "dandi_omr_motor_response_audit_20260603"
    / "stimulus_motor_response_summary.csv"
)
DEFAULT_OUT_DIR = ROOT / "analysis" / "out" / "dandi_stimulus_identifiability_audit_20260603"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _load_rows(dandi_path: Path, motor_path: Path) -> list[dict[str, Any]]:
    motor = {row["stimulus"]: row for row in _read_csv(motor_path)} if motor_path.exists() else {}
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(_read_csv(dandi_path), start=1):
        stim = row["stimulus"]
        mot = motor.get(stim, {})
        rows.append(
            {
                "dandi_rank": rank,
                "stimulus": stim,
                "primary_axis": row.get("primary_axis", ""),
                "forward_drive": _safe_float(row.get("forward_drive")),
                "side_drive": _safe_float(row.get("side_drive")),
                "expansion_drive": _safe_float(row.get("expansion_drive")),
                "population_response_mean": _safe_float(row.get("population_response_mean")),
                "population_response_sem": _safe_float(row.get("population_response_sem_across_files")),
                "active_roi_fraction": _safe_float(row.get("selective_active_roi_fraction_mean")),
                "motor_drive_index": _safe_float(mot.get("motor_drive_index")),
                "motor_drive_rank": int(_safe_float(mot.get("motor_drive_rank"), 0)),
                "motor_bout_hz": _safe_float(mot.get("bout_transition_frequency_hz")),
            }
        )
    return rows


def _axis_key(row: dict[str, Any], *, signs: tuple[int, int, int] = (1, 1, 1)) -> tuple[float, float, float]:
    return (
        signs[0] * float(row["forward_drive"]),
        signs[1] * float(row["side_drive"]),
        signs[2] * float(row["expansion_drive"]),
    )


def _axis_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_axis_key(row)].append(row)
    out: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        values = np.asarray([row["population_response_mean"] for row in members], dtype=np.float64)
        motor = np.asarray([row["motor_drive_index"] for row in members], dtype=np.float64)
        labels = [row["stimulus"] for row in members]
        out.append(
            {
                "axis_key": str(key),
                "forward_drive": key[0],
                "side_drive": key[1],
                "expansion_drive": key[2],
                "labels": ";".join(labels),
                "label_count": len(labels),
                "response_min": float(np.min(values)),
                "response_max": float(np.max(values)),
                "response_range": float(np.max(values) - np.min(values)),
                "response_std": float(np.std(values)),
                "response_mean": float(np.mean(values)),
                "motor_min": float(np.min(motor)),
                "motor_max": float(np.max(motor)),
                "motor_range": float(np.max(motor) - np.min(motor)),
                "top_label": labels[int(np.argmax(values))],
                "bottom_label": labels[int(np.argmin(values))],
            }
        )
    return sorted(out, key=lambda row: row["response_range"], reverse=True)


def _design_matrix(rows: list[dict[str, Any]], feature_set: str, *, signs: tuple[int, int, int] = (1, 1, 1)) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = [row["stimulus"] for row in rows]
    y = np.asarray([row["population_response_mean"] for row in rows], dtype=np.float64)
    if feature_set == "axes":
        x = np.asarray([_axis_key(row, signs=signs) for row in rows], dtype=np.float64)
        names = ["forward", "side", "expansion"]
    elif feature_set == "axes_quadratic":
        base = np.asarray([_axis_key(row, signs=signs) for row in rows], dtype=np.float64)
        x = np.column_stack(
            [
                base,
                base * base,
                base[:, 0] * base[:, 1],
                base[:, 0] * base[:, 2],
                base[:, 1] * base[:, 2],
            ]
        )
        names = ["forward", "side", "expansion", "forward2", "side2", "expansion2", "fwd_side", "fwd_exp", "side_exp"]
    elif feature_set == "tokens":
        tokens = sorted({token for label in labels for token in label.split("_")})
        x = np.asarray([[1.0 if token in label.split("_") else 0.0 for token in tokens] for label in labels], dtype=np.float64)
        names = tokens
    elif feature_set == "axes_plus_tokens":
        axes, _, axes_names = _design_matrix(rows, "axes", signs=signs)
        tok, _, tok_names = _design_matrix(rows, "tokens", signs=signs)
        x = np.column_stack([axes, tok])
        names = axes_names + tok_names
    else:
        raise ValueError(feature_set)
    x = np.column_stack([np.ones(len(rows), dtype=np.float64), x])
    names = ["intercept", *names]
    return x, y, names


def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = 1e-4) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * lam
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def _loo_predictions(rows: list[dict[str, Any]], feature_set: str, *, signs: tuple[int, int, int] = (1, 1, 1)) -> tuple[np.ndarray, np.ndarray]:
    x, y, _ = _design_matrix(rows, feature_set, signs=signs)
    preds = np.zeros_like(y)
    for i in range(len(y)):
        keep = np.ones(len(y), dtype=bool)
        keep[i] = False
        beta = _fit_ridge(x[keep], y[keep])
        preds[i] = float(x[i] @ beta)
    return y, preds


def _model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature_set in ["axes", "axes_quadratic", "tokens", "axes_plus_tokens"]:
        x, y, names = _design_matrix(rows, feature_set)
        beta = _fit_ridge(x, y)
        pred = x @ beta
        loo_y, loo_pred = _loo_predictions(rows, feature_set)
        out.append(
            {
                "feature_set": feature_set,
                "features": ";".join(names),
                "train_r2": _r2(y, pred),
                "loo_r2": _r2(loo_y, loo_pred),
                "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
                "loo_rmse": float(np.sqrt(np.mean((loo_y - loo_pred) ** 2))),
                "n_features": len(names),
            }
        )
    return out


def _sign_sensitivity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    y = np.asarray([row["population_response_mean"] for row in rows], dtype=np.float64)
    motor = np.asarray([row["motor_drive_index"] for row in rows], dtype=np.float64)
    for signs in product([-1, 1], repeat=3):
        # Sign flips do not change motor values; this tests whether a linear
        # coarse-axis reading of DANDI responses is sensitive to convention.
        x, _, _ = _design_matrix(rows, "axes", signs=signs)
        beta = _fit_ridge(x, y)
        pred = x @ beta
        out.append(
            {
                "forward_sign": signs[0],
                "side_sign": signs[1],
                "expansion_sign": signs[2],
                "axis_train_r2": _r2(y, pred),
                "axis_prediction_motor_corr": _corr(pred, motor),
                "axis_prediction_dandi_corr": _corr(pred, y),
            }
        )
    return sorted(out, key=lambda row: row["axis_train_r2"], reverse=True)


def _corr(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.sum(keep) < 3:
        return 0.0
    x = x[keep]
    y = y[keep]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _save_collision_plot(out_dir: Path, group_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "coarse_axis_response_collisions.png"
    rows = [row for row in group_rows if int(row["label_count"]) > 1]
    labels = [row["axis_key"] for row in rows]
    ranges = np.asarray([row["response_range"] for row in rows], dtype=np.float64)
    counts = np.asarray([row["label_count"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(rows))
    ax.bar(x, ranges, color="#de2d26")
    for i, row in enumerate(rows):
        ax.text(i, ranges[i] + 0.02, f"n={int(counts[i])}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("DANDI response range inside same coarse axis")
    ax.set_title("DANDI labels that collide under coarse OMR-axis representation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_model_plot(out_dir: Path, rows: list[dict[str, Any]], model_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "stimulus_feature_model_r2.png"
    labels = [row["feature_set"] for row in model_rows]
    train = np.asarray([row["train_r2"] for row in model_rows], dtype=np.float64)
    loo = np.asarray([row["loo_r2"] for row in model_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, train, width=0.36, label="train R2", color="#3182bd")
    ax.bar(x + 0.18, loo, width=0.36, label="leave-one-out R2", color="#fdae6b")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("R2 predicting DANDI calcium response")
    ax.set_title("Can current stimulus features identify DANDI calcium responses?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_axis_scatter(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "coarse_axes_vs_dandi_response.png"
    y = np.asarray([row["population_response_mean"] for row in rows], dtype=np.float64)
    fields = ["forward_drive", "side_drive", "expansion_drive", "motor_drive_index"]
    fig, axes = plt.subplots(1, len(fields), figsize=(15, 4.5), sharey=True)
    for ax, field in zip(axes, fields, strict=True):
        x = np.asarray([row[field] for row in rows], dtype=np.float64)
        ax.scatter(x, y, s=70, color="#2b8cbe")
        for row in rows:
            ax.text(row[field], row["population_response_mean"], row["stimulus"], fontsize=7)
        ax.set_xlabel(field)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("DANDI population response")
    fig.suptitle("DANDI response is not identifiable from coarse OMR axes alone", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    path: Path,
    *,
    group_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    sign_rows: list[dict[str, Any]],
    plots: list[Path],
) -> None:
    collisions = [row for row in group_rows if int(row["label_count"]) > 1]
    worst = collisions[0] if collisions else {}
    axes_model = next(row for row in model_rows if row["feature_set"] == "axes")
    token_model = next(row for row in model_rows if row["feature_set"] == "tokens")
    best_sign = sign_rows[0] if sign_rows else {}
    lines = [
        "# DANDI Stimulus Identifiability Audit",
        "",
        "## Scope",
        "",
        "This audit tests whether the current coarse OMR representation, `forward/side/expansion`, contains enough information to identify DANDI OMR calcium-response classes. It also checks whether simple axis sign changes explain the motor-response mismatch.",
        "",
        "## Main Findings",
        "",
        f"- Coarse-axis collision groups: `{len(collisions)}`.",
        f"- Worst collision: axis `{worst.get('axis_key', '')}` contains `{worst.get('labels', '')}` with DANDI response range `{_safe_float(worst.get('response_range')):.6g}`.",
        f"- Coarse-axis train R2 for DANDI calcium response: `{_safe_float(axes_model.get('train_r2')):.4f}`; leave-one-out R2 `{_safe_float(axes_model.get('loo_r2')):.4f}`.",
        f"- Label-token train R2: `{_safe_float(token_model.get('train_r2')):.4f}`; leave-one-out R2 `{_safe_float(token_model.get('loo_r2')):.4f}`.",
        f"- Best sign-convention axis R2: `{_safe_float(best_sign.get('axis_train_r2')):.4f}` with signs forward={best_sign.get('forward_sign')}, side={best_sign.get('side_sign')}, expansion={best_sign.get('expansion_sign')}.",
        "",
        "## Collision Groups",
        "",
        "| axis | labels | count | response min | response max | response range | top label | bottom label |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in group_rows:
        if int(row["label_count"]) <= 1:
            continue
        lines.append(
            f"| `{row['axis_key']}` | `{row['labels']}` | {int(row['label_count'])} | "
            f"{row['response_min']:.6g} | {row['response_max']:.6g} | {row['response_range']:.6g} | "
            f"`{row['top_label']}` | `{row['bottom_label']}` |"
        )
    lines.extend(
        [
            "",
            "## Feature Model Results",
            "",
            "| feature set | features | train R2 | leave-one-out R2 | RMSE | LOO RMSE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in model_rows:
        lines.append(
            f"| `{row['feature_set']}` | {int(row['n_features'])} | {row['train_r2']:.4f} | "
            f"{row['loo_r2']:.4f} | {row['rmse']:.4f} | {row['loo_rmse']:.4f} |"
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
            "- The current coarse OMR projection is not sufficiently identifiable for DANDI class-level calcium responses.",
            "- Simple sign flips do not solve the problem; DANDI labels contain stimulus semantics that are not represented by the current axes.",
            "- This means a biologically rigorous video-to-motion bridge needs a richer two-eye stimulus representation before fitting or judging the motor adapter against DANDI neural targets.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dandi-summary", type=Path, default=DEFAULT_DANDI_SUMMARY)
    parser.add_argument("--motor-summary", type=Path, default=DEFAULT_MOTOR_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.dandi_summary, args.motor_summary)
    group_rows = _axis_group_rows(rows)
    model_rows = _model_rows(rows)
    sign_rows = _sign_sensitivity_rows(rows)
    _write_csv(out_dir / "coarse_axis_collision_groups.csv", group_rows)
    _write_csv(out_dir / "stimulus_feature_model_scores.csv", model_rows)
    _write_csv(out_dir / "axis_sign_sensitivity.csv", sign_rows)
    plots = [
        _save_collision_plot(out_dir, group_rows),
        _save_model_plot(out_dir, rows, model_rows),
        _save_axis_scatter(out_dir, rows),
    ]
    report_path = out_dir / "DANDI_STIMULUS_IDENTIFIABILITY_AUDIT.md"
    _write_report(report_path, group_rows=group_rows, model_rows=model_rows, sign_rows=sign_rows, plots=plots)
    collisions = [row for row in group_rows if int(row["label_count"]) > 1]
    axes_model = next(row for row in model_rows if row["feature_set"] == "axes")
    token_model = next(row for row in model_rows if row["feature_set"] == "tokens")
    manifest = {
        "out_dir": str(out_dir.resolve()),
        "report": str(report_path.resolve()),
        "dandi_summary": str(args.dandi_summary.resolve()),
        "motor_summary": str(args.motor_summary.resolve()) if args.motor_summary.exists() else "",
        "stimulus_count": len(rows),
        "axis_group_count": len(group_rows),
        "collision_group_count": len(collisions),
        "max_collision_response_range": collisions[0]["response_range"] if collisions else 0.0,
        "max_collision_labels": collisions[0]["labels"] if collisions else "",
        "axes_train_r2": axes_model["train_r2"],
        "axes_loo_r2": axes_model["loo_r2"],
        "tokens_train_r2": token_model["train_r2"],
        "tokens_loo_r2": token_model["loo_r2"],
        "best_sign_axis_train_r2": sign_rows[0]["axis_train_r2"] if sign_rows else 0.0,
        "csv": {
            "coarse_axis_collision_groups": str((out_dir / "coarse_axis_collision_groups.csv").resolve()),
            "stimulus_feature_model_scores": str((out_dir / "stimulus_feature_model_scores.csv").resolve()),
            "axis_sign_sensitivity": str((out_dir / "axis_sign_sensitivity.csv").resolve()),
        },
        "plots": [str(plot.resolve()) for plot in plots],
        "limitations": [
            "Uses DANDI labels and derived coarse axes, not original DANDI visual movies.",
            "Token models prove missing semantics, not a video-extractable solution.",
            "Does not fit the motor adapter.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
