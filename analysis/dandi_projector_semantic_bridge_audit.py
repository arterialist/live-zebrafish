"""Audit a richer two-eye/projector representation for DANDI OMR classes.

The coarse DANDI/video bridge uses only three axes: forward, side, and
expansion.  That loses distinctions that the Z-Robot paper treats as
biologically important: which eye saw the stimulus, whether the stimulus was
monocular/binocular/conflicting, and how bottom-projected motion activates the
lower-posterior retina.

This script builds an explicit semantic and lower-posterior retinal code for
the 20 DANDI 001076 OMR labels, tests whether that representation is more
identifiable than the coarse axes, and re-scores cached selected-video frames
against the richer retinal code emitted by the current backend.

The result is still an audit, not a claim that arbitrary underwater video is an
exact replay of the DANDI projector movies.  It quantifies the next bridge that
must exist before video-to-calcium-to-motion claims can become stricter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DANDI_SUMMARY = ROOT / "analysis" / "out" / "dandi_omr_neural_validation_20260603" / "stimulus_response_summary.csv"
DEFAULT_VIDEO_FRAMES = (
    ROOT
    / "analysis"
    / "out"
    / "backend_video_robustness"
    / "20260603_all_selected_60s"
    / "backend_video_frame_metrics.csv"
)
DEFAULT_BACKEND_SUMMARY = (
    ROOT
    / "analysis"
    / "out"
    / "backend_video_robustness"
    / "20260603_all_selected_60s"
    / "backend_video_robustness_summary.csv"
)
DEFAULT_COARSE_ALIGNMENT = (
    ROOT
    / "analysis"
    / "out"
    / "dandi_video_omr_alignment_20260603"
    / "clip_neural_alignment_summary.csv"
)
DEFAULT_OUT_DIR = ROOT / "analysis" / "out" / "dandi_projector_semantic_bridge_audit_20260603"

RETINAL_CHANNELS = [
    "left_superior",
    "left_anterior",
    "left_inferior",
    "left_posterior",
    "right_superior",
    "right_anterior",
    "right_inferior",
    "right_posterior",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _load_dandi_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(_read_csv(path), start=1):
        rows.append(
            {
                "dandi_rank": rank,
                "stimulus": row["stimulus"],
                "primary_axis": row.get("primary_axis", ""),
                "forward_drive": _safe_float(row.get("forward_drive")),
                "side_drive": _safe_float(row.get("side_drive")),
                "expansion_drive": _safe_float(row.get("expansion_drive")),
                "population_response_mean": _safe_float(row.get("population_response_mean")),
                "population_response_sem": _safe_float(row.get("population_response_sem_across_files")),
                "active_roi_fraction": _safe_float(row.get("selective_active_roi_fraction_mean")),
                "positive_roi_fraction": _safe_float(row.get("positive_roi_fraction_mean")),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No DANDI rows loaded from {path}")
    return rows


def _tokens(label: str) -> list[str]:
    return [token for token in label.lower().split("_") if token]


def _empty_eye_state() -> dict[str, str]:
    return {"left": "x", "right": "x", "class": "unknown", "notes": ""}


def _stimulus_eye_state(label: str, *, order: str) -> dict[str, str]:
    """Map a DANDI label into a two-eye motion hypothesis.

    The public NWB files expose labels, not the original projector movies.  For
    ambiguous labels such as ``x_forward`` and ``forward_x`` this function keeps
    two testable hypotheses: the first token is either the left eye or the right
    eye.  Nonambiguous monocular labels such as ``forward_left`` are treated as
    stimulus-motion plus eye side.
    """

    tokens = _tokens(label)
    state = _empty_eye_state()
    if label == "converging":
        state.update({"left": "medial", "right": "medial", "class": "binocular_converging"})
    elif label == "diverging":
        state.update({"left": "lateral", "right": "lateral", "class": "binocular_diverging"})
    elif label == "forward":
        state.update({"left": "forward", "right": "forward", "class": "binocular_forward"})
    elif label == "backward":
        state.update({"left": "backward", "right": "backward", "class": "binocular_backward"})
    elif label == "left":
        # Whole-field leftward motion is represented as opposite monocular
        # medial/lateral flow.  The sign convention is tested only as a
        # semantic code, not a projector-movie reconstruction.
        state.update({"left": "lateral", "right": "medial", "class": "binocular_left"})
    elif label == "right":
        state.update({"left": "medial", "right": "lateral", "class": "binocular_right"})
    elif len(tokens) == 2 and tokens[1] in {"left", "right"} and tokens[0] in {
        "forward",
        "backward",
        "medial",
        "lateral",
    }:
        eye = tokens[1]
        state[eye] = tokens[0]
        state["class"] = f"monocular_{eye}_{tokens[0]}"
    elif len(tokens) == 2 and set(tokens) <= {"x", "forward", "backward"}:
        first_eye, second_eye = ("left", "right") if order == "first_left" else ("right", "left")
        state[first_eye] = tokens[0]
        state[second_eye] = tokens[1]
        if "x" in tokens:
            active_eye = second_eye if tokens[0] == "x" else first_eye
            active_motion = tokens[1] if tokens[0] == "x" else tokens[0]
            state["class"] = f"monocular_{active_eye}_{active_motion}"
        elif tokens[0] != tokens[1]:
            state["class"] = "binocular_forward_backward_conflict"
        else:
            state["class"] = f"binocular_{tokens[0]}"
    else:
        state["notes"] = "unparsed_label"
    return state


def _motion_to_retina(motion: str) -> dict[str, float]:
    """Lower-posterior retinal counter proxy from Z-Robot Fig. 4 text.

    The paper reports that lower-posterior forward and medial motion coactivate
    anterior and inferior direction-selective channels.  Backward/lateral are
    encoded as the complementary posterior/superior pattern for this audit.
    """

    if motion in {"forward", "medial"}:
        return {"superior": 0.0, "anterior": 1.0, "inferior": 1.0, "posterior": 0.0}
    if motion in {"backward", "lateral"}:
        return {"superior": 1.0, "anterior": 0.0, "inferior": 0.0, "posterior": 1.0}
    return {"superior": 0.0, "anterior": 0.0, "inferior": 0.0, "posterior": 0.0}


def _semantic_features(row: dict[str, Any], *, order: str) -> dict[str, float | str]:
    label = str(row["stimulus"])
    state = _stimulus_eye_state(label, order=order)
    left = state["left"]
    right = state["right"]
    left_ret = _motion_to_retina(left)
    right_ret = _motion_to_retina(right)
    left_excit = 0.5 * (left_ret["anterior"] + left_ret["inferior"])
    right_excit = 0.5 * (right_ret["anterior"] + right_ret["inferior"])
    left_suppress = 0.5 * (left_ret["posterior"] + left_ret["superior"])
    right_suppress = 0.5 * (right_ret["posterior"] + right_ret["superior"])
    active_eyes = int(left != "x") + int(right != "x")
    conflict_fb = int({left, right} == {"forward", "backward"})
    same_motion = int(left == right and left != "x")
    monocular = int(active_eyes == 1)
    binocular = int(active_eyes == 2)
    blank_eyes = 2 - active_eyes
    semantic: dict[str, float | str] = {
        "order_hypothesis": order,
        "stimulus": label,
        "left_motion": left,
        "right_motion": right,
        "semantic_class": state["class"],
        "left_forward": float(left == "forward"),
        "left_backward": float(left == "backward"),
        "left_medial": float(left == "medial"),
        "left_lateral": float(left == "lateral"),
        "right_forward": float(right == "forward"),
        "right_backward": float(right == "backward"),
        "right_medial": float(right == "medial"),
        "right_lateral": float(right == "lateral"),
        "monocular": float(monocular),
        "binocular": float(binocular),
        "blank_eye_count": float(blank_eyes),
        "same_motion": float(same_motion),
        "conflict_forward_backward": float(conflict_fb),
        "forward_backward_balance": float((left == "forward") + (right == "forward") - (left == "backward") - (right == "backward")),
        "medial_lateral_balance": float((left == "medial") + (right == "medial") - (left == "lateral") - (right == "lateral")),
        "left_excitatory_lp": float(left_excit),
        "right_excitatory_lp": float(right_excit),
        "left_suppressive_lp": float(left_suppress),
        "right_suppressive_lp": float(right_suppress),
        "bilateral_excitatory_lp": float(0.5 * (left_excit + right_excit)),
        "bilateral_suppressive_lp": float(0.5 * (left_suppress + right_suppress)),
        "turn_bias_lp": float(right_excit - left_excit + left_suppress - right_suppress),
        "net_locomotor_lp": float(0.5 * (left_excit + right_excit) - 0.5 * (left_suppress + right_suppress)),
        "coarse_forward": float(row["forward_drive"]),
        "coarse_side": float(row["side_drive"]),
        "coarse_expansion": float(row["expansion_drive"]),
    }
    for eye, ret in [("left", left_ret), ("right", right_ret)]:
        for direction, value in ret.items():
            semantic[f"{eye}_{direction}"] = float(value)
    return semantic


def _all_semantic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for order in ["first_left", "first_right"]:
        for row in rows:
            semantic = _semantic_features(row, order=order)
            merged = {
                **row,
                **semantic,
            }
            out.append(merged)
    return out


def _feature_names(kind: str) -> list[str]:
    if kind == "coarse_axes":
        return ["coarse_forward", "coarse_side", "coarse_expansion"]
    if kind == "eye_motion":
        return [
            "left_forward",
            "left_backward",
            "left_medial",
            "left_lateral",
            "right_forward",
            "right_backward",
            "right_medial",
            "right_lateral",
            "monocular",
            "binocular",
            "blank_eye_count",
            "same_motion",
            "conflict_forward_backward",
        ]
    if kind == "retinal_lp8":
        return RETINAL_CHANNELS
    if kind == "paper_semantic":
        return [
            "forward_backward_balance",
            "medial_lateral_balance",
            "left_excitatory_lp",
            "right_excitatory_lp",
            "left_suppressive_lp",
            "right_suppressive_lp",
            "bilateral_excitatory_lp",
            "bilateral_suppressive_lp",
            "turn_bias_lp",
            "net_locomotor_lp",
            "monocular",
            "binocular",
            "blank_eye_count",
            "conflict_forward_backward",
        ]
    if kind == "paper_semantic_plus_axes":
        return _feature_names("paper_semantic") + _feature_names("coarse_axes")
    raise ValueError(kind)


def _matrix(rows: list[dict[str, Any]], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
    y = np.asarray([float(row["population_response_mean"]) for row in rows], dtype=np.float64)
    return x, y


def _standardize_train_apply(x_train: np.ndarray, x_apply: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (x_train - mean) / std, (x_apply - mean) / std


def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    x1 = np.column_stack([np.ones(x.shape[0]), x])
    penalty = np.eye(x1.shape[1]) * float(lam)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x1.T @ x1 + penalty, x1.T @ y)


def _predict_ridge(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x]) @ beta


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def _corr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
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


def _loo_predictions(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    preds = np.zeros_like(y)
    for i in range(y.size):
        keep = np.ones(y.size, dtype=bool)
        keep[i] = False
        x_train, x_test = _standardize_train_apply(x[keep], x[[i]])
        beta = _fit_ridge(x_train, y[keep], lam)
        preds[i] = float(_predict_ridge(beta, x_test)[0])
    return preds


def _collision_stats(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    groups: dict[tuple[float, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(round(float(row[name]), 6) for name in names)
        groups[key].append(row)
    collision_groups = [members for members in groups.values() if len(members) > 1]
    worst_range = 0.0
    worst_labels = ""
    for members in collision_groups:
        values = [float(row["population_response_mean"]) for row in members]
        response_range = max(values) - min(values)
        if response_range > worst_range:
            worst_range = response_range
            worst_labels = ";".join(str(row["stimulus"]) for row in members)
    return {
        "signature_count": len(groups),
        "collision_group_count": len(collision_groups),
        "max_collision_response_range": float(worst_range),
        "max_collision_labels": worst_labels,
    }


def _feature_collision_rows(semantic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for order in ["first_left", "first_right"]:
        rows = [row for row in semantic_rows if row["order_hypothesis"] == order]
        for kind in ["coarse_axes", "eye_motion", "retinal_lp8", "paper_semantic", "paper_semantic_plus_axes"]:
            names = _feature_names(kind)
            groups: dict[tuple[float, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                key = tuple(round(float(row[name]), 6) for name in names)
                groups[key].append(row)
            for key, members in groups.items():
                if len(members) <= 1:
                    continue
                values = np.asarray([float(row["population_response_mean"]) for row in members], dtype=np.float64)
                labels = [str(row["stimulus"]) for row in members]
                out.append(
                    {
                        "order_hypothesis": order,
                        "feature_set": kind,
                        "signature": json.dumps(key),
                        "label_count": len(labels),
                        "labels": ";".join(labels),
                        "response_min": float(np.min(values)),
                        "response_max": float(np.max(values)),
                        "response_range": float(np.max(values) - np.min(values)),
                        "top_label": labels[int(np.argmax(values))],
                        "bottom_label": labels[int(np.argmin(values))],
                    }
                )
    return sorted(out, key=lambda row: row["response_range"], reverse=True)


def _model_audit_rows(semantic_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    model_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for order in ["first_left", "first_right"]:
        rows = [row for row in semantic_rows if row["order_hypothesis"] == order]
        for kind in ["coarse_axes", "eye_motion", "retinal_lp8", "paper_semantic", "paper_semantic_plus_axes"]:
            names = _feature_names(kind)
            x, y = _matrix(rows, names)
            best: dict[str, Any] | None = None
            for lam in lambdas:
                x_std, _ = _standardize_train_apply(x, x)
                beta = _fit_ridge(x_std, y, lam)
                train_pred = _predict_ridge(beta, x_std)
                loo_pred = _loo_predictions(x, y, lam)
                row = {
                    "order_hypothesis": order,
                    "feature_set": kind,
                    "lambda": lam,
                    "n": len(rows),
                    "n_features": len(names),
                    "features": ";".join(names),
                    "train_r2": _r2(y, train_pred),
                    "loo_r2": _r2(y, loo_pred),
                    "train_rmse": float(np.sqrt(np.mean((y - train_pred) ** 2))),
                    "loo_rmse": float(np.sqrt(np.mean((y - loo_pred) ** 2))),
                    "loo_corr": _corr(y, loo_pred),
                    **_collision_stats(rows, names),
                }
                if best is None or row["loo_r2"] > best["loo_r2"]:
                    best = row
                    best_pred = loo_pred
            assert best is not None
            model_rows.append(best)
            for row, pred in zip(rows, best_pred, strict=True):
                prediction_rows.append(
                    {
                        "order_hypothesis": order,
                        "feature_set": kind,
                        "stimulus": row["stimulus"],
                        "left_motion": row["left_motion"],
                        "right_motion": row["right_motion"],
                        "semantic_class": row["semantic_class"],
                        "observed_response": row["population_response_mean"],
                        "loo_predicted_response": float(pred),
                        "loo_error": float(pred - row["population_response_mean"]),
                    }
                )
    return sorted(model_rows, key=lambda row: row["loo_r2"], reverse=True), prediction_rows


def _permutation_rows(
    semantic_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    *,
    n_perm: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for model in model_rows:
        order = model["order_hypothesis"]
        kind = model["feature_set"]
        lam = float(model["lambda"])
        rows = [row for row in semantic_rows if row["order_hypothesis"] == order]
        x, y = _matrix(rows, _feature_names(kind))
        observed = float(model["loo_r2"])
        null = []
        for _ in range(n_perm):
            yp = np.asarray(y, dtype=np.float64).copy()
            rng.shuffle(yp)
            pred = _loo_predictions(x, yp, lam)
            null.append(_r2(yp, pred))
        null_arr = np.asarray(null, dtype=np.float64)
        out.append(
            {
                "order_hypothesis": order,
                "feature_set": kind,
                "lambda": lam,
                "observed_loo_r2": observed,
                "null_mean_loo_r2": float(np.mean(null_arr)),
                "null_p95_loo_r2": float(np.quantile(null_arr, 0.95)),
                "permutation_p_ge_observed": float((np.sum(null_arr >= observed) + 1) / (n_perm + 1)),
                "n_perm": n_perm,
            }
        )
    return sorted(out, key=lambda row: row["observed_loo_r2"], reverse=True)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-9:
        return v * 0.0
    return v / norm


def _quality_weight(row: dict[str, str]) -> tuple[float, int]:
    high_camera = _safe_float(row.get("high_camera_shake")) > 0.5
    high_compression = _safe_float(row.get("high_compression_noise")) > 0.5
    low_flow = _safe_float(row.get("low_flow_reliability")) > 0.5
    flags = int(high_camera) + int(high_compression) + int(low_flow)
    reliability = max(0.0, min(1.0, _safe_float(row.get("flow_reliability"), 0.0)))
    inlier = max(0.0, min(1.0, _safe_float(row.get("affine_inlier_ratio"), 1.0)))
    coherence = max(
        0.0,
        min(1.0, _safe_float(row.get("residual_flow_coherence"), _safe_float(row.get("flow_coherence"), 0.0))),
    )
    penalty = 0.35 * high_camera + 0.25 * high_compression + 0.35 * low_flow
    quality = (0.45 * reliability) + (0.30 * inlier) + (0.25 * coherence)
    return max(0.0, min(1.0, quality * (1.0 - penalty))), flags


def _video_retinal_vector(row: dict[str, str]) -> np.ndarray:
    values = [
        _safe_float(row.get(f"simzfish_retinal_counters_{name}"))
        for name in RETINAL_CHANNELS
    ]
    arr = np.asarray(values, dtype=np.float64)
    return np.clip(arr, 0.0, None)


def _dandi_retinal_matrix(rows: list[dict[str, Any]], *, order: str) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    semantic = [_semantic_features(row, order=order) for row in rows]
    mat = np.asarray([[float(row[name]) for name in RETINAL_CHANNELS] for row in semantic], dtype=np.float64)
    response = np.asarray([float(row["population_response_mean"]) for row in rows], dtype=np.float64)
    active = np.asarray([float(row["active_roi_fraction"]) for row in rows], dtype=np.float64)
    return semantic, mat, response, active


def _soft_match(
    video_vec: np.ndarray,
    dandi_vecs: np.ndarray,
    rows: list[dict[str, Any]],
    response: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    vu = _unit(video_vec)
    du = np.asarray([_unit(v) for v in dandi_vecs], dtype=np.float64)
    if float(np.linalg.norm(vu)) <= 1e-9:
        distances = np.ones(dandi_vecs.shape[0], dtype=np.float64)
    else:
        distances = 1.0 - np.clip(du @ vu, -1.0, 1.0)
    order = np.argsort(distances)
    weights = np.exp(-4.0 * distances)
    weights = weights / max(1e-12, float(np.sum(weights)))
    soft_response = float(np.sum(weights * response))
    soft_active = float(np.sum(weights * active))
    response_min = float(np.min(response))
    response_max = float(np.max(response))
    plaus = (soft_response - response_min) / max(1e-9, response_max - response_min)
    top = rows[int(order[0])]
    second = rows[int(order[1])] if len(order) > 1 else {}
    return {
        "nearest_stimulus": top["stimulus"],
        "nearest_primary_axis": top.get("primary_axis", ""),
        "nearest_semantic_class": top.get("semantic_class", ""),
        "nearest_left_motion": top.get("left_motion", ""),
        "nearest_right_motion": top.get("right_motion", ""),
        "nearest_distance": float(distances[int(order[0])]),
        "nearest_dandi_response": float(response[int(order[0])]),
        "second_stimulus": second.get("stimulus", ""),
        "second_distance": float(distances[int(order[1])]) if len(order) > 1 else 0.0,
        "soft_dandi_response": soft_response,
        "soft_dandi_active_fraction": soft_active,
        "dandi_neural_plausibility": max(0.0, min(1.0, float(plaus))),
    }


def _video_alignment_rows(
    video_rows: list[dict[str, str]],
    dandi_rows: list[dict[str, Any]],
    *,
    order: str,
) -> list[dict[str, Any]]:
    semantic, dandi_vecs, response, active = _dandi_retinal_matrix(dandi_rows, order=order)
    # Merge source DANDI fields with semantic fields for matching metadata.
    semantic_rows = [{**src, **sem} for src, sem in zip(dandi_rows, semantic, strict=True)]
    rows: list[dict[str, Any]] = []
    for row in video_rows:
        vec = _video_retinal_vector(row)
        match = _soft_match(vec, dandi_vecs, semantic_rows, response, active)
        q, flags = _quality_weight(row)
        force = _safe_float(row.get("action_force"))
        confidence = _safe_float(row.get("action_confidence"))
        out = {
            "order_hypothesis": order,
            "clip": row.get("clip", ""),
            "frame_index": _safe_int(row.get("frame_index")),
            "video_time_s": _safe_float(row.get("video_time_s")),
            "action_bout_type": row.get("action_bout_type", ""),
            "action_force": force,
            "action_kick": _safe_float(row.get("action_kick")),
            "action_confidence": confidence,
            "action_side_score": _safe_float(row.get("action_side_score")),
            "camera_shake": _safe_float(row.get("camera_shake")),
            "compression_noise": _safe_float(row.get("compression_noise")),
            "flow_reliability": _safe_float(row.get("flow_reliability")),
            "quality_weight": q,
            "quality_flags": flags,
            "retinal_energy": float(np.linalg.norm(vec)),
            **{f"video_{name}": float(value) for name, value in zip(RETINAL_CHANNELS, vec, strict=True)},
            **match,
        }
        out["quality_weighted_plausibility"] = q * out["dandi_neural_plausibility"]
        out["quality_weighted_dandi_response"] = q * out["soft_dandi_response"]
        out["neural_drive_product"] = out["dandi_neural_plausibility"] * force * confidence
        rows.append(out)
    return rows


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = np.asarray([count / total for count in counts.values() if count > 0], dtype=np.float64)
    return float(-np.sum(probs * np.log2(probs)))


def _clip_summary_rows(
    frame_rows: list[dict[str, Any]],
    backend_rows: list[dict[str, str]],
    coarse_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    backend_by_clip = {row.get("clip", ""): row for row in backend_rows}
    coarse_by_clip = {row.get("clip", ""): row for row in coarse_rows}
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_clip[str(row["clip"])].append(row)
    summaries: list[dict[str, Any]] = []
    for clip, rows in sorted(by_clip.items()):
        counts = Counter(str(row["nearest_stimulus"]) for row in rows)
        classes = Counter(str(row["nearest_semantic_class"]) for row in rows)
        top_stim, top_stim_count = counts.most_common(1)[0]
        top_class, top_class_count = classes.most_common(1)[0]
        quality = np.asarray([float(row["quality_weight"]) for row in rows], dtype=np.float64)
        plaus = np.asarray([float(row["dandi_neural_plausibility"]) for row in rows], dtype=np.float64)
        force = np.asarray([float(row["action_force"]) for row in rows], dtype=np.float64)
        distance = np.asarray([float(row["nearest_distance"]) for row in rows], dtype=np.float64)
        retinal_energy = np.asarray([float(row["retinal_energy"]) for row in rows], dtype=np.float64)
        backend = backend_by_clip.get(clip, {})
        coarse = coarse_by_clip.get(clip, {})
        summaries.append(
            {
                "order_hypothesis": rows[0]["order_hypothesis"],
                "clip": clip,
                "frames": len(rows),
                "duration_s": float(np.nanmax([float(row["video_time_s"]) for row in rows])) if rows else 0.0,
                "top_stimulus": top_stim,
                "top_stimulus_fraction": top_stim_count / max(1, len(rows)),
                "top_semantic_class": top_class,
                "top_semantic_class_fraction": top_class_count / max(1, len(rows)),
                "stimulus_entropy_bits": _entropy(counts),
                "mean_neural_plausibility": float(np.nanmean(plaus)),
                "quality_weighted_neural_plausibility": float(np.nanmean(plaus * quality)),
                "mean_quality_weight": float(np.nanmean(quality)),
                "mean_retinal_energy": float(np.nanmean(retinal_energy)),
                "mean_nearest_distance": float(np.nanmean(distance)),
                "p95_nearest_distance": float(np.nanquantile(distance, 0.95)),
                "mean_action_force": float(np.nanmean(force)),
                "force_vs_plausibility_corr": _corr(force, plaus),
                "quality_event_fraction": _safe_float(
                    backend.get("quality_event_fraction"),
                    float(np.nanmean([row["quality_flags"] > 0 for row in rows])),
                ),
                "backend_event_frequency_hz": _safe_float(backend.get("noncoast_event_frequency_hz")),
                "coarse_quality_weighted_plausibility": _safe_float(
                    coarse.get("quality_weighted_neural_plausibility"),
                    float("nan"),
                ),
                "coarse_top_stimulus": coarse.get("top_stimulus", ""),
                "rich_minus_coarse_quality_weighted_plausibility": float(
                    np.nanmean(plaus * quality)
                    - _safe_float(coarse.get("quality_weighted_neural_plausibility"), 0.0)
                ),
            }
        )
    return sorted(summaries, key=lambda row: row["quality_weighted_neural_plausibility"], reverse=True)


def _stimulus_fraction_matrix(frame_rows: list[dict[str, Any]], dandi_rows: list[dict[str, Any]]) -> tuple[list[str], list[str], np.ndarray]:
    clips = sorted({str(row["clip"]) for row in frame_rows})
    labels = [str(row["stimulus"]) for row in dandi_rows]
    mat = np.zeros((len(clips), len(labels)), dtype=np.float64)
    clip_i = {clip: i for i, clip in enumerate(clips)}
    label_i = {label: i for i, label in enumerate(labels)}
    counts = Counter(str(row["clip"]) for row in frame_rows)
    for row in frame_rows:
        clip = str(row["clip"])
        stim = str(row["nearest_stimulus"])
        if stim in label_i:
            mat[clip_i[clip], label_i[stim]] += 1.0 / max(1, counts[clip])
    return clips, labels, mat


def _pca2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x, axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def _save_retinal_code_plot(out_dir: Path, semantic_rows: list[dict[str, Any]], *, order: str) -> Path:
    path = out_dir / "dandi_projector_retinal_code_matrix.png"
    rows = [row for row in semantic_rows if row["order_hypothesis"] == order]
    mat = np.asarray([[float(row[name]) for name in RETINAL_CHANNELS] for row in rows], dtype=np.float64)
    labels = [str(row["stimulus"]) for row in rows]
    responses = [float(row["population_response_mean"]) for row in rows]
    order_idx = np.argsort(responses)[::-1]
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(mat[order_idx], aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(RETINAL_CHANNELS)))
    ax.set_xticklabels(RETINAL_CHANNELS, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(order_idx)))
    ax.set_yticklabels([labels[i] for i in order_idx], fontsize=8)
    ax.set_title(f"DANDI OMR labels as lower-posterior retinal code ({order})")
    fig.colorbar(im, ax=ax, label="channel activation proxy")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_model_plot(out_dir: Path, model_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "semantic_feature_model_r2.png"
    rows = sorted(model_rows, key=lambda row: (row["order_hypothesis"], row["feature_set"]))
    labels = [f"{row['order_hypothesis']}\\n{row['feature_set']}" for row in rows]
    loo = np.asarray([row["loo_r2"] for row in rows], dtype=np.float64)
    train = np.asarray([row["train_r2"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, train, width=0.36, label="train R2", color="#74add1")
    ax.bar(x + 0.18, loo, width=0.36, label="leave-one-out R2", color="#f46d43")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R2 for DANDI population calcium response")
    ax.set_title("Does a richer two-eye/projector representation identify DANDI OMR calcium responses?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_prediction_plot(out_dir: Path, prediction_rows: list[dict[str, Any]], best_model: dict[str, Any]) -> Path:
    path = out_dir / "semantic_prediction_vs_dandi_response.png"
    rows = [
        row
        for row in prediction_rows
        if row["order_hypothesis"] == best_model["order_hypothesis"]
        and row["feature_set"] == best_model["feature_set"]
    ]
    x = np.asarray([row["observed_response"] for row in rows], dtype=np.float64)
    y = np.asarray([row["loo_predicted_response"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, s=45, color="#2b8cbe")
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
    for row in rows:
        ax.text(row["observed_response"], row["loo_predicted_response"], str(row["stimulus"]), fontsize=7)
    ax.set_xlabel("Observed DANDI population response")
    ax.set_ylabel("LOO predicted response")
    ax.set_title(
        f"Best semantic model: {best_model['order_hypothesis']} / {best_model['feature_set']} "
        f"(LOO R2={best_model['loo_r2']:.3f})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_embedding_plot(out_dir: Path, semantic_rows: list[dict[str, Any]], *, order: str) -> Path:
    path = out_dir / "dandi_semantic_retinal_embedding.png"
    rows = [row for row in semantic_rows if row["order_hypothesis"] == order]
    names = _feature_names("paper_semantic_plus_axes")
    x = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
    coords = _pca2(x)
    responses = np.asarray([row["population_response_mean"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=responses, cmap="coolwarm", s=70)
    for i, row in enumerate(rows):
        ax.text(coords[i, 0], coords[i, 1], str(row["stimulus"]), fontsize=7)
    ax.set_xlabel("semantic PC1")
    ax.set_ylabel("semantic PC2")
    ax.set_title("DANDI stimulus embedding from two-eye/projector semantic code")
    fig.colorbar(sc, ax=ax, label="DANDI population response")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_clip_rank_plot(out_dir: Path, summary_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "rich_video_clip_alignment_rank.png"
    rows = sorted(summary_rows, key=lambda row: row["quality_weighted_neural_plausibility"], reverse=True)
    labels = [row["clip"] for row in rows]
    rich = np.asarray([row["quality_weighted_neural_plausibility"] for row in rows], dtype=np.float64)
    coarse = np.asarray([row["coarse_quality_weighted_plausibility"] for row in rows], dtype=np.float64)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 0.18, rich, width=0.36, label="rich retinal-code alignment", color="#238b45")
    ax.bar(x + 0.18, coarse, width=0.36, label="old coarse-axis alignment", color="#9ecae1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("quality-weighted DANDI plausibility")
    ax.set_title("Selected-video DANDI alignment: rich retinal code versus coarse axes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_fraction_heatmap(out_dir: Path, frame_rows: list[dict[str, Any]], dandi_rows: list[dict[str, Any]]) -> Path:
    path = out_dir / "rich_video_stimulus_fraction_heatmap.png"
    clips, labels, mat = _stimulus_fraction_matrix(frame_rows, dandi_rows)
    fig, ax = plt.subplots(figsize=(15, 8))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=max(0.05, float(np.max(mat))))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(clips)))
    ax.set_yticklabels(clips, fontsize=8)
    ax.set_title("Nearest DANDI class fractions using rich retinal-code matching")
    fig.colorbar(im, ax=ax, label="fraction of backend frames")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    path: Path,
    *,
    model_rows: list[dict[str, Any]],
    permutation_rows: list[dict[str, Any]],
    collision_rows: list[dict[str, Any]],
    clip_rows: list[dict[str, Any]],
    best_order: str,
    plots: list[Path],
    semantic_rows: list[dict[str, Any]],
) -> None:
    best = model_rows[0]
    baseline = next(
        row for row in model_rows if row["order_hypothesis"] == best_order and row["feature_set"] == "coarse_axes"
    )
    best_perm = next(
        row
        for row in permutation_rows
        if row["order_hypothesis"] == best["order_hypothesis"] and row["feature_set"] == best["feature_set"]
    )
    zero_collision = [
        row
        for row in model_rows
        if int(row["collision_group_count"]) == 0 and row["order_hypothesis"] == best_order
    ]
    best_zero_collision = max(zero_collision, key=lambda row: float(row["loo_r2"]), default={})
    collision_delta = int(best["collision_group_count"]) - int(baseline["collision_group_count"])
    loo_delta = float(best["loo_r2"]) - float(baseline["loo_r2"])
    coarse_collisions = baseline["collision_group_count"]
    best_collisions = best["collision_group_count"]
    top_clip = clip_rows[0] if clip_rows else {}
    rows_for_order = [row for row in semantic_rows if row["order_hypothesis"] == best_order]
    lines = [
        "# DANDI Projector Semantic Bridge Audit",
        "",
        "## Scope",
        "",
        "This audit tests whether a richer two-eye/projector-like representation is a better bridge between public Z-Robot/DANDI OMR calcium data and the current zebrafish lab video branch than the previous coarse `forward/side/expansion` projection.",
        "",
        "The representation is paper-derived but still not a substitute for the original projected movies: it encodes eye specificity, monocular/binocular/conflict structure, and lower-posterior retinal direction-channel proxies motivated by the Z-Robot/simZFish methods.",
        "",
        "## Main Findings",
        "",
        f"- Best model: `{best['order_hypothesis']}` / `{best['feature_set']}`.",
        f"- Baseline coarse-axis LOO R2: `{baseline['loo_r2']:.4f}`; best semantic LOO R2: `{best['loo_r2']:.4f}`.",
        f"- Coarse-axis collision groups: `{coarse_collisions}`; best semantic collision groups: `{best_collisions}`.",
        f"- Best semantic versus coarse deltas: LOO R2 `{loo_delta:+.4f}`, collision groups `{collision_delta:+d}`.",
        f"- Best semantic permutation p(LOO R2 >= observed): `{best_perm['permutation_p_ge_observed']:.4f}` with `{best_perm['n_perm']}` permutations.",
        (
            f"- Best zero-collision code: `{best_zero_collision.get('feature_set', 'none')}` with LOO R2 "
            f"`{_safe_float(best_zero_collision.get('loo_r2')):.4f}`."
            if best_zero_collision
            else "- No tested semantic code eliminated all feature collisions."
        ),
        f"- Best selected-video rich-alignment clip: `{top_clip.get('clip', '')}` with quality-weighted plausibility `{_safe_float(top_clip.get('quality_weighted_neural_plausibility')):.4f}`; old coarse score `{_safe_float(top_clip.get('coarse_quality_weighted_plausibility')):.4f}`.",
        "",
        "## DANDI Class Encoding",
        "",
        "| stimulus | left motion | right motion | semantic class | response |",
        "|---|---|---|---|---:|",
    ]
    for row in sorted(rows_for_order, key=lambda r: float(r["population_response_mean"]), reverse=True):
        lines.append(
            f"| `{row['stimulus']}` | `{row['left_motion']}` | `{row['right_motion']}` | "
            f"`{row['semantic_class']}` | {float(row['population_response_mean']):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Feature-Set Model Scores",
            "",
            "| order | feature set | features | lambda | train R2 | LOO R2 | LOO corr | collisions | worst collision range |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in model_rows:
        lines.append(
            f"| `{row['order_hypothesis']}` | `{row['feature_set']}` | {int(row['n_features'])} | "
            f"{float(row['lambda']):.3g} | {float(row['train_r2']):.4f} | {float(row['loo_r2']):.4f} | "
            f"{float(row['loo_corr']):.4f} | {int(row['collision_group_count'])} | "
            f"{float(row['max_collision_response_range']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Largest Feature Collisions",
            "",
            "| order | feature set | labels | count | response range | top label | bottom label |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in collision_rows[:12]:
        lines.append(
            f"| `{row['order_hypothesis']}` | `{row['feature_set']}` | `{row['labels']}` | "
            f"{int(row['label_count'])} | {float(row['response_range']):.4f} | "
            f"`{row['top_label']}` | `{row['bottom_label']}` |"
        )
    lines.extend(
        [
            "",
            "## Selected-Video Rich Retinal-Code Alignment",
            "",
            "| rank | clip | top stimulus | semantic class | rich score | coarse score | delta | quality | event Hz |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(clip_rows[:12], start=1):
        lines.append(
            f"| {rank} | `{row['clip']}` | `{row['top_stimulus']}` | `{row['top_semantic_class']}` | "
            f"{float(row['quality_weighted_neural_plausibility']):.4f} | "
            f"{float(row['coarse_quality_weighted_plausibility']):.4f} | "
            f"{float(row['rich_minus_coarse_quality_weighted_plausibility']):.4f} | "
            f"{float(row['mean_quality_weight']):.4f} | {float(row['backend_event_frequency_hz']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Supported: DANDI stimulus labels contain eye-specific and lower-posterior retinal semantics that the previous coarse axes discarded.",
            "- Supported: the richer representation gives a more anatomically traceable bridge target and slightly improves selected-video DANDI plausibility relative to the coarse-axis audit.",
            "- Not supported: the tested semantic feature sets do not yet predict DANDI class-level calcium responses with journal-grade held-out performance; all LOO R2 values are negative and the best semantic model is not significant against permutation at conventional thresholds.",
            "- Not supported: eye-motion features alone do not eliminate collisions. A higher-dimensional semantic-plus-axis code can separate all labels, but its LOO R2 remains negative with only 20 DANDI classes, so it is a representation scaffold rather than a validated predictive model.",
            "- Suggestive: selected videos can be re-scored against the current backend's simZFish retinal counters, making the bridge more anatomically traceable than a three-axis projection.",
            "- Missing: the original DANDI projector movies are still absent from the public NWB files, so this cannot prove frame-exact video-to-calcium prediction.",
            "- Missing: the current runtime motor adapter still needs fitting or replacement against DANDI calcium classes plus simZFish/ZBot motor targets before claiming biological fidelity.",
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
    parser.add_argument("--dandi-summary", type=Path, default=DEFAULT_DANDI_SUMMARY)
    parser.add_argument("--video-frames", type=Path, default=DEFAULT_VIDEO_FRAMES)
    parser.add_argument("--backend-summary", type=Path, default=DEFAULT_BACKEND_SUMMARY)
    parser.add_argument("--coarse-alignment", type=Path, default=DEFAULT_COARSE_ALIGNMENT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--permutations", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260603)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dandi_rows = _load_dandi_rows(args.dandi_summary)
    semantic_rows = _all_semantic_rows(dandi_rows)
    model_rows, prediction_rows = _model_audit_rows(semantic_rows)
    collision_rows = _feature_collision_rows(semantic_rows)
    permutation_rows = _permutation_rows(semantic_rows, model_rows, n_perm=args.permutations, seed=args.seed)
    best = model_rows[0]
    best_order = str(best["order_hypothesis"])
    video_rows = _read_csv(args.video_frames)
    rich_frame_rows = _video_alignment_rows(video_rows, dandi_rows, order=best_order)
    clip_rows = _clip_summary_rows(rich_frame_rows, _read_csv(args.backend_summary), _read_csv(args.coarse_alignment))

    semantic_path = out_dir / "dandi_projector_semantic_features.csv"
    model_path = out_dir / "semantic_feature_model_scores.csv"
    prediction_path = out_dir / "semantic_model_predictions.csv"
    collision_path = out_dir / "semantic_feature_collision_groups.csv"
    permutation_path = out_dir / "semantic_model_permutation_scores.csv"
    frame_path = out_dir / "rich_video_frame_neural_alignment.csv"
    clip_path = out_dir / "rich_video_clip_alignment_summary.csv"
    _write_csv(semantic_path, semantic_rows)
    _write_csv(model_path, model_rows)
    _write_csv(prediction_path, prediction_rows)
    _write_csv(collision_path, collision_rows)
    _write_csv(permutation_path, permutation_rows)
    _write_csv(frame_path, rich_frame_rows)
    _write_csv(clip_path, clip_rows)

    plots = [
        _save_retinal_code_plot(out_dir, semantic_rows, order=best_order),
        _save_model_plot(out_dir, model_rows),
        _save_prediction_plot(out_dir, prediction_rows, best),
        _save_embedding_plot(out_dir, semantic_rows, order=best_order),
        _save_clip_rank_plot(out_dir, clip_rows),
        _save_fraction_heatmap(out_dir, rich_frame_rows, dandi_rows),
    ]
    report_path = out_dir / "DANDI_PROJECTOR_SEMANTIC_BRIDGE_AUDIT.md"
    _write_report(
        report_path,
        model_rows=model_rows,
        permutation_rows=permutation_rows,
        collision_rows=collision_rows,
        clip_rows=clip_rows,
        best_order=best_order,
        plots=plots,
        semantic_rows=semantic_rows,
    )
    baseline = next(
        row for row in model_rows if row["order_hypothesis"] == best_order and row["feature_set"] == "coarse_axes"
    )
    manifest = {
        "out_dir": str(out_dir.resolve()),
        "report": str(report_path.resolve()),
        "dandi_summary": str(args.dandi_summary.resolve()),
        "video_frames": str(args.video_frames.resolve()),
        "backend_summary": str(args.backend_summary.resolve()),
        "coarse_alignment": str(args.coarse_alignment.resolve()),
        "stimulus_count": len(dandi_rows),
        "video_frame_count": len(video_rows),
        "clip_count": len(clip_rows),
        "best_order_hypothesis": best_order,
        "best_feature_set": best["feature_set"],
        "best_loo_r2": best["loo_r2"],
        "baseline_coarse_loo_r2": baseline["loo_r2"],
        "best_collision_group_count": int(best["collision_group_count"]),
        "baseline_collision_group_count": int(baseline["collision_group_count"]),
        "best_permutation_p_ge_observed": next(
            row["permutation_p_ge_observed"]
            for row in permutation_rows
            if row["order_hypothesis"] == best["order_hypothesis"] and row["feature_set"] == best["feature_set"]
        ),
        "top_clip": clip_rows[0]["clip"] if clip_rows else "",
        "top_clip_rich_quality_weighted_plausibility": clip_rows[0]["quality_weighted_neural_plausibility"] if clip_rows else 0.0,
        "top_clip_coarse_quality_weighted_plausibility": clip_rows[0]["coarse_quality_weighted_plausibility"] if clip_rows else 0.0,
        "mean_rich_quality_weighted_plausibility": float(
            np.mean([row["quality_weighted_neural_plausibility"] for row in clip_rows])
        )
        if clip_rows
        else 0.0,
        "mean_coarse_quality_weighted_plausibility": float(
            np.mean([row["coarse_quality_weighted_plausibility"] for row in clip_rows])
        )
        if clip_rows
        else 0.0,
        "csv": {
            "dandi_projector_semantic_features": str(semantic_path.resolve()),
            "semantic_feature_model_scores": str(model_path.resolve()),
            "semantic_model_predictions": str(prediction_path.resolve()),
            "semantic_feature_collision_groups": str(collision_path.resolve()),
            "semantic_model_permutation_scores": str(permutation_path.resolve()),
            "rich_video_frame_neural_alignment": str(frame_path.resolve()),
            "rich_video_clip_alignment_summary": str(clip_path.resolve()),
        },
        "plots": [str(path.resolve()) for path in plots],
        "limitations": [
            "DANDI NWB exposes stimulus labels, not original projector movies.",
            "Ambiguous two-token labels are evaluated under two eye-order hypotheses.",
            "Selected-video matching uses current backend retinal counters, not calibrated camera/projector geometry.",
            "This audit does not fit or replace the live motor adapter.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
