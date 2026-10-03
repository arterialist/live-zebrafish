"""Comprehensive live activity study for zebrafish action pipelines.

The runner drives the lab backend through the same REST endpoints used by the
UI and records the WebSocket stream consumed by the body/connectome viewers.
It intentionally keeps all outputs under ``analysis/out`` so a run can be
repeated without changing the simulation code.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PHYSICS_TIMESTEP_S = 0.005
MM_SCALE = 1e6
JOINT_SCALE = 1e4
MUSCLE_SCALE = 1e4
NEURAL_SCALE = 1e4
TOUCH_SCALE = 1e6

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "out" / "comprehensive_activity_study"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _request_json(
    rest_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    url = rest_url.rstrip("/") + path
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, default=_json_default).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {detail}") from exc


def _local_sample_video_duration_s(file_name: str) -> float:
    path = SCRIPT_DIR / "cache" / "video_stimuli" / "clips" / Path(file_name).name
    if not path.exists():
        return 0.0
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            return frames / fps if fps > 0.0 else 0.0
        finally:
            cap.release()
    except Exception:  # noqa: BLE001
        return 0.0


def _scaled(values: Any, scale: float) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(values, dtype=np.float64) / scale


def _segments_from_wire(raw: Any) -> np.ndarray:
    arr = np.asarray(raw or [], dtype=np.float64)
    if arr.size < 3:
        return np.zeros((0, 3), dtype=np.float64)
    return arr.reshape((-1, 3)) / MM_SCALE


def _decode_bits(b64: str, n: int) -> np.ndarray:
    if not b64 or n <= 0:
        return np.zeros(n, dtype=np.float64)
    raw = base64.b64decode(b64)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out[i] = 1.0 if raw[i >> 3] & (1 << (i & 7)) else 0.0
    return out


def _stats(values: Iterable[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _pad_stack(arrays: list[np.ndarray], *, fill: float = np.nan) -> np.ndarray:
    if not arrays:
        return np.zeros((0, 0), dtype=np.float64)
    width = max((int(a.size) for a in arrays), default=0)
    out = np.full((len(arrays), width), fill, dtype=np.float64)
    for i, arr in enumerate(arrays):
        flat = np.asarray(arr, dtype=np.float64).ravel()
        if flat.size:
            out[i, : flat.size] = flat
    return out


def _body_path_metrics(points: np.ndarray) -> dict[str, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return {}
    diffs = np.diff(points, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    path = float(np.sum(lengths))
    chord = float(np.linalg.norm(points[0] - points[-1]))
    xy_chord = float(np.linalg.norm((points[0] - points[-1])[:2]))
    tangents = diffs / np.maximum(lengths[:, None], 1e-12)
    local3: list[float] = []
    local2: list[float] = []
    for a, b in zip(tangents[:-1], tangents[1:]):
        local3.append(float(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))))
        a2 = a[:2] / max(1e-12, float(np.linalg.norm(a[:2])))
        b2 = b[:2] / max(1e-12, float(np.linalg.norm(b[:2])))
        local2.append(float(math.atan2(a2[0] * b2[1] - a2[1] * b2[0], np.dot(a2, b2))))
    local3_arr = np.asarray(local3, dtype=np.float64)
    local2_arr = np.asarray(local2, dtype=np.float64)
    return {
        "body_path_mm": path,
        "body_chord_mm": chord,
        "body_xy_chord_mm": xy_chord,
        "body_straightness": chord / path if path > 1e-12 else 0.0,
        "body_abs_curvature_2d_rad": float(np.sum(np.abs(local2_arr))) if local2_arr.size else 0.0,
        "body_signed_curvature_2d_rad": float(np.sum(local2_arr)) if local2_arr.size else 0.0,
        "body_max_local_bend_2d_rad": float(np.max(np.abs(local2_arr))) if local2_arr.size else 0.0,
        "body_abs_curvature_3d_rad": float(np.sum(local3_arr)) if local3_arr.size else 0.0,
        "body_max_local_bend_3d_rad": float(np.max(local3_arr)) if local3_arr.size else 0.0,
        "body_z_span_mm": float(np.ptp(points[:, 2])),
    }


def _matrix_groups(values: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        z = np.zeros(0, dtype=np.float64)
        return {"left": z, "right": z, "dorsal": z, "ventral": z}
    lower = [n.lower() for n in names]
    groups: dict[str, list[int]] = {"left": [], "right": [], "dorsal": [], "ventral": []}
    for i, name in enumerate(lower[: arr.size]):
        if "left" in name or name.endswith("_l") or "_l_" in name:
            groups["left"].append(i)
        if "right" in name or name.endswith("_r") or "_r_" in name:
            groups["right"].append(i)
        if "dorsal" in name or "_d_" in name:
            groups["dorsal"].append(i)
        if "ventral" in name or "_v_" in name:
            groups["ventral"].append(i)
    if not any(groups.values()) and arr.size >= 4:
        trimmed = arr[: (arr.size // 4) * 4].reshape((-1, 4))
        return {
            "left": trimmed[:, 0],
            "right": trimmed[:, 1],
            "dorsal": trimmed[:, 2],
            "ventral": trimmed[:, 3],
        }
    return {key: arr[idxs] if idxs else np.zeros(0, dtype=np.float64) for key, idxs in groups.items()}


@dataclass
class WsCapture:
    label: str
    neuron_names: list[str] = field(default_factory=list)
    neuron_meta: list[dict[str, Any]] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    muscle_names: list[str] = field(default_factory=list)
    touch_names: list[str] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)
    wall_start_s: float = field(default_factory=time.perf_counter)
    error: str = ""


def _frame_from_wire(payload: dict[str, Any], capture: WsCapture) -> dict[str, Any]:
    n_neurons = len(capture.neuron_names)
    frame = {
        "tick": int(payload.get("k", 0)),
        "wall_s": time.perf_counter() - capture.wall_start_s,
        "com_mm": np.asarray(payload.get("cm", [0, 0, 0]), dtype=np.float64) / MM_SCALE,
        "segments_mm": _segments_from_wire(payload.get("sm")),
        "heading_rad": float(payload.get("hd", 0.0)),
        "pitch_rad": float(payload.get("pt", 0.0)),
        "tail_yaw_rad": _scaled(payload.get("ta"), JOINT_SCALE),
        "tail_pitch_rad": _scaled(payload.get("tpa"), JOINT_SCALE),
        "joint_angles_rad": _scaled(payload.get("ja"), JOINT_SCALE),
        "joint_velocities_rad_s": _scaled(payload.get("jv"), JOINT_SCALE),
        "touch_forces": _scaled(payload.get("tc"), TOUCH_SCALE),
        "muscle_activations": _scaled(payload.get("ma"), MUSCLE_SCALE),
        "neuron_s": _scaled(payload.get("Si"), NEURAL_SCALE),
        "neuron_r": _scaled(payload.get("Ri"), NEURAL_SCALE),
        "neuron_b": _scaled(payload.get("Bi"), NEURAL_SCALE),
        "neuron_tref": _scaled(payload.get("Trefi"), NEURAL_SCALE),
        "neuron_fired": _decode_bits(str(payload.get("Fb", "")), n_neurons),
        "neuron_m0": _scaled(payload.get("M0i"), NEURAL_SCALE),
        "neuron_m1": _scaled(payload.get("M1i"), NEURAL_SCALE),
        "neuromod": np.asarray(payload.get("nm01", [0.0, 0.0]), dtype=np.float64),
        "free_energy": float(payload.get("fe", 0.0)),
    }
    return frame


async def capture_ws_until(
    ws_url: str,
    *,
    label: str,
    stop_event: asyncio.Event | None = None,
    sim_seconds: float | None = None,
    max_wall_s: float,
    settle_wall_s: float = 0.25,
) -> WsCapture:
    import websockets

    cap = WsCapture(label=label)
    first_tick: int | None = None
    start = time.perf_counter()
    try:
        async with websockets.connect(ws_url, open_timeout=10.0, ping_interval=20.0) as ws:
            while True:
                now = time.perf_counter()
                if now - start > max_wall_s:
                    break
                if stop_event is not None and stop_event.is_set() and now - start >= settle_wall_s:
                    break
                if sim_seconds is not None and first_tick is not None and cap.frames:
                    tick_span = cap.frames[-1]["tick"] - first_tick
                    if tick_span * PHYSICS_TIMESTEP_S >= sim_seconds:
                        break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                payload = json.loads(raw)
                if payload.get("t") == "h":
                    layout = payload.get("L") or {}
                    cap.neuron_names = list(layout.get("nm") or [])
                    cap.neuron_meta = list(payload.get("M") or [])
                    body = payload.get("L_body") or {}
                    cap.joint_names = list(body.get("joints") or [])
                    cap.muscle_names = list(body.get("muscles") or [])
                    cap.touch_names = list(body.get("touch") or [])
                    continue
                if payload.get("t") != "s":
                    continue
                frame = _frame_from_wire(payload, cap)
                if first_tick is None:
                    first_tick = int(frame["tick"])
                cap.frames.append(frame)
    except Exception as exc:  # noqa: BLE001
        cap.error = repr(exc)
    return cap


async def poll_endpoint(
    rest_url: str,
    path: str,
    *,
    interval_s: float,
    stop_event: asyncio.Event | None = None,
    max_wall_s: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    start = time.perf_counter()
    while time.perf_counter() - start < max_wall_s:
        if stop_event is not None and stop_event.is_set():
            break
        try:
            row = _request_json(rest_url, path, timeout_s=10.0)
            row["_wall_s"] = time.perf_counter() - start
            out.append(row)
        except Exception as exc:  # noqa: BLE001
            out.append({"_wall_s": time.perf_counter() - start, "error": repr(exc)})
        await asyncio.sleep(max(0.02, float(interval_s)))
    return out


def _rows_from_capture(cap: WsCapture) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not cap.frames:
        return rows
    tick0 = int(cap.frames[0]["tick"])
    com0 = np.asarray(cap.frames[0]["com_mm"], dtype=np.float64)
    prev: dict[str, Any] | None = None
    for frame in cap.frames:
        tick = int(frame["tick"])
        com = np.asarray(frame["com_mm"], dtype=np.float64)
        row: dict[str, float] = {
            "tick": float(tick),
            "sim_time_s": float((tick - tick0) * PHYSICS_TIMESTEP_S),
            "wall_s": float(frame["wall_s"]),
            "com_x_mm": float(com[0]) if com.size >= 1 else 0.0,
            "com_y_mm": float(com[1]) if com.size >= 2 else 0.0,
            "com_z_mm": float(com[2]) if com.size >= 3 else 0.0,
            "xy_displacement_mm": float(np.linalg.norm(com[:2] - com0[:2])) if com.size >= 2 else 0.0,
            "xyz_displacement_mm": float(np.linalg.norm(com - com0)) if com.size >= 3 else 0.0,
            "heading_rad": float(frame["heading_rad"]),
            "pitch_rad": float(frame["pitch_rad"]),
            "free_energy": float(frame["free_energy"]),
        }
        if prev is not None:
            prev_tick = int(prev["tick"])
            dt = max(PHYSICS_TIMESTEP_S, (tick - prev_tick) * PHYSICS_TIMESTEP_S)
            prev_com = np.asarray(prev["com_mm"], dtype=np.float64)
            delta = com - prev_com
            row["speed_xy_mm_s"] = float(np.linalg.norm(delta[:2]) / dt)
            row["speed_3d_mm_s"] = float(np.linalg.norm(delta) / dt)
            row["vertical_speed_mm_s"] = float(delta[2] / dt) if delta.size >= 3 else 0.0
            row["vertical_speed_abs_mm_s"] = abs(row["vertical_speed_mm_s"])
            row["heading_rate_rad_s"] = float(np.angle(np.exp(1j * (row["heading_rad"] - float(prev["heading_rad"])))) / dt)
            row["pitch_rate_rad_s"] = float((row["pitch_rad"] - float(prev["pitch_rad"])) / dt)
        else:
            row.update(
                {
                    "speed_xy_mm_s": 0.0,
                    "speed_3d_mm_s": 0.0,
                    "vertical_speed_mm_s": 0.0,
                    "vertical_speed_abs_mm_s": 0.0,
                    "heading_rate_rad_s": 0.0,
                    "pitch_rate_rad_s": 0.0,
                }
            )
        for prefix, arr_name in (
            ("tail_yaw", "tail_yaw_rad"),
            ("tail_pitch", "tail_pitch_rad"),
            ("joint_angle", "joint_angles_rad"),
            ("joint_velocity", "joint_velocities_rad_s"),
            ("touch", "touch_forces"),
            ("muscle", "muscle_activations"),
            ("neuron_s", "neuron_s"),
            ("neuron_r", "neuron_r"),
            ("neuron_b", "neuron_b"),
            ("neuron_tref", "neuron_tref"),
            ("neuron_m0", "neuron_m0"),
            ("neuron_m1", "neuron_m1"),
        ):
            arr = np.asarray(frame[arr_name], dtype=np.float64)
            if arr.size:
                row[f"{prefix}_mean"] = float(np.mean(arr))
                row[f"{prefix}_abs_mean"] = float(np.mean(np.abs(arr)))
                row[f"{prefix}_rms"] = float(np.sqrt(np.mean(arr * arr)))
                row[f"{prefix}_max"] = float(np.max(arr))
                row[f"{prefix}_abs_max"] = float(np.max(np.abs(arr)))
                row[f"{prefix}_sum"] = float(np.sum(arr))
            else:
                row[f"{prefix}_mean"] = 0.0
                row[f"{prefix}_abs_mean"] = 0.0
                row[f"{prefix}_rms"] = 0.0
                row[f"{prefix}_max"] = 0.0
                row[f"{prefix}_abs_max"] = 0.0
                row[f"{prefix}_sum"] = 0.0
        fired = np.asarray(frame["neuron_fired"], dtype=np.float64)
        row["neuron_fired_count"] = float(np.sum(fired)) if fired.size else 0.0
        row["neuron_fired_fraction"] = float(np.mean(fired)) if fired.size else 0.0
        neuromod = np.asarray(frame["neuromod"], dtype=np.float64)
        row["neuromod_m0"] = float(neuromod[0]) if neuromod.size >= 1 else 0.0
        row["neuromod_m1"] = float(neuromod[1]) if neuromod.size >= 2 else 0.0
        groups = _matrix_groups(np.asarray(frame["muscle_activations"], dtype=np.float64), cap.muscle_names)
        for key, arr in groups.items():
            row[f"muscle_{key}_mean"] = float(np.mean(arr)) if arr.size else 0.0
        left = row["muscle_left_mean"]
        right = row["muscle_right_mean"]
        dorsal = row["muscle_dorsal_mean"]
        ventral = row["muscle_ventral_mean"]
        row["muscle_lr_bias"] = float((right - left) / max(1e-9, right + left))
        row["muscle_dv_bias"] = float((dorsal - ventral) / max(1e-9, dorsal + ventral))
        row.update(_body_path_metrics(np.asarray(frame["segments_mm"], dtype=np.float64)))
        rows.append(row)
        prev = frame
    return rows


def _series_matrix(cap: WsCapture, key: str) -> np.ndarray:
    return _pad_stack([np.asarray(f[key], dtype=np.float64) for f in cap.frames])


def _write_rows_csv(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")


def _downsample_matrix(mat: np.ndarray, max_rows: int = 80, max_cols: int = 900) -> np.ndarray:
    out = np.asarray(mat, dtype=np.float64)
    if out.ndim != 2 or out.size == 0:
        return out
    if out.shape[0] > max_cols:
        idx = np.linspace(0, out.shape[0] - 1, max_cols).astype(int)
        out = out[idx]
    if out.shape[1] > max_rows:
        score = np.nanstd(out, axis=0) + 0.1 * np.nanmean(np.abs(out), axis=0)
        cols = np.argsort(-score)[:max_rows]
        out = out[:, cols]
    return out


def _metric_arrays(rows: list[dict[str, float]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not rows:
        return np.zeros(0, dtype=np.float64), {}
    x = np.asarray([r.get("sim_time_s", 0.0) for r in rows], dtype=np.float64)
    fields = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"tick", "sim_time_s", "wall_s"} and isinstance(value, (int, float, np.floating))
        }
    )
    arrays = {
        field: np.asarray([float(row.get(field, 0.0)) for row in rows], dtype=np.float64)
        for field in fields
    }
    return x, arrays


def _rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or samples <= 1:
        return arr
    samples = min(int(samples), arr.size)
    kernel = np.ones(samples, dtype=np.float64)
    valid = np.isfinite(arr).astype(np.float64)
    filled = np.where(np.isfinite(arr), arr, 0.0)
    numerator = np.convolve(filled, kernel, mode="same")
    denominator = np.maximum(1.0, np.convolve(valid, kernel, mode="same"))
    return numerator / denominator


def _median_dt_s(x: np.ndarray) -> float:
    if x.size < 3:
        return PHYSICS_TIMESTEP_S
    diffs = np.diff(x)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return PHYSICS_TIMESTEP_S
    return float(np.median(diffs))


def _plot_overview(run_dir: Path, label: str, rows: list[dict[str, float]], actions: list[dict[str, Any]]) -> None:
    if not rows:
        return
    x = np.asarray([r["sim_time_s"] for r in rows], dtype=np.float64)
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    axes[0].plot(x, [r["speed_xy_mm_s"] for r in rows], label="xy speed")
    axes[0].plot(x, [r["speed_3d_mm_s"] for r in rows], label="3D speed", alpha=0.8)
    axes[0].set_ylabel("mm/s")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"{label}: activity overview")

    axes[1].plot(x, [r["com_z_mm"] for r in rows], label="COM z")
    axes[1].plot(x, [r["pitch_rad"] for r in rows], label="pitch rad")
    axes[1].plot(x, [r["heading_rate_rad_s"] for r in rows], label="heading rate", alpha=0.7)
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("depth/orient")

    axes[2].plot(x, [r["tail_yaw_abs_mean"] for r in rows], label="tail yaw mean abs")
    axes[2].plot(x, [r["tail_yaw_abs_max"] for r in rows], label="tail yaw max abs", alpha=0.8)
    axes[2].plot(x, [r["tail_pitch_abs_mean"] for r in rows], label="tail pitch mean abs", alpha=0.7)
    axes[2].legend(loc="upper right")
    axes[2].set_ylabel("rad")

    axes[3].plot(x, [r["muscle_sum"] for r in rows], label="muscle sum")
    axes[3].plot(x, [r["muscle_lr_bias"] for r in rows], label="L/R bias")
    axes[3].plot(x, [r["muscle_dv_bias"] for r in rows], label="D/V bias")
    axes[3].legend(loc="upper right")
    axes[3].set_ylabel("motor")

    axes[4].plot(x, [r["neuron_s_mean"] for r in rows], label="mean S")
    axes[4].plot(x, [r["neuron_fired_count"] for r in rows], label="fired count")
    axes[4].plot(x, [r["neuron_m1_mean"] for r in rows], label="mean M1")
    axes[4].legend(loc="upper right")
    axes[4].set_ylabel("PAULA")

    axes[5].plot(x, [r["free_energy"] for r in rows], label="free energy")
    axes[5].plot(x, [r["touch_sum"] for r in rows], label="touch/contact")
    axes[5].legend(loc="upper right")
    axes[5].set_ylabel("FE/contact")
    axes[5].set_xlabel("simulation time (s)")

    if actions:
        ax2 = axes[2].twinx()
        ax = np.asarray(
            [
                float(
                    a.get(
                        "sim_time_s",
                        a.get("_sim_time_s", a.get("video_time_s", a.get("calcium_time_s", 0.0))),
                    )
                )
                for a in actions
            ]
        )
        force = np.asarray([float(a.get("action_force", a.get("force", 0.0)) or 0.0) for a in actions])
        side = np.asarray([float(a.get("action_side_score", a.get("side_score", 0.0)) or 0.0) for a in actions])
        ax2.plot(ax, force, color="tab:red", alpha=0.35, label="action force")
        ax2.plot(ax, side, color="tab:purple", alpha=0.30, label="action side")
        ax2.set_ylabel("action")
    fig.tight_layout()
    fig.savefig(run_dir / "overview_timeseries.png", dpi=170)
    plt.close(fig)


def _plot_all_scalar_metrics(run_dir: Path, label: str, rows: list[dict[str, float]]) -> str:
    x, arrays = _metric_arrays(rows)
    page_dir = run_dir / "scalar_metric_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    if x.size == 0 or not arrays:
        return str(page_dir)
    dt = _median_dt_s(x)
    smooth_short = max(3, int(round(5.0 / max(dt, 1e-6))))
    smooth_long = max(smooth_short + 1, int(round(30.0 / max(dt, 1e-6))))
    fields = list(arrays)
    per_page = 12
    for page, start in enumerate(range(0, len(fields), per_page), start=1):
        subset = fields[start : start + per_page]
        fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
        for ax, field in zip(axes.ravel(), subset):
            y = arrays[field]
            ax.plot(x, y, color="0.45", linewidth=0.5, alpha=0.45, label="raw")
            ax.plot(x, _rolling_mean(y, smooth_short), linewidth=1.0, label="5 s")
            ax.plot(x, _rolling_mean(y, smooth_long), linewidth=1.4, label="30 s")
            ax.set_title(field, fontsize=9)
            ax.grid(alpha=0.18)
        for ax in axes.ravel()[len(subset) :]:
            ax.axis("off")
        axes.ravel()[0].legend(loc="upper right", fontsize=7)
        fig.suptitle(f"{label}: every scalar metric, page {page}", y=0.995)
        fig.supxlabel("simulation time (s)")
        fig.tight_layout()
        fig.savefig(page_dir / f"scalar_metrics_page_{page:02d}.png", dpi=160)
        plt.close(fig)
    return str(page_dir)


def _plot_scalar_distributions(run_dir: Path, label: str, rows: list[dict[str, float]]) -> str:
    _, arrays = _metric_arrays(rows)
    page_dir = run_dir / "scalar_distribution_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    fields = list(arrays)
    per_page = 16
    for page, start in enumerate(range(0, len(fields), per_page), start=1):
        subset = fields[start : start + per_page]
        fig, axes = plt.subplots(4, 4, figsize=(18, 14))
        for ax, field in zip(axes.ravel(), subset):
            y = arrays[field]
            y = y[np.isfinite(y)]
            if y.size:
                ax.hist(y, bins=48, color="tab:blue", alpha=0.78)
                ax.axvline(float(np.mean(y)), color="white", linewidth=1.0, alpha=0.8)
            ax.set_title(field, fontsize=8)
            ax.grid(alpha=0.12)
        for ax in axes.ravel()[len(subset) :]:
            ax.axis("off")
        fig.suptitle(f"{label}: scalar metric distributions, page {page}", y=0.995)
        fig.tight_layout()
        fig.savefig(page_dir / f"scalar_distributions_page_{page:02d}.png", dpi=160)
        plt.close(fig)
    return str(page_dir)


def _plot_correlation_matrix(run_dir: Path, label: str, rows: list[dict[str, float]]) -> str:
    _, arrays = _metric_arrays(rows)
    fields = [field for field, arr in arrays.items() if np.nanstd(arr) > 1e-12]
    path = run_dir / "scalar_correlation_matrix.png"
    if len(fields) < 2:
        return str(path)
    # Keep every non-constant metric, but order by variance so related dynamic
    # channels remain visible even when the matrix is large.
    fields = sorted(fields, key=lambda f: float(np.nanstd(arrays[f])), reverse=True)
    data = np.vstack([arrays[field] for field in fields])
    corr = np.corrcoef(data)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    fig_size = max(12, min(28, 0.26 * len(fields)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    ax.set_title(f"{label}: all non-constant scalar metric correlations")
    ax.set_xticks(np.arange(len(fields)))
    ax.set_yticks(np.arange(len(fields)))
    ax.set_xticklabels(fields, rotation=90, fontsize=5)
    ax.set_yticklabels(fields, fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def _plot_trajectory(run_dir: Path, label: str, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    x = np.asarray([r["com_x_mm"] for r in rows])
    y = np.asarray([r["com_y_mm"] for r in rows])
    z = np.asarray([r["com_z_mm"] for r in rows])
    t = np.asarray([r["sim_time_s"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc = axes[0].scatter(x, y, c=t, s=8, cmap="viridis")
    axes[0].plot(x, y, color="white", alpha=0.18, linewidth=0.8)
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_title(f"{label}: XY trajectory")
    axes[0].set_xlabel("x mm")
    axes[0].set_ylabel("y mm")
    fig.colorbar(sc, ax=axes[0], label="sim time s")
    axes[1].plot(t, z)
    axes[1].set_title("vertical activity")
    axes[1].set_xlabel("sim time s")
    axes[1].set_ylabel("z mm")
    fig.tight_layout()
    fig.savefig(run_dir / "body_trajectory.png", dpi=170)
    plt.close(fig)


def _plot_body_shape_snapshots(run_dir: Path, label: str, cap: WsCapture) -> str:
    path = run_dir / "body_shape_snapshots.png"
    if not cap.frames:
        return str(path)
    idxs = np.linspace(0, len(cap.frames) - 1, min(18, len(cap.frames))).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.get_cmap("viridis")
    tick0 = int(cap.frames[0]["tick"])
    for j, idx in enumerate(idxs):
        frame = cap.frames[int(idx)]
        pts = np.asarray(frame["segments_mm"], dtype=np.float64)
        if pts.size == 0:
            continue
        t = (int(frame["tick"]) - tick0) * PHYSICS_TIMESTEP_S
        color = cmap(j / max(1, len(idxs) - 1))
        axes[0].plot(pts[:, 0], pts[:, 1], color=color, alpha=0.75, linewidth=1.1)
        axes[0].scatter(pts[0, 0], pts[0, 1], color=color, s=18)
        axes[1].plot(pts[:, 0], pts[:, 2], color=color, alpha=0.75, linewidth=1.1, label=f"{t:.0f}s")
        axes[1].scatter(pts[0, 0], pts[0, 2], color=color, s=18)
    axes[0].set_title("top view body shape snapshots")
    axes[0].set_xlabel("x mm")
    axes[0].set_ylabel("y mm")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set_title("side/depth body shape snapshots")
    axes[1].set_xlabel("x mm")
    axes[1].set_ylabel("z mm")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def _plot_heatmaps(run_dir: Path, cap: WsCapture) -> None:
    if not cap.frames:
        return
    mats = {
        "tail_yaw_heatmap.png": (_series_matrix(cap, "tail_yaw_rad"), "Tail yaw rad", "tail segment"),
        "tail_pitch_heatmap.png": (_series_matrix(cap, "tail_pitch_rad"), "Tail pitch rad", "tail segment"),
        "muscle_activation_heatmap.png": (_series_matrix(cap, "muscle_activations"), "Muscle activation", "muscle"),
        "neural_s_heatmap.png": (_series_matrix(cap, "neuron_s"), "PAULA S", "top neurons"),
        "neural_m1_heatmap.png": (_series_matrix(cap, "neuron_m1"), "PAULA M1", "top neurons"),
    }
    for filename, (matrix, title, ylabel) in mats.items():
        matrix = _downsample_matrix(matrix)
        if matrix.size == 0:
            continue
        t = np.asarray([(f["tick"] - cap.frames[0]["tick"]) * PHYSICS_TIMESTEP_S for f in cap.frames])
        if matrix.shape[0] != t.size:
            t = np.arange(matrix.shape[0], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(14, 5))
        extent = [float(t[0]), float(t[-1]) if t.size > 1 else 0.0, 0, matrix.shape[1]]
        im = ax.imshow(
            matrix.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=extent,
            cmap="magma",
        )
        ax.set_title(title)
        ax.set_xlabel("simulation time (s)")
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=170)
        plt.close(fig)


def _plot_all_matrix_heatmaps(run_dir: Path, cap: WsCapture) -> str:
    page_dir = run_dir / "matrix_heatmaps"
    page_dir.mkdir(parents=True, exist_ok=True)
    if not cap.frames:
        return str(page_dir)
    matrices = {
        "tail_yaw_rad": (_series_matrix(cap, "tail_yaw_rad"), "Tail yaw rad", "segment"),
        "tail_pitch_rad": (_series_matrix(cap, "tail_pitch_rad"), "Tail pitch rad", "segment"),
        "joint_angles_rad": (_series_matrix(cap, "joint_angles_rad"), "Joint angle rad", "joint"),
        "joint_velocities_rad_s": (_series_matrix(cap, "joint_velocities_rad_s"), "Joint velocity rad/s", "joint"),
        "touch_forces": (_series_matrix(cap, "touch_forces"), "Touch/contact force", "sensor"),
        "muscle_activations": (_series_matrix(cap, "muscle_activations"), "Muscle activation", "muscle"),
        "neuron_s": (_series_matrix(cap, "neuron_s"), "PAULA S", "neuron"),
        "neuron_r": (_series_matrix(cap, "neuron_r"), "PAULA r", "neuron"),
        "neuron_b": (_series_matrix(cap, "neuron_b"), "PAULA b", "neuron"),
        "neuron_tref": (_series_matrix(cap, "neuron_tref"), "PAULA t_ref", "neuron"),
        "neuron_fired": (_series_matrix(cap, "neuron_fired"), "PAULA fired bits", "neuron"),
        "neuron_m0": (_series_matrix(cap, "neuron_m0"), "PAULA M0 vector", "neuron"),
        "neuron_m1": (_series_matrix(cap, "neuron_m1"), "PAULA M1 vector", "neuron"),
    }
    t = np.asarray([(f["tick"] - cap.frames[0]["tick"]) * PHYSICS_TIMESTEP_S for f in cap.frames])
    for key, (matrix, title, ylabel) in matrices.items():
        matrix = _downsample_matrix(matrix, max_rows=120, max_cols=1400)
        if matrix.size == 0:
            continue
        tx = t
        if matrix.shape[0] != tx.size:
            tx = np.linspace(float(t[0]), float(t[-1]) if t.size > 1 else 0.0, matrix.shape[0])
        fig, ax = plt.subplots(figsize=(16, 5.5))
        extent = [float(tx[0]), float(tx[-1]) if tx.size > 1 else 0.0, 0, matrix.shape[1]]
        im = ax.imshow(matrix.T, aspect="auto", origin="lower", interpolation="nearest", extent=extent, cmap="magma")
        ax.set_title(f"{title} over full run")
        ax.set_xlabel("simulation time (s)")
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(page_dir / f"{key}_heatmap.png", dpi=165)
        plt.close(fig)
    return str(page_dir)


def _plot_frequency_analysis(run_dir: Path, label: str, rows: list[dict[str, float]]) -> str:
    path = run_dir / "frequency_analysis.png"
    x, arrays = _metric_arrays(rows)
    if x.size < 16:
        return str(path)
    fields = [
        "speed_xy_mm_s",
        "tail_yaw_abs_mean",
        "tail_yaw_abs_max",
        "muscle_sum",
        "muscle_lr_bias",
        "neuron_s_mean",
        "neuron_fired_count",
        "free_energy",
    ]
    fields = [field for field in fields if field in arrays and np.nanstd(arrays[field]) > 1e-12]
    if not fields:
        return str(path)
    dt = _median_dt_s(x)
    grid = np.arange(float(x[0]), float(x[-1]), dt)
    if grid.size < 16:
        return str(path)
    fig, axes = plt.subplots(len(fields), 1, figsize=(14, max(8, 2.2 * len(fields))), sharex=True)
    if len(fields) == 1:
        axes = [axes]
    for ax, field in zip(axes, fields):
        y = arrays[field]
        y = np.interp(grid, x, np.nan_to_num(y, nan=float(np.nanmean(y))))
        y = y - float(np.mean(y))
        window = np.hanning(y.size)
        spec = np.abs(np.fft.rfft(y * window)) ** 2
        freq = np.fft.rfftfreq(y.size, d=dt)
        keep = (freq > 0.0) & (freq <= 20.0)
        ax.semilogy(freq[keep], spec[keep] + 1e-18)
        ax.set_ylabel(field)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("frequency (Hz), interpolated from WS samples")
    fig.suptitle(f"{label}: frequency content of key activity metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def _action_time(row: dict[str, Any], fallback: int) -> float:
    for key in ("sim_time_s", "_sim_time_s", "video_time_s", "calcium_time_s"):
        try:
            value = row.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return float(fallback)


def _action_value(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            value = row.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def _plot_action_alignment(run_dir: Path, label: str, rows: list[dict[str, float]], actions: list[dict[str, Any]]) -> str:
    path = run_dir / "action_body_alignment.png"
    if not rows or not actions:
        return str(path)
    x = np.asarray([r["sim_time_s"] for r in rows], dtype=np.float64)
    action_t = np.asarray([_action_time(a, i) for i, a in enumerate(actions)], dtype=np.float64)
    force = np.asarray([_action_value(a, "action_force", "force") for a in actions], dtype=np.float64)
    side = np.asarray([_action_value(a, "action_side_score", "side_score") for a in actions], dtype=np.float64)
    confidence = np.asarray([_action_value(a, "action_confidence", "confidence") for a in actions], dtype=np.float64)
    kick = np.asarray([_action_value(a, "action_kick", "kick") for a in actions], dtype=np.float64)
    if action_t.size < 2:
        return str(path)
    force_i = np.interp(x, action_t, force)
    side_i = np.interp(x, action_t, side)
    confidence_i = np.interp(x, action_t, confidence)
    kick_i = np.interp(x, action_t, kick)
    tail = np.asarray([r.get("tail_yaw_abs_mean", 0.0) for r in rows], dtype=np.float64)
    lr = np.asarray([r.get("muscle_lr_bias", 0.0) for r in rows], dtype=np.float64)
    speed = np.asarray([r.get("speed_xy_mm_s", 0.0) for r in rows], dtype=np.float64)
    heading_rate = np.asarray([r.get("heading_rate_rad_s", 0.0) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(4, 2, figsize=(16, 15))
    axes[0, 0].plot(x, force_i, label="action force")
    axes[0, 0].plot(x, tail, label="tail yaw abs mean")
    axes[0, 0].plot(x, speed, label="xy speed", alpha=0.7)
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].set_title("force vs realized body motion")

    axes[1, 0].plot(x, side_i, label="action side score")
    axes[1, 0].plot(x, lr, label="muscle L/R bias")
    axes[1, 0].plot(x, heading_rate, label="heading rate", alpha=0.7)
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].set_title("side command vs motor/turning")

    axes[2, 0].plot(x, confidence_i, label="confidence")
    axes[2, 0].plot(x, kick_i, label="kick")
    axes[2, 0].legend(loc="upper right")
    axes[2, 0].set_title("decoder confidence and kick")

    axes[3, 0].plot(x, _rolling_mean(speed, max(3, int(round(10.0 / max(_median_dt_s(x), 1e-6))))), label="10s speed")
    axes[3, 0].plot(x, _rolling_mean(tail, max(3, int(round(10.0 / max(_median_dt_s(x), 1e-6))))), label="10s tail")
    axes[3, 0].legend(loc="upper right")
    axes[3, 0].set_title("long-timescale realized activity")

    axes[0, 1].scatter(force_i, tail, s=5, alpha=0.25)
    axes[0, 1].set_xlabel("action force")
    axes[0, 1].set_ylabel("tail yaw abs mean")
    axes[1, 1].scatter(side_i, lr, s=5, alpha=0.25)
    axes[1, 1].set_xlabel("action side score")
    axes[1, 1].set_ylabel("muscle L/R bias")
    axes[2, 1].scatter(force_i, speed, s=5, alpha=0.25)
    axes[2, 1].set_xlabel("action force")
    axes[2, 1].set_ylabel("xy speed")
    axes[3, 1].scatter(confidence_i, tail, s=5, alpha=0.25)
    axes[3, 1].set_xlabel("confidence")
    axes[3, 1].set_ylabel("tail yaw abs mean")
    for ax in axes.ravel():
        ax.grid(alpha=0.16)
    fig.suptitle(f"{label}: decoded action to body alignment")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def _plot_video_diagnostics(run_dir: Path, actions: list[dict[str, Any]]) -> None:
    if not actions:
        return
    x = np.asarray([float(a.get("driver_video_time_s", a.get("video_time_s", i))) for i, a in enumerate(actions)])
    fields = [
        ("motion_energy", "motion"),
        ("startle", "startle"),
        ("asymmetry", "asymmetry"),
        ("flow_reliability", "flow reliability"),
        ("camera_shake", "camera shake"),
        ("zapbench_distance", "ZAPBench NN distance"),
        ("simzfish_omr_drive", "SimZFish OMR drive"),
        ("simzfish_turn_bias", "SimZFish turn bias"),
    ]
    fig, axes = plt.subplots(len(fields), 1, figsize=(14, 18), sharex=True)
    for ax, (field_name, label) in zip(axes, fields):
        ax.plot(x, [float(a.get(field_name, 0.0) or 0.0) for a in actions], label=label)
        ax.legend(loc="upper right")
        ax.set_ylabel(label)
    axes[-1].set_xlabel("video time (s)")
    fig.tight_layout()
    fig.savefig(run_dir / "video_extractor_diagnostics.png", dpi=170)
    plt.close(fig)


def _plot_calcium_actions(run_dir: Path, actions: list[dict[str, Any]]) -> None:
    if not actions:
        return
    x = np.asarray([float(a.get("_sim_time_s", a.get("calcium_time_s", i))) for i, a in enumerate(actions)])
    fields = [
        ("row", "ZAPBench row"),
        ("force", "force"),
        ("kick", "kick"),
        ("side_score", "side score"),
        ("kick_score", "kick score"),
        ("confidence", "confidence"),
    ]
    fig, axes = plt.subplots(len(fields), 1, figsize=(14, 14), sharex=True)
    for ax, (field_name, label) in zip(axes, fields):
        ax.plot(x, [float(a.get(field_name, 0.0) or 0.0) for a in actions], label=label)
        ax.legend(loc="upper right")
        ax.set_ylabel(label)
    axes[-1].set_xlabel("simulation time (s)")
    fig.tight_layout()
    fig.savefig(run_dir / "calcium_replay_actions.png", dpi=170)
    plt.close(fig)


def _action_rows_from_video_responses(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first_tick: int | None = None
    for response in responses:
        features = dict(response.get("features") or {})
        diagnostics = dict(response.get("diagnostics") or {})
        state = dict(response.get("state") or {})
        tick = state.get("received_tick")
        if first_tick is None and isinstance(tick, int):
            first_tick = tick
        row = {**features, **diagnostics}
        row["manual_advance_ticks"] = state.get("manual_advance_ticks", 0)
        row["received_tick"] = tick if tick is not None else 0
        if first_tick is not None and isinstance(tick, int):
            row["sim_time_s"] = float((tick - first_tick) * PHYSICS_TIMESTEP_S)
        rows.append(row)
    return rows


def _calcium_action_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first_tick: int | None = None
    for sample in samples:
        replay = dict(sample.get("replay") or {})
        tick = sample.get("received_tick")
        if tick is None:
            tick = replay.get("received_tick")
        if first_tick is None and isinstance(tick, int):
            first_tick = tick
        row = {k: v for k, v in sample.items() if k != "replay"}
        row.update({f"replay_{k}": v for k, v in replay.items() if k not in row})
        if first_tick is not None and isinstance(tick, int):
            row["_sim_time_s"] = float((tick - first_tick) * PHYSICS_TIMESTEP_S)
            row["sim_time_s"] = row["_sim_time_s"]
        rows.append(row)
    return rows


def _summarize_run(
    cap: WsCapture,
    rows: list[dict[str, float]],
    actions: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    if not rows:
        return {"mode": mode, "frames": 0, "error": cap.error}
    fields = [
        "speed_xy_mm_s",
        "speed_3d_mm_s",
        "vertical_speed_mm_s",
        "vertical_speed_abs_mm_s",
        "heading_rate_rad_s",
        "pitch_rate_rad_s",
        "tail_yaw_abs_mean",
        "tail_yaw_abs_max",
        "tail_pitch_abs_mean",
        "joint_angle_abs_mean",
        "joint_velocity_abs_mean",
        "muscle_sum",
        "muscle_lr_bias",
        "muscle_dv_bias",
        "touch_sum",
        "neuron_s_mean",
        "neuron_s_abs_max",
        "neuron_r_mean",
        "neuron_b_mean",
        "neuron_tref_mean",
        "neuron_fired_count",
        "neuron_fired_fraction",
        "neuron_m0_mean",
        "neuron_m1_mean",
        "neuromod_m0",
        "neuromod_m1",
        "free_energy",
        "body_straightness",
        "body_abs_curvature_2d_rad",
        "body_abs_curvature_3d_rad",
        "body_z_span_mm",
    ]
    summary = {
        "mode": mode,
        "frames": len(rows),
        "tick_start": int(rows[0]["tick"]),
        "tick_end": int(rows[-1]["tick"]),
        "tick_span": int(rows[-1]["tick"] - rows[0]["tick"]),
        "sim_seconds": float(rows[-1]["sim_time_s"]),
        "wall_seconds": float(rows[-1]["wall_s"]),
        "travel_xy_mm": float(rows[-1]["xy_displacement_mm"]),
        "travel_3d_mm": float(rows[-1]["xyz_displacement_mm"]),
        "ws_error": cap.error,
        "metrics": {field_name: _stats([r.get(field_name, 0.0) for r in rows]) for field_name in fields},
        "actions": {
            "samples": len(actions),
            "force": _stats([float(a.get("action_force", a.get("force", 0.0)) or 0.0) for a in actions]),
            "side_score": _stats([float(a.get("action_side_score", a.get("side_score", 0.0)) or 0.0) for a in actions]),
            "confidence": _stats([float(a.get("action_confidence", a.get("confidence", 0.0)) or 0.0) for a in actions]),
            "kick": _stats([float(a.get("action_kick", a.get("kick", 0.0)) or 0.0) for a in actions]),
        },
    }
    return summary


def _write_capture_artifacts(
    run_dir: Path,
    cap: WsCapture,
    rows: list[dict[str, float]],
    actions: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(rows, run_dir / "state_metrics.csv")
    _write_jsonl(actions, run_dir / "action_samples.jsonl")
    metadata = {
        "label": label,
        "neuron_names": cap.neuron_names,
        "neuron_meta": cap.neuron_meta,
        "joint_names": cap.joint_names,
        "muscle_names": cap.muscle_names,
        "touch_names": cap.touch_names,
        "frames": len(cap.frames),
        "ws_error": cap.error,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if cap.frames:
        np.savez_compressed(
            run_dir / "state_matrices.npz",
            tick=np.asarray([f["tick"] for f in cap.frames], dtype=np.int64),
            wall_s=np.asarray([f["wall_s"] for f in cap.frames], dtype=np.float64),
            com_mm=np.vstack([f["com_mm"] for f in cap.frames]),
            tail_yaw_rad=_series_matrix(cap, "tail_yaw_rad"),
            tail_pitch_rad=_series_matrix(cap, "tail_pitch_rad"),
            joint_angles_rad=_series_matrix(cap, "joint_angles_rad"),
            joint_velocities_rad_s=_series_matrix(cap, "joint_velocities_rad_s"),
            touch_forces=_series_matrix(cap, "touch_forces"),
            muscle_activations=_series_matrix(cap, "muscle_activations"),
            neuron_s=_series_matrix(cap, "neuron_s"),
            neuron_r=_series_matrix(cap, "neuron_r"),
            neuron_b=_series_matrix(cap, "neuron_b"),
            neuron_tref=_series_matrix(cap, "neuron_tref"),
            neuron_fired=_series_matrix(cap, "neuron_fired"),
            neuron_m0=_series_matrix(cap, "neuron_m0"),
            neuron_m1=_series_matrix(cap, "neuron_m1"),
            neuromod=np.vstack([f["neuromod"] for f in cap.frames]),
            free_energy=np.asarray([f["free_energy"] for f in cap.frames], dtype=np.float64),
        )
    _plot_overview(run_dir, label, rows, actions)
    _plot_trajectory(run_dir, label, rows)
    _plot_heatmaps(run_dir, cap)
    scalar_pages = _plot_all_scalar_metrics(run_dir, label, rows)
    distribution_pages = _plot_scalar_distributions(run_dir, label, rows)
    matrix_heatmaps = _plot_all_matrix_heatmaps(run_dir, cap)
    correlation = _plot_correlation_matrix(run_dir, label, rows)
    frequency = _plot_frequency_analysis(run_dir, label, rows)
    action_alignment = _plot_action_alignment(run_dir, label, rows, actions)
    body_snapshots = _plot_body_shape_snapshots(run_dir, label, cap)
    return {
        "state_metrics_csv": str(run_dir / "state_metrics.csv"),
        "action_samples_jsonl": str(run_dir / "action_samples.jsonl"),
        "state_matrices_npz": str(run_dir / "state_matrices.npz"),
        "overview": str(run_dir / "overview_timeseries.png"),
        "trajectory": str(run_dir / "body_trajectory.png"),
        "scalar_metric_pages": scalar_pages,
        "scalar_distribution_pages": distribution_pages,
        "matrix_heatmaps": matrix_heatmaps,
        "correlation_matrix": correlation,
        "frequency_analysis": frequency,
        "action_body_alignment": action_alignment,
        "body_shape_snapshots": body_snapshots,
    }


async def _run_calcium(
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    label = f"calcium_{args.calcium_condition}"
    run_dir = out_dir / label
    _request_json(args.rest_url, "/video-stimulus/clear", method="POST", payload={}, timeout_s=20.0)
    state0 = _request_json(
        args.rest_url,
        "/calcium-stimulus/replay",
        method="POST",
        payload={
            "enabled": True,
            "condition": args.calcium_condition,
            "gain": args.calcium_gain,
            "loop": True,
        },
        timeout_s=30.0,
    )
    stop_event = asyncio.Event()
    poll_task = asyncio.create_task(
        poll_endpoint(
            args.rest_url,
            "/calcium-stimulus",
            interval_s=args.calcium_poll_interval_s,
            stop_event=stop_event,
            max_wall_s=args.calcium_max_wall_s,
        )
    )
    cap = await capture_ws_until(
        args.ws_url,
        label=label,
        sim_seconds=args.calcium_sim_seconds,
        max_wall_s=args.calcium_max_wall_s,
    )
    stop_event.set()
    calcium_samples = await poll_task
    clear_state = _request_json(args.rest_url, "/calcium-stimulus/clear", method="POST", payload={}, timeout_s=20.0)
    rows = _rows_from_capture(cap)
    actions = _calcium_action_rows(calcium_samples)
    artifacts = _write_capture_artifacts(run_dir, cap, rows, actions, label=label)
    _plot_calcium_actions(run_dir, actions)
    summary = _summarize_run(cap, rows, actions, mode="calcium")
    summary["initial_replay_state"] = state0
    summary["clear_state"] = clear_state
    return summary, artifacts


async def _drive_video_frames(
    args: argparse.Namespace,
    *,
    file_name: str,
    stop_event: asyncio.Event,
) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    _request_json(
        args.rest_url,
        "/video-stimulus/config",
        method="POST",
        payload={"enabled": True, "gain": args.video_gain, "file_name": file_name},
        timeout_s=20.0,
    )
    frames = int(math.ceil(args.video_seconds * args.video_sample_hz))
    source_duration_s = _local_sample_video_duration_s(file_name)
    for frame_index in range(frames):
        driver_video_time_s = float(frame_index / args.video_sample_hz)
        source_loop_index = 0
        source_video_time_s = driver_video_time_s
        if source_duration_s > 0.0 and source_video_time_s >= max(0.0, source_duration_s - 1e-6):
            if not args.video_loop_source:
                raise RuntimeError(
                    f"requested {args.video_seconds:.3f}s but {file_name} is only "
                    f"{source_duration_s:.3f}s; rerun with --video-loop-source or a longer sample"
                )
            source_loop_index = int(source_video_time_s // source_duration_s)
            source_video_time_s = source_video_time_s % source_duration_s
        payload = {
            "file_name": file_name,
            "frame_index": int(frame_index),
            "video_time_s": float(source_video_time_s),
            "sample_hz": float(args.video_sample_hz),
            "enabled": True,
        }
        response = _request_json(
            args.rest_url,
            "/video-stimulus/backend-frame",
            method="POST",
            payload=payload,
            timeout_s=args.video_frame_timeout_s,
        )
        response.setdefault("features", {})
        response["features"]["driver_video_time_s"] = driver_video_time_s
        response["features"]["source_video_time_s"] = source_video_time_s
        response["features"]["source_duration_s"] = source_duration_s
        response["features"]["source_loop_index"] = source_loop_index
        responses.append(response)
        await asyncio.sleep(max(0.0, args.video_post_sleep_s))
    stop_event.set()
    return responses


async def _run_video(
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    samples = _request_json(args.rest_url, "/video-stimulus/samples", timeout_s=20.0).get("samples", [])
    by_slug = {str(s.get("slug")): s for s in samples}
    if args.video_slug not in by_slug:
        raise RuntimeError(f"sample video slug not found: {args.video_slug}; available={sorted(by_slug)}")
    file_name = str(by_slug[args.video_slug]["file_name"])
    label = f"video_{args.video_slug}"
    run_dir = out_dir / label
    _request_json(args.rest_url, "/calcium-stimulus/clear", method="POST", payload={}, timeout_s=20.0)
    stop_event = asyncio.Event()
    capture_task = asyncio.create_task(
        capture_ws_until(
            args.ws_url,
            label=label,
            stop_event=stop_event,
            max_wall_s=args.video_max_wall_s,
            settle_wall_s=args.video_settle_wall_s,
        )
    )
    driver_task = asyncio.create_task(_drive_video_frames(args, file_name=file_name, stop_event=stop_event))
    responses = await driver_task
    cap = await capture_task
    clear_state = _request_json(args.rest_url, "/video-stimulus/clear", method="POST", payload={}, timeout_s=20.0)
    rows = _rows_from_capture(cap)
    actions = _action_rows_from_video_responses(responses)
    artifacts = _write_capture_artifacts(run_dir, cap, rows, actions, label=label)
    _plot_video_diagnostics(run_dir, actions)
    _write_jsonl(responses, run_dir / "backend_frame_responses.jsonl")
    summary = _summarize_run(cap, rows, actions, mode="video")
    summary["video_sample"] = by_slug[args.video_slug]
    summary["posted_frames"] = len(responses)
    summary["video_loop_source"] = bool(args.video_loop_source)
    summary["video_source_duration_s"] = float(actions[0].get("source_duration_s", 0.0)) if actions else 0.0
    summary["video_source_loop_count"] = int(max([int(a.get("source_loop_index", 0) or 0) for a in actions], default=0))
    summary["clear_state"] = clear_state
    return summary, artifacts


def _plot_comparison(out_dir: Path, summaries: dict[str, Any], run_rows: dict[str, list[dict[str, float]]]) -> str:
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=False)
    fields = [
        ("speed_xy_mm_s", "xy speed mm/s"),
        ("tail_yaw_abs_mean", "mean abs tail yaw rad"),
        ("muscle_sum", "muscle sum"),
        ("neuron_fired_count", "fired neurons"),
        ("free_energy", "free energy"),
    ]
    for ax, (field_name, label) in zip(axes, fields):
        for run_name, rows in run_rows.items():
            if not rows:
                continue
            x = [r["sim_time_s"] for r in rows]
            y = [r.get(field_name, 0.0) for r in rows]
            ax.plot(x, y, label=run_name, alpha=0.85)
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("simulation time (s)")
    fig.tight_layout()
    path = out_dir / "pipeline_comparison.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def _render_report(result: dict[str, Any]) -> str:
    calcium = result["runs"]["calcium"]["summary"]
    video = result["runs"]["video"]["summary"]
    comparison = result["artifacts"].get("pipeline_comparison", "")

    def metric(summary: dict[str, Any], name: str, key: str = "mean") -> float:
        return float(((summary.get("metrics") or {}).get(name) or {}).get(key, 0.0))

    lines = [
        "# Zebrafish Comprehensive Activity Study",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- backend: `{result['config']['rest_url']}`",
        f"- websocket: `{result['config']['ws_url']}`",
        f"- output_dir: `{result['output_dir']}`",
        f"- min_tick_target: `{result['config'].get('min_ticks', 0)}`",
        "",
        "## Pipelines Run",
        "",
        (
            f"- Calcium replay: condition `{result['config']['calcium_condition']}`, "
            f"{calcium['frames']} WS frames, {calcium['tick_span']} ticks, "
            f"{calcium['sim_seconds']:.2f} simulated seconds."
        ),
        (
            f"- Video stimulus: sample `{result['config']['video_slug']}`, "
            f"{video.get('posted_frames', 0)} decoded backend frames, {video['frames']} WS frames, "
            f"{video['tick_span']} ticks, {video['sim_seconds']:.2f} simulated seconds."
        ),
        (
            f"- Video source: duration {video.get('video_source_duration_s', 0.0):.2f}s, "
            f"loop_source={video.get('video_loop_source', False)}, "
            f"max_loop_index={video.get('video_source_loop_count', 0)}."
        ),
        "",
        "## Headline Metrics",
        "",
        "| metric | calcium | video |",
        "|---|---:|---:|",
        f"| mean xy speed (mm/s) | {metric(calcium, 'speed_xy_mm_s'):.4f} | {metric(video, 'speed_xy_mm_s'):.4f} |",
        f"| p95 xy speed (mm/s) | {metric(calcium, 'speed_xy_mm_s', 'p95'):.4f} | {metric(video, 'speed_xy_mm_s', 'p95'):.4f} |",
        f"| mean abs vertical speed (mm/s) | {metric(calcium, 'vertical_speed_abs_mm_s'):.4f} | {metric(video, 'vertical_speed_abs_mm_s'):.4f} |",
        f"| mean abs tail yaw (rad) | {metric(calcium, 'tail_yaw_abs_mean'):.4f} | {metric(video, 'tail_yaw_abs_mean'):.4f} |",
        f"| max abs tail yaw p95 (rad) | {metric(calcium, 'tail_yaw_abs_max', 'p95'):.4f} | {metric(video, 'tail_yaw_abs_max', 'p95'):.4f} |",
        f"| mean muscle sum | {metric(calcium, 'muscle_sum'):.4f} | {metric(video, 'muscle_sum'):.4f} |",
        f"| mean fired neurons | {metric(calcium, 'neuron_fired_count'):.4f} | {metric(video, 'neuron_fired_count'):.4f} |",
        f"| mean PAULA S | {metric(calcium, 'neuron_s_mean'):.6f} | {metric(video, 'neuron_s_mean'):.6f} |",
        f"| mean free energy | {metric(calcium, 'free_energy'):.6f} | {metric(video, 'free_energy'):.6f} |",
        f"| XY displacement (mm) | {calcium['travel_xy_mm']:.4f} | {video['travel_xy_mm']:.4f} |",
        "",
        "## Generated Plots",
        "",
        f"- Pipeline comparison: `{comparison}`",
    ]
    if result.get("long_timescale_checks"):
        lines.extend(["", "## Long-Timescale Checks", ""])
        for check in result["long_timescale_checks"]:
            status = "OK" if check.get("ok") else "SHORT"
            lines.append(
                f"- {check['run']}: {status}, tick_span={check['tick_span']}, target={check['target_ticks']}"
            )
    for run_name, run in result["runs"].items():
        artifacts = run["artifacts"]
        lines.append(f"- {run_name} overview: `{artifacts.get('overview', '')}`")
        lines.append(f"- {run_name} trajectory: `{artifacts.get('trajectory', '')}`")
        lines.append(f"- {run_name} body snapshots: `{artifacts.get('body_shape_snapshots', '')}`")
        lines.append(f"- {run_name} scalar metric pages: `{artifacts.get('scalar_metric_pages', '')}`")
        lines.append(f"- {run_name} scalar distributions: `{artifacts.get('scalar_distribution_pages', '')}`")
        lines.append(f"- {run_name} matrix heatmaps: `{artifacts.get('matrix_heatmaps', '')}`")
        lines.append(f"- {run_name} correlation matrix: `{artifacts.get('correlation_matrix', '')}`")
        lines.append(f"- {run_name} frequency analysis: `{artifacts.get('frequency_analysis', '')}`")
        lines.append(f"- {run_name} action/body alignment: `{artifacts.get('action_body_alignment', '')}`")
    lines.extend(
        [
            "",
            "## Interpretation Boundaries",
            "",
            (
                "- Calcium replay is grounded in the current ZAPBench-derived calcium-to-fictive-ephys "
                "artifact and is measured here as decoded action commands applied to the simulated body."
            ),
            (
                "- Video stimulus is measured through the backend optical-flow/stabilization/SimZFish OMR "
                "path plus auxiliary ZAPBench covariate diagnostics. It is not raw ZAPBench movie replay, "
                "because the ZAPBench release does not provide the original visual movies."
            ),
            (
                "- PAULA variables are recorded as activity diagnostics in this study; the active body drive "
                "for these two regimes is still the decoded calcium/video action path."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


async def run_study(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir) if args.output_dir else OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    initial_transport = _request_json(args.rest_url, "/sim/transport", timeout_s=20.0)
    initial_pacing = _request_json(args.rest_url, "/sim/pacing", timeout_s=20.0)
    health = _request_json(args.rest_url, "/health", timeout_s=20.0)
    result: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "output_dir": str(out_dir),
        "health_start": health,
        "initial_transport": initial_transport,
        "initial_pacing": initial_pacing,
        "config": vars(args),
        "runs": {},
        "artifacts": {},
    }
    run_rows: dict[str, list[dict[str, float]]] = {}
    try:
        if args.calcium_pacing_ms is not None:
            _request_json(
                args.rest_url,
                "/sim/pacing",
                method="POST",
                payload={"real_ms_per_physics_step": args.calcium_pacing_ms, "real_ms_per_neural_tick": None},
                timeout_s=20.0,
            )
        calcium_summary, calcium_artifacts = await _run_calcium(args, out_dir)
        result["runs"]["calcium"] = {"summary": calcium_summary, "artifacts": calcium_artifacts}
        calcium_rows = []
        with Path(calcium_artifacts["state_metrics_csv"]).open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                calcium_rows.append({k: float(v) for k, v in row.items() if v not in ("", None)})
        run_rows["calcium"] = calcium_rows

        _request_json(
            args.rest_url,
            "/sim/pacing",
            method="POST",
            payload={
                "real_ms_per_physics_step": initial_pacing.get("real_ms_per_physics_step"),
                "real_ms_per_neural_tick": initial_pacing.get("real_ms_per_neural_tick"),
            },
            timeout_s=20.0,
        )
        video_summary, video_artifacts = await _run_video(args, out_dir)
        result["runs"]["video"] = {"summary": video_summary, "artifacts": video_artifacts}
        video_rows = []
        with Path(video_artifacts["state_metrics_csv"]).open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                video_rows.append({k: float(v) for k, v in row.items() if v not in ("", None)})
        run_rows["video"] = video_rows
        if args.min_ticks > 0:
            result["long_timescale_checks"] = [
                {
                    "run": name,
                    "tick_span": int(run["summary"].get("tick_span", 0)),
                    "target_ticks": int(args.min_ticks),
                    "ok": int(run["summary"].get("tick_span", 0)) >= int(args.min_ticks),
                }
                for name, run in result["runs"].items()
            ]
        result["artifacts"]["pipeline_comparison"] = _plot_comparison(out_dir, result, run_rows)
    finally:
        try:
            _request_json(args.rest_url, "/calcium-stimulus/clear", method="POST", payload={}, timeout_s=20.0)
            _request_json(args.rest_url, "/video-stimulus/clear", method="POST", payload={}, timeout_s=20.0)
            _request_json(
                args.rest_url,
                "/sim/pacing",
                method="POST",
                payload={
                    "real_ms_per_physics_step": initial_pacing.get("real_ms_per_physics_step"),
                    "real_ms_per_neural_tick": initial_pacing.get("real_ms_per_neural_tick"),
                },
                timeout_s=20.0,
            )
            _request_json(
                args.rest_url,
                "/sim/transport",
                method="POST",
                payload={"action": "play" if initial_transport.get("running") else "pause"},
                timeout_s=20.0,
            )
        except Exception as exc:  # noqa: BLE001
            result["restore_error"] = repr(exc)
    result["health_end"] = _request_json(args.rest_url, "/health", timeout_s=20.0)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "REPORT.md").write_text(_render_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rest-url", default="http://127.0.0.1:8765/api")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765/ws/state")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--calcium-condition", default="all")
    parser.add_argument("--calcium-gain", type=float, default=1.0)
    parser.add_argument("--calcium-sim-seconds", type=float, default=90.0)
    parser.add_argument("--calcium-max-wall-s", type=float, default=80.0)
    parser.add_argument("--calcium-pacing-ms", type=float, default=1.0)
    parser.add_argument("--calcium-poll-interval-s", type=float, default=0.20)
    parser.add_argument("--video-slug", default="commons_gopro_crabbing")
    parser.add_argument("--video-gain", type=float, default=1.0)
    parser.add_argument("--video-seconds", type=float, default=60.0)
    parser.add_argument("--video-sample-hz", type=float, default=10.0)
    parser.add_argument("--video-post-sleep-s", type=float, default=0.02)
    parser.add_argument("--video-settle-wall-s", type=float, default=1.0)
    parser.add_argument("--video-max-wall-s", type=float, default=240.0)
    parser.add_argument("--video-frame-timeout-s", type=float, default=45.0)
    parser.add_argument("--video-loop-source", action="store_true")
    parser.add_argument("--min-ticks", type=int, default=0)
    args = parser.parse_args()
    result = asyncio.run(run_study(args))
    print(json.dumps(result, indent=2, default=_json_default, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
