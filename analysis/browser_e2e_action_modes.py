"""Browser-driven e2e recording checks for calcium and video action modes.

The browser UI is responsible for starting each mode. This recorder only
observes the same backend state and WebSocket stream used by the lab viewers,
then writes raw payloads, decoded behavioral metrics, and anomaly checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from zebrafish_reverse_engineering import _frame_from_wire, summarize


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "out" / "browser_e2e_action_modes"


def _request_json(rest_url: str, path: str) -> dict[str, Any]:
    url = rest_url.rstrip("/") + path
    with urllib.request.urlopen(url, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


async def _record_ws(ws_url: str, frames_target: int, timeout_s: float, raw_path: Path) -> tuple[list[Any], dict[str, Any]]:
    import websockets

    names: list[str] = []
    frames = []
    raw_count = 0
    first_tick = None
    last_tick = None
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as fh:
        async with websockets.connect(ws_url, open_timeout=timeout_s) as ws:
            while len(frames) < frames_target:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                payload = json.loads(raw)
                if payload.get("t") == "h":
                    names = list((payload.get("L") or {}).get("nm") or [])
                    continue
                if payload.get("t") != "s":
                    continue
                raw_count += 1
                tick = int(payload.get("k", 0))
                first_tick = tick if first_tick is None else first_tick
                last_tick = tick
                fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
                frames.append(_frame_from_wire(payload, names))
    return frames, {
        "raw_payloads": raw_count,
        "first_tick": first_tick,
        "last_tick": last_tick,
        "raw_path": str(raw_path),
    }


def _analyze(summary: dict[str, Any], start_state: dict[str, Any], end_state: dict[str, Any], mode: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, value: Any = None, threshold: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "value": value, "threshold": threshold})

    frames = int(summary.get("frames", 0))
    tick_span = int(summary.get("tick_span", 0))
    behavior = summary.get("behavior", {})
    body = summary.get("body", {})
    joints = summary.get("joints", {})
    motors = summary.get("motors", {})

    speed_mean = float((behavior.get("xy_speed_mm_s") or {}).get("mean", 0.0))
    travel = float(behavior.get("travel_mm", 0.0))
    free_energy_max = float((behavior.get("free_energy") or {}).get("max", 0.0))
    bend_p95 = float((body.get("max_local_bend_3d_rad") or {}).get("p95", 0.0))
    z_span_p95 = float((body.get("z_span_mm") or {}).get("p95", 0.0))
    pitch_p95 = float((body.get("body_pitch_rad") or {}).get("p95", 0.0))
    yaw_limit_mean = float((joints.get("fraction_yaw_joints_above_0p54_rad") or {}).get("mean", 0.0))
    axial_drive_mean = float((motors.get("axial_drive") or {}).get("mean", 0.0))

    add("enough_frames", frames >= 600, frames, ">= 600")
    add("ticks_advance", tick_span > 0, tick_span, "> 0")
    add("finite_speed", math.isfinite(speed_mean), speed_mean)
    add("finite_free_energy", math.isfinite(free_energy_max), free_energy_max)
    add("not_exploding_bend", bend_p95 < 1.25, bend_p95, "< 1.25 rad p95")
    add("not_vertical_pitch", abs(pitch_p95) < 1.25, pitch_p95, "|p95| < 1.25 rad")
    add("bounded_z_span", z_span_p95 < 25.0, z_span_p95, "< 25 mm p95")
    add("not_joint_saturated", yaw_limit_mean < 0.35, yaw_limit_mean, "< 35% saturated yaw joints")
    add("active_motor_drive", axial_drive_mean > 0.001, axial_drive_mean, "> 0.001")

    if mode == "calcium":
        add("calcium_enabled_at_start", bool(start_state.get("enabled")), start_state.get("enabled"))
        add("calcium_has_frame", bool(end_state.get("has_frame")), end_state.get("has_frame"))
        add("calcium_frame_advances", int(end_state.get("frame_index", 0)) > int(start_state.get("frame_index", -1)), {
            "start": start_state.get("frame_index"),
            "end": end_state.get("frame_index"),
        })
        add("calcium_command_nonzero", float(end_state.get("kick_score", 0.0)) > 0.0 or float(end_state.get("force", 0.0)) > 0.0, {
            "kick_score": end_state.get("kick_score"),
            "force": end_state.get("force"),
        })
    elif mode == "video":
        add("video_enabled_at_start", bool(start_state.get("enabled")), start_state.get("enabled"))
        add("video_backend_frame", bool(end_state.get("backend_extracted")), end_state.get("backend_extracted"))
        add("video_frame_advances", int(end_state.get("frame_index", 0)) > int(start_state.get("frame_index", -1)), {
            "start": start_state.get("frame_index"),
            "end": end_state.get("frame_index"),
        })
        add("video_action_nonzero", float(end_state.get("action_confidence", 0.0)) > 0.0 or float(end_state.get("action_force", 0.0)) > 0.0, {
            "confidence": end_state.get("action_confidence"),
            "force": end_state.get("action_force"),
        })

    return {
        "passed": all(item["ok"] for item in checks),
        "checks": checks,
        "headline": {
            "frames": frames,
            "tick_span": tick_span,
            "speed_mean_mm_s": speed_mean,
            "travel_mm": travel,
            "axial_drive_mean": axial_drive_mean,
            "bend_p95_rad": bend_p95,
            "body_pitch_p95_rad": pitch_p95,
            "z_span_p95_mm": z_span_p95,
            "free_energy_max": free_energy_max,
        },
    }


async def record_case(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir) if args.output_dir else OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = "/calcium-stimulus" if args.mode == "calcium" else "/video-stimulus"
    start_state = _request_json(args.rest_url, state_path)
    frames, raw_meta = await _record_ws(
        args.ws_url,
        frames_target=args.frames,
        timeout_s=args.timeout_s,
        raw_path=out_dir / f"{args.case_name}_raw_ws.ndjson",
    )
    end_state = _request_json(args.rest_url, state_path)
    summary = summarize(frames)
    analysis = _analyze(summary, start_state, end_state, args.mode)
    result = {
        "case": args.case_name,
        "mode": args.mode,
        "config": vars(args),
        "start_state": start_state,
        "end_state": end_state,
        "raw_recording": raw_meta,
        "summary": summary,
        "analysis": analysis,
    }
    (out_dir / f"{args.case_name}_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_report(out_dir / f"{args.case_name}_report.md", result)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return result


def _write_report(path: Path, result: dict[str, Any]) -> None:
    headline = result["analysis"]["headline"]
    checks = result["analysis"]["checks"]
    lines = [
        f"# Browser E2E Action Mode: {result['case']}",
        "",
        f"- mode: `{result['mode']}`",
        f"- passed: `{result['analysis']['passed']}`",
        f"- raw recording: `{result['raw_recording']['raw_path']}`",
        f"- frames: {headline['frames']}",
        f"- tick span: {headline['tick_span']}",
        f"- mean speed mm/s: {headline['speed_mean_mm_s']:.4f}",
        f"- travel mm: {headline['travel_mm']:.4f}",
        f"- mean axial drive: {headline['axial_drive_mean']:.4f}",
        f"- bend p95 rad: {headline['bend_p95_rad']:.4f}",
        f"- body pitch p95 rad: {headline['body_pitch_p95_rad']:.4f}",
        f"- z span p95 mm: {headline['z_span_p95_mm']:.4f}",
        f"- max free energy: {headline['free_energy_max']:.6f}",
        "",
        "## Checks",
        "",
    ]
    for item in checks:
        mark = "PASS" if item["ok"] else "FAIL"
        lines.append(f"- {mark} `{item['name']}` value=`{item.get('value')}` threshold=`{item.get('threshold')}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calcium", "video"], required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--rest-url", default="http://127.0.0.1:8811/api")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8811/ws/state")
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    asyncio.run(record_case(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
