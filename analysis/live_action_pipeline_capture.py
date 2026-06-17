"""Live REST/WS validation for calcium-to-action and video-to-action regimes.

This script drives the same FastAPI endpoints used by the lab UI and captures
the same WebSocket stream used by the visualizers.  It is intentionally separate
from the browser so the action pipeline can be measured repeatably.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from video_stimulus_pipeline import CLIP_DIR, FEATURE_NAMES, extract_video_features
from zebrafish_reverse_engineering import capture_live, summarize


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "out" / "live_action_pipeline"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _request_json(
    rest_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = rest_url.rstrip("/") + path
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {detail}") from exc


async def _capture_summary(ws_url: str, frames: int, timeout_s: float) -> dict[str, Any]:
    captured = await capture_live(ws_url, frames_target=frames, timeout_s=timeout_s)
    return summarize(captured)


async def _stream_video_features(
    rest_url: str,
    *,
    clip_path: Path,
    sample_hz: float,
    seconds: float,
    gain: float,
) -> dict[str, Any]:
    features = extract_video_features(clip_path, sample_hz=sample_hz, max_seconds=seconds)
    arr = np.asarray(features["features"], dtype=np.float32)
    _request_json(
        rest_url,
        "/video-stimulus/config",
        method="POST",
        payload={"enabled": True, "gain": gain, "file_name": clip_path.name},
    )
    interval_s = 1.0 / max(1.0, float(sample_hz))
    started = time.perf_counter()
    posted = 0
    max_frames = min(arr.shape[0], int(np.ceil(seconds * sample_hz)))
    for frame_index in range(max_frames):
        row = arr[frame_index]
        payload = {
            "file_name": clip_path.name,
            "frame_index": int(frame_index),
            "video_time_s": float(frame_index / sample_hz),
            "sample_hz": float(sample_hz),
            "enabled": True,
        }
        payload.update({name: float(row[i]) for i, name in enumerate(FEATURE_NAMES)})
        _request_json(rest_url, "/video-stimulus/frame", method="POST", payload=payload)
        posted += 1
        target_elapsed = (frame_index + 1) * interval_s
        delay = started + target_elapsed - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
    return {
        "clip_path": str(clip_path),
        "sample_hz": float(sample_hz),
        "requested_seconds": float(seconds),
        "posted_frames": int(posted),
        "feature_summary": {
            "frames": int(arr.shape[0]),
            "motion_mean": float(np.mean(arr[:, FEATURE_NAMES.index("motion_energy")])) if arr.size else 0.0,
            "startle_mean": float(np.mean(arr[:, FEATURE_NAMES.index("startle")])) if arr.size else 0.0,
            "asymmetry_abs_mean": float(np.mean(np.abs(arr[:, FEATURE_NAMES.index("asymmetry")]))) if arr.size else 0.0,
        },
    }


async def run_live_analysis(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir) if args.output_dir else OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    health = _request_json(args.rest_url, "/health")
    results: dict[str, Any] = {
        "config": vars(args),
        "health": health,
        "outputs": {"out_dir": str(out_dir)},
    }

    _request_json(args.rest_url, "/calcium-stimulus/clear", method="POST", payload={})
    calcium_state = _request_json(
        args.rest_url,
        "/calcium-stimulus/replay",
        method="POST",
        payload={
            "enabled": True,
            "condition": args.calcium_condition,
            "gain": args.calcium_gain,
            "loop": True,
        },
    )
    await asyncio.sleep(0.25)
    calcium_summary = await _capture_summary(args.ws_url, args.calcium_frames, args.timeout_s)
    _request_json(args.rest_url, "/calcium-stimulus/clear", method="POST", payload={})
    results["calcium_live"] = {
        "state": calcium_state,
        "summary": calcium_summary,
    }

    clip_path = CLIP_DIR / f"{args.video_slug}.mp4"
    if not clip_path.exists():
        raise SystemExit(f"missing cached video clip: {clip_path}")
    previous_pacing = _request_json(args.rest_url, "/sim/pacing")
    stream_task = asyncio.create_task(
        _stream_video_features(
            args.rest_url,
            clip_path=clip_path,
            sample_hz=args.video_sample_hz,
            seconds=args.video_seconds,
            gain=args.video_gain,
        )
    )
    await asyncio.sleep(0.25)
    video_summary = await _capture_summary(args.ws_url, args.video_frames, args.timeout_s)
    stream_info = await stream_task
    _request_json(args.rest_url, "/video-stimulus/clear", method="POST", payload={})
    results["video_live"] = {
        "stream": stream_info,
        "summary": video_summary,
        "pacing_before": previous_pacing,
    }

    (out_dir / "live_action_pipeline_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_render_report(results), encoding="utf-8")
    return results


def _render_report(results: dict[str, Any]) -> str:
    calcium = results["calcium_live"]["summary"]
    video = results["video_live"]["summary"]
    stream = results["video_live"]["stream"]
    lines = [
        "# Live Zebrafish Action Pipeline Capture",
        "",
        "## Calcium To Action",
        "",
        f"- frames: {calcium['frames']}",
        f"- tick span: {calcium['tick_span']}",
        f"- mean speed mm/s: {calcium['behavior']['xy_speed_mm_s']['mean']:.4f}",
        f"- mean max tail yaw rad: {calcium['joints']['max_abs_tail_yaw_rad']['mean']:.4f}",
        f"- mean axial motor drive: {calcium['motors']['axial_drive']['mean']:.4f}",
        f"- mean neural S: {calcium['neural']['mean_S']['mean']:.5f}",
        "",
        "## Video To Action",
        "",
        f"- clip: {stream['clip_path']}",
        f"- posted video frames: {stream['posted_frames']}",
        f"- captured frames: {video['frames']}",
        f"- tick span: {video['tick_span']}",
        f"- mean speed mm/s: {video['behavior']['xy_speed_mm_s']['mean']:.4f}",
        f"- mean max tail yaw rad: {video['joints']['max_abs_tail_yaw_rad']['mean']:.4f}",
        f"- mean axial motor drive: {video['motors']['axial_drive']['mean']:.4f}",
        f"- mean neural S: {video['neural']['mean_S']['mean']:.5f}",
        "",
        "## Notes",
        "",
        "- These captures validate the current action-controller-to-body path used by the lab backend.",
        "- PAULA state is captured as a diagnostic signal, but the current action source is still the decoded calcium/video command path, not a completed PAULA-muscle connectome.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rest-url", default="http://127.0.0.1:8811/api")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8811/ws/state")
    parser.add_argument("--calcium-condition", default="turning")
    parser.add_argument("--calcium-gain", type=float, default=1.0)
    parser.add_argument("--calcium-frames", type=int, default=900)
    parser.add_argument("--video-slug", default="commons_gopro_crabbing")
    parser.add_argument("--video-sample-hz", type=float, default=15.0)
    parser.add_argument("--video-seconds", type=float, default=20.0)
    parser.add_argument("--video-frames", type=int, default=900)
    parser.add_argument("--video-gain", type=float, default=1.0)
    parser.add_argument(
        "--video-real-ms-per-physics-step",
        type=float,
        default=5.0,
        help="Deprecated; video replay now advances physics manually per decoded frame.",
    )
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    results = asyncio.run(run_live_analysis(args))
    print(json.dumps(results, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
