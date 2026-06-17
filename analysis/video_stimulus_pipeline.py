"""Dataset-style validation for the zebrafish lab video-stimulus regime.

The lab UI streams low-dimensional visual features extracted from a selected
video.  This script makes that path reproducible offline:

* download/cache a curated set of underwater POV or fish-level videos;
* normalize each source into a local MP4 clip suitable for browser/lab use;
* extract the same retinal/optic-flow features as the React lab tab;
* drive the embodied zebrafish simulation through the public video-stimulus
  API fields; and
* write dataset, feature, behavior, neural, and ZAPBench-alignment summaries.

It deliberately does not modify ``neuron.py``.  ZAPBench is used as the current
calcium/ephys grounding source: the released ephys-derived labels define the
reference action distribution, while the current lab path remains a video
feature -> action controller -> body motor pathway.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = REPO_DIR.parent
CACHE_DIR = SCRIPT_DIR / "cache" / "video_stimuli"
RAW_DIR = CACHE_DIR / "raw"
CLIP_DIR = CACHE_DIR / "clips"
FEATURE_DIR = CACHE_DIR / "features"
OUT_ROOT = SCRIPT_DIR / "out" / "video_stimulus_pipeline"

FRAME_W = 96
FRAME_H = 54
PHYSICS_DT_S = 0.005


def _ticks_for_sample_frame(frame_index: int, sample_hz: float) -> int:
    """No-drift conversion from one decoded video frame to MuJoCo ticks."""
    hz = max(1.0, float(sample_hz))
    idx = max(0, int(frame_index))
    ticks_per_frame = 1.0 / (hz * PHYSICS_DT_S)
    start_tick = int(round(idx * ticks_per_frame))
    end_tick = int(round((idx + 1) * ticks_per_frame))
    return max(1, end_tick - start_tick)


def _total_ticks_for_sample_frames(frame_count: int, sample_hz: float) -> int:
    if frame_count <= 0:
        return 0
    hz = max(1.0, float(sample_hz))
    return max(1, int(round(frame_count / (hz * PHYSICS_DT_S))))


@dataclass(frozen=True)
class VideoSource:
    slug: str
    label: str
    page_url: str
    download_url: str | None = None
    commons_title: str | None = None
    local_fallback: str | None = None
    license_note: str = ""
    start_s: float = 0.0
    clip_s: float = 90.0
    raw_download: bool = True
    notes: str = ""


SOURCES: tuple[VideoSource, ...] = (
    VideoSource(
        slug="bassmaster_bass_fish_pov",
        label="Bassmaster bass-mounted GoPro POV",
        page_url="https://www.bassmaster.com/video/video-mercer-shows-a-bass-point-of-view/",
        local_fallback="/tmp/bass-fish-pov.mp4",
        license_note="Source page, not an open-license Commons asset; local prior capture is used when present.",
        start_s=42.0,
        clip_s=32.0,
        raw_download=False,
        notes="Exact fish-mounted POV; underwater section is shorter than 60 s, kept as a reference control.",
    ),
    VideoSource(
        slug="noaa_batfish_rov",
        label="NOAA Batfish ROV",
        page_url="https://oceanexplorer.noaa.gov/multimedia/video-shorts-ex1605-pizzafish/",
        download_url="https://oceanexplorer.noaa.gov/wp-content/uploads/2021/02/ex1605-pizzafish.mp4",
        license_note="NOAA .gov downloadable MP4.",
        clip_s=61.0,
        notes="ROV view of a benthic fish; verified duration is just over one minute.",
    ),
    VideoSource(
        slug="noaa_juvenile_fish_rov",
        label="NOAA juvenile fish ROV",
        page_url="https://oceanexplorer.noaa.gov/multimedia/okeanos-explorations-ex1905-dailyupdates-sept1-media-its-so-cute/",
        download_url="https://oceanexplorer.noaa.gov/wp-content/uploads/2025/05/its-so-cute-1920x1080-1.mp4",
        license_note="NOAA .gov downloadable HD MP4.",
        clip_s=61.0,
        notes="ROV view of a tiny juvenile fish; verified duration is just over one minute.",
    ),
    VideoSource(
        slug="commons_brycon_rio_prata",
        label="Brycon hilarii, Rio da Prata",
        page_url="https://commons.wikimedia.org/wiki/File:Brycon_hilarii_piraputanga_rio_prata_01.webm",
        commons_title="File:Brycon hilarii piraputanga rio prata 01.webm",
        license_note="Wikimedia Commons CC BY-SA 4.0.",
        clip_s=110.0,
        notes="Clear freshwater fish-level scene with visible teleost movement.",
    ),
    VideoSource(
        slug="commons_madeira_garajau",
        label="Madeira Garajau shallow dive",
        page_url="https://commons.wikimedia.org/wiki/File:Madeira_Garajau.webm",
        commons_title="File:Madeira Garajau.webm",
        license_note="Wikimedia Commons CC BY 3.0.",
        clip_s=109.0,
        notes="Diver/reef POV with fish and habitat structure.",
    ),
    VideoSource(
        slug="commons_char_school",
        label="USFWS char school",
        page_url="https://commons.wikimedia.org/wiki/File:Char_at_the_Russian_River_Ferry_(31666975855).webm",
        commons_title="File:Char at the Russian River Ferry (31666975855).webm",
        license_note="Public domain, U.S. Fish and Wildlife Service.",
        clip_s=76.0,
        notes="Underwater school of char; useful schooling-control stimulus.",
    ),
    VideoSource(
        slug="commons_gopro_crabbing",
        label="GoPro underwater crabbing",
        page_url="https://commons.wikimedia.org/wiki/File:GoPro_Underwater_Crabbing_from_Inflatable_Kayak.webm",
        commons_title="File:GoPro Underwater Crabbing from Inflatable Kayak.webm",
        license_note="Wikimedia Commons CC BY 3.0.",
        clip_s=120.0,
        notes="Bottom-level GoPro view with strong near-field optic flow.",
    ),
    VideoSource(
        slug="commons_black_rockfish_stereo_dov",
        label="Black rockfish stereo-DOV supplement",
        page_url="https://commons.wikimedia.org/wiki/File:Short-Term-Fidelity-Habitat-Use-and-Vertical-Movement-Behavior-of-the-Black-Rockfish-Sebastes-pone.0134381.s012.ogv",
        commons_title="File:Short-Term-Fidelity-Habitat-Use-and-Vertical-Movement-Behavior-of-the-Black-Rockfish-Sebastes-pone.0134381.s012.ogv",
        license_note="Wikimedia Commons CC BY 4.0; PLOS ONE supplemental video.",
        clip_s=98.0,
        notes="Scientific fish/habitat video from a stereo-DOV.",
    ),
    VideoSource(
        slug="commons_deep_landers",
        label="Autonomous deep-sea landers",
        page_url="https://commons.wikimedia.org/wiki/File:Autonomous_landers,_Observing_the_deepest_places_on_Earth.WebM",
        commons_title="File:Autonomous landers, Observing the deepest places on Earth.WebM",
        license_note="Wikimedia Commons CC BY 3.0.",
        clip_s=120.0,
        notes="Scientific static/lander POV with deep-sea fish.",
    ),
    VideoSource(
        slug="commons_precision_aquaculture",
        label="Precision aquaculture underwater robots",
        page_url="https://commons.wikimedia.org/wiki/File:Precision_aquaculture_unleashing_underwater_robots_(1080p).webm",
        commons_title="File:Precision aquaculture unleashing underwater robots (1080p).webm",
        license_note="Wikimedia Commons CC BY-SA 4.0.",
        clip_s=120.0,
        notes="Aquaculture/robot underwater POV; includes non-underwater sections but useful robot-context control.",
    ),
    VideoSource(
        slug="commons_lindquist_snorkeling",
        label="Snorkeling at Lindquist Beach",
        page_url="https://commons.wikimedia.org/wiki/File:Snorkeling_at_Lindquist_beach_-_US_Virgin_Islands_March_2021.webm",
        download_url="https://upload.wikimedia.org/wikipedia/commons/transcoded/4/45/Snorkeling_at_Lindquist_beach_-_US_Virgin_Islands_March_2021.webm/Snorkeling_at_Lindquist_beach_-_US_Virgin_Islands_March_2021.webm.720p.vp9.webm",
        commons_title="File:Snorkeling at Lindquist beach - US Virgin Islands March 2021.webm",
        license_note="Wikimedia Commons CC BY 3.0; license review pending on source page.",
        clip_s=180.0,
        raw_download=True,
        notes="Long continuous snorkeler POV. The original is 1.69 GB; this pipeline downloads the 720p Commons transcode and stores a normalized clip.",
    ),
    VideoSource(
        slug="commons_tenggol_underwater",
        label="Discovering Tenggol underwater segment",
        page_url="https://commons.wikimedia.org/wiki/File:Discovering_Tenggol-_Aerial_Views_and_Submerged_Delights_in_4K.webm",
        download_url="https://upload.wikimedia.org/wikipedia/commons/transcoded/8/86/Discovering_Tenggol-_Aerial_Views_and_Submerged_Delights_in_4K.webm/Discovering_Tenggol-_Aerial_Views_and_Submerged_Delights_in_4K.webm.720p.vp9.webm",
        commons_title="File:Discovering Tenggol- Aerial Views and Submerged Delights in 4K.webm",
        license_note="Wikimedia Commons CC BY 3.0.",
        start_s=14.0 * 60.0 + 36.0,
        clip_s=180.0,
        raw_download=True,
        notes="Original is 3.28 GB; this pipeline downloads the 720p Commons transcode and clips the documented dive section at 14:36.",
    ),
)


def _run(cmd: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, default=_json_default), flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _stats(values: np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _duration_s(path_or_url: str | Path) -> float:
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path_or_url),
        ],
        timeout=60,
    ).stdout.strip()
    return float(out) if out else 0.0


def _commons_original_url(title: str) -> str:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        }
    )
    req = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{params}",
        headers={"User-Agent": "agi-research-zebrafish-video-dataset/0.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    pages = data["query"]["pages"]
    info = next(iter(pages.values()))["imageinfo"][0]
    return str(info["url"]).split("?utm_", 1)[0]


def _download_url(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "agi-research-zebrafish-video-dataset/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.replace(path)


def _video_ext(url: str) -> str:
    clean = url.split("?", 1)[0]
    suffix = Path(urllib.parse.unquote(clean)).suffix.lower()
    return suffix if suffix else ".mp4"


def _make_clip(src: str | Path, out_path: Path, *, start_s: float, clip_s: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".part.mp4")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-t",
        f"{float(clip_s):.3f}",
        "-vf",
        "scale='min(1280,iw)':-2,fps=30",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    if start_s > 1e-6:
        cmd[5:5] = ["-ss", f"{max(0.0, float(start_s)):.3f}"]
    _run(cmd, timeout=60 * 20)
    tmp.replace(out_path)


def prepare_dataset(force: bool = False) -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for source in SOURCES:
        item: dict[str, Any] = asdict(source)
        item["downloaded"] = False
        item["raw_path"] = None
        item["clip_path"] = None
        item["error"] = None
        try:
            url = source.download_url
            if source.commons_title is not None and source.download_url is None:
                url = _commons_original_url(source.commons_title)
            src_for_clip: str | Path | None = url
            if source.local_fallback and Path(source.local_fallback).exists():
                src_for_clip = Path(source.local_fallback)
                raw_path = RAW_DIR / f"{source.slug}{Path(source.local_fallback).suffix.lower()}"
                if force or not raw_path.exists():
                    shutil.copy2(source.local_fallback, raw_path)
                item["raw_path"] = str(raw_path)
                item["raw_bytes"] = raw_path.stat().st_size
                item["raw_sha256"] = _sha256(raw_path)
            elif url and source.raw_download:
                raw_path = RAW_DIR / f"{source.slug}{_video_ext(url)}"
                if force or not raw_path.exists():
                    _download_url(url, raw_path)
                src_for_clip = raw_path
                item["raw_path"] = str(raw_path)
                item["raw_bytes"] = raw_path.stat().st_size
                item["raw_sha256"] = _sha256(raw_path)
            elif url:
                src_for_clip = url
            else:
                item["error"] = "No direct URL and no local fallback available."

            item["resolved_download_url"] = str(url) if url else None
            if src_for_clip is not None and item["error"] is None:
                clip_path = CLIP_DIR / f"{source.slug}.mp4"
                if force or not clip_path.exists():
                    _make_clip(src_for_clip, clip_path, start_s=source.start_s, clip_s=source.clip_s)
                item["clip_path"] = str(clip_path)
                item["clip_bytes"] = clip_path.stat().st_size
                item["clip_sha256"] = _sha256(clip_path)
                item["clip_duration_s"] = _duration_s(clip_path)
                if float(item["clip_duration_s"]) < min(1.0, source.clip_s):
                    raise RuntimeError(f"normalized clip has no video duration: {clip_path}")
                item["downloaded"] = True
        except Exception as exc:  # noqa: BLE001
            item["error"] = f"{type(exc).__name__}: {exc}"
        manifest.append(item)
        _emit({"prepared": source.slug, "downloaded": item["downloaded"], "error": item["error"]})
    return {"sources": manifest}


def extract_video_features(video_path: Path, sample_hz: float, max_seconds: float | None = None) -> dict[str, Any]:
    duration = _duration_s(video_path)
    seconds = min(duration, float(max_seconds)) if max_seconds else duration
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-t",
        f"{seconds:.3f}",
        "-vf",
        f"fps={float(sample_hz):.6f},scale={FRAME_W}:{FRAME_H}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    frame_bytes = FRAME_W * FRAME_H * 3
    gray_prev: np.ndarray | None = None
    features: list[list[float]] = []
    hist_bins = np.linspace(0.0, 1.0, 17)
    entropy_values: list[float] = []
    sharpness_values: list[float] = []

    while True:
        raw = proc.stdout.read(frame_bytes)
        if not raw:
            break
        if len(raw) != frame_bytes:
            break
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape((FRAME_H, FRAME_W, 3)).astype(np.float32) / 255.0
        lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        feat = _frame_features_from_luma(lum, gray_prev)
        features.append(feat)
        gray_prev = lum.copy()
        hist, _ = np.histogram(lum, bins=hist_bins, density=False)
        prob = hist.astype(np.float64)
        prob = prob / max(1.0, float(np.sum(prob)))
        entropy_values.append(float(-np.sum(prob[prob > 0.0] * np.log2(prob[prob > 0.0])) / 4.0))
        gx = np.diff(lum, axis=1, prepend=lum[:, :1])
        gy = np.diff(lum, axis=0, prepend=lum[:1, :])
        sharpness_values.append(float(np.mean(np.sqrt(gx * gx + gy * gy))))

    _stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    code = proc.wait(timeout=30)
    if code != 0:
        raise RuntimeError(f"ffmpeg feature extraction failed for {video_path}: {_stderr}")
    arr = np.asarray(features, dtype=np.float32)
    return {
        "path": str(video_path),
        "duration_s": duration,
        "analyzed_seconds": seconds,
        "sample_hz": float(sample_hz),
        "frame_count": int(arr.shape[0]),
        "feature_names": FEATURE_NAMES,
        "features": arr,
        "image_entropy": np.asarray(entropy_values, dtype=np.float32),
        "image_sharpness": np.asarray(sharpness_values, dtype=np.float32),
    }


FEATURE_NAMES = [
    "visual_left",
    "visual_right",
    "optic_flow_left",
    "optic_flow_right",
    "lateral_line_left",
    "lateral_line_right",
    "visual_up",
    "visual_down",
    "light_level",
    "startle",
    "motion_energy",
    "asymmetry",
]


def _frame_features_from_luma(lum: np.ndarray, prev_lum: np.ndarray | None) -> list[float]:
    mean = float(np.mean(lum))
    salience = np.abs(lum - mean)
    motion = np.abs(lum - prev_lum) if prev_lum is not None else np.zeros_like(lum)
    left = slice(0, FRAME_W // 2)
    right = slice(FRAME_W // 2, FRAME_W)
    up = slice(0, FRAME_H // 2)
    down = slice(FRAME_H // 2, FRAME_H)
    left_lum = float(np.mean(lum[:, left]))
    right_lum = float(np.mean(lum[:, right]))
    up_lum = float(np.mean(lum[up, :]))
    down_lum = float(np.mean(lum[down, :]))
    left_sal = float(np.mean(salience[:, left]))
    right_sal = float(np.mean(salience[:, right]))
    up_sal = float(np.mean(salience[up, :]))
    down_sal = float(np.mean(salience[down, :]))
    left_motion = float(np.mean(motion[:, left]))
    right_motion = float(np.mean(motion[:, right]))
    total_motion = float(np.mean(motion))

    visual_left = _clamp(0.45 * left_lum + 1.65 * left_sal + 3.0 * left_motion)
    visual_right = _clamp(0.45 * right_lum + 1.65 * right_sal + 3.0 * right_motion)
    visual_up = _clamp(0.35 * up_lum + 1.2 * up_sal)
    visual_down = _clamp(0.35 * down_lum + 1.2 * down_sal)
    optic_left = _clamp(left_motion * 5.0)
    optic_right = _clamp(right_motion * 5.0)
    motion_energy = _clamp(total_motion * 4.0)
    denom = max(0.02, visual_left + visual_right)
    return [
        visual_left,
        visual_right,
        optic_left,
        optic_right,
        _clamp(left_motion * 3.5),
        _clamp(right_motion * 3.5),
        visual_up,
        visual_down,
        _clamp(mean),
        _clamp((motion_energy - 0.18) * 4.0),
        motion_energy,
        float(np.clip((visual_left - visual_right) / denom, -1.0, 1.0)),
    ]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, value)))


def _load_zapbench_reference() -> dict[str, Any]:
    path = SCRIPT_DIR / "out" / "zapbench_ephys_action_decoder" / "direct_ephys_labels.npz"
    if not path.exists():
        return {"available": False, "path": str(path)}
    with np.load(path, allow_pickle=False) as data:
        force = np.asarray(data["force"], dtype=np.float64)
        kick = np.asarray(data["kick"], dtype=np.float64) > 0.5
        side_class = np.asarray(data["side_class"], dtype=np.int32)
        side_active = side_class != 0
        return {
            "available": True,
            "path": str(path),
            "frames": int(force.size),
            "kick_rate": float(np.mean(kick)),
            "force": _stats(force),
            "force_active": _stats(force[kick]),
            "side_active_rate": float(np.mean(side_active)),
            "side_counts": {
                "none": int(np.sum(side_class == 0)),
                "left": int(np.sum(side_class == 1)),
                "right": int(np.sum(side_class == 2)),
            },
        }


def _configure_imports() -> None:
    active = WORKSPACE_DIR / "active-inference"
    if str(active) not in sys.path:
        sys.path.insert(0, str(active))


def run_simulation_for_features(
    feature_arr: np.ndarray,
    *,
    file_name: str,
    sample_hz: float,
    max_seconds: float,
    gain: float,
    seed: int,
) -> dict[str, Any]:
    _configure_imports()
    from simulations.zebrafish.simulation import build_zebrafish_simulation

    engine, loop = build_zebrafish_simulation(
        food_positions=[],
        log_level="ERROR",
        record_neural_states=False,
        max_history=8,
        suppress_connectome_summary=True,
        seed=seed,
    )
    engine.reset(nervous_rebuild=False)
    env = engine.environment
    env.set_video_stimulus(enabled=True, gain=gain, file_name=file_name)
    ns = engine.nervous_system

    n_frames = min(int(feature_arr.shape[0]), int(math.ceil(max_seconds * sample_hz)))
    ticks_per_frame_mean = 1.0 / (max(1.0, float(sample_hz)) * PHYSICS_DT_S)
    max_ticks = _total_ticks_for_sample_frames(n_frames, sample_hz)

    speed = np.zeros(max_ticks, dtype=np.float32)
    z_mm = np.zeros(max_ticks, dtype=np.float32)
    swim = np.zeros(max_ticks, dtype=np.float32)
    turn = np.zeros(max_ticks, dtype=np.float32)
    pitch = np.zeros(max_ticks, dtype=np.float32)
    m0 = np.zeros(max_ticks, dtype=np.float32)
    m1 = np.zeros(max_ticks, dtype=np.float32)
    fe = np.zeros(max_ticks, dtype=np.float32)
    fired_frac = np.zeros(max_ticks, dtype=np.float32)
    tail_yaw_peak = np.zeros(max_ticks, dtype=np.float32)
    stimulus_frame = np.zeros(max_ticks, dtype=np.int32)

    tick_i = 0
    for fi in range(n_frames):
        f = feature_arr[fi]
        env.push_video_stimulus_frame(
            {
                "file_name": file_name,
                "frame_index": int(fi),
                "video_time_s": float(fi / sample_hz),
                "visual_left": float(f[0]),
                "visual_right": float(f[1]),
                "optic_flow_left": float(f[2]),
                "optic_flow_right": float(f[3]),
                "lateral_line_left": float(f[4]),
                "lateral_line_right": float(f[5]),
                "visual_up": float(f[6]),
                "visual_down": float(f[7]),
                "light_level": float(f[8]),
                "startle": float(f[9]),
                "motion_energy": float(f[10]),
                "asymmetry": float(f[11]),
                "enabled": True,
            }
        )
        frame_ticks = _ticks_for_sample_frame(fi, sample_hz)
        for _ in range(frame_ticks):
            step = engine.step()
            extra = step.body_state.extra or {}
            bs = ns.behavior_state
            speed[tick_i] = float(extra.get("speed_m_s", 0.0)) * 1000.0
            z_mm[tick_i] = float(step.body_state.position[2]) * 1000.0
            swim[tick_i] = float(bs.get("swim_drive", 0.0))
            turn[tick_i] = float(bs.get("turn_bias", 0.0))
            pitch[tick_i] = float(bs.get("pitch_bias", 0.0))
            m0[tick_i], m1[tick_i] = [float(x) for x in ns.neuromod_levels]
            if loop.log_free_energy and loop.free_energy_trace.prediction_error:
                fe[tick_i] = float(loop.free_energy_trace.prediction_error[-1])
            s_list, fired, _r = ns.get_compact_neural_snapshot()
            fired_frac[tick_i] = float(np.mean(fired)) if fired else 0.0
            tail = np.asarray(extra.get("tail_angles", []), dtype=np.float64)
            tail_yaw_peak[tick_i] = float(np.max(np.abs(tail))) if tail.size else 0.0
            stimulus_frame[tick_i] = int(fi)
            tick_i += 1

    arrays = {
        "speed_mm_s": speed[:tick_i],
        "z_mm": z_mm[:tick_i],
        "swim_drive": swim[:tick_i],
        "turn_bias": turn[:tick_i],
        "pitch_bias": pitch[:tick_i],
        "m0": m0[:tick_i],
        "m1": m1[:tick_i],
        "free_energy": fe[:tick_i],
        "fired_frac": fired_frac[:tick_i],
        "tail_yaw_peak_rad": tail_yaw_peak[:tick_i],
        "stimulus_frame": stimulus_frame[:tick_i],
    }
    return _summarize_simulation(arrays, sample_hz=sample_hz, ticks_per_frame=ticks_per_frame_mean)


def _event_count(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    starts = mask & np.concatenate([[True], ~mask[:-1]])
    return int(np.sum(starts))


def _summarize_simulation(
    arrays: dict[str, np.ndarray],
    *,
    sample_hz: float,
    ticks_per_frame: float,
) -> dict[str, Any]:
    swim = arrays["swim_drive"]
    turn = arrays["turn_bias"]
    pitch = arrays["pitch_bias"]
    ticks = int(swim.size)
    seconds = float(ticks * PHYSICS_DT_S)
    kick_mask = swim > 0.12
    turn_mask = kick_mask & (np.abs(turn) > 0.065)
    climb_mask = kick_mask & (pitch > 0.12)
    dive_mask = kick_mask & (pitch < -0.12)
    return {
        "ticks": ticks,
        "seconds": seconds,
        "sample_hz": float(sample_hz),
        "ticks_per_frame": float(ticks_per_frame),
        "speed_mm_s": _stats(arrays["speed_mm_s"]),
        "depth_z_mm": _stats(arrays["z_mm"]),
        "swim_drive": _stats(swim),
        "turn_bias": _stats(turn),
        "pitch_bias": _stats(pitch),
        "tail_yaw_peak_rad": _stats(arrays["tail_yaw_peak_rad"]),
        "free_energy": _stats(arrays["free_energy"]),
        "neural_fired_fraction": _stats(arrays["fired_frac"]),
        "neuromod_m0": _stats(arrays["m0"]),
        "neuromod_m1": _stats(arrays["m1"]),
        "kick_rate": float(np.mean(kick_mask)) if ticks else 0.0,
        "side_active_rate": float(np.mean(turn_mask)) if ticks else 0.0,
        "left_turn_rate": float(np.mean(kick_mask & (turn < -0.065))) if ticks else 0.0,
        "right_turn_rate": float(np.mean(kick_mask & (turn > 0.065))) if ticks else 0.0,
        "climb_rate": float(np.mean(climb_mask)) if ticks else 0.0,
        "dive_rate": float(np.mean(dive_mask)) if ticks else 0.0,
        "bout_events": _event_count(kick_mask),
        "turn_events": _event_count(turn_mask),
    }


def feature_summary(features: dict[str, Any]) -> dict[str, Any]:
    arr = np.asarray(features["features"], dtype=np.float64)
    out: dict[str, Any] = {
        "frame_count": int(arr.shape[0]),
        "duration_s": float(features["duration_s"]),
        "analyzed_seconds": float(features["analyzed_seconds"]),
        "sample_hz": float(features["sample_hz"]),
        "image_entropy": _stats(features["image_entropy"]),
        "image_sharpness": _stats(features["image_sharpness"]),
        "signals": {},
    }
    for i, name in enumerate(FEATURE_NAMES):
        out["signals"][name] = _stats(arr[:, i] if arr.size else [])
    if arr.size:
        out["motion_frames_rate"] = float(np.mean(arr[:, FEATURE_NAMES.index("motion_energy")] > 0.08))
        out["startle_frames_rate"] = float(np.mean(arr[:, FEATURE_NAMES.index("startle")] > 0.05))
        out["strong_asymmetry_rate"] = float(np.mean(np.abs(arr[:, FEATURE_NAMES.index("asymmetry")]) > 0.08))
    else:
        out["motion_frames_rate"] = 0.0
        out["startle_frames_rate"] = 0.0
        out["strong_asymmetry_rate"] = 0.0
    return out


def run_pipeline(
    *,
    sample_hz: float,
    sim_seconds: float,
    gain: float,
    seed: int,
    force_download: bool,
    skip_download: bool,
    simulate_slugs: set[str] | None,
    features_only: bool,
    download_only: bool,
) -> Path:
    run_dir = OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": []} if skip_download else prepare_dataset(force=force_download)
    if skip_download:
        for source in SOURCES:
            clip_path = CLIP_DIR / f"{source.slug}.mp4"
            item = {
                **asdict(source),
                "downloaded": clip_path.exists(),
                "clip_path": str(clip_path),
                "error": None if clip_path.exists() else "missing clip",
            }
            if clip_path.exists():
                item["clip_bytes"] = clip_path.stat().st_size
                try:
                    item["clip_duration_s"] = _duration_s(clip_path)
                except Exception as exc:  # noqa: BLE001
                    item["error"] = f"{type(exc).__name__}: {exc}"
                    item["downloaded"] = False
            manifest["sources"].append(item)
    (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    if download_only:
        (run_dir / "report.md").write_text(_render_download_report(manifest), encoding="utf-8")
        return run_dir

    zap = _load_zapbench_reference()
    feature_summaries: dict[str, Any] = {}
    sim_summaries: dict[str, Any] = {}
    for item in manifest["sources"]:
        if not item.get("downloaded") or not item.get("clip_path"):
            continue
        path = Path(item["clip_path"])
        features = extract_video_features(path, sample_hz=sample_hz, max_seconds=sim_seconds)
        arr = np.asarray(features["features"], dtype=np.float32)
        feature_path = FEATURE_DIR / f"{item['slug']}_features_{sample_hz:g}hz.npz"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feature_path,
            features=arr,
            feature_names=np.asarray(FEATURE_NAMES),
            image_entropy=features["image_entropy"],
            image_sharpness=features["image_sharpness"],
        )
        fs = feature_summary(features)
        fs["feature_npz"] = str(feature_path)
        feature_summaries[item["slug"]] = fs
        _emit({"features": item["slug"], "frames": fs["frame_count"]})
        if features_only or (simulate_slugs is not None and item["slug"] not in simulate_slugs):
            continue
        sim = run_simulation_for_features(
            arr,
            file_name=path.name,
            sample_hz=sample_hz,
            max_seconds=sim_seconds,
            gain=gain,
            seed=seed,
        )
        sim_summaries[item["slug"]] = sim
        _emit({"simulated": item["slug"], "ticks": sim["ticks"], "kick_rate": sim["kick_rate"]})

    aggregate = _aggregate_results(feature_summaries, sim_summaries, zap)
    outputs = {
        "config": {
            "sample_hz": sample_hz,
            "sim_seconds": sim_seconds,
            "gain": gain,
            "seed": seed,
            "physics_dt_s": PHYSICS_DT_S,
        },
        "zapbench_reference": zap,
        "features": feature_summaries,
        "simulation": sim_summaries,
        "aggregate": aggregate,
    }
    (run_dir / "analysis_summary.json").write_text(json.dumps(outputs, indent=2, default=_json_default), encoding="utf-8")
    (run_dir / "report.md").write_text(_render_report(manifest, outputs), encoding="utf-8")
    return run_dir


def _render_download_report(manifest: dict[str, Any]) -> str:
    lines = ["# Zebrafish Video Dataset Download Manifest", ""]
    lines.append("| stimulus | downloaded | clip seconds | raw path | clip path |")
    lines.append("|---|---:|---:|---|---|")
    for item in manifest["sources"]:
        lines.append(
            f"| {item['slug']} | {bool(item.get('downloaded'))} | "
            f"{float(item.get('clip_duration_s') or 0.0):.1f} | "
            f"{item.get('raw_path') or ''} | {item.get('clip_path') or ''} |"
        )
    return "\n".join(lines) + "\n"


def _aggregate_results(features: dict[str, Any], sims: dict[str, Any], zap: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for slug, sim in sims.items():
        fs = features.get(slug, {})
        rows.append(
            {
                "slug": slug,
                "motion_mean": fs.get("signals", {}).get("motion_energy", {}).get("mean", 0.0),
                "asymmetry_abs_mean": fs.get("signals", {}).get("asymmetry", {}).get("mean", 0.0),
                "kick_rate": sim.get("kick_rate", 0.0),
                "side_active_rate": sim.get("side_active_rate", 0.0),
                "speed_mean": sim.get("speed_mm_s", {}).get("mean", 0.0),
                "free_energy_mean": sim.get("free_energy", {}).get("mean", 0.0),
            }
        )
    if not rows:
        return {"n": 0}
    kick = np.asarray([r["kick_rate"] for r in rows], dtype=np.float64)
    side = np.asarray([r["side_active_rate"] for r in rows], dtype=np.float64)
    speed = np.asarray([r["speed_mean"] for r in rows], dtype=np.float64)
    motion = np.asarray([r["motion_mean"] for r in rows], dtype=np.float64)
    target_kick = float(zap.get("kick_rate", 0.0)) if zap.get("available") else 0.0
    target_side = float(zap.get("side_active_rate", 0.0)) if zap.get("available") else 0.0
    return {
        "n": len(rows),
        "rows": rows,
        "kick_rate": _stats(kick),
        "side_active_rate": _stats(side),
        "speed_mean_mm_s": _stats(speed),
        "motion_mean": _stats(motion),
        "zapbench_kick_rate_abs_error": float(abs(np.mean(kick) - target_kick)) if zap.get("available") else None,
        "zapbench_side_rate_abs_error": float(abs(np.mean(side) - target_side)) if zap.get("available") else None,
    }


def _render_report(manifest: dict[str, Any], outputs: dict[str, Any]) -> str:
    lines: list[str] = []
    cfg = outputs["config"]
    zap = outputs["zapbench_reference"]
    agg = outputs["aggregate"]
    lines.append("# Zebrafish Video-Stimulus Pipeline Analysis")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This run treats the downloaded videos as a stimulus dataset and drives the same "
        "backend video-stimulus fields used by the lab: visual left/right, optic-flow "
        "left/right, lateral-line left/right, vertical salience, light, startle, motion "
        "energy, and asymmetry."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(cfg, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## ZAPBench Grounding")
    lines.append("")
    if zap.get("available"):
        lines.append(f"- Direct ephys label frames: {zap['frames']}")
        lines.append(f"- Reference kick rate: {zap['kick_rate']:.3f}")
        lines.append(f"- Reference side-active rate: {zap['side_active_rate']:.3f}")
        lines.append(f"- Reference force mean: {zap['force']['mean']:.3f}")
    else:
        lines.append("- ZAPBench direct ephys labels were not available in this workspace.")
    lines.append("")
    lines.append("Important limitation: the current lab video tab does not predict a full ZAPBench calcium trace from arbitrary natural video. It sends retinal/optic-flow features into the body action controller. ZAPBench currently grounds the reference action distribution and the existing calcium/ephys-to-tail-action decoder, but no trained natural-video-to-calcium model is yet in this runtime path.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append("| stimulus | downloaded | clip seconds | notes |")
    lines.append("|---|---:|---:|---|")
    for item in manifest["sources"]:
        lines.append(
            f"| {item['slug']} | {bool(item.get('downloaded'))} | "
            f"{float(item.get('clip_duration_s') or 0.0):.1f} | {item.get('notes','')} |"
        )
    lines.append("")
    lines.append("## Aggregate Simulation")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({k: v for k, v in agg.items() if k != "rows"}, indent=2, default=_json_default))
    lines.append("```")
    lines.append("")
    lines.append("## Per-Video Summary")
    lines.append("")
    lines.append("| stimulus | motion mean | kick rate | side-active rate | speed mean mm/s | FE mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in agg.get("rows", []):
        lines.append(
            f"| {row['slug']} | {row['motion_mean']:.3f} | {row['kick_rate']:.3f} | "
            f"{row['side_active_rate']:.3f} | {row['speed_mean']:.2f} | {row['free_energy_mean']:.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- A useful PoC should show monotonic coupling between natural-video motion features and simulated bout drive, without saturating every video into continuous swimming.")
    lines.append("- Side-active rates should be nonzero for asymmetric/near-field clips and lower for static ROV/lander clips.")
    lines.append("- ZAPBench ephys labels are the current biological reference distribution; large deviations indicate the lab regime needs calibration or a learned video-to-brain model before robot-control claims are defensible.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-hz", type=float, default=15.0)
    parser.add_argument("--sim-seconds", type=float, default=75.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--simulate-slugs",
        default="",
        help="Comma-separated subset to simulate; features are still extracted for every downloaded clip.",
    )
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    simulate_slugs = {s.strip() for s in args.simulate_slugs.split(",") if s.strip()} or None
    run_dir = run_pipeline(
        sample_hz=args.sample_hz,
        sim_seconds=args.sim_seconds,
        gain=args.gain,
        seed=args.seed,
        force_download=args.force_download,
        skip_download=args.skip_download,
        simulate_slugs=simulate_slugs,
        features_only=args.features_only,
        download_only=args.download_only,
    )
    _emit({"run_dir": str(run_dir)})


if __name__ == "__main__":
    main()
