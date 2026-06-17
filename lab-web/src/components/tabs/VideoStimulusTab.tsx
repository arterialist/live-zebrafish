import {
  Pause,
  Play,
  Power,
  RotateCcw,
  Upload,
  Video,
  X,
} from "lucide-react";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ReactNode,
} from "react";

import {
  clearVideoStimulus,
  clearCalciumStimulus,
  getVideoStimulus,
  getVideoStimulusSamples,
  postBackendVideoStimulusFrame,
  setVideoStimulusConfig,
  uploadVideoStimulus,
  type BackendVideoStimulusDiagnostics,
  type BackendVideoStimulusFrameResult,
  type VideoStimulusSample,
  type VideoStimulusState,
} from "../../api/http";
import { TabShell } from "./TabShell";

type ExtractedFrame = BackendVideoStimulusFrameResult["features"];

type DiagnosticPreview = BackendVideoStimulusDiagnostics | null;

const EMPTY_PREVIEW: ExtractedFrame = {
  file_name: "",
  frame_index: 0,
  video_time_s: 0,
  visual_left: 0,
  visual_right: 0,
  optic_flow_left: 0,
  optic_flow_right: 0,
  lateral_line_left: 0,
  lateral_line_right: 0,
  visual_up: 0,
  visual_down: 0,
  light_level: 0,
  startle: 0,
  motion_energy: 0,
  asymmetry: 0,
  backend_extracted: false,
  action_kick: 0,
  action_force: 0,
  action_side_score: 0,
  action_kick_score: 0,
  action_confidence: 0,
  zapbench_row: 0,
};

export function VideoStimulusTab() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const pumpTimerRef = useRef<number | null>(null);
  const inFlightRef = useRef(false);
  const requestEpochRef = useRef(0);
  const lastSampleMsRef = useRef(0);
  const frameIndexRef = useRef(0);
  const enabledRef = useRef(false);
  const sampleHzRef = useRef(15);
  const gainRef = useRef(1);
  const fileNameRef = useRef("");
  const restoreVideoTimeRef = useRef<number | null>(null);

  const [objectUrl, setObjectUrl] = useState("");
  const [fileName, setFileName] = useState("");
  const [samples, setSamples] = useState<VideoStimulusSample[]>([]);
  const [selectedSample, setSelectedSample] = useState("");
  const [enabled, setEnabled] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(true);
  const [gain, setGain] = useState(1);
  const [sampleHz, setSampleHz] = useState(15);
  const [preview, setPreview] = useState<ExtractedFrame | null>(null);
  const [diagnostics, setDiagnostics] = useState<DiagnosticPreview>(null);
  const [serverState, setServerState] = useState<VideoStimulusState | null>(null);
  const [status, setStatus] = useState("Select a video to drive visual input.");
  const [error, setError] = useState("");

  useEffect(() => {
    enabledRef.current = enabled;
  }, [enabled]);

  useEffect(() => {
    sampleHzRef.current = sampleHz;
  }, [sampleHz]);

  useEffect(() => {
    gainRef.current = gain;
  }, [gain]);

  useEffect(() => {
    fileNameRef.current = fileName;
  }, [fileName]);

  useEffect(() => {
    return () => {
      if (pumpTimerRef.current !== null) {
        window.clearInterval(pumpTimerRef.current);
        pumpTimerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    return () => {
      if (objectUrl.startsWith("blob:")) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [objectUrl]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getVideoStimulusSamples(), getVideoStimulus()])
      .then(async ([nextSamples, nextState]) => {
        if (cancelled) {
          return;
        }
        setSamples(nextSamples);
        setServerState(nextState);
        setGain(nextState.gain);

        if (!nextState.file_name) {
          setStatus("Select a video to drive visual input.");
          return;
        }

        const restored = await restoreVideoSource(nextState, nextSamples);
        if (cancelled) {
          return;
        }
        if (!restored) {
          const cleared = await clearVideoStimulus();
          if (!cancelled) {
            setServerState(cleared);
            setEnabled(false);
            setStatus("Uploaded video source is not restored after reload; stimulus cleared.");
          }
          return;
        }

        setEnabled(nextState.enabled);
        frameIndexRef.current = Math.max(0, (nextState.frame_index ?? 0) + 1);
        lastSampleMsRef.current = 0;
        setStatus(
          nextState.enabled
            ? "Video-driven visual input restored from backend state."
            : "Video loaded from backend state.",
        );
      })
      .catch(() => {
        if (!cancelled) {
          setSamples([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!objectUrl) {
      return;
    }
    const video = videoRef.current;
    const restoreTime = restoreVideoTimeRef.current;
    if (!video || restoreTime == null || !Number.isFinite(restoreTime)) {
      return;
    }

    const applyRestoreTime = () => {
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
      const nextTime = loop && duration > 0 ? restoreTime % duration : Math.min(restoreTime, duration || restoreTime);
      video.currentTime = Math.max(0, nextTime);
      restoreVideoTimeRef.current = null;
    };

    if (video.readyState >= 1) {
      applyRestoreTime();
      return;
    }
    video.addEventListener("loadedmetadata", applyRestoreTime, { once: true });
    return () => {
      video.removeEventListener("loadedmetadata", applyRestoreTime);
    };
  }, [objectUrl, loop]);

  useEffect(() => {
    if (!enabled || !objectUrl) {
      return;
    }
    let cancelled = false;
    const playRestoredVideo = async () => {
      try {
        void videoRef.current?.play();
        if (!cancelled) {
          setPlaying(true);
          setError("");
        }
      } catch (err) {
        if (!cancelled) {
          setPlaying(false);
          setError(err instanceof Error ? err.message : String(err));
          setStatus("Video source restored; press play to continue playback.");
        }
      }
    };
    void playRestoredVideo();
    return () => {
      cancelled = true;
    };
  }, [enabled, objectUrl]);

  useEffect(() => {
    if (!enabled || !objectUrl) {
      if (pumpTimerRef.current !== null) {
        window.clearInterval(pumpTimerRef.current);
        pumpTimerRef.current = null;
      }
      return;
    }

    const pump = (nowMs: number) => {
      const video = videoRef.current;
      if (!video || video.readyState < 1) {
        return;
      }

      const intervalMs = 1000 / Math.max(1, sampleHzRef.current);
      if (nowMs - lastSampleMsRef.current < intervalMs) {
        return;
      }
      lastSampleMsRef.current = nowMs;

      if (video.paused || video.ended) {
        const dt = intervalMs / 1000;
        const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
        if (video.ended && !loop) {
          void setRegime(false);
          return;
        }
        if (duration > 0) {
          const nextTime = video.currentTime + dt;
          video.currentTime = loop ? nextTime % duration : Math.min(nextTime, duration);
        } else {
          video.currentTime = Math.max(0, video.currentTime + dt);
        }
      }

      if (inFlightRef.current) {
        return;
      }

      inFlightRef.current = true;
      const requestEpoch = requestEpochRef.current;
      const frameIndex = frameIndexRef.current;
      frameIndexRef.current += 1;

      postBackendVideoStimulusFrame({
        file_name: fileNameRef.current,
        frame_index: frameIndex,
        video_time_s: video.currentTime,
        sample_hz: sampleHzRef.current,
        enabled: true,
      })
        .then((result) => {
          if (requestEpoch !== requestEpochRef.current || !enabledRef.current) {
            void clearVideoStimulus().catch(() => undefined);
            return;
          }
          setPreview(result.features);
          setDiagnostics(result.diagnostics);
          setServerState(result.state);
          setError("");
        })
        .catch((err: unknown) => {
          setError(err instanceof Error ? err.message : String(err));
        })
        .finally(() => {
          inFlightRef.current = false;
        });
    };

    pump(performance.now());
    pumpTimerRef.current = window.setInterval(
      () => pump(performance.now()),
      Math.max(25, Math.floor(500 / Math.max(1, sampleHzRef.current))),
    );
    return () => {
      if (pumpTimerRef.current !== null) {
        window.clearInterval(pumpTimerRef.current);
        pumpTimerRef.current = null;
      }
    };
  }, [enabled, objectUrl]);

  useEffect(() => {
    const stopForCalcium = () => {
      void stopVideoRegime();
    };
    window.addEventListener("zebrafish:stop-video-stimulus", stopForCalcium);
    return () => {
      window.removeEventListener("zebrafish:stop-video-stimulus", stopForCalcium);
    };
  }, []);

  const canRun = objectUrl.length > 0;
  const effectiveState = useMemo(() => serverState ?? null, [serverState]);
  const displayPreview = preview ?? stateToPreview(effectiveState);

  async function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    await stopVideoRegime();
    if (objectUrl.startsWith("blob:")) {
      URL.revokeObjectURL(objectUrl);
    }
    const url = URL.createObjectURL(file);
    setStatus("Uploading video to backend extractor.");
    setObjectUrl(url);
    setSelectedSample("");
    setPreview(null);
    setDiagnostics(null);
    setError("");
    setPlaying(false);
    frameIndexRef.current = 0;
    lastSampleMsRef.current = 0;
    restoreVideoTimeRef.current = null;
    try {
      const upload = await uploadVideoStimulus(file);
      setFileName(upload.file_name);
      const state = await setVideoStimulusConfig({
        enabled: false,
        gain,
        file_name: upload.file_name,
      });
      setServerState(state);
      setStatus("Video uploaded. Backend extraction is ready.");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function onSampleChange(event: ChangeEvent<HTMLSelectElement>) {
    const slug = event.target.value;
    setSelectedSample(slug);
    const sample = samples.find((item) => item.slug === slug);
    if (!sample) {
      await stopVideoRegime();
      setObjectUrl("");
      setFileName("");
      setPreview(null);
      return;
    }
    await stopVideoRegime();
    if (objectUrl.startsWith("blob:")) {
      URL.revokeObjectURL(objectUrl);
    }
    setObjectUrl(sample.url);
    setFileName(sample.file_name);
    setPreview(null);
    setDiagnostics(null);
    setError("");
    setPlaying(false);
    frameIndexRef.current = 0;
    lastSampleMsRef.current = 0;
    restoreVideoTimeRef.current = null;
    try {
      const state = await setVideoStimulusConfig({
        enabled: false,
        gain,
        file_name: sample.file_name,
      });
      setServerState(state);
      setStatus("Cached video loaded. Enable the regime to stream visual input.");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function setRegime(nextEnabled: boolean) {
    if (nextEnabled && !canRun) {
      setError("Select a video before enabling video-driven simulation.");
      return;
    }
    try {
      if (!nextEnabled) {
        const state = await stopVideoRegime();
        setServerState(state);
        setStatus("Video-driven visual input is paused.");
        setError("");
        return;
      }
      if (nextEnabled) {
        window.dispatchEvent(new CustomEvent("zebrafish:stop-calcium-replay"));
        await clearCalciumStimulus().catch(() => undefined);
      }
      const state = await setVideoStimulusConfig({
        enabled: nextEnabled,
        gain,
        file_name: fileName || undefined,
      });
      setServerState(state);
      setEnabled(nextEnabled);
      setError("");
      if (nextEnabled) {
        setStatus("Video-driven visual input is active.");
        void videoRef.current?.play().catch(() => {
          setPlaying(false);
        });
      } else {
        setStatus("Video-driven visual input is paused.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function applyGain() {
    try {
      const state = await setVideoStimulusConfig({
        enabled,
        gain,
        file_name: fileName || undefined,
      });
      setServerState(state);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function clearRegime() {
    try {
      const state = await stopVideoRegime();
      setServerState(state);
      setStatus("Video stimulus cleared.");
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function togglePlayback() {
    const video = videoRef.current;
    if (!video || !objectUrl) {
      return;
    }
    if (video.paused) {
      void video.play().then(
        () => setPlaying(true),
        () => setPlaying(false),
      );
    } else {
      if (enabledRef.current) {
        void setRegime(false);
      } else {
        video.pause();
        setPlaying(false);
      }
    }
  }

  function restartVideo() {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    video.currentTime = 0;
    setDiagnostics(null);
    frameIndexRef.current = 0;
    lastSampleMsRef.current = 0;
  }

  async function stopVideoRegime(): Promise<VideoStimulusState> {
    requestEpochRef.current += 1;
    enabledRef.current = false;
    if (pumpTimerRef.current !== null) {
      window.clearInterval(pumpTimerRef.current);
      pumpTimerRef.current = null;
    }
    await videoRef.current?.pause();
    for (let i = 0; i < 40 && inFlightRef.current; i += 1) {
      await delay(25);
    }
    setPlaying(false);
    setEnabled(false);
    setPreview(null);
    setDiagnostics(null);
    frameIndexRef.current = 0;
    lastSampleMsRef.current = 0;
    restoreVideoTimeRef.current = null;
    const state = await clearVideoStimulus();
    return state;
  }

  async function restoreVideoSource(
    state: VideoStimulusState,
    sampleList: VideoStimulusSample[],
  ): Promise<boolean> {
    const sample = sampleList.find((item) => item.file_name === state.file_name);
    if (sample) {
      setObjectUrl(sample.url);
      setFileName(sample.file_name);
      setSelectedSample(sample.slug);
      setPreview(null);
      setPlaying(false);
      restoreVideoTimeRef.current = state.video_time_s;
      setDiagnostics(null);
      return true;
    }

    return false;
  }

  return (
    <TabShell
      title="VIDEO STIMULUS"
      subtitle="Drive zebrafish visual and lateral-line inputs from a selected local video."
    >
      <div className="space-y-4 text-sm">
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(300px,0.8fr)]">
          <section className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
            <div className="flex flex-wrap items-center gap-2">
              <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs font-medium text-zinc-100 hover:border-zinc-500">
                <Upload size={15} />
                Select video
                <input
                  className="hidden"
                  type="file"
                  accept="video/*"
                  onChange={onFileChange}
                />
              </label>
              <label className="inline-flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs font-medium text-zinc-100">
                cached sample
                <select
                  aria-label="Cached video sample"
                  className="max-w-[220px] bg-transparent text-xs text-zinc-100 outline-none"
                  value={selectedSample}
                  onChange={(event) => void onSampleChange(event)}
                >
                  <option value="">none</option>
                  {samples.map((sample) => (
                    <option key={sample.slug} value={sample.slug}>
                      {sample.slug}
                    </option>
                  ))}
                </select>
              </label>
              <IconButton
                disabled={!canRun}
                title={playing ? "Pause video" : "Play video"}
                onClick={togglePlayback}
              >
                {playing ? <Pause size={15} /> : <Play size={15} />}
              </IconButton>
              <IconButton disabled={!canRun} title="Restart video" onClick={restartVideo}>
                <RotateCcw size={15} />
              </IconButton>
              <button
                className={[
                  "inline-flex items-center gap-2 rounded-md px-3 py-2 text-xs font-medium",
                  enabled
                    ? "border border-emerald-500/40 bg-emerald-500/15 text-emerald-100"
                    : "border border-zinc-700 bg-zinc-900 text-zinc-200 hover:border-zinc-500",
                  !canRun ? "cursor-not-allowed opacity-40" : "",
                ].join(" ")}
                disabled={!canRun}
                onClick={() => void setRegime(!enabled)}
                type="button"
              >
                <Power size={15} />
                {enabled ? "Regime active" : "Enable regime"}
              </button>
              <IconButton disabled={!canRun && !enabled} title="Clear stimulus" onClick={clearRegime}>
                <X size={15} />
              </IconButton>
            </div>

            <div className="mt-4 overflow-hidden rounded-md border border-zinc-800 bg-black">
              {objectUrl ? (
                <video
                  ref={videoRef}
                  className="aspect-video w-full bg-black object-contain"
                  controls={false}
                  loop={loop}
                  muted
                  playsInline
                  src={objectUrl}
                  onPause={() => setPlaying(false)}
                  onPlay={() => setPlaying(true)}
                  onEnded={() => {
                    if (!loop) {
                      void setRegime(false);
                    }
                  }}
                />
              ) : (
                <div className="flex aspect-video items-center justify-center text-zinc-600">
                  <Video size={28} />
                </div>
              )}
            </div>

            <div className="mt-3 grid gap-2 text-xs text-zinc-400 sm:grid-cols-3">
              <StatusRow label="file" value={fileName || "none"} />
              <StatusRow label="backend" value={effectiveState?.enabled ? "video" : "arena"} />
              <StatusRow
                label="frame"
                value={effectiveState ? String(effectiveState.frame_index ?? 0) : "0"}
              />
            </div>
          </section>

          <section className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-zinc-300">
              Regime controls
            </h3>
            <div className="space-y-4">
              <RangeControl
                label="visual gain"
                min={0}
                max={2}
                step={0.05}
                value={gain}
                onChange={setGain}
              />
              <RangeControl
                label="sample Hz"
                min={2}
                max={30}
                step={1}
                value={sampleHz}
                onChange={setSampleHz}
              />
              <label className="flex items-center justify-between rounded-md border border-zinc-800 bg-zinc-900/50 px-3 py-2">
                <span className="text-xs uppercase text-zinc-400">loop video</span>
                <input
                  checked={loop}
                  className="h-4 w-4 accent-sky-500"
                  type="checkbox"
                  onChange={(event) => setLoop(event.target.checked)}
                />
              </label>
              <button
                className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs font-medium text-zinc-100 hover:border-zinc-500"
                type="button"
                onClick={() => void applyGain()}
              >
                Apply gain
              </button>
            </div>
          </section>
        </div>

        <section className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-300">
              Extracted sensory input
            </h3>
            <p className="text-xs text-zinc-500">{error || status}</p>
          </div>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <SignalBar label="visual left" value={displayPreview.visual_left} />
            <SignalBar label="visual right" value={displayPreview.visual_right} />
            <SignalBar label="optic flow L" value={displayPreview.optic_flow_left} />
            <SignalBar label="optic flow R" value={displayPreview.optic_flow_right} />
            <SignalBar label="lateral line L" value={displayPreview.lateral_line_left} />
            <SignalBar label="lateral line R" value={displayPreview.lateral_line_right} />
            <SignalBar label="vertical up" value={displayPreview.visual_up} />
            <SignalBar label="vertical down" value={displayPreview.visual_down} />
            <SignalBar label="light" value={displayPreview.light_level} />
            <SignalBar label="motion" value={displayPreview.motion_energy} />
            <SignalBar label="startle" value={displayPreview.startle} />
            <SignedSignalBar label="asymmetry" value={displayPreview.asymmetry} />
            <SignalBar label="ZAPBench kick" value={displayPreview.action_kick_score} />
            <SignalBar label="action force" value={displayPreview.action_force} />
            <SignedSignalBar label="action side" value={displayPreview.action_side_score} />
            <SignalBar label="action confidence" value={displayPreview.action_confidence} />
          </div>
        </section>
        <section className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-300">
              Backend extraction diagnostics
            </h3>
            <p className="text-xs text-zinc-500">
              {displayPreview.backend_extracted ? "OpenCV optical-flow + ZAPBench calibration" : "Waiting for backend frame."}
            </p>
          </div>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <SignalBar label="flow reliability" value={diagnostics?.flow_reliability ?? 0} />
            <SignalBar label="flow coherence" value={diagnostics?.flow_coherence ?? 0} />
            <SignalBar label="camera shake" value={diagnostics?.camera_shake ?? 0} />
            <SignalBar label="compression noise" value={diagnostics?.compression_noise ?? 0} />
            <SignalBar label="RANSAC inliers" value={diagnostics?.affine_inlier_ratio ?? 0} />
            <SignedSignalBar label="global flow X" value={diagnostics?.global_horizontal_flow ?? 0} />
            <SignedSignalBar label="global flow Y" value={diagnostics?.global_vertical_flow ?? 0} />
            <SignalBar label="ZAPBench match" value={diagnostics ? 1 / (1 + diagnostics.zapbench_distance) : 0} />
          </div>
          <div className="mt-3 grid gap-2 text-xs text-zinc-400 sm:grid-cols-2 xl:grid-cols-4">
            <StatusRow label="extractor" value={diagnostics?.backend ?? "none"} />
            <StatusRow label="true optical flow" value={diagnostics?.true_optical_flow ? "yes" : "no"} />
            <StatusRow label="camera stabilized" value={diagnostics?.camera_stabilized ? "yes" : "no"} />
            <StatusRow label="ZAPBench row" value={String(diagnostics?.zapbench_row ?? displayPreview.zapbench_row ?? 0)} />
          </div>
        </section>
      </div>
    </TabShell>
  );
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function IconButton({
  children,
  disabled = false,
  onClick,
  title,
}: {
  children: ReactNode;
  disabled?: boolean;
  onClick: () => void | Promise<void>;
  title: string;
}) {
  return (
    <button
      className={[
        "inline-flex h-9 w-9 items-center justify-center rounded-md border border-zinc-700 bg-zinc-900 text-zinc-200 hover:border-zinc-500",
        disabled ? "cursor-not-allowed opacity-40" : "",
      ].join(" ")}
      disabled={disabled}
      title={title}
      type="button"
      onClick={() => void onClick()}
    >
      {children}
    </button>
  );
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-md border border-zinc-800 bg-zinc-900/50 px-3 py-2">
      <div className="text-[10px] uppercase text-zinc-500">{label}</div>
      <div className="truncate font-mono text-zinc-200">{value}</div>
    </div>
  );
}

function stateToPreview(state: VideoStimulusState | null): ExtractedFrame {
  if (!state?.has_frame) {
    return EMPTY_PREVIEW;
  }
  return {
    file_name: state.file_name,
    frame_index: state.frame_index,
    video_time_s: state.video_time_s,
    visual_left: state.visual_left,
    visual_right: state.visual_right,
    optic_flow_left: state.optic_flow_left,
    optic_flow_right: state.optic_flow_right,
    lateral_line_left: state.lateral_line_left,
    lateral_line_right: state.lateral_line_right,
    visual_up: state.visual_up,
    visual_down: state.visual_down,
    light_level: state.light_level,
    startle: state.startle,
    motion_energy: state.motion_energy,
    asymmetry: state.asymmetry,
    backend_extracted: state.backend_extracted,
    action_kick: state.action_kick,
    action_force: state.action_force,
    action_side_score: state.action_side_score,
    action_kick_score: state.action_kick_score,
    action_confidence: state.action_confidence,
    zapbench_row: state.zapbench_row,
  };
}

function RangeControl({
  label,
  max,
  min,
  onChange,
  step,
  value,
}: {
  label: string;
  max: number;
  min: number;
  onChange: (value: number) => void;
  step: number;
  value: number;
}) {
  return (
    <label className="block rounded-md border border-zinc-800 bg-zinc-900/50 px-3 py-2">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-xs uppercase text-zinc-400">{label}</span>
        <span className="font-mono text-xs text-zinc-200">{value.toFixed(step < 1 ? 2 : 0)}</span>
      </div>
      <input
        className="w-full accent-sky-500"
        max={max}
        min={min}
        step={step}
        type="range"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function SignalBar({ label, value }: { label: string; value: number }) {
  const clamped = clamp(value, 0, 1);
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/50 px-3 py-2">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-[10px] uppercase text-zinc-500">{label}</span>
        <span className="font-mono text-xs text-zinc-300">{clamped.toFixed(3)}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
        <div
          className="h-full rounded-full bg-sky-400"
          style={{ width: `${Math.round(clamped * 100)}%` }}
        />
      </div>
    </div>
  );
}

function SignedSignalBar({ label, value }: { label: string; value: number }) {
  const clamped = clamp(value, -1, 1);
  const width = Math.round(Math.abs(clamped) * 50);
  const left = clamped < 0 ? 50 - width : 50;
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/50 px-3 py-2">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-[10px] uppercase text-zinc-500">{label}</span>
        <span className="font-mono text-xs text-zinc-300">{clamped.toFixed(3)}</span>
      </div>
      <div className="relative h-1.5 overflow-hidden rounded-full bg-zinc-800">
        <div className="absolute left-1/2 top-0 h-full w-px bg-zinc-600" />
        <div
          className="absolute top-0 h-full rounded-full bg-cyan-300"
          style={{ left: `${left}%`, width: `${width}%` }}
        />
      </div>
    </div>
  );
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
