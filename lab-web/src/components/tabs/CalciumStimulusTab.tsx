import {
  Activity,
  BarChart3,
  BrainCircuit,
  Database,
  FileText,
  LineChart,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useRef } from "react";

import {
  clearCalciumStimulus,
  clearVideoStimulus,
  getCalciumStimulus,
  getZapbenchReplayDetail,
  getZapbenchSummary,
  setCalciumReplay,
  type CalciumStimulusState,
  type ZapbenchBrainPreview,
  type ZapbenchReplayDetail,
  type ZapbenchStimulusCovariates,
  type ZapbenchSummary,
} from "../../api/http";
import { TabShell } from "./TabShell";

const DEFAULT_CONDITIONS = [
  "gain",
  "dots",
  "flash",
  "taxis",
  "turning",
  "position",
  "open loop",
  "rotation",
  "dark",
  "all",
];

export function CalciumStimulusTab() {
  const [state, setState] = useState<CalciumStimulusState | null>(null);
  const [condition, setCondition] = useState("turning");
  const [gain, setGain] = useState(1);
  const [loop, setLoop] = useState(true);
  const [status, setStatus] = useState("Loading calcium replay state.");
  const [error, setError] = useState("");
  const [zapbenchSummary, setZapbenchSummary] = useState<ZapbenchSummary | null>(null);
  const [zapbenchDetail, setZapbenchDetail] = useState<ZapbenchReplayDetail | null>(null);
  const [zapbenchError, setZapbenchError] = useState("");
  const didInitialSyncRef = useRef(false);

  const conditions = useMemo(() => {
    const names = state?.replay.condition_names ?? [];
    const merged = [...names, "all"].filter(Boolean);
    return merged.length > 1 ? Array.from(new Set(merged)) : DEFAULT_CONDITIONS;
  }, [state]);

  useEffect(() => {
    let cancelled = false;
    void getZapbenchSummary()
      .then((next) => {
        if (!cancelled) {
          setZapbenchSummary(next);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setZapbenchError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    const refresh = async () => {
      let nextDelayMs = 1000;
      try {
        const next = await getCalciumStimulus();
        nextDelayMs = next.replay.enabled ? 250 : 1000;
        if (cancelled) {
          return;
        }
        setState(next);
        setError("");
        const shouldSyncControls = !didInitialSyncRef.current;
        if (shouldSyncControls && next.replay.condition) {
          setCondition(next.replay.condition);
        }
        if (shouldSyncControls) {
          setGain(next.replay.gain);
          setLoop(next.replay.loop);
          didInitialSyncRef.current = true;
        }
        setStatus(next.replay.enabled ? "Calcium replay is driving body action commands." : "Calcium replay is idle.");
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(() => {
            void refresh();
          }, nextDelayMs);
        }
      }
    };
    void refresh();
    return () => {
      cancelled = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, []);

  useEffect(() => {
    const stopForVideo = () => {
      void clearReplay();
    };
    window.addEventListener("zebrafish:stop-calcium-replay", stopForVideo);
    return () => {
      window.removeEventListener("zebrafish:stop-calcium-replay", stopForVideo);
    };
  }, []);

  const replay = state?.replay ?? null;
  const canRun = Boolean(replay?.available);
  const enabled = Boolean(replay?.enabled);
  const hasFrame = Boolean(state?.has_frame);
  const selectedFrames = replay?.selected_frames ?? 0;
  const totalFrames = replay?.frames ?? 0;
  const currentFrameIndex = hasFrame ? state?.frame_index ?? 0 : replay?.frame_index ?? 0;
  const currentFrameNumber =
    hasFrame && selectedFrames > 0 ? Math.min(currentFrameIndex + 1, selectedFrames) : 0;
  const progressFraction =
    selectedFrames > 0 ? Math.min(1, Math.max(0, currentFrameNumber / selectedFrames)) : 0;
  const progressPercent = progressFraction * 100;
  const sourceRow = hasFrame ? state?.row ?? 0 : replay?.row ?? 0;
  const calciumTimeS = hasFrame ? state?.calcium_time_s ?? 0 : replay?.calcium_time_s ?? 0;
  const action = {
    kick: state?.kick ?? replay?.kick ?? 0,
    force: state?.force ?? replay?.force ?? 0,
    kickScore: state?.kick_score ?? replay?.kick_score ?? 0,
    confidence: state?.confidence ?? replay?.confidence ?? 0,
    sideScore: state?.side_score ?? replay?.side_score ?? 0,
    side: state?.side ?? replay?.side ?? "none",
  };
  const sideMeaning = action.side || "none";

  useEffect(() => {
    const row = Math.max(0, Math.round(Number(sourceRow) || 0));
    let cancelled = false;
    void getZapbenchReplayDetail(row)
      .then((next) => {
        if (!cancelled) {
          setZapbenchDetail(next);
          setZapbenchError("");
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setZapbenchError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [sourceRow]);

  async function refreshState() {
    try {
      const next = await getCalciumStimulus();
      setState(next);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function setReplay(nextEnabled: boolean) {
    try {
      if (nextEnabled) {
        window.dispatchEvent(new CustomEvent("zebrafish:stop-video-stimulus"));
        await clearVideoStimulus().catch(() => undefined);
      }
      const replayState = await setCalciumReplay({
        enabled: nextEnabled,
        condition,
        gain,
        loop,
      });
      setState((prev) =>
        prev
          ? {
              ...prev,
              replay: replayState,
              enabled: replayState.enabled,
              gain: replayState.gain,
            }
          : prev,
      );
      setError("");
      setStatus(nextEnabled ? "Calcium replay is driving body action commands." : "Calcium replay is paused.");
      window.setTimeout(() => void refreshState(), 150);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function applySettings() {
    try {
      const replayState = await setCalciumReplay({
        enabled,
        condition,
        gain,
        loop,
      });
      setState((prev) => (prev ? { ...prev, replay: replayState } : prev));
      setError("");
      setStatus("Calcium replay settings applied.");
      window.setTimeout(() => void refreshState(), 150);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function clearReplay() {
    try {
      const replayState = await clearCalciumStimulus();
      setState((prev) =>
        prev
          ? {
              ...prev,
              replay: replayState,
              enabled: false,
              has_frame: false,
            }
          : prev,
      );
      setStatus("Calcium replay cleared.");
      setError("");
      window.setTimeout(() => void refreshState(), 150);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function restartReplay() {
    void setReplay(true);
  }

  return (
    <TabShell
      title="CALCIUM REPLAY"
      subtitle="Drive tail-kick, side, and force commands from decoded ZAPBench calcium activity."
    >
      <div className="space-y-4 text-sm">
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(300px,0.8fr)]">
          <section className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
            <div className="flex flex-wrap items-center gap-2">
              <button
                className={[
                  "inline-flex items-center gap-2 rounded-md px-3 py-2 text-xs font-medium",
                  enabled
                    ? "border border-emerald-500/40 bg-emerald-500/15 text-emerald-100"
                    : "border border-zinc-700 bg-zinc-900 text-zinc-200 hover:border-zinc-500",
                  !canRun ? "cursor-not-allowed opacity-40" : "",
                ].join(" ")}
                disabled={!canRun}
                onClick={() => void setReplay(!enabled)}
                type="button"
              >
                {enabled ? <Pause size={15} /> : <Play size={15} />}
                {enabled ? "Replay active" : "Start replay"}
              </button>
              <IconButton disabled={!canRun} title="Restart replay" onClick={restartReplay}>
                <RotateCcw size={15} />
              </IconButton>
              <IconButton title="Refresh state" onClick={refreshState}>
                <RefreshCw size={15} />
              </IconButton>
              <IconButton disabled={!state?.has_frame && !enabled} title="Clear calcium replay" onClick={clearReplay}>
                <X size={15} />
              </IconButton>
            </div>

            <div className="mt-4 rounded-lg border border-zinc-800 bg-zinc-900/35 p-3">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="text-[10px] uppercase tracking-wide text-zinc-500">
                    replay progress
                  </div>
                  <div className="font-mono text-sm text-zinc-100">
                    {currentFrameNumber} / {selectedFrames}
                  </div>
                </div>
                <div className="text-right font-mono text-xs text-zinc-400">
                  {formatPercent(progressPercent)}
                  {replay?.loop ? " loop" : ""}
                </div>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full rounded-full bg-sky-400"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>

            <div className="mt-3 grid gap-2 text-xs text-zinc-400 sm:grid-cols-2 xl:grid-cols-4">
              <StatusRow label="artifact" value={canRun ? "ready" : "missing"} />
              <StatusRow label="condition" value={replay?.condition ?? condition} />
              <StatusRow label="selected frames" value={formatInt(selectedFrames)} />
              <StatusRow label="all frames" value={formatInt(totalFrames)} />
              <StatusRow label="ZAPBench row" value={formatInt(sourceRow)} />
              <StatusRow label="calcium time" value={`${formatSeconds(calciumTimeS)} s`} />
              <StatusRow
                label="age ticks"
                value={state?.age_ticks == null ? "none" : formatInt(state.age_ticks)}
              />
              <StatusRow label="decoded side" value={sideMeaning} />
              <StatusRow label="source" value={state?.source || replay?.source || "none"} />
              <StatusRow label="path" value={replay?.path ?? ""} />
            </div>
          </section>

          <section className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-zinc-300">
              Replay controls
            </h3>
            <div className="space-y-4">
              <label className="block rounded-md border border-zinc-800 bg-zinc-900/50 px-3 py-2">
                <div className="mb-2 flex items-center justify-between gap-3">
                  <span className="text-xs uppercase text-zinc-400">condition</span>
                  <Database size={14} className="text-zinc-500" />
                </div>
                <select
                  className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-100 outline-none focus:border-sky-500"
                  value={condition}
                  onChange={(event) => setCondition(event.target.value)}
                >
                  {conditions.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
              <RangeControl
                label="action gain"
                min={0}
                max={2}
                step={0.05}
                value={gain}
                onChange={setGain}
              />
              <label className="flex items-center justify-between rounded-md border border-zinc-800 bg-zinc-900/50 px-3 py-2">
                <span className="text-xs uppercase text-zinc-400">loop replay</span>
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
                onClick={() => void applySettings()}
              >
                Apply settings
              </button>
            </div>
          </section>
        </div>

        <section className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h3 className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-zinc-300">
              <BrainCircuit size={15} />
              Decoded calcium action
            </h3>
            <p className="text-xs text-zinc-500">
              {error || replay?.error || status}
            </p>
          </div>
          <div className="mb-3 grid gap-2 text-xs text-zinc-400 sm:grid-cols-2 xl:grid-cols-5">
            <StatusRow label="kick command" value={formatNumber(action.kick, 3)} />
            <StatusRow label="force magnitude" value={formatNumber(action.force, 3)} />
            <StatusRow label="kick score" value={formatNumber(action.kickScore, 3)} />
            <StatusRow label="confidence" value={formatNumber(action.confidence, 3)} />
            <StatusRow label="side score" value={formatSigned(action.sideScore, 3)} />
          </div>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
            <SignalBar
              label="kick command"
              value={action.kick}
              rawLabel={formatNumber(action.kick, 3)}
            />
            <SignalBar
              label="force magnitude"
              value={action.force}
              rawLabel={formatNumber(action.force, 3)}
            />
            <SignalBar
              label="kick score"
              value={action.kickScore}
              rawLabel={formatNumber(action.kickScore, 3)}
            />
            <SignalBar
              label="confidence"
              value={action.confidence}
              rawLabel={formatNumber(action.confidence, 3)}
            />
            <SignedSignalBar
              label={`side score (${sideMeaning})`}
              value={action.sideScore}
              rawLabel={formatSigned(action.sideScore, 3)}
            />
          </div>
        </section>

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(360px,0.9fr)]">
          <EphysTruthOverlay detail={zapbenchDetail} />
          <DecoderModelCard detail={zapbenchDetail} summary={zapbenchSummary} />
        </div>

        <ConditionTimeline detail={zapbenchDetail} />

        <div className="grid gap-4 xl:grid-cols-[minmax(360px,0.9fr)_minmax(0,1.1fr)]">
          <StimulusCovariateViewer covariates={zapbenchDetail?.stimulus_covariates ?? null} />
          <BrainTracePreview preview={zapbenchDetail?.brain_preview ?? null} />
        </div>

        {zapbenchError ? (
          <p className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
            ZAPBench evidence layer: {zapbenchError}
          </p>
        ) : null}
      </div>
    </TabShell>
  );
}

function EphysTruthOverlay({ detail }: { detail: ZapbenchReplayDetail | null }) {
  const truth = detail?.ephys_truth ?? null;
  const prediction = detail?.decoder_prediction ?? null;
  const replay = detail?.replay_action ?? null;
  const left = truth?.left_power ?? 0;
  const right = truth?.right_power ?? 0;
  const total = truth?.total_power ?? 0;
  const powerScale = Math.max(0.001, left, right, total);

  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
      <PanelTitle icon={<Activity size={15} />} title="Raw ephys truth overlay" />
      {!detail ? (
        <EvidencePlaceholder label="Loading row-aligned motor-nerve labels." />
      ) : truth?.available ? (
        <div className="space-y-3">
          <div className="grid gap-2 text-xs text-zinc-400 sm:grid-cols-2 xl:grid-cols-4">
            <StatusRow label="ground truth kick" value={truth.kick ? "yes" : "no"}/>
            <StatusRow label="ground truth side" value={truth.side_name ?? "none"} />
            <StatusRow label="ground truth force" value={formatMaybeNumber(truth.force, 3)} />
            <StatusRow
              label="raw samples"
              value={`${formatInt(truth.raw_start_sample ?? 0)}-${formatInt(truth.raw_end_sample ?? 0)}`}
            />
          </div>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <SignalBar label="left ephys power" rawLabel={formatMaybeNumber(left, 5)} value={left / powerScale} />
            <SignalBar
              label="right ephys power"
              rawLabel={formatMaybeNumber(right, 5)}
              value={right / powerScale}
            />
            <SignalBar
              label="total ephys power"
              rawLabel={formatMaybeNumber(total, 5)}
              value={total / powerScale}
            />
            <SignedSignalBar
              label="left/right ephys side"
              rawLabel={formatSigned(truth.side ?? 0, 3)}
              value={truth.side ?? 0}
            />
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <SignedSignalBar
              label={`decoder side (${prediction?.side_name ?? "none"})`}
              rawLabel={formatSigned(prediction?.side_score ?? 0, 3)}
              value={prediction?.side_score ?? 0}
            />
            <SignalBar
              label={`replay force (${replay?.side_name ?? "none"})`}
              rawLabel={formatMaybeNumber(replay?.force, 3)}
              value={Number(replay?.force ?? 0)}
            />
          </div>
        </div>
      ) : (
        <EvidencePlaceholder label={truth?.reason || truth?.error || "Raw ephys labels unavailable for this row."} />
      )}
    </section>
  );
}

function DecoderModelCard({
  detail,
  summary,
}: {
  detail: ZapbenchReplayDetail | null;
  summary: ZapbenchSummary | null;
}) {
  const model = detail?.model_card ?? null;
  const prediction = detail?.decoder_prediction ?? null;
  const thresholds = prediction?.thresholds ?? {};
  const metrics = Object.entries(model?.metrics ?? {}).filter((entry): entry is [string, number] => {
    return typeof entry[1] === "number" && Number.isFinite(entry[1]);
  });
  const conditionMetrics = Object.entries(model?.condition_metrics ?? {}).filter(
    (entry): entry is [string, number] => typeof entry[1] === "number" && Number.isFinite(entry[1]),
  );

  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
      <PanelTitle icon={<FileText size={15} />} title="Decoder model card" />
      {!detail || !model ? (
        <EvidencePlaceholder label="Loading decoder provenance and validation metrics." />
      ) : (
        <div className="space-y-3">
          <div className="grid gap-2 text-xs text-zinc-400 sm:grid-cols-2 xl:grid-cols-4">
            <StatusRow label="label source" value="raw tail ephys" />
            <StatusRow
              label="label lag"
              value={formatInt(Number(prediction?.metadata?.label_lag ?? 0))}
            />
            <StatusRow
              label="decoder neurons"
              value={formatInt(Number(model.artifact.neurons ?? 0))}
            />
            <StatusRow
              label="projection dims"
              value={formatInt(Number(model.artifact.projection_components ?? 0))}
            />
            <StatusRow
              label="row split"
              value={prediction?.is_test_row ? "test" : prediction?.is_train_row ? "train" : "unknown"}
            />
            <StatusRow label="condition" value={detail.condition.name} />
            <StatusRow label="source links" value={formatInt(summary?.sources.length ?? 0)} />
            <StatusRow label="label frames" value={formatInt(model.label_artifact.frames ?? 0)} />
          </div>

          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
            {metrics.map(([key, value]) => (
              <MetricCard key={key} label={formatMetricLabel(key)} value={formatMetricValue(key, value)} />
            ))}
          </div>

          <div className="grid gap-2 md:grid-cols-3">
            {Object.entries(thresholds).map(([key, value]) => (
              <StatusRow key={key} label={`threshold ${key}`} value={formatMaybeNumber(Number(value), 3)} />
            ))}
          </div>

          {conditionMetrics.length ? (
            <div>
              <div className="mb-2 text-[10px] uppercase tracking-wide text-zinc-500">
                current-condition validation
              </div>
              <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
                {conditionMetrics
                  .filter(([key]) =>
                    ["kick_accuracy", "side_accuracy", "force_mae", "force_r2"].includes(key),
                  )
                  .map(([key, value]) => (
                    <MetricCard key={key} label={formatMetricLabel(key)} value={formatMetricValue(key, value)} />
                  ))}
              </div>
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}

function ConditionTimeline({ detail }: { detail: ZapbenchReplayDetail | null }) {
  const timeline = detail?.condition_timeline ?? [];
  const total = timeline.reduce((sum, item) => sum + item.length, 0) || 1;
  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
      <PanelTitle icon={<BarChart3 size={15} />} title="Condition timeline" />
      {!detail ? (
        <EvidencePlaceholder label="Loading ZAPBench condition offsets." />
      ) : (
        <div className="space-y-3">
          <div className="flex h-9 overflow-hidden rounded-md border border-zinc-800 bg-zinc-950">
            {timeline.map((segment) => (
              <div
                key={segment.name}
                className={[
                  "relative min-w-[18px] border-r border-zinc-900",
                  segment.current ? "bg-sky-400" : "bg-zinc-800",
                ].join(" ")}
                style={{ width: `${(segment.length / total) * 100}%` }}
                title={`${segment.name}: rows ${segment.start_row}-${segment.end_row}`}
              >
                {segment.current ? (
                  <div
                    className="absolute inset-y-0 w-px bg-white"
                    style={{ left: `${Math.round((detail.condition.phase ?? 0) * 100)}%` }}
                  />
                ) : null}
              </div>
            ))}
          </div>
          <div className="grid gap-2 text-xs text-zinc-400 sm:grid-cols-2 xl:grid-cols-4">
            <StatusRow label="current condition" value={detail.condition.name} />
            <StatusRow label="row" value={formatInt(detail.row)} />
            <StatusRow
              label="rows in condition"
              value={`${formatInt(detail.condition.start_row)}-${formatInt(detail.condition.end_row)}`}
            />
            <StatusRow label="condition progress" value={formatPercent((detail.condition.phase ?? 0) * 100)} />
          </div>
        </div>
      )}
    </section>
  );
}

function StimulusCovariateViewer({
  covariates,
}: {
  covariates: ZapbenchStimulusCovariates | null;
}) {
  const columns = covariates?.columns ?? [];
  const active = covariates?.active_columns ?? [];
  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
      <PanelTitle icon={<Database size={15} />} title="Stimulus covariates" />
      {!covariates ? (
        <EvidencePlaceholder label="Loading 26-column ZAPBench stimulus vector." />
      ) : !covariates.available ? (
        <EvidencePlaceholder label={covariates.reason || covariates.error || "Stimulus cache unavailable."} />
      ) : (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {(active.length ? active : columns.slice(0, 4)).map((column) => (
              <div
                key={column.index}
                className="rounded-md border border-zinc-800 bg-zinc-900/60 px-2.5 py-1.5 text-xs"
                title={column.description}
              >
                <span className="mr-2 uppercase text-zinc-500">{column.name}</span>
                <span className="font-mono text-zinc-100">{formatMaybeNumber(column.value, 3)}</span>
              </div>
            ))}
          </div>
          <div className="max-h-72 overflow-y-auto rounded-md border border-zinc-800">
            {columns.map((column) => (
              <div
                key={column.index}
                className={[
                  "grid grid-cols-[38px_minmax(130px,0.8fr)_minmax(0,1fr)_86px] gap-2 border-b border-zinc-900 px-3 py-2 text-xs last:border-b-0",
                  column.active ? "bg-sky-400/10" : "bg-zinc-950/40",
                ].join(" ")}
              >
                <span className="font-mono text-zinc-500">{column.index}</span>
                <span className="truncate text-zinc-200" title={column.name}>
                  {column.name}
                </span>
                <span className="truncate text-zinc-500" title={column.description}>
                  {column.group}: {column.description}
                </span>
                <span className="text-right font-mono text-zinc-200">
                  {formatMaybeNumber(column.value, 3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function BrainTracePreview({ preview }: { preview: ZapbenchBrainPreview | null }) {
  const values = preview?.values ?? [];
  const timeCount = values.length;
  const neuronCount = values[0]?.length ?? 0;
  const min = preview?.min ?? 0;
  const max = preview?.max ?? 1;
  const currentColumn = preview?.current_column ?? -1;

  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
      <PanelTitle icon={<LineChart size={15} />} title="Brain activity raster" />
      {!preview ? (
        <EvidencePlaceholder label="Loading sampled calcium traces." />
      ) : !preview.available ? (
        <EvidencePlaceholder label={preview.reason || preview.error || "Trace preview cache unavailable."} />
      ) : (
        <div className="space-y-3">
          <div className="grid gap-2 text-xs text-zinc-400 sm:grid-cols-2 xl:grid-cols-4">
            <StatusRow label="row window" value={`${formatInt(preview.start_row ?? 0)}-${formatInt(preview.end_row ?? 0)}`} />
            <StatusRow label="sampled neurons" value={formatInt(neuronCount)} />
            <StatusRow label="trace min" value={formatMaybeNumber(min, 3)} />
            <StatusRow label="trace max" value={formatMaybeNumber(max, 3)} />
          </div>
          <div className="overflow-hidden rounded-md border border-zinc-800 bg-zinc-950 p-2">
            <div
              className="grid gap-px"
              style={{
                gridTemplateColumns: `repeat(${Math.max(1, timeCount)}, minmax(4px, 1fr))`,
              }}
            >
              {Array.from({ length: neuronCount }).flatMap((_, neuronIndex) =>
                Array.from({ length: timeCount }).map((__, timeIndex) => {
                  const value = values[timeIndex]?.[neuronIndex] ?? 0;
                  const current = timeIndex === currentColumn;
                  return (
                    <div
                      key={`${neuronIndex}-${timeIndex}`}
                      className={current ? "h-1.5 ring-1 ring-inset ring-white/70" : "h-1.5"}
                      style={{ backgroundColor: heatColor(value, min, max) }}
                      title={`neuron ${preview.neuron_ids?.[neuronIndex] ?? neuronIndex}, row ${preview.rows?.[timeIndex] ?? timeIndex}: ${formatMaybeNumber(value, 4)}`}
                    />
                  );
                }),
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function PanelTitle({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <h3 className="mb-3 inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-zinc-300">
      {icon}
      {title}
    </h3>
  );
}

function EvidencePlaceholder({ label }: { label: string }) {
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/40 px-3 py-3 text-xs text-zinc-500">
      {label}
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/50 px-3 py-2">
      <div className="text-[10px] uppercase text-zinc-500">{label}</div>
      <div className="font-mono text-sm text-zinc-100">{value}</div>
    </div>
  );
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
      <div className="truncate font-mono text-zinc-200">{value || "none"}</div>
    </div>
  );
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

function SignalBar({
  label,
  rawLabel,
  value,
}: {
  label: string;
  rawLabel: string;
  value: number;
}) {
  const clamped = clamp(value, 0, 1);
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/50 px-3 py-2">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-[10px] uppercase text-zinc-500">{label}</span>
        <span className="font-mono text-xs text-zinc-300">{rawLabel}</span>
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

function SignedSignalBar({
  label,
  rawLabel,
  value,
}: {
  label: string;
  rawLabel: string;
  value: number;
}) {
  const clamped = clamp(value, -1, 1);
  const width = Math.round(Math.abs(clamped) * 50);
  const left = clamped < 0 ? 50 - width : 50;
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/50 px-3 py-2">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-[10px] uppercase text-zinc-500">{label}</span>
        <span className="font-mono text-xs text-zinc-300">{rawLabel}</span>
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

function formatInt(value: number): string {
  return Number.isFinite(value) ? Math.round(value).toLocaleString("en-US") : "0";
}

function formatNumber(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : (0).toFixed(digits);
}

function formatMaybeNumber(value: number | null | undefined, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "none";
}

function formatSigned(value: number, digits = 3): string {
  if (!Number.isFinite(value)) {
    return `+${(0).toFixed(digits)}`;
  }
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function formatPercent(value: number): string {
  return `${formatNumber(value, 1)}%`;
}

function formatSeconds(value: number): string {
  return Number.isFinite(value) ? value.toFixed(2) : "0.00";
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : 0));
}

function formatMetricLabel(key: string): string {
  return key.replaceAll("_", " ");
}

function formatMetricValue(key: string, value: number): string {
  if (key.includes("accuracy") || key.includes("f1") || key.includes("precision") || key.includes("recall")) {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (key.endsWith("_r2") || key === "force_r2") {
    return value.toFixed(3);
  }
  return value.toFixed(3);
}

function heatColor(value: number, min: number, max: number): string {
  const span = Math.max(1e-6, max - min);
  const normalized = clamp((value - min) / span, 0, 1);
  const lightness = 12 + normalized * 58;
  const saturation = 70 + normalized * 12;
  return `hsl(195, ${saturation.toFixed(0)}%, ${lightness.toFixed(0)}%)`;
}
