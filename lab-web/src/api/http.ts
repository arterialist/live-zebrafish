import axios, { type AxiosInstance } from "axios";

export const http: AxiosInstance = axios.create({
  baseURL: "/api",
  timeout: 10_000,
  headers: { "Content-Type": "application/json" },
});

export interface TransportState {
  running: boolean;
  tick: number;
}

export type TransportAction = "play" | "pause" | "step";

export async function getTransport(): Promise<TransportState> {
  const r = await http.get<TransportState>("/sim/transport");
  return r.data;
}

export async function setTransport(action: TransportAction): Promise<TransportState> {
  const r = await http.post<TransportState>("/sim/transport", { action });
  return r.data;
}

/** Wall-clock pacing: ``0`` = no added delay (run as fast as the CPU allows). */
export interface SimPacing {
  real_ms_per_physics_step: number;
  real_ms_per_neural_tick: number;
}

export async function getPacing(): Promise<SimPacing> {
  const r = await http.get<SimPacing>("/sim/pacing");
  return r.data;
}

export async function setPacing(patch: {
  real_ms_per_physics_step?: number;
  real_ms_per_neural_tick?: number;
}): Promise<SimPacing> {
  const r = await http.post<SimPacing>("/sim/pacing", patch);
  return r.data;
}

export interface ParameterSpec {
  path: string;
  label: string;
  group: string;
  kind: "int" | "float" | "bool" | "enum" | "vec";
  apply: "live" | "rebuild";
  min: number | null;
  max: number | null;
  step: number | null;
  enum: string[] | null;
  help: string;
  value: unknown;
}

export interface SchemaResponse {
  specs: ParameterSpec[];
  pending: Record<string, unknown>;
}

export async function getSchema(): Promise<SchemaResponse> {
  const r = await http.get<SchemaResponse>("/schema");
  return r.data;
}

export interface NeuronInfo {
  id: number;
  name: string;
  type: string;
  class: "s" | "m" | "i" | "u";
  degree_in_chem: number;
  degree_out_chem: number;
  degree_in_gap: number;
  degree_out_gap: number;
  layout_x: number;
  layout_y: number;
}

export interface EdgeInfo {
  pre_id: number;
  post_id: number;
  type: "chemical" | "gap";
  weight: number;
}

export async function getConnectome(): Promise<{ neurons: NeuronInfo[]; edges: EdgeInfo[] }> {
  const r = await http.get<{ neurons: NeuronInfo[]; edges: EdgeInfo[] }>("/connectome");
  return r.data;
}

export interface NeuronPostsynapticView {
  id: number;
  pre_paula_id: number | null;
  pre_terminal: number | null;
  pre_name: string | null;
  info: number;
  plast: number;
  adapt: number[];
  potential: number;
}

export interface NeuronPresynapticView {
  id: number;
  u_o_info: number;
  u_o_mod: number[];
  u_i_retro: number;
}

export interface NeuronDetail {
  name: string;
  paula_id: number;
  S: number;
  O: number;
  r: number;
  b: number;
  t_ref: number;
  F_avg: number;
  t_last_fire: number | null;
  M_vector: number[];
  pq_len: number;
  postsynaptic: NeuronPostsynapticView[];
  presynaptic: NeuronPresynapticView[];
  params: {
    r_base: number;
    b_base: number;
    c: number;
    lambda_param: number;
    p: number;
    eta_post: number;
    eta_retro: number;
    delta_decay: number;
    beta_avg: number;
    gamma: number[];
    w_r: number[];
    w_b: number[];
    w_tref: number[];
    num_neuromodulators: number;
    num_inputs: number;
  };
}

export async function getNeuron(name: string): Promise<NeuronDetail> {
  const r = await http.get<NeuronDetail>(`/neurons/${encodeURIComponent(name)}`);
  return r.data;
}

export interface NeuronFieldPatch {
  field: string;
  value: unknown;
  index?: number;
  subfield?: string;
  vec_index?: number;
}

export async function patchNeuron(
  name: string,
  patches: NeuronFieldPatch[],
): Promise<{ applied: unknown[]; failed: { field: string; error: string }[] }> {
  const r = await http.post(`/neurons/${encodeURIComponent(name)}/patch`, {
    patches,
  });
  return r.data;
}

export interface BodyView {
  opt: { timestep: number; gravity: number[]; viscosity: number; density: number };
  bodies: { id: number; name: string; mass: number; inertia: number[] }[];
  joints: {
    id: number;
    name: string;
    range: number[];
    damping: number;
    armature: number;
  }[];
  actuators: {
    id: number;
    name: string;
    forcerange: number[];
    gear: number[];
    target_joint_id: number;
  }[];
  sensors: { id: number; name: string; dim: number }[];
  contact_pairs: {
    id: number;
    friction: number[];
    solref: number[];
    solimp: number[];
  }[];
}

export async function getBody(): Promise<BodyView> {
  const r = await http.get<BodyView>("/body");
  return r.data;
}

export interface VideoStimulusState {
  enabled: boolean;
  gain: number;
  has_frame: boolean;
  file_name: string;
  frame_index: number;
  video_time_s: number;
  received_tick: number | null;
  age_ticks: number | null;
  manual_advance_ticks?: number;
  manual_sample_hz?: number;
  visual_left: number;
  visual_right: number;
  optic_flow_left: number;
  optic_flow_right: number;
  lateral_line_left: number;
  lateral_line_right: number;
  visual_up: number;
  visual_down: number;
  light_level: number;
  startle: number;
  motion_energy: number;
  asymmetry: number;
  action_kick: number;
  action_force: number;
  action_side_score: number;
  action_kick_score: number;
  action_confidence: number;
  zapbench_row: number;
  backend_extracted: boolean;
}

export interface VideoStimulusFrame {
  file_name: string;
  frame_index: number;
  video_time_s: number;
  sample_hz?: number;
  visual_left: number;
  visual_right: number;
  optic_flow_left: number;
  optic_flow_right: number;
  lateral_line_left: number;
  lateral_line_right: number;
  visual_up: number;
  visual_down: number;
  light_level: number;
  startle: number;
  motion_energy: number;
  asymmetry: number;
  enabled?: boolean;
}

export interface VideoStimulusSample {
  slug: string;
  file_name: string;
  url: string;
  bytes: number;
  uploaded?: boolean;
}

export interface BackendVideoStimulusDiagnostics {
  backend: string;
  true_optical_flow: boolean;
  camera_stabilized: boolean;
  flow_coherence: number;
  residual_flow_coherence: number;
  camera_motion: number;
  camera_shake: number;
  compression_noise: number;
  contrast: number;
  flow_reliability: number;
  global_horizontal_flow: number;
  global_vertical_flow: number;
  affine_inlier_ratio: number;
  zapbench_grounded: boolean;
  zapbench_row: number;
  zapbench_distance: number;
  zapbench_neighbor_rows: number[];
  zapbench_neighbor_distance_mean: number;
  zapbench_vector?: number[];
}

export interface BackendVideoStimulusFrameResult {
  state: VideoStimulusState;
  features: VideoStimulusFrame & {
    backend_extracted: boolean;
    action_kick: number;
    action_force: number;
    action_side_score: number;
    action_kick_score: number;
    action_confidence: number;
    zapbench_row: number;
  };
  diagnostics: BackendVideoStimulusDiagnostics;
}

export async function getVideoStimulus(): Promise<VideoStimulusState> {
  const r = await http.get<VideoStimulusState>("/video-stimulus");
  return r.data;
}

export async function getVideoStimulusSamples(): Promise<VideoStimulusSample[]> {
  const r = await http.get<{ samples: VideoStimulusSample[] }>("/video-stimulus/samples");
  return r.data.samples;
}

export async function uploadVideoStimulus(file: File): Promise<VideoStimulusSample> {
  const r = await http.post<VideoStimulusSample>("/video-stimulus/upload", file, {
    params: { file_name: file.name },
    headers: { "Content-Type": file.type || "application/octet-stream" },
  });
  return r.data;
}

export async function setVideoStimulusConfig(patch: {
  enabled?: boolean;
  gain?: number;
  file_name?: string;
}): Promise<VideoStimulusState> {
  const r = await http.post<VideoStimulusState>("/video-stimulus/config", patch);
  return r.data;
}

export async function postVideoStimulusFrame(
  frame: VideoStimulusFrame,
): Promise<VideoStimulusState> {
  const r = await http.post<VideoStimulusState>("/video-stimulus/frame", frame);
  return r.data;
}

export async function postBackendVideoStimulusFrame(frame: {
  file_name: string;
  frame_index: number;
  video_time_s: number;
  sample_hz?: number;
  enabled?: boolean;
}): Promise<BackendVideoStimulusFrameResult> {
  const r = await http.post<BackendVideoStimulusFrameResult>("/video-stimulus/backend-frame", frame);
  return r.data;
}

export async function clearVideoStimulus(): Promise<VideoStimulusState> {
  const r = await http.post<VideoStimulusState>("/video-stimulus/clear");
  return r.data;
}

export interface CalciumReplayState {
  available: boolean;
  enabled: boolean;
  loop: boolean;
  gain: number;
  condition: string;
  path: string;
  source: string;
  frame_index: number;
  row: number;
  calcium_time_s: number;
  kick: number;
  side: "none" | "left" | "right" | string;
  side_score: number;
  force: number;
  kick_score: number;
  confidence: number;
  frames: number;
  selected_frames: number;
  condition_names: string[];
  error: string;
}

export interface CalciumStimulusState {
  enabled: boolean;
  gain: number;
  has_frame: boolean;
  source: string;
  row: number;
  frame_index: number;
  calcium_time_s: number;
  received_tick: number | null;
  age_ticks: number | null;
  kick: number;
  side: "none" | "left" | "right" | string;
  side_score: number;
  force: number;
  kick_score: number;
  confidence: number;
  replay: CalciumReplayState;
}

export async function getCalciumStimulus(): Promise<CalciumStimulusState> {
  const r = await http.get<CalciumStimulusState>("/calcium-stimulus");
  return r.data;
}

export async function setCalciumReplay(patch: {
  enabled?: boolean;
  gain?: number;
  condition?: string;
  loop?: boolean;
  replay_path?: string;
}): Promise<CalciumReplayState> {
  const r = await http.post<CalciumReplayState>("/calcium-stimulus/replay", patch);
  return r.data;
}

export async function clearCalciumStimulus(): Promise<CalciumReplayState> {
  const r = await http.post<CalciumReplayState>("/calcium-stimulus/clear");
  return r.data;
}

export interface ZapbenchThresholdMetrics {
  kick_accuracy: number;
  kick_f1: number;
  kick_precision: number;
  kick_recall: number;
  side_accuracy: number;
  side_active_accuracy: number;
  side_macro_accuracy: number;
  side_non_none_rate_pred: number;
  side_non_none_rate_true: number;
}

export interface ZapbenchSummary {
  sources: { label: string; url: string }[];
  bucket: Record<string, string>;
  condition_names: string[];
  condition_offsets: number[];
  connectome_public: boolean;
  connectome_note: string;
  linear_direct: {
    metrics: {
      kick_accuracy: number;
      force_r2: number;
      force_mae: number;
      side_accuracy: number;
      direct_extra?: {
        kick_f1: number;
        side_active_accuracy: number;
      };
      thresholds?: {
        default?: {
          thresholds: Record<string, number>;
          test: ZapbenchThresholdMetrics;
        };
        calibrated?: {
          thresholds: Record<string, number>;
          train: ZapbenchThresholdMetrics;
          test: ZapbenchThresholdMetrics;
        };
      };
    } | null;
    artifact: {
      path: string;
      exists: boolean;
      context?: number;
      neurons?: number;
      projection_components?: number;
      metadata?: Record<string, unknown>;
      thresholds?: Record<string, number>;
      error?: string;
    };
    lag0_artifact: { path: string; exists: boolean; error?: string };
    labels: {
      path: string;
      exists: boolean;
      frames?: number;
      kick_rate?: number;
      force_mean?: number;
      force_p95?: number;
      side_counts?: { none: number; left: number; right: number };
      samples?: { row: number; force: number; kick: boolean; side: number }[];
      error?: string;
    };
    report_path: string;
  };
  stimulus_fallback: {
    metrics: Record<string, unknown> | null;
    artifact: { path: string; exists: boolean; error?: string };
  };
  nonlinear_experimental: {
    metrics: {
      metrics?: Record<string, Record<string, number>>;
    } | null;
    artifact_path: string;
  };
  research_notes: string[];
}

export async function getZapbenchSummary(): Promise<ZapbenchSummary> {
  const r = await http.get<ZapbenchSummary>("/zapbench");
  return r.data;
}

export interface ZapbenchConditionSegment {
  id: number;
  name: string;
  start_row: number;
  end_row: number;
  length: number;
  current?: boolean;
  phase?: number;
}

export interface ZapbenchEphysTruth {
  available: boolean;
  path: string;
  row?: number;
  left_power?: number;
  right_power?: number;
  total_power?: number;
  kick?: boolean;
  side?: number;
  side_class?: number;
  side_name?: string;
  force?: number;
  raw_start_sample?: number;
  raw_end_sample?: number;
  raw_window_samples?: number;
  reason?: string;
  error?: string;
}

export interface ZapbenchActionRow {
  available: boolean;
  path: string;
  artifact_path?: string;
  row?: number;
  calcium_time_s?: number;
  kick?: boolean | number;
  kick_score?: number;
  side_score?: number;
  side_class?: number;
  side_name?: string;
  force?: number;
  confidence?: number;
  is_train_row?: boolean;
  is_test_row?: boolean;
  thresholds?: Record<string, number>;
  metadata?: Record<string, unknown>;
  reason?: string;
  error?: string;
}

export interface ZapbenchStimulusColumn {
  index: number;
  name: string;
  group: string;
  description: string;
  value: number;
  active: boolean;
}

export interface ZapbenchStimulusCovariates {
  available: boolean;
  path: string;
  source?: string;
  row?: number;
  columns?: ZapbenchStimulusColumn[];
  active_columns?: ZapbenchStimulusColumn[];
  reason?: string;
  error?: string;
}

export interface ZapbenchBrainPreview {
  available: boolean;
  path: string;
  source?: string;
  artifact_path?: string;
  row?: number;
  start_row?: number;
  end_row?: number;
  current_column?: number;
  neuron_ids?: number[];
  rows?: number[];
  values?: number[][];
  min?: number;
  max?: number;
  shape?: number[];
  reason?: string;
  error?: string;
}

export interface ZapbenchModelCard {
  label: string;
  metrics: Record<string, number | null | undefined>;
  condition_metrics?: Record<string, number>;
  nonlinear_experimental?: Record<string, unknown>;
  artifact: ZapbenchSummary["linear_direct"]["artifact"];
  label_artifact: ZapbenchSummary["linear_direct"]["labels"];
  decoder_note: string;
}

export interface ZapbenchReplayDetail {
  row: number;
  condition: ZapbenchConditionSegment;
  condition_timeline: ZapbenchConditionSegment[];
  ephys_truth: ZapbenchEphysTruth;
  replay_action: ZapbenchActionRow;
  decoder_prediction: ZapbenchActionRow;
  stimulus_covariates: ZapbenchStimulusCovariates;
  brain_preview: ZapbenchBrainPreview;
  model_card: ZapbenchModelCard;
}

export async function getZapbenchReplayDetail(row?: number): Promise<ZapbenchReplayDetail> {
  const r = await http.get<ZapbenchReplayDetail>("/zapbench/replay-detail", {
    params: typeof row === "number" ? { row } : undefined,
  });
  return r.data;
}

export interface MuscleDetail {
  name: string;
  id: number;
  ctrl: number;
  activation: number;
  forcerange: number[];
  gear: number[];
  target_joint_id: number;
}

export async function getMuscle(name: string): Promise<MuscleDetail> {
  const r = await http.get<MuscleDetail>(`/muscles/${encodeURIComponent(name)}`);
  return r.data;
}

export type BodyPatchTarget = "joint" | "actuator" | "body" | "pair" | "opt";

export interface BodyPatch {
  target: BodyPatchTarget;
  field: string;
  value: unknown;
  id?: number;
  index?: number;
}

export interface BodyPatchResult {
  applied: BodyPatch[];
  failed: (BodyPatch & { error: string })[];
}

export async function patchBody(patches: BodyPatch[]): Promise<BodyPatchResult> {
  const r = await http.post<BodyPatchResult>("/body/patch", { patches });
  return r.data;
}

export interface Patch {
  path: string;
  value: unknown;
}

export interface PatchResult {
  applied: string[];
  pending: string[];
  failed: { path: string; error: string }[];
}

export async function postPatches(patches: Patch[]): Promise<PatchResult> {
  const r = await http.post<PatchResult>("/patch", { patches });
  return r.data;
}

export interface ApplyPendingResult {
  applied: string[];
  failed: { path: string; error: string }[];
}

export async function applyPending(): Promise<ApplyPendingResult> {
  const r = await http.post<ApplyPendingResult>("/apply-pending");
  return r.data;
}

export async function resetSim(): Promise<{ ok: boolean }> {
  const r = await http.post<{ ok: boolean }>("/reset");
  return r.data;
}
