export const FISH_BODY_LENGTH_MM = 4.2;
export const FISH_VIEW_LENGTH_MM = 5.0;

const SECTION_COUNT = 18;
const MAX_SEGMENT_YAW_RAD = 0.34;
const MAX_VISUAL_YAW_RAD = 0.72;
const MAX_SEGMENT_PITCH_RAD = 0.22;
const MAX_VISUAL_PITCH_RAD = 0.34;

export interface FishPoint {
  x: number;
  y: number;
  z: number;
  frac: number;
  halfWidth: number;
  halfHeight: number;
  tangentX: number;
  tangentY: number;
  tangentZ: number;
  normalX: number;
  normalY: number;
}

export interface FishGeometry {
  points: FishPoint[];
  outlineLeft: Array<[number, number, number]>;
  outlineRight: Array<[number, number, number]>;
  com: [number, number, number];
}

export interface FishGeometryInput {
  com_mm: [number, number, number];
  heading_rad?: number;
  pitch_rad?: number;
  tail_angles?: ArrayLike<number>;
  tail_pitch_angles?: ArrayLike<number>;
  segments_mm?: ArrayLike<number>;
}

interface RawPoint {
  x: number;
  y: number;
  z: number;
  frac: number;
}

export function buildFishGeometry(latest: FishGeometryInput): FishGeometry {
  const [cx, cy, cz] = latest.com_mm;
  const segmentPoints = rawPointsFromSegments(latest.segments_mm);
  const raw = segmentPoints ?? buildSyntheticRawPoints(latest);

  const shifted = segmentPoints
    ? raw
    : (() => {
        const centroid = weightedCentroid(raw);
        return raw.map((p) => ({
          ...p,
          x: p.x - centroid[0] + cx,
          y: p.y - centroid[1] + cy,
          z: p.z - centroid[2] + cz,
        }));
      })();

  const points: FishPoint[] = shifted.map((p, i) => {
    const prev = shifted[Math.max(0, i - 1)];
    const next = shifted[Math.min(shifted.length - 1, i + 1)];
    let tx = next.x - prev.x;
    let ty = next.y - prev.y;
    let tz = next.z - prev.z;
    const len = Math.hypot(tx, ty, tz) || 1;
    tx /= len;
    ty /= len;
    tz /= len;
    const nLen = Math.hypot(tx, ty) || 1;
    return {
      x: p.x,
      y: p.y,
      z: p.z,
      frac: p.frac,
      halfWidth: fishHalfWidthMm(p.frac),
      halfHeight: fishHalfHeightMm(p.frac),
      tangentX: tx,
      tangentY: ty,
      tangentZ: tz,
      normalX: -ty / nLen,
      normalY: tx / nLen,
    };
  });

  const outlineLeft: Array<[number, number, number]> = [];
  const outlineRight: Array<[number, number, number]> = [];
  for (const p of points) {
    outlineLeft.push([p.x + p.normalX * p.halfWidth, p.y + p.normalY * p.halfWidth, p.z]);
    outlineRight.push([p.x - p.normalX * p.halfWidth, p.y - p.normalY * p.halfWidth, p.z]);
  }

  return { points, outlineLeft, outlineRight, com: [cx, cy, cz] };
}

function buildSyntheticRawPoints(latest: FishGeometryInput): RawPoint[] {
  const heading = finite(latest.heading_rad, fallbackHeading(latest));
  const pitch = clamp(finite(latest.pitch_rad, 0), -MAX_VISUAL_PITCH_RAD, MAX_VISUAL_PITCH_RAD);
  const yawAngles = latest.tail_angles ?? [];
  const pitchAngles = latest.tail_pitch_angles ?? [];
  const segLen = FISH_BODY_LENGTH_MM / SECTION_COUNT;
  const raw: RawPoint[] = [{ x: 0, y: 0, z: 0, frac: 0 }];

  let visualYaw = 0;
  let visualPitch = -pitch;
  for (let i = 1; i <= SECTION_COUNT; i++) {
    const frac = i / SECTION_COUNT;
    const yawInput = clamp(finite(yawAngles[i - 1], 0), -MAX_SEGMENT_YAW_RAD, MAX_SEGMENT_YAW_RAD);
    const pitchInput = clamp(
      finite(pitchAngles[i - 1], 0),
      -MAX_SEGMENT_PITCH_RAD,
      MAX_SEGMENT_PITCH_RAD,
    );
    visualYaw = clamp(
      visualYaw + yawInput * (0.22 + 0.38 * frac),
      -MAX_VISUAL_YAW_RAD,
      MAX_VISUAL_YAW_RAD,
    );
    visualPitch = clamp(
      visualPitch + pitchInput * (0.14 + 0.22 * frac),
      -MAX_VISUAL_PITCH_RAD,
      MAX_VISUAL_PITCH_RAD,
    );

    const posteriorYaw = heading + Math.PI + visualYaw;
    const cPitch = Math.cos(visualPitch);
    const prev = raw[i - 1];
    raw.push({
      x: prev.x + Math.cos(posteriorYaw) * cPitch * segLen,
      y: prev.y + Math.sin(posteriorYaw) * cPitch * segLen,
      z: prev.z + Math.sin(visualPitch) * segLen,
      frac,
    });
  }
  return raw;
}

function rawPointsFromSegments(segments: ArrayLike<number> | undefined): RawPoint[] | null {
  if (!segments || segments.length < 6) return null;
  const n = Math.floor(segments.length / 3);
  const coords: Array<[number, number, number]> = [];
  for (let i = 0; i < n; i++) {
    const x = Number(segments[i * 3]);
    const y = Number(segments[i * 3 + 1]);
    const z = Number(segments[i * 3 + 2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return null;
    }
    const prev = coords[coords.length - 1];
    if (prev && Math.hypot(x - prev[0], y - prev[1], z - prev[2]) < 1e-6) {
      continue;
    }
    coords.push([x, y, z]);
  }
  if (coords.length < 2) return null;

  const cumulative = [0];
  for (let i = 1; i < coords.length; i++) {
    const prev = coords[i - 1];
    const cur = coords[i];
    cumulative.push(
      cumulative[i - 1] +
        Math.hypot(cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]),
    );
  }
  const total = cumulative[cumulative.length - 1];
  if (!Number.isFinite(total) || total <= 1e-6) return null;

  return coords.map((p, i) => ({
    x: p[0],
    y: p[1],
    z: p[2],
    frac: cumulative[i] / total,
  }));
}

export function buildNeutralFishGeometry(): FishGeometry {
  return buildFishGeometry({
    com_mm: [0, 0, 0],
    heading_rad: 0,
    pitch_rad: 0,
    tail_angles: [],
    tail_pitch_angles: [],
    segments_mm: [],
  });
}

function weightedCentroid(points: RawPoint[]): [number, number, number] {
  let sx = 0;
  let sy = 0;
  let sz = 0;
  let sw = 0;
  for (const p of points) {
    const w = Math.max(0.01, fishHalfWidthMm(p.frac) ** 2);
    sx += p.x * w;
    sy += p.y * w;
    sz += p.z * w;
    sw += w;
  }
  return sw > 0 ? [sx / sw, sy / sw, sz / sw] : [0, 0, 0];
}

function fallbackHeading(latest: FishGeometryInput): number {
  const seg = latest.segments_mm;
  if (seg && seg.length >= 6) {
    const last = seg.length - 3;
    return Math.atan2(seg[1] - seg[last + 1], seg[0] - seg[last]);
  }
  return 0;
}

function fishHalfWidthMm(frac: number): number {
  if (frac < 0.16) return lerp(0.255, 0.225, frac / 0.16);
  if (frac < 0.54) return lerp(0.225, 0.155, (frac - 0.16) / 0.38);
  return lerp(0.155, 0.024, (frac - 0.54) / 0.46);
}

function fishHalfHeightMm(frac: number): number {
  if (frac < 0.18) return lerp(0.095, 0.12, frac / 0.18);
  if (frac < 0.58) return lerp(0.12, 0.085, (frac - 0.18) / 0.4);
  return lerp(0.085, 0.018, (frac - 0.58) / 0.42);
}

function finite(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function lerp(a: number, b: number, t: number): number {
  const tt = clamp(t, 0, 1);
  return a + (b - a) * tt;
}
