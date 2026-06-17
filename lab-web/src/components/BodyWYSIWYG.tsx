import { useEffect, useRef } from "react";
import type { MouseEvent } from "react";
import {
  FISH_VIEW_LENGTH_MM,
  buildNeutralFishGeometry,
  type FishGeometry,
  type FishPoint,
} from "../rendering/fish-body";
import { useBodyStore } from "../state/body";
import { useLabStore } from "../state/store";

const MUSCLE_NAME_RE = /^tail_(\d+)_(left|right|dorsal|ventral)$/;
const TAIL_MOTOR_START_FRAC = 0.22;
const TAIL_MOTOR_END_FRAC = 0.985;
const STATIC_ANATOMY_GEOMETRY = buildNeutralFishGeometry();

type Side = "L" | "R" | "D" | "V";
type CanvasPoint = [number, number];

interface MotorActuator {
  id: number;
  name: string;
}

type MuscleLookup = Record<number, Partial<Record<Side, MotorActuator>>>;

interface HitRegion {
  id: number;
  name: string;
  polygon: CanvasPoint[];
}

interface Transform {
  scale: number;
  toScreen: (p: [number, number, number]) => CanvasPoint;
}

const SIDE_BY_NAME = {
  left: "L",
  right: "R",
  dorsal: "D",
  ventral: "V",
} as const;

function buildLookup(actuators: { id: number; name: string }[]): {
  lookup: MuscleLookup;
  segmentCount: number;
} {
  const lookup: MuscleLookup = {};
  let maxSeg = -1;
  for (const actuator of actuators) {
    const m = MUSCLE_NAME_RE.exec(actuator.name);
    if (!m) continue;
    const seg = Number(m[1]);
    const side = SIDE_BY_NAME[m[2] as keyof typeof SIDE_BY_NAME];
    if (!lookup[seg]) lookup[seg] = {};
    lookup[seg][side] = { id: actuator.id, name: actuator.name };
    maxSeg = Math.max(maxSeg, seg);
  }
  return { lookup, segmentCount: maxSeg + 1 };
}

function activationIntensity(a: number): number {
  return Math.min(1, Math.max(0, Math.abs(a)));
}

function activationFill(
  a: number,
  base: [number, number, number],
  alpha = 0.72,
): string {
  const t = activationIntensity(a);
  const target: [number, number, number] = a >= 0 ? [242, 128, 72] : [74, 166, 230];
  const r = Math.round(base[0] + (target[0] - base[0]) * t);
  const g = Math.round(base[1] + (target[1] - base[1]) * t);
  const b = Math.round(base[2] + (target[2] - base[2]) * t);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function buildTransform(geometry: FishGeometry, width: number, height: number): Transform {
  const origin = geometry.com;
  const head = geometry.points[0];
  const tail = geometry.points[geometry.points.length - 1];
  const heading =
    head && tail ? Math.atan2(head.y - tail.y, head.x - tail.x) : 0;
  const rotate = Math.PI - heading;
  const cos = Math.cos(rotate);
  const sin = Math.sin(rotate);
  const toBodyFrame = (p: [number, number, number]): [number, number, number] => {
    const dx = p[0] - origin[0];
    const dy = p[1] - origin[1];
    return [dx * cos - dy * sin, dx * sin + dy * cos, p[2] - origin[2]];
  };

  const all = [...geometry.outlineLeft, ...geometry.outlineRight].map(toBodyFrame);
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of all) {
    minX = Math.min(minX, p[0]);
    maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]);
    maxY = Math.max(maxY, p[1]);
  }
  if (!Number.isFinite(minX)) {
    minX = -2;
    maxX = 2;
    minY = -0.5;
    maxY = 0.5;
  }

  const modelWidth = Math.max(FISH_VIEW_LENGTH_MM, maxX - minX + 0.5);
  const modelHeight = Math.max(1.25, maxY - minY + 0.35);
  const scale = Math.max(
    1,
    Math.min(
      Math.max(20, width - 76) / modelWidth,
      Math.max(20, height - 52) / modelHeight,
    ),
  );
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  return {
    scale,
    toScreen: (p) => {
      const local = toBodyFrame(p);
      return [width * 0.5 + (local[0] - cx) * scale, height * 0.5 - (local[1] - cy) * scale];
    },
  };
}

function tracePolygon(
  ctx: CanvasRenderingContext2D,
  polygon: CanvasPoint[],
): void {
  if (!polygon.length) return;
  ctx.beginPath();
  ctx.moveTo(polygon[0][0], polygon[0][1]);
  for (let i = 1; i < polygon.length; i++) ctx.lineTo(polygon[i][0], polygon[i][1]);
  ctx.closePath();
}

function drawPolygon(
  ctx: CanvasRenderingContext2D,
  polygon: CanvasPoint[],
  fill: string,
  stroke?: string,
  lineWidth = 1,
): void {
  tracePolygon(ctx, polygon);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }
}

function pointInPolygon(point: CanvasPoint, polygon: CanvasPoint[]): boolean {
  const [px, py] = point;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    const crosses = yi > py !== yj > py;
    if (!crosses) continue;
    const atX = ((xj - xi) * (py - yi)) / (yj - yi || 1e-9) + xi;
    if (px < atX) inside = !inside;
  }
  return inside;
}

function samplePoint(geometry: FishGeometry, frac: number): FishPoint {
  const points = geometry.points;
  if (!points.length) {
    return {
      x: 0,
      y: 0,
      z: 0,
      frac: 0,
      halfWidth: 0.2,
      halfHeight: 0.08,
      tangentX: -1,
      tangentY: 0,
      tangentZ: 0,
      normalX: 0,
      normalY: -1,
    };
  }
  const clamped = Math.max(0, Math.min(1, frac));
  const scaled = clamped * (points.length - 1);
  const lo = Math.floor(scaled);
  const hi = Math.min(points.length - 1, lo + 1);
  const t = scaled - lo;
  const a = points[lo];
  const b = points[hi];
  const tangentX = lerp(a.tangentX, b.tangentX, t);
  const tangentY = lerp(a.tangentY, b.tangentY, t);
  const tangentZ = lerp(a.tangentZ, b.tangentZ, t);
  const normalX = lerp(a.normalX, b.normalX, t);
  const normalY = lerp(a.normalY, b.normalY, t);
  const normalLen = Math.hypot(normalX, normalY) || 1;
  const tangentLen = Math.hypot(tangentX, tangentY, tangentZ) || 1;
  return {
    x: lerp(a.x, b.x, t),
    y: lerp(a.y, b.y, t),
    z: lerp(a.z, b.z, t),
    frac: clamped,
    halfWidth: lerp(a.halfWidth, b.halfWidth, t),
    halfHeight: lerp(a.halfHeight, b.halfHeight, t),
    tangentX: tangentX / tangentLen,
    tangentY: tangentY / tangentLen,
    tangentZ: tangentZ / tangentLen,
    normalX: normalX / normalLen,
    normalY: normalY / normalLen,
  };
}

function offsetByNormal(p: FishPoint, mm: number): [number, number, number] {
  return [p.x + p.normalX * mm, p.y + p.normalY * mm, p.z];
}

function yawMusclePolygon(
  geometry: FishGeometry,
  transform: Transform,
  startFrac: number,
  endFrac: number,
  side: "L" | "R",
): CanvasPoint[] {
  const a = samplePoint(geometry, startFrac);
  const b = samplePoint(geometry, endFrac);
  // FishGeometry tangents run from head to tail, so anatomical left is -normal.
  const sign = side === "L" ? -1 : 1;
  const outerA = transform.toScreen(offsetByNormal(a, sign * a.halfWidth * 0.96));
  const outerB = transform.toScreen(offsetByNormal(b, sign * b.halfWidth * 0.96));
  const innerB = transform.toScreen(offsetByNormal(b, sign * b.halfWidth * 0.18));
  const innerA = transform.toScreen(offsetByNormal(a, sign * a.halfWidth * 0.18));
  return [outerA, outerB, innerB, innerA];
}

function pitchMarkerPolygon(
  geometry: FishGeometry,
  transform: Transform,
  frac: number,
  side: "D" | "V",
): CanvasPoint[] {
  const center = transform.toScreen(pointTuple(samplePoint(geometry, frac)));
  const r = Math.max(3.5, Math.min(7.5, transform.scale * 0.036));
  if (side === "D") {
    return [
      [center[0], center[1] - r * 1.45],
      [center[0] - r, center[1] - r * 0.1],
      [center[0] + r, center[1] - r * 0.1],
    ];
  }
  return [
    [center[0], center[1] + r * 1.45],
    [center[0] - r, center[1] + r * 0.1],
    [center[0] + r, center[1] + r * 0.1],
  ];
}

function pointTuple(p: FishPoint): [number, number, number] {
  return [p.x, p.y, p.z];
}

function drawOrientedEllipse(
  ctx: CanvasRenderingContext2D,
  transform: Transform,
  center: FishPoint,
  rxMm: number,
  ryMm: number,
  fill: string,
  stroke?: string,
): void {
  const [sx, sy] = transform.toScreen(pointTuple(center));
  const tangentScreen = transform.toScreen([
    center.x + center.tangentX,
    center.y + center.tangentY,
    center.z,
  ]);
  const angle = Math.atan2(tangentScreen[1] - sy, tangentScreen[0] - sx);
  ctx.beginPath();
  ctx.ellipse(
    sx,
    sy,
    Math.max(1, rxMm * transform.scale),
    Math.max(1, ryMm * transform.scale),
    angle,
    0,
    Math.PI * 2,
  );
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function drawSegmentRib(
  ctx: CanvasRenderingContext2D,
  geometry: FishGeometry,
  transform: Transform,
  frac: number,
): void {
  const p = samplePoint(geometry, frac);
  const a = transform.toScreen(offsetByNormal(p, -p.halfWidth * 0.92));
  const b = transform.toScreen(offsetByNormal(p, p.halfWidth * 0.92));
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
}

function drawEye(
  ctx: CanvasRenderingContext2D,
  transform: Transform,
  p: FishPoint,
  side: "L" | "R",
): void {
  const sign = side === "L" ? -1 : 1;
  const center = transform.toScreen(offsetByNormal(p, sign * p.halfWidth * 0.58));
  const r = Math.max(3.2, p.halfWidth * transform.scale * 0.22);
  ctx.beginPath();
  ctx.arc(center[0], center[1], r, 0, Math.PI * 2);
  ctx.fillStyle = "#05070a";
  ctx.fill();
  ctx.strokeStyle = "rgba(210, 235, 235, 0.55)";
  ctx.lineWidth = 1;
  ctx.stroke();
}

function isSelected(selectionId: number | null, id: number): boolean {
  return selectionId === id;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function BodyWYSIWYG() {
  const view = useBodyStore((s) => s.view);
  const selection = useBodyStore((s) => s.selection);
  const select = useBodyStore((s) => s.select);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const hitRegionsRef = useRef<HitRegion[]>([]);

  useEffect(() => {
    if (!view) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { lookup, segmentCount } = buildLookup(view.actuators);

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const W = rect.width;
      const H = rect.height;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#07090d";
      ctx.fillRect(0, 0, W, H);

      const latest = useLabStore.getState().latest;
      const geometry = STATIC_ANATOMY_GEOMETRY;
      const transform = buildTransform(geometry, W, H);
      const ma = latest?.ma ?? null;
      const selectedId =
        selection && selection.kind === "muscle" ? selection.id : null;
      const hitRegions: HitRegion[] = [];

      const outline = [
        ...geometry.outlineLeft.map(transform.toScreen),
        ...geometry.outlineRight.slice().reverse().map(transform.toScreen),
      ];
      drawPolygon(ctx, outline, "#c7e4e0", "rgba(220, 250, 248, 0.58)", 1.2);

      drawOrientedEllipse(
        ctx,
        transform,
        samplePoint(geometry, 0.13),
        0.45,
        0.23,
        "rgba(214, 238, 234, 0.45)",
      );
      drawOrientedEllipse(
        ctx,
        transform,
        samplePoint(geometry, 0.25),
        0.34,
        0.13,
        "rgba(235, 180, 78, 0.62)",
        "rgba(255, 221, 140, 0.36)",
      );
      drawOrientedEllipse(
        ctx,
        transform,
        samplePoint(geometry, 0.31),
        0.23,
        0.07,
        "rgba(160, 220, 232, 0.48)",
      );

      const tailSpan = TAIL_MOTOR_END_FRAC - TAIL_MOTOR_START_FRAC;
      const nSegments = Math.max(1, segmentCount);
      for (let seg = 0; seg < nSegments; seg++) {
        const startFrac = TAIL_MOTOR_START_FRAC + (tailSpan * seg) / nSegments;
        const endFrac = TAIL_MOTOR_START_FRAC + (tailSpan * (seg + 1)) / nSegments;
        const midFrac = (startFrac + endFrac) * 0.5;
        const ids = lookup[seg] ?? {};

        for (const side of ["L", "R"] as const) {
          const motor = ids[side];
          if (!motor) continue;
          const activation = ma ? ma[motor.id] ?? 0 : 0;
          const polygon = yawMusclePolygon(geometry, transform, startFrac, endFrac, side);
          hitRegions.push({ id: motor.id, name: motor.name, polygon });
          drawPolygon(
            ctx,
            polygon,
            activationFill(activation, side === "L" ? [65, 76, 84] : [58, 69, 78], 0.68),
            isSelected(selectedId, motor.id) ? "#7ab6ff" : undefined,
            2,
          );
        }

        for (const side of ["D", "V"] as const) {
          const motor = ids[side];
          if (!motor) continue;
          const activation = ma ? ma[motor.id] ?? 0 : 0;
          const polygon = pitchMarkerPolygon(geometry, transform, midFrac, side);
          hitRegions.push({ id: motor.id, name: motor.name, polygon });
          drawPolygon(
            ctx,
            polygon,
            activationFill(activation, side === "D" ? [42, 63, 86] : [76, 62, 45], 0.76),
            isSelected(selectedId, motor.id) ? "#7ab6ff" : "rgba(7, 9, 13, 0.45)",
            isSelected(selectedId, motor.id) ? 2 : 1,
          );
        }
      }

      ctx.strokeStyle = "rgba(35, 51, 59, 0.68)";
      ctx.lineWidth = 1;
      for (let seg = 0; seg <= nSegments; seg++) {
        const frac = TAIL_MOTOR_START_FRAC + (tailSpan * seg) / nSegments;
        drawSegmentRib(ctx, geometry, transform, frac);
      }

      ctx.strokeStyle = "rgba(230, 255, 252, 0.72)";
      ctx.lineWidth = 1.25;
      tracePolygon(ctx, outline);
      ctx.stroke();

      drawEye(ctx, transform, samplePoint(geometry, 0.055), "L");
      drawEye(ctx, transform, samplePoint(geometry, 0.055), "R");

      hitRegionsRef.current = hitRegions;
      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [view, selection]);

  const handleClick = (evt: MouseEvent<HTMLCanvasElement>) => {
    if (!view) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const point: CanvasPoint = [evt.clientX - rect.left, evt.clientY - rect.top];
    const regions = hitRegionsRef.current;
    for (let i = regions.length - 1; i >= 0; i--) {
      const region = regions[i];
      if (!pointInPolygon(point, region.polygon)) continue;
      if (selection && selection.kind === "muscle" && selection.id === region.id) {
        select(null);
      } else {
        select({ kind: "muscle", id: region.id, name: region.name });
      }
      return;
    }
  };

  if (!view) {
    return (
      <div className="flex h-full items-center justify-center text-zinc-500 text-xs">
        Loading body…
      </div>
    );
  }

  return (
    <div className="relative h-full w-full">
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        className="h-full w-full cursor-pointer"
      />
    </div>
  );
}
