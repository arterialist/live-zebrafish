import { useEffect, useRef, useState, type PointerEvent, type WheelEvent } from "react";
import {
  FISH_VIEW_LENGTH_MM,
  buildFishGeometry,
  type FishPoint,
} from "../rendering/fish-body";
import { useLabStore } from "../state/store";
import { useAppSettings } from "../state/app-settings";

const ZOOM_MIN = 0.5;
const ZOOM_MAX = 1.0;

export function ZebrafishCanvas() {
  const latest = useLabStore((s) => s.latest);
  const lockCameraOnSubject = useAppSettings((s) => s.lockCameraOnSubject);
  const showGrid = useAppSettings((s) => s.showGrid);
  const showHudText = useAppSettings((s) => s.showHudText);
  const showTrail = useAppSettings((s) => s.showTrail);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trailRef = useRef<Array<[number, number]>>([]);
  const cameraCenterRef = useRef<[number, number] | null>(null);
  const lastLockRef = useRef(lockCameraOnSubject);
  const panRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    startCx: number;
    startCy: number;
  } | null>(null);
  const touchRef = useRef(new Map<number, { x: number; y: number }>());
  const pinchRef = useRef<{
    startDistance: number;
    startScreenX: number;
    startScreenY: number;
    startCx: number;
    startCy: number;
    startZoom: number;
  } | null>(null);
  const [zoom, setZoom] = useState(ZOOM_MAX);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const drawFrame = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const dpr = window.devicePixelRatio || 1;
      const w = parent.clientWidth;
      const h = parent.clientHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#0a0a0a";
      ctx.fillRect(0, 0, w, h);

      if (!latest) {
        drawStatus(ctx, w, h, "Waiting for simulation...");
        return;
      }

      const minDim = Math.min(w, h);
      const pxPerMm = (minDim / FISH_VIEW_LENGTH_MM) * zoom;
      const [cx, cy, cz] = latest.com_mm;
      const fish = buildFishGeometry(latest);
      if (!lockCameraOnSubject && lastLockRef.current) {
        cameraCenterRef.current = [cx, cy];
      }
      if (lockCameraOnSubject || !cameraCenterRef.current) {
        cameraCenterRef.current = [cx, cy];
      }
      lastLockRef.current = lockCameraOnSubject;
      const [viewCx, viewCy] = cameraCenterRef.current;
      const toPx = (mmx: number, mmy: number): [number, number] => [
        w / 2 + (mmx - viewCx) * pxPerMm,
        h / 2 - (mmy - viewCy) * pxPerMm,
      ];

      if (showGrid) drawMmGrid(ctx, w, h, pxPerMm, viewCx, viewCy);

      if (showTrail) {
        const trail = trailRef.current;
        trail.push([cx, cy]);
        if (trail.length > 600) trail.shift();
        ctx.strokeStyle = "rgba(127,255,191,0.28)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < trail.length; i++) {
          const [px, py] = toPx(trail[i][0], trail[i][1]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      } else if (trailRef.current.length) {
        trailRef.current = [];
      }

      drawTailFin(ctx, fish.points[fish.points.length - 1], toPx, pxPerMm);
      drawPectoralFins(ctx, fish.points[3], toPx);
      drawBodySilhouette(ctx, fish, toPx, pxPerMm, w, h);
      drawMidline(ctx, fish.points, toPx, pxPerMm);
      drawYolk(ctx, fish.points[4], toPx, pxPerMm);
      drawEyes(ctx, fish.points[0], toPx, pxPerMm);

      if (showHudText) {
        ctx.fillStyle = "#a1a1aa";
        ctx.font =
          "12px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
        ctx.fillText(
          `tick=${latest.tick}  zoom=${(zoom * 100).toFixed(0)}%  COM=(${cx.toFixed(2)}, ${cy.toFixed(2)}, ${cz.toFixed(3)}) mm`,
          12,
          20,
        );
        if (!latest.running) {
          ctx.fillStyle = "#fca5a5";
          ctx.fillText("paused", 12, 38);
        }
      }
    };

    let raf = requestAnimationFrame(function loop() {
      drawFrame();
      raf = requestAnimationFrame(loop);
    });
    return () => cancelAnimationFrame(raf);
  }, [latest, zoom, lockCameraOnSubject, showGrid, showHudText, showTrail]);

  const onWheel = (e: WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const delta = -e.deltaY * 0.001;
    zoomAt(e.clientX, e.clientY, clamp(zoom + delta, ZOOM_MIN, ZOOM_MAX));
  };

  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (lockCameraOnSubject) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.setPointerCapture(e.pointerId);
    if (e.pointerType === "touch") {
      touchRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (touchRef.current.size >= 2) startPinch();
      else startPan(e.pointerId, e.clientX, e.clientY);
      e.preventDefault();
      return;
    }
    if (e.button === 0 || e.button === 1 || e.button === 2) {
      startPan(e.pointerId, e.clientX, e.clientY);
      e.preventDefault();
    }
  };

  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    if (lockCameraOnSubject) return;
    if (e.pointerType === "touch") {
      touchRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (touchRef.current.size >= 2) {
        updatePinch();
      } else if (panRef.current?.pointerId === e.pointerId) {
        updatePan(e.clientX, e.clientY);
      }
      e.preventDefault();
      return;
    }
    if (panRef.current?.pointerId === e.pointerId) {
      updatePan(e.clientX, e.clientY);
      e.preventDefault();
    }
  };

  const onPointerUp = (e: PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === "touch") {
      touchRef.current.delete(e.pointerId);
      if (touchRef.current.size >= 2) {
        startPinch();
      } else {
        pinchRef.current = null;
        const remaining = touchRef.current.entries().next().value as
          | [number, { x: number; y: number }]
          | undefined;
        if (remaining) startPan(remaining[0], remaining[1].x, remaining[1].y);
        else panRef.current = null;
      }
    }
    if (panRef.current?.pointerId === e.pointerId) panRef.current = null;
    try {
      canvasRef.current?.releasePointerCapture(e.pointerId);
    } catch {
      // Pointer capture may already be gone after a cancelled touch.
    }
  };

  const startPan = (pointerId: number, clientX: number, clientY: number) => {
    const [cx, cy] = currentCameraCenter();
    panRef.current = {
      pointerId,
      startX: clientX,
      startY: clientY,
      startCx: cx,
      startCy: cy,
    };
  };

  const updatePan = (clientX: number, clientY: number) => {
    const pan = panRef.current;
    if (!pan) return;
    const metrics = canvasMetrics();
    if (!metrics) return;
    const pxPerMm = (Math.min(metrics.w, metrics.h) / FISH_VIEW_LENGTH_MM) * zoom;
    cameraCenterRef.current = [
      pan.startCx - (clientX - pan.startX) / pxPerMm,
      pan.startCy + (clientY - pan.startY) / pxPerMm,
    ];
  };

  const startPinch = () => {
    const pinch = pinchMetrics();
    if (!pinch) return;
    const [cx, cy] = currentCameraCenter();
    pinchRef.current = {
      ...pinch,
      startCx: cx,
      startCy: cy,
      startZoom: zoom,
    };
  };

  const updatePinch = () => {
    const start = pinchRef.current;
    const pinch = pinchMetrics();
    const metrics = canvasMetrics();
    if (!start || !pinch || !metrics) return;
    const nextZoom = clamp(
      start.startZoom * (pinch.startDistance / start.startDistance),
      ZOOM_MIN,
      ZOOM_MAX,
    );
    const startPxPerMm = (Math.min(metrics.w, metrics.h) / FISH_VIEW_LENGTH_MM) * start.startZoom;
    const nextPxPerMm = (Math.min(metrics.w, metrics.h) / FISH_VIEW_LENGTH_MM) * nextZoom;
    const worldX = start.startCx + (start.startScreenX - metrics.w / 2) / startPxPerMm;
    const worldY = start.startCy - (start.startScreenY - metrics.h / 2) / startPxPerMm;
    cameraCenterRef.current = [
      worldX - (pinch.startScreenX - metrics.w / 2) / nextPxPerMm,
      worldY + (pinch.startScreenY - metrics.h / 2) / nextPxPerMm,
    ];
    setZoom(nextZoom);
  };

  const zoomAt = (clientX: number, clientY: number, nextZoom: number) => {
    if (!Number.isFinite(nextZoom)) return;
    if (!lockCameraOnSubject) {
      const metrics = canvasMetrics();
      if (metrics) {
        const [cx, cy] = currentCameraCenter();
        const sx = clientX - metrics.left;
        const sy = clientY - metrics.top;
        const oldPxPerMm = (Math.min(metrics.w, metrics.h) / FISH_VIEW_LENGTH_MM) * zoom;
        const nextPxPerMm = (Math.min(metrics.w, metrics.h) / FISH_VIEW_LENGTH_MM) * nextZoom;
        const worldX = cx + (sx - metrics.w / 2) / oldPxPerMm;
        const worldY = cy - (sy - metrics.h / 2) / oldPxPerMm;
        cameraCenterRef.current = [
          worldX - (sx - metrics.w / 2) / nextPxPerMm,
          worldY + (sy - metrics.h / 2) / nextPxPerMm,
        ];
      }
    }
    setZoom(nextZoom);
  };

  const currentCameraCenter = (): [number, number] => {
    if (cameraCenterRef.current) return cameraCenterRef.current;
    if (latest) return [latest.com_mm[0], latest.com_mm[1]];
    return [0, 0];
  };

  const canvasMetrics = () => {
    const canvas = canvasRef.current;
    const parent = canvas?.parentElement;
    if (!canvas || !parent) return null;
    const rect = canvas.getBoundingClientRect();
    return {
      left: rect.left,
      top: rect.top,
      w: parent.clientWidth,
      h: parent.clientHeight,
    };
  };

  const pinchMetrics = () => {
    const touches = [...touchRef.current.values()];
    if (touches.length < 2) return null;
    const a = touches[0];
    const b = touches[1];
    return {
      startDistance: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)),
      startScreenX: (a.x + b.x) * 0.5,
      startScreenY: (a.y + b.y) * 0.5,
    };
  };

  return (
    <canvas
      ref={canvasRef}
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onContextMenu={(e) => {
        if (!lockCameraOnSubject) e.preventDefault();
      }}
      className="absolute inset-0 h-full w-full cursor-grab"
      style={{ touchAction: lockCameraOnSubject ? "auto" : "none" }}
      aria-label="Zebrafish camera"
    />
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function drawBodySilhouette(
  ctx: CanvasRenderingContext2D,
  fish: ReturnType<typeof buildFishGeometry>,
  toPx: (x: number, y: number) => [number, number],
  pxPerMm: number,
  w: number,
  h: number,
) {
  const grad = ctx.createLinearGradient(w * 0.35, h * 0.18, w * 0.65, h * 0.82);
  grad.addColorStop(0, "#dff8f3");
  grad.addColorStop(0.46, "#aacdce");
  grad.addColorStop(1, "#7895a7");

  ctx.beginPath();
  fish.outlineLeft.forEach(([x, y], i) => {
    const [px, py] = toPx(x, y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  for (let i = fish.outlineRight.length - 1; i >= 0; i--) {
    const [px, py] = toPx(fish.outlineRight[i][0], fish.outlineRight[i][1]);
    ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.strokeStyle = "rgba(215, 245, 244, 0.88)";
  ctx.lineWidth = Math.max(1.25, pxPerMm * 0.02);
  ctx.stroke();
}

function drawMidline(
  ctx: CanvasRenderingContext2D,
  points: FishPoint[],
  toPx: (x: number, y: number) => [number, number],
  pxPerMm: number,
) {
  ctx.strokeStyle = "rgba(35, 64, 72, 0.34)";
  ctx.lineWidth = Math.max(1, pxPerMm * 0.012);
  ctx.lineCap = "round";
  ctx.beginPath();
  points.forEach((p, i) => {
    const [px, py] = toPx(p.x, p.y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
}

function drawTailFin(
  ctx: CanvasRenderingContext2D,
  tail: FishPoint,
  toPx: (x: number, y: number) => [number, number],
  pxPerMm: number,
) {
  const tip = [
    tail.x + tail.tangentX * 0.28,
    tail.y + tail.tangentY * 0.28,
  ] as const;
  const rootA = [
    tail.x + tail.normalX * Math.max(0.09, tail.halfWidth * 2.8),
    tail.y + tail.normalY * Math.max(0.09, tail.halfWidth * 2.8),
  ] as const;
  const rootB = [
    tail.x - tail.normalX * Math.max(0.09, tail.halfWidth * 2.8),
    tail.y - tail.normalY * Math.max(0.09, tail.halfWidth * 2.8),
  ] as const;
  const [tx, ty] = toPx(tip[0], tip[1]);
  const [ax, ay] = toPx(rootA[0], rootA[1]);
  const [bx, by] = toPx(rootB[0], rootB[1]);
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(tx, ty);
  ctx.lineTo(bx, by);
  ctx.closePath();
  ctx.fillStyle = "rgba(120, 160, 176, 0.62)";
  ctx.fill();
  ctx.strokeStyle = "rgba(205, 236, 236, 0.55)";
  ctx.lineWidth = Math.max(1, pxPerMm * 0.012);
  ctx.stroke();
}

function drawPectoralFins(
  ctx: CanvasRenderingContext2D,
  p: FishPoint,
  toPx: (x: number, y: number) => [number, number],
) {
  for (const side of [-1, 1]) {
    const base = [
      p.x + p.normalX * p.halfWidth * side,
      p.y + p.normalY * p.halfWidth * side,
    ] as const;
    const outer = [
      p.x + p.normalX * p.halfWidth * 2.15 * side - p.tangentX * 0.12,
      p.y + p.normalY * p.halfWidth * 2.15 * side - p.tangentY * 0.12,
    ] as const;
    const rear = [
      p.x + p.normalX * p.halfWidth * 1.05 * side + p.tangentX * 0.22,
      p.y + p.normalY * p.halfWidth * 1.05 * side + p.tangentY * 0.22,
    ] as const;
    const [bx, by] = toPx(base[0], base[1]);
    const [ox, oy] = toPx(outer[0], outer[1]);
    const [rx, ry] = toPx(rear[0], rear[1]);
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(ox, oy);
    ctx.lineTo(rx, ry);
    ctx.closePath();
    ctx.fillStyle = "rgba(167, 210, 216, 0.28)";
    ctx.fill();
  }
}

function drawYolk(
  ctx: CanvasRenderingContext2D,
  p: FishPoint,
  toPx: (x: number, y: number) => [number, number],
  pxPerMm: number,
) {
  const [px, py] = toPx(p.x, p.y);
  const angle = Math.atan2(-p.tangentY, p.tangentX);
  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(angle);
  ctx.fillStyle = "rgba(235, 238, 208, 0.40)";
  ctx.beginPath();
  ctx.ellipse(0, 0, pxPerMm * 0.20, pxPerMm * 0.105, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawEyes(
  ctx: CanvasRenderingContext2D,
  head: FishPoint,
  toPx: (x: number, y: number) => [number, number],
  pxPerMm: number,
) {
  const eyeBaseX = head.x - head.tangentX * 0.14;
  const eyeBaseY = head.y - head.tangentY * 0.14;
  const radius = Math.max(1.15, pxPerMm * 0.026);
  const offset = 0.105;
  ctx.fillStyle = "#061012";
  for (const side of [-1, 1]) {
    const [ex, ey] = toPx(
      eyeBaseX + head.normalX * offset * side,
      eyeBaseY + head.normalY * offset * side,
    );
    ctx.beginPath();
    ctx.arc(ex, ey, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawStatus(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  msg: string,
) {
  ctx.fillStyle = "#71717a";
  ctx.font = "14px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(msg, w / 2, h / 2);
  ctx.textAlign = "start";
}

function drawMmGrid(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  pxPerMm: number,
  cx: number,
  cy: number,
) {
  const gridMm = 0.1;
  const pxGrid = pxPerMm * gridMm;
  if (pxGrid < 6) return;
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;
  const offsetX = ((cx / gridMm) % 1) * pxGrid;
  const offsetY = ((-cy / gridMm) % 1) * pxGrid;
  ctx.beginPath();
  for (let x = (w / 2 - offsetX) % pxGrid; x < w; x += pxGrid) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
  }
  for (let y = (h / 2 - offsetY) % pxGrid; y < h; y += pxGrid) {
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
  }
  ctx.stroke();
}
