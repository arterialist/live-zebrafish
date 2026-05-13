import { useEffect, useRef, useState } from "react";
import { useLabStore } from "../state/store";
import { useAppSettings } from "../state/app-settings";

/** 5-7 dpf larval zebrafish are about 4.2 mm long; leave visual padding. */
const BODY_LENGTH_MM = 5.0;
const ZOOM_MIN = 0.5; // 2x smaller than max
const ZOOM_MAX = 1.0;

export function ZebrafishCanvas() {
  const latest = useLabStore((s) => s.latest);
  const showGrid = useAppSettings((s) => s.showGrid);
  const showHudText = useAppSettings((s) => s.showHudText);
  const showTrail = useAppSettings((s) => s.showTrail);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trailRef = useRef<Array<[number, number]>>([]);
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

      // Backdrop grid
      ctx.fillStyle = "#0a0a0a";
      ctx.fillRect(0, 0, w, h);

      if (!latest || latest.segments_mm.length < 9) {
        drawStatus(ctx, w, h, "Waiting for simulation…");
        return;
      }

      // mm_per_px at max zoom: fit BODY_LENGTH_MM to smaller viewport dim.
      const minDim = Math.min(w, h);
      const pxPerMm = (minDim / BODY_LENGTH_MM) * zoom;
      const [cx, cy] = latest.com_mm;
      const cz = latest.com_mm[2];

      if (showGrid) drawMmGrid(ctx, w, h, pxPerMm, cx, cy);

      // Zebrafish body as a tapered axial body, head first.
      const seg = latest.segments_mm;
      const toPx = (mmx: number, mmy: number): [number, number] => [
        w / 2 + (mmx - cx) * pxPerMm,
        h / 2 - (mmy - cy) * pxPerMm, // invert y
      ];

      if (showTrail) {
        const trail = trailRef.current;
        trail.push([cx, cy]);
        if (trail.length > 600) trail.shift();
        ctx.strokeStyle = "rgba(127,255,191,0.35)";
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

      ctx.lineWidth = Math.max(2, pxPerMm * 0.18);
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, "#9bd5d4");
      grad.addColorStop(0.45, "#d9ede8");
      grad.addColorStop(1, "#89a7b5");
      ctx.strokeStyle = grad;
      ctx.beginPath();
      for (let i = 0; i < seg.length; i += 3) {
        const [px, py] = toPx(seg[i], seg[i + 1]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // Segment points taper toward the tail tip.
      for (let i = 0; i < seg.length; i += 3) {
        const [px, py] = toPx(seg[i], seg[i + 1]);
        const frac = i / Math.max(3, seg.length - 3);
        const radius = Math.max(1.2, pxPerMm * (0.11 - 0.075 * frac));
        ctx.fillStyle = frac < 0.12 ? "#d8f2ed" : "#9fc3c8";
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();
      }
      // Head marker and eye spots.
      ctx.fillStyle = "#dff8f3";
      const [hx, hy] = toPx(seg[0], seg[1]);
      ctx.beginPath();
      ctx.arc(hx, hy, Math.max(3, pxPerMm * 0.18), 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#071113";
      ctx.beginPath();
      ctx.arc(hx, hy - Math.max(1.5, pxPerMm * 0.065), Math.max(1.0, pxPerMm * 0.035), 0, Math.PI * 2);
      ctx.arc(hx, hy + Math.max(1.5, pxPerMm * 0.065), Math.max(1.0, pxPerMm * 0.035), 0, Math.PI * 2);
      ctx.fill();

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
  }, [latest, zoom, showGrid, showHudText, showTrail]);

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    const delta = -e.deltaY * 0.001;
    setZoom((z) => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, z + delta)));
  };

  return (
    <canvas
      ref={canvasRef}
      onWheel={onWheel}
      className="absolute inset-0 h-full w-full cursor-grab"
      aria-label="Zebrafish camera"
    />
  );
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
  const gridMm = 0.1; // 100 µm
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
