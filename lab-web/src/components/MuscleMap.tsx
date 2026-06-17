import { useEffect, useRef } from "react";
import type { MouseEvent } from "react";
import { useBodyStore } from "../state/body";
import { useLabStore } from "../state/store";

const ROWS = [
  { side: "L", label: "left yaw", base: [58, 69, 78] },
  { side: "R", label: "right yaw", base: [58, 69, 78] },
  { side: "D", label: "dorsal pitch", base: [42, 63, 86] },
  { side: "V", label: "ventral pitch", base: [76, 62, 45] },
] as const;

type Side = (typeof ROWS)[number]["side"];

interface MuscleCell {
  side: Side;
  seg: number;
  id: number;
  name: string;
}

const MUSCLE_NAME_RE = /^tail_(\d+)_(left|right|dorsal|ventral)$/;
const SIDE_BY_NAME = {
  left: "L",
  right: "R",
  dorsal: "D",
  ventral: "V",
} as const;

function parseMuscles(actuators: { id: number; name: string }[]): MuscleCell[] {
  const cells: MuscleCell[] = [];
  for (const actuator of actuators) {
    const m = MUSCLE_NAME_RE.exec(actuator.name);
    if (!m) continue;
    cells.push({
      side: SIDE_BY_NAME[m[2] as keyof typeof SIDE_BY_NAME],
      seg: Number(m[1]),
      id: actuator.id,
      name: actuator.name,
    });
  }
  return cells;
}

function activationColor(a: number, base: readonly number[]): string {
  const t = Math.max(0, Math.min(1, Math.abs(a)));
  const target = a >= 0 ? [242, 128, 72] : [74, 166, 230];
  const r = Math.round(base[0] + (target[0] - base[0]) * t);
  const g = Math.round(base[1] + (target[1] - base[1]) * t);
  const b = Math.round(base[2] + (target[2] - base[2]) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

export function MuscleMap() {
  const view = useBodyStore((s) => s.view);
  const selection = useBodyStore((s) => s.selection);
  const select = useBodyStore((s) => s.select);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);

  const cells = view ? parseMuscles(view.actuators) : [];
  const segmentCount = cells.reduce((max, c) => Math.max(max, c.seg + 1), 0);
  const cellIndexRef = useRef<MuscleCell[]>(cells);
  cellIndexRef.current = cells;

  useEffect(() => {
    if (!view) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const W = rect.width;
      const H = rect.height;
      ctx.clearRect(0, 0, W, H);

      const padL = 96;
      const padT = 34;
      const padR = 18;
      const padB = 28;
      const count = Math.max(segmentCount, 1);
      const gx = (W - padL - padR) / count;
      const gy = (H - padT - padB) / ROWS.length;

      ctx.font = "11px ui-monospace, monospace";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#8ba3c7";
      ctx.textAlign = "left";
      ctx.fillText("tail base", padL, 14);
      ctx.textAlign = "right";
      ctx.fillText("tail tip", W - padR, 14);

      ctx.strokeStyle = "#242833";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, 23);
      ctx.lineTo(W - padR, 23);
      ctx.stroke();

      for (let row = 0; row < ROWS.length; row++) {
        const y = padT + gy * row;
        ctx.fillStyle = row < 2 ? "rgba(75, 91, 105, 0.12)" : "rgba(76, 73, 94, 0.12)";
        ctx.fillRect(padL, y, W - padL - padR, gy - 1);
        ctx.fillStyle = "#8ba3c7";
        ctx.textAlign = "left";
        ctx.fillText(ROWS[row].label, 12, y + gy * 0.5);
      }

      ctx.textAlign = "center";
      ctx.fillStyle = "#667895";
      for (let seg = 0; seg < count; seg++) {
        if (seg % 2 === 0 || count <= 12) {
          ctx.fillText(String(seg).padStart(2, "0"), padL + gx * (seg + 0.5), H - 12);
        }
      }

      const latest = useLabStore.getState().latest;
      const ma = latest?.ma ?? null;

      for (const cell of cellIndexRef.current) {
        const row = ROWS.findIndex((r) => r.side === cell.side);
        if (row < 0) continue;
        const x = padL + gx * cell.seg;
        const y = padT + gy * row;
        const a = ma ? ma[cell.id] ?? 0 : 0;
        ctx.fillStyle = activationColor(a, ROWS[row].base);
        ctx.fillRect(x + 1, y + 1, gx - 2, gy - 2);

        ctx.strokeStyle = "rgba(5, 7, 10, 0.42)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 1, y + 1, gx - 2, gy - 2);

        if (selection && selection.kind === "muscle" && selection.id === cell.id) {
          ctx.strokeStyle = "#7ab6ff";
          ctx.lineWidth = 2;
          ctx.strokeRect(x + 2, y + 2, gx - 4, gy - 4);
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [view, selection, segmentCount]);

  const handleClick = (evt: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !view) return;
    const rect = canvas.getBoundingClientRect();
    const mx = evt.clientX - rect.left;
    const my = evt.clientY - rect.top;

    const padL = 96;
    const padT = 34;
    const padR = 18;
    const padB = 28;
    const count = Math.max(segmentCount, 1);
    const gx = (rect.width - padL - padR) / count;
    const gy = (rect.height - padT - padB) / ROWS.length;
    const col = Math.floor((mx - padL) / gx);
    const row = Math.floor((my - padT) / gy);
    if (col < 0 || col >= count || row < 0 || row >= ROWS.length) return;
    const side = ROWS[row].side;
    const cell = cellIndexRef.current.find((c) => c.seg === col && c.side === side);
    if (!cell) return;
    if (selection && selection.kind === "muscle" && selection.id === cell.id) {
      select(null);
      return;
    }
    select({ kind: "muscle", id: cell.id, name: cell.name });
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
