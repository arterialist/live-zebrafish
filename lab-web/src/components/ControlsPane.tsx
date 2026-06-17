import { clsx } from "clsx";
import { BrainCircuit, ChevronDown, ChevronUp, Video } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { useLabStore, type Tab } from "../state/store";
import { SimulationTab } from "./tabs/SimulationTab";
import { MujocoEngineTab } from "./tabs/MujocoEngineTab";
import { ConnectomeTab } from "./tabs/ConnectomeTab";
import { BodyTab } from "./tabs/BodyTab";
import { AppSettingsTab } from "./tabs/AppSettingsTab";
import { CalciumStimulusTab } from "./tabs/CalciumStimulusTab";
import { VideoStimulusTab } from "./tabs/VideoStimulusTab";
import { KeyHint } from "./ui/KeyHint";

const TABS: { id: Tab; label: string; key: string }[] = [
  { id: "sim", label: "Simulation settings", key: "1" },
  { id: "mujoco", label: "MuJoCo engine", key: "2" },
  { id: "connectome", label: "Connectome", key: "3" },
  { id: "body", label: "Body", key: "4" },
  { id: "app", label: "App settings", key: "5" },
];

type BottomSheet = "video" | "calcium";
const BOTTOM_SHEET_STORAGE_KEY = "zebrafish.lab.bottomSheet";

export function ControlsPane() {
  const activeTab = useLabStore((s) => s.activeTab);
  const setTab = useLabStore((s) => s.setTab);
  const [openSheet, setOpenSheet] = useState<BottomSheet | null>(() => {
    try {
      const saved = window.localStorage.getItem(BOTTOM_SHEET_STORAGE_KEY);
      return saved === "video" || saved === "calcium" ? saved : null;
    } catch {
      return null;
    }
  });

  useEffect(() => {
    try {
      if (openSheet) {
        window.localStorage.setItem(BOTTOM_SHEET_STORAGE_KEY, openSheet);
      } else {
        window.localStorage.removeItem(BOTTOM_SHEET_STORAGE_KEY);
      }
    } catch {
      // Browser storage may be disabled; the lab still works without sheet persistence.
    }
  }, [openSheet]);

  return (
    <div className="relative flex h-full flex-col overflow-hidden bg-zinc-950">
      <nav
        className="flex shrink-0 items-end gap-0 border-b border-zinc-800 px-3"
        role="tablist"
        aria-label="Lab sections"
      >
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={activeTab === t.id}
            onClick={() => setTab(t.id)}
            title={`${t.label} (press ${t.key})`}
            className={clsx(
              "flex items-center gap-2 px-4 py-2.5 text-sm transition-colors",
              "border-b-2 -mb-px",
              activeTab === t.id
                ? "border-accent text-zinc-100"
                : "border-transparent text-zinc-400 hover:text-zinc-200",
            )}
          >
            <span>{t.label}</span>
            <KeyHint size="xxs">{t.key}</KeyHint>
          </button>
        ))}
      </nav>
      <div className="min-h-0 flex-1 overflow-auto pb-12">
        {activeTab === "sim" && <SimulationTab />}
        {activeTab === "mujoco" && <MujocoEngineTab />}
        {activeTab === "connectome" && <ConnectomeTab />}
        {activeTab === "body" && <BodyTab />}
        {activeTab === "app" && <AppSettingsTab />}
      </div>
      <div
        className={clsx(
          "absolute inset-x-0 bottom-0 z-40 flex flex-col overflow-hidden rounded-t-lg border border-zinc-800 bg-zinc-950 shadow-2xl shadow-black/60",
          "transition-[height] duration-200 ease-out",
          openSheet ? "h-[78%]" : "h-11",
        )}
      >
        <div className="grid h-11 shrink-0 grid-cols-2 border-b border-zinc-800 bg-zinc-900/95">
          <SheetHandle
            active={openSheet === "video"}
            icon={<Video size={15} />}
            label="Video stimulus"
            onClick={() => setOpenSheet(openSheet === "video" ? null : "video")}
          />
          <SheetHandle
            active={openSheet === "calcium"}
            icon={<BrainCircuit size={15} />}
            label="Calcium replay"
            onClick={() => setOpenSheet(openSheet === "calcium" ? null : "calcium")}
          />
        </div>
        <div className="relative min-h-0 flex-1 overflow-hidden bg-zinc-950">
          <SheetContent active={openSheet === "video"}>
            <VideoStimulusTab />
          </SheetContent>
          <SheetContent active={openSheet === "calcium"}>
            <CalciumStimulusTab />
          </SheetContent>
        </div>
      </div>
    </div>
  );
}

function SheetHandle({
  active,
  icon,
  label,
  onClick,
}: {
  active: boolean;
  icon: ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      className={clsx(
        "flex items-center justify-center gap-2 border-x border-zinc-800 px-3 text-xs font-medium uppercase tracking-wide transition-colors",
        active
          ? "bg-zinc-950 text-zinc-100"
          : "bg-zinc-900 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100",
      )}
      type="button"
      onClick={onClick}
      title={active ? `Collapse ${label}` : `Expand ${label}`}
    >
      {icon}
      <span>{label}</span>
      {active ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
    </button>
  );
}

function SheetContent({ active, children }: { active: boolean; children: ReactNode }) {
  return (
    <div
      aria-hidden={!active}
      className={clsx(
        "absolute inset-0 min-h-0 overflow-auto transition-opacity duration-150",
        active ? "pointer-events-auto visible opacity-100" : "pointer-events-none invisible opacity-0",
      )}
    >
      {children}
    </div>
  );
}
