# Larval zebrafish live WebSocket demo & virtual lab

This package ships **two** ways to drive the same [active-inference](https://github.com/arterialist/active-inference) larval zebrafish stack (PAULA + MuJoCo):

1. **Canvas demo** — `zebrafish-demo-server` streams compact physical and neural state to the static **`web/`** Three.js aquarium viewer.
2. **Virtual lab** — `zebrafish-lab-server` (**`lab/`**) exposes REST + a richer WebSocket protocol; **`lab-web/`** is a React + Vite app for body, muscle, connectome, and environment controls, neuron inspector, video stimulus pipeline (optical flow extraction), and calcium stimulus replays.

Video replay requires a local or uploaded video. Downloaded clips and derived
stimulus caches under `analysis/cache/` are excluded from Git; a clean checkout
does not include the `commons_black_rockfish_stereo_dov` sample.

The core lab starts without cached research data. Calibrated backend video
extraction additionally requires the local ZAPBench stimulus cache at
`analysis/cache/zapbench/zapbench_stimulus_features.npz` and direct ephys labels
at `analysis/out/zapbench_ephys_action_decoder/direct_ephys_labels.npz`.
`POST /api/video-stimulus/backend-frame` returns HTTP 503 when either is missing.
It does not substitute generated calibration data.

Both entrypoints use the same simulation builder:
`simulations.zebrafish.simulation.build_zebrafish_simulation(...)`.

## Repository layout (local dev)

This package depends on **`active-inference`** as an editable path dependency (`../active-inference` in `pyproject.toml`). That simulation, in turn, loads PAULA from a **sibling directory** named **`neuron-model`** (see `active-inference/simulations/paula_loader.py`). For `uv sync` / imports to work the same way as a local checkout, clone repos **side by side** under a common parent:

```text
your-workspace/
  zebrafish-live-demo/   # this repo (includes lab/ + lab-web/)
  active-inference/      # required — editable dep
  neuron-model/          # required — PAULA (github.com/arterialist/neuron-model)
```

Inside **`zebrafish-live-demo/`**:

- **`lab/`** — FastAPI lab backend (importable package + `zebrafish-lab-server` entrypoint).
- **`lab-web/`** — Vite + React + TypeScript frontend for the lab.
- **`web/`** — static canvas client for the classic aquarium demo.
- **`zebrafish_live_demo/`** — asyncio WebSocket server for the canvas demo.
- **`analysis/`** — comprehensive behavior, calibration, uncertainty, and fidelity analysis scripts.

## Prerequisites

- Same environment expectations as `active-inference` (Python 3.11+, MuJoCo, connectome cache after first run).
- **`neuron-model`** and **`active-inference`** available as siblings.
- **Virtual lab:** Node.js 20+ (or current LTS) for `lab-web/` (`npm install` / `npm run dev`).

## Virtual lab (`zebrafish-lab-server` + `lab-web/`)

The lab runs a **background simulation thread** (same `LabSimRuntime` / connectome as the canvas demo) and serves:

- **REST** under `/api/` — health, connectome, per-neuron detail + patch, body/MuJoCo introspection, parameter schema + live/rebuild patches, simulation transport (play / pause / step), pacing, video upload/config, calcium stimulus config/replay, etc. (see `lab/server.py` docstring and `lab/rest_routes.py`).
- **WebSocket** `GET /ws/state` — lab wire protocol (v7 compact frames: segment geometry, COM, neural summaries, joints, muscles, neuromods; see `lab/wire.py` and `lab-web/src/api/wire.ts`).

**Run locally** (default API + WS on **8765**; Vite proxies to it in dev):

```bash
# from zebrafish-live-demo/
uv sync
uv run zebrafish-lab-server --host 127.0.0.1 --port 8765
```

```bash
# second terminal — from zebrafish-live-demo/lab-web/
npm install
npm run dev
```

Then open the URL Vite prints (typically `http://127.0.0.1:5173`).

### Video Stimulus Pipeline
Exposes a backend video-to-action extraction tool. The browser client acts as a controller that uploads video, and the backend performs deterministic frame decoding, stabilized optical-flow extraction, and ZAPBench-conditioned action calibration.

### Calcium Stimulus Replay
Enables replaying experimental action sequences extracted from calcium imaging datasets based on specific conditions (`turning`, `taxis`, `flash`, `dark`, etc.) with looping and playback transport controls.

## Canvas demo server (`zebrafish-demo-server`)

From this directory:

```bash
uv sync
uv run zebrafish-demo-server --host 127.0.0.1 --port 8775
```

The WebSocket server accepts two commands: `ping` and `startle`. The startle command triggers a startle stimulus in the physical arena.

## Run the static client locally

```bash
cd web && python -m http.server 8085
```

Open `http://127.0.0.1:8085/` and pass your local server, for example:

```text
http://127.0.0.1:8085/?ws=ws://127.0.0.1:8775
```

## How to deploy

**Canvas demo**
1. Deploy the `web/` folder to any static host.
2. Set `DEFAULT_WS_URL` in `app.js` to your public `wss://` endpoint or pass `?ws=`.
3. Keep `zebrafish-demo-server` running.

**Virtual lab**
1. Run `zebrafish-lab-server` on a reachable host/port.
2. Build `lab-web` (`npm run build`) and deploy `lab-web/dist/` behind the same host (path-based) or another static origin; configure the production base URL / proxy so browser calls reach the lab server.

## Protocol specifications

### Canvas Demo Protocol (version 1)
JSON frames sent from the demo server at 30Hz:
- `p`: protocol version (`1`)
- `t`: `"state"`
- `tick`: simulation step index
- `arena_radius_m`: physics arena radius in meters
- `arena_depth_m`: depth of the fluid arena in meters
- `points_m`: 3D coordinates `[x,y,z]` of 17 body segments
- `com_m`: Center of mass coordinate `[x,y,z]`
- `head_m`: Head coordinate `[x,y,z]`
- `heading_rad`: Yaw heading angle in radians
- `pitch_rad`: Pitch angle in radians
- `speed_m_s`: Forward speed in meters/second
- `vertical_velocity_m_s`: Vertical velocity in meters/second
- `tail_angles`: Yaw joint angles of the tail segments
- `tail_pitch_angles`: Pitch joint angles of the tail segments
- `behavior`: Dictionary containing behavioral classifiers
- `neural`: `{ names: [...], s: [...], fired: [...] }` containing membrane potentials and firing states.

### Virtual Lab Wire Protocol (version 7)
WebSocket binary/quantized messages on `/ws/state`:
- `p`: protocol version (`7`)
- `t`: `"s"` (state)
- `k`: tick index
- `sm`: flattened `[x,y,z]` body segments in nanometers (rounded `mm * 1e6`)
- `cm`: center of mass `[x,y,z]` in nanometers
- `hd`: quantized heading angle
- `pt`: quantized pitch angle
- `ta`: scaled tail yaw angles (`JOINT_INT_SCALE`)
- `tpa`: scaled tail pitch angles (`JOINT_INT_SCALE`)
- `Si`: scaled membrane potentials (`NEURAL_INT_SCALE`)
- `Ri`: scaled primary threshold (`NEURAL_INT_SCALE`)
- `Bi`: scaled base threshold (`NEURAL_INT_SCALE`)
- `Trefi`: scaled refractory time (`NEURAL_INT_SCALE`)
- `Fb`: base64 bit-packed neuron firing states
- `ja`: scaled joint angles
- `jv`: scaled joint velocities
- `tc`: scaled touch forces
- `ma`: scaled muscle activations
- `nm01`: scaled neuromodulators
- `M0i` / `M1i`: scaled learning components of M-vectors
- `fe`: scaled free energy metric
