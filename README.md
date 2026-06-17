# Larval zebrafish live demo & virtual lab

Interactive 3D simulation, neural connectome visualizer, and virtual lab suite for the MuJoCo embodied larval zebrafish model.

This repository provides two primary entrypoints for interacting with the larval zebrafish simulation:

1. **Canvas demo**: `zebrafish-demo-server` streams compact physical and joint state to the static `web/` Three.js aquarium viewer.
2. **Virtual lab**: `zebrafish-lab-server` exposes a REST and WebSocket API used by `lab-web/` (a React/TypeScript application), enabling interactive control over zebrafish body, connectome, muscle, and fluid-environment parameters.


Both entrypoints use the same simulation builder:
`simulations.zebrafish.simulation.build_zebrafish_simulation(...)`.

## Repository layout

This package depends on `active-inference` as an editable path dependency
(`../active-inference` in `pyproject.toml`). The simulation uses PAULA through
the sibling `neuron-model` checkout, matching the rest of this workspace.

```text
agi-research/
  active-inference/
  neuron-model/
  zebrafish-live-demo/
    lab/
    lab-web/
    web/
    zebrafish_live_demo/
```

## Run the virtual lab

Backend:

```bash
cd zebrafish-live-demo
uv sync
uv run zebrafish-lab-server --host 127.0.0.1 --port 8811
```

Frontend:

```bash
cd zebrafish-live-demo/lab-web
npm install
npm run dev -- --host 127.0.0.1
```

Open the Vite URL, normally `http://127.0.0.1:5173/`. The dev proxy forwards
`/api` and `/ws` to `http://127.0.0.1:8811`.

## Run the simple aquarium demo

WebSocket backend:

```bash
cd zebrafish-live-demo
uv run zebrafish-demo-server --host 127.0.0.1 --port 8776
```

Static viewer:

```bash
cd zebrafish-live-demo/web
python -m http.server 8086
```

Open:

```text
http://127.0.0.1:8086/?ws=ws://127.0.0.1:8776
```

## Simulation Integration Points

The virtual lab maps system parameters and layouts to larval zebrafish biology at the following integration points:

- Body and MuJoCo model: `active-inference/simulations/zebrafish/body.py`
- Aquatic arena and stimuli: `active-inference/simulations/zebrafish/environment.py`
- PAULA nervous-system adapter: `active-inference/simulations/zebrafish/neuron_mapping.py`
- Simulation builder: `active-inference/simulations/zebrafish/simulation.py`
- Lab connectome layout: `lab/connectome_layout.py`
- Lab parameter registry: `lab/parameters/simulation_params.py`
- Lab 2D and 3D viewers: `lab-web/src/components/ZebrafishCanvas.tsx` and
  `lab-web/src/components/ZebrafishCanvas3D.tsx`

Food is intentionally absent in this path. The demo accepts `startle` and
`ping` commands only; the lab exposes water, hydrodynamic, neural, MuJoCo, body,
muscle, and connectome controls.
