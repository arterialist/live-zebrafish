# Larval zebrafish live demo & virtual lab

This package is a source-level copy of `celegans-live-demo` adapted to the
`active-inference` larval zebrafish stack. It keeps the same split:

1. **Canvas demo**: `zebrafish-demo-server-v2` streams compact no-food state to
   the static `web/` Three.js aquarium viewer.
2. **Virtual lab**: `zebrafish-lab-server-v2` exposes the REST and lab
   WebSocket API used by `lab-web/`, with the copied C. elegans lab controls
   retargeted to zebrafish body, connectome, muscle, and water-environment
   parameters.

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
  zebrafish_live_demo_v2/
    lab/
    lab-web/
    web/
    zebrafish_live_demo/
```

## Run the virtual lab

Backend:

```bash
cd zebrafish_live_demo_v2
uv sync
uv run zebrafish-lab-server-v2 --host 127.0.0.1 --port 8811
```

Frontend:

```bash
cd zebrafish_live_demo_v2/lab-web
npm install
npm run dev -- --host 127.0.0.1
```

Open the Vite URL, normally `http://127.0.0.1:5173/`. The dev proxy forwards
`/api` and `/ws` to `http://127.0.0.1:8811`.

## Run the simple aquarium demo

WebSocket backend:

```bash
cd zebrafish_live_demo_v2
uv run zebrafish-demo-server-v2 --host 127.0.0.1 --port 8776
```

Static viewer:

```bash
cd zebrafish_live_demo_v2/web
python -m http.server 8086
```

Open:

```text
http://127.0.0.1:8086/?ws=ws://127.0.0.1:8776
```

## Biological substitution points

The copied C. elegans infrastructure remains recognizable, but the organism
specific paths now point at zebrafish:

- Body and MuJoCo model: `active-inference/simulations/zebrafish/body.py`
- Aquatic arena and stimuli: `active-inference/simulations/zebrafish/environment.py`
- PAULA nervous-system adapter: `active-inference/simulations/zebrafish/neuron_mapping.py`
- Simulation builder: `active-inference/simulations/zebrafish/simulation.py`
- Lab connectome layout: `lab/connectome_layout.py`
- Lab parameter registry: `lab/parameters/simulation_params.py`
- Lab 2D and 3D viewers: `lab-web/src/components/ZebrafishCanvas.tsx` and
  `lab-web/src/components/ZebrafishCanvas3D.tsx`

Food is intentionally absent in this v2 path. The demo accepts `startle` and
`ping` commands only; the lab exposes water, hydrodynamic, neural, MuJoCo, body,
muscle, and connectome controls.
