import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const DEFAULT_WS_URL = "ws://127.0.0.1:8775";
const params = new URLSearchParams(window.location.search);
const wsUrl = params.get("ws") || DEFAULT_WS_URL;

const canvas = document.getElementById("scene");
const canvasHost = document.getElementById("canvas-wrap");
const statusEl = document.getElementById("status");
const onlineEl = document.getElementById("online");
const tickEl = document.getElementById("tick");
const speedEl = document.getElementById("speed");
const depthEl = document.getElementById("depth");
const pitchEl = document.getElementById("pitch");
const driveEl = document.getElementById("drive");
const turnEl = document.getElementById("turn");
const neuronsEl = document.getElementById("neurons");
const neuralCountEl = document.getElementById("neural-count");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
const initialSize = rendererSize();
renderer.setSize(initialSize.width, initialSize.height, false);
renderer.shadowMap.enabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x081115);
scene.fog = new THREE.Fog(0x081115, 0.09, 0.18);

const camera = new THREE.PerspectiveCamera(48, initialSize.width / initialSize.height, 0.001, 0.35);
camera.position.set(0.0, -0.105, 0.085);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);
controls.maxDistance = 0.22;
controls.minDistance = 0.025;

const hemi = new THREE.HemisphereLight(0xbcefff, 0x0b1a1e, 1.9);
scene.add(hemi);
const key = new THREE.DirectionalLight(0xffffff, 2.1);
key.position.set(-0.03, -0.05, 0.08);
key.castShadow = true;
scene.add(key);

const arenaGroup = new THREE.Group();
scene.add(arenaGroup);
const arenaMat = new THREE.MeshPhysicalMaterial({
  color: 0x2b7885,
  roughness: 0.52,
  metalness: 0.0,
  transmission: 0.18,
  transparent: true,
  opacity: 0.72,
});
const arena = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.004, 128), arenaMat);
arena.rotation.x = Math.PI / 2;
arena.position.z = -0.0025;
arena.receiveShadow = true;
arenaGroup.add(arena);
const surface = new THREE.Mesh(
  new THREE.CircleGeometry(0.0495, 128),
  new THREE.MeshPhysicalMaterial({
    color: 0x83d7e6,
    roughness: 0.18,
    metalness: 0.0,
    transparent: true,
    opacity: 0.2,
    transmission: 0.45,
    side: THREE.DoubleSide,
  })
);
surface.position.z = 0.00008;
arenaGroup.add(surface);
const rim = new THREE.Mesh(
  new THREE.TorusGeometry(0.05, 0.00065, 8, 128),
  new THREE.MeshStandardMaterial({ color: 0xc3e1df, roughness: 0.35 })
);
arenaGroup.add(rim);

const fishGroup = new THREE.Group();
scene.add(fishGroup);
const bodyMat = new THREE.MeshStandardMaterial({ color: 0xe9eef0, roughness: 0.44 });
const stripeMat = new THREE.MeshStandardMaterial({ color: 0x2e4f5a, roughness: 0.55 });
const finMat = new THREE.MeshStandardMaterial({ color: 0xf0c878, roughness: 0.5, transparent: true, opacity: 0.58, side: THREE.DoubleSide });
const segmentMeshes = [];
const connectorMeshes = [];
const maxPoints = 17;

for (let i = 0; i < maxPoints; i += 1) {
  const radius = i === 0 ? 0.00085 : Math.max(0.00018, 0.00056 * (1 - i / maxPoints));
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 18, 12), i % 3 === 0 ? stripeMat : bodyMat);
  mesh.castShadow = true;
  segmentMeshes.push(mesh);
  fishGroup.add(mesh);
}
for (let i = 0; i < maxPoints - 1; i += 1) {
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.00034, 0.00026, 0.001, 12),
    bodyMat
  );
  mesh.castShadow = true;
  connectorMeshes.push(mesh);
  fishGroup.add(mesh);
}
const finGeom = new THREE.BufferGeometry();
finGeom.setAttribute(
  "position",
  new THREE.Float32BufferAttribute([0, 0, 0, -0.004, 0.003, 0, -0.004, -0.003, 0], 3)
);
finGeom.computeVertexNormals();
const tailFin = new THREE.Mesh(finGeom, finMat);
fishGroup.add(tailFin);

const trailPoints = [];
const trailGeom = new THREE.BufferGeometry();
const trailMat = new THREE.LineBasicMaterial({ color: 0x9de1ff, transparent: true, opacity: 0.5 });
const trail = new THREE.Line(trailGeom, trailMat);
scene.add(trail);

const DEFAULT_SWIM_DEPTH_M = -0.0014;

let ws = null;
let latest = null;
let neuronCells = [];
let renderWidth = 0;
let renderHeight = 0;

function rendererSize() {
  const rect = canvasHost?.getBoundingClientRect();
  return {
    width: Math.max(1, Math.floor(rect?.width || window.innerWidth)),
    height: Math.max(1, Math.floor(rect?.height || window.innerHeight)),
  };
}

function connect() {
  if (ws) ws.close();
  statusEl.textContent = "connecting";
  statusEl.className = "";
  ws = new WebSocket(wsUrl);
  ws.addEventListener("open", () => {
    statusEl.textContent = "live";
    statusEl.className = "ok";
  });
  ws.addEventListener("close", () => {
    statusEl.textContent = "offline";
    statusEl.className = "err";
    setTimeout(connect, 1800);
  });
  ws.addEventListener("error", () => {
    statusEl.textContent = "error";
    statusEl.className = "err";
  });
  ws.addEventListener("message", (event) => {
    const msg = JSON.parse(event.data);
    if (msg.t === "state") {
      latest = msg;
      updateHud(msg);
      updateNeurons(msg.neural);
    } else if (msg.t === "presence") {
      onlineEl.textContent = `${msg.n || 0} online`;
    }
  });
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function updateHud(msg) {
  tickEl.textContent = String(msg.tick || 0);
  speedEl.textContent = ((msg.speed_m_s || 0) * 1000).toFixed(1);
  const depth = Array.isArray(msg.com_m) ? Math.max(0, -Number(msg.com_m[2] || 0) * 1000) : 0;
  depthEl.textContent = depth.toFixed(2);
  pitchEl.textContent = ((Number(msg.pitch_rad || 0) * 180) / Math.PI).toFixed(1);
  const behavior = msg.behavior || {};
  driveEl.textContent = Number(behavior.swim_drive || 0).toFixed(2);
  turnEl.textContent = Number(behavior.turn_bias || 0).toFixed(2);
}

function updateNeurons(neural) {
  if (!neural || !Array.isArray(neural.names)) return;
  if (neuronCells.length !== neural.names.length) {
    neuronsEl.textContent = "";
    neuralCountEl.textContent = `${neural.names.length} PAULA`;
    neuronCells = neural.names.map((name) => {
      const el = document.createElement("div");
      el.className = "neuron";
      el.title = name;
      neuronsEl.appendChild(el);
      return el;
    });
  }
  const s = neural.s || [];
  const fired = neural.fired || [];
  for (let i = 0; i < neuronCells.length; i += 1) {
    const v = Math.max(0, Math.min(1, Number(s[i] || 0)));
    neuronCells[i].style.opacity = String(0.35 + 0.65 * v);
    neuronCells[i].classList.toggle("fired", Boolean(fired[i]));
  }
}

function updateFish(points) {
  if (!Array.isArray(points) || points.length < 2) return;
  for (let i = 0; i < segmentMeshes.length; i += 1) {
    const p = points[Math.min(i, points.length - 1)];
    const z = Number(p[2] ?? DEFAULT_SWIM_DEPTH_M) + 0.00012 * Math.sin(i * 0.7);
    segmentMeshes[i].position.set(p[0], p[1], z);
    segmentMeshes[i].visible = i < points.length;
  }
  const quat = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < connectorMeshes.length; i += 1) {
    const a = points[i];
    const b = points[i + 1];
    const mesh = connectorMeshes[i];
    if (!a || !b) {
      mesh.visible = false;
      continue;
    }
    const av = new THREE.Vector3(a[0], a[1], Number(a[2] ?? DEFAULT_SWIM_DEPTH_M));
    const bv = new THREE.Vector3(b[0], b[1], Number(b[2] ?? DEFAULT_SWIM_DEPTH_M));
    const mid = av.clone().add(bv).multiplyScalar(0.5);
    const dir = bv.clone().sub(av);
    const len = dir.length();
    mesh.position.copy(mid);
    mesh.scale.set(1, Math.max(len, 0.0001), 1);
    quat.setFromUnitVectors(up, dir.normalize());
    mesh.quaternion.copy(quat);
    mesh.visible = true;
  }
  const tail = points[points.length - 1];
  const prev = points[points.length - 2];
  tailFin.position.set(tail[0], tail[1], Number(tail[2] ?? DEFAULT_SWIM_DEPTH_M));
  tailFin.rotation.z = Math.atan2(tail[1] - prev[1], tail[0] - prev[0]);
}

function updateTrail(com) {
  if (!Array.isArray(com)) return;
  trailPoints.push(new THREE.Vector3(com[0], com[1], Number(com[2] ?? DEFAULT_SWIM_DEPTH_M) + 0.00022));
  if (trailPoints.length > 1200) trailPoints.shift();
  trailGeom.setFromPoints(trailPoints);
}

function render() {
  requestAnimationFrame(render);
  resize();
  if (latest) {
    updateFish(latest.points_m || []);
    updateTrail(latest.com_m);
  }
  controls.update();
  renderer.render(scene, camera);
}

function resize() {
  const size = rendererSize();
  if (size.width === renderWidth && size.height === renderHeight) return;
  renderWidth = size.width;
  renderHeight = size.height;
  camera.aspect = size.width / size.height;
  camera.updateProjectionMatrix();
  renderer.setSize(size.width, size.height, false);
}

document.getElementById("startle").addEventListener("click", () => {
  send({ p: 1, t: "startle" });
});
document.getElementById("reconnect").addEventListener("click", connect);
window.addEventListener("resize", resize);

connect();
render();
