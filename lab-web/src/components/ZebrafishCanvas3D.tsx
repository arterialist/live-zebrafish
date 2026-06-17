import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  FISH_VIEW_LENGTH_MM,
  buildFishGeometry,
  type FishGeometry,
  type FishPoint,
} from "../rendering/fish-body";
import { useLabStore } from "../state/store";
import { useAppSettings } from "../state/app-settings";

const MM_TO_WORLD = 1 / FISH_VIEW_LENGTH_MM;
const Z_EXAGGERATION = 2.4;
const BODY_RINGS = 19;
const BODY_RING_SEGMENTS = 14;
const X_AXIS = new THREE.Vector3(1, 0, 0);

export function ZebrafishCanvas3D() {
  const rootRef = useRef<HTMLDivElement>(null);
  const canvasHostRef = useRef<HTMLDivElement>(null);
  const resetViewRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const canvasHost = canvasHostRef.current;
    if (!canvasHost) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.Fog(0x0a0a0a, 5.5, 22);

    const camera = new THREE.PerspectiveCamera(48, 1, 0.02, 80);
    camera.up.set(0, 1, 0);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio ?? 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    canvasHost.appendChild(renderer.domElement);
    renderer.domElement.classList.add("block", "h-full", "w-full", "cursor-grab");
    renderer.domElement.addEventListener("mousedown", () => {
      renderer.domElement.classList.replace("cursor-grab", "cursor-grabbing");
    });
    renderer.domElement.addEventListener("mouseup", () => {
      renderer.domElement.classList.replace("cursor-grabbing", "cursor-grab");
    });

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.screenSpacePanning = true;
    controls.enablePan = true;
    controls.minDistance = 0.28;
    controls.maxDistance = 48;
    controls.minPolarAngle = 0;
    controls.maxPolarAngle = Math.PI;
    controls.rotateSpeed = 0.65;
    controls.zoomSpeed = 0.85;
    controls.panSpeed = 0.75;
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    };

    const hemi = new THREE.HemisphereLight(0x9db4c8, 0x080808, 0.85);
    hemi.position.set(0, 1, 0);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 0.6);
    key.position.set(2.2, 4.5, 1.4);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xb8c9e0, 0.24);
    fill.position.set(-2.5, 2.2, -2);
    scene.add(fill);

    const gridSize = 6;
    const grid = new THREE.GridHelper(gridSize, 60, 0x5a6a82, 0x303844);
    grid.position.y = -0.006;
    const gridMat = grid.material as THREE.LineBasicMaterial;
    gridMat.transparent = true;
    gridMat.opacity = 0.42;
    gridMat.depthWrite = false;
    scene.add(grid);

    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x0f1218,
      metalness: 0.05,
      roughness: 0.92,
      transparent: true,
      opacity: 0.88,
    });
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(gridSize * 2, gridSize * 2),
      floorMat,
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.008;
    scene.add(floor);

    const fishGroup = new THREE.Group();
    fishGroup.visible = false;
    scene.add(fishGroup);

    const bodyGeom = createBodyGeometry();
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0xbad8d6,
      metalness: 0.04,
      roughness: 0.55,
      transparent: true,
      opacity: 0.96,
      side: THREE.DoubleSide,
    });
    const bodyMesh = new THREE.Mesh(bodyGeom, bodyMat);
    fishGroup.add(bodyMesh);

    const headGeom = new THREE.SphereGeometry(1, 28, 18);
    const headMat = new THREE.MeshStandardMaterial({
      color: 0xdaf4ef,
      metalness: 0.08,
      roughness: 0.42,
      emissive: 0x071412,
      emissiveIntensity: 0.18,
    });
    const head = new THREE.Mesh(headGeom, headMat);
    fishGroup.add(head);

    const eyeGeom = new THREE.SphereGeometry(1, 12, 10);
    const eyeMat = new THREE.MeshStandardMaterial({
      color: 0x020608,
      metalness: 0.08,
      roughness: 0.28,
    });
    const leftEye = new THREE.Mesh(eyeGeom, eyeMat);
    const rightEye = new THREE.Mesh(eyeGeom, eyeMat);
    fishGroup.add(leftEye, rightEye);

    const finGeom = new THREE.BufferGeometry();
    finGeom.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array([0, 0, 1, 1, 0, 0, 0, 0, -1]), 3),
    );
    finGeom.setIndex([0, 1, 2]);
    finGeom.computeVertexNormals();
    const finMat = new THREE.MeshStandardMaterial({
      color: 0x88b9c8,
      metalness: 0.02,
      roughness: 0.7,
      transparent: true,
      opacity: 0.48,
      side: THREE.DoubleSide,
    });
    const tailFin = new THREE.Mesh(finGeom, finMat);
    const dorsalFin = new THREE.Mesh(finGeom, finMat);
    dorsalFin.visible = false;
    const leftPectoral = new THREE.Mesh(finGeom, finMat);
    const rightPectoral = new THREE.Mesh(finGeom, finMat);
    fishGroup.add(tailFin, dorsalFin, leftPectoral, rightPectoral);

    const tmpA = new THREE.Vector3();
    const tmpB = new THREE.Vector3();
    const tmpDir = new THREE.Vector3();
    const tmpNorm = new THREE.Vector3();
    const tmpUp = new THREE.Vector3();
    const tmpQuat = new THREE.Quaternion();

    const resetView = () => {
      const latest = useLabStore.getState().latest;
      const heading = latest?.heading_rad ?? 0;
      const lateralX = -Math.sin(heading);
      const lateralZ = Math.cos(heading);
      camera.position.set(lateralX * 2.05, 0.95, lateralZ * 2.05);
      controls.target.set(0, 0, 0);
      camera.up.set(0, 1, 0);
      camera.updateProjectionMatrix();
      controls.update();
    };
    resetView();
    resetViewRef.current = resetView;

    let raf = 0;
    let lastGeomMs = 0;
    let didInitialReset = false;
    let anchorMm: [number, number, number] | null = null;
    let lastLock = useAppSettings.getState().lockCameraOnSubject;

    const resize = () => {
      const w = canvasHost.clientWidth;
      const h = Math.max(canvasHost.clientHeight, 1);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    };

    const ro = new ResizeObserver(resize);
    ro.observe(canvasHost);
    resize();

    const tick = (timeMs: number) => {
      raf = requestAnimationFrame(tick);
      const cap = useAppSettings.getState().renderFpsCap;
      const minInterval = cap > 0 ? 1000 / cap : 0;
      const geomDue =
        minInterval <= 0 || timeMs - lastGeomMs >= minInterval - 0.5;

      if (geomDue) {
        lastGeomMs = timeMs;
        const latest = useLabStore.getState().latest;
        if (latest) {
          const lockCameraOnSubject = useAppSettings.getState().lockCameraOnSubject;
          if (!lockCameraOnSubject && lastLock) {
            anchorMm = [...latest.com_mm];
          }
          if (lockCameraOnSubject || !anchorMm) {
            anchorMm = [...latest.com_mm];
          }
          lastLock = lockCameraOnSubject;
          if (lockCameraOnSubject) {
            controls.target.set(0, 0, 0);
          }
          updateFish(
            buildFishGeometry(latest),
            anchorMm,
            bodyGeom,
            head,
            leftEye,
            rightEye,
            tailFin,
            dorsalFin,
            leftPectoral,
            rightPectoral,
            tmpA,
            tmpB,
            tmpDir,
            tmpNorm,
            tmpUp,
            tmpQuat,
          );
          fishGroup.visible = true;
          if (!didInitialReset) {
            resetView();
            didInitialReset = true;
          }
        } else {
          fishGroup.visible = false;
        }
      }

      controls.update();
      renderer.render(scene, camera);
    };

    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      resetViewRef.current = null;
      ro.disconnect();
      controls.dispose();
      bodyGeom.dispose();
      bodyMat.dispose();
      headGeom.dispose();
      headMat.dispose();
      eyeGeom.dispose();
      eyeMat.dispose();
      finGeom.dispose();
      finMat.dispose();
      floor.geometry.dispose();
      floorMat.dispose();
      grid.dispose();
      scene.fog = null;
      renderer.dispose();
      if (renderer.domElement.parentNode === canvasHost) {
        canvasHost.removeChild(renderer.domElement);
      }
    };
  }, []);

  useEffect(() => {
    rootRef.current?.focus({ preventScroll: true });
  }, []);

  return (
    <div
      ref={rootRef}
      className="relative h-full min-h-0 w-full outline-none focus-visible:ring-1 focus-visible:ring-accent/50 focus-visible:ring-offset-2 focus-visible:ring-offset-black"
      tabIndex={0}
      role="application"
      aria-label="Zebrafish 3D view"
      onKeyDown={(e) => {
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        if (e.code === "KeyR" || e.code === "Home") {
          e.preventDefault();
          resetViewRef.current?.();
        }
      }}
    >
      <div ref={canvasHostRef} className="absolute inset-0 min-h-0" />

      <div className="pointer-events-none absolute bottom-3 left-3 max-w-[220px] rounded-lg border border-zinc-800/90 bg-zinc-950/85 p-2.5 shadow-lg ring-1 ring-black/40 backdrop-blur-sm">
        <div className="mb-1.5 text-[10px] font-semibold tracking-wider text-zinc-400 uppercase">
          Controls
        </div>
        <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 text-[11px] leading-tight text-zinc-300">
          <dt className="font-mono text-zinc-500">Rotate</dt>
          <dd>Left-drag</dd>
          <dt className="font-mono text-zinc-500">Pan</dt>
          <dd>Right-drag · Shift+left</dd>
          <dt className="font-mono text-zinc-500">Zoom</dt>
          <dd>Scroll · pinch</dd>
          <dt className="font-mono text-zinc-500">Reset</dt>
          <dd>
            <kbd className="rounded border border-zinc-700 bg-zinc-900 px-1 font-mono text-[10px] text-zinc-200">
              R
            </kbd>{" "}
            ·{" "}
            <kbd className="rounded border border-zinc-700 bg-zinc-900 px-1 font-mono text-[10px] text-zinc-200">
              Home
            </kbd>
          </dd>
        </dl>
      </div>

      <button
        type="button"
        className="pointer-events-auto absolute right-3 bottom-3 rounded-md border border-zinc-700 bg-zinc-900/90 px-2.5 py-1.5 text-[11px] font-medium text-zinc-200 shadow-md backdrop-blur-sm hover:bg-zinc-800 hover:text-white focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none"
        onClick={() => resetViewRef.current?.()}
      >
        Reset view
      </button>
    </div>
  );
}

function createBodyGeometry(): THREE.BufferGeometry {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(BODY_RINGS * BODY_RING_SEGMENTS * 3), 3),
  );
  const indices: number[] = [];
  for (let ring = 0; ring < BODY_RINGS - 1; ring++) {
    for (let seg = 0; seg < BODY_RING_SEGMENTS; seg++) {
      const a = ring * BODY_RING_SEGMENTS + seg;
      const b = ring * BODY_RING_SEGMENTS + ((seg + 1) % BODY_RING_SEGMENTS);
      const c = (ring + 1) * BODY_RING_SEGMENTS + seg;
      const d = (ring + 1) * BODY_RING_SEGMENTS + ((seg + 1) % BODY_RING_SEGMENTS);
      indices.push(a, c, b, b, c, d);
    }
  }
  geom.setIndex(indices);
  return geom;
}

function updateFish(
  fish: FishGeometry,
  anchorMm: [number, number, number],
  bodyGeom: THREE.BufferGeometry,
  head: THREE.Mesh,
  leftEye: THREE.Mesh,
  rightEye: THREE.Mesh,
  tailFin: THREE.Mesh,
  dorsalFin: THREE.Mesh,
  leftPectoral: THREE.Mesh,
  rightPectoral: THREE.Mesh,
  tmpA: THREE.Vector3,
  tmpB: THREE.Vector3,
  tmpDir: THREE.Vector3,
  tmpNorm: THREE.Vector3,
  tmpUp: THREE.Vector3,
  tmpQuat: THREE.Quaternion,
) {
  updateBodyMesh(fish, anchorMm, bodyGeom, tmpA, tmpDir, tmpNorm, tmpUp);

  const headPoint = fish.points[0];
  pointToWorld(headPoint, anchorMm, tmpA);
  vectorToWorld(-headPoint.tangentX, -headPoint.tangentY, -headPoint.tangentZ, tmpDir);
  tmpQuat.setFromUnitVectors(X_AXIS, tmpDir);
  head.position.copy(tmpA);
  head.quaternion.copy(tmpQuat);
  head.scale.set(0.076, 0.038, 0.070);

  normalToWorld(headPoint, tmpNorm);
  updateEye(leftEye, headPoint, anchorMm, tmpDir, tmpNorm, 1, tmpA, tmpB);
  updateEye(rightEye, headPoint, anchorMm, tmpDir, tmpNorm, -1, tmpA, tmpB);

  const tailPoint = fish.points[fish.points.length - 1];
  orientFin(tailFin, tailPoint, anchorMm, 0.078, 0.038, 1, tmpA, tmpDir, tmpNorm, tmpQuat);
  orientFin(dorsalFin, fish.points[6], anchorMm, 0.076, 0.030, 1, tmpA, tmpDir, tmpNorm, tmpQuat, true);
  updatePectoralFin(leftPectoral, fish.points[3], anchorMm, 1, tmpA, tmpDir, tmpNorm, tmpQuat);
  updatePectoralFin(rightPectoral, fish.points[3], anchorMm, -1, tmpA, tmpDir, tmpNorm, tmpQuat);
}

function updateBodyMesh(
  fish: FishGeometry,
  anchorMm: [number, number, number],
  geom: THREE.BufferGeometry,
  tmpCenter: THREE.Vector3,
  tmpTangent: THREE.Vector3,
  tmpLateral: THREE.Vector3,
  tmpUp: THREE.Vector3,
) {
  const attr = geom.getAttribute("position") as THREE.BufferAttribute;
  for (let ring = 0; ring < BODY_RINGS; ring++) {
    const point = fish.points[Math.min(ring, fish.points.length - 1)];
    pointToWorld(point, anchorMm, tmpCenter);
    vectorToWorld(point.tangentX, point.tangentY, point.tangentZ, tmpTangent);
    normalToWorld(point, tmpLateral);
    tmpUp.crossVectors(tmpLateral, tmpTangent);
    if (tmpUp.lengthSq() < 1e-12) tmpUp.set(0, 1, 0);
    else tmpUp.normalize();
    if (tmpUp.y < 0) tmpUp.multiplyScalar(-1);

    const width = Math.max(0.018, point.halfWidth * MM_TO_WORLD * 1.55);
    const height = Math.max(0.010, point.halfHeight * MM_TO_WORLD * 1.45);
    for (let seg = 0; seg < BODY_RING_SEGMENTS; seg++) {
      const theta = (seg / BODY_RING_SEGMENTS) * Math.PI * 2;
      const lateral = Math.cos(theta) * width;
      const vertical = Math.sin(theta) * height;
      const idx = ring * BODY_RING_SEGMENTS + seg;
      attr.setXYZ(
        idx,
        tmpCenter.x + tmpLateral.x * lateral + tmpUp.x * vertical,
        tmpCenter.y + tmpLateral.y * lateral + tmpUp.y * vertical,
        tmpCenter.z + tmpLateral.z * lateral + tmpUp.z * vertical,
      );
    }
  }
  attr.needsUpdate = true;
  geom.computeVertexNormals();
  geom.computeBoundingSphere();
}

function updateEye(
  mesh: THREE.Mesh,
  head: FishPoint,
  com: [number, number, number],
  headDir: THREE.Vector3,
  normal: THREE.Vector3,
  side: number,
  tmpA: THREE.Vector3,
  tmpB: THREE.Vector3,
) {
  pointToWorld(head, com, tmpA);
  tmpB.copy(tmpA);
  tmpB.addScaledVector(headDir, 0.032);
  tmpB.addScaledVector(normal, side * 0.027);
  tmpB.y += 0.018;
  mesh.position.copy(tmpB);
  mesh.scale.setScalar(0.009);
}

function orientFin(
  mesh: THREE.Mesh,
  point: FishPoint,
  com: [number, number, number],
  length: number,
  width: number,
  side: number,
  tmpPos: THREE.Vector3,
  tmpDir: THREE.Vector3,
  tmpNorm: THREE.Vector3,
  tmpQuat: THREE.Quaternion,
  dorsal = false,
) {
  pointToWorld(point, com, tmpPos);
  vectorToWorld(point.tangentX, point.tangentY, point.tangentZ, tmpDir);
  normalToWorld(point, tmpNorm);
  if (dorsal) {
    tmpPos.y += point.halfHeight * MM_TO_WORLD * 1.6;
    tmpNorm.set(0, 1, 0);
  }
  tmpQuat.setFromUnitVectors(X_AXIS, tmpDir);
  mesh.position.copy(tmpPos);
  mesh.quaternion.copy(tmpQuat);
  mesh.scale.set(length, 1, side * width);
}

function updatePectoralFin(
  mesh: THREE.Mesh,
  point: FishPoint,
  com: [number, number, number],
  side: number,
  tmpPos: THREE.Vector3,
  tmpDir: THREE.Vector3,
  tmpNorm: THREE.Vector3,
  tmpQuat: THREE.Quaternion,
) {
  pointToWorld(point, com, tmpPos);
  vectorToWorld(point.tangentX, point.tangentY, point.tangentZ, tmpDir);
  normalToWorld(point, tmpNorm);
  tmpPos.addScaledVector(tmpNorm, side * point.halfWidth * MM_TO_WORLD * 0.85);
  tmpPos.addScaledVector(tmpDir, 0.006);
  tmpQuat.setFromUnitVectors(X_AXIS, tmpDir);
  mesh.position.copy(tmpPos);
  mesh.quaternion.copy(tmpQuat);
  mesh.scale.set(0.050, 1, side * 0.026);
}

function pointToWorld(
  p: FishPoint,
  com: [number, number, number],
  out: THREE.Vector3,
) {
  out.set(
    (p.x - com[0]) * MM_TO_WORLD,
    (p.z - com[2]) * MM_TO_WORLD * Z_EXAGGERATION,
    (p.y - com[1]) * MM_TO_WORLD,
  );
}

function vectorToWorld(
  x: number,
  y: number,
  z: number,
  out: THREE.Vector3,
) {
  out.set(x * MM_TO_WORLD, z * MM_TO_WORLD * Z_EXAGGERATION, y * MM_TO_WORLD);
  if (out.lengthSq() < 1e-12) out.set(1, 0, 0);
  else out.normalize();
}

function normalToWorld(point: FishPoint, out: THREE.Vector3) {
  out.set(point.normalX, 0, point.normalY);
  if (out.lengthSq() < 1e-12) out.set(0, 0, 1);
  else out.normalize();
}
