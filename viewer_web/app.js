import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

const state = {
  manifest: null,
  currentSceneIdx: 0,
  lastFile: null,
  lastFileType: null,
  lastJsonText: "",
  renderToken: 0,
  loadedAssetCount: 0,
  failedAssetCount: 0,
};

const cache = {
  assetPromiseByUrl: new Map(),
};

const els = {
  fileInput: document.getElementById("file-input"),
  dropZone: document.getElementById("drop-zone"),
  status: document.getElementById("load-status"),
  assetMap: document.getElementById("asset-map"),
  sceneSelect: document.getElementById("scene-select"),
  prevScene: document.getElementById("prev-scene"),
  nextScene: document.getElementById("next-scene"),
  reloadButton: document.getElementById("reload-button"),
  sceneMeta: document.getElementById("scene-meta"),
  wallHeight: document.getElementById("wall-height"),
  wallThickness: document.getElementById("wall-thickness"),
  sceneLimit: document.getElementById("scene-limit"),
  toggleAssets: document.getElementById("toggle-assets"),
  toggleBoxes: document.getElementById("toggle-boxes"),
  toggleWalls: document.getElementById("toggle-walls"),
  toggleFloor: document.getElementById("toggle-floor"),
  toggleGrid: document.getElementById("toggle-grid"),
  toggleAutoFit: document.getElementById("toggle-auto-fit"),
  canvas: document.getElementById("viewer-canvas"),
};

const three = {
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  gltfLoader: null,
  groups: {
    floor: new THREE.Group(),
    walls: new THREE.Group(),
    assets: new THREE.Group(),
    boxes: new THREE.Group(),
    helpers: new THREE.Group(),
  },
};

const FLOOR_MATERIAL = new THREE.MeshStandardMaterial({
  color: 0x7f95a2,
  roughness: 0.92,
  metalness: 0.05,
});
const WALL_MATERIAL = new THREE.MeshStandardMaterial({
  color: 0xc4d2df,
  roughness: 0.83,
  metalness: 0.03,
});

function setStatus(message, type = "idle") {
  els.status.textContent = message;
  els.status.className = `status ${type}`;
}

function parseNumber(value, fallbackValue) {
  if (value === undefined || value === null || value === "") {
    return fallbackValue;
  }
  const n = Number(value);
  if (Number.isFinite(n)) {
    return n;
  }
  return fallbackValue;
}

function proxiedAssetUrl(url) {
  const trimmed = String(url || "").trim();
  if (!trimmed) {
    return "";
  }
  if (trimmed.startsWith("/")) {
    return trimmed;
  }
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return `/api/proxy_asset?url=${encodeURIComponent(trimmed)}`;
  }
  return trimmed;
}

function hashColor(label) {
  let h = 2166136261;
  for (let i = 0; i < label.length; i += 1) {
    h ^= label.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const r = 80 + (h & 0x7f);
  const g = 80 + ((h >> 8) & 0x7f);
  const b = 80 + ((h >> 16) & 0x7f);
  return new THREE.Color(r / 255, g / 255, b / 255);
}

function validateManifest(manifest) {
  if (!manifest || typeof manifest !== "object") {
    throw new Error("Manifest must be a JSON object");
  }
  if (!Array.isArray(manifest.scenes)) {
    throw new Error("Manifest missing 'scenes' array");
  }
  if (manifest.scenes.length === 0) {
    throw new Error("Manifest has no scenes");
  }
}

function parseAssetMapText() {
  const raw = (els.assetMap.value || "").trim();
  if (!raw) {
    return "";
  }
  JSON.parse(raw);
  return raw;
}

function initThree() {
  three.scene = new THREE.Scene();
  three.scene.background = new THREE.Color(0x0f1720);

  three.camera = new THREE.PerspectiveCamera(48, 1, 0.01, 2500);
  three.camera.position.set(6, 5, 6);

  three.renderer = new THREE.WebGLRenderer({ canvas: els.canvas, antialias: true });
  three.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  three.renderer.shadowMap.enabled = true;
  three.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  three.controls = new OrbitControls(three.camera, three.renderer.domElement);
  three.controls.target.set(0, 0.8, 0);
  three.controls.enableDamping = true;
  three.controls.dampingFactor = 0.06;

  three.gltfLoader = new GLTFLoader();
  const dracoLoader = new DRACOLoader();
  dracoLoader.setDecoderPath("https://www.gstatic.com/draco/versioned/decoders/1.5.6/");
  three.gltfLoader.setDRACOLoader(dracoLoader);

  const hemi = new THREE.HemisphereLight(0xd9ecff, 0x24313e, 0.82);
  three.scene.add(hemi);

  const key = new THREE.DirectionalLight(0xffffff, 0.74);
  key.position.set(6, 10, 8);
  key.castShadow = true;
  key.shadow.mapSize.width = 2048;
  key.shadow.mapSize.height = 2048;
  key.shadow.camera.near = 0.1;
  key.shadow.camera.far = 70;
  three.scene.add(key);

  const fill = new THREE.PointLight(0x9ec7ff, 0.32, 40);
  fill.position.set(-5, 4, -5);
  three.scene.add(fill);

  Object.values(three.groups).forEach((group) => {
    three.scene.add(group);
  });

  resizeRenderer();
  window.addEventListener("resize", resizeRenderer);

  const animate = () => {
    requestAnimationFrame(animate);
    three.controls.update();
    three.renderer.render(three.scene, three.camera);
  };
  animate();
}

function resizeRenderer() {
  const { clientWidth, clientHeight } = els.canvas;
  if (clientWidth <= 0 || clientHeight <= 0) {
    return;
  }
  three.renderer.setSize(clientWidth, clientHeight, false);
  three.camera.aspect = clientWidth / clientHeight;
  three.camera.updateProjectionMatrix();
}

function disposeObject3D(node) {
  if (!node) {
    return;
  }
  node.traverse((child) => {
    if (child.geometry) {
      child.geometry.dispose();
    }
    if (child.material) {
      if (Array.isArray(child.material)) {
        child.material.forEach((mat) => mat.dispose());
      } else {
        child.material.dispose();
      }
    }
  });
}

function clearGroup(group) {
  while (group.children.length > 0) {
    const child = group.children[0];
    group.remove(child);
    disposeObject3D(child);
  }
}

function clearRenderedScene() {
  Object.values(three.groups).forEach((group) => clearGroup(group));
  state.loadedAssetCount = 0;
  state.failedAssetCount = 0;
}

function roomBoundary(room) {
  if (Array.isArray(room?.boundary_xz) && room.boundary_xz.length >= 3) {
    return room.boundary_xz;
  }

  if (Array.isArray(room?.floor_vertices) && room.floor_vertices.length >= 3) {
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minZ = Number.POSITIVE_INFINITY;
    let maxZ = Number.NEGATIVE_INFINITY;
    room.floor_vertices.forEach((v) => {
      const x = Number(v[0]);
      const z = Number(v[2]);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    });
    if (Number.isFinite(minX) && Number.isFinite(maxX) && Number.isFinite(minZ) && Number.isFinite(maxZ)) {
      return [
        [minX, minZ],
        [maxX, minZ],
        [maxX, maxZ],
        [minX, maxZ],
      ];
    }
  }

  return null;
}

function addFloor(room) {
  if (!els.toggleFloor.checked || !room) {
    return;
  }

  if (Array.isArray(room.floor_vertices) && Array.isArray(room.floor_faces) && room.floor_faces.length > 0) {
    const positions = [];
    room.floor_vertices.forEach((v) => {
      positions.push(Number(v[0]), Number(v[1]), Number(v[2]));
    });

    const indices = [];
    room.floor_faces.forEach((f) => {
      indices.push(Number(f[0]), Number(f[1]), Number(f[2]));
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const mesh = new THREE.Mesh(geometry, FLOOR_MATERIAL.clone());
    mesh.receiveShadow = true;
    mesh.name = "floor";
    three.groups.floor.add(mesh);
  }
}

function addWalls(room) {
  if (!els.toggleWalls.checked || !room) {
    return;
  }

  const boundary = roomBoundary(room);
  if (!boundary) {
    return;
  }

  const wallHeight = parseNumber(room.wall_height, parseNumber(els.wallHeight.value, 2.6));
  const wallThickness = parseNumber(room.wall_thickness, parseNumber(els.wallThickness.value, 0.08));

  for (let i = 0; i < boundary.length; i += 1) {
    const a = boundary[i];
    const b = boundary[(i + 1) % boundary.length];
    const dx = Number(b[0]) - Number(a[0]);
    const dz = Number(b[1]) - Number(a[1]);
    const length = Math.hypot(dx, dz);
    if (length < 1e-4) {
      continue;
    }

    const geometry = new THREE.BoxGeometry(length, wallHeight, wallThickness);
    const mesh = new THREE.Mesh(geometry, WALL_MATERIAL.clone());
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.position.set((Number(a[0]) + Number(b[0])) * 0.5, wallHeight * 0.5, (Number(a[1]) + Number(b[1])) * 0.5);

    const dir = new THREE.Vector3(dx / length, 0, dz / length);
    const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(1, 0, 0), dir);
    mesh.quaternion.copy(quat);
    mesh.name = `wall_${i}`;
    three.groups.walls.add(mesh);
  }
}

function addGrid(sceneRoom, objects) {
  if (!els.toggleGrid.checked) {
    return;
  }

  const pts = [];
  const boundary = roomBoundary(sceneRoom) || [];
  boundary.forEach((p) => {
    pts.push(new THREE.Vector3(Number(p[0]), 0, Number(p[1])));
  });

  (objects || []).forEach((obj) => {
    const tr = obj.translation || [0, 0, 0];
    pts.push(new THREE.Vector3(Number(tr[0]), Number(tr[1]), Number(tr[2])));
  });

  let size = 12;
  if (pts.length > 0) {
    const box = new THREE.Box3().setFromPoints(pts);
    size = Math.max(6, box.max.distanceTo(box.min) + 2);
  }

  const divisions = Math.max(8, Math.floor(size));
  const grid = new THREE.GridHelper(size, divisions, 0x4f7fa4, 0x2a4157);
  grid.position.y = 0.002;
  grid.material.transparent = true;
  grid.material.opacity = 0.68;
  three.groups.helpers.add(grid);
}

function addBoxFurniture(obj) {
  if (!els.toggleBoxes.checked) {
    return;
  }

  const he = obj.half_extent || [0.4, 0.4, 0.4];
  const extents = [
    Math.max(0.02, Math.abs(Number(he[0])) * 2),
    Math.max(0.02, Math.abs(Number(he[1])) * 2),
    Math.max(0.02, Math.abs(Number(he[2])) * 2),
  ];
  const geometry = new THREE.BoxGeometry(extents[0], extents[1], extents[2]);

  const material = new THREE.MeshStandardMaterial({
    color: hashColor(String(obj.label || "unknown")),
    roughness: 0.65,
    metalness: 0.12,
    transparent: true,
    opacity: 0.52,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;

  const tr = obj.translation || [0, 0, 0];
  mesh.position.set(Number(tr[0]), Number(tr[1]), Number(tr[2]));
  mesh.rotation.y = Number(obj.rotation_y || 0);
  mesh.name = obj.name || obj.label || "box";
  three.groups.boxes.add(mesh);
}

function applyAssetTransform(instance, obj) {
  const userScale = obj.scale || [1, 1, 1];
  instance.scale.set(Number(userScale[0] || 1), Number(userScale[1] || 1), Number(userScale[2] || 1));

  if (els.toggleAutoFit.checked) {
    const he = obj.half_extent || [0.5, 0.5, 0.5];
    const target = new THREE.Vector3(
      Math.max(0.02, Math.abs(Number(he[0])) * 2),
      Math.max(0.02, Math.abs(Number(he[1])) * 2),
      Math.max(0.02, Math.abs(Number(he[2])) * 2)
    );

    const bbox = new THREE.Box3().setFromObject(instance);
    const size = bbox.getSize(new THREE.Vector3());
    if (size.x > 1e-6 && size.y > 1e-6 && size.z > 1e-6) {
      const fitScale = Math.min(target.x / size.x, target.y / size.y, target.z / size.z);
      if (Number.isFinite(fitScale) && fitScale > 0) {
        instance.scale.multiplyScalar(fitScale);
      }
    }
  }

  const bbox = new THREE.Box3().setFromObject(instance);
  const center = bbox.getCenter(new THREE.Vector3());
  instance.position.sub(center);

  const offset = obj.offset || [0, 0, 0];
  instance.position.x += Number(offset[0] || 0);
  instance.position.y += Number(offset[1] || 0);
  instance.position.z += Number(offset[2] || 0);
}

async function loadAssetPrototype(url) {
  const resolved = proxiedAssetUrl(url);
  if (!resolved) {
    throw new Error("Empty asset url");
  }

  if (!cache.assetPromiseByUrl.has(resolved)) {
    const promise = new Promise((resolve, reject) => {
      three.gltfLoader.load(
        resolved,
        (gltf) => resolve(gltf.scene),
        undefined,
        (err) => reject(err)
      );
    });
    cache.assetPromiseByUrl.set(resolved, promise);
  }
  return cache.assetPromiseByUrl.get(resolved);
}

async function addAssetFurniture(obj, token) {
  if (!els.toggleAssets.checked) {
    return;
  }
  const url = obj.asset_url;
  if (!url) {
    return;
  }

  try {
    const prototype = await loadAssetPrototype(url);
    if (token !== state.renderToken) {
      return;
    }

    const holder = new THREE.Group();
    const tr = obj.translation || [0, 0, 0];
    holder.position.set(Number(tr[0]), Number(tr[1]), Number(tr[2]));
    holder.rotation.y = Number(obj.rotation_y || 0);
    holder.name = obj.name || obj.label || "asset";

    const instance = prototype.clone(true);
    instance.traverse((child) => {
      if (child.isMesh) {
        if (child.geometry) {
          child.geometry = child.geometry.clone();
        }
        if (Array.isArray(child.material)) {
          child.material = child.material.map((mat) => mat.clone());
        } else if (child.material) {
          child.material = child.material.clone();
        }
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });
    applyAssetTransform(instance, obj);
    holder.add(instance);
    three.groups.assets.add(holder);
    state.loadedAssetCount += 1;
    updateSceneMeta();
    scheduleFit();
  } catch (err) {
    state.failedAssetCount += 1;
    updateSceneMeta();
    console.warn("Failed to load asset", obj.asset_url, err);
  }
}

function visibleGroupsForFit() {
  return [
    els.toggleFloor.checked ? three.groups.floor : null,
    els.toggleWalls.checked ? three.groups.walls : null,
    els.toggleBoxes.checked ? three.groups.boxes : null,
    els.toggleAssets.checked ? three.groups.assets : null,
  ].filter(Boolean);
}

function fitCameraToContent() {
  const groups = visibleGroupsForFit();
  const box = new THREE.Box3();
  let any = false;
  groups.forEach((g) => {
    if (g.children.length > 0) {
      box.expandByObject(g);
      any = true;
    }
  });
  if (!any) {
    return;
  }

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(0.8, size.length() * 0.5);

  const dist = radius * 1.55;
  three.camera.position.set(center.x + dist, center.y + dist * 0.72, center.z + dist);
  three.controls.target.copy(center);
  three.controls.update();

  three.camera.near = Math.max(0.01, radius / 200);
  three.camera.far = Math.max(120, radius * 30);
  three.camera.updateProjectionMatrix();
}

let fitTimer = null;
function scheduleFit() {
  if (fitTimer) {
    clearTimeout(fitTimer);
  }
  fitTimer = setTimeout(() => {
    fitCameraToContent();
  }, 60);
}

function updateSceneMeta() {
  if (!state.manifest) {
    els.sceneMeta.textContent = "No scene loaded";
    return;
  }
  const scenes = state.manifest.scenes;
  const scene = scenes[state.currentSceneIdx];
  const roomId = scene?.room?.room_id || `scene_${state.currentSceneIdx}`;
  const objectCount = scene?.objects?.length || 0;
  const assetObjects = (scene?.objects || []).filter((o) => !!o.asset_url).length;
  els.sceneMeta.textContent = [
    `Scene ${state.currentSceneIdx + 1}/${scenes.length}`,
    `${roomId}`,
    `${objectCount} objects`,
    `${state.loadedAssetCount}/${assetObjects} assets loaded`,
    `${state.failedAssetCount} failed`,
  ].join(" | ");
}

function populateSceneSelect() {
  els.sceneSelect.innerHTML = "";
  if (!state.manifest) {
    return;
  }

  state.manifest.scenes.forEach((scene, idx) => {
    const opt = document.createElement("option");
    const roomId = scene?.room?.room_id || `scene_${idx}`;
    opt.value = String(idx);
    opt.textContent = `${idx + 1}. ${roomId}`;
    els.sceneSelect.appendChild(opt);
  });
}

function renderScene(refit = true) {
  if (!state.manifest) {
    return;
  }

  state.renderToken += 1;
  const token = state.renderToken;

  clearRenderedScene();

  const scene = state.manifest.scenes[state.currentSceneIdx];
  if (!scene) {
    return;
  }

  addFloor(scene.room || {});
  addWalls(scene.room || {});
  addGrid(scene.room || {}, scene.objects || []);

  (scene.objects || []).forEach((obj) => {
    addBoxFurniture(obj);
    addAssetFurniture(obj, token);
  });

  updateSceneMeta();
  if (refit) {
    scheduleFit();
  }
}

function setManifest(manifest, sourceLabel) {
  validateManifest(manifest);
  state.manifest = manifest;
  state.currentSceneIdx = 0;
  populateSceneSelect();
  els.sceneSelect.value = "0";
  renderScene(true);
  setStatus(`Loaded ${manifest.scenes.length} scenes from ${sourceLabel}`, "ok");
}

async function parsePklWithServer(file) {
  const assetMapJson = parseAssetMapText();
  const form = new FormData();
  form.append("results_file", file);
  form.append("wall_height", String(parseNumber(els.wallHeight.value, 2.6)));
  form.append("wall_thickness", String(parseNumber(els.wallThickness.value, 0.08)));

  const limit = parseNumber(els.sceneLimit.value, null);
  if (limit) {
    form.append("limit", String(limit));
  }
  if (assetMapJson) {
    form.append("asset_map_json", assetMapJson);
  }

  const response = await fetch("/api/parse_results", {
    method: "POST",
    body: form,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Server error ${response.status}`);
  }
  return payload;
}

async function parseGpcJsonWithServer(file) {
  const assetMapJson = parseAssetMapText();
  const form = new FormData();
  form.append("gpc_file", file);
  form.append("wall_height", String(parseNumber(els.wallHeight.value, 2.6)));
  form.append("wall_thickness", String(parseNumber(els.wallThickness.value, 0.12)));

  if (assetMapJson) {
    form.append("asset_map_json", assetMapJson);
  }

  const response = await fetch("/api/parse_gpc_json", {
    method: "POST",
    body: form,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Server error ${response.status}`);
  }
  return payload;
}

function applyViewerWallOverrides(manifest) {
  const wallHeight = parseNumber(els.wallHeight.value, 2.6);
  const wallThickness = parseNumber(els.wallThickness.value, 0.08);
  (manifest.scenes || []).forEach((scene) => {
    if (!scene.room) {
      scene.room = {};
    }
    scene.room.wall_height = wallHeight;
    scene.room.wall_thickness = wallThickness;
  });
}

async function loadFromFile(file, remember = true) {
  const lower = (file.name || "").toLowerCase();
  if (!lower.endsWith(".json") && !lower.endsWith(".pkl")) {
    throw new Error("Unsupported file type. Use .pkl or .json");
  }

  setStatus(`Loading ${file.name}...`, "warn");

  if (lower.endsWith(".json")) {
    const text = await file.text();
    const parsed = JSON.parse(text);

    let manifest;
    if (Array.isArray(parsed?.rootNodes)) {
      manifest = await parseGpcJsonWithServer(file);
    } else {
      manifest = parsed;
      validateManifest(manifest);
      applyViewerWallOverrides(manifest);
    }

    setManifest(manifest, file.name);
    if (remember) {
      state.lastFile = file;
      state.lastFileType = Array.isArray(parsed?.rootNodes) ? "gpc_json" : "json";
      state.lastJsonText = text;
    }
    return;
  }

  const manifest = await parsePklWithServer(file);
  setManifest(manifest, file.name);
  if (remember) {
    state.lastFile = file;
    state.lastFileType = "pkl";
    state.lastJsonText = "";
  }
}

async function reloadLastFile() {
  if (!state.lastFileType) {
    setStatus("Load a file first", "warn");
    return;
  }

  try {
    if (state.lastFileType === "pkl") {
      const manifest = await parsePklWithServer(state.lastFile);
      setManifest(manifest, state.lastFile.name);
      return;
    }

    if (state.lastFileType === "gpc_json") {
      const manifest = await parseGpcJsonWithServer(state.lastFile);
      setManifest(manifest, state.lastFile.name);
      return;
    }

    const manifest = JSON.parse(state.lastJsonText);
    validateManifest(manifest);
    applyViewerWallOverrides(manifest);
    setManifest(manifest, state.lastFile.name);
  } catch (err) {
    setStatus(`Reload failed: ${err.message}`, "error");
  }
}

function handleDropVisualState(enable) {
  if (enable) {
    els.dropZone.classList.add("dragover");
  } else {
    els.dropZone.classList.remove("dragover");
  }
}

function bindEvents() {
  ["dragenter", "dragover"].forEach((ev) => {
    els.dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      handleDropVisualState(true);
    });
  });

  ["dragleave", "drop"].forEach((ev) => {
    els.dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      handleDropVisualState(false);
    });
  });

  els.dropZone.addEventListener("drop", async (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (!file) {
      return;
    }
    try {
      await loadFromFile(file, true);
    } catch (err) {
      setStatus(`Load failed: ${err.message}`, "error");
    }
  });

  els.fileInput.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      await loadFromFile(file, true);
    } catch (err) {
      setStatus(`Load failed: ${err.message}`, "error");
    }
  });

  els.sceneSelect.addEventListener("change", () => {
    if (!state.manifest) {
      return;
    }
    state.currentSceneIdx = parseInt(els.sceneSelect.value, 10) || 0;
    renderScene(true);
  });

  els.prevScene.addEventListener("click", () => {
    if (!state.manifest) {
      return;
    }
    state.currentSceneIdx = (state.currentSceneIdx - 1 + state.manifest.scenes.length) % state.manifest.scenes.length;
    els.sceneSelect.value = String(state.currentSceneIdx);
    renderScene(true);
  });

  els.nextScene.addEventListener("click", () => {
    if (!state.manifest) {
      return;
    }
    state.currentSceneIdx = (state.currentSceneIdx + 1) % state.manifest.scenes.length;
    els.sceneSelect.value = String(state.currentSceneIdx);
    renderScene(true);
  });

  [els.toggleAssets, els.toggleBoxes, els.toggleWalls, els.toggleFloor, els.toggleGrid, els.toggleAutoFit].forEach((el) => {
    el.addEventListener("change", () => renderScene(false));
  });

  els.reloadButton.addEventListener("click", () => {
    reloadLastFile();
  });
}

function bootstrap() {
  initThree();
  bindEvents();
  setStatus("Ready. Drag and drop a results file.", "idle");
}

bootstrap();
