
import * as THREE           from 'three';
import { GUI              } from '../node_modules/three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls    } from '../node_modules/three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { setupGUI, downloadExampleScenesFolder, loadSceneFromURL, drawTendonsAndFlex, getPosition, getQuaternion, toMujocoPos, standardNormal } from './mujocoUtils.js';
import { ImplicitPolicyController } from './implicitPolicyController.js';
import   load_mujoco        from '../node_modules/mujoco-js/dist/mujoco_wasm.js';

function showStartupError(error) {
  const pre = document.createElement('pre');
  pre.style.position = 'fixed';
  pre.style.left = '10px';
  pre.style.top = '10px';
  pre.style.right = '10px';
  pre.style.padding = '12px';
  pre.style.background = '#2a0000';
  pre.style.color = '#ffd7d7';
  pre.style.whiteSpace = 'pre-wrap';
  pre.style.zIndex = '9999';
  pre.textContent = "Startup error:\n" + (error && error.stack ? error.stack : String(error));
  document.body.appendChild(pre);
}

// Load the MuJoCo Module
let mujoco;
try {
  mujoco = await load_mujoco();
} catch (error) {
  console.error(error);
  showStartupError(error);
  throw error;
}

// Set up Emscripten's Virtual File System
var bootstrapScene = "simple.xml";
var initialScene = "pongbot_r2/PONGBOT_R2_V2.xml";
mujoco.FS.mkdir('/working');
mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
mujoco.FS.writeFile("/working/" + bootstrapScene, await(await fetch("./assets/scenes/" + bootstrapScene)).text());

export class MuJoCoDemo {
  constructor() {
    this.mujoco = mujoco;

    // Load in the state from XML
    this.model = mujoco.MjModel.loadFromXML("/working/" + bootstrapScene);
    this.data  = new mujoco.MjData(this.model);

    // Define Random State Variables
    this.params = {
      scene: initialScene,
      paused: false,
      help: false,
      ctrlnoiserate: 0.0,
      ctrlnoisestd: 0.0,
      keyframeNumber: 0,
      cmdX: 0.0,
      cmdY: 0.0,
      cmdYaw: 0.0,
      policyStatus: "loading",
      terrainMode: "stair",
      autoForward: true,
    };
    this.mujoco_time = 0.0;
    this.bodies  = {}, this.lights = {};
    this.tmpVec  = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();
    this.updateGUICallbacks = [];
    this.implicitPolicy = null;
    this.pressedKeys = new Set();
    this.terrainGeoms = [];
    this.terrainBodyIds = new Set();
    this.appliedTerrainMode = null;
    this.followBodyId = -1;
    this.followLastPosition = new THREE.Vector3();
    this.followInitialized = false;
    this.stairLabels = [];
    this.stairLabelPoints = [
      { text: "STEP 0.10 m", position: new THREE.Vector3(4.1, 1.05, -3.2) },
      { text: "STEP 0.15 m", position: new THREE.Vector3(10.0, 1.25, -3.2) },
      { text: "STEP 0.18 m", position: new THREE.Vector3(15.9, 1.40, -3.2) },
      { text: "STEP 0.20 m", position: new THREE.Vector3(21.8, 1.55, -3.2) },
    ];
    this.courseClearEvents = [
      { id: 1, thresholdX: 6.8, title: "COURSE 1 CLEAR", detail: "0.10 m steps complete" },
      { id: 2, thresholdX: 12.7, title: "COURSE 2 CLEAR", detail: "0.15 m steps complete" },
      { id: 3, thresholdX: 18.6, title: "COURSE 3 CLEAR", detail: "0.18 m steps complete" },
      { id: 4, thresholdX: 24.5, title: "COURSE 4 CLEAR", detail: "0.20 m steps complete" },
    ];
    this.clearedCourses = new Set();
    this.lastPolicyStatus = "";
    this.startBannerShown = false;
    this.textDecoder = new TextDecoder("utf-8");
    this.nullChar = this.textDecoder.decode(new ArrayBuffer(1));

    this.container = document.createElement( 'div' );
    document.body.appendChild( this.container );
    this.createControlHelpOverlay();
    this.createWalkReadyOverlay();
    this.createStairLabels();

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.001, 100 );
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(2.0, 1.7, 1.7);
    this.scene.add(this.camera);

    this.scene.background = new THREE.Color(0.10, 0.14, 0.18);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5 );
    this.createFloorGuide();

    this.ambientLight = new THREE.AmbientLight( 0xffffff, 0.28 * 3.14 );
    this.ambientLight.name = 'AmbientLight';
    this.scene.add( this.ambientLight );

    this.spotlight = new THREE.SpotLight();
    this.spotlight.angle = 1.11;
    this.spotlight.distance = 10000;
    this.spotlight.penumbra = 0.5;
    this.spotlight.castShadow = true; // default false
    this.spotlight.intensity = this.spotlight.intensity * 3.14 * 15.0;
    this.spotlight.shadow.mapSize.width = 1024; // default
    this.spotlight.shadow.mapSize.height = 1024; // default
    this.spotlight.shadow.camera.near = 0.1; // default
    this.spotlight.shadow.camera.far = 100; // default
    this.spotlight.position.set(0, 3, 3);
    const targetObject = new THREE.Object3D();
    this.scene.add(targetObject);
    this.spotlight.target = targetObject;
    targetObject.position.set(0, 1, 0);
    this.scene.add( this.spotlight );

    this.renderer = new THREE.WebGLRenderer( { antialias: true } );
    this.renderer.setPixelRatio(1.0);////window.devicePixelRatio );
    this.renderer.setSize( window.innerWidth, window.innerHeight );
    this.renderer.shadowMap.enabled = false;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
    THREE.ColorManagement.enabled = false;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    //this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    //this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    //this.renderer.toneMappingExposure = 2.0;
    this.renderer.useLegacyLights = true;

    this.renderer.setAnimationLoop( this.render.bind(this) );

    this.container.appendChild( this.renderer.domElement );

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.7, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    window.addEventListener('resize', this.onWindowResize.bind(this));
    window.addEventListener('keydown', this.onKeyDown.bind(this));
    window.addEventListener('keyup', this.onKeyUp.bind(this));
    // Initialize the Drag State Manager.
    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);
  }

  async init() {
    // Download the the examples to MuJoCo's virtual file system
    await downloadExampleScenesFolder(mujoco);

    // Initialize the three.js Scene using the .xml Model in initialScene
    [this.model, this.data, this.bodies, this.lights] =
      await loadSceneFromURL(mujoco, initialScene, this);
    this.configureTerrainModes();
    this.applyTerrainMode();
    this.configureCameraFollow();
    this.resetCameraView();
    await this.startImplicitPolicyController();

    this.gui = new GUI();
    setupGUI(this);
  }

  async startImplicitPolicyController() {
    this.implicitPolicy = new ImplicitPolicyController(this);
    this.implicitPolicy.captureWalkReadyStart();
    try {
      await this.implicitPolicy.initialize();
      this.params.policyStatus = this.implicitPolicy.status;
    } catch (error) {
      this.params.policyStatus = "onnx missing";
      console.error(error);
      showStartupError(
        "Implicit ONNX policy failed to load. Place files at:\n" +
        "assets/policies/implicit/encoder.onnx\n" +
        "assets/policies/implicit/policy.onnx\n\n" +
        "If using local ONNX Runtime, also place ort.min.js and wasm files under assets/onnxruntime-web/.\n\n" +
        String(error && error.stack ? error.stack : error)
      );
    }
  }

  createFloorGuide() {
    const grid = new THREE.GridHelper(80, 80, 0x4f78a8, 0x263647);
    grid.name = "Floor Motion Grid";
    grid.position.y = 0.006;
    grid.material.opacity = 0.58;
    grid.material.transparent = true;
    this.scene.add(grid);

    const centerLineMaterial = new THREE.LineBasicMaterial({ color: 0xffc857, transparent: true, opacity: 0.82 });
    const centerLineGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-40, 0.009, 0),
      new THREE.Vector3(40, 0.009, 0),
    ]);
    const centerLine = new THREE.Line(centerLineGeometry, centerLineMaterial);
    centerLine.name = "Floor Center Line";
    this.scene.add(centerLine);
  }

  onKeyDown(event) {
    if (event.code === "Digit1") {
      this.setTerrainMode("flat");
      event.preventDefault();
      return;
    }
    if (event.code === "Digit2") {
      this.setTerrainMode("stair");
      event.preventDefault();
      return;
    }
    if (event.code === "KeyF") {
      this.params.autoForward = !this.params.autoForward;
      this.updateControlHelpOverlay();
      event.preventDefault();
      return;
    }
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "KeyA", "KeyD", "KeyW", "KeyS", "KeyQ", "KeyE"].includes(event.code)) {
      this.pressedKeys.add(event.code);
      event.preventDefault();
    }
  }

  onKeyUp(event) {
    this.pressedKeys.delete(event.code);
  }

  updateKeyboardCommand() {
    const manualForward = (this.pressedKeys.has("ArrowUp") || this.pressedKeys.has("KeyW") ? 1 : 0) -
      (this.pressedKeys.has("ArrowDown") || this.pressedKeys.has("KeyS") ? 1 : 0);
    const lateral = (this.pressedKeys.has("KeyA") ? 1 : 0) - (this.pressedKeys.has("KeyD") ? 1 : 0);
    const yaw = (this.pressedKeys.has("ArrowLeft") || this.pressedKeys.has("KeyQ") ? 1 : 0) -
      (this.pressedKeys.has("ArrowRight") || this.pressedKeys.has("KeyE") ? 1 : 0);
    this.params.cmdX = manualForward !== 0 ? manualForward * 1.0 : (this.params.autoForward ? 0.5 : 0.0);
    this.params.cmdY = lateral * 1.0;
    this.params.cmdYaw = yaw * 1.0;
    if (this.implicitPolicy) {
      this.implicitPolicy.setCommand(this.params.cmdX, this.params.cmdY, this.params.cmdYaw);
      this.params.policyStatus = this.implicitPolicy.status;
    }
  }

  findModelName(adr) {
    return this.textDecoder.decode(this.model.names.subarray(adr)).split(this.nullChar)[0];
  }

  findBodyId(name) {
    if (!this.model) { return -1; }
    for (let i = 0; i < this.model.nbody; i++) {
      if (this.findModelName(this.model.name_bodyadr[i]) === name) { return i; }
    }
    return -1;
  }

  configureCameraFollow() {
    this.followBodyId = this.findBodyId("root");
    if (this.followBodyId < 0 && this.model && this.model.nbody > 1) {
      this.followBodyId = 1;
    }
    this.followInitialized = false;
  }

  resetCameraView() {
    if (this.followBodyId < 0 || !this.data || !this.controls) { return; }
    const current = getPosition(this.data.xpos, this.followBodyId, new THREE.Vector3());
    if (this.params.terrainMode === "stair") {
      this.controls.target.copy(current).add(new THREE.Vector3(7.2, 0.55, -2.6));
      this.camera.position.copy(current).add(new THREE.Vector3(-4.2, 3.8, 6.8));
    } else {
      this.controls.target.copy(current).add(new THREE.Vector3(0.0, 0.05, 0.0));
      this.camera.position.copy(current).add(new THREE.Vector3(2.0, 1.0, 1.7));
    }
    this.followLastPosition.copy(current);
    this.followInitialized = true;
    this.controls.update();
  }

  updateCameraFollow() {
    if (this.followBodyId < 0 || !this.data || !this.controls) { return; }
    const current = getPosition(this.data.xpos, this.followBodyId, new THREE.Vector3());
    if (!this.followInitialized) {
      this.followLastPosition.copy(current);
      this.followInitialized = true;
      return;
    }
    const delta = current.clone().sub(this.followLastPosition);
    if (delta.lengthSq() > 0.0) {
      this.camera.position.add(delta);
      this.controls.target.add(delta);
      this.followLastPosition.copy(current);
    }
  }

  configureTerrainModes() {
    this.terrainGeoms = [];
    this.terrainBodyIds.clear();
    if (!this.model) { return; }

    for (let g = 0; g < this.model.ngeom; g++) {
      const bodyId = this.model.geom_bodyid[g];
      const bodyName = this.findModelName(this.model.name_bodyadr[bodyId]);
      if (!bodyName.toLowerCase().includes("stairs")) { continue; }
      this.terrainBodyIds.add(bodyId);
      this.terrainGeoms.push({
        geomId: g,
        bodyId,
        contype: this.model.geom_contype[g],
        conaffinity: this.model.geom_conaffinity[g],
      });
    }
  }

  applyTerrainMode() {
    if (!this.model) { return; }
    const stairEnabled = this.params.terrainMode === "stair";
    for (const entry of this.terrainGeoms) {
      this.model.geom_contype[entry.geomId] = stairEnabled ? entry.contype : 0;
      this.model.geom_conaffinity[entry.geomId] = stairEnabled ? entry.conaffinity : 0;
    }
    for (const bodyId of this.terrainBodyIds) {
      if (this.bodies[bodyId]) {
        this.bodies[bodyId].visible = stairEnabled;
      }
    }
    this.appliedTerrainMode = this.params.terrainMode;
    this.updateControlHelpOverlay();
  }

  setTerrainMode(mode) {
    if (mode !== "flat" && mode !== "stair") { return; }
    const modeChanged = this.appliedTerrainMode !== mode;
    this.params.terrainMode = mode;
    this.params.autoForward = mode === "stair";
    if (this.mujoco && this.model && this.data) {
      if (modeChanged) {
        this.mujoco.mj_resetData(this.model, this.data);
        this.resetCourseProgress();
      }
      this.applyTerrainMode();
      this.mujoco.mj_forward(this.model, this.data);
      if (modeChanged && this.implicitPolicy) {
        this.implicitPolicy.reset({ immediatePolicy: true });
      }
      if (modeChanged) {
        this.followInitialized = false;
        this.resetCameraView();
      }
    } else {
      this.applyTerrainMode();
    }
    this.updateControlHelpOverlay();
    this.updateStairLabels();
  }

  resetCourseProgress() {
    this.clearedCourses.clear();
    this.lastPolicyStatus = "";
    this.startBannerShown = false;
  }

  createControlHelpOverlay() {
    this.helpOverlay = document.createElement("div");
    this.helpOverlay.style.position = "fixed";
    this.helpOverlay.style.top = "12px";
    this.helpOverlay.style.left = "12px";
    this.helpOverlay.style.width = "280px";
    this.helpOverlay.style.padding = "12px";
    this.helpOverlay.style.background = "rgba(12, 16, 22, 0.78)";
    this.helpOverlay.style.color = "#ffffff";
    this.helpOverlay.style.font = "13px Inter, Arial, sans-serif";
    this.helpOverlay.style.lineHeight = "1.35";
    this.helpOverlay.style.border = "1px solid rgba(255, 255, 255, 0.16)";
    this.helpOverlay.style.borderRadius = "8px";
    this.helpOverlay.style.boxShadow = "0 12px 30px rgba(0, 0, 0, 0.24)";
    this.helpOverlay.style.zIndex = "20";
    this.helpOverlay.style.pointerEvents = "none";
    document.body.appendChild(this.helpOverlay);
    this.updateControlHelpOverlay();
  }

  updateControlHelpOverlay() {
    if (!this.helpOverlay) { return; }
    const isStair = this.params.terrainMode === "stair";
    const modeLabel = isStair ? "Stair" : "Flat";
    const cruiseLabel = this.params.autoForward ? "Cruise on" : "Cruise off";
    const row = (icon, label, keys) => `
      <div style="display:flex; align-items:center; gap:9px; margin-top:7px;">
        <div style="width:22px; color:#91d7ff; text-align:center; font-size:15px;">${icon}</div>
        <div style="flex:1; color:#e9eef5;">${label}</div>
        <div style="color:#b9c6d8; font-size:12px;">${keys}</div>
      </div>`;
    this.helpOverlay.innerHTML = `
      <div style="display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px;">
        <div style="font-weight:700; font-size:14px; letter-spacing:0;">Controls</div>
        <div style="padding:3px 8px; border-radius:999px; background:${isStair ? "#4b3820" : "#153b2a"}; color:${isStair ? "#ffd599" : "#a5f2c7"};">
          ${modeLabel}
        </div>
      </div>
      ${row("◆", "Terrain mode", "1 Flat / 2 Stair")}
      ${row("▶", cruiseLabel, "F or GUI")}
      ${row("▲", "Forward / back", "W/S or ↑/↓")}
      ${row("↔", "Strafe", "A/D")}
      ${row("↺", "Yaw", "Q/E or ←/→")}
      ${row("⟲", "Reset", "Backspace")}
      ${row("Ⅱ", "Pause", "Space")}
      <div style="height:1px; background:rgba(255,255,255,0.12); margin:10px 0 8px;"></div>
      <div style="color:#aeb9c8; font-size:12px;">Camera follows the robot automatically. Drag to orbit.</div>`;
  }

  createWalkReadyOverlay() {
    this.walkReadyOverlay = document.createElement("div");
    this.walkReadyOverlay.style.position = "fixed";
    this.walkReadyOverlay.style.left = "50%";
    this.walkReadyOverlay.style.top = "22%";
    this.walkReadyOverlay.style.transform = "translate(-50%, -50%)";
    this.walkReadyOverlay.style.padding = "14px 22px";
    this.walkReadyOverlay.style.borderRadius = "10px";
    this.walkReadyOverlay.style.background = "rgba(8, 12, 18, 0.82)";
    this.walkReadyOverlay.style.border = "1px solid rgba(105, 210, 255, 0.85)";
    this.walkReadyOverlay.style.color = "#ffffff";
    this.walkReadyOverlay.style.font = "13px Arial, sans-serif";
    this.walkReadyOverlay.style.lineHeight = "1.35";
    this.walkReadyOverlay.style.textAlign = "center";
    this.walkReadyOverlay.style.zIndex = "30";
    this.walkReadyOverlay.style.pointerEvents = "none";
    this.walkReadyOverlay.style.boxShadow = "0 12px 30px rgba(0, 0, 0, 0.24)";
    this.walkReadyOverlay.innerHTML = `
      <div style="font-size:22px; font-weight:800;">WALK READY</div>
      <div style="font-size:13px; color:#f6d997; margin-top:3px;">Robot is held in the air while joints move to the policy start pose.</div>`;
    document.body.appendChild(this.walkReadyOverlay);
    this.updateWalkReadyOverlay();
  }

  updateWalkReadyOverlay() {
    if (!this.walkReadyOverlay) { return; }
    const status = this.implicitPolicy ? this.implicitPolicy.status : this.params.policyStatus;
    const show = status === "walk-ready" || status === "settling" || status === "loading onnx" || status === "initializing";
    this.walkReadyOverlay.style.display = show ? "block" : "none";
    const wasPreparing = this.lastPolicyStatus === "walk-ready" || this.lastPolicyStatus === "settling" ||
      this.lastPolicyStatus === "loading onnx" || this.lastPolicyStatus === "initializing";
    if (!this.startBannerShown && wasPreparing && status === "policy") {
      this.startBannerShown = true;
      this.showStatusBanner("START!", "Walking policy is now active.", "rgba(165, 242, 199, 0.85)");
    }
    this.lastPolicyStatus = status;
  }

  createStairLabels() {
    for (const point of this.stairLabelPoints) {
      const label = document.createElement("div");
      label.textContent = point.text;
      label.style.position = "fixed";
      label.style.padding = "5px 9px";
      label.style.borderRadius = "999px";
      label.style.background = "rgba(9, 13, 18, 0.76)";
      label.style.border = "1px solid rgba(255, 210, 120, 0.75)";
      label.style.color = "#ffd578";
      label.style.font = "700 12px Arial, sans-serif";
      label.style.letterSpacing = "0";
      label.style.textShadow = "0 1px 2px rgba(0, 0, 0, 0.45)";
      label.style.transform = "translate(-50%, -50%)";
      label.style.pointerEvents = "none";
      label.style.zIndex = "18";
      document.body.appendChild(label);
      this.stairLabels.push({ ...point, element: label });
    }
    this.updateStairLabels();
  }

  updateStairLabels() {
    if (!this.stairLabels.length || !this.camera) { return; }
    const visible = this.params.terrainMode === "stair";
    const width = window.innerWidth;
    const height = window.innerHeight;
    for (const label of this.stairLabels) {
      if (!visible) {
        label.element.style.display = "none";
        continue;
      }
      const projected = label.position.clone().project(this.camera);
      const inFront = projected.z > -1 && projected.z < 1;
      label.element.style.display = inFront ? "block" : "none";
      label.element.style.left = `${(projected.x * 0.5 + 0.5) * width}px`;
      label.element.style.top = `${(-projected.y * 0.5 + 0.5) * height}px`;
    }
  }

  checkCourseClearEffects() {
    if (this.params.terrainMode !== "stair" || this.followBodyId < 0 || !this.data) { return; }
    const rootX = this.data.xpos[this.followBodyId * 3 + 0];
    for (const event of this.courseClearEvents) {
      if (this.clearedCourses.has(event.id) || rootX < event.thresholdX) { continue; }
      this.clearedCourses.add(event.id);
      this.showCourseClearEffect(event);
    }
  }

  showCourseClearEffect(event) {
    this.showStatusBanner(event.title, event.detail, "rgba(255, 213, 120, 0.85)", true);
  }

  showStatusBanner(title, detail, borderColor, withParticles = false) {
    const banner = document.createElement("div");
    banner.innerHTML = `
      <div style="font-size:22px; font-weight:800;">${title}</div>
      <div style="font-size:13px; color:#f6d997; margin-top:3px;">${detail}</div>`;
    banner.style.position = "fixed";
    banner.style.left = "50%";
    banner.style.top = "22%";
    banner.style.transform = "translate(-50%, -50%) scale(0.92)";
    banner.style.padding = "14px 22px";
    banner.style.borderRadius = "10px";
    banner.style.background = "rgba(8, 12, 18, 0.82)";
    banner.style.border = `1px solid ${borderColor}`;
    banner.style.color = "#ffffff";
    banner.style.textAlign = "center";
    banner.style.zIndex = "30";
    banner.style.pointerEvents = "none";
    banner.style.opacity = "0";
    banner.style.transition = "opacity 180ms ease, transform 180ms ease";
    document.body.appendChild(banner);
    requestAnimationFrame(() => {
      banner.style.opacity = "1";
      banner.style.transform = "translate(-50%, -50%) scale(1)";
    });
    setTimeout(() => {
      banner.style.opacity = "0";
      banner.style.transform = "translate(-50%, -55%) scale(0.96)";
    }, 1150);
    setTimeout(() => banner.remove(), 1500);

    if (!withParticles) { return; }
    const colors = ["#ffd578", "#69d2ff", "#9cf2b6", "#ff8f70"];
    for (let i = 0; i < 22; i++) {
      const particle = document.createElement("div");
      const angle = (Math.PI * 2 * i) / 22;
      const distance = 55 + Math.random() * 70;
      particle.style.position = "fixed";
      particle.style.left = "50%";
      particle.style.top = "22%";
      particle.style.width = "7px";
      particle.style.height = "7px";
      particle.style.borderRadius = "50%";
      particle.style.background = colors[i % colors.length];
      particle.style.zIndex = "29";
      particle.style.pointerEvents = "none";
      particle.style.opacity = "0.95";
      particle.style.transition = "transform 850ms ease-out, opacity 850ms ease-out";
      document.body.appendChild(particle);
      requestAnimationFrame(() => {
        particle.style.transform = `translate(${Math.cos(angle) * distance}px, ${Math.sin(angle) * distance}px)`;
        particle.style.opacity = "0";
      });
      setTimeout(() => particle.remove(), 900);
    }
  }

  stepImplicitPolicyControl() {
    if (!this.implicitPolicy) { return; }
    this.implicitPolicy.stepControl();
    this.params.policyStatus = this.implicitPolicy.status;
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize( window.innerWidth, window.innerHeight );
  }

  render(timeMS) {
    this.controls.update();

    if (!this.params["paused"]) {
      this.updateKeyboardCommand();
      let timestep = this.model.opt.timestep;
      if (timeMS - this.mujoco_time > 35.0) { this.mujoco_time = timeMS; }
      while (this.mujoco_time < timeMS) {

        // Jitter the control state with gaussian random noise
        if (this.params["ctrlnoisestd"] > 0.0) {
          let rate  = Math.exp(-timestep / Math.max(1e-10, this.params["ctrlnoiserate"]));
          let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
          let currentCtrl = this.data.ctrl;
          for (let i = 0; i < currentCtrl.length; i++) {
            currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
            this.params["Actuator " + i] = currentCtrl[i];
          }
        }

        this.stepImplicitPolicyControl();

        // Clear old perturbations, apply new ones.
        for (let i = 0; i < this.data.qfrc_applied.length; i++) { this.data.qfrc_applied[i] = 0.0; }
        let dragged = this.dragStateManager.physicsObject;
        if (dragged && dragged.bodyID) {
          for (let b = 0; b < this.model.nbody; b++) {
            if (this.bodies[b]) {
              getPosition  (this.data.xpos , b, this.bodies[b].position);
              getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
              this.bodies[b].updateWorldMatrix();
            }
          }
          let bodyID = dragged.bodyID;
          this.dragStateManager.update(); // Update the world-space force origin
          let force = toMujocoPos(this.dragStateManager.currentWorld.clone().sub(this.dragStateManager.worldHit).multiplyScalar(this.model.body_mass[bodyID] * 250));
          let point = toMujocoPos(this.dragStateManager.worldHit.clone());
          mujoco.mj_applyFT(this.model, this.data, [force.x, force.y, force.z], [0, 0, 0], [point.x, point.y, point.z], bodyID, this.data.qfrc_applied);

          // TODO: Apply pose perturbations (mocap bodies only).
        }

        mujoco.mj_step(this.model, this.data);

        this.mujoco_time += timestep * 1000.0;
      }

    } else if (this.params["paused"]) {
      this.dragStateManager.update(); // Update the world-space force origin
      let dragged = this.dragStateManager.physicsObject;
      if (dragged && dragged.bodyID) {
        let b = dragged.bodyID;
        getPosition  (this.data.xpos , b, this.tmpVec , false); // Get raw coordinate from MuJoCo
        getQuaternion(this.data.xquat, b, this.tmpQuat, false); // Get raw coordinate from MuJoCo

        let offset = toMujocoPos(this.dragStateManager.currentWorld.clone()
          .sub(this.dragStateManager.worldHit).multiplyScalar(0.3));
        if (this.model.body_mocapid[b] >= 0) {
          // Set the root body's mocap position...
          console.log("Trying to move mocap body", b);
          let addr = this.model.body_mocapid[b] * 3;
          let pos  = this.data.mocap_pos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        } else {
          // Set the root body's position directly...
          let root = this.model.body_rootid[b];
          let addr = this.model.jnt_qposadr[this.model.body_jntadr[root]];
          let pos  = this.data.qpos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        }
      }

      mujoco.mj_forward(this.model, this.data);
    }

    // Update body transforms.
    for (let b = 0; b < this.model.nbody; b++) {
      if (this.bodies[b]) {
        getPosition  (this.data.xpos , b, this.bodies[b].position);
        getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    // Update light transforms.
    for (let l = 0; l < this.model.nlight; l++) {
      if (this.lights[l]) {
        getPosition(this.data.light_xpos, l, this.lights[l].position);
        getPosition(this.data.light_xdir, l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }

    // Draw Tendons and Flex verts
    drawTendonsAndFlex(this.mujocoRoot, this.model, this.data);

    this.updateCameraFollow();
    this.controls.update();
    this.updateStairLabels();
    this.updateWalkReadyOverlay();
    this.checkCourseClearEffects();

    // Render!
    this.renderer.render( this.scene, this.camera );
  }
}

try {
  let demo = new MuJoCoDemo();
  await demo.init();
} catch (error) {
  console.error(error);
  showStartupError(error);
  throw error;
}
