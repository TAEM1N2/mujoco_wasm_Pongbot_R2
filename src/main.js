
import * as THREE           from 'three';
import { GUI              } from '../node_modules/three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls    } from '../node_modules/three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { setupGUI, downloadExampleScenesFolder, loadSceneFromURL, drawTendonsAndFlex, getPosition, getQuaternion, toMujocoPos, standardNormal } from './mujocoUtils.js';
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
var initialScene = "pongbot_r2/Pongbot_R2_no_link_ver2.xml";
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
    this.params = { scene: initialScene, paused: false, help: false, ctrlnoiserate: 0.0, ctrlnoisestd: 0.0, keyframeNumber: 0 };
    this.mujoco_time = 0.0;
    this.bodies  = {}, this.lights = {};
    this.tmpVec  = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();
    this.updateGUICallbacks = [];
    this.pongbotPD = {
      active: false,
      startTime: 0.0,
      duration: 2.0,
      kp: 500.0,
      kd: 1.5,
      // KP_joint in request is interpreted as KN_JOINT in this model.
      targets: { HR: 0.0, HP: 0.8, KN: -1.5 },
      channels: [],
    };

    this.container = document.createElement( 'div' );
    document.body.appendChild( this.container );

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.001, 100 );
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(2.0, 1.7, 1.7);
    this.scene.add(this.camera);

    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5 );

    this.ambientLight = new THREE.AmbientLight( 0xffffff, 0.1 * 3.14 );
    this.ambientLight.name = 'AmbientLight';
    this.scene.add( this.ambientLight );

    this.spotlight = new THREE.SpotLight();
    this.spotlight.angle = 1.11;
    this.spotlight.distance = 10000;
    this.spotlight.penumbra = 0.5;
    this.spotlight.castShadow = true; // default false
    this.spotlight.intensity = this.spotlight.intensity * 3.14 * 10.0;
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
    // Initialize the Drag State Manager.
    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);
  }

  async init() {
    // Download the the examples to MuJoCo's virtual file system
    await downloadExampleScenesFolder(mujoco);

    // Initialize the three.js Scene using the .xml Model in initialScene
    [this.model, this.data, this.bodies, this.lights] =
      await loadSceneFromURL(mujoco, initialScene, this);
    this.configurePongbotPDChannels();
    this.startPongbotPDControl();

    this.gui = new GUI();
    setupGUI(this);
  }

  startPongbotPDControl() {
    if (!this.pongbotPD.channels.length) {
      console.warn("Pongbot PD channels were not found in current scene.");
      return;
    }
    this.pongbotPD.active = true;
    this.pongbotPD.startTime = this.data.time;
    for (let i = 0; i < this.pongbotPD.channels.length; i++) {
      const ch = this.pongbotPD.channels[i];
      ch.startQ = this.data.qpos[ch.qposAdr];
    }
  }

  configurePongbotPDChannels() {
    const textDecoder = new TextDecoder("utf-8");
    const nullChar = textDecoder.decode(new ArrayBuffer(1));
    const getNameAt = (adr) => textDecoder.decode(this.model.names.subarray(adr)).split(nullChar)[0];

    const findJointIDByName = (name) => {
      for (let i = 0; i < this.model.njnt; i++) {
        if (getNameAt(this.model.name_jntadr[i]) === name) { return i; }
      }
      return -1;
    };

    const findActuatorIDByName = (name) => {
      for (let i = 0; i < this.model.nu; i++) {
        if (getNameAt(this.model.name_actuatoradr[i]) === name) { return i; }
      }
      return -1;
    };

    const legs = ["FL", "FR", "RL", "RR"];
    const types = ["HR", "HP", "KN"];
    this.pongbotPD.channels = [];

    for (let l = 0; l < legs.length; l++) {
      for (let t = 0; t < types.length; t++) {
        const type = types[t];
        const name = legs[l] + "_" + type + "_JOINT";
        const jointID = findJointIDByName(name);
        const actuatorID = findActuatorIDByName(name);
        if (jointID < 0 || actuatorID < 0) { continue; }
        this.pongbotPD.channels.push({
          actuatorID: actuatorID,
          actuatorName: name,
          qposAdr: this.model.jnt_qposadr[jointID],
          qvelAdr: this.model.jnt_dofadr[jointID],
          target: this.pongbotPD.targets[type],
          startQ: 0.0,
        });
      }
    }
  }

  applyPongbotPDControl() {
    if (!this.pongbotPD.active) { return; }

    const ctrl = this.data.ctrl;
    const qpos = this.data.qpos;
    const qvel = this.data.qvel;
    const actRange = this.model.actuator_ctrlrange;
    const elapsed = this.data.time - this.pongbotPD.startTime;
    const tau = Math.max(0.0, Math.min(1.0, elapsed / this.pongbotPD.duration));
    const blend = 0.5 * (1.0 - Math.cos(Math.PI * tau));

    for (let i = 0; i < this.pongbotPD.channels.length; i++) {
      const ch = this.pongbotPD.channels[i];
      const qHome = ch.startQ + (ch.target - ch.startQ) * blend;
      const qRef = qHome;
      const q = qpos[ch.qposAdr];
      const qd = qvel[ch.qvelAdr];
      let u = this.pongbotPD.kp * (qRef - q) - this.pongbotPD.kd * qd;
      const uMin = actRange[2 * ch.actuatorID + 0];
      const uMax = actRange[2 * ch.actuatorID + 1];
      u = Math.max(uMin, Math.min(uMax, u));
      ctrl[ch.actuatorID] = u;
      this.params[ch.actuatorName] = u;
    }
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize( window.innerWidth, window.innerHeight );
  }

  render(timeMS) {
    this.controls.update();

    if (!this.params["paused"]) {
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

        this.applyPongbotPDControl();

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
