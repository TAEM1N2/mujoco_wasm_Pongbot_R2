import { loadOnnxRuntime } from "./onnxRuntimeLoader.js";

const POLICY_CONFIG = {
  encoderPath: "./assets/policies/implicit/encoder.onnx",
  policyPath: "./assets/policies/implicit/policy.onnx",
  historyLength: 5,
  observationSize: 48,
  commandSize: 3,
  gaitCommand: [1.5, 0.5, 0.5, 0.05],
  homeByType: { HR: 0.0, HP: 0.64, KN: -1.25 },
  actionScale: 0.25,
  policyKp: 150.0,
  policyKd: 5.5,
  walkReadyKp: 1000.0,
  walkReadyKd: 10.0,
  walkReadyDuration: 3.0,
  fixedBaseReleaseTime: 5.0,
  autoPolicyStartTime: 6.2,
  initialBasePos: [0.0, 0.0, 0.63],
  policyDecimation: 5,
};

const POLICY_JOINT_ORDER = [
  "FL_HR_JOINT", "FR_HR_JOINT", "RL_HR_JOINT", "RR_HR_JOINT",
  "FL_HP_JOINT", "FR_HP_JOINT", "RL_HP_JOINT", "RR_HP_JOINT",
  "FL_KN_JOINT", "FR_KN_JOINT", "RL_KN_JOINT", "RR_KN_JOINT",
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function tensorSizeFromDims(dims) {
  if (!Array.isArray(dims) || dims.length === 0) { return 0; }
  return dims.reduce((acc, dim) => acc * (typeof dim === "number" && dim > 0 ? dim : 1), 1);
}

function getMetadataShape(session, inputName) {
  const metadata = session.inputMetadata || session.inputMetadata_ || {};
  const info = metadata[inputName];
  return info && Array.isArray(info.dimensions) ? info.dimensions : null;
}

function inferTensorDims(session, inputName, flatLength) {
  const shape = getMetadataShape(session, inputName);
  if (!shape || shape.length === 0) { return [flatLength]; }
  const dims = shape.map((dim, index) => {
    if (typeof dim === "number" && dim > 0) { return dim; }
    return index === 0 && shape.length > 1 ? 1 : flatLength;
  });
  const total = tensorSizeFromDims(dims);
  if (total === flatLength) { return dims; }
  if (shape.length > 1 && tensorSizeFromDims([1, flatLength]) === flatLength) { return [1, flatLength]; }
  return [flatLength];
}

function candidateTensorDims(session, inputName, flatLength) {
  const primary = inferTensorDims(session, inputName, flatLength);
  const candidates = [primary, [flatLength], [1, flatLength]];
  const seen = new Set();
  return candidates.filter((dims) => {
    const key = dims.join("x");
    if (seen.has(key)) { return false; }
    seen.add(key);
    return tensorSizeFromDims(dims) === flatLength;
  });
}

function quatConjugate(q) {
  return [q[0], -q[1], -q[2], -q[3]];
}

function rotateVectorByQuat(v, q) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  const tx = 2.0 * (y * vz - z * vy);
  const ty = 2.0 * (z * vx - x * vz);
  const tz = 2.0 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

export class ImplicitPolicyController {
  constructor(demo) {
    this.demo = demo;
    this.config = POLICY_CONFIG;
    this.textDecoder = new TextDecoder("utf-8");
    this.nullChar = this.textDecoder.decode(new ArrayBuffer(1));
    this.channels = [];
    this.command = [0.0, 0.0, 0.0];
    this.previousAction = new Array(12).fill(0.0);
    this.encoderOutput = [];
    this.history = [];
    this.stepCount = 0;
    this.policyActionValid = false;
    this.policyEnabled = false;
    this.inferencePending = false;
    this.ready = false;
    this.loading = false;
    this.status = "initializing";
    this.encoderSession = null;
    this.policySession = null;
    this.ort = null;
    this.rootBodyId = -1;
    this.rootJointId = -1;
    this.rootQposAdr = -1;
    this.rootQvelAdr = -1;
    this.skipStartupSequence = false;
  }

  async initialize() {
    if (this.loading || this.ready) { return; }
    this.loading = true;
    this.status = "loading onnx";
    this.configureModelBindings();
    this.captureWalkReadyStart();
    this.seedHistory();

    this.ort = await loadOnnxRuntime();
    this.encoderSession = await this.ort.InferenceSession.create(this.config.encoderPath, {
      executionProviders: ["wasm"],
    });
    this.policySession = await this.ort.InferenceSession.create(this.config.policyPath, {
      executionProviders: ["wasm"],
    });

    this.ready = true;
    this.loading = false;
    this.status = "walk-ready";
  }

  reset(options = {}) {
    const immediatePolicy = options.immediatePolicy === true;
    this.policyEnabled = this.ready;
    this.policyActionValid = false;
    this.previousAction.fill(0.0);
    this.encoderOutput = [];
    this.stepCount = this.ready ? this.config.policyDecimation : 0;
    this.configureModelBindings();
    this.captureWalkReadyStart();
    this.seedHistory();
    this.skipStartupSequence = immediatePolicy;
    this.status = this.ready ? "policy" : this.status;
  }

  configureModelBindings() {
    const model = this.demo.model;
    if (!model) { return; }

    this.rootBodyId = this.findBodyId("root");
    this.rootJointId = this.findJointId("root");
    if (this.rootJointId < 0) { this.rootJointId = this.findJointId("floating_base"); }
    this.rootQposAdr = this.rootJointId >= 0 ? model.jnt_qposadr[this.rootJointId] : -1;
    this.rootQvelAdr = this.rootJointId >= 0 ? model.jnt_dofadr[this.rootJointId] : -1;

    this.channels = POLICY_JOINT_ORDER.map((jointName, policyIndex) => {
      const jointId = this.findJointId(jointName);
      const actuatorId = this.findActuatorId(jointName);
      const type = jointName.includes("_HR_") ? "HR" : jointName.includes("_HP_") ? "HP" : "KN";
      if (jointId < 0 || actuatorId < 0) {
        throw new Error(`Missing PongBot joint or actuator: ${jointName}`);
      }
      return {
        policyIndex,
        jointName,
        actuatorId,
        qposAdr: model.jnt_qposadr[jointId],
        qvelAdr: model.jnt_dofadr[jointId],
        home: this.config.homeByType[type],
        startQ: 0.0,
      };
    });
  }

  captureWalkReadyStart() {
    const qpos = this.demo.data.qpos;
    for (const ch of this.channels) {
      ch.startQ = qpos[ch.qposAdr];
    }
  }

  seedHistory() {
    if (!this.demo.model || !this.demo.data || !this.channels.length) { return; }
    const obs = this.buildObservation();
    this.history = [];
    for (let i = 0; i < this.config.historyLength; i++) {
      this.history.push(new Float32Array(obs));
    }
  }

  findName(adr) {
    return this.textDecoder.decode(this.demo.model.names.subarray(adr)).split(this.nullChar)[0];
  }

  findJointId(name) {
    for (let i = 0; i < this.demo.model.njnt; i++) {
      if (this.findName(this.demo.model.name_jntadr[i]) === name) { return i; }
    }
    return -1;
  }

  findActuatorId(name) {
    for (let i = 0; i < this.demo.model.nu; i++) {
      if (this.findName(this.demo.model.name_actuatoradr[i]) === name) { return i; }
    }
    return -1;
  }

  findBodyId(name) {
    for (let i = 0; i < this.demo.model.nbody; i++) {
      if (this.findName(this.demo.model.name_bodyadr[i]) === name) { return i; }
    }
    return -1;
  }

  setCommand(x, y, yaw) {
    this.command[0] = clamp(x, -1.0, 1.0);
    this.command[1] = clamp(y, -1.0, 1.0);
    this.command[2] = clamp(yaw, -1.0, 1.0);
  }

  fixBaseUntilPolicyStart() {
    if (this.rootQposAdr < 0 || this.rootQvelAdr < 0) { return; }
    const qpos = this.demo.data.qpos;
    const qvel = this.demo.data.qvel;
    qpos[this.rootQposAdr + 0] = this.config.initialBasePos[0];
    qpos[this.rootQposAdr + 1] = this.config.initialBasePos[1];
    qpos[this.rootQposAdr + 2] = this.config.initialBasePos[2];
    qpos[this.rootQposAdr + 3] = 1.0;
    qpos[this.rootQposAdr + 4] = 0.0;
    qpos[this.rootQposAdr + 5] = 0.0;
    qpos[this.rootQposAdr + 6] = 0.0;
    for (let i = 0; i < 6; i++) { qvel[this.rootQvelAdr + i] = 0.0; }
  }

  applyWalkReadyControl() {
    const data = this.demo.data;
    const model = this.demo.model;
    const elapsed = data.time;
    const tau = clamp(elapsed / this.config.walkReadyDuration, 0.0, 1.0);
    const blend = 0.5 * (1.0 - Math.cos(Math.PI * tau));

    if (data.time < this.config.fixedBaseReleaseTime) {
      this.fixBaseUntilPolicyStart();
    }
    for (const ch of this.channels) {
      const target = ch.startQ + (ch.home - ch.startQ) * blend;
      const q = data.qpos[ch.qposAdr];
      const qd = data.qvel[ch.qvelAdr];
      let torque = this.config.walkReadyKp * (target - q) - this.config.walkReadyKd * qd;
      torque = clamp(torque, model.actuator_ctrlrange[ch.actuatorId * 2], model.actuator_ctrlrange[ch.actuatorId * 2 + 1]);
      data.ctrl[ch.actuatorId] = torque;
    }
  }

  computeBodyState() {
    if (this.rootBodyId < 0) {
      return { lin: [0, 0, 0], ang: [0, 0, 0], gravity: [0, 0, -1] };
    }
    const data = this.demo.data;
    const body = this.rootBodyId;
    const q = [
      data.xquat[body * 4 + 0],
      data.xquat[body * 4 + 1],
      data.xquat[body * 4 + 2],
      data.xquat[body * 4 + 3],
    ];
    const inv = quatConjugate(q);
    const worldAng = [data.cvel[body * 6 + 0], data.cvel[body * 6 + 1], data.cvel[body * 6 + 2]];
    const worldLin = [data.cvel[body * 6 + 3], data.cvel[body * 6 + 4], data.cvel[body * 6 + 5]];
    return {
      lin: rotateVectorByQuat(worldLin, inv),
      ang: rotateVectorByQuat(worldAng, inv),
      gravity: rotateVectorByQuat([0, 0, -1], inv),
    };
  }

  buildObservation() {
    const obs = [];
    const state = this.computeBodyState();
    obs.push(state.ang[0] * 0.25, state.ang[1] * 0.25, state.ang[2] * 0.25);
    obs.push(state.gravity[0], state.gravity[1], state.gravity[2]);

    for (const ch of this.channels) {
      obs.push(this.demo.data.qpos[ch.qposAdr] - ch.home);
    }
    for (const ch of this.channels) {
      obs.push(this.demo.data.qvel[ch.qvelAdr] * 0.05);
    }
    for (let i = 0; i < this.previousAction.length; i++) {
      obs.push(this.previousAction[i]);
    }

    const gaitIndex = (this.demo.data.time * this.config.gaitCommand[0]) % 1.0;
    const gaitAngle = 2.0 * Math.PI * gaitIndex;
    obs.push(Math.sin(gaitAngle), Math.cos(gaitAngle));
    for (const value of this.config.gaitCommand) { obs.push(value); }

    if (obs.length !== this.config.observationSize) {
      throw new Error(`Implicit observation size mismatch: ${obs.length}`);
    }
    return obs;
  }

  flattenHistory() {
    const flat = new Float32Array(this.config.historyLength * this.config.observationSize);
    for (let i = 0; i < this.history.length; i++) {
      flat.set(this.history[i], i * this.config.observationSize);
    }
    return flat;
  }

  async runSession(session, values, extraDims = []) {
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const input = values instanceof Float32Array ? values : new Float32Array(values);
    const dimsList = [];
    const seen = new Set();
    const addDims = (dims) => {
      if (!Array.isArray(dims) || tensorSizeFromDims(dims) !== input.length) { return; }
      const key = dims.join("x");
      if (seen.has(key)) { return; }
      seen.add(key);
      dimsList.push(dims);
    };
    for (const dims of extraDims) { addDims(dims); }
    for (const dims of candidateTensorDims(session, inputName, input.length)) { addDims(dims); }
    let lastError = null;
    for (const dims of dimsList) {
      try {
        const tensor = new this.ort.Tensor("float32", input, dims);
        const result = await session.run({ [inputName]: tensor });
        return Array.from(result[outputName].data);
      } catch (error) {
        lastError = error;
      }
    }
    throw new Error(
      `ONNX inference failed for input '${inputName}' length=${input.length} ` +
      `tried_dims=${dimsList.map((dims) => `[${dims.join(",")}]`).join(", ")}: ` +
      String(lastError && lastError.message ? lastError.message : lastError)
    );
  }

  async computePolicyAction() {
    const historyInput = this.flattenHistory();
    this.encoderOutput = await this.runSession(this.encoderSession, historyInput, [
      [1, this.config.historyLength, this.config.observationSize],
    ]);

    const currentObs = this.buildObservation();
    this.history.shift();
    this.history.push(new Float32Array(currentObs));

    const policyInput = new Float32Array(this.encoderOutput.length + currentObs.length + this.command.length);
    policyInput.set(this.encoderOutput, 0);
    policyInput.set(currentObs, this.encoderOutput.length);
    policyInput.set(this.command, this.encoderOutput.length + currentObs.length);

    const action = await this.runSession(this.policySession, policyInput);
    if (action.length !== this.channels.length) {
      throw new Error(`Policy output size mismatch: ${action.length}, expected ${this.channels.length}`);
    }
    return action.map((value) => clamp(value, -10.0, 10.0));
  }

  applyPolicyAction(action) {
    const data = this.demo.data;
    const model = this.demo.model;
    for (let i = 0; i < this.channels.length; i++) {
      const ch = this.channels[i];
      const target = ch.home + this.config.actionScale * action[i];
      const q = data.qpos[ch.qposAdr];
      const qd = data.qvel[ch.qvelAdr];
      let torque = this.config.policyKp * (target - q) - this.config.policyKd * qd;
      torque = clamp(torque, model.actuator_ctrlrange[ch.actuatorId * 2], model.actuator_ctrlrange[ch.actuatorId * 2 + 1]);
      data.ctrl[ch.actuatorId] = torque;
    }
  }

  stepControl() {
    if (!this.ready || !this.channels.length) {
      this.applyWalkReadyControl();
      return;
    }

    if (!this.skipStartupSequence && this.demo.data.time < this.config.autoPolicyStartTime) {
      this.status = this.demo.data.time < this.config.fixedBaseReleaseTime ? "walk-ready" : "settling";
      this.applyWalkReadyControl();
      this.previousAction.fill(0.0);
      return;
    }

    if (!this.policyEnabled) {
      this.policyEnabled = true;
      this.policyActionValid = false;
      this.inferencePending = false;
      this.seedHistory();
      this.status = "policy";
    }

    const shouldUpdatePolicy = !this.policyActionValid || this.stepCount % this.config.policyDecimation === 0;
    if (shouldUpdatePolicy && !this.inferencePending) {
      this.inferencePending = true;
      this.computePolicyAction()
        .then((action) => {
          this.previousAction = action;
          this.policyActionValid = true;
          this.applyPolicyAction(this.previousAction);
        })
        .catch((error) => {
          this.status = "policy error";
          this.demo.params.policyStatus = this.status;
          this.demo.params.paused = true;
          console.error(error);
        })
        .finally(() => {
          this.inferencePending = false;
        });
    }

    if (this.policyActionValid) {
      this.applyPolicyAction(this.previousAction);
    }
    this.stepCount += 1;
  }
}
