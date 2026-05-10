import * as THREE from 'three';
import { MuJoCoDemo } from './main.js';

function createGroundTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  const colors = ["#394b5c", "#46596a"];
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      ctx.fillStyle = colors[(x + y) % 2];
      ctx.fillRect(x * 32, y * 32, 32, 32);
    }
  }
  ctx.strokeStyle = "rgba(255, 255, 255, 0.20)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i++) {
    ctx.beginPath();
    ctx.moveTo(i * 32, 0);
    ctx.lineTo(i * 32, 256);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * 32);
    ctx.lineTo(256, i * 32);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(255, 200, 87, 0.45)";
  ctx.fillRect(122, 0, 12, 256);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(30, 30);
  texture.needsUpdate = true;
  return texture;
}

export async function reloadFunc() {
  // Delete the old scene and load the new scene
  this.scene.remove(this.scene.getObjectByName("MuJoCo Root"));
  [this.model, this.data, this.bodies, this.lights] =
    await loadSceneFromURL(this.mujoco, this.params.scene, this);
  if (typeof this.configureTerrainModes === "function") {
    this.configureTerrainModes();
  }
  if (typeof this.applyTerrainMode === "function") {
    this.applyTerrainMode();
  }
  if (typeof this.configureCameraFollow === "function") {
    this.configureCameraFollow();
  }
  if (typeof this.startImplicitPolicyController === "function") {
    await this.startImplicitPolicyController();
  }
  this.mujoco.mj_forward(this.model, this.data);
  for (let i = 0; i < this.updateGUICallbacks.length; i++) {
    this.updateGUICallbacks[i](this.model, this.data, this.params);
  }
}

/** @param {MuJoCoDemo} parentContext*/
export function setupGUI(parentContext) {

  // Make sure we reset the camera when the scene is changed or reloaded.
  parentContext.updateGUICallbacks.length = 0;
  parentContext.updateGUICallbacks.push((model, data, params) => {
    // TODO: Use free camera parameters from MuJoCo
    parentContext.camera.position.set(2.0, 1.7, 1.7);
    parentContext.controls.target.set(0, 0.7, 0);
    parentContext.controls.update(); });

  // Add scene selection dropdown.
  let reload = reloadFunc.bind(parentContext);
  parentContext.gui.add(parentContext.params, 'scene', {
    "Pongbot R2": "pongbot_r2/PONGBOT_R2_V2.xml",
  }).name('Example Scene').onChange(reload);

  let simulationFolder = parentContext.gui.addFolder("Simulation");

  // Add pause simulation checkbox.
  // Parameters:
  //  Under "Simulation" folder.
  //  Name: "Pause Simulation".
  //  When paused, a "pause" text in white is displayed in the top left corner.
  //  Can also be triggered by pressing the spacebar.
  const pauseSimulation = simulationFolder.add(parentContext.params, 'paused').name('Pause Simulation');
  pauseSimulation.onChange((value) => {
    if (value) {
      const pausedText = document.createElement('div');
      pausedText.style.position = 'absolute';
      pausedText.style.top = '10px';
      pausedText.style.left = '10px';
      pausedText.style.color = 'white';
      pausedText.style.font = 'normal 18px Arial';
      pausedText.innerHTML = 'pause';
      parentContext.container.appendChild(pausedText);
    } else {
      parentContext.container.removeChild(parentContext.container.lastChild);
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.code === 'Space') {
      parentContext.params.paused = !parentContext.params.paused;
      pauseSimulation.setValue(parentContext.params.paused);
      event.preventDefault();
    }
  });
  // Add reset simulation button.
  // Parameters:
  //  Under "Simulation" folder.
  //  Name: "Reset".
  //  When pressed, resets the simulation to the initial state.
  //  Can also be triggered by pressing backspace.
  const resetSimulation = () => {
    parentContext.mujoco.mj_resetData(parentContext.model, parentContext.data);
    if (typeof parentContext.resetCourseProgress === "function") {
      parentContext.resetCourseProgress();
    }
    if (typeof parentContext.applyTerrainMode === "function") {
      parentContext.applyTerrainMode();
    }
    parentContext.mujoco.mj_forward(parentContext.model, parentContext.data);
    if (parentContext.implicitPolicy) {
      parentContext.implicitPolicy.reset({ immediatePolicy: true });
    }
  };
  simulationFolder.add({reset: () => { resetSimulation(); }}, 'reset').name('Reset');
  document.addEventListener('keydown', (event) => {
    if (event.code === 'Backspace') { resetSimulation(); event.preventDefault(); }});

  simulationFolder.add(parentContext.params, "terrainMode", {
    "Flat": "flat",
    "Stair": "stair",
  }).name("Terrain").onChange((value) => {
    if (typeof parentContext.setTerrainMode === "function") {
      parentContext.setTerrainMode(value);
    }
  }).listen();
  simulationFolder.add(parentContext.params, "autoForward").name("Forward 0.5 m/s").onChange(() => {
    if (typeof parentContext.updateControlHelpOverlay === "function") {
      parentContext.updateControlHelpOverlay();
    }
  }).listen();

  // Add function that resets the camera to the default position.
  // Can be triggered by pressing ctrl + A.
  document.addEventListener('keydown', (event) => {
    if (event.ctrlKey && event.code === 'KeyA') {
      if (typeof parentContext.resetCameraView === "function") {
        parentContext.resetCameraView();
      }
      event.preventDefault();
    }
  });

  parentContext.gui.open();
}


/** Loads a scene for MuJoCo
 * @param {mujoco} mujoco This is a reference to the mujoco namespace object
 * @param {string} filename This is the name of the .xml file in the /working/ directory of the MuJoCo/Emscripten Virtual File System
 * @param {MuJoCoDemo} parent The three.js Scene Object to add the MuJoCo model elements to
 */
export async function loadSceneFromURL(mujoco, filename, parent) {
    const fastPongbotLoad = filename.includes("pongbot_r2/");

    // Free the old data.
    if (parent.data != null) {
      parent.data.delete();
      parent.model = null;
      parent.data  = null;
    }

    // Load in the state from XML.
    parent.model = mujoco.MjModel.loadFromXML("/working/"+filename);
    parent.data  = new mujoco.MjData(parent.model);

    let model = parent.model;
    let data = parent.data;

    // Decode the null-terminated string names.
    let textDecoder = new TextDecoder("utf-8");
    let names_array = new Uint8Array(model.names);
    let fullString = textDecoder.decode(model.names);
    let names = fullString.split(textDecoder.decode(new ArrayBuffer(1)));

    // Create the root object.
    let mujocoRoot = new THREE.Group();
    mujocoRoot.name = "MuJoCo Root";
    parent.scene.add(mujocoRoot);

    /** @type {Object.<number, THREE.Group>} */
    let bodies = {};
    /** @type {Object.<number, THREE.BufferGeometry>} */
    let meshes = {};
    /** @type {THREE.Light[]} */
    let lights = [];

    // Default material definition.
    let material = new THREE.MeshPhysicalMaterial();
    material.color = new THREE.Color(1, 1, 1);
    
    // Loop through the MuJoCo geoms and recreate them in three.js.
    for (let g = 0; g < model.ngeom; g++) {
      // Only visualize geom groups up to 2 (same default behavior as simulate).
      if (!(model.geom_group[g] < 3)) { continue; }

      // Get the body ID and type of the geom.
      let b    = model.geom_bodyid[g];
      let type = model.geom_type  [g];
      let size = [
        model.geom_size[(g*3) + 0],
        model.geom_size[(g*3) + 1],
        model.geom_size[(g*3) + 2]
      ];

      // Create the body if it doesn't exist.
      if (!(b in bodies)) {
        bodies[b] = new THREE.Group();
        
        let start_idx = model.name_bodyadr[b];
        let end_idx = start_idx;
        while (end_idx < names_array.length && names_array[end_idx] !== 0) {
          end_idx++;
        }
        let name_buffer = names_array.subarray(start_idx, end_idx);
        bodies[b].name = textDecoder.decode(name_buffer);
        
        bodies[b].bodyID = b;
        bodies[b].has_custom_mesh = false;
      }

      // Set the default geometry. In MuJoCo, this is a sphere.
      let geometry = new THREE.SphereGeometry(size[0] * 0.5);
      if (type == mujoco.mjtGeom.mjGEOM_PLANE.value) {
        // Special handling for plane later.
      } else if (type == mujoco.mjtGeom.mjGEOM_HFIELD.value) {
        // TODO: Implement this.
      } else if (type == mujoco.mjtGeom.mjGEOM_SPHERE.value) {
        geometry = new THREE.SphereGeometry(size[0]);
      } else if (type == mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
        geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 20, 20);
      } else if (type == mujoco.mjtGeom.mjGEOM_ELLIPSOID.value) {
        geometry = new THREE.SphereGeometry(1); // Stretch this below
      } else if (type == mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
        geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0);
      } else if (type == mujoco.mjtGeom.mjGEOM_BOX.value) {
        geometry = new THREE.BoxGeometry(size[0] * 2.0, size[2] * 2.0, size[1] * 2.0);
      } else if (type == mujoco.mjtGeom.mjGEOM_MESH.value) {
        let meshID = model.geom_dataid[g];

        if (!(meshID in meshes)) {
          geometry = new THREE.BufferGeometry();

          let vertex_buffer = new Float32Array(model.mesh_vert.subarray(
             model.mesh_vertadr[meshID] * 3,
            (model.mesh_vertadr[meshID]  + model.mesh_vertnum[meshID]) * 3));
          for (let v = 0; v < vertex_buffer.length; v+=3){
            //vertex_buffer[v + 0] =  vertex_buffer[v + 0];
            let temp             =  vertex_buffer[v + 1];
            vertex_buffer[v + 1] =  vertex_buffer[v + 2];
            vertex_buffer[v + 2] = -temp;
          }

          let normal_buffer = new Float32Array(model.mesh_normal.subarray(
             model.mesh_normaladr[meshID] * 3,
            (model.mesh_normaladr[meshID]  + model.mesh_normalnum[meshID]) * 3));
          for (let v = 0; v < normal_buffer.length; v+=3){
            //normal_buffer[v + 0] =  normal_buffer[v + 0];
            let temp             =  normal_buffer[v + 1];
            normal_buffer[v + 1] =  normal_buffer[v + 2];
            normal_buffer[v + 2] = -temp;
          }

          let face_to_vertex_buffer = model.mesh_face.subarray(
             model.mesh_faceadr[meshID] * 3,
            (model.mesh_faceadr[meshID]  + model.mesh_facenum[meshID]) * 3);

          geometry.setAttribute("position", new THREE.BufferAttribute(vertex_buffer, 3));
          geometry.setIndex    (Array.from(face_to_vertex_buffer));
          if (fastPongbotLoad && normal_buffer.length === vertex_buffer.length) {
            geometry.setAttribute("normal", new THREE.BufferAttribute(normal_buffer, 3));
          } else {
            let uv_buffer = model.mesh_texcoord.subarray(
               model.mesh_texcoordadr[meshID] * 2,
              (model.mesh_texcoordadr[meshID]  + model.mesh_texcoordnum[meshID]) * 2);
            let face_to_uv_buffer = model.mesh_facetexcoord.subarray(
               model.mesh_faceadr[meshID] * 3,
              (model.mesh_faceadr[meshID]  + model.mesh_facenum[meshID]) * 3);
            let face_to_normal_buffer = model.mesh_facenormal.subarray(
               model.mesh_faceadr[meshID] * 3,
              (model.mesh_faceadr[meshID]  + model.mesh_facenum[meshID]) * 3);

            let swizzled_uv_buffer      = new Float32Array((vertex_buffer.length / 3) * 2);
            let swizzled_normal_buffer  = new Float32Array(vertex_buffer.length);
            for (let t = 0; t < face_to_vertex_buffer.length / 3; t++) {
              let vi0 = face_to_vertex_buffer[(t * 3) + 0];
              let vi1 = face_to_vertex_buffer[(t * 3) + 1];
              let vi2 = face_to_vertex_buffer[(t * 3) + 2];
              let uvi0 = face_to_uv_buffer[(t * 3) + 0];
              let uvi1 = face_to_uv_buffer[(t * 3) + 1];
              let uvi2 = face_to_uv_buffer[(t * 3) + 2];
              let nvi0 = face_to_normal_buffer[(t * 3) + 0];
              let nvi1 = face_to_normal_buffer[(t * 3) + 1];
              let nvi2 = face_to_normal_buffer[(t * 3) + 2];
              swizzled_uv_buffer[(vi0 * 2) + 0] = uv_buffer[(uvi0 * 2) + 0];
              swizzled_uv_buffer[(vi0 * 2) + 1] = uv_buffer[(uvi0 * 2) + 1];
              swizzled_uv_buffer[(vi1 * 2) + 0] = uv_buffer[(uvi1 * 2) + 0];
              swizzled_uv_buffer[(vi1 * 2) + 1] = uv_buffer[(uvi1 * 2) + 1];
              swizzled_uv_buffer[(vi2 * 2) + 0] = uv_buffer[(uvi2 * 2) + 0];
              swizzled_uv_buffer[(vi2 * 2) + 1] = uv_buffer[(uvi2 * 2) + 1];
              swizzled_normal_buffer[(vi0 * 3) + 0] = normal_buffer[(nvi0 * 3) + 0];
              swizzled_normal_buffer[(vi0 * 3) + 1] = normal_buffer[(nvi0 * 3) + 1];
              swizzled_normal_buffer[(vi0 * 3) + 2] = normal_buffer[(nvi0 * 3) + 2];
              swizzled_normal_buffer[(vi1 * 3) + 0] = normal_buffer[(nvi1 * 3) + 0];
              swizzled_normal_buffer[(vi1 * 3) + 1] = normal_buffer[(nvi1 * 3) + 1];
              swizzled_normal_buffer[(vi1 * 3) + 2] = normal_buffer[(nvi1 * 3) + 2];
              swizzled_normal_buffer[(vi2 * 3) + 0] = normal_buffer[(nvi2 * 3) + 0];
              swizzled_normal_buffer[(vi2 * 3) + 1] = normal_buffer[(nvi2 * 3) + 1];
              swizzled_normal_buffer[(vi2 * 3) + 2] = normal_buffer[(nvi2 * 3) + 2];
            }
            geometry.setAttribute("normal", new THREE.BufferAttribute(swizzled_normal_buffer, 3));
            geometry.setAttribute("uv"    , new THREE.BufferAttribute(swizzled_uv_buffer, 2));
            geometry.computeVertexNormals(); // fallback for non-pongbot assets
          }
          meshes[meshID] = geometry;
        } else {
          geometry = meshes[meshID];
        }

        bodies[b].has_custom_mesh = true;
      }
      // Done with geometry creation.

      // Set the Material Properties of incoming bodies
      let texture = undefined;
      let color = [
        model.geom_rgba[(g * 4) + 0],
        model.geom_rgba[(g * 4) + 1],
        model.geom_rgba[(g * 4) + 2],
        model.geom_rgba[(g * 4) + 3]];
      if (model.geom_matid[g] != -1) {
        let matId = model.geom_matid[g];
        color = [
          model.mat_rgba[(matId * 4) + 0],
          model.mat_rgba[(matId * 4) + 1],
          model.mat_rgba[(matId * 4) + 2],
          model.mat_rgba[(matId * 4) + 3]];

        // Construct Texture from model.tex_data
        texture = undefined;
        // mat_texid is now a matrix (nmat x mjNTEXROLE)
        // We use mjTEXROLE_RGB (value 1) for standard diffuse/color textures
        const mjNTEXROLE = 10; // Total number of texture roles
        const mjTEXROLE_RGB = 1; // RGB texture role
        let texId = model.mat_texid[(matId * mjNTEXROLE) + mjTEXROLE_RGB];
        
        if (texId != -1) {
          let width    = model.tex_width [texId];
          let height   = model.tex_height[texId];
          let offset   = model.tex_adr   [texId];
          let channels = model.tex_nchannel[texId];
          let texData  = model.tex_data;
          let rgbaArray = new Uint8Array(width * height * 4);
          for (let p = 0; p < width * height; p++){
            rgbaArray[(p * 4) + 0] = texData[offset + ((p * channels) + 0)];
            rgbaArray[(p * 4) + 1] = channels > 1 ? texData[offset + ((p * channels) + 1)] : rgbaArray[(p * 4) + 0];
            rgbaArray[(p * 4) + 2] = channels > 2 ? texData[offset + ((p * channels) + 2)] : rgbaArray[(p * 4) + 0];
            rgbaArray[(p * 4) + 3] = channels > 3 ? texData[offset + ((p * channels) + 3)] : 255;
          }
          texture = new THREE.DataTexture(rgbaArray, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
          if (texId == 2) {
            texture.repeat = new THREE.Vector2(50, 50);
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
          } else {
            texture.repeat = new THREE.Vector2(model.mat_texrepeat[(model.geom_matid[g] * 2) + 0],
                                               model.mat_texrepeat[(model.geom_matid[g] * 2) + 1]);
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
          }

          texture.needsUpdate = true;
        }
      }

      // Create a new material for each geom to avoid cross-contamination
      let currentMaterial = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(color[0], color[1], color[2]),
        transparent: color[3] < 1.0,
        opacity: color[3]/255.,
        specularIntensity: model.geom_matid[g] != -1 ?       model.mat_specular   [model.geom_matid[g]] : undefined,
        reflectivity     : model.geom_matid[g] != -1 ?       model.mat_reflectance[model.geom_matid[g]] : undefined,
        roughness        : model.geom_matid[g] != -1 ? 1.0 - model.mat_shininess  [model.geom_matid[g]] : undefined,
        metalness        : model.geom_matid[g] != -1 ?       0.1 : undefined, //model.mat_metallic   [model.geom_matid[g]]
        map              : texture
      });

      let mesh;// = new THREE.Mesh();
      if (type == 0) {
        const groundMaterial = new THREE.MeshPhysicalMaterial({
          color: new THREE.Color(0.42, 0.48, 0.52),
          roughness: 0.92,
          metalness: 0.0,
          map: texture || createGroundTexture(),
        });
        mesh = new THREE.Mesh(new THREE.PlaneGeometry(120, 120), groundMaterial);
        mesh.rotateX( - Math.PI / 2 );
      } else {
        mesh = new THREE.Mesh(geometry, currentMaterial);
      }

      mesh.castShadow = g == 0 ? false : true;
      mesh.receiveShadow = type != 7;
      mesh.bodyID = b;
      bodies[b].add(mesh);
      getPosition  (model.geom_pos, g, mesh.position  );
      if (type != 0) { getQuaternion(model.geom_quat, g, mesh.quaternion); }
      if (type == 4) { mesh.scale.set(size[0], size[2], size[1]); } // Stretch the Ellipsoid
    }

    // Parse tendons.
    let tendonMat = new THREE.MeshPhongMaterial();
    tendonMat.color = new THREE.Color(0.8, 0.3, 0.3);
    mujocoRoot.cylinders = new THREE.InstancedMesh(
        new THREE.CylinderGeometry(1, 1, 1),
        tendonMat, 1023);
    mujocoRoot.cylinders.receiveShadow = true;
    mujocoRoot.cylinders.castShadow    = true;
    mujocoRoot.add(mujocoRoot.cylinders);
    mujocoRoot.spheres = new THREE.InstancedMesh(
        new THREE.SphereGeometry(1, 10, 10),
        tendonMat, 1023);
    mujocoRoot.spheres.receiveShadow = true;
    mujocoRoot.spheres.castShadow    = true;
    mujocoRoot.add(mujocoRoot.spheres);

    // Parse lights.
    for (let l = 0; l < model.nlight; l++) {
      let light = new THREE.DirectionalLight();
      if (model.light_type[l] == 0) {
        light = new THREE.SpotLight();
        light.angle = 1.51;//model.light_cutoffangle[l];
      } else if (model.light_type[l] == 1) {
        light = new THREE.DirectionalLight();
      } else if (model.light_type[l] == 2) {
        light = new THREE.PointLight();
      }else if (model.light_type[l] == 3) {
        light = new THREE.HemisphereLight();
      }

      light.angle = 1.11;

      light.decay = model.light_attenuation[l] * 100;
      light.penumbra = 0.5;
      light.castShadow = true; // default false
      light.intensity = light.intensity * 3.14 * 1.0;

      light.shadow.mapSize.width = 1024; // default
      light.shadow.mapSize.height = 1024; // default
      light.shadow.camera.near = 0.1; // default
      light.shadow.camera.far = 10; // default
      //bodies[model.light_bodyid()].add(light);
      if (bodies[0]) {
        bodies[0].add(light);
      } else {
        mujocoRoot.add(light);
      }
      lights.push(light);
    }
    if (model.nlight == 0) {
      let light = new THREE.DirectionalLight();
      mujocoRoot.add(light);
    }

    for (let b = 0; b < model.nbody; b++) {
      //let parent_body = model.body_parentid()[b];
      if (b == 0 || !bodies[0]) {
        mujocoRoot.add(bodies[b]);
      } else if(bodies[b]){
        bodies[0].add(bodies[b]);
      } else {
        console.log("Body without Geometry detected; adding to bodies", b, bodies[b]);
        bodies[b] = new THREE.Group(); bodies[b].name = names[b + 1]; bodies[b].bodyID = b; bodies[b].has_custom_mesh = false;
        bodies[0].add(bodies[b]);
      }
    }
  
    parent.mujocoRoot = mujocoRoot;

    return [model, data, bodies, lights];
}

export function drawTendonsAndFlex(mujocoRoot, model, data) {
  // Update tendon transforms.
  let identityQuat = new THREE.Quaternion();
  let numWraps = 0;
  if (mujocoRoot && mujocoRoot.cylinders) {
    let mat = new THREE.Matrix4();
    for (let t = 0; t < model.ntendon; t++) {
      let startW = data.ten_wrapadr[t];
      let r = model.tendon_width[t];
      for (let w = startW; w < startW + data.ten_wrapnum[t] -1 ; w++) {
        let tendonStart = getPosition(data.wrap_xpos, w    , new THREE.Vector3());
        let tendonEnd   = getPosition(data.wrap_xpos, w + 1, new THREE.Vector3());
        let tendonAvg   = new THREE.Vector3().addVectors(tendonStart, tendonEnd).multiplyScalar(0.5);

        let validStart = tendonStart.length() > 0.01;
        let validEnd   = tendonEnd  .length() > 0.01;

        if (validStart) { mujocoRoot.spheres.setMatrixAt(numWraps    , mat.compose(tendonStart, identityQuat, new THREE.Vector3(r, r, r))); }
        if (validEnd  ) { mujocoRoot.spheres.setMatrixAt(numWraps + 1, mat.compose(tendonEnd  , identityQuat, new THREE.Vector3(r, r, r))); }
        if (validStart && validEnd) {
          mat.compose(tendonAvg, identityQuat.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0), tendonEnd.clone().sub(tendonStart).normalize()),
            new THREE.Vector3(r, tendonStart.distanceTo(tendonEnd), r));
          mujocoRoot.cylinders.setMatrixAt(numWraps, mat);
          numWraps++;
        }
      }
    }

    let curFlexSphereInd = numWraps;
    let tempvertPos = new THREE.Vector3();
    let tempvertRad = new THREE.Vector3();
    for (let i = 0; i < model.nflex; i++) {
      for(let j = 0; j < model.flex_vertnum[i]; j++) {
        let vertIndex = model.flex_vertadr[i] + j;
        getPosition(data.flexvert_xpos, vertIndex, tempvertPos);
        let r   = 0.01;
        mat.compose(tempvertPos, identityQuat, tempvertRad.set(r, r, r));

        mujocoRoot.spheres.setMatrixAt(curFlexSphereInd, mat);
        curFlexSphereInd++;
      }
    }
    mujocoRoot.cylinders.count = numWraps;
    mujocoRoot.spheres  .count = curFlexSphereInd;
    mujocoRoot.cylinders.instanceMatrix.needsUpdate = true;
    mujocoRoot.spheres  .instanceMatrix.needsUpdate = true;
  }
}

/** Downloads the scenes/assets folder to MuJoCo's virtual filesystem
 * @param {mujoco} mujoco */
export async function downloadExampleScenesFolder(mujoco) {
  let allFiles = [
    // Keep only Pongbot R2 assets to avoid loading unused examples.
    "pongbot_r2/BODY.STL",
    "pongbot_r2/FL_CALF.STL",
    "pongbot_r2/FL_HIP.STL",
    "pongbot_r2/FL_THIGH.STL",
    "pongbot_r2/FL_TIP.STL",
    "pongbot_r2/FR_CALF.STL",
    "pongbot_r2/FR_HIP.STL",
    "pongbot_r2/FR_THIGH.STL",
    "pongbot_r2/FR_TIP.STL",
    "pongbot_r2/PONGBOT_R2_V2.xml",
    "pongbot_r2/RL_CALF.STL",
    "pongbot_r2/RL_HIP.STL",
    "pongbot_r2/RL_THIGH.STL",
    "pongbot_r2/RL_TIP.STL",
    "pongbot_r2/RR_CALF.STL",
    "pongbot_r2/RR_HIP.STL",
    "pongbot_r2/RR_THIGH.STL",
    "pongbot_r2/RR_TIP.STL",
  ];

  let requests = allFiles.map((url) => fetch("./assets/scenes/" + url));
  let responses = await Promise.all(requests);
  for (let i = 0; i < responses.length; i++) {
      let split = allFiles[i].split("/");
      let working = '/working/';
      for (let f = 0; f < split.length - 1; f++) {
          working += split[f];
          if (!mujoco.FS.analyzePath(working).exists) { mujoco.FS.mkdir(working); }
          working += "/";
      }

      const lower = allFiles[i].toLowerCase();
      if (lower.endsWith(".png") || lower.endsWith(".stl") || lower.endsWith(".skn")) {
          mujoco.FS.writeFile("/working/" + allFiles[i], new Uint8Array(await responses[i].arrayBuffer()));
      } else {
          mujoco.FS.writeFile("/working/" + allFiles[i], await responses[i].text());
      }
  }
}

/** Access the vector at index, swizzle for three.js, and apply to the target THREE.Vector3
 * @param {Float32Array|Float64Array} buffer
 * @param {number} index
 * @param {THREE.Vector3} target */
export function getPosition(buffer, index, target, swizzle = true) {
  if (swizzle) {
    return target.set(
       buffer[(index * 3) + 0],
       buffer[(index * 3) + 2],
      -buffer[(index * 3) + 1]);
  } else {
    return target.set(
       buffer[(index * 3) + 0],
       buffer[(index * 3) + 1],
       buffer[(index * 3) + 2]);
  }
}

/** Access the quaternion at index, swizzle for three.js, and apply to the target THREE.Quaternion
 * @param {Float32Array|Float64Array} buffer
 * @param {number} index
 * @param {THREE.Quaternion} target */
export function getQuaternion(buffer, index, target, swizzle = true) {
  if (swizzle) {
    return target.set(
      -buffer[(index * 4) + 1],
      -buffer[(index * 4) + 3],
       buffer[(index * 4) + 2],
      -buffer[(index * 4) + 0]);
  } else {
    return target.set(
       buffer[(index * 4) + 0],
       buffer[(index * 4) + 1],
       buffer[(index * 4) + 2],
       buffer[(index * 4) + 3]);
  }
}

/** Converts this Vector3's Handedness to MuJoCo's Coordinate Handedness
 * @param {THREE.Vector3} target */
export function toMujocoPos(target) { return target.set(target.x, -target.z, target.y); }

/** Standard normal random number generator using Box-Muller transform */
export function standardNormal() {
  return Math.sqrt(-2.0 * Math.log( Math.random())) *
         Math.cos ( 2.0 * Math.PI * Math.random()); }
