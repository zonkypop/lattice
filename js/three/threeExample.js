

import * as THREE from "three";
import { GLTFLoader } from "./GLTFLoader.js";
import { OrbitControls } from "./OrbitControls.js";




const gltfTest = async (scene) => {
  try {
    const data = Deno.readFileSync("./js/three/vipers_helmet.glb");
    const arrayBuffer = data.buffer.slice(
      data.byteOffset,
      data.byteOffset + data.byteLength
    );

    // browser
    // const response = await fetch("./vipers_helmet.glb");
    // const arrayBuffer = await response.arrayBuffer();

    let start = performance.now();
    const loader = new GLTFLoader();
    loader.parse(
      arrayBuffer,
      "", 
      (gltf) => {
        console.log("time to load GLTF: ", performance.now() - start);
        const obj = gltf.scene;
        scene.add(obj);
        obj.scale.set(3, 3, 3);
        loadedModel = obj;
      },
      (error) => {
        console.error("Error loading model:", error);
      }
    );
  } catch (err) {
    console.log(err);
  }
};


const runThree = async () => {
  const canvas = document.createElement("canvas");
  canvas.style.width = "800px";
  canvas.style.height = "600px";
  canvas.style.display = "block";
  document.body.appendChild(canvas);

  const renderer = new THREE.WebGPURenderer({
    antialias: true,
    samples: 4,
    colorBufferType: THREE.UnsignedByteType,
    canvas: canvas,
    logarithmicDepthBuffer: false,
  });
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  await renderer.init();

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x202020);

  const light = new THREE.DirectionalLight(0xffffff, 2);
  light.position.set(5, 5, 5);
  scene.add(light);

  const ambient = new THREE.AmbientLight(0xffffff, 0.1);
  scene.add(ambient);

  const aspect = window.innerWidth / window.innerHeight;
  const camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 100);
  camera.position.set(0, 0, 1);

  window.addEventListener("resize", () => {
    const width = window.innerWidth;
    const height = window.innerHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    renderer.setSize(width, height);
  });

  let controls = new OrbitControls(camera, renderer.domElement);

  await gltfTest(scene);


  const basicCube = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshBasicMaterial({ color: "#00FF00" })
  );
  basicCube.position.z = -2;

  function animate() {
    if (!basicCube.parent) {
      scene.add(basicCube);
    }
    try {
      controls.update();
      renderer.render(scene, camera);
    } catch (e) {
      console.error("render error", e);
    }

    requestAnimationFrame(animate);
  }

  animate();
};

runThree();
