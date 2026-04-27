import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Get placeholders from window object injected by Python/HTML
const GLB_DATA_URL = window.GLB_DATA;
let cameraMode = window.CAMERA_MODE_DATA;

const container = document.getElementById('canvas-container');
const instructions = document.getElementById('instructions');
const hud = document.getElementById('hud');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
// 關閉 Fog 讓全域背景乾淨
// scene.fog = new THREE.Fog(0x0f172a, 10, 100);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 1.5, 5);

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
// Fix the Aspect Ratio bug by calculating dimensions relative to the parent bounding box
const containerRect = container.getBoundingClientRect();
renderer.setSize(containerRect.width, containerRect.height);
container.appendChild(renderer.domElement);

// lightness
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);

// unified controller
let orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.enableDamping = true;
orbitControls.dampingFactor = 0.05;

let moveForward = false, moveBackward = false, moveLeft = false, moveRight = false;
let moveUp = false, moveDown = false;
let moveFast = false;
let prevTime = performance.now();

instructions.style.display = 'none';
hud.innerText = "自由視角: 滑鼠左鍵旋轉 | 右鍵/WASD平移 | 空白鍵上升 | Ctrl下降 | 滾輪縮放 | Shift加速";

const onKeyDown = function (event) {
    switch (event.code) {
        case 'ArrowUp': case 'KeyW': moveForward = true; break;
        case 'ArrowLeft': case 'KeyA': moveLeft = true; break;
        case 'ArrowDown': case 'KeyS': moveBackward = true; break;
        case 'ArrowRight': case 'KeyD': moveRight = true; break;
        case 'ControlLeft': case 'ControlRight': moveDown = true; break;
        case 'Space': moveUp = true; break;
        case 'ShiftLeft': case 'ShiftRight': moveFast = true; break;
    }
};
const onKeyUp = function (event) {
    switch (event.code) {
        case 'ArrowUp': case 'KeyW': moveForward = false; break;
        case 'ArrowLeft': case 'KeyA': moveLeft = false; break;
        case 'ArrowDown': case 'KeyS': moveBackward = false; break;
        case 'ArrowRight': case 'KeyD': moveRight = false; break;
        case 'ControlLeft': case 'ControlRight': moveDown = false; break;
        case 'Space': moveUp = false; break;
        case 'ShiftLeft': case 'ShiftRight': moveFast = false; break;
    }
};
document.addEventListener('keydown', onKeyDown);
document.addEventListener('keyup', onKeyUp);

// floor grid default
let gridHelper = new THREE.GridHelper(12, 12, 0x8b5cf6, 0x444444);
gridHelper.material.opacity = 0.4;
gridHelper.material.transparent = true;
gridHelper.visible = false; // hidden
scene.add(gridHelper);

// GLB Loading is now handled via postMessage 'load_glb_b64' event to avoid OOM

function animate() {
    requestAnimationFrame(animate);

    const time = performance.now();
    const delta = (time - prevTime) / 1000;
    prevTime = time;

    // Handle WASD QE movement
    if (moveForward || moveBackward || moveLeft || moveRight || moveUp || moveDown) {
        // Dynamic speed based on max scene dimension could be used, but static 5.0 is fine for base.
        // If the user wants to move fast, shift gives 5x speed
        let speed = moveFast ? 50.0 : 10.0;

        // If the camera is very far (large model), scale speed automatically
        const distToTarget = camera.position.distanceTo(orbitControls.target);
        if (distToTarget > 100) speed *= (distToTarget / 50.0);

        const actualSpeed = speed * delta;

        const dir = new THREE.Vector3();
        camera.getWorldDirection(dir);

        const right = new THREE.Vector3();
        right.crossVectors(camera.up, dir).normalize();

        const up = new THREE.Vector3().copy(camera.up);

        const moveVec = new THREE.Vector3();
        if (moveForward) moveVec.add(dir);
        if (moveBackward) moveVec.sub(dir);
        if (moveLeft) moveVec.add(right);
        if (moveRight) moveVec.sub(right);
        if (moveUp) moveVec.add(up);
        if (moveDown) moveVec.sub(up);

        moveVec.normalize().multiplyScalar(actualSpeed);

        camera.position.add(moveVec);
        orbitControls.target.add(moveVec);
    }

    orbitControls.update();
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', onWindowResize);
function onWindowResize() {
    const containerRect = container.getBoundingClientRect();
    camera.aspect = containerRect.width / containerRect.height;
    camera.updateProjectionMatrix();
    renderer.setSize(containerRect.width, containerRect.height);
}

let isRecording = false;
let recordedTrajectory = [];
let recordIntervalId = null;
let loadedModels = [];

// catch Gradio screenshot request
window.addEventListener('message', (event) => {

    function getCameraInfo() {
        const info = {
            position: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
            rotation: { x: camera.rotation.x, y: camera.rotation.y, z: camera.rotation.z },
            fov: camera.fov,
            near: camera.near,
            far: camera.far,
            aspect: camera.aspect,
            mode: 'free_cam',
            target: {
                x: orbitControls.target.x,
                y: orbitControls.target.y,
                z: orbitControls.target.z
            }
        };
        return info;
    }

    if (event.data && event.data.type === 'request_camera') {
        const cameraInfo = getCameraInfo();
        console.log('收到 request_camera 指令，準備回傳:', cameraInfo);
        event.source.postMessage({
            type: 'camera_result',
            camera: cameraInfo
        }, event.origin);
    }
    else if (event.data && event.data.type === 'load_glb_b64') {
        const dataUrl = event.data.data;
        const gltfLoader = new GLTFLoader();
        console.log('解析 GLB Base64 資料...');

        // Remove old models to prevent overlap
        loadedModels.forEach(model => scene.remove(model));
        loadedModels = [];

        gltfLoader.load(dataUrl, function (gltf) {
            const object = gltf.scene;
            scene.add(object);
            loadedModels.push(object);
            console.log('GLB 模型載入成功');
            const box = new THREE.Box3().setFromObject(object);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);

            // 動態相機遠近與目標設定
            const fovRad = camera.fov * (Math.PI / 180);
            const distanceY = size.y / (2 * Math.tan(fovRad / 2));
            const distanceX = size.x / (2 * camera.aspect * Math.tan(fovRad / 2));
            const distanceZ = size.z / 2;
            const distance = maxDim > 0 ? (Math.max(distanceY, distanceX) + distanceZ) * 1.2 : 5;

            camera.position.set(center.x, center.y + distance * 0.3, center.z + distance);
            camera.near = maxDim > 0 ? Math.min(0.1, maxDim * 0.001) : 0.1;
            camera.far = maxDim > 0 ? Math.max(10000, maxDim * 100) : 10000;
            camera.updateProjectionMatrix();

            if (orbitControls) {
                orbitControls.target.copy(center);
                orbitControls.update();
            }

            // dynamic changing grid size 
            scene.remove(gridHelper);
            gridHelper = new THREE.GridHelper(maxDim * 3, 50, 0x8b5cf6, 0xaaaaaa);
            gridHelper.position.set(center.x, box.min.y, center.z);
            gridHelper.material.opacity = 0.5;
            gridHelper.material.transparent = true;
            gridHelper.visible = false; // hidden
            scene.add(gridHelper);

            // compute normals + set materials
            object.traverse(function (child) {
                if (child.isMesh) {
                    if (!child.geometry.attributes.normal) {
                        child.geometry.computeVertexNormals();
                    }
                    // fix texture problem, ensure surrounding is white
                    if (child.material) {
                        if (child.material.color && child.material.color.getHex() === 0x000000) {
                            child.material.color.setHex(0xffffff);
                        }
                        child.material.side = THREE.DoubleSide;
                        child.material.needsUpdate = true;
                    }
                }
            });

        }, undefined, function (error) {
            console.error('載入 GLB 失敗:', error);
        });
    }
    else if (event.data && event.data.type === 'start_record') {
        if (isRecording) return;
        isRecording = true;
        recordedTrajectory = [];
        console.log('開始錄製軌跡 2fps');
        recordIntervalId = setInterval(() => {
            recordedTrajectory.push(getCameraInfo());
        }, 500); // 2 frames per second
    }
    else if (event.data && event.data.type === 'stop_record') {
        if (!isRecording) return;
        isRecording = false;
        clearInterval(recordIntervalId);
        console.log('停止錄製軌跡，共收集:', recordedTrajectory.length, 'frames');
        event.source.postMessage({
            type: 'trajectory_result',
            trajectory: recordedTrajectory
        }, event.origin);
    }
    else if (event.data && event.data.type === 'request_depth') {
        const oldBackground = scene.background;
        scene.background = new THREE.Color(0xffffff);

        scene.overrideMaterial = new THREE.MeshDepthMaterial();
        renderer.render(scene, camera);

        const dataURL = renderer.domElement.toDataURL('image/png');

        scene.overrideMaterial = null;
        scene.background = oldBackground;
        renderer.render(scene, camera);

        const cameraInfo = getCameraInfo();

        console.log('[Camera]', JSON.stringify(cameraInfo));
        event.source.postMessage({
            type: 'depth_result',
            depth: dataURL,
            camera: cameraInfo
        }, event.origin);
    }
});
