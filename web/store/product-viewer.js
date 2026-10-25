import * as THREE from '../vendor/three/three.module.js';
import { OrbitControls } from '../vendor/three/examples/jsm/controls/OrbitControls.js';
import URDFLoader from '../vendor/urdf-loader/URDFLoader.js';

// This viewer displays native geometry and recorded data. It does not run physics.
const URDF_URL = '/urdf/q1_inertia/quanta_robot_legged.urdf';
const SOURCES = {
  sidestep: {
    url: '/docs/results/upper_body_13115_baseline_replay.json',
    label: '歷史原生 PhysX 記錄',
    provenance: 'x86 · V5 13,115-update 原設定 · 7.2 秒歷史物理軌跡',
    physical: true,
    warning: '歷史記錄尚未通過完整操作驗收；不是本輪 Spark 的新成果。',
  },
  upper: {
    url: '/docs/results/upper_body_13115_experimental_replay.json',
    label: '腰臂控制實驗 · 歷史物理記錄',
    provenance: 'x86 · V5 13,115-update＋腰臂幅度限制 · 7.2 秒歷史軌跡',
    physical: true,
    warning: '同份舊 PPO 的控制修正實驗，沒有重訓或正式驗收。',
  },
};
const MODES = {
  stand: {
    label: '原生 URDF · 站立展示',
    provenance: '原生 URDF、關節限位與 V5 站姿 IK 參考',
    physical: false,
    warning: '此畫面是幾何展示，不是即時物理模擬。',
  },
  inplace: {
    label: '原地抬腳 · IK 參考預覽',
    provenance: 'config/v5_gaits_lift6.json · 原生關節 IK 前後步態平均',
    physical: false,
    warning: '程序化幾何參考，沒有在此執行 PPO 或物理驗收。',
  },
};
async function readJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('模型資料無法載入：' + url + ' (' + response.status + ')');
  return response.json();
}
function samplePlan(plan, phase) {
  const position = THREE.MathUtils.clamp(phase, 0, 1) * (plan.q.length - 1);
  const index = Math.min(plan.q.length - 2, Math.floor(position)), fraction = position - index;
  return plan.q[index].map((value, i) => THREE.MathUtils.lerp(value, plan.q[index + 1][i], fraction));
}
function sampleReplay(frames, time, jointCount) {
  let low = 0, high = frames.length - 1;
  while (low < high - 1) {
    const middle = (low + high) >> 1;
    if (frames[middle].time > time) high = middle;else low = middle;
  }
  const first = frames[low], second = frames[Math.min(low + 1, frames.length - 1)];
  const a = THREE.MathUtils.clamp((time - first.time) / Math.max(.00001, second.time - first.time), 0, 1);
  const q = Array.from({ length: jointCount }, (_, i) => THREE.MathUtils.lerp(first.q[i], second.q[i], a));
  // Isaac Lab 2.x recordings store WXYZ, while Three.js uses XYZW.
  const quat = new THREE.Quaternion(first.quaternion[1], first.quaternion[2], first.quaternion[3], first.quaternion[0]);
  quat.slerp(new THREE.Quaternion(second.quaternion[1], second.quaternion[2], second.quaternion[3], second.quaternion[0]), a);
  return { q, root: first.root.map((value, i) => THREE.MathUtils.lerp(value, second.root[i], a)), quat };
}

/** Resource-bounded native URDF product stage. Exhibit scenes are schematic. */
export async function createProductViewer(container, {
  onReady = () => {}, onError = () => {}, onMotionChange = () => {},
} = {}) {
  if (!(container instanceof Element)) throw new TypeError('A viewer container is required');
  let disposed = false, robot, renderer, controls, scene, camera, floor, contexts, measurements;
  let raf = 0, visible = true, playing = true, lastRender = 0, animationTime = 0;
  let mode = 'stand', sceneName = 'studio', requestVersion = 0, activeReplay = null;
  let contract, gait, info, resizeObserver, intersectionObserver;
  const replayCache = new Map();
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const viewCenter = new THREE.Vector3(0, 0, .83);
  const homePosition = new THREE.Vector3(2.15, -2.75, 1.65);

  function schedule() {
    if (!disposed && !raf && visible && !document.hidden) raf = requestAnimationFrame(animate);
  }
  function resize() {
    if (!renderer || disposed) return;
    const width = Math.max(1, container.clientWidth), height = Math.max(1, container.clientHeight);
    camera.aspect = width / height;camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);schedule();
  }
  function setPose(q, root, quaternion) {
    robot.setJointValues(Object.fromEntries(contract.joint_names.map((name, i) => [name, q[i]])));
    robot.position.fromArray(root);
    if (quaternion) robot.quaternion.copy(quaternion);else robot.quaternion.identity();
  }
  function updatePose() {
    if (!robot) return;
    if (activeReplay) {
      const frames = activeReplay.frames, duration = frames.at(-1).time - frames[0].time;
      const frame = sampleReplay(frames, frames[0].time + (animationTime % Math.max(duration, .02)), contract.joint_names.length);
      setPose(frame.q, frame.root, frame.quat);
    } else if (mode === 'inplace') {
      const phase = (animationTime / gait.walk_cycle_s) % 1;
      const forward = samplePlan(gait.plans.forward, phase), backward = samplePlan(gait.plans.backward, phase);
      setPose(forward.map((value, i) => (value + backward[i]) / 2), [0, 0, gait.root_height]);
    } else setPose(gait.plans.stand.q[0], [0, 0, gait.root_height]);
  }
  function animate(now) {
    raf = 0;
    if (disposed || !visible || document.hidden) return;
    schedule();
    // Production PPO shares GB10: this viewer is capped at 30 fps and DPR 1.4.
    if (now - lastRender < 1000 / 30) return;
    const delta = lastRender ? Math.min((now - lastRender) / 1000, .1) : 0;lastRender = now;
    if (playing && !reducedMotion) animationTime += delta;
    updatePose();controls.update();renderer.render(scene, camera);
  }
  function onVisibility() {
    if (document.hidden) { cancelAnimationFrame(raf);raf = 0;lastRender = 0; }
    else schedule();
  }
  function updateInfo() {
    if (!robot) return;
    robot.updateMatrixWorld(true);
    const size = new THREE.Box3().setFromObject(robot).getSize(new THREE.Vector3());
    info = {
      mode, ...(SOURCES[mode] || MODES[mode]),
      scene: sceneName,
      scene_scope: '示意展示場景；不是已驗證的工作範圍或地形能力。',
      bounds_m: { x: size.x, y: size.y, z: size.z },
      bounds_scope: '原生 URDF mesh 的當前姿態包圍盒；不是實機尺寸量測。',
      root_height_m: contract.root_height, mass_kg: contract.total_mass_kg,
      action_joints: contract.joint_names.length, urdf_url: URDF_URL, urdf_sha256: contract.urdf_sha256,
    };
  }
  const api = {
    async setMotion(value) {
      const selected = typeof value === 'string' ? value : value?.viewerMode || value?.mode || value?.id;
      const aliases = { 'spark-balance-v1': 'stand', v6: 'stand', 'v5-pair': 'sidestep', v5: 'sidestep' };
      const next = aliases[selected] || selected;
      if (!(next in MODES) && !(next in SOURCES)) throw new Error('Unknown viewer motion: ' + next);
      const currentRequest = ++requestVersion;
      try {
        let replay = null;
        if (SOURCES[next]) {
          if (!replayCache.has(next)) replayCache.set(next, readJSON(SOURCES[next].url));
          replay = await replayCache.get(next);
          if (!Array.isArray(replay.frames) || replay.frames.length < 2 ||
              replay.frames.some(f => f.q.length !== contract.joint_names.length || f.root.length !== 3 || f.quaternion.length !== 4)) {
            throw new Error('歷史軌跡與 Q1 關節契約不符');
          }
        }
        if (disposed || currentRequest !== requestVersion) return api;
        mode = next;activeReplay = replay;animationTime = 0;updatePose();updateInfo();
        onMotionChange({ ...info });schedule();return api;
      } catch (error) { if (!disposed) onError(error);throw error; }
    },
    setScene(value) {
      if (!['studio', 'workspace', 'gallery'].includes(value)) throw new Error('Unknown exhibit scene: ' + value);
      sceneName = value;contexts.children.forEach(group => { group.visible = group.name === value; });
      floor.material.color.set(value === 'studio' ? 0xf5f5f7 : value === 'workspace' ? 0xe9edf1 : 0xdedfdc);
      scene.background = value === 'studio' ? null : new THREE.Color(value === 'workspace' ? 0xf0f3f6 : 0xf0f0eb);
      updateInfo();schedule();return api;
    },
    setPlaying(value) { playing = Boolean(value);lastRender = 0;schedule();return api; },
    setTime(seconds) {
      animationTime = Math.max(0, Number(seconds) || 0);
      updatePose();controls.update();renderer.render(scene, camera);updateInfo();return api;
    },
    seek(value) {
      const duration = activeReplay ? activeReplay.frames.at(-1).time - activeReplay.frames[0].time : gait.walk_cycle_s;
      return api.setTime(THREE.MathUtils.clamp(Number(value) || 0, 0, .999999) * duration);
    },
    captureImage() { updatePose();controls.update();renderer.render(scene, camera);return renderer.domElement.toDataURL('image/png'); },
    setMeasurements(value) { measurements.visible = Boolean(value);schedule();return api; },
    resetView() { camera.position.copy(homePosition);controls.target.copy(viewCenter);controls.update();schedule();return api; },
    getInfo() { updateInfo();return { ...info }; },
    dispose() {
      if (disposed) return;disposed = true;requestVersion++;cancelAnimationFrame(raf);
      resizeObserver?.disconnect();intersectionObserver?.disconnect();document.removeEventListener('visibilitychange', onVisibility);
      controls?.dispose();
      const geometry = new Set(), materials = new Set();
      scene?.traverse(object => {
        if (object.geometry && !geometry.has(object.geometry)) { object.geometry.dispose();geometry.add(object.geometry); }
        (Array.isArray(object.material) ? object.material : [object.material]).filter(Boolean).forEach(material => {
          if (!materials.has(material)) { material.dispose();materials.add(material); }
        });
      });
      renderer?.dispose();renderer?.domElement.remove();
    },
  };
  try {
    [contract, gait] = await Promise.all([readJSON('/config/robot_contract.json'), readJSON('/config/v5_gaits_lift6.json')]);
    if (!contract.native_cad_ready || contract.proxy || contract.joint_names.length !== 23 ||
        JSON.stringify(contract.joint_names) !== JSON.stringify(gait.joint_names)) {
      throw new Error('Q1 原生幾何或關節資料尚未就緒');
    }
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true, powerPreference: 'low-power' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.4));
    renderer.outputColorSpace = THREE.SRGBColorSpace;renderer.toneMapping = THREE.ACESFilmicToneMapping;renderer.toneMappingExposure = 1.0;
    renderer.shadowMap.enabled = true;renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.domElement.style.cssText = 'display:block;width:100%;height:100%;touch-action:none;outline:none';
    renderer.domElement.tabIndex = 0;
    renderer.domElement.setAttribute('aria-label', 'Q1 原生 URDF 互動模型，拖曳旋轉，捲動縮放。場景為示意。');
    container.append(renderer.domElement);
    scene = new THREE.Scene();scene.up.set(0, 0, 1);
    camera = new THREE.PerspectiveCamera(31, 1, .025, 30);camera.up.set(0, 0, 1);camera.position.copy(homePosition);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(viewCenter);controls.enableDamping = true;controls.dampingFactor = .08;
    controls.minDistance = 1.7;controls.maxDistance = 5.8;controls.maxPolarAngle = Math.PI * .495;controls.minPolarAngle = .3;
    controls.enablePan = false;controls.autoRotate = false;
    scene.add(new THREE.HemisphereLight(0xf8fbff, 0x8693a1, 1.85));
    const key = new THREE.DirectionalLight(0xfffaf1, 3.2);key.position.set(2, -3, 5);key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);key.shadow.camera.left = -2;key.shadow.camera.right = 2;
    key.shadow.camera.top = 2;key.shadow.camera.bottom = -2;key.shadow.camera.near = .5;key.shadow.camera.far = 10;
    key.shadow.normalBias = .015;key.shadow.bias = -.00008;scene.add(key);
    const fill = new THREE.DirectionalLight(0xe0eaff, 1.05);fill.position.set(-2.5, 1.5, 2.5);scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffffff, 2.0);rim.position.set(1.5, 3, 3.3);scene.add(rim);
    floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.MeshStandardMaterial({ color: 0xf5f5f7, roughness: .88 }));
    floor.receiveShadow = true;floor.position.z = -.006;scene.add(floor);
    contexts = new THREE.Group();scene.add(contexts);
    for (const name of ['studio', 'workspace', 'gallery']) {
      const group = new THREE.Group();group.name = name;group.visible = name === 'studio';contexts.add(group);
      if (name === 'workspace') {
        const grid = new THREE.GridHelper(3, 12, 0xb6c1cb, 0xd8dfe6);grid.rotation.x = Math.PI / 2;grid.position.z = .001;
        grid.material.transparent = true;grid.material.opacity = .55;group.add(grid);
        const zone = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-.6, -.6, .004),new THREE.Vector3(.6, -.6, .004),
          new THREE.Vector3(.6, .6, .004),new THREE.Vector3(-.6, .6, .004),
        ]), new THREE.LineBasicMaterial({ color: 0x8aadb8, transparent: true, opacity: .65 }));
        group.add(zone);
      } else if (name === 'gallery') {
        const ring = new THREE.Mesh(new THREE.RingGeometry(.72, .735, 80), new THREE.MeshBasicMaterial({ color: 0xb9b7af, transparent: true, opacity: .7, side: THREE.DoubleSide }));
        ring.position.z = .004;group.add(ring);
      }
    }
    const manager = new THREE.LoadingManager(), failedAssets = [];
    const meshesLoaded = new Promise(resolve => { manager.onLoad = resolve; });
    manager.onError = url => failedAssets.push(url);
    robot = await new URDFLoader(manager).loadAsync(URDF_URL);await meshesLoaded;
    if (failedAssets.length) throw new Error('原生 mesh 載入失敗：' + failedAssets[0]);
    robot.traverse(object => {
      if (!object.isMesh) return;object.castShadow = true;object.receiveShadow = true;
      const prior = Array.isArray(object.material) ? object.material[0] : object.material;
      // URDFVisual containers often have no useful name; classify by their
      // nearest native link rather than accidentally painting every mesh alike.
      let link = object.parent;
      while (link && !link.isURDFLink) link = link.parent;
      const linkName = (link?.urdfName || link?.name || '').toLowerCase();
      const isFoot = /foot|sole/.test(linkName);
      const isHand = /wrist|gripper|finger/.test(linkName);
      const isCore = /center_waist|ankle_roll/.test(linkName);
      object.material = new THREE.MeshStandardMaterial({
        color: isFoot ? 0x3d4651 : isHand ? 0x62707e : isCore ? 0x8995a3 : 0xb7c0cc,
        roughness: isFoot ? .53 : isHand ? .4 : .32,
        metalness: isFoot ? .22 : .38,
      });
      if (prior) prior.dispose();
    });
    scene.add(robot);updatePose();robot.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(robot), x = bounds.max.x + .08, y = bounds.max.y + .08;
    measurements = new THREE.Group();
    measurements.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x, y, bounds.min.z),new THREE.Vector3(x, y, bounds.max.z),
      new THREE.Vector3(x-.035, y, bounds.min.z),new THREE.Vector3(x+.035, y, bounds.min.z),
      new THREE.Vector3(x-.035, y, bounds.max.z),new THREE.Vector3(x+.035, y, bounds.max.z),
    ]),new THREE.LineBasicMaterial({ color: 0x567b98, transparent: true, opacity: .8 })));
    measurements.visible = false;scene.add(measurements);
    resizeObserver = new ResizeObserver(resize);resizeObserver.observe(container);
    intersectionObserver = new IntersectionObserver(entries => {
      visible = entries[0]?.isIntersecting ?? true;
      if (!visible) { cancelAnimationFrame(raf);raf = 0;lastRender = 0; }else schedule();
    }, { rootMargin: '80px' });
    intersectionObserver.observe(container);document.addEventListener('visibilitychange', onVisibility);
    resize();updateInfo();onReady({ ...info });schedule();return api;
  } catch (error) { api.dispose();onError(error);throw error; }
}
