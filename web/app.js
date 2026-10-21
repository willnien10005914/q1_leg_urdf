import * as THREE from './vendor/three/three.module.js';
import {OrbitControls} from './vendor/three/examples/jsm/controls/OrbitControls.js';
import URDFLoader from './vendor/urdf-loader/URDFLoader.js';

const $=id=>document.getElementById(id);

// Resize both the video panel and robot viewport without reloading either.
const layout=document.querySelector('main'),splitter=$('pane-resizer');
let panelWidth=420,previousPanelWidth=420,videoExpanded=false;
try{panelWidth=Number(localStorage.getItem('q1-leg-panel-width'))||420;}catch{}
function setPanelWidth(width,persist=true){
 if(innerWidth<=850)return;
 panelWidth=Math.max(320,Math.min(width,layout.clientWidth-332));
 layout.style.setProperty('--sidebar-width',`${panelWidth}px`);
 splitter.setAttribute('aria-valuemin','320');splitter.setAttribute('aria-valuemax',String(layout.clientWidth-332));
 splitter.setAttribute('aria-valuenow',String(Math.round(panelWidth)));splitter.setAttribute('aria-valuetext',`影片面板寬 ${Math.round(panelWidth)} 像素`);
 if(persist)try{localStorage.setItem('q1-leg-panel-width',String(panelWidth));}catch{}
}
function expandedState(value){videoExpanded=value;$('expand-video').textContent=value?'還原寬度':'放大影片';$('expand-video').setAttribute('aria-pressed',String(value));}
setPanelWidth(panelWidth,false);
let resizingPointer=null;
splitter.addEventListener('pointerdown',event=>{if(event.button!==0)return;event.preventDefault();resizingPointer=event.pointerId;splitter.setPointerCapture(event.pointerId);document.body.classList.add('resizing');expandedState(false);});
splitter.addEventListener('pointermove',event=>{if(resizingPointer===event.pointerId)setPanelWidth(layout.getBoundingClientRect().right-event.clientX);});
function endPanelResize(){resizingPointer=null;document.body.classList.remove('resizing');}
splitter.addEventListener('pointerup',endPanelResize);splitter.addEventListener('pointercancel',endPanelResize);splitter.addEventListener('lostpointercapture',endPanelResize);
splitter.addEventListener('dblclick',()=>{setPanelWidth(420);expandedState(false);});
splitter.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home'].includes(event.key))return;event.preventDefault();expandedState(false);setPanelWidth(event.key==='Home'?420:panelWidth+(event.key==='ArrowLeft'?1:-1)*(event.shiftKey?100:30));});
$('expand-video').onclick=()=>{if(!videoExpanded){previousPanelWidth=panelWidth;setPanelWidth(layout.clientWidth*.68);expandedState(true);}else{setPanelWidth(previousPanelWidth);expandedState(false);}};
addEventListener('resize',()=>setPanelWidth(panelWidth,false));
document.querySelectorAll('[data-video-source]').forEach(button=>button.addEventListener('click',()=>{
 const video=$('reference-video'),time=video.currentTime,wasPlaying=!video.paused;
 video.src=button.dataset.videoSource==='overlay'?'/docs/reference/sidestep_pose_overlay.mp4':'/docs/reference/sidestep_0_7.mp4';
 video.addEventListener('loadedmetadata',()=>{video.currentTime=Math.min(time,video.duration);if(wasPlaying)video.play().catch(()=>{});},{once:true});
 document.querySelectorAll('[data-video-source]').forEach(other=>{other.classList.toggle('active',other===button);other.setAttribute('aria-pressed',String(other===button));});
}));

const canvas=$('canvas'), scene=new THREE.Scene();
scene.up.set(0,0,1);
const camera=new THREE.PerspectiveCamera(42,1,.01,100);
camera.up.set(0,0,1);camera.position.set(2.8,2.5,1.9);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));canvas.appendChild(renderer.domElement);
renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','機器人即時畫面；啟用鍵盤後可使用 WASD');
renderer.outputColorSpace=THREE.SRGBColorSpace;
const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,0,.85);controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xd7e6ff,0x4e3a25,3));
const light=new THREE.DirectionalLight(0xffe3bd,3);light.position.set(2,1,4);scene.add(light);
const grid=new THREE.GridHelper(12,60,0x66768c,0x344155);grid.rotation.x=Math.PI/2;scene.add(grid);
const floor=new THREE.Mesh(new THREE.PlaneGeometry(12,12),new THREE.MeshStandardMaterial({color:0x1b2535,transparent:true,opacity:.3}));floor.position.z=-.003;scene.add(floor);
new ResizeObserver(()=>{camera.aspect=canvas.clientWidth/canvas.clientHeight;camera.updateProjectionMatrix();renderer.setSize(canvas.clientWidth,canvas.clientHeight);}).observe(canvas);
let robot,contract,motion,rollout,status,liveState,liveError,livePending=false,skill='sidestep',source='reference',t=0,playing=true,previous=performance.now(),duration=6.95,loadVersion=0;
let keyboardArmed=false,sequencePending=false,commandQueue=Promise.resolve();
const heldKeys=new Set(),movementKeys=new Set(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE']);
const isInteractive=()=>['q1_leg_v4','q1_leg_v5'].includes(liveState?.controller_version);
const controllerLabel=()=>liveState?.controller_version==='q1_leg_v5'?'v5':'v4';
const liveControlsReady=()=>source==='live'&&isInteractive()&&liveState.ready===true&&!liveError&&!liveState.error&&!['paused','failed','starting'].includes(liveState.status);
const sequenceActive=()=>sequencePending||Boolean(liveState?.sequence_active);
async function json(url){const r=await fetch(url);if(!r.ok)throw Error(`${url}: ${r.status}`);return r.json();}
function setTime(value){t=Math.min(Math.max(value,0),duration);$('timeline').value=t;$('time').textContent=`${t.toFixed(2)} s`;}
function interpolate(time,times,poses,roots,quats){
 let i=0;while(i<times.length-2&&times[i+1]<time)i++;
 const j=Math.min(i+1,times.length-1),a=Math.max(0,Math.min(1,(time-times[i])/Math.max(times[j]-times[i],1e-6)));
 return {q:poses[i].map((v,k)=>v+(poses[j][k]-v)*a),root:roots[i].map((v,k)=>v+(roots[j][k]-v)*a),quaternion:quats?new THREE.Quaternion(quats[i][1],quats[i][2],quats[i][3],quats[i][0]).slerp(new THREE.Quaternion(quats[j][1],quats[j][2],quats[j][3],quats[j][0]),a):new THREE.Quaternion()};
}
function refFrame(){
 if(skill==='sidestep')return interpolate(t,motion.time,motion.q,motion.root);
 const q=[...contract.default];
 if(skill==='walk')for(const [side,sign,offset] of [['left',1,0],['right',-1,Math.PI]]){
  const swing=Math.sin(2*Math.PI*t/.9+offset),lift=Math.max(0,swing);
  for(const [part,v] of [['hip_Pitch',.15+.18*swing],['knee_Pitch',.3+.25*lift],['ankle_Pitch',-.15+.18*swing-.25*lift],['shoulder_Pitch',.12*swing]])q[contract.joint_names.indexOf(`${side}_${part}_joint`)]=sign*v;
 }
 return {q,root:[skill==='walk'?.2*t:0,0,contract.root_height],quaternion:new THREE.Quaternion()};
}
function showStatus(){
 const e=status?.skills[skill];
 $('mode-label').textContent=source==='live'?'即時 PPO · Isaac / PhysX':source==='reference'?(skill==='sidestep'?'MediaPipe → 原生關節 IK':'程序化參考姿態'):`PPO → ${e?.backend??'模擬'} 物理評估記錄`;
 $('status').textContent=source==='reference'?(skill==='sidestep'?'已擷取原片 0–7 秒；關節映射預覽，尚非 PPO 成果。':'程序化參考姿態；用於訓練目標與動作檢查。'):(rollout?`PPO 評估記錄 · ${rollout.proxy?'簡化模型':'原生 CAD'} · ${rollout.passed?'通過門檻':'尚未通過門檻'} · ${rollout.frames.at(-1).time.toFixed(2)} 秒`:'此動作尚無 PPO 評估記錄，請先執行訓練。');
 if(source==='live'){
  const labels={running:'執行中',paused:'已暫停',finished:'回合完成，可重播',failed:'回合終止',starting:'初始化中'};
  $('status').textContent=liveError??liveState?.error??`原生 CAD 即時模擬 · ${labels[liveState?.status]??'連線中'}。${isInteractive()?`${controllerLabel()} 方向控制與閃身步連續執行；「回到起點」會重置模擬。`:'切換動作會重置模擬。'}`;
  $('play').textContent=liveState?.status==='running'?'暫停':liveState?.status==='finished'?'重播':'播放';
 }
 const metrics=source==='reference'&&skill==='sidestep'?[[`${(motion.quality.coverage*100).toFixed(0)}%`,'MediaPipe 有效影格'],[String(motion.quality.accepted_frames),'擷取影格']]:[[e?.survival_rate!==undefined?`${(e.survival_rate*100).toFixed(0)}%`:'—','PPO 回合存活率'],[e?.pose_rmse!==undefined?`${e.pose_rmse.toFixed(3)}`:'—','關節 RMSE / rad']];
 $('metrics').replaceChildren(...metrics.map(([value,label])=>{const div=document.createElement('div');div.className='metric';const strong=document.createElement('strong'),span=document.createElement('span');strong.textContent=value;span.textContent=label;div.append(strong,span);return div;}));
 const training=status?.training;
 $('training').textContent=training?`訓練：${training.skill} · ${training.state}${training.max_iterations?` · ${training.live_iteration_estimate??training.completed_iterations??0} / ${training.max_iterations} iterations · ${training.num_envs} envs`:''}`:'PPO 尚未開始';
 updateKeyboardControls();
}
async function select(){
 const version=++loadVersion;
 playing=false;$('play').textContent='播放';rollout=null;
 $('timeline').disabled=source==='live';$('speed').disabled=source==='live';
 if(source==='live'){liveError=null;setTime(0);try{liveState=await json('/api/live');if(version!==loadVersion)return;if(!isInteractive())await liveCommand('restart');}catch(e){liveError=e.message;}showStatus();return;}
 if(source==='policy')try{const data=await json(status?.skills[skill]?.rollout??`/outputs/${skill}_policy_rollout.json`);if(version!==loadVersion)return;rollout=data;}catch(e){if(version!==loadVersion)return;}
 if(version!==loadVersion)return;
 duration=source==='policy'?(rollout?.frames.at(-1).time??0):(skill==='sidestep'?motion.time.at(-1):10);
 $('timeline').max=duration;$('duration').textContent=`${duration.toFixed(2)} s`;setTime(0);showStatus();
 if(source==='reference'||rollout){playing=true;$('play').textContent='暫停';}
}
async function liveCommand(action,parameters={}){
 const response=await fetch('/api/live/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(isInteractive()?{action,...parameters}:{skill,action,...parameters})});
 const value=await response.json();if(!response.ok)throw Error(value.error??'即時模擬無法執行');liveError=null;
}
function queueControl(action,parameters={}){
 commandQueue=commandQueue.catch(()=>{}).then(()=>liveCommand(action,parameters));
 commandQueue.catch(e=>{liveError=e.message;clearHeldKeys();keyboardArmed=false;showStatus();});
 return commandQueue;
}
function clearHeldKeys(){heldKeys.clear();document.querySelectorAll('[data-control-key]').forEach(b=>b.classList.remove('held'));}
function releaseKeyboard(sendStop=true,disarm=true){
 clearHeldKeys();if(disarm)keyboardArmed=false;
 if(sendStop&&source==='live'&&isInteractive())queueControl('stop');
 updateKeyboardControls();
}
function directionCommand(){return {vx:.22*(Number(heldKeys.has('KeyW'))-Number(heldKeys.has('KeyS'))),vy:.18*(Number(heldKeys.has('KeyA'))-Number(heldKeys.has('KeyD'))),wz:.6*(Number(heldKeys.has('KeyQ'))-Number(heldKeys.has('KeyE')))};}
function sendDirection(){if(!liveControlsReady()||sequenceActive())return;queueControl('velocity',directionCommand());}
function updateKeyboardControls(){
 const ready=liveControlsReady(),sequence=sequenceActive();
 if(!ready){keyboardArmed=false;clearHeldKeys();}
 if(sequence)clearHeldKeys();
 $('keyboard-toggle').disabled=!ready||sequence;
 $('keyboard-toggle').setAttribute('aria-pressed',String(keyboardArmed));
 $('keyboard-toggle').textContent=keyboardArmed?'鍵盤已啟用':'啟用鍵盤';
 $('keyboard-stop').disabled=source!=='live'||!isInteractive();
 $('sidestep-sequence').disabled=!ready||sequence;
 $('reset').disabled=source==='live'&&isInteractive()&&liveState.status!=='failed';
 $('sidestep-sequence').textContent=sequence?'閃身步執行中 · 完成後回站立':'閃身步 · 左右各一次 → 站立';
 document.querySelectorAll('[data-control-key]').forEach(b=>{b.disabled=!ready||sequence;b.classList.toggle('held',heldKeys.has(b.dataset.controlKey));});
 document.querySelectorAll('[data-skill]').forEach(b=>{b.disabled=source==='live'&&isInteractive();});
 $('keyboard-status').textContent=source!=='live'?'切換至「即時 PPO / Isaac PhysX」才能控制移動。':!isInteractive()?'目前為舊版播放服務；WASD 與一鍵閃身需通過完整驗證的 v4 或 v5 控制器。':!ready?`${controllerLabel()} 控制器尚未就緒，方向控制已鎖定；請等待載入或通過訓練驗證。`:sequence?'正在左右閃身，方向輸入暫時鎖定；Space 可停止。':keyboardArmed?'鍵盤已啟用。放開按鍵、切換視窗或離開即時模式會停止移動。':`${controllerLabel()} 控制器已就緒。點「啟用鍵盤」，或按住方向按鈕移動。`;
 const modes={stand:'站立',walk:'方向行走',sidestep_left:'向左閃身',sidestep_right:'向右閃身',recover:'回復站姿'};
 $('controller-mode').textContent=source==='live'&&isInteractive()?`即時狀態：${modes[liveState.mode]??liveState.mode??'等待控制器'}`:'控制器尚未就緒';
 const c=source==='live'&&isInteractive()?liveState.command??{}:{};
 const fmt=value=>Number.isFinite(Number(value))?Number(value).toFixed(2):'0.00';
 $('velocity-command').textContent=`前後 ${fmt(c.vx)} · 左右 ${fmt(c.vy)} m/s · 旋轉 ${fmt(c.wz)} rad/s`;
}
function interactiveTarget(target){return target instanceof Element&&Boolean(target.closest('input,textarea,select,button,summary,video,[contenteditable]:not([contenteditable="false"]),[role="separator"]'));}
$('keyboard-toggle').onclick=()=>{if(!liveControlsReady()||sequenceActive())return;if(keyboardArmed)releaseKeyboard();else{keyboardArmed=true;renderer.domElement.focus({preventScroll:true});updateKeyboardControls();}};
$('keyboard-stop').onclick=()=>releaseKeyboard(true,false);
$('sidestep-sequence').onclick=async()=>{
 if(!liveControlsReady()||sequenceActive())return;
 renderer.domElement.focus({preventScroll:true});
 clearHeldKeys();sequencePending=true;updateKeyboardControls();
 try{await queueControl('sidestep_sequence');liveState=await json('/api/live');}catch(e){liveError=e.message;}finally{sequencePending=false;showStatus();}
};
addEventListener('keydown',event=>{
 if(source!=='live'||!isInteractive()||interactiveTarget(event.target)||event.altKey||event.ctrlKey||event.metaKey)return;
 if(event.code==='Space'){event.preventDefault();if(!event.repeat)releaseKeyboard(true,false);return;}
 if(!keyboardArmed||!movementKeys.has(event.code)||!liveControlsReady()||sequenceActive())return;
 event.preventDefault();if(event.repeat||heldKeys.has(event.code))return;
 heldKeys.add(event.code);updateKeyboardControls();sendDirection();
});
addEventListener('keyup',event=>{if(!heldKeys.delete(event.code))return;event.preventDefault();updateKeyboardControls();sendDirection();});
addEventListener('blur',()=>releaseKeyboard());
document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseKeyboard();});
document.addEventListener('focusin',event=>{if(heldKeys.size&&interactiveTarget(event.target))releaseKeyboard();});
addEventListener('pagehide',()=>{if(source==='live'&&isInteractive())fetch('/api/live/command',{method:'POST',keepalive:true,headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'stop'})}).catch(()=>{});});
setInterval(()=>{if(heldKeys.size)sendDirection();},120);
document.querySelectorAll('[data-control-key]').forEach(button=>{
 let activePointer=null;
 button.addEventListener('pointerdown',event=>{if(event.button!==0||!liveControlsReady()||sequenceActive())return;event.preventDefault();keyboardArmed=true;renderer.domElement.focus({preventScroll:true});activePointer=event.pointerId;button.setPointerCapture(event.pointerId);heldKeys.add(button.dataset.controlKey);updateKeyboardControls();sendDirection();});
 const release=event=>{if(activePointer!==event.pointerId)return;activePointer=null;heldKeys.delete(button.dataset.controlKey);updateKeyboardControls();sendDirection();};
 button.addEventListener('pointerup',release);button.addEventListener('pointercancel',release);button.addEventListener('lostpointercapture',release);
});
document.querySelectorAll('[data-skill]').forEach(b=>b.addEventListener('click',()=>{skill=b.dataset.skill;document.querySelectorAll('[data-skill]').forEach(x=>x.classList.toggle('active',x===b));select();}));
$('source').onchange=()=>{releaseKeyboard();source=$('source').value;updateKeyboardControls();select();};
$('play').onclick=async()=>{if(source==='live'){releaseKeyboard();try{await queueControl(liveState?.status==='running'?'pause':liveState?.status==='finished'?'restart':'play');}catch(e){liveError=e.message;showStatus();}return;}if(source==='policy'&&!rollout)return;if(t>=duration)setTime(0);playing=!playing;$('play').textContent=playing?'暫停':'播放';};
$('reset').onclick=async()=>{if(source==='live'){releaseKeyboard();try{await queueControl('restart');}catch(e){liveError=e.message;showStatus();}}else setTime(0);};
$('timeline').oninput=()=>{playing=false;$('play').textContent='播放';setTime(Number($('timeline').value));};
function animate(now){requestAnimationFrame(animate);const dt=Math.min((now-previous)/1000,.1);previous=now;
 if(robot&&motion&&contract){
  if(playing&&source!=='live'){setTime(t+dt*Number($('speed').value));if(t>=duration){playing=false;$('play').textContent='重播';}}
  let frame=null;
  if(source==='reference')frame=refFrame();
  else if(source==='live'&&(isInteractive()||liveState?.skill===skill)&&liveState.frame&&!liveError){const f=liveState.frame;frame={q:f.q,root:f.root,quaternion:new THREE.Quaternion(f.quaternion[1],f.quaternion[2],f.quaternion[3],f.quaternion[0])};}
  else if(rollout){const f=rollout.frames;frame=interpolate(t,f.map(x=>x.time),f.map(x=>x.q),f.map(x=>x.root),f.map(x=>x.quaternion));}
  if(frame){robot.visible=true;robot.setJointValues(Object.fromEntries(contract.joint_names.map((n,i)=>[n,frame.q[i]])));robot.position.fromArray(frame.root);robot.quaternion.copy(frame.quaternion);}else robot.visible=false;
 }
 controls.update();renderer.render(scene,camera);
}
requestAnimationFrame(animate);
try{
 [contract,motion,status]=await Promise.all([json('/config/robot_contract.json'),json('/docs/reference/sidestep_motion.json'),json('/api/status')]);
 $('geometry').textContent=contract.native_cad_ready?'原生 CAD 已齊全':'簡化雙足 CAD';
 const manager=new THREE.LoadingManager();manager.onError=url=>{$('geometry').textContent='模型素材載入失敗';console.error(url);};
 robot=await new URDFLoader(manager).loadAsync(contract.native_cad_ready?'/urdf/q1_inertia/quanta_robot_legged.urdf':'/urdf/leg_humanoid_proxy.urdf');scene.add(robot);
 robot.traverse(o=>{if(o.isMesh&&o.material){o.material=o.material.clone();if(!o.material.color.equals(new THREE.Color(.95,.49,.12)))o.material.color.set(0xadb8c8);o.material.roughness=.65;o.material.metalness=.15;}});
 await select();
 setInterval(async()=>{try{const old=JSON.stringify(status?.skills[skill]);status=await json('/api/status');if(source==='policy'&&old!==JSON.stringify(status.skills[skill]))await select();showStatus();}catch(e){console.error(e);}},5000);
 setInterval(async()=>{if(source!=='live'||livePending)return;livePending=true;try{liveState=await json('/api/live');liveError=null;if((isInteractive()||liveState.skill===skill)&&liveState.frame){duration=Number.isFinite(liveState.duration)?liveState.duration:Math.max(10,liveState.frame.time);$('timeline').max=duration;$('duration').textContent=`${duration.toFixed(2)} s`;setTime(liveState.frame.time);}showStatus();}catch(e){liveError='即時 PPO 模擬連線失敗，請使用參考動作或評估記錄。';releaseKeyboard();showStatus();}finally{livePending=false;}},75);
}catch(e){$('status').textContent=`初始化失敗：${e.message}`;console.error(e);}
