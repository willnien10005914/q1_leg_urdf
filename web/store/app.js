import {createProductViewer} from './product-viewer.js';
const $=id=>document.getElementById(id),num=v=>Number.isFinite(v)?v.toLocaleString('en-US'):'—',pct=v=>Number.isFinite(v)?(v*100).toFixed(1)+'%':'—';
const states={training:'PPO 訓練中',evaluating:'物理評估中',prepared:'準備訓練',waiting_for_other_compute:'等待 GPU',queued:'訓練佇列中',budget_exhausted:'預算已完成',runtime_error:'訓練已停止',stopped:'已停止',not_started:'尚未開始',updating:'狀態更新中'};
const statusNames={measured:'模擬量測',command:'命令設定',source:'來源資料',unmeasured:'待量測',unverified:'待驗證'};
const specOrder=['speed','workspace','battery','energy','control','duration','footprint','clearance'];
const specLabels={speed:'速度',workspace:'安全工作範圍',battery:'電量與續航',energy:'動作耗電',control:'控制關節',duration:'記錄／驗測時長',footprint:'已記錄的軌跡',clearance:'腳底抬升'};
const sceneCopy={studio:['每個細節，都看得見。','以原生 URDF 幾何呈現關節與連桿，從你喜歡的角度觀察 Q1。'],workspace:['讓空間，成為新的視角。','切換地面與空間參考，理解動作與模型的相對比例；示意區域不是已驗證的安全工作範圍。'],gallery:['把動作，放在聚光燈下。','用展示舞台比較站姿與歷史動作，保留每一份軌跡的來源與版本。']};
let data,filter='all',loading=false,currentItem=null,lastCatalogKey='',compareIds=[],initial=true;
function node(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function groupButtons(selector,button){document.querySelectorAll(selector).forEach(b=>{const key=selector.includes('scene')?'scene':'motion',active=b.dataset[key]===button.dataset[key];b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});}
function sourceInfo(info){
 $('viewer-status').textContent=info.label;
 if(info.bounds_m?.z)$('model-height').textContent=info.bounds_m.z.toFixed(2);
 $('viewer-source').textContent=info.provenance+'。'+info.warning+' '+info.scene_scope;
}
const viewerReady=createProductViewer($('product-stage'),{
 onReady:sourceInfo,onMotionChange:sourceInfo,onError:error=>{
  $('viewer-status').textContent='模型載入失敗，可使用影片或完整檢視器。';
  $('viewer-source').textContent=error.message;
  const pending=$('product-stage').querySelector('.viewer-loading');if(pending)pending.textContent='3D 模型暫時無法載入。請重新整理或使用完整 3D 檢視器。';
 }
}).then(viewer=>{window.Q1ProductViewer=viewer;viewer.setPlaying(false);$('product-stage').querySelector('.viewer-loading')?.remove();return viewer;}).catch(()=>null);
function renderProgress(t){
 $('state').textContent=states[t.state]||t.state||'未知狀態';
 $('connection').className=t.stale||t.error?'error':'';
 $('connection').textContent=t.error?'工作已停止：'+t.error:t.stale?'狀態超過兩分鐘未更新，請檢查訓練服務。':t.state==='not_started'?'尚未收到訓練資料。':'即時讀取 Spark 的保存狀態。此版本尚未完成正式驗收。';
 const saved=t.saved_iterations??0,budget=t.max_new_iterations??15000;
 $('saved').textContent=num(saved);$('budget').textContent=num(budget);$('percent').textContent=pct(saved/budget);$('progress').max=budget;$('progress').value=saved;
 $('envs').textContent=num(t.num_envs??6144);$('live').textContent=t.state==='training'&&t.live_iteration>saved?num(t.live_iteration):'—';
 $('rate').textContent=t.state==='training'?num(t.latest_steps_per_second):'評估／等待中';
 $('eta').textContent=['budget_exhausted','stopped','runtime_error'].includes(t.state)?'—':Number.isFinite(t.estimated_remaining_s)?(t.estimated_remaining_s/3600).toFixed(1)+' 小時':'估算中';
 const evals=t.evaluations||[],last=evals.at(-1);
 $('eval-version').textContent=last?'第 '+num(last.completed_iterations)+' 次 · '+(last.episodes??'—')+' 回合':'尚未評估';
 $('survival').textContent=pct(last?.four_step_survival);$('completion').textContent=pct(last?.completed_four_step_rate);
 $('updated').textContent=t.updated_at?'更新於 '+new Date(t.updated_at*1000).toLocaleString('zh-TW'):'尚無更新';
 $('commit').textContent=t.source_git_commit?'來源 '+t.source_git_commit.slice(0,7):'';renderChart(evals);
}
function svg(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
function renderChart(evals){
 const el=$('chart');el.replaceChildren();
 for(const value of[0,.5,1]){const y=118-value*98;el.append(svg('line',{x1:36,x2:688,y1:y,y2:y,stroke:'#ededf0','stroke-width':1}));el.append(svg('text',{x:0,y:y+3,fill:'#86868b','font-size':9},value*100+'%'));}
 const points=evals.filter(v=>Number.isFinite(v.completed_iterations));
 if(!points.length){el.append(svg('text',{x:350,y:72,'text-anchor':'middle',fill:'#86868b','font-size':11},'完成首次評估後顯示曲線'));return;}
 const maximum=Math.max(...points.map(v=>v.completed_iterations),500),x=value=>36+value/maximum*645;
 for(const[key,color]of[['four_step_survival','#0071e3'],['completed_four_step_rate','#a9a3c8']]){
  const values=points.filter(v=>Number.isFinite(v[key]));
  el.append(svg('polyline',{points:values.map(v=>x(v.completed_iterations)+','+(118-v[key]*98)).join(' '),fill:'none',stroke:color,'stroke-width':2}));
  for(const p of values){const c=svg('circle',{cx:x(p.completed_iterations),cy:118-p[key]*98,r:3,fill:color});c.append(svg('title',{},p.completed_iterations+' updates: '+pct(p[key])));el.append(c);}
 }
 el.append(svg('text',{x:36,y:140,fill:'#86868b','font-size':9},'0'));el.append(svg('text',{x:680,y:140,'text-anchor':'end',fill:'#86868b','font-size':9},num(maximum)+' updates'));
}
function renderCards(force=false){
 const query=$('search').value.trim().toLowerCase(),key=JSON.stringify([data.catalog,filter,query,data.training.state]);
 if(!force&&key===lastCatalogKey)return;lastCatalogKey=key;
 const items=data.catalog.filter(v=>(filter==='all'||v.family===filter)&&[v.title,v.version,v.product_name,v.family_name].join(' ').toLowerCase().includes(query));
 $('cards').replaceChildren();$('empty').hidden=items.length>0;
 for(const item of items){
  const card=node('article',undefined,'product-card'),media=node('button',undefined,'product-media');media.type='button';media.style.border='0';media.style.padding='0';media.setAttribute('aria-label','探索 '+item.product_name);
  const image=item.product_poster||item.poster;
  if(image){const img=node('img');img.src=image;img.alt=item.product_name+' 原生 Q1 模型展示';img.loading='lazy';media.append(img);}
  else{const abstract=node('div',undefined,'abstract-product');abstract.append(node('span','Q1','brand'));media.append(abstract);}
  media.append(node('span',item.version,'product-version'));media.onclick=()=>openDetail(item);
  const copy=node('div',undefined,'product-card-copy');
  copy.append(node('p',item.category==='spark'?'新實驗 · '+(states[data.training.state]||item.label):item.label,'product-new'),node('h3',item.product_name||item.title),node('p',item.tagline,'tagline'),node('p',item.description,'description'));
  const actions=node('div',undefined,'product-card-actions'),learn=node('button','進一步了解','learn-button'),film=node('button',item.video?'觀看影片 ›':'查看訓練 ›','video-button');
  learn.type=film.type='button';learn.onclick=()=>openDetail(item);film.onclick=()=>{if(item.video)openDetail(item,'video');else $('training').scrollIntoView({behavior:'smooth'});};
  actions.append(learn,film);card.append(media,copy,actions,node('p',item.origin,'product-footer'));$('cards').append(card);
 }
}
function specValue(spec){return spec?String(spec.value)+(spec.unit?' '+spec.unit:''):'待量測';}
function statusPill(spec){return node('span',statusNames[spec?.status]||'待量測','spec-status '+(spec?.status||'unmeasured'));}
function renderCompare(){
 const catalog=data.catalog;
 if(!compareIds.length)compareIds=[catalog.find(v=>v.id==='spark-balance-v1')?.id,catalog.find(v=>v.family==='stability')?.id,catalog.find(v=>v.family==='inplace')?.id].filter(Boolean);
 $('compare-selectors').replaceChildren();
 compareIds.forEach((id,index)=>{
  const select=node('select');select.setAttribute('aria-label','比較動作 '+(index+1));
  catalog.forEach(item=>{const option=node('option',item.product_name+' · '+item.version);option.value=item.id;select.append(option);});select.value=id;
  select.onchange=()=>{compareIds[index]=select.value;renderCompare();};$('compare-selectors').append(select);
 });
 const items=compareIds.map(id=>catalog.find(v=>v.id===id)).filter(Boolean),head=node('tr');
 head.append(node('th',''));items.forEach(item=>{const th=node('th',item.product_name);th.scope='col';head.append(th);});$('spec-head').replaceChildren(head);$('spec-body').replaceChildren();
 for(const key of specOrder){
  const row=node('tr'),label=node('th',specLabels[key]);label.scope='row';row.append(label);
  items.forEach(item=>{const spec=item.specs?.find(v=>v.id===key),cell=node('td');cell.append(node('strong',specValue(spec)),statusPill(spec));if(spec?.note)cell.append(node('small',spec.note));row.append(cell);});
  $('spec-body').append(row);
 }
}
function selectTab(name){
 document.querySelectorAll('[data-tab]').forEach(button=>{const active=button.dataset.tab===name;button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;});
 ['overview','specs','video'].forEach(key=>{$('detail-'+key).hidden=key!==name;});
 if(name!=='video')$('media').querySelector('video')?.pause();
}
async function showModel(item){
 closeDetail();
 const viewer=await viewerReady;if(!viewer)return;
 const mode=item.category==='spark'?'stand':(item.motion||'stand');
 await viewer.setMotion(mode);viewer.setPlaying(false);$('motion-playing').checked=false;
 document.querySelectorAll('[data-motion]').forEach(b=>{const active=b.dataset.motion===mode;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
 if(item.category==='spark')$('viewer-source').textContent='本輪 Spark 尚未錄製結果，現在顯示原生 URDF 站姿。不是本輪策略的動態成果。';
 $('hero-title').scrollIntoView({behavior:'smooth',block:'start'});
}
function openDetail(item,tab='overview'){
 currentItem=item;$('detail-origin').textContent=item.family_name+' · '+item.origin;$('detail-title').textContent=item.product_name;$('detail-description').textContent=item.tagline+' '+item.description;
 $('detail-overview').replaceChildren();const highlights=node('div',undefined,'detail-highlight-grid');
 (item.highlights||[]).forEach(value=>highlights.append(node('div',value,'detail-highlight')));$('detail-overview').append(highlights);
 const scenes=node('div',undefined,'detail-scenes');(item.scenes||[]).forEach(s=>scenes.append(node('span',s.name)));$('detail-overview').append(scenes,node('p',item.spec_scope,'detail-source'));
 const modelButton=node('button','以原生 URDF 探索','detail-model-button');modelButton.type='button';modelButton.onclick=()=>showModel(item);$('detail-overview').append(modelButton);
 $('detail-specs').replaceChildren();
 for(const spec of item.specs||[]){
  const row=node('div',undefined,'detail-spec-row'),value=node('div',undefined,'detail-spec-value');value.append(node('span',specValue(spec)),node('br'),statusPill(spec),node('p',spec.note||''));
  if(spec.source){const source=node('a','資料來源 ↗','source');source.href=spec.source;source.target='_blank';source.rel='noopener';value.append(source);}
  row.append(node('strong',spec.label),value);$('detail-specs').append(row);
 }
 $('media').replaceChildren();
 if(item.video){const video=node('video');video.controls=true;video.playsInline=true;video.preload='metadata';video.src=item.video;if(item.poster)video.poster=item.poster;$('media').append(video);}
 else{const box=node('div',undefined,'no-video');box.append(node('p','本輪 Spark 影片尚未錄製。'),node('p','不以歷史 x86 影片代替本輪結果。'));const a=node('a','查看即時訓練 ↑');a.href='#training';a.onclick=closeDetail;box.append(a);$('media').append(box);}
 $('detail-meta').textContent=item.version+' · '+item.backend+' · '+num(item.iterations)+' updates · 尚未完成正式驗收';
 selectTab(tab);$('detail').showModal();
}
function closeDetail(){$('media').querySelector('video')?.pause();$('detail').close();currentItem=null;}
$('close').onclick=closeDetail;$('detail').addEventListener('close',()=>{$('media').querySelector('video')?.pause();});
$('detail').addEventListener('click',event=>{if(event.target===$('detail')){const r=$('detail').getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)closeDetail();}});
document.querySelectorAll('[data-tab]').forEach(button=>{button.onclick=()=>selectTab(button.dataset.tab);button.onkeydown=event=>{const keys=['overview','specs','video'],index=keys.indexOf(button.dataset.tab);let target;if(event.key==='ArrowRight')target=(index+1)%3;else if(event.key==='ArrowLeft')target=(index+2)%3;else if(event.key==='Home')target=0;else if(event.key==='End')target=2;if(target!==undefined){event.preventDefault();selectTab(keys[target]);$('tab-'+keys[target]).focus();}};});
$('search').oninput=()=>renderCards();
function setFamily(family){filter=family;document.querySelectorAll('[data-filter]').forEach(b=>{const active=b.dataset.filter===family;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});if(data)renderCards();}
document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>setFamily(button.dataset.filter));
document.querySelectorAll('[data-family-link]').forEach(a=>a.addEventListener('click',()=>setFamily(a.dataset.familyLink)));
document.querySelectorAll('[data-scene]').forEach(button=>button.onclick=async()=>{const viewer=await viewerReady;if(!viewer)return;viewer.setScene(button.dataset.scene);groupButtons('[data-scene]',button);const copy=sceneCopy[button.dataset.scene];$('scene-title').textContent=copy[0];$('scene-description').textContent=copy[1];sourceInfo(viewer.getInfo());});
document.querySelectorAll('[data-motion]').forEach(button=>button.onclick=async()=>{const viewer=await viewerReady;if(!viewer)return;button.disabled=true;try{await viewer.setMotion(button.dataset.motion);groupButtons('[data-motion]',button);}finally{button.disabled=false;}});
$('motion-playing').onchange=async()=>{const viewer=await viewerReady;viewer?.setPlaying($('motion-playing').checked);};
$('model-measurements').onchange=async()=>{const viewer=await viewerReady;viewer?.setMeasurements($('model-measurements').checked);};
$('reset-view').onclick=async()=>{const viewer=await viewerReady;viewer?.resetView();};
async function refresh(){
 if(loading)return;loading=true;
 try{const response=await fetch('/api/store',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);data=await response.json();renderProgress(data.training);renderCards();
 if(data.robot?.available){$('robot-joints').textContent=data.robot.active_joints;$('robot-mass').replaceChildren(node('span',data.robot.mass_kg.toFixed(2)),node('small',' kg'));}
 if(initial){renderCompare();initial=false;const id=new URLSearchParams(location.search).get('motion');const item=data.catalog.find(v=>v.id===id);if(item)openDetail(item);}
 }catch(error){$('connection').textContent='無法取得最新資料；保留上次顯示，請檢查連線。';$('connection').className='error';}
 finally{loading=false;}
}
refresh();setInterval(refresh,5000);
