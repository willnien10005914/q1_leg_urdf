const $=id=>document.getElementById(id),num=v=>Number.isFinite(v)?v.toLocaleString('en-US'):'—',pct=v=>Number.isFinite(v)?(v*100).toFixed(1)+'%':'—';
const states={training:'PPO 訓練中',evaluating:'物理評估中',prepared:'準備訓練',waiting_for_other_compute:'等待 GPU',queued:'訓練佇列中',budget_exhausted:'預算已完成',runtime_error:'訓練已停止',stopped:'已停止',not_started:'尚未開始',updating:'狀態更新中'};
let data,filter='all',loading=false;
function node(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function renderProgress(t){
 $('state').textContent=states[t.state]||t.state||'未知狀態';
 $('connection').className=t.stale||t.error?'error':'';
 $('connection').textContent=t.error?'工作已停止：'+t.error:t.stale?'狀態超過兩分鐘未更新，請檢查訓練服務。':t.state==='not_started'?'尚未收到訓練資料。':'直接讀取 Spark 保存的狀態 · 未通過完整驗收';
 const saved=t.saved_iterations??0,budget=t.max_new_iterations??15000;
 $('saved').textContent=num(saved);$('budget').textContent=num(budget);$('percent').textContent=pct(saved/budget);$('progress').max=budget;$('progress').value=saved;
 $('envs').textContent=num(t.num_envs??6144);$('live').textContent=t.state==='training'&&t.live_iteration>saved?num(t.live_iteration):'—';
 $('rate').textContent=t.state==='training'?num(t.latest_steps_per_second):'評估／等待中';
 $('eta').textContent=['budget_exhausted','stopped','runtime_error'].includes(t.state)?'—':Number.isFinite(t.estimated_remaining_s)?(t.estimated_remaining_s/3600).toFixed(1)+' 小時':'估算中';
 const evals=t.evaluations||[],last=evals.at(-1);
 $('eval-version').textContent=last?'第 '+num(last.completed_iterations)+' 次 · '+(last.episodes??'—')+' 回合':'尚未評估';
 $('survival').textContent=pct(last?.four_step_survival);$('completion').textContent=pct(last?.completed_four_step_rate);
 $('updated').textContent=t.updated_at?'更新於 '+new Date(t.updated_at*1000).toLocaleString('zh-TW'):'尚無更新';
 $('commit').textContent=t.source_git_commit?'來源 '+t.source_git_commit.slice(0,7):'';chart(evals);
}
function svg(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
function chart(evals){
 const el=$('chart');el.replaceChildren();
 for(const v of[0,.5,1]){const y=118-v*98;el.append(svg('line',{x1:36,x2:688,y1:y,y2:y,stroke:'#e6ece5','stroke-width':1}));el.append(svg('text',{x:0,y:y+3,fill:'#8a9991','font-size':9},v*100+'%'));}
 const points=evals.filter(v=>Number.isFinite(v.completed_iterations));
 if(!points.length){el.append(svg('text',{x:350,y:72,'text-anchor':'middle',fill:'#8a9991','font-size':11},'完成首次評估後顯示曲線'));return;}
 const maximum=Math.max(...points.map(v=>v.completed_iterations),500),x=v=>36+v/maximum*645;
 for(const[key,color]of[['four_step_survival','#187b60'],['completed_four_step_rate','#91aa9b']]){
  const values=points.filter(v=>Number.isFinite(v[key]));
  el.append(svg('polyline',{points:values.map(v=>x(v.completed_iterations)+','+(118-v[key]*98)).join(' '),fill:'none',stroke:color,'stroke-width':2}));
  for(const p of values){const c=svg('circle',{cx:x(p.completed_iterations),cy:118-p[key]*98,r:3,fill:color});c.append(svg('title',{},p.completed_iterations+' updates: '+pct(p[key])));el.append(c);}
 }
 el.append(svg('text',{x:36,y:140,fill:'#8a9991','font-size':9},'0'));el.append(svg('text',{x:680,y:140,'text-anchor':'end',fill:'#8a9991','font-size':9},num(maximum)+' updates'));
}
function renderCards(){
 const q=$('search').value.trim().toLowerCase(),items=(data?.catalog||[]).filter(v=>(filter==='all'||v.category===filter)&&[v.title,v.version,v.description].join(' ').toLowerCase().includes(q));
 $('cards').replaceChildren();$('count').textContent=items.length+' 個版本';$('empty').hidden=items.length>0;
 for(const item of items){
  const card=node('button',undefined,'card');card.type='button';card.setAttribute('aria-label',item.title+'，'+(item.video?'查看影片':'查看進度'));const media=node('div',undefined,'card-media');
  if(item.poster){const img=node('img');img.src=item.poster;img.alt=item.title+' 原生物理診斷';img.loading='lazy';media.append(img);}
  else{const art=node('div',undefined,'spark-art'),group=node('div');group.append(node('strong','6,144'),node('small','PARALLEL ENVIRONMENTS'));art.append(group);media.append(art);}
  media.append(node('span',item.id==='spark-balance-v1'?(states[data.training.state]||'Spark 訓練'):item.label,'pill'+(item.id==='v6'?' warning':'')));if(item.video)media.append(node('span','▶','play'));
  const body=node('div',undefined,'card-body');body.append(node('h3',item.title),node('p',item.description),node('div',item.origin,'origin'));
  const meta=node('div',undefined,'card-meta');meta.append(node('span',item.version),node('span',item.backend),node('span',num(item.iterations)+' updates'));
  card.append(media,body,meta);card.onclick=()=>openDetail(item);$('cards').append(card);
 }
}
function openDetail(item){
 $('detail-origin').textContent=item.origin;$('detail-title').textContent=item.title;$('detail-description').textContent=item.description;$('detail-meta').textContent=item.version+' · '+item.backend+' · '+num(item.iterations)+' updates · 尚未正式驗收';$('media').replaceChildren();
 if(item.video){const video=node('video');video.controls=true;video.playsInline=true;video.preload='metadata';video.src=item.video;video.poster=item.poster;$('media').append(video);}
 else{const box=node('div',undefined,'no-video');box.append(node('p','本輪 Spark 結果影片尚未錄製。'),node('p','目前顯示真實訓練進度，歷史影片保留為獨立版本。'));const a=node('a','查看即時訓練 ↑');a.href='#training';a.onclick=closeDetail;box.append(a);$('media').append(box);}
 $('detail').showModal();
}
function closeDetail(){const video=$('media').querySelector('video');if(video)video.pause();$('detail').close();}
$('close').onclick=closeDetail;$('detail').addEventListener('close',()=>{const v=$('media').querySelector('video');if(v)v.pause();});
$('detail').addEventListener('click',e=>{if(e.target===$('detail')){const r=$('detail').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)closeDetail();}});
$('search').oninput=renderCards;
document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>{filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});renderCards();});
async function refresh(){
 if(loading)return;loading=true;
 try{const response=await fetch('/api/store',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);data=await response.json();renderProgress(data.training);renderCards();}
 catch(error){$('connection').textContent='無法取得最新資料；保留上次顯示，請檢查連線。';$('connection').className='error';}
 finally{loading=false;}
}
refresh();setInterval(refresh,5000);
