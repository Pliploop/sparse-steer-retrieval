import {densitySurface} from './density.js?v=canvas2';
export const palette=['#6366f1','#f43f5e','#06b6d4','#f59e0b','#10b981','#a855f7'];
const $=s=>document.querySelector(s),clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
const node=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e;};
const BACKGROUND_OPACITY=.34;
const STAR='M0-9 2.65-3.1 9.5-2.9 4.1 1.45 5.7 8 0 4.35-5.7 8-4.1 1.45-9.5-2.9-2.65-3.1Z';
// Camera coordinates are applied before drawing, never as a CSS bitmap scale.
// Only the seed and retrieval results have DOM hit targets; the cloud is canvas.
export function createSpace({playButton,chooseSeed,highlight,isBusy,getTrack,loadTrack}){
  const stage=$('#map-stage'),world=$('#space-world'),canvas=node('canvas','map-canvas');
  canvas.id='map-canvas';canvas.setAttribute('role','img');canvas.setAttribute('aria-label','Music map. Bright dots are the ten nearest neighbours in the selected similarity space. Hover a dot to preview; drag to pan; scroll to zoom.');
  world.replaceChildren(canvas);const ctx=canvas.getContext('2d');
  const points=new Map(),cards=new Map(),trackIndex=new Map(),metadataCache=new Map(),densityCache=new Map();
  let result,compare=false,width=0,height=0,pixelRatio=0,zoom=1,pan={x:0,y:0},drag=null,moved=false;
  let frame=0,start=0,duration=0,openId=null,focusedId=null,closeTimer=0,metadataEpoch=0,activeGroup=null,fields=[],showDensity=false,grid=new Map();
  const densityWorker=new Worker(new URL('./density-worker.js?v=canvas2',import.meta.url),{type:'module'});
  densityWorker.onmessage=({data})=>{
    const surfaces=data.fields.map(f=>({...f,surface:densitySurface(f,palette[f.id%palette.length],24)}));
    densityCache.set(data.key,surfaces);if(densityCache.size>12)densityCache.delete(densityCache.keys().next().value);
    if(result&&data.key===densityKey()){fields=surfaces;schedule();}
  };
  function densityKey(){return result.signature+':'+result.method+':'+result.plot.length;}
  function anchor(p){return{x:width*(.5+p.x*.265),y:height*(.5-p.y*.29)};}
  function screen(p){const a=anchor(p);return{x:width/2+(a.x-width/2)*zoom+pan.x,y:height/2+(a.y-height/2)*zoom+pan.y};}
  function schedule(){if(!frame)frame=requestAnimationFrame(paint);}
  function fit(){zoom=1;pan={x:0,y:0};schedule();}
  function resize(){
    const rect=stage.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
    if(width!==rect.width||height!==rect.height||pixelRatio!==ratio){width=rect.width;height=rect.height;pixelRatio=ratio;canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);world.style.width=width+'px';world.style.height=height+'px';}
    schedule();
  }
  function paint(now){
    frame=0;if(pixelRatio!==(window.devicePixelRatio||1))resize();if(!width||!height)return;
    ctx.setTransform(canvas.width/width,0,0,canvas.height/height,0,0);ctx.clearRect(0,0,width,height);
    if(!result)return;
    const t=duration?clamp((now-start)/duration,0,1):1,eased=t*t*(3-2*t),radius=clamp(1.8*Math.sqrt(zoom),1.35,4.4);
    stage.classList.toggle('transitioning',t<1);world.dataset.zoom=zoom.toFixed(3);
    canvas.dataset.pointCount=points.size;canvas.dataset.neighbours=JSON.stringify(result.results.map(t=>t.id));
    // Density is deliberately soft; circles are always rasterized afresh at native DPR.
    if(showDensity){ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';for(const f of fields){const a=screen({x:f.bounds.left,y:f.bounds.top}),b=screen({x:f.bounds.right,y:f.bounds.bottom});ctx.globalAlpha=activeGroup!==null&&f.id!==activeGroup ? .15 : 1;ctx.drawImage(f.surface,a.x,a.y,b.x-a.x,b.y-a.y);}ctx.globalAlpha=1;}
    const batches=Array.from({length:18},()=>[]);grid=new Map();
    for(const p of points.values()){
      p.current={x:p.from.x+(p.x-p.from.x)*eased,y:p.from.y+(p.y-p.from.y)*eased};
      const pos=screen(p.current);p.screen=pos;
      if(pos.x< -15||pos.y< -15||pos.x>width+15||pos.y>height+15)continue;
      const cell=Math.floor(pos.x/32)+','+Math.floor(pos.y/32);if(!grid.has(cell))grid.set(cell,[]);grid.get(cell).push(p);
      if(p.kind==='seed')continue;
      const layer=p.kind==='result'?2:p.kind==='base'?1:0;
      batches[layer*6+p.cluster%6].push(p);
    }
    // All dots share one radius; opacity alone identifies exact retrieval results.
    for(let b=0;b<batches.length;b++){
      const group=b%6,layer=Math.floor(b/6);ctx.fillStyle=palette[group];ctx.globalAlpha=(layer===2?1:layer===1?.45:BACKGROUND_OPACITY)*(activeGroup!==null&&activeGroup!==group?.13:1);
      ctx.beginPath();for(const p of batches[b]){ctx.moveTo(p.screen.x+radius,p.screen.y);ctx.arc(p.screen.x,p.screen.y,radius,0,Math.PI*2);}ctx.fill();
    }
    ctx.globalAlpha=1;
    for(const[id,c]of cards){const p=points.get(id);if(p?.screen)c.style.transform=`translate(${p.screen.x}px,${p.screen.y}px)`;}
    if(openId)placeCard(cards.get(openId));
    if(t<1)schedule();
  }
  function focus(id,on){focusedId=on?id:null;/* Hover never promotes a non-neighbour to full opacity. */}
  function closeCard(immediate=false){
    clearTimeout(closeTimer);
    const close=()=>{const c=cards.get(openId);if(!immediate&&c?.matches(':hover'))return;if(c){c.classList.remove('expanded');c.querySelector('.spatial-marker').setAttribute('aria-expanded','false');if(c.dataset.transient){c.remove();cards.delete(openId);}}openId=null;};
    if(immediate)close();else closeTimer=setTimeout(close,180);
  }
  function placeCard(c){
    if(!c)return;const p=points.get(c.track.id)?.screen;if(!p)return;
    const panel=c.querySelector('.spatial-content'),panelWidth=Math.min(220,width-30),reserve=width<700?150:105;
    c.style.setProperty('--popup-width',panelWidth+'px');panel.style.maxHeight=Math.min(300,height-80-reserve)+'px';
    panel.style.left=clamp(19,15-p.x,width-15-p.x-panelWidth)+'px';
    panel.style.top=clamp(-25,90-p.y,height-reserve-p.y-(panel.offsetHeight||190))+'px';
  }
  function openCard(id){
    clearTimeout(closeTimer);if(isBusy()||drag)return;
    const p=points.get(id);if(!p)return;
    if(openId!==id){closeCard(true);openId=id;}
    const c=cards.get(id)||ensureCard(trackFor(p),p.kind,p,true);
    c.classList.add('expanded');c.querySelector('.spatial-marker').setAttribute('aria-expanded','true');
    placeCard(c);hydrate(c.track,c);
  }
  function cardContent(track,kind){
    const content=node('div','spatial-content');content.id='preview-'+track.id;content.setAttribute('aria-label',`${track.title} preview`);
    const top=node('div','spatial-top');top.append(node('span','spatial-kind',kind==='seed'?'Starting song':kind==='result'?`Neighbour ${String(track.rank).padStart(2,'0')}`:kind==='base'?'Full-space neighbour':'Explore this song'),node('i','cluster-dot'));content.append(top);
    const body=node('div','spatial-body'),text=node('div','spatial-text'),title=node('button','spatial-title',track.title);
    title.onclick=()=>{if(!isBusy())chooseSeed(track);};text.append(title,node('span','spatial-artist',track.artist));body.append(text,playButton(track));content.append(body);
    const tags=node('div','spatial-tags');for(const attr of(track.attributes||[]).slice(0,2))tags.append(node('span','spatial-attribute',attr.label));content.append(tags);
    const details=node('div','spatial-details');details.append(highlight(track.caption,track.attributes));
    if(kind!=='seed'){const next=node('button','spatial-next','Explore from here');next.onclick=()=>chooseSeed(track);details.append(next);}content.append(details);return content;
  }
  function trackFor(p){return trackIndex.get(p.id)||{...(getTrack?.(p.id)||metadataCache.get(p.id)||{id:p.id,title:p.title,artist:p.artist,audio:`/api/audio/${p.id}`,start:0,end:30}),cluster:p.cluster};}
  async function hydrate(track,c){
    if(!track||(track.caption&&(!loadTrack||Array.isArray(track.attributes))))return;
    const epoch=metadataEpoch;
    try{
      let full;if(loadTrack)full=await loadTrack(track.id);else {full=metadataCache.get(track.id);if(!full){const response=await fetch('/api/track/'+encodeURIComponent(track.id));if(!response.ok)return;full=await response.json();metadataCache.set(track.id,full);}}
      if(epoch!==metadataEpoch)return;Object.assign(track,full);
      if(c.isConnected&&c.track?.id===track.id){c.querySelector('.card-caption')?.replaceWith(highlight(track.caption,track.attributes));if(full.attributes)c.querySelector('.spatial-tags').replaceChildren(...full.attributes.slice(0,2).map(a=>node('span','spatial-attribute',a.label)));placeCard(c);}
    }catch{}
  }
  function ensureCard(track,kind,p,transient=false){
    const c=node('article',`map-card ${kind}`);c.track=track;c.dataset.track=track.id;c.dataset.cluster=p.cluster;if(transient)c.dataset.transient='true';c.style.setProperty('--cluster',palette[p.cluster%palette.length]);
    const marker=node('button','spatial-marker');marker.setAttribute('aria-label',`${kind==='seed'?'Starting song: ':''}${track.title} — ${track.artist}. Show preview`);marker.setAttribute('aria-expanded','false');marker.setAttribute('aria-controls','preview-'+track.id);
    if(kind==='seed'){
      const icon=document.createElementNS('http://www.w3.org/2000/svg','svg');icon.setAttribute('viewBox','-12 -12 24 24');icon.setAttribute('aria-hidden','true');icon.setAttribute('class','seed-star');const path=document.createElementNS(icon.namespaceURI,'path');path.setAttribute('d',STAR);icon.append(path);marker.append(icon);
    }
    marker.onclick=e=>{e.stopPropagation();openCard(track.id);};marker.onfocus=()=>openCard(track.id);
    c.append(marker,cardContent(track,kind));c.onmouseenter=()=>openCard(track.id);c.onmouseleave=()=>closeCard();
    c.addEventListener('focusin',()=>openCard(track.id));c.addEventListener('focusout',e=>{if(!c.contains(e.relatedTarget))closeCard();});
    if(p.screen)c.style.transform=`translate(${p.screen.x}px,${p.screen.y}px)`;
    $('#map-cards').append(c);cards.set(track.id,c);return c;
  }
  function hit(x,y){
    const cx=Math.floor(x/32),cy=Math.floor(y/32);let best=null,dist=100;
    for(let i=-1;i<=1;i++)for(let j=-1;j<=1;j++)for(const p of grid.get((cx+i)+','+(cy+j))||[]){const d=(p.screen.x-x)**2+(p.screen.y-y)**2;if(d<dist){dist=d;best=p;}}
    return best;
  }
  function render(r,options={}){
    closeCard(true);metadataEpoch++;result=r;compare=!!options.compare;activeGroup=null;trackIndex.clear();
    for(const t of[r.seed,...r.results,...r.baseline,...(r.representatives||[])])trackIndex.set(t.id,t);
    const bright=new Set(r.results.map(t=>t.id)),base=new Set(compare?r.baseline.map(t=>t.id):[]),old=new Map(points);points.clear();
    for(const p of r.plot){const previous=old.get(p.id);points.set(p.id,{...p,kind:p.id===r.seed.id?'seed':bright.has(p.id)?'result':base.has(p.id)?'base':'background',from:previous?.current||{x:p.x,y:p.y},current:previous?.current||{x:p.x,y:p.y}});}
    for(const c of cards.values())c.remove();cards.clear();
    for(const t of[r.seed,...r.results,...(compare?r.baseline:[])]){if(cards.has(t.id))continue;const p=points.get(t.id);if(p)ensureCard(t,p.kind,p);}
    const key=densityKey();fields=densityCache.get(key)||[];
    if(!densityCache.has(key))densityWorker.postMessage({key,points:r.plot,groups:r.clusters||[]});
    start=performance.now();duration=old.size&&!matchMedia('(prefers-reduced-motion: reduce)').matches?1150:0;
    $('#map-empty').hidden=true;$('#map-title').textContent=r.method==='base'?'Full embedding space':`${r.category.name} space`;
    $('#map-explanation').textContent='Bright dots: 10 nearest neighbours · hover to preview · drag / scroll to explore';
    $('#cluster-legend').replaceChildren();resize();
  }
  canvas.addEventListener('pointermove',e=>{if(drag||e.pointerType==='touch')return;const rect=stage.getBoundingClientRect(),p=hit(e.clientX-rect.left,e.clientY-rect.top);canvas.style.cursor=p?'pointer':'grab';if(p)openCard(p.id);else closeCard();});
  canvas.addEventListener('pointerleave',e=>{if(!e.relatedTarget?.closest('.map-card'))closeCard();});
  stage.addEventListener('wheel',e=>{if(e.target.closest('.spatial-content'))return;e.preventDefault();closeCard(true);const rect=stage.getBoundingClientRect(),x=e.clientX-rect.left-width/2,y=e.clientY-rect.top-height/2,next=clamp(zoom*Math.exp(-e.deltaY*.0015),.55,6),ratio=next/zoom;pan={x:x-(x-pan.x)*ratio,y:y-(y-pan.y)*ratio};zoom=next;schedule();},{passive:false});
  stage.addEventListener('pointerdown',e=>{if(e.target.closest('button,.spatial-content'))return;closeCard(true);drag={x:e.clientX,y:e.clientY,px:pan.x,py:pan.y};moved=false;canvas.style.cursor='grabbing';});
  window.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>5)moved=true;if(moved){pan={x:drag.px+dx,y:drag.py+dy};schedule();}});
  window.addEventListener('pointerup',e=>{if(!drag)return;drag=null;canvas.style.cursor='grab';if(!moved){const rect=stage.getBoundingClientRect(),p=hit(e.clientX-rect.left,e.clientY-rect.top);if(p)openCard(p.id);}});
  window.addEventListener('pointercancel',()=>{drag=null;});
  document.addEventListener('pointerdown',e=>{if(!stage.contains(e.target))closeCard(true);});
  stage.addEventListener('keydown',e=>{if(e.key==='Escape'){closeCard(true);document.activeElement?.blur();}});
  $('#reset-map').onclick=()=>{closeCard(true);fit();};$('#zoom-in').onclick=()=>{closeCard(true);zoom=clamp(zoom*1.2,.55,6);schedule();};$('#zoom-out').onclick=()=>{closeCard(true);zoom=clamp(zoom/1.2,.55,6);schedule();};
  $('#groups-toggle').onclick=()=>{showDensity=!showDensity;$('#groups-toggle').setAttribute('aria-pressed',String(showDensity));schedule();};
  function watchResolution(){const query=matchMedia(`(resolution: ${window.devicePixelRatio||1}dppx)`);query.addEventListener('change',()=>{resize();watchResolution();},{once:true});}
  watchResolution();new ResizeObserver(resize).observe(stage);window.addEventListener('resize',resize);document.addEventListener('themechange',schedule);
  return{render,focus,fit,setStudy(study,hidden){closeCard(true);$('#study-map-cover').hidden=!(study&&hidden);stage.classList.toggle('blind',study&&hidden);},clear(){closeCard(true);result=null;points.clear();cards.clear();trackIndex.clear();$('#map-cards').replaceChildren();$('#cluster-legend').replaceChildren();schedule();}};
}
