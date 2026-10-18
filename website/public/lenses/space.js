import {createClusterLabels} from './clusters.js?v=paper-touch1';
import {createTrackLabels} from './labels.js?v=paper-touch1';
import {densitySurface} from './density.js?v=canvas2';
export const palette=['#635bff','#ff3d71','#00c5e8','#ffb000','#00d99a','#b35cff'];
const $=s=>document.querySelector(s),clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
const node=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e;};
const BACKGROUND_OPACITY=.72;
const rgb=palette.map(c=>[1,3,5].map(i=>parseInt(c.slice(i,i+2),16)));
const mix=(a,b,t)=>a+(b-a)*t;
const opacity=kind=>kind==='result'?1:kind==='base'?.88:BACKGROUND_OPACITY;
const STAR='M0-9 2.65-3.1 9.5-2.9 4.1 1.45 5.7 8 0 4.35-5.7 8-4.1 1.45-9.5-2.9-2.65-3.1Z';
// Camera coordinates are applied before drawing, never as a CSS bitmap scale.
// Only the seed and retrieval results have DOM hit targets; the cloud is canvas.
export function createSpace({playButton,chooseSeed,highlight,isBusy,getTrack,loadTrack}){
  const stage=$('#map-stage'),world=$('#space-world'),canvas=node('canvas','map-canvas');
  canvas.id='map-canvas';canvas.setAttribute('role','img');canvas.setAttribute('aria-label','Music map. Bright dots are the ten nearest neighbours in the selected similarity space. Hover a dot to preview; drag to pan; scroll to zoom.');
  world.replaceChildren(canvas);const ctx=canvas.getContext('2d');
  const densityLayers=new Map();let densityElapsed=1000;
  let motionEnabled=localStorage.getItem('music-lenses-map-motion')!=='off';
  const points=new Map(),cards=new Map(),trackIndex=new Map(),metadataCache=new Map(),densityCache=new Map();
  let result,compare=false,width=0,height=0,pixelRatio=0,zoom=1,pan={x:0,y:0},drag=null,moved=false;
  const touches=new Map();let pinch=null,suppressTapUntil=0;
  let frame=0,elapsed=0,lastFrame=null,duration=0,openId=null,focusedId=null,closeTimer=0,metadataEpoch=0,activeGroup=null,showDensity=false,grid=new Map();
  const densityWorker=new Worker(new URL('./density-worker.js?v=canvas2',import.meta.url),{type:'module'});
  densityWorker.onmessage=({data})=>{
    // Composite the six soft fields once at their native grid resolution.
    // Camera frames then scale one texture per view, including during a blend.
    const first=data.fields[0];if(!first)return;
    const surface=node('canvas');surface.width=first.width;surface.height=first.height;
    const densityContext=surface.getContext('2d');
    for(const field of data.fields)densityContext.drawImage(densitySurface(field,palette[field.id%palette.length],24),0,0);
    const surfaces=[{bounds:first.bounds,surface,modes:data.fields.map(f=>({id:f.id,mode:f.mode}))}];
    densityCache.set(data.key,surfaces);if(densityCache.size>12)densityCache.delete(densityCache.keys().next().value);
    if(result&&data.key===densityKey())setDensity(data.key,surfaces);
  };
  function setDensity(key,fields){
    if(densityLayers.get(key)?.target===1)return;
    for(const layer of densityLayers.values()){layer.from=layer.opacity;layer.target=0;}
    let layer=densityLayers.get(key);if(!layer){layer={fields,opacity:0,from:0,target:1};densityLayers.set(key,layer);}else layer.target=1;
    densityElapsed=motionEnabled?0:1000;schedule();
  }
  const clusterLabels=createClusterLabels({stage,palette});
  const trackLabels=createTrackLabels({stage,points,openCard,closeCard,schedule});
  function densityKey(){return result.signature+':'+result.method+':'+result.plot.length;}
  function anchor(p){const mobile=width<700;return{x:width*(.5+p.x*.38),y:height*((mobile?.48:.47)-p.y*(mobile?.30:.36))};}
  function screen(p){const a=anchor(p);return{x:width/2+(a.x-width/2)*zoom+pan.x,y:height/2+(a.y-height/2)*zoom+pan.y};}
  function schedule(){if(!frame)frame=requestAnimationFrame(paint);}
  function fit(){zoom=1;pan={x:0,y:0};schedule();}
  function resize(){
    const rect=stage.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
    if(width!==rect.width||height!==rect.height||pixelRatio!==ratio){width=rect.width;height=rect.height;pixelRatio=ratio;canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);world.style.width=width+'px';world.style.height=height+'px';}
    schedule();
  }
  function paint(now){
    frame=0;
    // Advance by displayed frames, with a capped delta. A slow first frame or
    // a stalled main thread must never consume the entire transition unseen.
    const delta=lastFrame===null?0:clamp(now-lastFrame,0,50);lastFrame=now;
    elapsed=Math.min(duration,elapsed+delta);densityElapsed=Math.min(1000,densityElapsed+delta);
    if(pixelRatio!==(window.devicePixelRatio||1))resize();if(!width||!height)return;
    ctx.setTransform(canvas.width/width,0,0,canvas.height/height,0,0);ctx.clearRect(0,0,width,height);
    if(!result)return;
    const t=duration?clamp(elapsed/duration,0,1):1,eased=t*t*(3-2*t),radius=clamp((width<700?2.2:2.8)*Math.sqrt(zoom),1.8,6.8);
    stage.classList.toggle('transitioning',t<1);world.dataset.zoom=zoom.toFixed(3);
    canvas.dataset.pointCount=points.size;canvas.dataset.neighbours=JSON.stringify(result.results.map(t=>t.id));
    canvas.dataset.transitionProgress=t.toFixed(5);
    // Keep old density fields until the next ones are ready; only these soft
    // backgrounds blend. The songs themselves travel along continuous paths.
    const densityT=clamp(densityElapsed/1000,0,1);
    ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='low';
    for(const[key,layer]of densityLayers){
      layer.opacity=mix(layer.from,layer.target,densityT);
      if(densityT===1&&layer.target===0){densityLayers.delete(key);continue;}
      if(showDensity&&layer.opacity>0)for(const f of layer.fields){const a=screen({x:f.bounds.left,y:f.bounds.top}),b=screen({x:f.bounds.right,y:f.bounds.bottom});ctx.globalAlpha=layer.opacity;ctx.drawImage(f.surface,a.x,a.y,b.x-a.x,b.y-a.y);}
    }
    const batches=new Map();grid=new Map();
    for(const p of points.values()){
      const color=rgb[p.cluster%6],a=p.from;
      p.current={x:mix(a.x,p.x,eased),y:mix(a.y,p.y,eased),r:mix(a.r,color[0],eased),g:mix(a.g,color[1],eased),b:mix(a.b,color[2],eased)};
      const pos=screen(p.current);p.screen=pos;
      if(pos.x< -15||pos.y< -15||pos.x>width+15||pos.y>height+15)continue;
      const cell=Math.floor(pos.x/32)+','+Math.floor(pos.y/32);if(!grid.has(cell))grid.set(cell,[]);grid.get(cell).push(p);
      if(p.kind==='seed')continue;
      const layer=p.kind==='result'?2:p.kind==='base'?1:0,c=p.current;
      const packed=(Math.round(c.r)<<16)|(Math.round(c.g)<<8)|Math.round(c.b),key=layer*0x1000000+packed;
      let batch=batches.get(key);if(!batch){batch={layer,color:'#'+packed.toString(16).padStart(6,'0'),points:[]};batches.set(key,batch);}batch.points.push(p);
    }
    // Batch matching colours instead of creating a DOM element per song. Colour
    // interpolation survives interrupted changes just like position does.
    for(const batch of [...batches.values()].sort((a,b)=>a.layer-b.layer)){
      ctx.fillStyle=batch.color;ctx.globalAlpha=opacity(batch.layer===2?'result':batch.layer===1?'base':'background');
      ctx.beginPath();for(const p of batch.points){ctx.moveTo(p.screen.x+radius,p.screen.y);ctx.arc(p.screen.x,p.screen.y,radius,0,Math.PI*2);}ctx.fill();
    }
    ctx.globalAlpha=1;
    for(const[id,c]of cards){const p=points.get(id);if(p?.screen)c.style.transform=`translate(${p.screen.x}px,${p.screen.y}px)`;}
    clusterLabels.update({signature:densityKey(),groups:result.clusters,modes:densityLayers.get(densityKey())?.fields[0]?.modes,visible:showDensity&&t===1,screen,width,height});
    trackLabels.update(now,{zoom,radius,width,height,transitioning:t<1});
    if(openId)placeCard(cards.get(openId));
    if(t<1||(showDensity&&densityT<1))schedule();
  }
  function focus(id,on){focusedId=on?id:null;/* Hover never promotes a non-neighbour to full opacity. */}
  function closeCard(immediate=false){
    clearTimeout(closeTimer);
    const close=()=>{const c=cards.get(openId);if(!immediate&&(c?.matches(':hover')||trackLabels.isHovered(openId)))return;if(c){c.classList.remove('expanded');c.querySelector('.spatial-marker').setAttribute('aria-expanded','false');if(c.dataset.transient){c.remove();cards.delete(openId);}}openId=null;};
    if(immediate)close();else closeTimer=setTimeout(close,180);
  }
  function placeCard(c){
    if(!c)return;const p=points.get(c.track.id)?.screen;if(!p)return;
    const panel=c.querySelector('.spatial-content'),panelWidth=Math.min(220,width-30),reserve=92;
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
    const remaining=Math.max(0,duration-elapsed);
    const viewChanged=!result||result.signature!==r.signature||result.method!==r.method;
    closeCard(true);metadataEpoch++;result=r;compare=!!options.compare;activeGroup=null;trackIndex.clear();
    for(const t of[r.seed,...r.results,...r.baseline,...(r.representatives||[])])trackIndex.set(t.id,t);
    const bright=new Set(r.results.map(t=>t.id)),base=new Set(compare?r.baseline.map(t=>t.id):[]),old=new Map(points);points.clear();
    for(const p of r.plot){
      const previous=old.get(p.id),color=rgb[p.cluster%6];
      const from=previous?.current?{...previous.current}:{x:p.x,y:p.y,r:color[0],g:color[1],b:color[2]};
      points.set(p.id,{...p,kind:p.id===r.seed.id?'seed':bright.has(p.id)?'result':base.has(p.id)?'base':'background',from,current:{...from}});
    }
    for(const c of cards.values())c.remove();cards.clear();
    for(const t of[r.seed,...r.results,...(compare?r.baseline:[])]){if(cards.has(t.id))continue;const p=points.get(t.id);if(p)ensureCard(t,p.kind,p);}
    const key=densityKey();if(densityCache.has(key))setDensity(key,densityCache.get(key));
    else densityWorker.postMessage({key,points:r.plot,groups:r.clusters||[]});
    elapsed=0;lastFrame=null;duration=old.size&&motionEnabled?(viewChanged?1650:remaining):0;
    $('#map-empty').hidden=true;$('#map-title').textContent=r.method==='base'?'Full embedding space':`${r.category.name} space`;
    $('#map-explanation').textContent='Bright dots: 10 nearest neighbours · hover to preview · drag / scroll to explore';
    $('#cluster-legend').replaceChildren();resize();
  }
  canvas.addEventListener('pointermove',e=>{if(drag||e.pointerType==='touch')return;const rect=stage.getBoundingClientRect(),p=hit(e.clientX-rect.left,e.clientY-rect.top);canvas.style.cursor=p?'pointer':'grab';if(p)openCard(p.id);else closeCard();});
  canvas.addEventListener('pointerleave',e=>{if(!e.relatedTarget?.closest('.map-card'))closeCard();});
  stage.addEventListener('wheel',e=>{if(e.target.closest('.spatial-content'))return;e.preventDefault();closeCard(true);const rect=stage.getBoundingClientRect(),x=e.clientX-rect.left-width/2,y=e.clientY-rect.top-height/2,next=clamp(zoom*Math.exp(-e.deltaY*.0015),.55,16),ratio=next/zoom;pan={x:x-(x-pan.x)*ratio,y:y-(y-pan.y)*ratio};zoom=next;schedule();},{passive:false});
  function touchPose(){
    const [a,b]=[...touches.values()],rect=stage.getBoundingClientRect();
    return{x:(a.x+b.x)/2-rect.left-width/2,y:(a.y+b.y)/2-rect.top-height/2,distance:Math.max(1,Math.hypot(b.x-a.x,b.y-a.y))};
  }
  stage.addEventListener('pointerdown',e=>{
    if(e.target.closest('.spatial-content')||e.pointerType!=='touch'&&e.target.closest('button'))return;
    closeCard(true);
    if(e.pointerType==='touch'){
      touches.set(e.pointerId,{x:e.clientX,y:e.clientY});
      if(touches.size>=2){
        if(touches.size===2){pinch={...touchPose(),zoom,pan:{...pan}};drag=null;moved=true;for(const id of touches.keys())stage.setPointerCapture(id);}
        return;
      }
    }
    drag={x:e.clientX,y:e.clientY,px:pan.x,py:pan.y};moved=false;canvas.style.cursor='grabbing';
  });
  window.addEventListener('pointermove',e=>{
    if(touches.has(e.pointerId))touches.set(e.pointerId,{x:e.clientX,y:e.clientY});
    if(pinch&&touches.size>=2){
      const pose=touchPose(),next=clamp(pinch.zoom*pose.distance/pinch.distance,.55,16),ratio=next/pinch.zoom;
      pan={x:pose.x-(pinch.x-pinch.pan.x)*ratio,y:pose.y-(pinch.y-pinch.pan.y)*ratio};zoom=next;moved=true;schedule();return;
    }
    if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
    if(Math.abs(dx)+Math.abs(dy)>5)moved=true;
    if(moved){pan={x:drag.px+dx,y:drag.py+dy};schedule();}
  });
  function endPointer(e,cancelled=false){
    const wasTouch=touches.delete(e.pointerId);
    if(wasTouch&&pinch){
      suppressTapUntil=performance.now()+400;
      if(touches.size>=2)pinch={...touchPose(),zoom,pan:{...pan}};
      else{pinch=null;const remaining=touches.values().next().value;drag=remaining?{x:remaining.x,y:remaining.y,px:pan.x,py:pan.y}:null;}
      moved=true;return;
    }
    if(!drag)return;drag=null;canvas.style.cursor='grab';
    if(!cancelled&&!moved&&performance.now()>suppressTapUntil){const rect=stage.getBoundingClientRect(),p=hit(e.clientX-rect.left,e.clientY-rect.top);if(p)openCard(p.id);}
  }
  window.addEventListener('pointerup',e=>endPointer(e));
  window.addEventListener('pointercancel',e=>endPointer(e,true));
  stage.addEventListener('click',e=>{if(performance.now()<suppressTapUntil){e.preventDefault();e.stopPropagation();}},true);
  document.addEventListener('pointerdown',e=>{if(!stage.contains(e.target))closeCard(true);});
  stage.addEventListener('keydown',e=>{if(e.key==='Escape'){closeCard(true);document.activeElement?.blur();}});
  $('#reset-map').onclick=()=>{closeCard(true);fit();};$('#zoom-in').onclick=()=>{closeCard(true);zoom=clamp(zoom*1.2,.55,16);schedule();};$('#zoom-out').onclick=()=>{closeCard(true);zoom=clamp(zoom/1.2,.55,16);schedule();};
  $('#groups-toggle').onclick=()=>{showDensity=!showDensity;$('#groups-toggle').setAttribute('aria-pressed',String(showDensity));schedule();};
  const motionToggle=$('#motion-toggle');
  function updateMotionControl(){
    canvas.dataset.motion=motionEnabled?'on':'off';
    if(motionToggle){motionToggle.setAttribute('aria-checked',String(motionEnabled));motionToggle.querySelector('.motion-state').textContent=motionEnabled?'On':'Off';}
  }
  if(motionToggle)motionToggle.onclick=()=>{
    motionEnabled=!motionEnabled;localStorage.setItem('music-lenses-map-motion',motionEnabled?'on':'off');
    if(!motionEnabled){elapsed=duration;densityElapsed=1000;}updateMotionControl();schedule();
  };
  updateMotionControl();
  function watchResolution(){const query=matchMedia(`(resolution: ${window.devicePixelRatio||1}dppx)`);query.addEventListener('change',()=>{resize();watchResolution();},{once:true});}
  watchResolution();new ResizeObserver(resize).observe(stage);window.addEventListener('resize',resize);document.addEventListener('themechange',schedule);
  return{render,focus,fit,setStudy(study,hidden){closeCard(true);$('#study-map-cover').hidden=!(study&&hidden);stage.classList.toggle('blind',study&&hidden);},clear(){closeCard(true);result=null;trackLabels.clear();clusterLabels.clear();densityLayers.clear();points.clear();cards.clear();trackIndex.clear();$('#map-cards').replaceChildren();$('#cluster-legend').replaceChildren();schedule();}};
}
