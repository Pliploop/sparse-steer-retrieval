import {estimateDensities,densityImage} from './density.js';
// Measured points first. One hover/focus card at a time; no displaced card grid.
const NS='http://www.w3.org/2000/svg';
export const palette=['#6366f1','#f43f5e','#06b6d4','#f59e0b','#10b981','#a855f7'];
const $=s=>document.querySelector(s);
const svg=(tag,attrs={})=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);return e;};
const node=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e;};
const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
const BACKGROUND_OPACITY=.18;
const STAR='M0-9 2.65-3.1 9.5-2.9 4.1 1.45 5.7 8 0 4.35-5.7 8-4.1 1.45-9.5-2.9-2.65-3.1Z';
export function createSpace({playButton,chooseSeed,highlight,isBusy,getTrack,loadTrack}){
  const stage=$('#map-stage'),world=$('#space-world');
  stage.style.setProperty('--background-dot-opacity',BACKGROUND_OPACITY);
  let result,compare=false,showDensity=false,activeGroup=null,zoom=1,pan={x:0,y:0},drag=null,moved=false,width=1000,height=620;
  let frame=0,resizeFrame=0,cameraFrame=0,tilesTimer=0,lastTileUpdate=0,openId=null,pinnedId=null,closeTimer=null,focusedId=null;
  let shownTiles=new Set(),metadataEpoch=0;
  const densityCache=new Map(),densityLayers=new Map(),metadataCache=new Map(),tiles=new Map();
  const points=new Map(),cards=new Map(),motions=new Map(),trackIndex=new Map();
  function screen(v){return{x:width/2+(v.x-width/2)*zoom+pan.x,y:height/2+(v.y-height/2)*zoom+pan.y};}
  function position(element,v,overlay=false){const p=overlay?screen(v):v;element.style.transform=`translate(${p.x}px,${p.y}px)`;}
  function overlayPositions(){
    for(const card of cards.values()){const v=motions.get(card)?.current;if(v)position(card,v,true);}
    for(const[id,tile]of tiles){const v=motions.get(points.get(id))?.current;if(v)position(tile,v,true);}
  }
  function camera(){
    if(cameraFrame)return;
    cameraFrame=requestAnimationFrame(()=>{
      cameraFrame=0;world.style.transform=`translate(${pan.x}px,${pan.y}px) scale(${zoom})`;world.dataset.zoom=zoom.toFixed(3);overlayPositions();
      clearTimeout(tilesTimer);
      if(performance.now()-lastTileUpdate>80){updateTiles();lastTileUpdate=performance.now();}
      tilesTimer=setTimeout(updateTiles,85);
    });
  }
  function fit(){zoom=1;pan={x:0,y:0};camera();}
  function anchor(p){return{x:width*(.5+p.x*.265),y:height*(.5-p.y*.29)};}
  function move(element,target,kind='point'){
    let entry=motions.get(element);
    if(!entry){const initial=kind==='cloud'?{...target,rx:0,ry:0}:{x:width/2,y:height/2};entry={current:initial,kind};motions.set(element,entry);}
    entry.target=target;
  }
  function animate(){
    cancelAnimationFrame(frame);
    const entries=[...motions].filter(([element])=>{if(element.isConnected)return true;motions.delete(element);return false;});
    for(const[,entry]of entries)entry.from={...entry.current};
    const duration=matchMedia('(prefers-reduced-motion: reduce)').matches?0:1150,start=performance.now();
    stage.classList.toggle('transitioning',duration>0);
    function tick(now){
      const t=duration?clamp((now-start)/duration,0,1):1;
      const eased=t*t*(3-2*t);
      for(const[element,entry]of entries){
        for(const key of Object.keys(entry.target))entry.current[key]=(entry.from[key]??entry.target[key])+(entry.target[key]-(entry.from[key]??entry.target[key]))*eased;
        const v=entry.current;
        if(entry.kind==='cloud')for(const key of ['cx','cy','rx','ry'])element.setAttribute(key,v[key]);
        else position(element,v,element.classList.contains('map-card'));
      }
      overlayPositions();
      if(t<1)frame=requestAnimationFrame(tick);else {stage.classList.remove('transitioning');updateTiles();}
    }
    frame=requestAnimationFrame(tick);
  }
  function focus(id,on){if(focusedId)points.get(focusedId)?.classList.remove('focus');focusedId=on?id:null;if(focusedId)points.get(focusedId)?.classList.add('focus');}
  function closeCard(force=false){
    clearTimeout(closeTimer);
    if(pinnedId&&!force)return;
    if(force)pinnedId=null;
    const close=()=>{const current=cards.get(openId);if(current){current.classList.remove('expanded');current.querySelector('.spatial-marker').setAttribute('aria-expanded','false');}openId=null;};
    if(force)close();else closeTimer=setTimeout(close,150);
  }
  function openCard(id,pin=false){
    clearTimeout(closeTimer);if(isBusy()||drag)return;
    if(openId!==id){closeCard(true);openId=id;}
    if(pin)pinnedId=id;
    const c=cards.get(id);if(!c)return;
    c.classList.add('expanded');c.querySelector('.spatial-marker').setAttribute('aria-expanded','true');
    // Keep the expanded card inside the viewport, including while zoomed or panned.
    const pos=motions.get(c)?.current||{x:width/2,y:height/2};
    const screenX=width/2+(pos.x-width/2)*zoom+pan.x,screenY=height/2+(pos.y-height/2)*zoom+pan.y;
    const panel=c.querySelector('.spatial-content'),panelWidth=Math.min(220,width-30);
    c.style.setProperty('--popup-width',panelWidth+'px');
    const bottomReserve=width<700?150:105;
    panel.style.maxHeight=Math.min(300,height-80-bottomReserve)+'px';
    const popupHeight=panel.offsetHeight||190;
    // Counter-scale the card: its type and controls stay the same screen size.
    const left=clamp(27,23-screenX,width-1-screenX-panelWidth);
    const top=clamp(-25,86-screenY,height+11-bottomReserve-screenY-popupHeight);
    panel.style.left=left+'px';panel.style.top=top+'px';
    $('#map-tooltip').hidden=true;
    hydrate(c.track,c);
  }
  function selectedCards(r){
    const items=new Map();const add=(track,kind)=>{if(!items.has(track.id))items.set(track.id,{track,kind});};
    add(r.seed,'seed');r.results.forEach(t=>add(t,'result'));
    if(compare)r.baseline.forEach(t=>add(t,'base'));else(r.representatives||[]).forEach(t=>add(t,'context'));
    return[...items.values()];
  }
  function cardContent(track,kind,group){
    const content=node('div','spatial-content');content.id='preview-'+track.id;
    content.setAttribute('aria-label',`${track.title} preview`);
    const top=node('div','spatial-top');top.append(node('span','spatial-kind',kind==='seed'?'Starting song':kind==='result'?`Match ${String(track.rank).padStart(2,'0')}`:kind==='base'?'Full-space match':'From this group'),node('i','cluster-dot'));content.append(top);
    const body=node('div','spatial-body'),text=node('div','spatial-text');
    const title=node('button','spatial-title',track.title);title.onclick=()=>{if(!isBusy())chooseSeed(track);};
    text.append(title,node('span','spatial-artist',track.artist));body.append(text,playButton(track));content.append(body);
    const tags=node('div','spatial-tags');
    for(const attr of(track.attributes||[]).slice(0,2))tags.append(node('span','spatial-attribute',attr.label));
    content.append(tags);
    const details=node('div','spatial-details');details.append(highlight(track.caption,track.attributes));
    if(kind!=='seed'){const next=node('button','spatial-next','Explore from here');next.onclick=()=>chooseSeed(track);details.append(next);}content.append(details);
    return content;
  }
  function trackFor(p){return trackIndex.get(p.id)||getTrack?.(p.id)||metadataCache.get(p.id)||{id:p.id,title:p.title,artist:p.artist,audio:`/api/audio/${p.id}`,start:0,end:30,cluster:p.cluster};}
  async function hydrate(track,c){
    if(!track||(track.caption&&(!loadTrack||track.attributes?.length)))return;
    const epoch=metadataEpoch;
    try{
      let full;
      if(loadTrack)full=await loadTrack(track.id);
      else {full=metadataCache.get(track.id);if(!full){const response=await fetch('/api/track/'+encodeURIComponent(track.id));if(!response.ok)return;full=await response.json();metadataCache.set(track.id,full);}}
      if(epoch!==metadataEpoch)return;
      Object.assign(track,full);
      if(c.isConnected&&c.track?.id===track.id){
        c.querySelector('.card-caption')?.replaceWith(highlight(track.caption,track.attributes));
        if(full.attributes)c.querySelector('.spatial-tags').replaceChildren(...full.attributes.slice(0,2).map(a=>node('span','spatial-attribute',a.label)));
      }
    }catch{}
  }
  function ensureCard(track,kind,p,immediate=false){
    let c=cards.get(track.id);const group=result.clusters?.find(g=>g.id===track.cluster),color=palette[(track.cluster||0)%palette.length];
    if(!c){c=node('article','map-card');c.dataset.track=track.id;$('#map-cards').append(c);cards.set(track.id,c);c.addEventListener('focusin',()=>openCard(c.dataset.track));c.addEventListener('focusout',e=>{if(!c.contains(e.relatedTarget))closeCard();});}
    c.track=track;c.className=`map-card ${kind}${activeGroup!==null&&track.cluster!==activeGroup?' muted':''}`;c.dataset.cluster=track.cluster;c.style.setProperty('--cluster',color);
    const marker=node('button','spatial-marker');marker.setAttribute('aria-label',`${kind==='seed'?'Starting song: ':''}${track.title} — ${track.artist}. Show preview`);marker.setAttribute('aria-expanded','false');marker.setAttribute('aria-controls','preview-'+track.id);
    if(kind==='seed'){const icon=svg('svg',{viewBox:'-12 -12 24 24','aria-hidden':'true',class:'seed-star'});icon.append(svg('path',{d:STAR}));marker.append(icon);}
    // The transparent button is only a hit target; every non-seed dot is drawn
    // in the same SVG layer, so neighbours share its exact zoom behaviour.
    marker.onclick=e=>{e.stopPropagation();if(pinnedId===track.id)closeCard(true);else openCard(track.id,true);};marker.onfocus=()=>openCard(track.id);
    c.replaceChildren(marker,cardContent(track,kind,group));c.onmouseenter=()=>{openCard(track.id);focus(track.id,true);};c.onmouseleave=()=>{focus(track.id,false);closeCard();};
    move(c,anchor(p));if(immediate){const value=anchor(p);motions.get(c).current={...value};motions.get(c).target={...value};position(c,value,true);}return c;
  }
  function previewPoint(p,pin=false){const track=trackFor(p);if(!cards.has(p.id))ensureCard(track,p.kind||'context',p,true);openCard(p.id,pin);}
  function updateTiles(){
    if(!result)return;let layer=$('#zoom-tiles');if(!layer){layer=node('div','zoom-tiles');layer.id='zoom-tiles';stage.append(layer);}
    const amount=clamp((zoom-1.7)/1.35,0,1),shown=new Set(),occupied=[];
    const budget=Math.max(4,Math.min(16,Math.floor(width*height/70000)));
    if(amount>0){
      const candidates=[];
      for(const p of result.plot){if(p.id===result.seed.id)continue;const v=motions.get(points.get(p.id))?.current||anchor(p),pos=screen(v);if(pos.x>90&&pos.x<width-90&&pos.y>110&&pos.y<height-155)candidates.push({p,v,...pos});}
      candidates.sort((a,b)=>(a.p.kind==='result'?0:1)-(b.p.kind==='result'?0:1)||((a.x-width/2)**2+(a.y-height/2)**2)-((b.x-width/2)**2+(b.y-height/2)**2));
      for(const item of candidates){if(shown.size>=budget)break;if(occupied.some(o=>Math.abs(item.x-o.x)<190&&Math.abs(item.y-o.y)<86))continue;occupied.push(item);shown.add(item.p.id);
        let tile=tiles.get(item.p.id);const track=trackFor(item.p);
        if(!tile){tile=node('article','zoom-tile');tile.dataset.track=item.p.id;const text=node('button','zoom-tile-text');text.append(node('strong','',track.title),node('span','',track.artist));text.onclick=()=>previewPoint(item.p,true);const surface=node('div','zoom-tile-surface');surface.append(text,playButton(track));tile.append(surface);layer.append(tile);tiles.set(item.p.id,tile);}
        tile.style.setProperty('--cluster',palette[item.p.cluster%palette.length]);position(tile,item.v,true);tile.style.opacity=amount;tile.style.pointerEvents=amount>.3?'auto':'none';
      }
    }
    // Diff only the small visible set. Invisible cards are removed, never accumulated.
    for(const id of shownTiles)if(!shown.has(id)){const point=points.get(id);if(point){point.classList.remove('has-tile');point.firstElementChild.style.opacity='';}const c=cards.get(id);c?.classList.remove('has-tile');if(c)c.querySelector('.spatial-marker').style.opacity='';}
    for(const id of shown){const point=points.get(id);point?.classList.add('has-tile');if(point)point.firstElementChild.style.opacity=point.classList.contains('has-marker')?'0':String((point.classList.contains('result')?1:point.classList.contains('base')?.6:BACKGROUND_OPACITY)*(1-amount));const c=cards.get(id);c?.classList.add('has-tile');if(c)c.querySelector('.spatial-marker').style.opacity=1-amount;}
    for(const[id,tile]of tiles)if(!shown.has(id)){tile.remove();tiles.delete(id);}
    shownTiles=shown;
  }
  function draw(){
    if(!result)return;const r=result,rect=stage.getBoundingClientRect();width=Math.round(rect.width);height=Math.round(rect.height);if(!width||!height)return;
    $('#map').setAttribute('viewBox',`0 0 ${width} ${height}`);world.style.width=width+'px';world.style.height=height+'px';
    const selected=selectedCards(r),selectedIds=new Set(selected.map(s=>s.track.id)),byId=new Map(r.plot.map(p=>[p.id,p]));
    const ids=new Set(byId.keys());
    for(const[id,p]of points)if(!ids.has(id)){p.remove();points.delete(id);}
    for(const p of r.plot){
      let g=points.get(p.id);
      if(!g){g=svg('g',{'data-track':p.id});g.append(svg('path',{class:'point-glyph',d:'M0 0h.001','stroke-linecap':'round'}));$('#points').append(g);points.set(p.id,g);}
      g.setAttribute('class',`map-point ${p.kind}${p.kind==='seed'?' has-marker':''}${activeGroup!==null&&p.cluster!==activeGroup?' muted':''}`);
      g.dataset.cluster=p.cluster;g.style.setProperty('--cluster',palette[p.cluster%palette.length]);move(g,anchor(p));
      const circle=g.firstElementChild;circle.setAttribute('fill',palette[p.cluster%palette.length]);circle.setAttribute('stroke',palette[p.cluster%palette.length]);circle.setAttribute('vector-effect','non-scaling-stroke');
      g.onmouseenter=()=>{if(isBusy())return;if(cards.has(p.id)){openCard(p.id);return;}focus(p.id,true);$('#map-tooltip').replaceChildren(node('strong','',p.title),node('small','',p.artist));$('#map-tooltip').hidden=false;};
      g.onmouseleave=e=>{focus(p.id,false);$('#map-tooltip').hidden=true;const card=cards.get(p.id);if(card&&!card.contains(e.relatedTarget))closeCard();};
      g.onclick=()=>{if(!moved&&!isBusy())previewPoint(p,true);};
    }
    const densityKey=r.signature+':'+r.method;
    let fields=densityCache.get(densityKey);
    if(!fields){fields=estimateDensities(r.plot.filter(p=>p.kind==='background'),r.clusters||[]);for(const field of fields)field.image=densityImage(field,palette[field.id%palette.length]);densityCache.set(densityKey,fields);if(densityCache.size>12)densityCache.delete(densityCache.keys().next().value);}
    const existingLayer=densityLayers.get(densityKey);
    if(!existingLayer){
      for(const[key,layer]of densityLayers){layer.classList.add('density-out');clearTimeout(layer.removal);layer.removal=setTimeout(()=>{layer.remove();densityLayers.delete(key);},1200);}
      const layer=svg('g',{class:'density-layer','data-density-method':'gaussian-kde'});
      for(const field of fields){const image=svg('image',{href:field.image,'data-cluster':field.id,preserveAspectRatio:'none'});image.field=field;layer.append(image);}
      $('#cluster-clouds').append(layer);densityLayers.set(densityKey,layer);requestAnimationFrame(()=>layer.classList.add('density-in'));
    }
    if(existingLayer){clearTimeout(existingLayer.removal);existingLayer.classList.remove('density-out');existingLayer.classList.add('density-in');}
    for(const layer of densityLayers.values())for(const image of layer.children){const b=image.field.bounds,topLeft=anchor({x:b.left,y:b.top}),bottomRight=anchor({x:b.right,y:b.bottom});image.setAttribute('x',topLeft.x);image.setAttribute('y',topLeft.y);image.setAttribute('width',bottomRight.x-topLeft.x);image.setAttribute('height',bottomRight.y-topLeft.y);image.style.opacity=activeGroup!==null&&Number(image.dataset.cluster)!==activeGroup ? .1 : 1;}
    $('#cluster-labels').replaceChildren();
    $('#cluster-clouds').style.display=showDensity?'':'none';
    for(const[id,c]of cards)if(!selectedIds.has(id)){if(openId===id)closeCard(true);c.remove();cards.delete(id);}
    for(const{track,kind}of selected)ensureCard(track,kind,byId.get(track.id));
    $('#leaders').replaceChildren();
    $('#cluster-legend').replaceChildren(...(r.clusters||[]).map(group=>{const b=node('button','cluster-chip'+(activeGroup===group.id?' active':''));b.style.setProperty('--cluster',palette[group.id%palette.length]);b.setAttribute('aria-pressed',String(activeGroup===group.id));b.append(node('i'),node('span','',group.label),node('small','',String(group.count)));b.onclick=()=>{activeGroup=activeGroup===group.id?null:group.id;draw();};return b;}));
    $('#map-empty').hidden=true;$('#map-title').textContent=r.method==='base'?'Full embedding space':`${r.category.name} space`;
    $('#map-explanation').textContent=`${r.clusters?.length||0} colour groups · ★ starting song · click / hover to play · zoom in for cards`;
    camera();animate();
  }
  function render(r,options={}){const changed=!result||result.category.id!==r.category.id||result.method!==r.method||result.info.k!==r.info.k||result.basis_method!==r.basis_method;closeCard(true);metadataEpoch++;result=r;trackIndex.clear();for(const t of[r.seed,...r.results,...r.baseline,...(r.representatives||[])])trackIndex.set(t.id,t);compare=!!options.compare;if(changed)activeGroup=null;draw();}
  new ResizeObserver(()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(draw);}).observe(stage);
  stage.addEventListener('wheel',e=>{if(e.target.closest('.spatial-content'))return;e.preventDefault();closeCard(true);zoom=clamp(zoom*(e.deltaY<0?1.12:.89),.55,6);camera();},{passive:false});
  stage.addEventListener('pointerdown',e=>{if(e.target.closest('button,.spatial-content'))return;closeCard(true);drag={x:e.clientX,y:e.clientY,px:pan.x,py:pan.y};moved=false;world.classList.add('dragging');});
  window.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>5)moved=true;pan={x:drag.px+dx,y:drag.py+dy};camera();});
  window.addEventListener('pointerup',()=>{drag=null;world.classList.remove('dragging');});
  stage.addEventListener('keydown',e=>{if(e.key==='Escape'){closeCard(true);document.activeElement?.blur();}});
  stage.addEventListener('pointerdown',e=>{if(!e.target.closest('.map-card'))closeCard(true);});
  $('#reset-map').onclick=()=>{closeCard(true);fit();};$('#zoom-in').onclick=()=>{closeCard(true);zoom=clamp(zoom*1.2,.55,6);camera();};$('#zoom-out').onclick=()=>{closeCard(true);zoom=clamp(zoom/1.2,.55,6);camera();};
  $('#groups-toggle').onclick=()=>{showDensity=!showDensity;$('#groups-toggle').setAttribute('aria-pressed',String(showDensity));$('#cluster-clouds').style.display=showDensity?'':'none';};
  return{render,focus,fit,setStudy(study,hidden){closeCard(true);$('#study-map-cover').hidden=!(study&&hidden);stage.classList.toggle('blind',study&&hidden);},clear(){closeCard(true);cancelAnimationFrame(frame);cancelAnimationFrame(cameraFrame);clearTimeout(tilesTimer);result=null;points.clear();cards.clear();trackIndex.clear();shownTiles.clear();motions.clear();densityLayers.clear();tiles.clear();$('#zoom-tiles')?.remove();for(const id of ['points','map-cards','leaders','cluster-clouds','cluster-labels','cluster-legend'])$('#'+id).replaceChildren();}};
}
