// Static envelopes are real clip summaries. Only one live SVG consumes the
// transport's audio samples; a signal frame never redraws the music cloud.
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const ns='http://www.w3.org/2000/svg';
const STAR='M0-9 2.65-3.1 9.5-2.9 4.1 1.45 5.7 8 0 4.35-5.7 8-4.1 1.45-9.5-2.9-2.65-3.1Z';
function element(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text)e.textContent=text;return e;}
function icon(kind){
  const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','-12 -12 24 24');svg.setAttribute('aria-hidden','true');
  if(kind==='star'){const p=document.createElementNS(ns,'path');p.setAttribute('d',STAR);svg.append(p);}
  else{const c=document.createElementNS(ns,'circle');c.setAttribute('r',8);const p=document.createElementNS(ns,'path');p.setAttribute('d','M0-1v6M0-5v.1');svg.append(c,p);}
  return svg;
}

export function createWaveformStore({base,indexFor,count}){
  let manifest;
  const shards=new Map();
  async function shard(i,m){
    if(!shards.has(i)){
      const promise=fetch(new URL(`${i}.bin?v=${m.version}`,base)).then(async r=>{
        if(!r.ok)throw Error('Waveform unavailable');const bytes=new Uint8Array(await r.arrayBuffer());
        if(bytes.length!==Math.min(m.shard_size,m.count-i*m.shard_size)*m.bins)throw Error('Invalid waveform shard');return bytes;
      }).catch(e=>{shards.delete(i);throw e;});shards.set(i,promise);
    }
    return shards.get(i);
  }
  return async ids=>{
    manifest||=fetch(new URL('manifest.json',base),{cache:'no-cache'}).then(r=>{if(!r.ok)throw Error('Waveforms unavailable');return r.json();}).catch(e=>{manifest=null;throw e;});
    const m=await manifest;if(m.format!==1||m.bins!==64||m.count!==count())throw Error('Waveforms do not match this catalog');
    const groups=new Map(),output=new Map();
    for(const id of ids){const i=indexFor(id);if(!Number.isInteger(i))continue;const s=Math.floor(i/m.shard_size);if(!groups.has(s))groups.set(s,[]);groups.get(s).push([id,i]);}
    const queue=[...groups];
    // Bound the network fan-out when a new viewport crosses many catalog shards.
    await Promise.all(Array.from({length:Math.min(6,queue.length)},async()=>{while(queue.length){const[s,rows]=queue.shift(),bytes=await shard(s,m);for(const[id,i]of rows){const offset=i%m.shard_size*m.bins;output.set(id,bytes.subarray(offset,offset+m.bins));}}}));
    return output;
  };
}

export function createWaveformLayer({stage,points,loadPeaks,play,inspect,chooseSeed,trackFor,schedule}){
  const layer=element('div','music-waveforms');layer.id='music-waveforms';layer.dataset.loading='0';stage.append(layer);
  const glyphs=new Map(),peaks=new Map(),pending=new Set(),failed=new Map();
  let selected=null,playing=null,hovered=null,requestTimer=0,selectionTime=-Infinity,frameTime=0,liveLevels=null,reveal=0,revealTime=0,hitGrid=new Map();
  const queue=new Set();
  function request(ids){
    for(const id of ids)if(!peaks.has(id)&&!pending.has(id)&&(failed.get(id)||0)<performance.now()){pending.add(id);queue.add(id);}
    layer.dataset.loading=pending.size;
    if(!queue.size||requestTimer)return;
    requestTimer=setTimeout(async()=>{
      requestTimer=0;const ids=[...queue];queue.clear();
      try{const values=await loadPeaks(ids);for(const[id,data]of values)if(data.length===64)peaks.set(id,data);}
      catch{for(const id of ids)failed.set(id,performance.now()+15000);}
      finally{for(const id of ids)pending.delete(id);layer.dataset.loading=pending.size;selectionTime=-Infinity;schedule();}
    },80);
  }
  function select(id){selected=id;selectionTime=-Infinity;schedule();}
  function activate(id){const p=points.get(id);if(!p)return;select(id);play(trackFor(p));}
  function showInfo(id){select(id);inspect(id);}
  function mount(p,g){
    const root=element('div','music-wave');root.dataset.track=p.id;
    const button=element('button','wave-play');button.type='button';button.title=`Play / pause ${p.title}`;
    // Pointer gestures are resolved by the map, so a drag/pinch cannot play.
    button.onclick=e=>{e.stopPropagation();if(e.detail===0)activate(p.id);};
    const live=document.createElementNS(ns,'svg');live.classList.add('wave-live');live.setAttribute('viewBox','0 0 100 32');live.setAttribute('aria-hidden','true');
    const bars=[];button.append(live);
    const badge=element('span','wave-seed');badge.append(icon('star'));badge.title='Starting song';button.append(badge);
    const label=element('div','wave-label');label.append(element('strong',null,p.title),element('span',null,p.artist));
    const actions=element('div','wave-actions'),info=element('button','wave-action wave-info'),star=element('button','wave-action wave-star');
    info.type=star.type='button';info.setAttribute('aria-label',`Show details for ${p.title}`);info.title='Track details';info.append(icon('info'));
    star.setAttribute('aria-label',`Use ${p.title} as starting song`);star.title='Use as starting song';star.append(icon('star'));
    info.onclick=e=>{e.stopPropagation();showInfo(p.id);};star.onclick=e=>{e.stopPropagation();const current=points.get(p.id);if(current&&current.kind!=='seed')chooseSeed(trackFor(current));};
    actions.append(info,star);root.append(button,label,actions);layer.append(root);
    root.onpointerenter=e=>{if(e.pointerType==='mouse')hover(p.id);};root.onpointerleave=e=>{if(e.pointerType==='mouse')hover(null);};
    return Object.assign(g,{root,button,badge,live,bars,star});
  }
  function hover(id){if(hovered!==id){hovered=id;selectionTime=-Infinity;schedule();}}
  function liveShape(g,levels){
    if(!g.root)mount(points.get(g.id),g);
    if(!g.bars.length)for(let i=0;i<25;i++){const line=document.createElementNS(ns,'line');line.setAttribute('x1',2+i*4);line.setAttribute('x2',2+i*4);g.live.append(line);g.bars.push(line);}
    for(let i=0;i<25;i++){const h=3+levels[i]*24;g.bars[i].setAttribute('y1',((32-h)/2).toFixed(2));g.bars[i].setAttribute('y2',((32+h)/2).toFixed(2));}
  }
  document.addEventListener('music-lenses-playback',({detail})=>{
    const previous=playing;playing=detail.playing?detail.id:null;if(detail.playing&&detail.id!==previous)select(detail.id);
    liveLevels=null;schedule();
  });
  document.addEventListener('music-lenses-signal',({detail})=>{
    if(detail.id!==playing)return;const first=!liveLevels;liveLevels=detail.levels;
    const g=glyphs.get(playing);if(g&&g.reveal>.01){liveShape(g,liveLevels);g.root.dataset.live='true';}
    if(first)schedule(); // One handoff repaint removes the frozen shape under the live SVG.
  });
  function update(now,{zoom,width,height,transitioning}){
    frameTime=now;
    const targetReveal=clamp((zoom-1.65)/1.55,0,1),dt=revealTime?Math.min(50,now-revealTime):16;revealTime=now;
    reveal+=(targetReveal-reveal)*(1-Math.exp(-dt/65));
    if(Math.abs(targetReveal-reveal)<.002)reveal=targetReveal;else schedule();
    const waveWidth=width<700?72:88;
    layer.dataset.reveal=reveal.toFixed(3);layer.dataset.selected=selected||'';
    if(!reveal&&!glyphs.size){layer.dataset.count='0';return;}
    for(const p of points.values())p.waveReveal=0;
    if(now-selectionTime>=100){
      selectionTime=now;
      const occupied=[],chosen=new Set(),labels=new Set(),candidates=[...points.values()].filter(p=>p.screen&&p.screen.x>=0&&p.screen.x<=width&&p.screen.y>=0&&p.screen.y<=height);
      layer.dataset.viewportCount=candidates.length;
      const priority=p=>p.id===selected?0:p.id===playing?1:p.id===hovered?2:p.kind==='seed'?3:p.kind==='result'?4:glyphs.get(p.id)?.root?5:6;
      candidates.sort((a,b)=>priority(a)-priority(b)||((a.screen.x-width/2)**2+(a.screen.y-height/2)**2)-((b.screen.x-width/2)**2+(b.screen.y-height/2)**2));
      const rect=stage.getBoundingClientRect();
      for(const e of document.querySelectorAll('.floating-controls,.top-left-tools,.top-right-tools,.zoom-pill,#player:not([hidden]),#cluster-labels[data-visible=true] button')){const r=e.getBoundingClientRect();occupied.push({x:r.x-rect.x-10,y:r.y-rect.y-8,w:r.width+20,h:r.height+16});}
      // Every on-frame track gets a frozen waveform. Only labels and DOM controls
      // are collision-limited; the complete waveform layer stays in the canvas.
      if(reveal>0)for(const p of candidates){
        chosen.add(p.id);
        if(labels.size>=Math.min(150,Math.floor(width*height/4800))&&p.id!==hovered&&p.id!==selected&&p.id!==playing)continue;
        const special=p.id===selected||p.id===playing,w=Math.max(waveWidth+(special?24:0),reveal>.45?(width<700?116:140):0);
        const below=p.screen.y<(width<1000?210:150);
        const r={x:p.screen.x-w/2-7,y:p.screen.y-(special&&!below?75:25),w:w+14,h:reveal>.45?(special?140:78):46};
        if(!special&&p.id!==hovered&&(r.x<5||r.x+r.w>width-5||r.y<5||r.y+r.h>height-7||occupied.some(b=>r.x<b.x+b.w&&r.x+r.w>b.x&&r.y<b.y+b.h&&r.y+r.h>b.y)))continue;
        labels.add(p.id);occupied.push(r);
      }
      if(transitioning)for(const id of [selected,playing])if(points.has(id)){chosen.add(id);labels.add(id);}
      request([...chosen]);
      for(const[id,g]of glyphs){const target=chosen.has(id)&&peaks.has(id)?1:0;if(g.target!==target){g.from=g.reveal;g.target=target;g.start=now;}}
      for(const id of chosen)if(peaks.has(id)&&!glyphs.has(id)){
        const g={id,root:null,bars:[],reveal:0,from:0,target:1,start:frameTime,base:new Float32Array(25),width:80},values=peaks.get(id);for(let i=0;i<25;i++){const lo=Math.floor(i*64/25),hi=Math.floor((i+1)*64/25);let sum=0;for(let j=lo;j<hi;j++)sum+=values[j];g.base[i]=sum/(hi-lo)/255;}glyphs.set(id,g);
      }
      for(const[id,g]of glyphs){if(labels.has(id)&&!g.root)mount(points.get(id),g);else if(!labels.has(id)&&g.root&&!g.root.contains(document.activeElement)){g.root.remove();g.root=null;g.bars=[];}}
    }
    let moving=false;hitGrid=new Map();
    const nearby=new Map();for(const p of points.values()){if(!p.screen)continue;const key=Math.floor(p.screen.x/64)+','+Math.floor(p.screen.y/64);if(!nearby.has(key))nearby.set(key,[]);nearby.get(key).push(p);}
    for(const[id,g]of glyphs){
      const p=points.get(id);if(!p){g.root?.remove();glyphs.delete(id);continue;}
      const t=clamp((now-g.start)/220,0,1);g.reveal=g.from+(g.target-g.from)*(t*t*(3-2*t));if(t<1)moving=true;
      if(g.target===0&&t===1){g.root?.remove();glyphs.delete(id);continue;}
      const amount=g.reveal*reveal;p.waveReveal=amount;g.amount=amount;
      const active=id===playing,expanded=id===selected||active,scale=expanded?1.12:1;
      let spacing=waveWidth/0.72;const cx=Math.floor(p.screen.x/64),cy=Math.floor(p.screen.y/64);
      for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(const q of nearby.get((cx+x)+','+(cy+y))||[])if(q!==p)spacing=Math.min(spacing,Math.hypot(q.screen.x-p.screen.x,q.screen.y-p.screen.y));
      g.width=expanded||id===hovered?waveWidth*scale:clamp(spacing*.72,14,waveWidth);g.height=expanded||id===hovered?28*scale:Math.max(7,28*g.width/waveWidth);
      if(p.kind==='result'&&!expanded&&id!==hovered){g.width=Math.min(waveWidth,g.width*1.25);g.height=Math.max(7,28*g.width/waveWidth);}
      const c=p.current;g.color=p.kind==='seed'?'#f59e0b':`rgb(${Math.round(c.r)},${Math.round(c.g)},${Math.round(c.b)})`;
      if(amount>.35)for(let x=Math.floor((p.screen.x-g.width/2)/32);x<=Math.floor((p.screen.x+g.width/2)/32);x++)for(let y=Math.floor((p.screen.y-15)/32);y<=Math.floor((p.screen.y+15)/32);y++){const key=x+','+y;if(!hitGrid.has(key))hitGrid.set(key,[]);hitGrid.get(key).push(g);}
      if(!g.root)continue;
      g.root.style.transform=`translate(${p.screen.x}px,${p.screen.y}px)`;g.root.style.width=g.width+'px';g.root.style.setProperty('--wave-color',g.color);g.root.style.opacity=amount;
      g.root.style.setProperty('--wave-label-opacity',clamp((reveal-.35)/.5,0,1));g.root.style.setProperty('--wave-scale',scale);g.root.style.setProperty('--wave-morph',amount);g.root.style.setProperty('--wave-height',g.height+'px');
      g.root.style.setProperty('--wave-actions-top',p.screen.y<(width<1000?210:150)?'59px':width<700?'-65px':'-54px');
      g.root.classList.toggle('is-selected',id===selected);g.root.classList.toggle('is-playing',active);g.root.classList.toggle('is-seed',p.kind==='seed');
      g.root.dataset.live=String(active&&!!liveLevels);g.root.dataset.kind=p.kind;
      if(!active&&g.bars.length){g.live.replaceChildren();g.bars.length=0;}
      g.root.style.pointerEvents=amount>.35?'':'none';g.root.inert=amount<=.35;
      g.button.setAttribute('aria-label',`${active?'Pause':'Play'} ${p.title} — ${p.artist}`);g.button.setAttribute('aria-pressed',String(active));
      g.star.disabled=p.kind==='seed';g.star.setAttribute('aria-label',p.kind==='seed'?'Current starting song':`Use ${p.title} as starting song`);
      if(active&&liveLevels)liveShape(g,liveLevels);
    }
    layer.dataset.count=glyphs.size;layer.dataset.controls=layer.children.length;if(moving)schedule();
  }
  function draw(ctx){
    ctx.lineCap='round';
    let rendered=0;for(const[id,g]of glyphs){
      const p=points.get(id),a=g.amount;if(!p?.screen||a<=0)continue;
      if(p.screen.x>=0&&p.screen.x<=ctx.canvas.clientWidth&&p.screen.y>=0&&p.screen.y<=ctx.canvas.clientHeight)rendered++;
      // The live SVG replaces only this shape, without repainting any other track.
      ctx.globalAlpha=(id===playing&&liveLevels?0:1)*a*(p.kind==='background'?.8:1);ctx.strokeStyle=g.color;
      const w=(p.radius*2+(g.width-p.radius*2)*a),h=g.height*a;
      ctx.lineWidth=Math.max(1.25,w/42);ctx.beginPath();
      const bars=clamp(Math.floor(w/3),5,25);for(let i=0;i<bars;i++){const x=p.screen.x-w/2+(i+.5)*w/bars,v=2+g.base[Math.floor(i*25/bars)]*(h-2);ctx.moveTo(x,p.screen.y-v/2);ctx.lineTo(x,p.screen.y+v/2);}ctx.stroke();
      if(p.kind==='seed'){ctx.save();ctx.globalAlpha=a;ctx.translate(p.screen.x-w/2-8,p.screen.y);ctx.scale(.65,.65);ctx.fillStyle='#f59e0b';ctx.strokeStyle=getComputedStyle(stage).backgroundColor;ctx.lineWidth=1.6;const star=new Path2D(STAR);ctx.fill(star);ctx.stroke(star);ctx.restore();}
    }
    layer.dataset.renderedCount=rendered;
    ctx.globalAlpha=1;ctx.lineCap='butt';
  }
  return{update,draw,activate,inspect:showInfo,select,hover,hit(x,y){let best=null,d=Infinity;for(const g of hitGrid.get(Math.floor(x/32)+','+Math.floor(y/32))||[]){const p=points.get(g.id),dx=x-p.screen.x,dy=y-p.screen.y,dist=dx*dx+dy*dy;if(Math.abs(dx)<=Math.max(8,g.width/2)&&Math.abs(dy)<=15&&dist<d){best=p;d=dist;}}return best;},isVisible:id=>(glyphs.get(id)?.amount||0)>.35,
    get reveal(){return reveal*Math.min(1,glyphs.size/Math.max(1,Number(layer.dataset.viewportCount)||1));},
    isHovered:id=>!!glyphs.get(id)?.root?.matches(':hover'),
    clear(){for(const g of glyphs.values())g.root?.remove();glyphs.clear();selected=null;selectionTime=-Infinity;layer.dataset.count='0';},
  };
}
