// Screen-space text only: no artwork, blur surfaces, or thousands of card nodes.
export function createTrackLabels({stage,points,openCard,closeCard,schedule}){
  const layer=document.createElement('div');layer.id='track-labels';layer.className='track-labels';layer.hidden=true;stage.append(layer);
  const labels=new Map();let lastSelection=-Infinity,timer=0;
  const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
  function clear(){clearTimeout(timer);timer=0;labels.clear();layer.replaceChildren();layer.hidden=true;lastSelection=-Infinity;}
  function update(now,{zoom,radius,width,height,transitioning}){
    const reveal=clamp((zoom-1.9)/.9,0,1);layer.hidden=reveal===0;layer.style.opacity=reveal;layer.style.pointerEvents=reveal>.35?'':'none';
    if(!reveal){if(labels.size)clear();return;}
    for(const[id,label]of labels)if(!points.has(id)){label.element.remove();labels.delete(id);}
    const select=!transitioning||!labels.size;
    if(select&&now-lastSelection>=90){
      clearTimeout(timer);timer=0;lastSelection=now;
      const buckets=new Map(),cell=64,chosen=new Map(),mobile=width<700,labelWidth=mobile?122:142,labelHeight=29;
      function cells(r,visit){for(let x=Math.floor(r.x/cell);x<=Math.floor((r.x+r.w)/cell);x++)for(let y=Math.floor(r.y/cell);y<=Math.floor((r.y+r.h)/cell);y++)visit(x+','+y);}
      function add(r){cells(r,key=>{if(!buckets.has(key))buckets.set(key,[]);buckets.get(key).push(r);});}
      function collides(r){let hit=false;cells(r,key=>{for(const b of buckets.get(key)||[])if(r.x<b.x+b.w&&r.x+r.w>b.x&&r.y<b.y+b.h&&r.y+r.h>b.y){hit=true;break;}});return hit;}
      const stageRect=stage.getBoundingClientRect();
      for(const element of document.querySelectorAll('.floating-controls,.top-left-tools,.top-right-tools,.canvas-context,.zoom-pill,#cluster-labels[data-visible=true] button,#player:not([hidden])')){const r=element.getBoundingClientRect();add({x:r.x-stageRect.x-10,y:r.y-stageRect.y-8,w:r.width+20,h:r.height+16});}
      const oldOrder=new Map([...labels.keys()].map((id,i)=>[id,i]));
      const priority=p=>p.kind==='seed'?0:p.kind==='result'?1:labels.has(p.id)?2:3;
      const candidates=[...points.values()].filter(p=>p.screen&&p.screen.x>12&&p.screen.x<width-12&&p.screen.y>12&&p.screen.y<height-12);
      candidates.sort((a,b)=>priority(a)-priority(b)||(labels.has(a.id)&&labels.has(b.id)?oldOrder.get(a.id)-oldOrder.get(b.id):0)||((a.screen.x-width/2)**2+(a.screen.y-height/2)**2)-((b.screen.x-width/2)**2+(b.screen.y-height/2)**2));
      const budget=Math.min(160,Math.floor(width*height/6000));
      for(const p of candidates){
        if(chosen.size>=budget)break;
        const gap=radius+7,offsets=[[gap,-14,'right'],[-labelWidth-gap,-14,'left'],[-labelWidth/2,gap,'below'],[-labelWidth/2,-labelHeight-gap,'above']];
        const preferred=labels.get(p.id)?.side;if(preferred)offsets.sort((a,b)=>(a[2]===preferred?-1:0)-(b[2]===preferred?-1:0));
        for(const[dx,dy,side]of offsets){const r={x:p.screen.x+dx-4,y:p.screen.y+dy-3,w:labelWidth+8,h:labelHeight+6};if(r.x<10||r.y<10||r.x+r.w>width-10||r.y+r.h>height-12||collides(r))continue;chosen.set(p.id,{dx,dy,side,width:labelWidth});add(r);break;}
      }
      for(const[id,label]of labels)if(!chosen.has(id)){label.element.remove();labels.delete(id);}
      for(const[id,pose]of chosen){
        let label=labels.get(id);const p=points.get(id);
        if(!label){const element=document.createElement('button');element.type='button';element.className='track-label';element.dataset.track=id;element.tabIndex=-1;
          const title=document.createElement('strong'),artist=document.createElement('span');title.textContent=p.title;artist.textContent=p.artist;element.append(title,artist);element.setAttribute('aria-label',`${p.title} — ${p.artist}. Show preview`);
          element.onmouseenter=()=>openCard(id);element.onmouseleave=e=>{if(!e.relatedTarget?.closest('.map-card'))closeCard();};element.onclick=e=>{e.stopPropagation();openCard(id);};element.onfocus=()=>openCard(id);element.onblur=()=>closeCard();layer.append(element);label={element};labels.set(id,label);
        }
        Object.assign(label,pose);label.element.dataset.side=pose.side;label.element.style.width=pose.width+'px';
      }
    }else if(select&&!timer){timer=setTimeout(()=>{timer=0;lastSelection=-Infinity;schedule();},95);}
    for(const[id,label]of labels){const p=points.get(id);if(p?.screen)label.element.style.transform=`translate(${p.screen.x+label.dx}px,${p.screen.y+label.dy}px)`;}
    layer.dataset.count=labels.size;
  }
  return{update,clear,isHovered:id=>!!labels.get(id)?.element.matches(':hover')};
}
