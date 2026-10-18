// A few evidence-backed descriptors at KDE peaks, never hard class labels.
export function createClusterLabels({stage,palette}){
  const layer=document.createElement('div');layer.id='cluster-labels';layer.className='cluster-themes';stage.append(layer);
  const details=document.createElement('div');details.id='cluster-evidence';details.className='cluster-evidence';details.popover='auto';document.body.append(details);
  let key=null,labels=[],evidenceAnchor=null;
  const directory=document.createElement('details');directory.id='profile-directory';directory.className='profile-directory';document.querySelector('#settings-popover')?.append(directory);
  const percent=value=>Math.round(value*100)+'%';
  const element=(tag,text,cls)=>{const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e;};
  function close(){if(details.matches(':popover-open'))details.hidePopover();}
  function placeEvidence(){
    if(!evidenceAnchor?.isConnected||!details.matches(':popover-open'))return;
    const r=evidenceAnchor.getBoundingClientRect(),width=Math.min(300,innerWidth-28);details.style.width=width+'px';
    details.style.left=Math.max(14,Math.min(innerWidth-width-14,r.left))+'px';
    const h=details.getBoundingClientRect().height;details.style.top=Math.max(14,Math.min(innerHeight-h-14,r.bottom+12))+'px';
  }
  new ResizeObserver(placeEvidence).observe(details);window.addEventListener('resize',placeEvidence);
  function explain(button,group){
    const t=group.theme,heading=element('div',null,'cluster-evidence-heading'),dismiss=element('button','×');dismiss.type='button';dismiss.setAttribute('aria-label','Close cluster evidence');dismiss.onclick=close;
    heading.append(element('strong',t.label),dismiss);
    if(t.kind==='profile'){
      const source=t.source==='caption-profile'?'caption-defined':'metadata-defined';
      details.replaceChildren(heading,element('p',`${t.count.toLocaleString()} songs in this ${source} group.`));
      if(t.source==='caption-profile')details.append(element('p','Every assigned caption explicitly mentions all the listed attributes. Additional instruments or attributes may also be present. These are caption profiles, not measured audio classes.', 'evidence-muted'));
      else details.append(element('p',`Metadata rule: ${t.rule}.`, 'evidence-muted'));
      details.append(element('p',`${t.selection.count.toLocaleString()} songs in the discovery half; ${t.validation.count.toLocaleString()} in the confirmation half.`, 'evidence-muted'));
    }else{
      const source=t.source==='metadata'?'dataset metadata':'generated captions';
      details.replaceChildren(heading,element('p',`${t.matches.toLocaleString()} of ${t.count.toLocaleString()} songs (${percent(t.support)}) match this theme in ${source}.`),element('p',`${percent(t.baseline)} across the collection · ${t.lift.toFixed(1)}× more common here.`, 'evidence-muted'));
      if(t.source==='captions')details.append(element('p','This is a recurring description in generated captions.', 'evidence-muted'));
      else details.append(element('p',`Metadata field: ${t.rule}.`, 'evidence-muted'));
    }
    const tempo=group.statistics?.tempo;
    if(tempo)details.append(element('p',`Median tempo ${Math.round(tempo.median)} BPM; middle half ${Math.round(tempo.q25)}–${Math.round(tempo.q75)} BPM.`, 'evidence-muted'));
    const counts=group.statistics?.speed_counts;
    if(counts)details.append(element('p',`Speed labels: ${Object.entries(counts).map(([name,n])=>name+' '+percent(n/group.count)).join(' · ')}.`, 'evidence-muted'));
    if((t.source==='captions'||t.source==='caption-profile')&&t.examples?.length){
      const examples=element('details',null,'evidence-examples');examples.append(element('summary','See matching caption phrases'));
      for(const example of t.examples.slice(0,2)){
        const quote=element('blockquote'),phrases=example.phrases||[example.phrase],text=example.excerpt,ranges=[];
        for(const phrase of phrases.filter(Boolean)){const at=text.indexOf(phrase);if(at>=0)ranges.push({at,end:at+phrase.length});}
        ranges.sort((a,b)=>a.at-b.at);let cursor=0;
        for(const range of ranges){if(range.at<cursor)continue;quote.append(text.slice(cursor,range.at),element('mark',text.slice(range.at,range.end)));cursor=range.end;}
        quote.append(text.slice(cursor));examples.append(quote);
      }details.append(examples);
    }
    evidenceAnchor=button;details.showPopover();placeEvidence();
  }
  function clear(){close();key=null;labels=[];layer.replaceChildren();layer.hidden=true;}
  function update({signature,groups,grouping,modes,visible,screen,width,height}){
    if(key!==signature){
      clear();key=signature;directory.replaceChildren();
      const named=(groups||[]).filter(g=>g.theme),summary=element('summary',`${named.length} groups${grouping?` · ${percent(grouping.coverage)} assigned`:''}`);directory.append(summary);
      for(const group of groups||[]){const row=element(group.theme?'button':'div',null,'profile-row');if(group.theme){row.type='button';row.onclick=()=>explain(row,group);}row.style.setProperty('--cluster',palette[group.id%palette.length]);row.append(element('i'),element('span',group.label||group.theme?.label),element('small',group.count.toLocaleString()));directory.append(row);}
    }
    if(!visible||!modes?.length){layer.dataset.visible='false';layer.style.opacity='0';layer.style.pointerEvents='none';close();return;}
    if(!labels.length){
      for(const group of groups||[]){const mode=modes.find(m=>m.id===group.id);if(!group.theme||!mode)continue;
        const button=element('button',null,'cluster-theme');button.type='button';button.dataset.group=group.id;button.style.setProperty('--cluster',palette[group.id%palette.length]);
        button.append(element('span',group.theme.label),element('small',group.theme.kind==='profile'?`${group.count.toLocaleString()} songs`:`${percent(group.theme.support)} of ${group.theme.source==='captions'?'captions':'metadata'}`));
        button.setAttribute('aria-label',group.theme.kind==='profile'?`${group.theme.label}. ${group.count.toLocaleString()} songs. Show evidence`:`${group.theme.label}. ${percent(group.theme.support)} support. Show evidence`);button.onclick=()=>explain(button,group);layer.append(button);labels.push({button,modes:mode.peaks?.length?mode.peaks:[mode.mode]});
      }
    }
    layer.hidden=false;layer.dataset.visible='true';layer.style.opacity='1';layer.style.pointerEvents='';
    const occupied=[...document.querySelectorAll('.floating-controls,.top-left-tools,.top-right-tools,.canvas-context,.zoom-pill,#player:not([hidden])')].map(e=>e.getBoundingClientRect());
    for(const{button,modes}of labels){
      const w=button.offsetWidth,h=button.offsetHeight;let placement=null;
      for(const mode of modes){const p=screen(mode);
        for(const dy of [0,-h*.7,h*.7]){const x=p.x-w/2,y=p.y-h/2+dy;
          const blocked=x<12||y<12||x+w>width-12||y+h>height-12||occupied.some(r=>x<r.right+5&&x+w>r.left-5&&y<r.bottom+5&&y+h>r.top-5);
          if(!blocked){placement={x,y};break;}
        }if(placement)break;
      }
      button.style.visibility=placement?'':'hidden';
      if(placement){const{x,y}=placement;button.style.transform=`translate(${x}px,${y}px)`;occupied.push({left:x,top:y,right:x+w,bottom:y+h});}
    }
  }
  return{update,clear};
}
