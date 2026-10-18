// Select-only comboboxes. Native selects remain the state source for both apps.
const icons={
  instrumentation:'M4 7h16M7 4v16m5-16v16m5-16v16M4 17h16',
  vocals:'M9 5a3 3 0 0 1 6 0v6a3 3 0 0 1-6 0Zm-3 5v1a6 6 0 0 0 12 0v-1m-6 7v4m-3 0h6',
  mood:'M4 13c3-9 5 9 8 0s5 9 8-2',groove:'m4 15 4-6 4 9 4-12 4 6',
  language:'M3 5h12M9 3v2m-4 3c1 4 4 7 8 9M13 5c-1 6-5 10-10 12m11 4 4-12 4 12m-6-4h4',
  speed:'M4 16a8 8 0 0 1 16 0m-8-1 5-7M5 19h14',
  energy:'m13 3-8 11h6l-1 7 9-12h-6z',production:'M5 4v16M12 4v16M19 4v16M3 9h4m3 7h4m3-9h4',
  __full:'M4 4h6v6H4zm10 0h6v6h-6zM4 14h6v6H4zm10 0h6v6h-6z',
  text:'M5 5h14M12 5v14m-4 0h8',sparse:'m12 3 9 5-9 5-9-5zm-9 9 9 5 9-5M3 16l9 5 9-5'
};
const descriptions={instrumentation:'Instruments & arrangement',vocals:'Voices & delivery',language:'Lyrics & spoken languages',mood:'Emotion & atmosphere',groove:'Rhythm & movement',speed:'Slow, medium & fast tempo',energy:'Intensity & dynamics',production:'Recording & texture',__full:'All 512 audio dimensions',text:'Direct prompt embeddings',sparse:'Adam-inverted concept embeddings'};
const svg=path=>{const s=document.createElementNS('http://www.w3.org/2000/svg','svg');s.setAttribute('viewBox','0 0 24 24');s.setAttribute('aria-hidden','true');const p=document.createElementNS(s.namespaceURI,'path');p.setAttribute('d',path);s.append(p);return s;};
export function stylizeControls(){
  const controls=['lens-select','basis-select'].map(id=>{
    const select=document.getElementById(id),field=select.closest('.control-field'),label=field.querySelector('label');
    select.classList.add('native-select');select.tabIndex=-1;select.setAttribute('aria-hidden','true');
    if(id==='basis-select'){
      label.classList.add('sr-only');label.removeAttribute('for');
      const group=document.createElement('div');group.className='basis-segments';group.setAttribute('role','group');group.setAttribute('aria-label','Basis construction method');field.append(group);
      function sync(){for(const option of select.options){let button=[...group.children].find(b=>b.dataset.basis===option.value);if(!button){button=document.createElement('button');button.type='button';button.dataset.basis=option.value;button.textContent=option.value==='text'?'Text':'Sparse';button.setAttribute('aria-label',option.value==='text'?'Use text prompt basis':'Use Adam sparse inversion basis');button.title=option.value==='text'?'Direct text embeddings':'Adam sparse inversion';button.onclick=()=>{select.value=option.value;sync();select.dispatchEvent(new Event('input',{bubbles:true}));select.dispatchEvent(new Event('change',{bubbles:true}));};group.append(button);}button.disabled=select.disabled||option.disabled;button.setAttribute('aria-pressed',String(option.value===select.value));}}
      select.addEventListener('change',sync);new MutationObserver(sync).observe(select,{childList:true,subtree:true,attributes:true,attributeFilter:['disabled','selected']});sync();return sync;
    }
    const trigger=document.createElement('button');trigger.className='select-trigger';trigger.id=id+'-trigger';trigger.type='button';trigger.setAttribute('role','combobox');trigger.setAttribute('aria-haspopup','listbox');trigger.setAttribute('aria-expanded','false');trigger.setAttribute('aria-label',select.getAttribute('aria-label')||label.textContent);
    const value=document.createElement('span');value.className='select-value';trigger.append(value,svg('m8 10 4 4 4-4'));field.append(trigger);label.htmlFor=trigger.id;
    const menu=document.createElement('div');menu.id=id+'-menu';menu.className='select-menu';menu.popover='auto';menu.setAttribute('role','listbox');menu.setAttribute('aria-label',label.textContent);document.body.append(menu);trigger.setAttribute('aria-controls',menu.id);
    let active=0,options=[],typeahead='',typeTimer;
    function position(){const r=trigger.getBoundingClientRect(),width=Math.min(id==='lens-select'?276:284,innerWidth-28);menu.style.width=width+'px';menu.style.left=Math.max(14,Math.min(r.left,innerWidth-width-14))+'px';menu.style.bottom=(innerHeight-r.top+12)+'px';menu.style.maxHeight=Math.max(120,Math.min(440,r.top-28))+'px';}
    function highlight(index){if(!options.length)return;active=(index+options.length)%options.length;for(let i=0;i<menu.children.length;i++)menu.children[i].classList.toggle('is-active',i===active);trigger.setAttribute('aria-activedescendant',menu.children[active].id);menu.children[active].scrollIntoView({block:'nearest'});}
    function choose(index){const option=options[index];if(!option||option.disabled)return;select.value=option.value;menu.hidePopover();sync();trigger.focus();select.dispatchEvent(new Event('input',{bubbles:true}));select.dispatchEvent(new Event('change',{bubbles:true}));}
    function sync(){
      value.textContent=select.selectedOptions[0]?.textContent||'Choose a lens';trigger.disabled=select.disabled||!select.options.length;
      options=[...select.options].filter(o=>!o.disabled);menu.replaceChildren(...options.map((option,i)=>{
        const row=document.createElement('div');row.className='select-option';row.id=id+'-option-'+i;row.setAttribute('role','option');row.setAttribute('aria-selected',String(option.value===select.value));row.dataset.value=option.value;
        const icon=document.createElement('span');icon.className='select-option-icon';icon.append(svg(icons[option.value]||icons.__full));
        const text=document.createElement('span');text.className='select-option-text';const title=document.createElement('strong');title.textContent=option.textContent;const note=document.createElement('small');note.textContent=descriptions[option.value]||'Custom musical axis';text.append(title,note);
        const check=svg('m6 12 4 4 8-8');check.classList.add('select-check');row.append(icon,text,check);row.onpointermove=()=>highlight(i);row.onclick=()=>choose(i);return row;
      }));
      if(menu.matches(':popover-open')){position();highlight(Math.max(0,options.findIndex(o=>o.value===select.value)));}
    }
    function open(){if(trigger.disabled)return;sync();position();menu.showPopover();highlight(Math.max(0,options.findIndex(o=>o.value===select.value)));}
    trigger.onclick=()=>menu.matches(':popover-open')?menu.hidePopover():open();
    menu.addEventListener('toggle',()=>{const open=menu.matches(':popover-open');trigger.setAttribute('aria-expanded',String(open));if(!open)trigger.removeAttribute('aria-activedescendant');});
    trigger.onkeydown=e=>{
      if(e.key==='Tab'){if(menu.matches(':popover-open'))menu.hidePopover();return;}
      if(e.key==='Escape'){if(menu.matches(':popover-open')){e.preventDefault();e.stopPropagation();menu.hidePopover();}return;}
      if(['ArrowDown','ArrowUp','Home','End','Enter',' '].includes(e.key)){
        e.preventDefault();e.stopPropagation();if(!menu.matches(':popover-open')){open();return;}
        if(e.key==='Enter'||e.key===' ')choose(active);else highlight(e.key==='Home'?0:e.key==='End'?options.length-1:active+(e.key==='ArrowDown'?1:-1));
      }else if(e.key.length===1&&!e.ctrlKey&&!e.metaKey&&!e.altKey){e.preventDefault();if(!menu.matches(':popover-open'))open();clearTimeout(typeTimer);typeahead+=e.key.toLowerCase();typeTimer=setTimeout(()=>{typeahead='';},650);const i=options.findIndex(o=>o.textContent.toLowerCase().startsWith(typeahead));if(i>=0)highlight(i);}
    };
    select.addEventListener('change',sync);new MutationObserver(sync).observe(select,{childList:true,subtree:true,attributes:true,attributeFilter:['disabled','label','selected']});
    window.addEventListener('resize',()=>{if(menu.matches(':popover-open'))position();});sync();return sync;
  });
  return()=>controls.forEach(sync=>sync());
}
