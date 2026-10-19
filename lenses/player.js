// One transport surface for exact clips and the optional Spotify fallback.
// The bars indicate playback activity; they do not claim to analyse Spotify audio.
export function createTransport({onToggle}){
  const panel=document.querySelector('#player');panel.className='player music-player';
  panel.setAttribute('aria-label','Music player');
  const toggle=document.createElement('button');toggle.id='player-play';toggle.className='play-button';toggle.type='button';toggle.onclick=onToggle;
  const wave=document.createElement('div');wave.className='player-wave';wave.setAttribute('aria-hidden','true');
  for(let i=0;i<25;i++){const bar=document.createElement('i');bar.style.setProperty('--bar-height',(5+((i*17+11)%19))+'px');bar.style.setProperty('--bar-delay',(-i*.137)+'s');bar.style.setProperty('--bar-duration',(.65+(i%7)*.11)+'s');wave.append(bar);}
  const text=document.createElement('div');text.className='player-track';
  const title=document.createElement('strong');title.id='player-title';
  const artist=document.createElement('span');artist.id='player-artist';text.append(title,artist);
  const status=document.createElement('span');status.className='sr-only';status.setAttribute('role','status');
  panel.replaceChildren(toggle,wave,text,status);
  let current=null,playing=false,loading=false,color='#9b9284';
  function refresh(){
    panel.hidden=!current;panel.dataset.track=current?.id||'';panel.dataset.playing=String(playing);panel.dataset.loading=String(loading);panel.style.setProperty('--playing-color',color);
    title.textContent=current?.title||'';title.title=current?.title||'';artist.textContent=current?.artist||'';
    toggle.textContent=playing?'Ⅱ':'▶';toggle.setAttribute('aria-label',`${playing?'Pause':'Play'} ${current?.title||'excerpt'}`);toggle.setAttribute('aria-pressed',String(playing));toggle.setAttribute('aria-busy',String(loading));
    for(const b of document.querySelectorAll('[data-play]')){const active=current?.id===b.dataset.play&&playing;b.setAttribute('aria-label',`${active?'Pause':'Play'} ${b.dataset.title||'excerpt'}`);b.setAttribute('aria-pressed',String(active));}
    for(const c of document.querySelectorAll('.map-card,.track-card'))c.classList.toggle('playing',playing&&c.dataset.track===current?.id);
    status.textContent=loading?'Loading audio':playing?'Playing':'Paused';
    document.dispatchEvent(new CustomEvent('music-lenses-playback',{detail:{id:current?.id,playing,color}}));
  }
  return{
    update(value={}){if('track'in value)current=value.track;if('playing'in value)playing=!!value.playing;if('loading'in value)loading=!!value.loading;if(value.color)color=value.color;refresh();},
    refresh,
    get track(){return current;},get playing(){return playing;}
  };
}

export function createClipPlayer({onError,resolveSource,remote}){
  const audio=new Audio();audio.preload='metadata';audio.id='clip-audio';audio.hidden=true;audio.setAttribute('aria-hidden','true');document.body.append(audio);
  let selected=null,reference=null,source=null,generation=0,loading=false,mode=null,remoteState={playing:false,loading:false},colorFor=()=>null;
  const transport=createTransport({onToggle:()=>{if(selected||reference)play(selected||reference);}});
  function refresh(){
    const track=selected||reference;
    transport.update({track,playing:mode==='spotify'?remoteState.playing:mode==='audio'&&!audio.paused&&!audio.ended&&!loading,loading,color:track&&colorFor(track.id)});
  }
  function pause(){generation++;loading=false;audio.pause();remote?.pause();refresh();}
  async function play(track){
    if(selected?.id===track.id&&mode==='audio'&&source){
      if(!audio.paused){generation++;audio.pause();return;}
      if(audio.currentTime>=source.end-.05)audio.currentTime=source.start;
      const epoch=++generation;try{await audio.play();}catch(e){if(epoch===generation&&e.name!=='AbortError')onError(`Could not play this excerpt: ${e.message}`);}return;
    }
    if(selected?.id===track.id&&mode==='spotify'){remote.play(track);return;}
    const epoch=++generation;audio.pause();remote?.pause();selected=track;source=null;mode=null;loading=true;refresh();
    try{
      const resolved=await resolveSource(track);if(epoch!==generation)return;
      if(!resolved){
        loading=false;mode='spotify';refresh();
        if(remote)remote.play(track);else onError('This excerpt is unavailable.');return;
      }
      mode='audio';source={start:0,end:30,...resolved};
      audio.src=source.url;audio.currentTime=source.start;await audio.play();
    }catch(e){if(epoch===generation){loading=false;audio.pause();refresh();onError(`Could not play this excerpt: ${e.message}`);}}
  }
  ['playing','pause','ended','waiting','stalled','error'].forEach(type=>audio.addEventListener(type,()=>{
    if(mode!=='audio')return;
    if(['playing','pause','ended'].includes(type))loading=false;
    if(type==='waiting')loading=true;
    if(type==='error'){loading=false;audio.pause();onError('This excerpt could not load. Please try another track.');}
    refresh();
  }));
  audio.addEventListener('loadedmetadata',()=>{if(mode==='audio'&&source)audio.currentTime=source.start;});
  audio.addEventListener('timeupdate',()=>{if(mode==='audio'&&source&&audio.currentTime>=source.end){audio.pause();audio.currentTime=source.start;}});
  remote?.onState(value=>{remoteState=value;if(mode==='spotify'){loading=!!value.loading;transport.update({track:selected,playing:!!value.playing,loading,color:colorFor(selected.id)});}});
  function button(track){
    const b=document.createElement('button');b.type='button';b.className='play-button';b.dataset.play=track.id;b.dataset.title=track.title;b.textContent='▶';const active=transport.playing&&transport.track?.id===track.id;b.setAttribute('aria-label',`${active?'Pause':'Play'} ${track.title}`);b.setAttribute('aria-pressed',String(active));b.onclick=e=>{e.stopPropagation();play(track);};return b;
  }
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)transport.refresh();});
  return{play,button,pause,refresh,
    setContext({seed,color}){reference=seed;colorFor=color||colorFor;refresh();},
    get current(){return selected;},get audio(){return audio;}
  };
}
