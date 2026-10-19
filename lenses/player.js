// One transport surface for exact clips and the optional Spotify fallback.
// Exact clips drive the waveform from live audio samples; Spotify stays unanalysed.
export function createTransport({onToggle}){
  const panel=document.querySelector('#player');panel.className='player music-player';
  panel.setAttribute('aria-label','Music player');
  const toggle=document.createElement('button');toggle.id='player-play';toggle.className='play-button';toggle.type='button';toggle.onclick=onToggle;
  const wave=document.createElement('div');wave.className='player-wave';wave.setAttribute('aria-hidden','true');
  const ns='http://www.w3.org/2000/svg',drawing=document.createElementNS(ns,'svg');drawing.setAttribute('viewBox','0 0 100 32');
  const bars=[],levels=new Float32Array(25);
  for(let i=0;i<25;i++){const bar=document.createElementNS(ns,'line');bar.setAttribute('x1',2+i*4);bar.setAttribute('x2',2+i*4);bar.setAttribute('y1',14);bar.setAttribute('y2',18);drawing.append(bar);bars.push(bar);}wave.append(drawing);
  const text=document.createElement('div');text.className='player-track';
  const title=document.createElement('strong');title.id='player-title';
  const artist=document.createElement('span');artist.id='player-artist';text.append(title,artist);
  const status=document.createElement('span');status.className='sr-only';status.setAttribute('role','status');
  panel.replaceChildren(toggle,wave,text,status);
  let current=null,playing=false,loading=false,color='#9b9284',analyser=null,samples=null,signal=false,waveFrame=0,lastWave=-Infinity;
  const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
  function sampleWave(){
    analyser.getFloatTimeDomainData(samples);
    const step=Math.floor(samples.length/bars.length);
    for(let i=0;i<bars.length;i++){
      let power=0;for(let j=i*step;j<(i+1)*step;j++)power+=samples[j]*samples[j];
      const target=Math.min(1,Math.pow(Math.sqrt(power/step),.65)*1.9);
      levels[i]+=(target-levels[i])*(target>levels[i]?.58:.24);
      const height=3+levels[i]*24;
      bars[i].setAttribute('y1',((32-height)/2).toFixed(2));bars[i].setAttribute('y2',((32+height)/2).toFixed(2));
    }
  }
  function animateWave(now){
    waveFrame=0;if(!playing||!signal||!analyser||document.hidden||reducedMotion.matches)return;
    // Sample only this 100 × 32 SVG at 30 Hz. Never request a map repaint here.
    if(now-lastWave>=32){sampleWave();lastWave=now;}
    waveFrame=requestAnimationFrame(animateWave);
  }
  function syncWave(){
    wave.dataset.mode=signal&&analyser?'signal':'idle';
    if(!playing||!signal||!analyser||document.hidden||reducedMotion.matches){cancelAnimationFrame(waveFrame);waveFrame=0;return;}
    if(!waveFrame){lastWave=-Infinity;waveFrame=requestAnimationFrame(animateWave);}
  }
  reducedMotion.addEventListener('change',syncWave);document.addEventListener('visibilitychange',syncWave);
  function refresh(){
    panel.hidden=!current;panel.dataset.track=current?.id||'';panel.dataset.playing=String(playing);panel.dataset.loading=String(loading);panel.style.setProperty('--playing-color',color);
    title.textContent=current?.title||'';title.title=current?.title||'';artist.textContent=current?.artist||'';
    toggle.textContent=playing?'Ⅱ':'▶';toggle.setAttribute('aria-label',`${playing?'Pause':'Play'} ${current?.title||'excerpt'}`);toggle.setAttribute('aria-pressed',String(playing));toggle.setAttribute('aria-busy',String(loading));
    for(const b of document.querySelectorAll('[data-play]')){const active=current?.id===b.dataset.play&&playing;b.setAttribute('aria-label',`${active?'Pause':'Play'} ${b.dataset.title||'excerpt'}`);b.setAttribute('aria-pressed',String(active));}
    for(const c of document.querySelectorAll('.map-card,.track-card'))c.classList.toggle('playing',playing&&c.dataset.track===current?.id);
    status.textContent=loading?'Loading audio':playing?'Playing':'Paused';
    syncWave();
    document.dispatchEvent(new CustomEvent('music-lenses-playback',{detail:{id:current?.id,playing,color}}));
  }
  return{
    update(value={}){if('track'in value)current=value.track;if('playing'in value)playing=!!value.playing;if('loading'in value)loading=!!value.loading;if('signal'in value)signal=!!value.signal;if(value.color)color=value.color;refresh();},
    refresh,
    setAnalyser(node){analyser=node;samples=new Float32Array(node.fftSize);syncWave();},
    get track(){return current;},get playing(){return playing;}
  };
}

export function createClipPlayer({onError,resolveSource,remote}){
  const audio=new Audio();audio.crossOrigin='anonymous';audio.preload='metadata';audio.id='clip-audio';audio.hidden=true;audio.setAttribute('aria-hidden','true');document.body.append(audio);
  let selected=null,reference=null,source=null,generation=0,loading=false,mode=null,remoteState={playing:false,loading:false},colorFor=()=>null;
  let context=null,mediaSource=null;
  const transport=createTransport({onToggle:()=>{if(selected||reference)play(selected||reference);}});
  function unlockAudio(){
    const AudioContext=window.AudioContext||window.webkitAudioContext;if(!AudioContext)return;
    try{
      if(!context){context=new AudioContext({latencyHint:'interactive'});context.addEventListener('statechange',refresh);}
      const connect=()=>{
        if(mediaSource||context.state!=='running')return;
        const analyser=context.createAnalyser();analyser.fftSize=512;analyser.connect(context.destination);
        mediaSource=context.createMediaElementSource(audio);mediaSource.connect(analyser);transport.setAnalyser(analyser);refresh();
      };
      // Resume synchronously in the user's play gesture, before any source await.
      if(context.state==='running')connect();else context.resume().then(connect).catch(()=>{});
    }catch{/* Unsupported Web Audio leaves the native audio player available. */}
  }
  function refresh(){
    const track=selected||reference;
    transport.update({track,playing:mode==='spotify'?remoteState.playing:mode==='audio'&&!audio.paused&&!audio.ended&&!loading&&(!mediaSource||context.state==='running'),loading,signal:mode==='audio',color:track&&colorFor(track.id)});
  }
  function pause(){generation++;loading=false;audio.pause();remote?.pause();refresh();}
  async function play(track){
    if(selected?.id===track.id&&mode==='audio'&&source){
      if(!audio.paused&&(!mediaSource||context.state==='running')){generation++;audio.pause();return;}
      unlockAudio();
      if(audio.currentTime>=source.end-.05)audio.currentTime=source.start;
      const epoch=++generation;try{await audio.play();}catch(e){if(epoch===generation&&e.name!=='AbortError')onError(`Could not play this excerpt: ${e.message}`);}return;
    }
    if(selected?.id===track.id&&mode==='spotify'){remote.play(track);return;}
    unlockAudio();
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
  remote?.onState(value=>{remoteState=value;if(mode==='spotify'){loading=!!value.loading;transport.update({track:selected,playing:!!value.playing,loading,signal:false,color:colorFor(selected.id)});}});
  function button(track){
    const b=document.createElement('button');b.type='button';b.className='play-button';b.dataset.play=track.id;b.dataset.title=track.title;b.textContent='▶';const active=transport.playing&&transport.track?.id===track.id;b.setAttribute('aria-label',`${active?'Pause':'Play'} ${track.title}`);b.setAttribute('aria-pressed',String(active));b.onclick=e=>{e.stopPropagation();play(track);};return b;
  }
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)transport.refresh();});
  return{play,button,pause,refresh,
    setContext({seed,color}){reference=seed;colorFor=color||colorFor;refresh();},
    get current(){return selected;},get audio(){return audio;}
  };
}
