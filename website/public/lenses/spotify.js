// Spotify remains an explicit fallback: its widget is never hidden during playback.
// https://developer.spotify.com/documentation/embeds/terms
let apiPromise;
function api(){
  if(!apiPromise)apiPromise=new Promise((resolve,reject)=>{
    window.onSpotifyIframeApiReady=resolve;
    const script=document.createElement('script');script.src='https://open.spotify.com/embed/iframe-api/v1';script.async=true;
    script.onerror=()=>{apiPromise=null;reject(Error('Spotify could not load. Use the Open in Spotify link.'));};document.head.append(script);
  });return apiPromise;
}
export function createSpotifyPlayer({onError}){
  const panel=document.querySelector('#spotify-panel');let controller=null,current=null,paused=true,generation=0,listener=()=>{};
  const emit=(loading=false)=>listener({playing:!!controller&&!paused&&!loading,loading});
  function pause(){controller?.pause();paused=true;emit();}
  function close(){generation++;controller?.destroy();controller=null;current=null;paused=true;panel.replaceChildren();panel.hidden=true;emit();}
  async function play(track){
    if(!track.spotify){onError('This track is not available on Spotify.');return;}
    if(current?.id===track.id&&controller){controller.togglePlay();return;}
    const epoch=++generation;current=track;paused=true;panel.hidden=false;controller?.destroy();controller=null;
    const header=document.createElement('div');header.className='spotify-heading';
    const title=document.createElement('span');title.textContent='Spotify playback';
    const link=document.createElement('a');link.href='https://open.spotify.com/track/'+track.spotify;link.target='_blank';link.rel='noopener noreferrer';link.textContent='Open Spotify ↗';
    const dismiss=document.createElement('button');dismiss.type='button';dismiss.textContent='×';dismiss.id='player-close';dismiss.setAttribute('aria-label','Close Spotify player');dismiss.onclick=close;header.append(title,link,dismiss);
    const mount=document.createElement('div');mount.id='spotify-mount';
    const note=document.createElement('p');note.textContent='Spotify requires its player to remain visible while playing here.';
    const activate=document.createElement('button');activate.className='spotify-activate';activate.textContent='Play here with Spotify';activate.type='button';
    mount.append(note,activate);panel.replaceChildren(header,mount);emit();
    activate.onclick=async()=>{
      activate.disabled=true;emit(true);
      try{
        const spotify=await api();if(epoch!==generation)return;
        spotify.createController(mount,{uri:'spotify:track:'+track.spotify,width:'100%',height:152},value=>{
          if(epoch!==generation){value.destroy();return;}controller=value;
          value.addListener('playback_update',event=>{if(epoch!==generation)return;paused=event.data.isPaused;emit(event.data.isBuffering);});
          value.addListener('ready',()=>{if(epoch===generation){emit();value.play();}});
        });
      }catch(e){if(epoch===generation){activate.disabled=false;emit();onError(e.message);}}
    };
  }
  return{play,pause,close,onState(callback){listener=callback;}};
}
