// Keep one visible, unmodified Spotify embed; custom buttons use its public API.
// https://developer.spotify.com/documentation/embeds/references/iframe-api
let apiPromise;
function api(){
  if(!apiPromise)apiPromise=new Promise((resolve,reject)=>{
    window.onSpotifyIframeApiReady=resolve;
    const script=document.createElement('script');script.src='https://open.spotify.com/embed/iframe-api/v1';script.async=true;
    script.onerror=()=>{apiPromise=null;reject(Error('Spotify could not load. Use the Open in Spotify link.'));};document.head.append(script);
  });
  return apiPromise;
}
export function createSpotifyPlayer({onError}){
  const panel=document.querySelector('#player');let controller=null,current=null,paused=true,generation=0;
  const update=()=>{
    for(const button of document.querySelectorAll('[data-play]')){
      const active=current?.id===button.dataset.play&&!paused;
      button.setAttribute('aria-label',`${active?'Pause':'Play'} ${button.dataset.title} on Spotify`);
    }
  };
  const close=()=>{generation++;controller?.destroy();controller=null;current=null;paused=true;panel.replaceChildren();panel.hidden=true;update();};
  async function play(track){
    if(!track.spotify){onError('Spotify does not have a preview for this track.');return;}
    if(current?.id===track.id&&controller){controller.togglePlay();return;}
    const epoch=++generation;current=track;paused=true;panel.hidden=false;
    const header=document.createElement('div');header.className='spotify-heading';
    const title=document.createElement('span');title.textContent='Spotify';
    const link=document.createElement('a');link.href='https://open.spotify.com/track/'+track.spotify;link.target='_blank';link.rel='noopener noreferrer';link.textContent='Open in Spotify ↗';
    const dismiss=document.createElement('button');dismiss.textContent='×';dismiss.id='player-close';dismiss.setAttribute('aria-label','Close Spotify player');dismiss.onclick=close;
    header.append(title,link,dismiss);
    const mount=document.createElement('div');mount.id='spotify-mount';mount.textContent='Loading Spotify…';
    controller?.destroy();controller=null;panel.replaceChildren(header,mount);update();
    try{
      const spotify=await api();if(epoch!==generation)return;
      spotify.createController(mount,{uri:'spotify:track:'+track.spotify,width:'100%',height:152},value=>{
        if(epoch!==generation){value.destroy();return;}controller=value;
        value.addListener('playback_update',event=>{if(epoch!==generation)return;paused=event.data.isPaused;update();});
        // Browsers may require the listener to press Play inside the visible embed.
        value.addListener('ready',()=>{if(epoch===generation)value.play();});
      });
    }catch(e){if(epoch===generation){mount.textContent=e.message;onError(e.message);}}
  }
  function button(track){
    const b=document.createElement('button');b.className='play-button';b.dataset.play=track.id;b.dataset.title=track.title;b.setAttribute('aria-label',`Play ${track.title} on Spotify`);b.disabled=!track.spotify;b.textContent='▶';b.onclick=e=>{e.stopPropagation();play(track);};return b;
  }
  return{play,button,close};
}
