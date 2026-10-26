// Match the embedded teaching figure's natural height without nested scroll traps.
const frame=document.querySelector('.method-figure iframe');
addEventListener('message',event=>{if(!frame||event.origin!==location.origin||event.source!==frame.contentWindow||event.data?.type!=='music-lenses-method-size')return;const h=Number(event.data.height);if(Number.isFinite(h)&&h>=300&&h<=5000)frame.style.height=Math.ceil(h)+'px';});
