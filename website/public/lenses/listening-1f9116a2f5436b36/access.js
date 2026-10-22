const configResponse=await fetch('./config.json',{cache:'no-store'});
if(!configResponse.ok)throw Error('Listening room configuration could not load');
const config=await configResponse.json(),apiOrigin=new URL(config.api).origin;
if(!apiOrigin.startsWith('https://')&&!/^http:\/\/(127\.0\.0\.1|localhost)(:|$)/.test(apiOrigin))throw Error('The study service must use HTTPS');
const sessionKey='music-lenses-study:'+apiOrigin;
let token=sessionStorage.getItem(sessionKey)||'';
export const serviceOrigin=apiOrigin;
export function lock(message=''){token='';sessionStorage.removeItem(sessionKey);document.body.classList.add('locked');document.querySelector('#access-error').textContent=message;document.querySelector('#access-password').value='';document.dispatchEvent(new CustomEvent('study-locked'));}
export async function response(path,value){
 const r=await fetch(apiOrigin+'/api/'+path,{method:value===undefined?'GET':'POST',cache:'no-store',credentials:'omit',referrerPolicy:'no-referrer',headers:{...(token?{Authorization:'Bearer '+token}:{}),...(value===undefined?{}:{'Content-Type':'application/json','X-Music-Lenses-Admin':'1'})},...(value===undefined?{}:{body:JSON.stringify(value)})});
 if(r.status===401){lock('Unlock the listening room to continue.');throw Error('Your session needs to be unlocked.');}
 if(!r.ok){let message='The study service could not respond. Please retry.';try{message=(await r.json()).error||message;}catch{}throw Error(message);}
 return r;
}
export async function api(path,value){return(await response(path,value)).json();}
export async function setupAccess(){
 document.querySelector('#access-form').onsubmit=async event=>{
  event.preventDefault();const button=document.querySelector('#unlock'),input=document.querySelector('#access-password'),error=document.querySelector('#access-error');button.disabled=true;error.textContent='';
  try{
   const r=await fetch(apiOrigin+'/api/login',{method:'POST',credentials:'omit',referrerPolicy:'no-referrer',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:input.value})}),value=await r.json();
   if(!r.ok)throw Error(value.error||'The room could not be unlocked');
   token=value.token;sessionStorage.setItem(sessionKey,token);input.value='';
   document.body.classList.remove('locked');document.dispatchEvent(new CustomEvent('study-unlocked'));
  }catch(e){error.textContent=e.message;}finally{button.disabled=false;}
 };
 document.querySelector('#lock-room').onclick=async()=>{try{await api('logout',{});lock();}catch(e){lock(e.message);}};
 if(!token)return false;
 try{await api('study');document.body.classList.remove('locked');return true;}catch(e){document.querySelector('#access-error').textContent=e.message;return false;}
}
export async function download(path,name){const blob=await(await response(path)).blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
