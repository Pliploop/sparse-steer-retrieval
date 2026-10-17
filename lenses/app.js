import {stylizeControls} from './selectors.js?v=motion1';
import {createSpace} from './space.js?v=motion1';
import {createSpotifyPlayer} from './spotify.js?v=canvas2';
const $=s=>document.querySelector(s),base=new URL('./data/',import.meta.url);
const state={manifest:null,catalog:null,category:'instrumentation',basis:'text',full:false,seed:0,compare:false,busy:true,request:0};
const trackIndices=new Map(),views=new Map(),captions=new Map(),affinities=new Map(),pending=new Map();
let sequence=0,searchTimer;
const syncSelectors=stylizeControls();
const worker=new Worker(new URL('./search-worker.js?v=canvas2',import.meta.url),{type:'module'});
worker.onmessage=({data})=>{const item=pending.get(data.id);if(!item)return;pending.delete(data.id);data.error?item.reject(Error(data.error)):item.resolve(data.result);};
worker.onerror=()=>{for(const item of pending.values())item.reject(Error('Search could not start. Please reload the page.'));pending.clear();};
function rpc(type,value={}){return new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});worker.postMessage({id,type,...value});});}
function el(tag,cls,text){const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;}
function error(message=''){$('#alert').textContent=message;$('#alert').hidden=!message;}
function busy(value,label='Loading this lens…'){syncSelectors();state.busy=value;$('#map-stage').setAttribute('aria-busy',String(value));$('#busy').hidden=!value;$('#busy-text').textContent=label;}
function url(path){return new URL(path+'?v='+(path.startsWith('maps/')||path.startsWith('views/')?state.manifest.map_version||state.manifest.version:state.manifest.dataset_version||state.manifest.version),base);}
async function resource(path,Type){const r=await fetch(url(path));if(!r.ok)throw Error('This part of the collection could not load. Please try again.');return Type?new Type(await r.arrayBuffer()):r.json();}
function cached(cache,key,load,limit=24){if(cache.has(key)){const value=cache.get(key);cache.delete(key);cache.set(key,value);return value;}const value=load().catch(e=>{cache.delete(key);throw e;});cache.set(key,value);while(cache.size>limit)cache.delete(cache.keys().next().value);return value;}
function basic(index){const row=state.catalog.tracks[index];return{id:row[0],title:row[1],artist:state.catalog.artists[row[2]],spotify:row[3]};}
function category(){return state.manifest.categories.find(c=>c.id===state.category);}
async function view(key){return cached(views,key,async()=>{const v=await resource(state.manifest.views[key]);const [coordinates,groups]=await Promise.all([resource(v.coordinates,Float32Array),resource(v.groups,Uint8Array)]);return{...v,key,coordinates,groups};},3);}
async function description(index){const shard=Math.floor(index/state.manifest.shard_size);const values=await cached(captions,shard,()=>resource(`captions/${shard}.json`));return values[index%state.manifest.shard_size];}
async function scores(index,c){const shard=Math.floor(index/state.manifest.shard_size),key=c.id+':'+shard;const values=await cached(affinities,key,()=>resource(`attributes/${c.id}/${shard}.bin`,Float32Array));const start=index%state.manifest.shard_size*c.members.length;return values.subarray(start,start+c.members.length);}
async function loadTrack(id){
  const index=trackIndices.get(id),seed=state.seed,c=category(),full=state.full;
  const text=await description(index);let attributes=[];
  if(!full){const [candidate,reference]=await Promise.all([scores(index,c),scores(seed,c)]);attributes=Array.from(candidate,(_,j)=>j).sort((a,b)=>Math.min(candidate[b],reference[b])-Math.min(candidate[a],reference[a])).slice(0,3).map(j=>({label:c.members[j],score:candidate[j],seed_score:reference[j]}));}
  return{...basic(index),caption:text,attributes};
}
function highlight(text,attributes=[]){const p=el('p','card-caption');if(!text){p.textContent='Loading description…';return p;}const words=[...new Set(attributes.flatMap(a=>a.label.toLowerCase().match(/[a-z]{4,}/g)||[]))];if(!words.length){p.textContent=text;return p;}const regex=new RegExp('\\b('+words.join('|')+')(?:s|ing|ed)?\\b','gi');let last=0;for(const match of text.matchAll(regex)){p.append(text.slice(last,match.index),el('mark','',match[0]));last=match.index+match[0].length;}p.append(text.slice(last));return p;}
const player=createSpotifyPlayer({onError:error});
const space=createSpace({playButton:player.button,chooseSeed:t=>choose(trackIndices.get(t.id)),highlight,isBusy:()=>state.busy,getTrack:id=>basic(trackIndices.get(id)),loadTrack});
function seedPanel(){const track=basic(state.seed),actions=el('div','seed-actions');actions.append(player.button(track));$('#seed-panel').replaceChildren(el('h3','',track.title),el('p','artist',track.artist),actions);}
function resultTrack(row,map,rank){return{...basic(row.index),score:row.score,rank,cluster:map.groups[row.index]};}
async function run(){
  const request=++state.request;error();busy(true);seedPanel();const key=state.full?'full':state.category+'-'+state.basis;
  try{
    const map=await view(key);if(request!==state.request)return;
    const result=await rpc('retrieve',{seed:state.seed,compare:state.compare,view:{key,vectors:map.vectors,dimensions:map.dimensions,invalid:map.invalid}});if(request!==state.request)return;
    const top=new Set(result.results.map(r=>r.index)),baseline=new Set(result.baseline.map(r=>r.index));
    const shown=new Set([...state.manifest.background,state.seed,...top,...baseline,...map.representatives]);
    const plot=[...shown].map(i=>({...basic(i),x:map.coordinates[i*2],y:map.coordinates[i*2+1],cluster:map.groups[i],kind:i===state.seed?'seed':top.has(i)?'result':baseline.has(i)?'base':'background'}));
    const seed={...basic(state.seed),cluster:map.groups[state.seed]};
    const output={seed,category:state.full?state.manifest.categories[0]:category(),basis_method:state.basis,method:state.full?'base':'lens',info:map.info,signature:map.signature,results:result.results.map((row,i)=>resultTrack(row,map,i+1)),baseline:result.baseline.map((row,i)=>resultTrack(row,map,i+1)),representatives:map.representatives.map((index,i)=>resultTrack({index},map,i+1)),plot,clusters:map.clusters};
    space.render(output,{compare:state.compare});
    $('#results-subtitle').textContent=`${state.manifest.count.toLocaleString()} songs · ${state.full?'full space':state.basis==='sparse'?'Adam sparse basis':'text basis'} · ${map.dimensions} dimensions`;
    $('#subspace-note').textContent=state.full?'Full 512-dimensional MuQ-MuLan audio similarity.':`${map.dimensions} orthogonal directions retain ${(map.info.variance*100).toFixed(0)}% of member variance.`;
    $('#spectrum').replaceChildren(...(state.full?[]:map.info.spectrum).map((v,i)=>{const bar=el('i',i>=map.info.k?'off':'');bar.style.height=Math.max(2,v*160)+'px';return bar;}));
    const inversion=map.info.inversion;$('#basis-diagnostics').textContent=!state.full&&inversion?`Adam · ${inversion.settings.num_steps} steps · mean support ${inversion.mean_support.toFixed(1)} · reconstruction cosine ${inversion.mean_text_cosine.toFixed(3)}`:'';
    $('#basis-select').disabled=state.full;$('#inspect-prompts').disabled=state.full;$('#compare-toggle').disabled=state.full;
    const params=new URLSearchParams({song:seed.id,lens:state.full?'full':state.category,basis:state.basis});history.replaceState(null,'','?'+params);
  }catch(e){if(request===state.request)error(e.message);}finally{if(request===state.request)busy(false);}
}
function choose(index){if(!Number.isInteger(index))return;state.seed=index;$('#search-popover').hidePopover();$('#search-results').hidden=true;$('#search').value='';run();}
$('#lens-select').onchange=e=>{state.full=e.target.value==='__full';if(!state.full)state.category=e.target.value;run();};
$('#basis-select').onchange=e=>{state.basis=e.target.value;run();};
$('#compare-toggle').onclick=()=>{state.compare=!state.compare;$('#compare-toggle').setAttribute('aria-pressed',String(state.compare));run();};
$('#shuffle').onclick=()=>choose(Math.floor(Math.random()*state.manifest.count));
$('#search').oninput=e=>{clearTimeout(searchTimer);const q=e.target.value;searchTimer=setTimeout(async()=>{try{const hits=await rpc('search',{query:q});if($('#search').value!==q)return;$('#search-results').replaceChildren(...hits.map(index=>{const track=basic(index),b=el('button','search-hit'),text=el('span');text.append(el('strong','',track.title),el('small','',track.artist));b.append(text);b.onclick=()=>choose(index);return b;}));$('#search-results').hidden=!hits.length;$('#search').setAttribute('aria-expanded',String(!!hits.length));}catch(e){error(e.message);}},150);};
$('#inspect-prompts').onclick=()=>{const c=category();$('#prompts-title').textContent=c.name;$('#prompts-list').replaceChildren(...c.members.map((m,i)=>{const row=el('div','prompt-row');row.append(el('span','',String(i+1).padStart(2,'0')),el('div','prompt-description',c.prompt_overrides?.[m]||c.template.replace('{member}',m)));if(c.controls?.includes(m))row.append(el('small','control-member','Control'));return row;}));$('#prompts-dialog').showModal();};
$('#close-prompts').onclick=()=>$('#prompts-dialog').close();
window.addEventListener('keydown',e=>{if(e.target.closest('input,select,textarea,dialog,[role=combobox]'))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){const list=state.manifest.categories,index=list.findIndex(c=>c.id===state.category),next=list[(index+(e.key==='ArrowRight'?1:list.length-1))%list.length];state.category=next.id;state.full=false;$('#lens-select').value=next.id;run();}});
async function init(){
  try{
    const r=await fetch(new URL('manifest.json',base),{cache:'no-cache'});if(!r.ok)throw Error('The music collection could not load.');state.manifest=await r.json();
    state.catalog=await resource(state.manifest.catalog);state.catalog.tracks.forEach((row,i)=>trackIndices.set(row[0],i));
    await rpc('init',{manifest:state.manifest,catalog:state.catalog});
    const params=new URLSearchParams(location.search);state.seed=trackIndices.get(params.get('song'))??state.manifest.suggestions[0];
    const lens=params.get('lens');if(lens==='full')state.full=true;else if(state.manifest.categories.some(c=>c.id===lens))state.category=lens;
    if(params.get('basis')==='sparse')state.basis='sparse';$('#basis-select').value=state.basis;
    $('#lens-select').replaceChildren(...state.manifest.categories.map(c=>{const o=el('option','',c.name);o.value=c.id;return o;}),Object.assign(el('option','','Full embedding'),{value:'__full'}));$('#lens-select').value=state.full?'__full':state.category;
    $('#seed-chips').replaceChildren(...state.manifest.suggestions.map(index=>{const b=el('button','',basic(index).artist);b.onclick=()=>choose(index);return b;}));
    $('#connection').textContent=state.manifest.count.toLocaleString()+' songs';$('#status-dot').classList.add('ready');await run();
  }catch(e){error(e.message);busy(false);}
}
init();
