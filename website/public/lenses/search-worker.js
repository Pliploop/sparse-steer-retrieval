// No server and no audio/model code: exact cosine over a lazy float32 lens matrix.
let manifest,catalog,artists,searchText,base;
const matrices=new Map();
const dataURL=path=>{const c=path.match(/^vectors\/(.+)-(?:text|sparse)\./)?.[1];return new URL('data/'+path+'?v='+(manifest.category_versions?.[c]||manifest.dataset_version||manifest.version),import.meta.url);};
async function binary(path,Type){const r=await fetch(dataURL(path));if(!r.ok)throw Error('Could not load this lens. Please try again.');return new Type(await r.arrayBuffer());}
async function matrix(view){
  if(matrices.has(view.key)){const value=matrices.get(view.key);matrices.delete(view.key);matrices.set(view.key,value);return value;}
  const value=await binary(view.vectors,Float32Array);
  if(value.length!==manifest.count*view.dimensions)throw Error('This lens package is incomplete.');
  matrices.set(view.key,value);while(matrices.size>3)matrices.delete(matrices.keys().next().value);return value;
}
function nearest(x,seed,d,invalid){
  const top=[],bad=new Set(invalid||[]),offset=seed*d;let worst=-Infinity,worstAt=0,norm=0;
  for(let j=0;j<d;j++)norm+=x[offset+j]**2;
  if(norm<1e-12)throw Error('This song has no usable projection for this lens. Try another lens.');
  for(let i=0;i<manifest.count;i++){
    if(i===seed||bad.has(i))continue;let score=0;const start=i*d;
    for(let j=0;j<d;j++)score+=x[start+j]*x[offset+j];
    if(top.length<10)top.push({index:i,score});else if(score>worst)top[worstAt]={index:i,score};else continue;
    worst=Infinity;for(let j=0;j<top.length;j++)if(top[j].score<worst){worst=top[j].score;worstAt=j;}
  }
  return top.sort((a,b)=>b.score-a.score);
}
async function baseline(seed){
  if(!base)base=Promise.all([binary(manifest.full_neighbours,Uint32Array),binary(manifest.full_scores,Float32Array)]).catch(e=>{base=null;throw e;});
  const [indices,scores]=await base;return Array.from({length:10},(_,j)=>({index:indices[seed*10+j],score:scores[seed*10+j]}));
}
self.onmessage=async({data})=>{
  const {id,type}=data;
  try{
    let result;
    if(type==='init'){
      manifest=data.manifest;catalog=data.catalog.tracks;artists=data.catalog.artists;
      searchText=catalog.map(row=>(row[1]+' '+artists[row[2]]).toLowerCase());result={count:catalog.length};
    }else if(type==='search'){
      const words=data.query.toLowerCase().trim().split(/\s+/).filter(Boolean);result=[];
      for(let i=0;i<searchText.length&&result.length<24;i++)if(words.every(w=>searchText[i].includes(w)))result.push(i);
    }else if(type==='retrieve'){
      const started=performance.now();
      const results=data.view.key==='full'?await baseline(data.seed):nearest(await matrix(data.view),data.seed,data.view.dimensions,data.view.invalid);
      result={results,baseline:data.compare&&data.view.key!=='full'?await baseline(data.seed):[],elapsed_ms:performance.now()-started};
    }else throw Error('Unknown search request');
    self.postMessage({id,result});
  }catch(error){self.postMessage({id,error:error.message});}
};
