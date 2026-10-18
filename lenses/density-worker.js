import {estimateDensities} from './density.js?v=profiles1';
self.onmessage=({data})=>{
  const fields=estimateDensities(data.points,data.groups,{width:192,height:144,sigma:3});
  self.postMessage({key:data.key,fields},fields.map(f=>f.values.buffer));
};
