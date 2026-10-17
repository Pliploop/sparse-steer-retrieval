// Binned Gaussian KDE: actual 2D samples, not an ellipse fitted to their covariance.
export function estimateDensities(points,groups,{width=96,height=76,sigma=3.1}={}){
  const bounds={left:-1.9,right:1.9,bottom:-1.7,top:1.7};
  const radius=Math.ceil(sigma*3),kernel=[];
  for(let d=-radius;d<=radius;d++)kernel.push(Math.exp(-.5*(d/sigma)**2));
  const total=kernel.reduce((a,b)=>a+b,0);for(let i=0;i<kernel.length;i++)kernel[i]/=total;
  return groups.map(group=>{
    const field=new Float32Array(width*height),middle=new Float32Array(field.length),smooth=new Float32Array(field.length);
    const members=points.filter(p=>p.cluster===group.id);
    for(const p of members){const x=Math.round((p.x-bounds.left)/(bounds.right-bounds.left)*(width-1)),y=Math.round((bounds.top-p.y)/(bounds.top-bounds.bottom)*(height-1));if(x>=0&&x<width&&y>=0&&y<height)field[y*width+x]+=1/Math.max(members.length,1);}
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){let value=0;for(let d=-radius;d<=radius;d++)if(x+d>=0&&x+d<width)value+=field[y*width+x+d]*kernel[d+radius];middle[y*width+x]=value;}
    let maximum=0,peak=0;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){let value=0;for(let d=-radius;d<=radius;d++)if(y+d>=0&&y+d<height)value+=middle[(y+d)*width+x]*kernel[d+radius];smooth[y*width+x]=value;if(value>maximum){maximum=value;peak=y*width+x;}}
    return{id:group.id,width,height,bounds,values:smooth,maximum,mode:{x:bounds.left+(peak%width)/(width-1)*(bounds.right-bounds.left),y:bounds.top-Math.floor(peak/width)/(height-1)*(bounds.top-bounds.bottom)}};
  });
}
export function densityImage(field,color){
  const canvas=document.createElement('canvas');canvas.width=field.width;canvas.height=field.height;const ctx=canvas.getContext('2d'),image=ctx.createImageData(field.width,field.height);
  const rgb=[1,3,5].map(start=>parseInt(color.slice(start,start+2),16));
  for(let i=0;i<field.values.length;i++){const density=field.maximum?field.values[i]/field.maximum:0;image.data[i*4]=rgb[0];image.data[i*4+1]=rgb[1];image.data[i*4+2]=rgb[2];image.data[i*4+3]=density<.008?0:Math.round(density**.75*49);}
  ctx.putImageData(image,0,0);return canvas.toDataURL('image/png');
}
