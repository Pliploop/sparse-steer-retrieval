(()=>{
  const saved=localStorage.getItem('ssr-theme');
  document.documentElement.dataset.theme=saved|| (matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  document.addEventListener('DOMContentLoaded',()=>{
    const button=document.querySelector('#theme-toggle');if(!button)return;
    const update=()=>{const dark=document.documentElement.dataset.theme==='dark';button.setAttribute('aria-label',`Switch to ${dark?'light':'dark'} theme`);button.setAttribute('title',`Switch to ${dark?'light':'dark'} theme`);};
    button.onclick=()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=theme;localStorage.setItem('ssr-theme',theme);update();window.dispatchEvent(new Event('themechange'));};update();
  });
})();
