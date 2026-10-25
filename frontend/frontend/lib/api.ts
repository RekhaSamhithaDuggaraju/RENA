export const API=process.env.NEXT_PUBLIC_API_URL||"http://localhost:8000";
export function authHeaders(extra:Record<string,string>={}){
  const token=typeof window!=='undefined'?localStorage.getItem('rena_token'):null;
  return {...extra,...(token?{Authorization:`Bearer ${token}`}:{})};
}
export async function apiFetch(path:string, init:RequestInit={}){
  const headers=new Headers(init.headers||{});
  const token=typeof window!=='undefined'?localStorage.getItem('rena_token'):null;
  if(token) headers.set('Authorization',`Bearer ${token}`);
  const res=await fetch(API+path,{...init,headers,cache:'no-store'});
  if(!res.ok){let msg=`Request failed (${res.status})`;try{const data=await res.json();msg=data.detail||msg}catch{}throw new Error(msg)}
  return res;
}
export async function auth(mode:'login'|'register',email:string,password:string){
  const res=await fetch(`${API}/production/auth/${mode}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email,password}),cache:'no-store'});
  const data=await res.json().catch(()=>({}));
  if(!res.ok) throw new Error(data.detail||'Authentication failed');
  if(typeof window!=='undefined'){localStorage.setItem('rena_token',data.token);document.cookie=`rena_token=${encodeURIComponent(data.token)}; Path=/; SameSite=Lax`;}
  return data;
}
export function logout(){if(typeof window!=='undefined'){localStorage.removeItem('rena_token');document.cookie='rena_token=; Path=/; Max-Age=0; SameSite=Lax';}}
export async function me(){const r=await apiFetch('/production/auth/me');return r.json();}


export async function listAudits(limit=50){
  const r=await apiFetch(`/audits?limit=${encodeURIComponent(String(limit))}`);
  return r.json();
}

export async function createAudit(algorithm:string, forget_fraction=0.1, split_seed=42, reference_retrains=10, reference_seed=50000){
  const r=await apiFetch('/audits',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({algorithm,forget_fraction,split_seed,reference_retrains,reference_seed})
  });
  return r.json();
}
