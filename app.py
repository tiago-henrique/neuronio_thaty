import json
import os

import anthropic
import streamlit as st
import streamlit.components.v1 as components

MODEL = "claude-sonnet-5-5"  # ajuste se quiser outro modelo

st.set_page_config(page_title="Neurônio com agente de IA", layout="wide")

# ---------------- estado e limites ----------------
DEFAULT = dict(na=145.0, k=4.0, ca=2.0, iext=8.0)
LIM = dict(na=(80.0, 180.0, 1.0), k=(1.0, 30.0, 0.5), ca=(0.1, 6.0, 0.1), iext=(0.0, 30.0, 0.5))
for key, val in DEFAULT.items():
    st.session_state.setdefault(key, val)
st.session_state.setdefault("chat", [])
st.session_state.setdefault("api_msgs", [])


def ajustar(key, v):
    lo, hi, step = LIM[key]
    return round(min(hi, max(lo, float(v))) / step) * step


# ---------------- agente ----------------
TOOLS = [
    {
        "name": "definir_ions",
        "description": "Define concentrações iônicas EXTRACELULARES em mM. Omita o que não deve mudar. "
        "Valores fisiológicos: Na+ 145, K+ 4, Ca2+ 2.",
        "input_schema": {
            "type": "object",
            "properties": {"na": {"type": "number"}, "k": {"type": "number"}, "ca": {"type": "number"}},
        },
    },
    {
        "name": "definir_estimulo",
        "description": "Define a corrente injetada no soma (µA/cm²). 0 = sem estímulo; a partir de ~7 há disparos repetitivos.",
        "input_schema": {"type": "object", "properties": {"iext": {"type": "number"}}, "required": ["iext"]},
    },
    {
        "name": "restaurar_fisiologico",
        "description": "Volta Na+, K+, Ca2+ e estímulo aos valores padrão.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def estado_txt():
    s = st.session_state
    return f"Na+ {s.na} mM, K+ {s.k} mM, Ca2+ {s.ca} mM, corrente {s.iext} µA/cm²"


def run_tool(name, args):
    if name == "definir_ions":
        for key in ("na", "k", "ca"):
            if key in args:
                st.session_state[key] = ajustar(key, args[key])
    elif name == "definir_estimulo":
        st.session_state["iext"] = ajustar("iext", args["iext"])
    elif name == "restaurar_fisiologico":
        for key, val in DEFAULT.items():
            st.session_state[key] = val
    return "Estado atual: " + estado_txt()


def system_prompt():
    return (
        "Você é o agente que controla um simulador de neurônio (Hodgkin-Huxley com canais de Na+, K+ e Ca2+). "
        "O usuário vê uma animação em tempo real do fluxo de íons, do potencial de membrana e da liberação de "
        "neurotransmissor (dependente do Ca2+ intracelular). Para alterar o neurônio, SEMPRE chame as ferramentas; "
        "nunca diga que alterou sem chamá-las. Depois explique em 2-4 frases, em português, o que deve ser visto "
        "(ex.: K+ alto -> E_K menos negativo -> despolarização, possível bloqueio por despolarização; Na+ baixo -> "
        "E_Na menor -> potencial de ação menor; Ca2+ baixo -> menos influxo e menos liberação). "
        "Limites do modelo: sem bombas iônicas e sem efeito de carga de superfície do Ca2+. "
        f"Estado atual: {estado_txt()}."
    )


def get_key():
    return st.session_state.get("api_key") or os.getenv("ANTHROPIC_API_KEY")


def run_agent(texto):
    client = anthropic.Anthropic(api_key=get_key())
    msgs = st.session_state.api_msgs
    msgs.append({"role": "user", "content": texto})
    chamadas = []
    while True:
        resp = client.messages.create(
            model=MODEL, max_tokens=1000, system=system_prompt(), tools=TOOLS, messages=msgs
        )
        msgs.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                chamadas.append(f"{b.name}({json.dumps(b.input, ensure_ascii=False)})")
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b.name, b.input)})
        msgs.append({"role": "user", "content": results})
    return "".join(b.text for b in resp.content if b.type == "text"), chamadas


# O agente roda ANTES dos sliders, para poder alterar o estado deles
prompt = st.chat_input("Ex.: aumente o potássio para 12 mM")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    if not get_key():
        st.session_state.chat.append({"role": "assistant", "content": "Informe a chave da API na barra lateral."})
    else:
        try:
            txt, calls = run_agent(prompt)
            st.session_state.chat.append({"role": "assistant", "content": txt, "tools": calls})
        except Exception as e:
            st.session_state.chat.append({"role": "assistant", "content": f"Erro ao chamar a API: {e}"})

# ---------------- barra lateral ----------------
with st.sidebar:
    st.header("Sinais externos")
    st.slider("Na⁺ extracelular (mM)", *LIM["na"][:2], step=LIM["na"][2], key="na")
    st.slider("K⁺ extracelular (mM)", *LIM["k"][:2], step=LIM["k"][2], key="k")
    st.slider("Ca²⁺ extracelular (mM)", *LIM["ca"][:2], step=LIM["ca"][2], key="ca")
    st.slider("Corrente injetada (µA/cm²)", *LIM["iext"][:2], step=LIM["iext"][2], key="iext")
    st.text_input("Chave da API Anthropic", type="password", key="api_key", help="Ou defina ANTHROPIC_API_KEY")

# ---------------- animação (HH rodando no navegador) ----------------
HTML = """
<canvas id="c" width="960" height="640" style="width:100%;max-width:960px;height:auto;border-radius:12px;display:block"></canvas>
<script>
const P = __PARAMS__;
const cv=document.getElementById('c'), ctx=cv.getContext('2d'), W=960;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const RTF=26.7, ENa=50+RTF*Math.log(P.na/145), EK=-77+RTF*Math.log(P.k/4);
const EL=-54.4,gNa=120,gK=36,gL=0.3,gCa=0.1,CAI0=1e-4;
let V=-65,m=0.0529,h=0.5961,n=0.3177,mc=0,cai=CAI0,tsim=0,INa=0,IK=0,ICa=0;
const vtrap=(x,k)=>Math.abs(x/k)<1e-6?k:x/(1-Math.exp(-x/k));
function step(dt){
  const am=0.1*vtrap(V+40,10),bm=4*Math.exp(-(V+65)/18),ah=0.07*Math.exp(-(V+65)/20),bh=1/(1+Math.exp(-(V+35)/10));
  const an=0.01*vtrap(V+55,10),bn=0.125*Math.exp(-(V+65)/80);
  m+=dt*(am*(1-m)-bm*m); h+=dt*(ah*(1-h)-bh*h); n+=dt*(an*(1-n)-bn*n);
  mc+=dt*(1/(1+Math.exp(-(V+10)/7))-mc)/3;
  const ECa=13.35*Math.log(P.ca/Math.max(cai,1e-5));
  INa=gNa*m*m*m*h*(V-ENa); IK=gK*n*n*n*n*(V-EK); ICa=gCa*mc*mc*(V-ECa);
  V+=dt*(P.iext-INa-IK-gL*(V-EL)-ICa);
  cai+=dt*(-0.00005*ICa-(cai-CAI0)/50);
  if(!isFinite(V)){V=-65;m=.05;h=.6;n=.32;mc=0;cai=CAI0;}
}
const rel=c=>{const a=Math.pow(c,4),b=Math.pow(0.002,4);return a/(a+b);};
const COL={Na:'#38bdf8',K:'#4ade80',Ca:'#fb923c'}, CH={Na:120,K:270,Ca:420};
const memColor=v=>{const t=clamp((v+80)/120,0,1);return `hsl(${220-200*t},80%,${40+10*t}%)`;};

// íons de fundo (quantidade proporcional à concentração)
const bg=[],flux=[],pulses=[],nts=[],trace=[],spikeT=[];
function addBg(type,side,cnt){for(let i=0;i<cnt;i++){const o=side==='o';
  bg.push({type,x:10+Math.random()*600,y:o?36+Math.random()*128:206+Math.random()*144,
  vx:Math.random()-.5,vy:Math.random()-.5,y0:o?36:206,y1:o?164:350});}}
addBg('Na','o',Math.round(P.na/145*26));addBg('K','o',clamp(Math.round(P.k/4*3),1,45));
addBg('Ca','o',Math.round(P.ca/2*8));addBg('Na','i',7);addBg('K','i',24);addBg('Ca','i',1);
function ion(x,y,type,a){ctx.globalAlpha=a;ctx.fillStyle=COL[type];ctx.beginPath();ctx.arc(x,y,5,0,7);ctx.fill();ctx.globalAlpha=1;}
function spawn(type,I,scale){let e=Math.min(Math.abs(I)/scale,5);const d=I<0?1:-1;
  while(e>0&&flux.length<250){if(Math.random()<Math.min(e,1))
    flux.push({x:CH[type]+(Math.random()-.5)*8,y:d>0?120:250,vy:d*(3+Math.random()*1.5),type});e--;}}
function channel(x,gap,col,label){
  ctx.fillStyle='#1e293b';ctx.strokeStyle=col;ctx.lineWidth=2;
  for(const bx of [x-28,x+gap/2]){ctx.fillRect(bx,164,28-gap/2,42);ctx.strokeRect(bx,164,28-gap/2,42);}
  ctx.fillStyle=col;ctx.font='bold 12px sans-serif';ctx.textAlign='center';ctx.fillText(label,x,226);ctx.textAlign='left';}
function onSpike(){spikeT.push(tsim);pulses.push({x:194});}
function release(){const k=Math.round(rel(cai)*16);
  for(let i=0;i<k;i++)nts.push({x:845,y:510+(Math.random()-.5)*20,vx:1.2+Math.random()*1.6,vy:(Math.random()-.5)*2.4,life:38});}
function bar(x,y,w,f,col,lab){ctx.fillStyle='#94a3b8';ctx.font='12px sans-serif';ctx.fillText(lab,x,y-6);
  ctx.fillStyle='#1e293b';ctx.fillRect(x,y,w,14);ctx.fillStyle=col;ctx.fillRect(x,y,w*clamp(f,0,1),14);}

function frame(){
  let mxNa=0,mxK=0,mxCa=0;
  for(let i=0;i<20;i++){const pv=V;step(0.025);tsim+=0.025;if(pv<=0&&V>0)onSpike();
    if(Math.abs(INa)>Math.abs(mxNa))mxNa=INa;if(Math.abs(IK)>Math.abs(mxK))mxK=IK;if(Math.abs(ICa)>Math.abs(mxCa))mxCa=ICa;
    if(i%4===0)trace.push([tsim,V]);}
  while(trace.length&&tsim-trace[0][0]>150)trace.shift();
  while(spikeT.length&&tsim-spikeT[0]>200)spikeT.shift();
  spawn('Na',mxNa,500);spawn('K',mxK,350);spawn('Ca',mxCa,5);

  ctx.fillStyle='#0b1220';ctx.fillRect(0,0,W,640);
  ctx.fillStyle='#0f1f3a';ctx.fillRect(0,0,620,170);ctx.fillStyle='#0d2a2a';ctx.fillRect(0,200,620,160);
  ctx.fillStyle=memColor(V);ctx.fillRect(0,170,620,30);
  ctx.fillStyle='#64748b';ctx.font='11px sans-serif';ctx.fillText('EXTRACELULAR',10,24);ctx.fillText('INTRACELULAR',10,354);
  channel(CH.Na,4+40*clamp(m*m*m*h*2.5,0,1),COL.Na,'Na⁺');
  channel(CH.K,4+40*clamp(Math.pow(n,4)*1.6,0,1),COL.K,'K⁺');
  channel(CH.Ca,4+40*clamp(mc*mc,0,1),COL.Ca,'Ca²⁺');
  for(const p of bg){p.x+=p.vx;p.y+=p.vy;p.vx=clamp(p.vx+(Math.random()-.5)*.4,-1.2,1.2);p.vy=clamp(p.vy+(Math.random()-.5)*.4,-1.2,1.2);
    if(p.x<8||p.x>612)p.vx*=-1;if(p.y<p.y0||p.y>p.y1)p.vy*=-1;p.x=clamp(p.x,8,612);p.y=clamp(p.y,p.y0,p.y1);ion(p.x,p.y,p.type,.55);}
  for(let i=flux.length-1;i>=0;i--){const f=flux[i];f.y+=f.vy;
    if((f.vy>0&&f.y>260)||(f.vy<0&&f.y<110)){flux.splice(i,1);continue;}
    ctx.shadowBlur=10;ctx.shadowColor=COL[f.type];ion(f.x,f.y,f.type,1);ctx.shadowBlur=0;}

  // gráfico de Vm
  const x0=650,y0=40,w=290,hh=160,yv=v=>y0+hh*(1-(v+90)/150);
  ctx.fillStyle='#e2e8f0';ctx.font='bold 13px sans-serif';ctx.fillText('Potencial de membrana: '+V.toFixed(1)+' mV',x0,28);
  ctx.fillStyle='#0f172a';ctx.fillRect(x0,y0,w,hh);ctx.strokeStyle='#334155';ctx.lineWidth=1;ctx.strokeRect(x0,y0,w,hh);
  ctx.setLineDash([4,4]);ctx.font='10px sans-serif';
  for(const v of [0,-65]){ctx.beginPath();ctx.moveTo(x0,yv(v));ctx.lineTo(x0+w,yv(v));ctx.stroke();ctx.fillStyle='#64748b';ctx.fillText(v+' mV',x0+3,yv(v)-3);}
  ctx.setLineDash([]);ctx.beginPath();ctx.strokeStyle='#f8fafc';ctx.lineWidth=1.5;
  trace.forEach((p,i)=>{const x=x0+(p[0]-(tsim-150))/150*w,y=yv(clamp(p[1],-90,60));i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  ctx.fillStyle='#94a3b8';ctx.font='11px sans-serif';
  ctx.fillText('E_Na '+ENa.toFixed(0)+' mV · E_K '+EK.toFixed(0)+' mV · '+(spikeT.length/0.2).toFixed(0)+' Hz',x0,218);
  bar(x0,250,w,cai/0.004,'#fb923c','Ca²⁺ intracelular: '+(cai*1000).toFixed(2)+' µM');
  const pr=rel(cai);bar(x0,300,w,pr,'#f472b6','Prob. de liberação do neurotransmissor: '+(pr*100).toFixed(0)+'%');
  let st='Repouso';
  if(V>-25)st='Potencial de ação';else if(spikeT.length)st='Disparando';
  else if(tsim>200&&V>-50)st='Bloqueio por despolarização';else if(V>-55)st='Despolarizado';else if(V<-72)st='Hiperpolarizado';
  ctx.fillStyle='#fde68a';ctx.font='bold 13px sans-serif';ctx.fillText('Estado: '+st,x0,344);

  // neurônio e sinapse
  ctx.lineCap='round';ctx.lineWidth=4;ctx.strokeStyle=V>-30?'#fbbf24':'#475569';
  for(const d of [[112,480,60,440],[60,440,30,415],[60,440,40,458],[112,540,70,585],[70,585,40,600],[125,468,125,425],[125,425,100,400],[125,425,150,402]]){
    ctx.beginPath();ctx.moveTo(d[0],d[1]);ctx.lineTo(d[2],d[3]);ctx.stroke();}
  ctx.lineWidth=6;ctx.strokeStyle='#475569';ctx.beginPath();ctx.moveTo(194,510);ctx.lineTo(810,510);ctx.stroke();
  ctx.shadowBlur=V>-30?25:0;ctx.shadowColor='#fbbf24';ctx.fillStyle=memColor(V);
  ctx.beginPath();ctx.arc(150,510,44,0,7);ctx.fill();ctx.shadowBlur=0;
  ctx.fillStyle='#334155';ctx.beginPath();ctx.arc(830,510,22,0,7);ctx.fill();ctx.fillRect(895,470,12,80);
  for(let i=pulses.length-1;i>=0;i--){const q=pulses[i];q.x+=14;
    const g=ctx.createRadialGradient(q.x,510,0,q.x,510,20);g.addColorStop(0,'rgba(250,204,21,.95)');g.addColorStop(1,'rgba(250,204,21,0)');
    ctx.fillStyle=g;ctx.beginPath();ctx.arc(q.x,510,20,0,7);ctx.fill();
    if(q.x>=815){pulses.splice(i,1);release();}}
  ctx.fillStyle='#f472b6';
  for(let i=nts.length-1;i>=0;i--){const t=nts[i];t.x+=t.vx;t.y+=t.vy;
    if(--t.life<=0||t.x>893){nts.splice(i,1);continue;}ctx.beginPath();ctx.arc(t.x,t.y,3,0,7);ctx.fill();}
  ctx.fillStyle='#64748b';ctx.font='11px sans-serif';ctx.textAlign='center';
  ctx.fillText('Soma',150,572);ctx.fillText('Axônio',500,540);ctx.fillText('Terminal',830,550);ctx.fillText('Pós-sináptico',900,575);
  ctx.textAlign='left';
  [['Na⁺',COL.Na],['K⁺',COL.K],['Ca²⁺',COL.Ca],['Neurotransmissor','#f472b6']].forEach((l,i)=>{
    ctx.fillStyle=l[1];ctx.beginPath();ctx.arc(20+i*90+(i===3?0:0),625,5,0,7);ctx.fill();ctx.fillStyle='#94a3b8';ctx.fillText(l[0],30+i*90,629);});
  requestAnimationFrame(frame);
}
frame();
</script>
"""

params = {k: float(st.session_state[k]) for k in DEFAULT}
col_anim, col_chat = st.columns([3, 1.2])

with col_anim:
    st.subheader("Neurônio em tempo real")
    components.html(HTML.replace("__PARAMS__", json.dumps(params)), height=660)
    st.caption("Ao mudar um sinal (slider ou agente) a animação reinicia com os novos parâmetros.")

with col_chat:
    st.subheader("Agente")
    box = st.container(height=560)
    with box:
        if not st.session_state.chat:
            st.info("Peça, por exemplo: “simule hipercalemia”, “reduza o cálcio para 0,5 mM” ou “volte ao normal”.")
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("tools"):
                    st.caption("⚙️ " + " · ".join(msg["tools"]))
