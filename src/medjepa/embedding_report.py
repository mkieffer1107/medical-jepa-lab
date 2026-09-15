"""Offline, interactive exploration of encoder PCA coordinates."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from plotly.offline import get_plotlyjs


def class_colors(count: int) -> list[str]:
    palette = plt.get_cmap("tab20" if count > 10 else "tab10")
    return [to_hex(palette(i)) for i in range(count)]


def plot_3d(coordinates, labels, class_names, title: str, output: Path, prefix="PC") -> None:
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    for class_id, (name, color) in enumerate(zip(class_names, class_colors(len(class_names)))):
        selected = labels == class_id
        if selected.any():
            axis.scatter(*coordinates[selected].T, s=14, alpha=0.75, color=color, label=name)
    axis.set(xlabel=f"{prefix} 1", ylabel=f"{prefix} 2", zlabel=f"{prefix} 3", title=title)
    axis.legend(title="Class", loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_interactive_report(coordinates: np.ndarray, tsne2: np.ndarray, tsne3: np.ndarray, labels: np.ndarray,
                             class_names: list[str], variance: np.ndarray,
                             run_name: str, checkpoint: str, output: Path, *,
                             samples: dict | None = None, clusters: dict | None = None,
                             assignments: np.ndarray | None = None) -> None:
    groups = []
    for class_id, (name, color) in enumerate(zip(class_names, class_colors(len(class_names)))):
        selected = labels == class_id
        if selected.any():
            groups.append({"name": name, "color": color,
                           "pca": coordinates[selected].tolist(),
                           "tsne2": tsne2[selected].tolist(), "tsne3": tsne3[selected].tolist(),
                           "indices": np.flatnonzero(selected).tolist()})
    payload = json.dumps({"groups": groups, "variance": variance.tolist(),
                          "samples": samples, "clusters": clusters, "class_names": class_names,
                          "assignments": assignments.tolist() if assignments is not None else []}, allow_nan=False)
    # Keep user-controlled labels and paths out of executable HTML/script contexts.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Encoder embeddings · __RUN__</title>
<style>
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:#f6f7f9;color:#172538;font:15px/1.5 system-ui,-apple-system,sans-serif}
header{padding:26px 36px 20px;border-bottom:1px solid #dce2e9;background:#fff}
.eyebrow{text-transform:uppercase;letter-spacing:.16em;color:#53677e;font-size:11px;font-weight:700}
h1{font-size:28px;letter-spacing:-.04em;margin:6px 0 3px;font-weight:650}.run{color:#5a687a;overflow-wrap:anywhere}
main{padding:22px 36px}nav{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.tabs{display:flex;gap:3px;background:#e6ebf1;padding:4px;border-radius:9px}
button{font:inherit;font-weight:600;border:0;border-radius:6px;padding:8px 16px;cursor:pointer;background:transparent;color:#53677e;transition:background .15s,color .15s}
button:hover{background:#dce4ed}button[aria-pressed=true]{background:#fff;color:#173d70;box-shadow:0 1px 4px #132a4414}
button:focus-visible{outline:3px solid #619dea;outline-offset:2px}.reset{border:1px solid #cbd5e1;background:#fff}.stats{margin-left:auto;color:#53677e;font-variant-numeric:tabular-nums}
.workspace{position:sticky;top:10px;z-index:5;background:white;border-radius:10px;overflow:hidden}.plot{width:100%;height:52vh;min-height:340px}
.hint{padding:12px 18px;border-top:1px solid #edf0f4;color:#64748b;font-size:13px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
footer{margin-top:17px;color:#657489;font-size:12px}details{margin-top:9px}summary{cursor:pointer}.path{overflow-wrap:anywhere;font-family:monospace;margin:6px 0}
@media(max-width:640px){header{padding:20px}main{padding:16px 10px}h1{font-size:24px}.stats{margin-left:0;width:100%}.plot{height:65vh;min-height:450px}}
@media(prefers-reduced-motion:reduce){button{transition:none}}
.plot-stage{display:flex;align-items:stretch}.plots{flex:1;min-width:0}.sample-preview{width:230px;padding:22px 18px;border-left:1px solid #edf0f4;font-size:13px}.sample-preview h2{font-size:15px;margin:0 0 8px}.sample-preview img{width:100%;height:144px;object-fit:contain;background:#f1f4f7;border-radius:8px;margin:14px 0}.sample-preview p{overflow-wrap:anywhere}.sample-preview a{display:block;color:#175da6;margin:8px 0}.sample-preview button{padding:6px 8px;font-size:12px}
.cluster-section{margin-top:32px}.cluster-section h2{font-size:23px;letter-spacing:-.03em;margin:0 0 5px}.cluster-section p{color:#5a687a}.cluster-list{list-style:none;padding:0}.cluster-row{padding:22px 0;border-top:1px solid #dce2e9}.cluster-row[data-selected=true]{border-top-color:#2473be}.cluster-heading{display:flex;align-items:center;flex-wrap:wrap;gap:12px}.cluster-heading button{padding:5px 8px 5px 0;color:#193e6b;text-align:left}.cluster-heading span{color:#657489;font-size:13px}.examples{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:14px;max-width:1100px;margin-top:14px}.example{margin:0;min-width:0}.example button{display:block;padding:0;width:100%;overflow:hidden;border-radius:8px;background:#e9edf2}.example img{display:block;width:100%;height:125px;object-fit:contain}.example figcaption{font-size:12px;color:#5a687a;margin-top:6px}.example a{font-size:12px;color:#175da6}.example strong{display:block;color:#273b50}.sample-id{font:11px/1.5 monospace;overflow-wrap:anywhere}.empty{color:#64748b}
@media(max-width:760px){.workspace{position:static}.plot-stage{display:block}.sample-preview{width:auto;border-left:0;border-top:1px solid #edf0f4}.sample-preview img{width:125px;float:left;margin:4px 15px 10px 0}.sample-preview:after{content:'';display:block;clear:both}.examples{grid-template-columns:repeat(3,minmax(0,1fr))}.example img{height:100px}}
</style><script>__PLOTLY__</script></head>
<body><header><div class="eyebrow">Medical JEPA Lab / representation explorer</div><h1>Encoder embeddings</h1><div class="run">__RUN__ · test split · __COUNT__ samples</div></header>
<main><nav aria-label="Projection controls"><div class="tabs"><button id="method-pca" aria-pressed="true">PCA</button><button id="method-tsne" aria-pressed="false">t-SNE</button></div><div class="tabs"><button id="view2" aria-pressed="false">2D</button><button id="view3" aria-pressed="true">3D</button></div><button class="reset" id="reset">Reset view</button><span class="stats" id="variance"></span></nav>
<section class="workspace" aria-label="Embedding plot"><div class="plot-stage"><div class="plots"><div id="plot3" class="plot" role="img" aria-label="Rotatable 3D scatter plot colored by class"></div><div id="plot2" class="plot" hidden role="img" aria-label="2D scatter plot colored by class"></div></div><aside class="sample-preview" aria-label="Sample preview"><h2>Sample inspector</h2><p id="sample-help">Hover a point to preview its image. Click a point to pin it.</p><img id="sample-image" hidden alt=""><p id="sample-title" class="sample-id"></p><p id="sample-meta"></p><a id="sample-download" hidden>Download PNG</a><a id="sample-source" hidden target="_blank" rel="noopener noreferrer">Source archive</a><button id="sample-unpin" hidden>Unpin sample</button></aside></div><div class="hint"><span id="gesture">Drag to rotate · scroll to zoom · hover to inspect</span><span>Click a class to toggle · double-click to isolate</span></div></section>
<section class="cluster-section" id="cluster-section" aria-label="K-means clusters"><h2>Explore clusters</h2><p id="cluster-description"></p><button class="reset" id="clear-cluster">Show all clusters</button><ol class="cluster-list" id="cluster-list"></ol></section>
<footer>Each point is one test image encoded by the saved model. Colors show its class; labels do not determine the projection. PCA axes are fitted on encoder embeddings from the training split. t-SNE is fitted separately in 2D and 3D on test embeddings; distances between clusters and cluster sizes need not reflect the original space. A 2D/3D projection shows only part of the representation; visual clusters alone do not establish accuracy.
<details><summary>Checkpoint used</summary><div class="path">__CHECKPOINT__</div></details></footer></main>
<script type="application/json" id="embedding-data">__DATA__</script><script>
const data=JSON.parse(document.getElementById('embedding-data').textContent);
let active=3,method='pca',highlighted=null,pinnedCluster=null,pinnedSample=false;const initialCamera={eye:{x:1.5,y:1.5,z:1.15}};
const axis=i=>({title:{text:method==='pca'?`PC ${i+1} (${(100*data.variance[i]).toFixed(1)}%)`:`t-SNE ${i+1}`},gridcolor:'#e5eaf0',zerolinecolor:'#ccd5df',showbackground:false});
const config={responsive:true,displaylogo:false,scrollZoom:true,toImageButtonOptions:{format:'png',filename:'encoder-embeddings',scale:2}};
function traces(dim){return data.groups.map(g=>{const points=method==='pca'?g.pca:g['tsne'+dim];return ({type:dim===3?'scatter3d':'scattergl',mode:'markers',name:g.name,
 x:points.map(p=>p[0]),y:points.map(p=>p[1]),...(dim===3?{z:points.map(p=>p[2])}:{}),
 customdata:g.indices,marker:{color:g.indices.map(i=>highlighted===null||data.assignments[i]===highlighted?g.color:'rgba(170,178,188,0.13)'),size:dim===3?4:6,opacity:.8},
 hovertemplate:'Test sample %{customdata}<br>Axis 1: %{x:.3f}<br>Axis 2: %{y:.3f}'+(dim===3?'<br>Axis 3: %{z:.3f}':'')+'<extra>'+g.name.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')+'</extra>'});});}
async function show(dim){active=dim;
 for(const d of [2,3]){document.getElementById('plot'+d).hidden=d!==dim;document.getElementById('view'+d).setAttribute('aria-pressed',String(d===dim));}
 document.getElementById('variance').textContent=method==='tsne'?'Local neighborhoods · independent '+dim+'D fit':(data.variance.slice(0,dim).reduce((a,b)=>a+b,0)*100).toFixed(1)+'% variance explained';
 document.getElementById('gesture').textContent=dim===3?'Drag to rotate · scroll to zoom · hover to inspect':'Drag to select · scroll to zoom · hover to inspect';
 const el=document.getElementById('plot'+dim);
 await Plotly.react(el,traces(dim),{paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'system-ui, sans-serif',color:'#34465c'},margin:{l:65,r:20,t:25,b:65},legend:{title:{text:'Class'},orientation:'h',y:-.12,x:0},xaxis:axis(0),yaxis:axis(1),scene:{xaxis:axis(0),yaxis:axis(1),zaxis:axis(2),camera:initialCamera,aspectmode:'data',dragmode:'orbit'},uirevision:method+dim},config);
 if(!el._sampleHandlers){
 el.on('plotly_hover',event=>{if(!pinnedSample)showSample(event.points[0].customdata);});
 el.on('plotly_click',event=>{pinnedSample=true;showSample(event.points[0].customdata);document.getElementById('sample-unpin').hidden=false;});
 el._sampleHandlers=true;}
 Plotly.Plots.resize(el);
}
for(const m of ['pca','tsne'])document.getElementById('method-'+m).onclick=()=>{method=m;for(const other of ['pca','tsne'])document.getElementById('method-'+other).setAttribute('aria-pressed',String(m===other));show(active);};
document.getElementById('view2').onclick=()=>show(2);document.getElementById('view3').onclick=()=>show(3);
document.getElementById('reset').onclick=()=>Plotly.relayout('plot'+active,active===3?{'scene.camera':initialCamera}:{'xaxis.autorange':true,'yaxis.autorange':true});
const imageURLs=new Map();
function imageURL(index){
 if(imageURLs.has(index))return imageURLs.get(index);
 const sample=data.samples?.samples[index];if(!sample)return null;
 const bytes=Uint8Array.from(atob(sample.png),c=>c.charCodeAt(0));
 const url=URL.createObjectURL(new Blob([bytes],{type:'image/png'}));imageURLs.set(index,url);return url;
}
window.addEventListener('pagehide',()=>{for(const url of imageURLs.values())URL.revokeObjectURL(url);imageURLs.clear();});
window.addEventListener('pageshow',event=>{if(event.persisted){for(const img of document.querySelectorAll('img[data-index]'))if(img.src)img.src=imageURL(Number(img.dataset.index));const index=document.getElementById('sample-image').dataset.index;if(index!==undefined)showSample(Number(index));}});
function showSample(index){
 const sample=data.samples?.samples[index];if(!sample)return;
 const img=document.getElementById('sample-image');img.src=imageURL(index);img.dataset.index=index;img.alt=sample.id;img.hidden=false;
 document.getElementById('sample-title').textContent=sample.id;
 document.getElementById('sample-meta').textContent=`${data.class_names[sample.label]} · test row ${sample.row} · Cluster ${data.assignments[index]+1}`;
 const link=document.getElementById('sample-download');link.href=img.src;link.download=sample.id+'.png';link.hidden=false;
 const source=document.getElementById('sample-source');
 if(data.samples.archive_url?.startsWith('https://')){source.href=data.samples.archive_url;source.hidden=false;}
}
document.getElementById('sample-unpin').onclick=()=>{pinnedSample=false;document.getElementById('sample-unpin').hidden=true;};
function highlightCluster(id){
 highlighted=id;
 const el=document.getElementById('plot'+active);if(!el.data)return;
 Plotly.restyle(el,{'marker.color':data.groups.map(g=>g.indices.map(i=>id===null||data.assignments[i]===id?g.color:'rgba(170,178,188,0.13)'))});
 for(const row of document.querySelectorAll('.cluster-row'))row.dataset.selected=String(Number(row.dataset.cluster)===id);
}
document.getElementById('clear-cluster').onclick=()=>{pinnedCluster=null;highlightCluster(null);for(const b of document.querySelectorAll('.cluster-select'))b.setAttribute('aria-pressed','false');};
function make(tag,text,cls){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el;}
const thumbnails=new IntersectionObserver(entries=>{for(const item of entries)if(item.isIntersecting){const img=item.target;img.src=imageURL(Number(img.dataset.index));thumbnails.unobserve(img);}}, {rootMargin:'200px'});
if(data.clusters&&data.samples){
 document.getElementById('cluster-description').textContent=`${data.clusters.actual_k} K-means clusters in full encoder space (Euclidean distance). Hover or focus a cluster to fade the others; click to keep it selected. Examples show the nearest and farthest members from its centroid, followed by diverse members.`;
 for(const cluster of data.clusters.clusters){
 const row=make('li',undefined,'cluster-row');row.dataset.cluster=cluster.id;
 const heading=make('div',undefined,'cluster-heading');const select=make('button','Cluster '+(cluster.id+1),'cluster-select');select.setAttribute('aria-pressed','false');
 select.onclick=()=>{pinnedCluster=pinnedCluster===cluster.id?null:cluster.id;highlightCluster(pinnedCluster);for(const b of document.querySelectorAll('.cluster-select'))b.setAttribute('aria-pressed',String(b===select&&pinnedCluster!==null));};
 heading.append(select,make('span',cluster.size+(cluster.size===1?' sample':' samples')),make('span',cluster.class_counts.map(c=>c.name+': '+c.count).join(' · ')));row.append(heading);
 row.onmouseenter=()=>highlightCluster(cluster.id);row.onmouseleave=()=>highlightCluster(pinnedCluster);
 row.onfocusin=()=>highlightCluster(cluster.id);row.onfocusout=event=>{if(!row.contains(event.relatedTarget))highlightCluster(pinnedCluster);};
 const examples=make('div',undefined,'examples');
 for(const example of cluster.examples){
 const sample=data.samples.samples[example.index];const figure=make('figure',undefined,'example');const button=make('button');button.setAttribute('aria-label','Inspect '+sample.id);
 const img=make('img');img.alt=sample.id;img.dataset.index=example.index;img.loading='lazy';button.append(img);thumbnails.observe(img);
 button.onmouseenter=()=>{if(!pinnedSample)showSample(example.index);};button.onfocus=()=>{if(!pinnedSample)showSample(example.index);};button.onclick=()=>{pinnedSample=true;showSample(example.index);document.getElementById('sample-unpin').hidden=false;};
 const caption=make('figcaption');caption.append(make('strong',example.role),make('span',data.class_names[sample.label]+' · distance '+example.distance.toFixed(3)),make('div',sample.id,'sample-id'));
 const link=make('a','Download PNG');link.href='#';link.download=sample.id+'.png';link.onclick=()=>{link.href=imageURL(example.index);};figure.append(button,caption,link);examples.append(figure);
 }
 row.append(examples);document.getElementById('cluster-list').append(row);
 }
}else{document.getElementById('cluster-section').hidden=true;}

show(3);
</script></body></html>'''
    # Replace in one pass so placeholder-like run names remain literal text.
    import re
    values = {"__RUN__": html.escape(run_name), "__COUNT__": f"{len(labels):,}",
              "__CHECKPOINT__": html.escape(checkpoint), "__DATA__": payload,
              "__PLOTLY__": get_plotlyjs()}
    output.write_text(re.sub(r"__(?:RUN|COUNT|CHECKPOINT|DATA|PLOTLY)__", lambda m: values[m[0]], template), encoding="utf-8")
