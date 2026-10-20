"""Publish to website: a self-contained folder with a WebGL 360 viewer for a panorama."""
from __future__ import annotations

import html
import os

import cv2
import numpy as np

from .imageio import load_image, save_image

VIEWER_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>html,body{margin:0;height:100%;background:#000;overflow:hidden;font:13px system-ui,sans-serif;color:#ddd}
canvas{width:100%;height:100%;display:block;touch-action:none;cursor:grab}
#hud{position:fixed;left:12px;bottom:10px;opacity:.75;text-shadow:0 1px 2px #000}</style></head>
<body><canvas id="c"></canvas><div id="hud">__TITLE__ - drag to look around, scroll to zoom</div>
<script>
const cv=document.getElementById('c'),gl=cv.getContext('webgl');
const vs='attribute vec2 p;varying vec2 q;void main(){q=p;gl_Position=vec4(p,0.,1.);}';
const fs=`precision highp float;varying vec2 q;uniform sampler2D t;uniform float yaw,pitch,f,asp;uniform vec2 ext;
void main(){vec3 d=normalize(vec3(q.x*asp*f,q.y*f,1.));
float cp=cos(pitch),sp=sin(pitch);d=vec3(d.x,d.y*cp+d.z*sp,-d.y*sp+d.z*cp);
float cy=cos(yaw),sy=sin(yaw);d=vec3(d.x*cy+d.z*sy,d.y,-d.x*sy+d.z*cy);
float lon=atan(d.x,d.z),lat=asin(clamp(d.y,-1.,1.));
vec2 uv=vec2(lon/ext.x+.5,.5-lat/ext.y);
if(uv.x<0.||uv.x>1.||uv.y<0.||uv.y>1.){gl_FragColor=vec4(0,0,0,1);return;}
gl_FragColor=texture2D(t,uv);}`;
function sh(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);return s}
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(pr);gl.useProgram(pr);
const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
const loc=gl.getAttribLocation(pr,'p');gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
const U=n=>gl.getUniformLocation(pr,n);let yaw=0,pitch=0,fov=1.4;
const EXT=[__HFOV__*Math.PI/180,__VFOV__*Math.PI/180];
const img=new Image();img.onload=()=>{const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,img);
gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);draw()};img.src='__IMAGE__';
function draw(){const w=cv.clientWidth*devicePixelRatio,h=cv.clientHeight*devicePixelRatio;if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h}
gl.viewport(0,0,w,h);gl.uniform1f(U('yaw'),yaw);gl.uniform1f(U('pitch'),pitch);gl.uniform1f(U('f'),Math.tan(fov/2));
gl.uniform1f(U('asp'),w/h);gl.uniform2f(U('ext'),EXT[0],EXT[1]);gl.drawArrays(gl.TRIANGLES,0,3)}
let drag=null;cv.onpointerdown=e=>{drag=[e.clientX,e.clientY,yaw,pitch];cv.setPointerCapture(e.pointerId)};
cv.onpointermove=e=>{if(!drag)return;const s=Math.tan(fov/2)*2/cv.clientHeight;yaw=drag[2]-(e.clientX-drag[0])*s;
pitch=Math.max(-1.55,Math.min(1.55,drag[3]+(e.clientY-drag[1])*s));draw()};cv.onpointerup=()=>drag=null;
cv.onwheel=e=>{e.preventDefault();fov=Math.max(.2,Math.min(2.6,fov*Math.exp(e.deltaY*.001)));draw()};addEventListener('resize',draw);
</script></body></html>"""


def publish_viewer(pano_path: str, folder: str | None = None, title: str = "Panorama", max_width: int = 8192,
                   hfov: float = 360.0, vfov: float = 180.0) -> str:
    folder = folder or os.path.splitext(pano_path)[0] + "_web"
    os.makedirs(folder, exist_ok=True)
    img = load_image(pano_path).rgb
    if img.shape[1] > max_width:
        s = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    save_image(os.path.join(folder, "panorama.jpg"), np.clip(img, 0, 1), quality=90)
    page = (VIEWER_HTML.replace("__TITLE__", html.escape(title)).replace("__IMAGE__", "panorama.jpg")
            .replace("__HFOV__", f"{hfov:.6g}").replace("__VFOV__", f"{vfov:.6g}"))
    with open(os.path.join(folder, "index.html"), "w") as f:
        f.write(page)
    return folder
