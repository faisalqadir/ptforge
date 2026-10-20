// Full-screen viewer for stitched results: interactive 360 view for equirectangular
// panoramas, pan/zoom for everything else.
import { h } from './ui.js';

const VS = `#version 300 es
in vec2 aPos; out vec2 q; void main(){ q = aPos; gl_Position = vec4(aPos, 0.0, 1.0); }`;
const FS = `#version 300 es
precision highp float; in vec2 q; out vec4 o;
uniform sampler2D t; uniform float yaw, pitch, f, asp; uniform vec2 ext, off;
void main(){
  vec3 d = normalize(vec3(q.x*asp*f, q.y*f, 1.0));
  float cp = cos(pitch), sp = sin(pitch); d = vec3(d.x, d.y*cp + d.z*sp, -d.y*sp + d.z*cp);
  float cy = cos(yaw), sy = sin(yaw); d = vec3(d.x*cy + d.z*sy, d.y, -d.x*sy + d.z*cy);
  float lon = atan(d.x, d.z), lat = asin(clamp(d.y, -1.0, 1.0));
  vec2 uv = vec2((lon - off.x)/ext.x + 0.5, 0.5 - (lat - off.y)/ext.y);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { o = vec4(0.05, 0.05, 0.06, 1.0); return; }
  o = texture(t, uv);
}`;

export function openViewer(url, { projection = 'equirectangular', hfov = 360, vfov = 180, offLon = 0, offLat = 0, title = '' } = {}) {
  const back = h('div', { class: 'viewer-back' });
  const close = () => { back.remove(); document.removeEventListener('keydown', onKey); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  document.addEventListener('keydown', onKey);
  back.append(h('button', { class: 'btn close', onclick: close }, 'Close (Esc)'));
  document.body.append(back);
  if (projection !== 'equirectangular') return flatViewer(back, url, title);
  const cv = h('canvas');
  back.prepend(cv);
  back.append(h('div', { class: 'hud' }, `${title}  —  drag to look around, scroll to zoom`));
  const gl = cv.getContext('webgl2');
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const pr = gl.createProgram();
  gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS));
  gl.bindAttribLocation(pr, 0, 'aPos'); gl.linkProgram(pr); gl.useProgram(pr);
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const U = (n) => gl.getUniformLocation(pr, n);
  // offLon/offLat: centre of a cropped panorama window (degrees)
  let yaw = offLon * Math.PI / 180, pitch = offLat * Math.PI / 180, fov = 1.5, ready = false;
  const draw = () => {
    if (!back.isConnected) return;
    const w = cv.clientWidth * devicePixelRatio, hh = cv.clientHeight * devicePixelRatio;
    if (cv.width !== w || cv.height !== hh) { cv.width = w; cv.height = hh; }
    gl.viewport(0, 0, w, hh);
    if (!ready) { gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT); return; }
    gl.uniform1f(U('yaw'), yaw); gl.uniform1f(U('pitch'), pitch); gl.uniform1f(U('f'), Math.tan(fov / 2));
    gl.uniform1f(U('asp'), w / hh); gl.uniform2f(U('ext'), hfov * Math.PI / 180, vfov * Math.PI / 180);
    gl.uniform2f(U('off'), offLon * Math.PI / 180, offLat * Math.PI / 180);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
  const img = new Image();
  img.onload = () => {
    const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    ready = true; draw();
  };
  img.src = url;
  let drag = null;
  cv.onpointerdown = (e) => { drag = [e.clientX, e.clientY, yaw, pitch]; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; };
  cv.onpointermove = (e) => {
    if (!drag) return;
    const s = Math.tan(fov / 2) * 2 / cv.clientHeight;
    yaw = drag[2] - (e.clientX - drag[0]) * s;
    pitch = Math.max(-1.55, Math.min(1.55, drag[3] + (e.clientY - drag[1]) * s));
    draw();
  };
  cv.onpointerup = () => { drag = null; cv.style.cursor = 'grab'; };
  cv.onwheel = (e) => { e.preventDefault(); fov = Math.max(0.15, Math.min(2.7, fov * Math.exp(e.deltaY * 0.001))); draw(); };
  new ResizeObserver(draw).observe(cv);
}

function flatViewer(back, url, title) {
  const img = h('img', { src: url, style: { position: 'absolute', left: '0', top: '0', transformOrigin: '0 0', maxWidth: 'none' } });
  const box = h('div', { style: { position: 'absolute', inset: '0', overflow: 'hidden', cursor: 'grab' } }, img);
  back.prepend(box);
  back.append(h('div', { class: 'hud' }, `${title}  —  drag to pan, scroll to zoom`));
  let s = 1, x = 0, y = 0, drag = null;
  const apply = () => { img.style.transform = `translate(${x}px, ${y}px) scale(${s})`; };
  img.onload = () => {
    s = Math.min(box.clientWidth / img.naturalWidth, box.clientHeight / img.naturalHeight);
    x = (box.clientWidth - img.naturalWidth * s) / 2; y = (box.clientHeight - img.naturalHeight * s) / 2; apply();
  };
  box.onpointerdown = (e) => { drag = [e.clientX - x, e.clientY - y]; box.setPointerCapture(e.pointerId); };
  box.onpointermove = (e) => { if (drag) { x = e.clientX - drag[0]; y = e.clientY - drag[1]; apply(); } };
  box.onpointerup = () => { drag = null; };
  box.onwheel = (e) => {
    e.preventDefault();
    const k = Math.exp(-e.deltaY * 0.0015);
    x = e.clientX - (e.clientX - x) * k; y = e.clientY - (e.clientY - y) * k; s *= k; apply();
  };
}
