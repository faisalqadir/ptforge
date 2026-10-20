// WebGL2 real-time panorama renderer: re-implements the camera model and projections
// of panoforge/geometry.py in GLSL and blends all source images with feathered weights.
import { PROJ_IDS, canvasInfo, cameraFor, transpose } from './geom.js';

const VS = `#version 300 es
in vec2 aPos; out vec2 vNdc;
void main(){ vNdc = aPos; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const COMMON = `#version 300 es
precision highp float;
const float PI = 3.141592653589793;
uniform int uProj; uniform vec2 uProjP; // panini d, vcompress
vec3 fromLL(float lon, float lat){ float c = cos(lat); return vec3(sin(lon)*c, sin(lat), cos(lon)*c); }
float vfun(float lat){ return (1.0-uProjP.y)*tan(lat) + uProjP.y*lat; }
float vinv(float y){
  if (uProjP.y <= 0.0) return atan(y);
  float lat = uProjP.y < 1.0 ? atan(y) : y;
  for (int k = 0; k < 8; k++) { float f = vfun(lat) - y; float c = cos(lat); float df = (1.0-uProjP.y)/(c*c) + uProjP.y; lat = clamp(lat - f/df, -1.55, 1.55); }
  return lat;
}
bool paniniInv(float u, float v, out vec3 d){
  float dd = uProjP.x; float k = u*u/((dd+1.0)*(dd+1.0));
  float disc = k*k*dd*dd - (k+1.0)*(k*dd*dd-1.0);
  float cl = (-k*dd + sqrt(max(disc, 0.0)))/(k+1.0);
  float lon = sign(u)*acos(clamp(cl, -1.0, 1.0));
  float S = (dd+1.0)/(dd+cos(lon));
  d = fromLL(lon, vinv(v/S)); return disc >= 0.0;
}
bool azimInv(int kind, float u, float v, out vec3 d){
  float r = length(vec2(u, v)); float th; bool ok = true;
  if (kind == 4) { th = r; ok = r <= PI; }
  else if (kind == 5) { th = 2.0*atan(r/2.0); }
  else if (kind == 6) { th = 2.0*asin(min(1.0, r/2.0)); ok = r <= 2.0; }
  else { th = asin(min(1.0, r)); ok = r <= 1.0; }
  float s = r > 1e-12 ? sin(th)/r : 1.0;
  d = vec3(u*s, v*s, cos(th)); return ok;
}
bool baseInv(int p, float u, float v, out vec3 d){
  if (p == 0) { d = fromLL(u, clamp(v, -PI/2.0, PI/2.0)); return abs(v) <= PI/2.0 + 1e-6 && abs(u) <= PI + 1e-6; }
  if (p == 1) { d = fromLL(u, atan(v)); return abs(u) <= PI + 1e-6; }
  if (p == 2) { d = fromLL(u, atan(sinh(v))); return abs(u) <= PI + 1e-6; }
  if (p == 3) { d = normalize(vec3(u, v, 1.0)); return true; }
  if (p >= 4 && p <= 7) return azimInv(p, u, v, d);
  if (p == 8) return paniniInv(u, v, d);
  if (p == 9) { float cl = cos(v); float lon = cl > 1e-9 ? u/cl : 0.0; d = fromLL(lon, v); return abs(lon) <= PI && abs(v) <= PI/2.0; }
  if (p == 10) { float lat = v > 0.0 ? atan(sinh(v/1.25))/0.8 : v; d = fromLL(u, lat); return abs(u) <= PI && abs(lat) <= PI/2.0; }
  d = vec3(0.0, 0.0, 1.0); return false;
}
bool projInv(float u, float v, out vec3 d){
  if (uProj >= 11) {  // transverse variants
    int b = uProj == 11 ? 0 : (uProj == 12 ? 1 : (uProj == 13 ? 2 : 8));
    vec3 d2; bool ok = baseInv(b, -v, u, d2);
    d = vec3(d2.y, -d2.x, d2.z); return ok;
  }
  return baseInv(uProj, u, v, d);
}
float srgb2lin(float x){ return x <= 0.04045 ? x/12.92 : pow((x+0.055)/1.055, 2.4); }
float lin2srgb(float x){ return x <= 0.0031308 ? x*12.92 : 1.055*pow(max(x,0.0), 1.0/2.4) - 0.055; }
`;

const FS_IMAGE = COMMON + `
in vec2 vNdc;
uniform vec2 uCenter; uniform vec2 uHalf;      // view in plane units
uniform mat3 uRT;                              // world -> camera
uniform vec2 uSize;                            // image w, h
uniform int uLens; uniform float uK, uF;       // 0 radial 1 cylindrical 2 equirect
uniform vec3 uABC; uniform vec2 uDE, uGT; uniform float uNormR, uRLim;
uniform int uCropMode; uniform vec4 uCrop;     // circle: cx, cy, r ; rect: l, t, r, b
uniform vec3 uGain; uniform vec3 uVig; uniform vec2 uVigC; uniform float uVigNorm2;
uniform float uFeather; uniform int uHard;     // 1 = nearest-to-centre look (unblended)
uniform sampler2D uTex;
out vec4 outColor;
float lensR(float th){ if (uK > 1e-9) return sin(uK*th)/uK; if (uK < -1e-9) return tan(min(-uK*th, PI/2.0*0.999))/(-uK); return th; }
bool lensOk(float th){ if (uK > 1e-9) return th < min(PI/(2.0*uK), PI); if (uK < -1e-9) return th < PI/(2.0*(-uK))*0.999; return th < PI*0.999; }
void main(){
  vec2 uv = uCenter + vNdc * uHalf;
  vec3 d; if (!projInv(uv.x, uv.y, d)) discard;
  vec3 c = uRT * d;
  float xi, yi; bool ok = true;
  if (uLens == 0) {
    float rho = length(c.xy); float th = atan(rho, c.z);
    ok = lensOk(th); float r = lensR(th) * uF; float s = rho > 1e-12 ? r/rho : uF;
    xi = c.x*s; yi = -c.y*s;
  } else if (uLens == 1) {
    float hh = length(c.xz); xi = uF*atan(c.x, c.z); yi = -uF*c.y/max(hh, 1e-12); ok = hh > 1e-9;
  } else {
    xi = uF*atan(c.x, c.z); yi = -uF*atan(c.y, length(c.xz));
  }
  if (!ok) discard;
  if (uABC != vec3(0.0)) {
    float rn = length(vec2(xi, yi))/uNormR; if (rn > uRLim) discard;
    float d0 = 1.0 - (uABC.x+uABC.y+uABC.z);
    float sc = ((uABC.x*rn + uABC.y)*rn + uABC.z)*rn + d0; xi *= sc; yi *= sc;
  }
  float xs = xi + uGT.x*yi, ys = yi + uGT.y*xi;
  float px = xs + uSize.x/2.0 + uDE.x;  // edge coordinates (0..w)
  float py = ys + uSize.y/2.0 + uDE.y;
  if (px < 0.0 || py < 0.0 || px > uSize.x || py > uSize.y) discard;
  float edge = min(min(px, uSize.x - px), min(py, uSize.y - py));
  if (uCropMode == 1) { float r = length(vec2(px, py) - uCrop.xy); if (r > uCrop.z) discard; edge = min(edge, uCrop.z - r); }
  else if (uCropMode == 2) { if (px < uCrop.x || py < uCrop.y || px > uCrop.z || py > uCrop.w) discard;
    edge = min(edge, min(min(px - uCrop.x, uCrop.z - px), min(py - uCrop.y, uCrop.w - py))); }
  vec3 col = texture(uTex, vec2(px/uSize.x, py/uSize.y)).rgb;
  col = vec3(srgb2lin(col.r), srgb2lin(col.g), srgb2lin(col.b)) * uGain;
  if (uVig != vec3(0.0)) {
    vec2 q = vec2(px, py) - (uSize/2.0 + uVigC); float r2 = dot(q, q)/uVigNorm2;
    col /= max(1.0 + r2*(uVig.x + r2*(uVig.y + r2*uVig.z)), 0.05);
  }
  float scale = min(uSize.x, uSize.y);
  float w = clamp(edge/(uFeather*scale), 0.0, 1.0);
  if (uHard == 1) w = pow(max(edge/scale, 1e-4), 10.0);  // closest-to-centre wins (unblended look)
  w = max(w, 1e-6);
  outColor = vec4(col*w, w);
}`;

const FS_RESOLVE = COMMON + `
in vec2 vNdc;
uniform sampler2D uAcc; uniform float uExposure;
uniform vec2 uCenter; uniform vec2 uHalf; uniform vec2 uCanvasHalf; uniform float uVOff;
out vec4 outColor;
void main(){
  vec2 tc = vNdc*0.5 + 0.5;
  vec4 a = texture(uAcc, tc);
  vec2 uv = uCenter + vNdc*uHalf;
  bool inCanvas = abs(uv.x) <= uCanvasHalf.x && abs(uv.y - uVOff) <= uCanvasHalf.y;
  // checkerboard background where nothing is covered
  vec2 g = floor(gl_FragCoord.xy/12.0); float chk = mod(g.x+g.y, 2.0) < 1.0 ? 0.16 : 0.2;
  vec3 bg = vec3(chk);
  if (a.a <= 1e-5) { outColor = vec4(inCanvas ? bg : bg*0.45, 1.0); return; }
  vec3 c = a.rgb/a.a * exp2(uExposure);
  c = vec3(lin2srgb(c.r), lin2srgb(c.g), lin2srgb(c.b));
  if (!inCanvas) c = mix(c, vec3(0.0), 0.55);
  outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(gl, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VS));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, 'aPos');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let k = 0; k < n; k++) { const info = gl.getActiveUniform(p, k); u[info.name] = gl.getUniformLocation(p, info.name); }
  return { p, u };
}

export class PanoRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 is not available');
    this.gl = gl;
    this.floatOK = !!gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('OES_texture_float_linear');
    this.progImg = program(gl, FS_IMAGE);
    this.progRes = program(gl, FS_RESOLVE);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.textures = new Map(); // key -> {tex, w, h, loading}
    this.fbo = null; this.fboTex = null; this.fbSize = [0, 0];
    this.onTextureLoaded = null;
    this.texSize = 1536;
    // view: centre and zoom relative to the fitted canvas
    this.view = { cx: 0, cy: 0, zoom: 1 };
    this.opts = { feather: 0.12, hard: false, exposure: 0, photometric: true, highlight: -1, only: null };
  }

  texture(project, i) {
    const im = project.images[i];
    const key = `${im.abspath}|${this.texSize}|${project.boot}`;
    let t = this.textures.get(key);
    if (t) return t.ready ? t : null;
    const gl = this.gl;
    t = { tex: gl.createTexture(), ready: false };
    this.textures.set(key, t);
    const img = new Image();
    img.onload = () => {
      gl.bindTexture(gl.TEXTURE_2D, t.tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const ext = gl.getExtension('EXT_texture_filter_anisotropic');
      if (ext) gl.texParameterf(gl.TEXTURE_2D, ext.TEXTURE_MAX_ANISOTROPY_EXT, 8);
      t.ready = true;
      if (this.onTextureLoaded) this.onTextureLoaded();
    };
    img.src = `/api/image/${i}/texture?size=${this.texSize}&v=${encodeURIComponent(im.abspath)}&b=${project.boot}`;
    return null;
  }

  // plane-units half extents of the visible area for the current canvas size
  viewport(project) {
    const ci = canvasInfo(project);
    const cw = this.canvas.clientWidth, chh = this.canvas.clientHeight;
    const margin = 0.92;
    const fit = Math.min(cw / ci.uExt, chh / ci.vExt) * margin; // pixels per plane unit at zoom 1
    const s = fit * this.view.zoom;
    return { ci, pxPerUnit: s, half: [cw / 2 / s, chh / 2 / s], center: [this.view.cx, this.view.cy + ci.vOff] };
  }

  // screen (css px) -> plane (u, v)
  screenToPlane(project, x, y) {
    const vp = this.viewport(project);
    const cw = this.canvas.clientWidth, chh = this.canvas.clientHeight;
    return [vp.center[0] + (x - cw / 2) / vp.pxPerUnit, vp.center[1] - (y - chh / 2) / vp.pxPerUnit];
  }

  planeToScreen(project, u, v) {
    const vp = this.viewport(project);
    const cw = this.canvas.clientWidth, chh = this.canvas.clientHeight;
    return [cw / 2 + (u - vp.center[0]) * vp.pxPerUnit, chh / 2 - (v - vp.center[1]) * vp.pxPerUnit];
  }

  ensureFbo(w, h) {
    const gl = this.gl;
    if (this.fbo && this.fbSize[0] === w && this.fbSize[1] === h) return;
    if (this.fbo) { gl.deleteFramebuffer(this.fbo); gl.deleteTexture(this.fboTex); }
    this.fboTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.fboTex);
    const fmt = this.floatOK ? gl.RGBA16F : gl.RGBA8;
    gl.texStorage2D(gl.TEXTURE_2D, 1, fmt, w, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboTex, 0);
    this.fbSize = [w, h];
  }

  // rotations: optional map image index -> 3x3 camera->world matrix (live drag preview)
  render(project, rotations = null) {
    const gl = this.gl;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const H = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== W || this.canvas.height !== H) { this.canvas.width = W; this.canvas.height = H; }
    if (!project || !project.images.length) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H); gl.clearColor(0.07, 0.07, 0.08, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }
    const vp = this.viewport(project);
    this.ensureFbo(W, H);
    gl.bindVertexArray(this.vao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    const { p: prog, u } = this.progImg;
    gl.useProgram(prog);
    const pn = project.panorama;
    gl.uniform1i(u.uProj, PROJ_IDS[pn.projection] ?? 0);
    gl.uniform2f(u.uProjP, (pn.proj_params || {}).d ?? 1.0, (pn.proj_params || {}).vcompress ?? 0.0);
    gl.uniform2f(u.uCenter, vp.center[0], vp.center[1]);
    gl.uniform2f(u.uHalf, vp.half[0], vp.half[1]);
    gl.uniform1f(u.uFeather, this.opts.feather);
    gl.uniform1i(u.uHard, this.opts.hard ? 1 : 0);
    const evs = project.images.filter((im) => im.include).map((im) => im.ev);
    const evRef = evs.length ? evs.slice().sort((a, b) => a - b)[Math.floor(evs.length / 2)] : 0;
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(u.uTex, 0);
    for (let i = 0; i < project.images.length; i++) {
      const im = project.images[i];
      if (!im.include) continue;
      if (this.opts.only && !this.opts.only.has(i)) continue;
      const tex = this.texture(project, i);
      if (!tex) continue;
      const cam = cameraFor(project, i, rotations && rotations.get(i));
      const lens = project.lenses[im.lens];
      const RT = cam.RT;
      // mat3 uniform is column-major: pass the transpose of RT's row layout
      gl.uniformMatrix3fv(u.uRT, false, new Float32Array([RT[0][0], RT[1][0], RT[2][0], RT[0][1], RT[1][1], RT[2][1], RT[0][2], RT[1][2], RT[2][2]]));
      gl.uniform2f(u.uSize, im.width, im.height);
      gl.uniform1i(u.uLens, cam.radial ? 0 : (cam.type === 'cylindrical' ? 1 : 2));
      gl.uniform1f(u.uK, cam.k);
      gl.uniform1f(u.uF, cam.f);
      gl.uniform3f(u.uABC, cam.a, cam.b, cam.c);
      gl.uniform2f(u.uDE, cam.d, cam.e);
      gl.uniform2f(u.uGT, cam.g, cam.t);
      gl.uniform1f(u.uNormR, cam.normR);
      gl.uniform1f(u.uRLim, Math.min(cam.rlim, 1e6));
      const cr = lens.crop;
      if (cr.mode === 'circle') {
        gl.uniform1i(u.uCropMode, 1);
        gl.uniform4f(u.uCrop, cr.cx || im.width / 2, cr.cy || im.height / 2, cr.radius || Math.min(im.width, im.height) / 2, 0);
      } else if (cr.mode === 'rect') {
        gl.uniform1i(u.uCropMode, 2);
        gl.uniform4f(u.uCrop, cr.left, cr.top, cr.right || im.width, cr.bottom || im.height);
      } else gl.uniform1i(u.uCropMode, 0);
      const ph = this.opts.photometric;
      const gain = ph ? Math.pow(2, im.ev - evRef) : 1;
      gl.uniform3f(u.uGain, gain / (ph ? im.wb_r : 1), gain, gain / (ph ? im.wb_b : 1));
      const vig = ph ? lens.vig : [0, 0, 0];
      gl.uniform3f(u.uVig, vig[0], vig[1], vig[2]);
      gl.uniform2f(u.uVigC, lens.vig_cx || 0, lens.vig_cy || 0);
      const norm2 = cr.mode === 'circle' ? (cr.radius || Math.min(im.width, im.height) / 2) ** 2 : (im.width ** 2 + im.height ** 2) / 4;
      gl.uniform1f(u.uVigNorm2, norm2);
      gl.bindTexture(gl.TEXTURE_2D, tex.tex);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.disable(gl.BLEND);
    // resolve to screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    const r = this.progRes;
    gl.useProgram(r.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fboTex);
    gl.uniform1i(r.u.uAcc, 0);
    gl.uniform1f(r.u.uExposure, this.opts.exposure + (project.exposure.exposure || 0));
    gl.uniform2f(r.u.uCenter, vp.center[0], vp.center[1]);
    gl.uniform2f(r.u.uHalf, vp.half[0], vp.half[1]);
    gl.uniform2f(r.u.uCanvasHalf, vp.ci.uExt / 2, vp.ci.vExt / 2);
    gl.uniform1f(r.u.uVOff, vp.ci.vOff);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

export { transpose };
