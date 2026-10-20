// Panorama Editor: real-time WebGL preview with direct manipulation.
import { PanoRenderer } from './glpano.js';
import { DEG, PROJ_LABELS, PROJ_MAX, axisAngle, cameraFor, canvasInfo, cross, dot, matMul, norm, rotPitch, rotYaw, rotation } from './geom.js';
import { store } from './store.js';
import { checkbox, fmt, h, icon, modal, numberInput, select, status, toast } from './ui.js';

const ICONS = {
  fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
  center: '<circle cx="12" cy="12" r="3"/><path d="M12 2v5M12 17v5M2 12h5M17 12h5"/>',
  level: '<path d="M3 12h18M7 8l-4 4 4 4M17 8l4 4-4 4"/>',
  crop: '<path d="M6 2v16h16M2 6h16v16"/>',
  num: '<path d="M4 9h16M4 15h16M10 3 8 21M16 3l-2 18"/>',
  outline: '<rect x="3" y="5" width="18" height="14" rx="2"/>',
  detail: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
  blend: '<circle cx="9" cy="12" r="6"/><circle cx="15" cy="12" r="6"/>',
  target: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="1"/>',
};

export class PanoEditor {
  constructor(root) {
    this.root = root;
    this.mode = 'pano'; // pano | image | crop | center | detail
    this.showNumbers = true;
    this.showOutlines = false;
    this.drag = null;
    this.live = null; // Map image -> rotation during drag
    this.build();
    store.onChange(() => { this.syncSide(); this.redraw(); });
  }

  build() {
    const bar = h('div', { class: 'editor-bar' });
    const btn = (name, title, fn, toggle) => {
      const b = h('button', { class: 'tb-btn', title, onclick: () => fn(b) }, icon(ICONS[name] || ''), title.split(' ')[0]);
      if (toggle) b.classList.toggle('on', toggle());
      return b;
    };
    this.modeSeg = h('div', { class: 'seg' },
      h('button', { class: 'on', onclick: (e) => this.setMode('pano', e.target) }, 'Edit panorama'),
      h('button', { onclick: (e) => this.setMode('image', e.target) }, 'Edit individual images'));
    bar.append(this.modeSeg, h('span', { class: 'tb-sep' }),
      btn('fit', 'Fit panorama', () => store.post('/api/fit', {})),
      btn('center', 'Center panorama', () => store.post('/api/fit', { center: true, horizontal: true })),
      btn('level', 'Straighten panorama', () => store.post('/api/straighten', {})),
      btn('target', 'Set center point (click)', (b) => this.setMode('center', null, b)),
      h('span', { class: 'tb-sep' }),
      btn('crop', 'Crop (drag rectangle)', (b) => this.setMode('crop', null, b)),
      btn('detail', 'Detail viewer (click)', (b) => this.setMode('detail', null, b)),
      h('span', { class: 'tb-sep' }),
      btn('num', 'Numbers', (b) => { this.showNumbers = !this.showNumbers; b.classList.toggle('on', this.showNumbers); this.redraw(); }, () => this.showNumbers),
      btn('outline', 'Outlines', (b) => { this.showOutlines = !this.showOutlines; b.classList.toggle('on', this.showOutlines); this.redraw(); }, () => this.showOutlines),
      btn('blend', 'Blended', (b) => { this.r.opts.hard = !this.r.opts.hard; b.classList.toggle('on', !this.r.opts.hard); this.redraw(); }, () => true),
      h('span', { class: 'tb-spacer' }),
      this.info = h('span', { class: 'muted mono' }));
    this.wrap = h('div', { class: 'editor-wrap' });
    this.glc = h('canvas');
    this.ovl = h('canvas');
    this.wrap.append(this.glc, this.ovl);
    this.side = h('div', { class: 'editor-side' });
    this.root.append(bar, h('div', { class: 'editor-body' }, this.wrap, this.side));
    try {
      this.r = new PanoRenderer(this.glc);
    } catch (e) {
      this.wrap.append(h('div', { style: { padding: '20px' } }, `WebGL2 unavailable: ${e.message}`));
      return;
    }
    this.r.onTextureLoaded = () => this.redraw();
    new ResizeObserver(() => this.redraw()).observe(this.wrap);
    this.bindEvents();
    this.buildSide();
  }

  setMode(mode, segBtn, btn) {
    if (segBtn) {
      this.modeSeg.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b === segBtn));
      this.mode = mode;
    } else {
      this.mode = this.mode === mode ? 'pano' : mode;
      this.root.querySelectorAll('.editor-bar .tb-btn.on').forEach((b) => {
        if (b !== btn && ['Crop', 'Detail', 'Set'].some((t) => b.textContent.startsWith(t))) b.classList.remove('on');
      });
      if (btn) btn.classList.toggle('on', this.mode === mode);
    }
    this.wrap.style.cursor = { crop: 'crosshair', center: 'crosshair', detail: 'zoom-in' }[this.mode] || 'grab';
    status({ pano: 'Drag to move the panorama, right-drag to roll, wheel to zoom', image: 'Drag an image to move it', crop: 'Drag a rectangle to crop', center: 'Click the point that should become the center', detail: 'Click to inspect at 100%' }[this.mode] || '');
    this.redraw();
  }

  // ---------------------------------------------------------------- side panel
  buildSide() {
    this.side.replaceChildren();
    const p = store.project;
    if (!p) return;
    const pn = p.panorama;
    const sec = (title, ...kids) => h('div', { class: 'side-sec' }, h('h4', {}, title), ...kids);
    const projSel = select(Object.entries(PROJ_LABELS), pn.projection, (v) => {
      const [mh, mv] = PROJ_MAX[v];
      store.patch({ panorama: { projection: v, hfov: Math.min(pn.hfov, mh), vfov: Math.min(pn.vfov, mv) } });
    });
    projSel.style.width = '100%';
    const [mh, mv] = PROJ_MAX[pn.projection] || [360, 180];
    const fovRow = (label, key, max) => {
      const range = h('input', { type: 'range', min: 1, max, step: 0.5, value: pn[key], style: { flex: 1 } });
      const num = numberInput(pn[key], (v) => store.patch({ panorama: { [key]: Math.max(1, Math.min(max, v)) } }), { digits: 1, width: '64px' });
      range.addEventListener('input', () => { num.value = range.value; this.previewPatch({ [key]: parseFloat(range.value) }); });
      range.addEventListener('change', () => store.patch({ panorama: { [key]: parseFloat(range.value) } }));
      return h('div', { class: 'row' }, h('label', {}, label), range, num);
    };
    const pp = pn.proj_params || {};
    const panini = pn.projection.includes('panini') ? [
      h('div', { class: 'row' }, h('label', {}, 'Compression d'), numberInput(pp.d ?? 1, (v) => store.patch({ panorama: { proj_params: { ...pp, d: v } } }), { digits: 2, step: 0.05 })),
      h('div', { class: 'row' }, h('label', {}, 'Vertical squeeze'), numberInput(pp.vcompress ?? 0, (v) => store.patch({ panorama: { proj_params: { ...pp, vcompress: Math.max(0, Math.min(1, v)) } } }), { digits: 2, step: 0.05 })),
    ] : [];
    const yaw = numberInput(0, () => {}, { digits: 1, width: '70px' });
    const pitch = numberInput(0, () => {}, { digits: 1, width: '70px' });
    const roll = numberInput(0, () => {}, { digits: 1, width: '70px' });
    const ex = h('input', { type: 'range', min: -3, max: 3, step: 0.05, value: this.r.opts.exposure, style: { flex: 1 } });
    ex.addEventListener('input', () => { this.r.opts.exposure = parseFloat(ex.value); this.redraw(); });
    const feather = h('input', { type: 'range', min: 0.01, max: 0.5, step: 0.01, value: this.r.opts.feather, style: { flex: 1 } });
    feather.addEventListener('input', () => { this.r.opts.feather = parseFloat(feather.value); this.redraw(); });
    const crop = pn.crop;
    const opt = p.optimum_size || [0, 0];
    this.side.append(
      sec('Projection', projSel, fovRow('Horizontal', 'hfov', mh), fovRow('Vertical', 'vfov', mv), ...panini,
        h('div', { class: 'row' }, h('button', { class: 'btn small', onclick: () => store.post('/api/fit', { choose_projection: true }) }, 'Choose suitable projection'))),
      sec('Numerical transform', h('div', { class: 'hint' }, 'Shift the panorama by:'),
        h('div', { class: 'row' }, h('label', {}, 'Yaw'), yaw), h('div', { class: 'row' }, h('label', {}, 'Pitch'), pitch), h('div', { class: 'row' }, h('label', {}, 'Roll'), roll),
        h('div', { class: 'row' }, h('button', { class: 'btn small primary', onclick: () => store.post('/api/transform', { yaw: +yaw.value, pitch: +pitch.value, roll: +roll.value }) }, 'Apply'))),
      sec('Display', h('div', { class: 'row' }, h('label', {}, 'Exposure'), ex), h('div', { class: 'row' }, h('label', {}, 'Feather'), feather),
        h('div', { class: 'row' }, checkbox(this.r.opts.photometric, (v) => { this.r.opts.photometric = v; this.redraw(); }, 'Exposure correction'))),
      sec('Crop', h('div', { class: 'mono muted' }, `${crop.map((v) => fmt(v * 100, 1)).join(' / ')} %`),
        h('div', { class: 'row' }, h('button', { class: 'btn small', onclick: () => store.patch({ panorama: { crop: [0, 0, 1, 1] } }) }, 'Reset crop'),
          h('button', { class: 'btn small', onclick: () => this.autoCrop() }, 'Auto crop'))),
      sec('Size', h('div', { class: 'muted' }, `Optimum: ${opt[0]} × ${opt[1]} px (${fmt(opt[0] * opt[1] / 1e6, 1)} MP)`)),
    );
  }

  syncSide() {
    if (!this.r) return;
    if (!this.sideKey || this.sideKey !== this.keyOf()) { this.buildSide(); this.sideKey = this.keyOf(); }
  }

  keyOf() {
    const p = store.project;
    return p ? JSON.stringify([p.panorama, p.optimum_size]) : '';
  }

  previewPatch(vals) {
    // live FOV update without a server round trip
    Object.assign(store.project.panorama, vals);
    this.redraw();
  }

  autoCrop() {
    // largest rectangle fully covered by images, sampled on a coarse grid
    const p = store.project;
    const ci = canvasInfo(p);
    const N = 160, M = Math.max(8, Math.round(N * ci.aspect));
    const cams = p.images.map((im, i) => (im.include ? cameraFor(p, i) : null));
    const cov = [];
    for (let y = 0; y < M; y++) {
      const row = [];
      for (let x = 0; x < N; x++) {
        const u = ((x + 0.5) / N - 0.5) * ci.uExt, v = (0.5 - (y + 0.5) / M) * ci.vExt + ci.vOff;
        const [d, ok] = ci.P.inv(u, v);
        let c = false;
        if (ok) for (const cam of cams) { if (!cam) continue; const [px, py, v2] = cam.worldToPixel(d); if (v2 && px >= 0 && py >= 0 && px < cam.w && py < cam.h) { c = true; break; } }
        row.push(c ? 1 : 0);
      }
      cov.push(row);
    }
    // maximal rectangle in a binary matrix (histogram method)
    let best = [0, 0, N, M], area = 0;
    const hgt = new Array(N).fill(0);
    for (let y = 0; y < M; y++) {
      for (let x = 0; x < N; x++) hgt[x] = cov[y][x] ? hgt[x] + 1 : 0;
      const st = [];
      for (let x = 0; x <= N; x++) {
        const hh = x < N ? hgt[x] : 0;
        let start = x;
        while (st.length && st[st.length - 1][1] >= hh) {
          const [sx, sh] = st.pop();
          const a = sh * (x - sx);
          if (a > area) { area = a; best = [sx, y - sh + 1, x, y + 1]; }
          start = sx;
        }
        st.push([start, hh]);
      }
    }
    store.patch({ panorama: { crop: [best[0] / N, best[1] / M, best[2] / N, best[3] / M] } });
  }

  // ---------------------------------------------------------------- interaction
  dirAt(x, y) {
    const p = store.project;
    const [u, v] = this.r.screenToPlane(p, x, y);
    const ci = canvasInfo(p);
    const [d, ok] = ci.P.inv(u, v);
    return ok ? norm(d) : null;
  }

  imageAt(x, y) {
    const p = store.project;
    const d = this.dirAt(x, y);
    if (!d) return -1;
    let best = -1, bestScore = -1;
    p.images.forEach((im, i) => {
      if (!im.include) return;
      const cam = cameraFor(p, i);
      const [px, py, ok] = cam.worldToPixel(d);
      if (!ok || px < 0 || py < 0 || px > cam.w || py > cam.h) return;
      const s = Math.min(px, cam.w - px, py, cam.h - py) / Math.min(cam.w, cam.h);
      if (s > bestScore) { bestScore = s; best = i; }
    });
    return best;
  }

  bindEvents() {
    const el = this.ovl;
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => {
      const p = store.project;
      if (!p || !p.images.length) return;
      el.setPointerCapture(e.pointerId);
      const x = e.offsetX, y = e.offsetY;
      if (e.button === 1 || (e.button === 0 && e.shiftKey && this.mode !== 'crop')) {
        this.drag = { kind: 'pan', x, y, cx: this.r.view.cx, cy: this.r.view.cy };
        return;
      }
      if (this.mode === 'crop') { this.drag = { kind: 'crop', x0: x, y0: y, x, y }; return; }
      if (this.mode === 'center') { this.setCenterAt(x, y); return; }
      if (this.mode === 'detail') { this.openDetail(x, y); return; }
      const d0 = this.dirAt(x, y);
      if (!d0) return;
      let images = null;
      if (this.mode === 'image') {
        const i = this.imageAt(x, y);
        if (i < 0) return;
        images = this.groupOf(i);
      }
      const roll = e.button === 2 || e.altKey;
      this.drag = { kind: roll ? 'roll' : 'rotate', x, y, d0, images, R: null, base: this.baseRotations(images) };
    });
    el.addEventListener('pointermove', (e) => {
      const x = e.offsetX, y = e.offsetY;
      if (!this.drag) { this.hover(x, y); return; }
      const dr = this.drag;
      const p = store.project;
      if (dr.kind === 'pan') {
        const vp = this.r.viewport(p);
        this.r.view.cx = dr.cx - (x - dr.x) / vp.pxPerUnit;
        this.r.view.cy = dr.cy + (y - dr.y) / vp.pxPerUnit;
        this.redraw();
      } else if (dr.kind === 'crop') {
        dr.x = x; dr.y = y; this.drawOverlay();
      } else if (dr.kind === 'rotate') {
        const d1 = this.dirAt(x, y);
        if (!d1) return;
        let R;
        if (this.mode === 'pano' && !e.ctrlKey && !e.metaKey) {
          // yaw about the vertical axis, pitch about the horizontal axis: keeps verticals vertical while moving sideways
          const lon0 = Math.atan2(dr.d0[0], dr.d0[2]), lat0 = Math.asin(dr.d0[1]);
          const lon1 = Math.atan2(d1[0], d1[2]), lat1 = Math.asin(d1[1]);
          R = matMul(rotPitch(lat1 - lat0), rotYaw(lon1 - lon0));
        } else {
          const axis = cross(dr.d0, d1);
          const s = Math.hypot(...axis);
          R = s < 1e-9 ? [[1, 0, 0], [0, 1, 0], [0, 0, 1]] : axisAngle(axis, Math.atan2(s, dot(dr.d0, d1)));
        }
        dr.R = R;
        this.applyLive(R, dr);
      } else if (dr.kind === 'roll') {
        const cw = this.ovl.clientWidth / 2, chh = this.ovl.clientHeight / 2;
        const a0 = Math.atan2(dr.y - chh, dr.x - cw), a1 = Math.atan2(y - chh, x - cw);
        const axis = this.dirAt(cw, chh) || [0, 0, 1];
        const R = axisAngle(axis, -(a1 - a0));
        dr.R = R;
        this.applyLive(R, dr);
      }
    });
    el.addEventListener('pointerup', (e) => {
      const dr = this.drag;
      this.drag = null;
      if (!dr) return;
      if (dr.kind === 'crop') {
        const p = store.project;
        const ci = canvasInfo(p);
        const a = this.r.screenToPlane(p, Math.min(dr.x0, dr.x), Math.min(dr.y0, dr.y));
        const b = this.r.screenToPlane(p, Math.max(dr.x0, dr.x), Math.max(dr.y0, dr.y));
        const fx = (u) => Math.max(0, Math.min(1, u / ci.uExt + 0.5));
        const fy = (v) => Math.max(0, Math.min(1, 0.5 - (v - ci.vOff) / ci.vExt));
        const crop = [fx(a[0]), fy(a[1]), fx(b[0]), fy(b[1])];
        if (crop[2] - crop[0] > 0.01 && crop[3] - crop[1] > 0.01) store.patch({ panorama: { crop } });
        return;
      }
      if ((dr.kind === 'rotate' || dr.kind === 'roll') && dr.R) {
        const body = { matrix: dr.R };
        if (dr.images) body.images = [...dr.images];
        store.post('/api/transform', body).then(() => { this.live = null; this.redraw(); });
      }
    });
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      const p = store.project;
      if (!p) return;
      const before = this.r.screenToPlane(p, e.offsetX, e.offsetY);
      this.r.view.zoom = Math.max(0.2, Math.min(40, this.r.view.zoom * Math.exp(-e.deltaY * 0.0015)));
      const after = this.r.screenToPlane(p, e.offsetX, e.offsetY);
      this.r.view.cx += before[0] - after[0];
      this.r.view.cy += before[1] - after[1];
      this.redraw();
    }, { passive: false });
    el.addEventListener('dblclick', () => { this.r.view = { cx: 0, cy: 0, zoom: 1 }; this.redraw(); });
  }

  groupOf(i) {
    const p = store.project;
    const g = p.images[i].link_group;
    const s = new Set([i]);
    if (g >= 0) p.images.forEach((im, k) => { if (im.link_group === g) s.add(k); });
    return s;
  }

  baseRotations(images) {
    const p = store.project;
    const m = new Map();
    p.images.forEach((im, i) => { if (!images || images.has(i)) m.set(i, rotation(im.yaw, im.pitch, im.roll)); });
    return m;
  }

  applyLive(R, dr) {
    const live = new Map();
    for (const [i, Ri] of dr.base) live.set(i, matMul(R, Ri));
    this.live = live;
    this.redraw();
  }

  setCenterAt(x, y) {
    const d = this.dirAt(x, y);
    if (!d) return;
    const lon = Math.atan2(d[0], d[2]), lat = Math.asin(d[1]);
    const R = matMul(rotPitch(-lat), rotYaw(-lon));
    store.post('/api/transform', { matrix: R });
    this.setMode('pano', null, null);
  }

  async openDetail(x, y) {
    const p = store.project;
    const ci = canvasInfo(p);
    const [u, v] = this.r.screenToPlane(p, x, y);
    const fx = u / ci.uExt + 0.5, fy = 0.5 - (v - ci.vOff) / ci.vExt;
    const opt = p.optimum_size || [4000, 2000];
    const W = p.panorama.width || opt[0];
    const H = Math.round(W * ci.aspect);
    const ww = 1100 / W, hh = 720 / H;
    const crop = [fx - ww / 2, fy - hh / 2, fx + ww / 2, fy + hh / 2];
    status('Rendering detail view…');
    try {
      const r = await fetch('/api/render_region', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ crop, width: W }) });
      if (!r.ok) throw new Error(await r.text());
      const url = URL.createObjectURL(await r.blob());
      status('Detail view: final quality at 100%');
      await modal('Detail viewer (100%, final blending)', h('img', { src: url, style: { maxWidth: '100%', display: 'block', background: '#000' } }));
      URL.revokeObjectURL(url);
    } catch (e) { toast(e.message, 'err'); }
  }

  hover(x, y) {
    const p = store.project;
    if (!p || !p.images.length) return;
    const d = this.dirAt(x, y);
    if (!d) { this.info.textContent = ''; return; }
    const lon = Math.atan2(d[0], d[2]) / DEG, lat = Math.asin(d[1]) / DEG;
    const i = this.imageAt(x, y);
    this.info.textContent = `yaw ${fmt(lon, 1)}°  pitch ${fmt(lat, 1)}°${i >= 0 ? `  image ${i}` : ''}`;
    if (this.mode === 'image') { this.hoverImage = i; this.drawOverlay(); }
  }

  // ---------------------------------------------------------------- drawing
  redraw() {
    if (!this.r || this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = null;
      if (!this.root.offsetParent) return; // hidden
      this.r.render(store.project, this.live);
      this.drawOverlay();
    });
  }

  drawOverlay() {
    const c = this.ovl;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(c.clientWidth * dpr), H = Math.round(c.clientHeight * dpr);
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, c.clientWidth, c.clientHeight);
    const p = store.project;
    if (!p || !p.images.length) {
      g.fillStyle = '#9aa2ab'; g.font = '15px system-ui'; g.textAlign = 'center';
      g.fillText('Load images to start (Project Assistant)', c.clientWidth / 2, c.clientHeight / 2);
      return;
    }
    const ci = canvasInfo(p);
    // canvas frame
    const tl = this.r.planeToScreen(p, -ci.uExt / 2, ci.vExt / 2 + ci.vOff);
    const br = this.r.planeToScreen(p, ci.uExt / 2, -ci.vExt / 2 + ci.vOff);
    g.strokeStyle = 'rgba(255,255,255,.35)'; g.lineWidth = 1;
    g.strokeRect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]);
    // crop
    const [l, t, r, b] = p.panorama.crop;
    if (l > 0 || t > 0 || r < 1 || b < 1) {
      const X = (f) => tl[0] + f * (br[0] - tl[0]), Y = (f) => tl[1] + f * (br[1] - tl[1]);
      g.fillStyle = 'rgba(0,0,0,.45)';
      g.fillRect(tl[0], tl[1], br[0] - tl[0], Y(t) - tl[1]);
      g.fillRect(tl[0], Y(b), br[0] - tl[0], br[1] - Y(b));
      g.fillRect(tl[0], Y(t), X(l) - tl[0], Y(b) - Y(t));
      g.fillRect(X(r), Y(t), br[0] - X(r), Y(b) - Y(t));
      g.strokeStyle = '#3d8bfd'; g.lineWidth = 2; g.strokeRect(X(l), Y(t), X(r) - X(l), Y(b) - Y(t));
    }
    // horizon
    g.strokeStyle = 'rgba(255,214,10,.55)'; g.lineWidth = 1; g.setLineDash([6, 5]);
    this.polyline(g, p, ci, Array.from({ length: 181 }, (_, k) => { const lon = (k / 180 - 0.5) * 2 * Math.PI; return [Math.sin(lon), 0, Math.cos(lon)]; }));
    g.setLineDash([]);
    const rot = this.live;
    p.images.forEach((im, i) => {
      if (!im.include) return;
      const cam = cameraFor(p, i, rot && rot.get(i));
      const hi = this.mode === 'image' && (this.hoverImage === i || (this.drag && this.drag.images && this.drag.images.has(i)));
      if (this.showOutlines || hi) {
        const pts = [];
        const n = 48;
        const cr = p.lenses[im.lens].crop;
        if (cr.mode === 'circle') {
          for (let k = 0; k <= 4 * n; k++) { const a = k / (4 * n) * 2 * Math.PI; pts.push(cam.pixelToWorld(cr.cx - 0.5 + cr.radius * Math.cos(a), cr.cy - 0.5 + cr.radius * Math.sin(a))); }
        } else {
          for (let k = 0; k <= n; k++) pts.push(cam.pixelToWorld(k / n * im.width - 0.5, -0.5));
          for (let k = 0; k <= n; k++) pts.push(cam.pixelToWorld(im.width - 0.5, k / n * im.height - 0.5));
          for (let k = n; k >= 0; k--) pts.push(cam.pixelToWorld(k / n * im.width - 0.5, im.height - 0.5));
          for (let k = n; k >= 0; k--) pts.push(cam.pixelToWorld(-0.5, k / n * im.height - 0.5));
        }
        g.strokeStyle = hi ? '#3d8bfd' : 'rgba(255,255,255,.55)'; g.lineWidth = hi ? 2 : 1;
        this.polyline(g, p, ci, pts);
      }
      if (this.showNumbers) {
        const cr = p.lenses[im.lens].crop;
        const cx = cr.mode === 'circle' ? cr.cx - 0.5 : im.width / 2 - 0.5, cy = cr.mode === 'circle' ? cr.cy - 0.5 : im.height / 2 - 0.5;
        const d = cam.pixelToWorld(cx, cy);
        const [u, v, ok] = ci.P.fwd(d);
        if (!ok) return;
        const [sx, sy] = this.r.planeToScreen(p, u, v);
        g.fillStyle = hi ? '#3d8bfd' : 'rgba(0,0,0,.6)';
        g.beginPath(); g.arc(sx, sy, 11, 0, 2 * Math.PI); g.fill();
        g.fillStyle = '#fff'; g.font = '600 11px system-ui'; g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText(String(i), sx, sy);
      }
    });
    if (this.drag && this.drag.kind === 'crop') {
      const d = this.drag;
      g.strokeStyle = '#3d8bfd'; g.lineWidth = 2; g.setLineDash([5, 4]);
      g.strokeRect(Math.min(d.x0, d.x), Math.min(d.y0, d.y), Math.abs(d.x - d.x0), Math.abs(d.y - d.y0));
      g.setLineDash([]);
    }
  }

  polyline(g, p, ci, dirs) {
    let prev = null;
    g.beginPath();
    for (const d of dirs) {
      const [u, v, ok] = ci.P.fwd(d);
      if (!ok) { prev = null; continue; }
      const s = this.r.planeToScreen(p, u, v);
      if (prev && Math.hypot(s[0] - prev[0], s[1] - prev[1]) < this.ovl.clientWidth * 0.4) g.lineTo(s[0], s[1]);
      else g.moveTo(s[0], s[1]);
      prev = s;
    }
    g.stroke();
  }
}

