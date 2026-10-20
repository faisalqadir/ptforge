// Single-image tools: mask painting and lens crop editing.
import { store } from './store.js';
import { fmt, h, numberInput, select, status, toast } from './ui.js';

class ZoomView {
  constructor(onDraw) {
    this.canvas = h('canvas');
    this.wrap = h('div', { class: 'editor-wrap' }, this.canvas);
    this.onDraw = onDraw;
    this.scale = 1; this.ox = 0; this.oy = 0; this.iw = 1; this.ih = 1;
    this.img = null;
    new ResizeObserver(() => { if (!this.fitted) this.fit(); this.draw(); }).observe(this.wrap);
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const [x, y] = this.toImage(e.offsetX, e.offsetY);
      this.scale = Math.max(0.02, Math.min(32, this.scale * Math.exp(-e.deltaY * 0.0018)));
      this.ox = x - e.offsetX / this.scale; this.oy = y - e.offsetY / this.scale;
      this.draw();
    }, { passive: false });
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }
  setImage(url, w, hgt) {
    this.iw = w; this.ih = hgt; this.fitted = false;
    const im = new Image();
    im.onload = () => { this.img = im; this.fit(); this.draw(); };
    im.src = url;
    this.img = null;
  }
  fit() {
    const W = this.wrap.clientWidth, H = this.wrap.clientHeight;
    if (!W || !H) return;
    this.scale = Math.min(W / this.iw, H / this.ih) * 0.97;
    this.ox = (this.iw - W / this.scale) / 2; this.oy = (this.ih - H / this.scale) / 2;
    this.fitted = true;
  }
  // edge coordinates (0..w)
  toImage(sx, sy) { return [this.ox + sx / this.scale, this.oy + sy / this.scale]; }
  toScreen(x, y) { return [(x - this.ox) * this.scale, (y - this.oy) * this.scale]; }
  draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const W = this.wrap.clientWidth, H = this.wrap.clientHeight;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#111'; g.fillRect(0, 0, W, H);
    if (this.img) {
      g.imageSmoothingEnabled = this.scale < 2;
      g.drawImage(this.img, -this.ox * this.scale, -this.oy * this.scale, this.iw * this.scale, this.ih * this.scale);
    }
    if (this.onDraw) this.onDraw(g);
  }
}

function imageStrip(current, onPick) {
  const p = store.project;
  return h('div', { class: 'strip' }, p.images.map((im, i) => h('div', { class: `s ${i === current ? 'sel' : ''}`, title: im.name, onclick: () => onPick(i) },
    h('img', { src: store.imgUrl(i, 'thumb', 160), loading: 'lazy' }), h('span', {}, i))));
}

// ======================================================================== masks
export class MaskEditor {
  constructor(root) {
    this.root = root;
    this.idx = -1;
    this.tool = 'red';
    this.color = 'red';
    this.size = 60; // brush radius in source pixels
    this.view = new ZoomView((g) => this.overlay(g));
    this.mask = document.createElement('canvas');
    this.build();
    store.onChange(() => this.refresh());
  }
  build() {
    const tools = h('div', { class: 'seg' }, [['red', 'Exclude (red)'], ['green', 'Include (green)'], ['erase', 'Eraser'], ['fill', 'Fill']].map(([t, l]) =>
      h('button', { class: t === this.tool ? 'on' : '', onclick: (e) => {
        this.tool = t;
        if (t === 'red' || t === 'green') this.color = t; // fill paints with the last chosen colour
        e.target.parentNode.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b === e.target));
      } }, l)));
    const size = h('input', { type: 'range', min: 4, max: 600, value: this.size });
    size.addEventListener('input', () => { this.size = +size.value; });
    this.bar = h('div', { class: 'editor-bar' }, tools, h('span', { class: 'tb-sep' }), h('span', { class: 'muted' }, 'Pencil size'), size,
      h('span', { class: 'tb-sep' }),
      h('button', { class: 'tb-btn', onclick: () => this.clear() }, 'Clear mask'),
      h('button', { class: 'tb-btn', onclick: () => this.copyToAll() }, 'Copy mask to all images'),
      h('span', { class: 'tb-spacer' }), h('span', { class: 'muted' }, 'Right-drag or Shift-drag pans, wheel zooms. Red areas never appear; green areas are preferred when blending.'));
    this.stripBox = h('div');
    this.root.append(this.bar, this.stripBox, this.view.wrap);
    const c = this.view.canvas;
    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      if (e.button !== 0 || e.shiftKey) { this.pan = { sx: e.offsetX, sy: e.offsetY, ox: this.view.ox, oy: this.view.oy }; return; }
      const [x, y] = this.view.toImage(e.offsetX, e.offsetY);
      if (this.tool === 'fill') { this.fill(x, y); return; }
      this.painting = true;
      this.last = [x, y];
      this.stroke(x, y, x, y);
    });
    c.addEventListener('pointermove', (e) => {
      const [x, y] = this.view.toImage(e.offsetX, e.offsetY);
      this.cursor = [e.offsetX, e.offsetY];
      if (this.pan) {
        this.view.ox = this.pan.ox - (e.offsetX - this.pan.sx) / this.view.scale;
        this.view.oy = this.pan.oy - (e.offsetY - this.pan.sy) / this.view.scale;
      } else if (this.painting) {
        this.stroke(this.last[0], this.last[1], x, y);
        this.last = [x, y];
      }
      this.view.draw();
    });
    c.addEventListener('pointerup', () => {
      this.pan = null;
      if (this.painting) { this.painting = false; this.scheduleSave(); }
    });
    c.addEventListener('pointerleave', () => { this.cursor = null; this.view.draw(); });
  }
  refresh() {
    const p = store.project;
    if (!p) return;
    if (this.idx >= p.images.length) this.idx = -1;
    if (this.idx < 0 && p.images.length) this.pick(0);
    else this.stripBox.replaceChildren(imageStrip(this.idx, (i) => this.pick(i)));
  }
  pick(i) {
    this.idx = i;
    const im = store.project.images[i];
    this.stripBox.replaceChildren(imageStrip(i, (k) => this.pick(k)));
    this.view.setImage(store.imgUrl(i, 'texture', 2048), im.width, im.height);
    const s = Math.min(1, 2048 / Math.max(im.width, im.height));
    this.mask.width = Math.max(1, Math.round(im.width * s));
    this.mask.height = Math.max(1, Math.round(im.height * s));
    this.ms = s;
    const g = this.mask.getContext('2d');
    g.clearRect(0, 0, this.mask.width, this.mask.height);
    if (im.mask) {
      const m = new Image();
      m.onload = () => { g.imageSmoothingEnabled = false; g.drawImage(m, 0, 0, this.mask.width, this.mask.height); this.view.draw(); };
      m.src = `/api/image/${i}/mask?t=${Date.now()}`;
    }
  }
  stroke(x0, y0, x1, y1) {
    const g = this.mask.getContext('2d');
    g.save();
    g.lineCap = 'round';
    g.lineWidth = this.size * 2 * this.ms;
    if (this.tool === 'erase') { g.globalCompositeOperation = 'destination-out'; g.strokeStyle = '#000'; }
    else { g.globalCompositeOperation = 'source-over'; g.strokeStyle = this.tool === 'red' ? 'rgb(255,0,0)' : 'rgb(0,255,0)'; }
    g.beginPath(); g.moveTo(x0 * this.ms, y0 * this.ms); g.lineTo(x1 * this.ms + 0.01, y1 * this.ms); g.stroke();
    g.restore();
    this.view.draw();
  }
  fill(x, y) {
    // flood fill the contiguous region (same mask value) under the cursor
    const g = this.mask.getContext('2d');
    const W = this.mask.width, H = this.mask.height;
    const d = g.getImageData(0, 0, W, H);
    const px = Math.floor(x * this.ms), py = Math.floor(y * this.ms);
    if (px < 0 || py < 0 || px >= W || py >= H) return;
    const key = (k) => (d.data[k * 4 + 3] > 127 ? (d.data[k * 4] > 127 ? 1 : 2) : 0);
    const target = key(py * W + px);
    const col = this.color === 'red' ? [255, 0, 0] : [0, 255, 0];
    const seen = new Uint8Array(W * H);
    const stack = [py * W + px];
    while (stack.length) {
      const k = stack.pop();
      if (seen[k] || key(k) !== target) continue;
      seen[k] = 1;
      d.data[k * 4] = col[0]; d.data[k * 4 + 1] = col[1]; d.data[k * 4 + 2] = col[2]; d.data[k * 4 + 3] = 255;
      const xx = k % W, yy = (k / W) | 0;
      if (xx > 0) stack.push(k - 1); if (xx < W - 1) stack.push(k + 1);
      if (yy > 0) stack.push(k - W); if (yy < H - 1) stack.push(k + W);
    }
    g.putImageData(d, 0, 0);
    this.view.draw();
    this.scheduleSave();
  }
  overlay(g) {
    const v = this.view;
    g.globalAlpha = 0.45;
    g.imageSmoothingEnabled = false;
    g.drawImage(this.mask, -v.ox * v.scale, -v.oy * v.scale, v.iw * v.scale, v.ih * v.scale);
    g.globalAlpha = 1;
    if (this.cursor && this.tool !== 'fill') {
      g.strokeStyle = this.tool === 'red' ? '#f85149' : this.tool === 'green' ? '#3fb950' : '#fff';
      g.lineWidth = 1.5;
      g.beginPath(); g.arc(this.cursor[0], this.cursor[1], this.size * v.scale, 0, 2 * Math.PI); g.stroke();
    }
  }
  scheduleSave() {
    clearTimeout(this.saveT);
    this.saveT = setTimeout(() => this.save(), 500);
  }
  async save(idx = this.idx) {
    const blob = await new Promise((r) => this.mask.toBlob(r, 'image/png'));
    try {
      const res = await fetch(`/api/image/${idx}/mask`, { method: 'PUT', body: blob });
      if (!res.ok) throw new Error(await res.text());
      const j = await res.json();
      store.set(j.project);
      status('Mask saved');
    } catch (e) { toast(`Saving mask failed: ${e.message}`, 'err'); }
  }
  clear() {
    this.mask.getContext('2d').clearRect(0, 0, this.mask.width, this.mask.height);
    this.view.draw();
    this.save();
  }
  async copyToAll() {
    const p = store.project;
    for (let i = 0; i < p.images.length; i++) {
      if (i === this.idx) continue;
      if (p.images[i].width !== p.images[this.idx].width || p.images[i].height !== p.images[this.idx].height) continue;
      await this.save(i);
    }
    toast('Mask copied to all images of the same size', 'ok');
  }
}

// ======================================================================== crop
export class CropEditor {
  constructor(root) {
    this.root = root;
    this.idx = -1;
    this.view = new ZoomView((g) => this.overlay(g));
    this.form = h('div', { class: 'editor-side' });
    this.stripBox = h('div');
    this.root.append(this.stripBox, h('div', { class: 'editor-body' }, this.view.wrap, this.form));
    store.onChange(() => this.refresh());
    const c = this.view.canvas;
    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      const [x, y] = this.view.toImage(e.offsetX, e.offsetY);
      const cr = this.crop();
      if (e.button !== 0 || e.shiftKey || !cr) { this.drag = { kind: 'pan', sx: e.offsetX, sy: e.offsetY, ox: this.view.ox, oy: this.view.oy }; return; }
      this.drag = { kind: this.handleAt(cr, x, y), x0: x, y0: y, start: { ...cr } };
    });
    c.addEventListener('pointermove', (e) => {
      if (!this.drag) return;
      const [x, y] = this.view.toImage(e.offsetX, e.offsetY);
      const d = this.drag;
      if (d.kind === 'pan') {
        this.view.ox = d.ox - (e.offsetX - d.sx) / this.view.scale; this.view.oy = d.oy - (e.offsetY - d.sy) / this.view.scale;
        this.view.draw(); return;
      }
      const cr = this.crop();
      const dx = x - d.x0, dy = y - d.y0, s = d.start;
      if (cr.mode === 'circle') {
        if (d.kind === 'radius') cr.radius = Math.max(10, Math.hypot(x - s.cx, y - s.cy));
        else { cr.cx = s.cx + dx; cr.cy = s.cy + dy; }
      } else if (cr.mode === 'rect') {
        if (d.kind.includes('l')) cr.left = s.left + dx;
        if (d.kind.includes('r')) cr.right = s.right + dx;
        if (d.kind.includes('t')) cr.top = s.top + dy;
        if (d.kind.includes('b')) cr.bottom = s.bottom + dy;
        if (d.kind === 'move') { cr.left = s.left + dx; cr.right = s.right + dx; cr.top = s.top + dy; cr.bottom = s.bottom + dy; }
      }
      this.view.draw();
    });
    c.addEventListener('pointerup', () => {
      const d = this.drag; this.drag = null;
      if (d && d.kind !== 'pan') this.commit();
    });
  }
  lens() { const p = store.project; return p.lenses[p.images[this.idx].lens]; }
  crop() {
    if (this.idx < 0) return null;
    const cr = this.lens().crop;
    const im = store.project.images[this.idx];
    if (cr.mode === 'circle') {
      cr.cx = cr.cx || im.width / 2; cr.cy = cr.cy || im.height / 2; cr.radius = cr.radius || Math.min(im.width, im.height) / 2;
    } else if (cr.mode === 'rect') {
      cr.right = cr.right || im.width; cr.bottom = cr.bottom || im.height;
    }
    return cr.mode === 'none' ? null : cr;
  }
  handleAt(cr, x, y) {
    const tol = 14 / this.view.scale;
    if (cr.mode === 'circle') return Math.abs(Math.hypot(x - cr.cx, y - cr.cy) - cr.radius) < tol ? 'radius' : 'move';
    let k = '';
    if (Math.abs(x - cr.left) < tol) k += 'l'; if (Math.abs(x - cr.right) < tol) k += 'r';
    if (Math.abs(y - cr.top) < tol) k += 't'; if (Math.abs(y - cr.bottom) < tol) k += 'b';
    return k || 'move';
  }
  commit() {
    const im = store.project.images[this.idx];
    store.patch({ lenses: { [im.lens]: { crop: { ...this.lens().crop } } } });
  }
  refresh() {
    const p = store.project;
    if (!p) return;
    if (this.idx >= p.images.length) this.idx = -1;
    if (this.idx < 0 && p.images.length) { this.pick(0); return; }
    this.stripBox.replaceChildren(imageStrip(this.idx, (i) => this.pick(i)));
    this.buildForm();
    this.view.draw();
  }
  pick(i) {
    this.idx = i;
    const im = store.project.images[i];
    this.stripBox.replaceChildren(imageStrip(i, (k) => this.pick(k)));
    this.view.setImage(store.imgUrl(i, 'texture', 2048), im.width, im.height);
    this.buildForm();
  }
  buildForm() {
    if (this.idx < 0) { this.form.replaceChildren(); return; }
    const im = store.project.images[this.idx];
    const cr = this.lens().crop;
    const setv = (k) => (v) => { cr[k] = v; this.commit(); };
    const rows = [];
    if (cr.mode === 'circle') {
      rows.push(['Center X', 'cx'], ['Center Y', 'cy'], ['Radius', 'radius']);
    } else if (cr.mode === 'rect') {
      rows.push(['Left', 'left'], ['Right', 'right'], ['Top', 'top'], ['Bottom', 'bottom']);
    }
    this.form.replaceChildren(
      h('div', { class: 'side-sec' }, h('h4', {}, 'Cropping'),
        select([['none', 'No crop'], ['rect', 'Rectangle'], ['circle', 'Circle (circular fisheye)']], cr.mode, (v) => {
          cr.mode = v;
          if (v === 'circle' && !cr.radius) { cr.cx = im.width / 2; cr.cy = im.height / 2; cr.radius = Math.min(im.width, im.height) / 2; }
          if (v === 'rect' && !cr.right) { cr.left = 0; cr.top = 0; cr.right = im.width; cr.bottom = im.height; }
          this.commit();
        }),
        ...rows.map(([l, k]) => h('div', { class: 'row' }, h('label', {}, l), numberInput(cr[k] || 0, setv(k), { digits: 1 }), h('span', { class: 'muted' }, 'px'))),
        h('div', { class: 'row' }, h('button', { class: 'btn small', onclick: () => store.post(`/api/image/${this.idx}/detect_crop`) }, 'Detect image circle')),
        h('p', { class: 'hint' }, `The crop belongs to lens ${im.lens} and applies to all ${store.project.images.filter((x) => x.lens === im.lens).length} images using it. Drag the outline to move it, drag its edge to resize.`)),
      h('div', { class: 'side-sec' }, h('h4', {}, 'Image'), h('div', { class: 'mono muted' }, `${im.name}\n${im.width} × ${im.height}`)));
  }
  overlay(g) {
    const cr = this.crop();
    if (!cr) return;
    const v = this.view;
    g.save();
    g.fillStyle = 'rgba(0,0,0,.55)';
    g.beginPath();
    g.rect(0, 0, v.wrap.clientWidth, v.wrap.clientHeight);
    if (cr.mode === 'circle') {
      const [sx, sy] = v.toScreen(cr.cx, cr.cy);
      g.arc(sx, sy, cr.radius * v.scale, 0, 2 * Math.PI, true);
    } else {
      const [x0, y0] = v.toScreen(cr.left, cr.top), [x1, y1] = v.toScreen(cr.right, cr.bottom);
      g.rect(x1, y0, x0 - x1, y1 - y0);
    }
    g.fill('evenodd');
    g.strokeStyle = '#3d8bfd'; g.lineWidth = 2;
    if (cr.mode === 'circle') {
      const [sx, sy] = v.toScreen(cr.cx, cr.cy);
      g.beginPath(); g.arc(sx, sy, cr.radius * v.scale, 0, 2 * Math.PI); g.stroke();
      g.beginPath(); g.moveTo(sx - 8, sy); g.lineTo(sx + 8, sy); g.moveTo(sx, sy - 8); g.lineTo(sx, sy + 8); g.stroke();
    } else {
      const [x0, y0] = v.toScreen(cr.left, cr.top), [x1, y1] = v.toScreen(cr.right, cr.bottom);
      g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }
    g.restore();
    g.fillStyle = '#ddd'; g.font = '12px system-ui';
    g.fillText(cr.mode === 'circle' ? `circle r=${fmt(cr.radius, 0)} px` : `${fmt(cr.right - cr.left, 0)} × ${fmt(cr.bottom - cr.top, 0)} px`, 10, 20);
  }
}

