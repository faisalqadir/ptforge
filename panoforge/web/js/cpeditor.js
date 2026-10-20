// Control point editor: two zoomable panes with add / move / delete / auto-add / fine-tune.
import { api } from './api.js';
import { store } from './store.js';
import { checkbox, fmt, h, select, status, toast } from './ui.js';

const cpColor = (k) => `hsl(${(k * 67) % 360} 85% 60%)`;
const errColor = (e) => (e < 3 ? '#3fb950' : e < 8 ? '#d29922' : '#f85149');

class ImagePane {
  constructor(side, editor) {
    this.side = side; // 0 left, 1 right
    this.ed = editor;
    this.idx = -1;
    this.scale = 1;
    this.ox = 0; this.oy = 0; // image coordinate at the canvas top-left
    this.img = null; this.full = null;
    this.strip = h('div', { class: 'strip' });
    this.canvas = h('canvas');
    this.wrap = h('div', { class: 'editor-wrap' }, this.canvas);
    this.mag = h('div', { class: 'magnifier' }, this.magCanvas = h('canvas', { width: 180, height: 180 }));
    this.wrap.append(this.mag);
    this.el = h('div', { class: 'cp-pane' }, this.strip, this.wrap);
    new ResizeObserver(() => this.draw()).observe(this.wrap);
    this.bind();
  }

  buildStrip() {
    const p = store.project;
    this.strip.replaceChildren(...p.images.map((im, i) => {
      const s = h('div', { class: `s ${i === this.idx ? 'sel' : ''}`, title: im.name, onclick: () => this.ed.select(this.side, i) },
        h('img', { src: store.imgUrl(i, 'thumb', 160), loading: 'lazy' }), h('span', {}, i));
      return s;
    }));
  }

  setImage(i) {
    if (i === this.idx && this.img) { this.buildStrip(); return; }
    this.idx = i;
    this.buildStrip();
    const im = store.project.images[i];
    this.img = null; this.full = null;
    const tex = new Image();
    tex.onload = () => { if (this.idx === i) { this.img = tex; if (!this.fitted) this.fit(); this.draw(); } };
    tex.src = store.imgUrl(i, 'texture', 2048);
    this.fitted = false;
  }

  needFull() {
    if (this.full || this.loadingFull || this.idx < 0) return;
    const im = store.project.images[this.idx];
    if (this.scale * Math.max(im.width, im.height) < 2048 * 1.2) return;
    this.loadingFull = true;
    const i = this.idx;
    const f = new Image();
    f.onload = () => { this.loadingFull = false; status('Full resolution loaded'); if (this.idx === i) { this.full = f; this.draw(); } };
    f.onerror = () => { this.loadingFull = false; };
    f.src = store.imgUrl(i, 'full');
    status('Loading full resolution…');
  }

  fit() {
    const im = store.project.images[this.idx];
    if (!im) return;
    const W = this.wrap.clientWidth, H = this.wrap.clientHeight;
    this.scale = Math.min(W / im.width, H / im.height) * 0.98;
    this.ox = (im.width - W / this.scale) / 2;
    this.oy = (im.height - H / this.scale) / 2;
    this.fitted = true;
  }

  toImage(sx, sy) { return [this.ox + sx / this.scale - 0.5, this.oy + sy / this.scale - 0.5]; }
  toScreen(x, y) { return [(x + 0.5 - this.ox) * this.scale, (y + 0.5 - this.oy) * this.scale]; }

  centerOn(x, y, zoom) {
    if (zoom) this.scale = zoom;
    this.ox = x + 0.5 - this.wrap.clientWidth / 2 / this.scale;
    this.oy = y + 0.5 - this.wrap.clientHeight / 2 / this.scale;
    this.draw();
  }

  bind() {
    const c = this.canvas;
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const [x, y] = this.toImage(e.offsetX, e.offsetY);
      const k = Math.exp(-e.deltaY * 0.0018);
      this.scale = Math.max(0.02, Math.min(32, this.scale * k));
      this.ox = x + 0.5 - e.offsetX / this.scale;
      this.oy = y + 0.5 - e.offsetY / this.scale;
      this.draw();
      if (this.ed.linkViews) this.ed.syncOther(this);
    }, { passive: false });
    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      const [x, y] = this.toImage(e.offsetX, e.offsetY);
      if (e.button === 1 || e.button === 2 || e.shiftKey) { this.drag = { kind: 'pan', sx: e.offsetX, sy: e.offsetY, ox: this.ox, oy: this.oy }; return; }
      const hit = this.ed.hitTest(this, e.offsetX, e.offsetY);
      if (hit >= 0) { this.ed.selectCp(hit); this.drag = { kind: 'cp', k: hit, moved: false }; return; }
      this.ed.click(this, x, y);
    });
    c.addEventListener('pointermove', (e) => {
      const [x, y] = this.toImage(e.offsetX, e.offsetY);
      this.ed.hoverXY = [this.side, x, y];
      if (this.drag && this.drag.kind === 'pan') {
        this.ox = this.drag.ox - (e.offsetX - this.drag.sx) / this.scale;
        this.oy = this.drag.oy - (e.offsetY - this.drag.sy) / this.scale;
        this.draw();
        if (this.ed.linkViews) this.ed.syncOther(this);
      } else if (this.drag && this.drag.kind === 'cp') {
        this.drag.moved = true;
        this.ed.moveCpLocal(this, this.drag.k, x, y);
      }
      this.magnify(e.offsetX, e.offsetY);
    });
    c.addEventListener('pointerup', () => {
      const d = this.drag;
      this.drag = null;
      if (d && d.kind === 'cp' && d.moved) this.ed.commitMove(d.k);
    });
    c.addEventListener('pointerleave', () => { this.mag.style.display = 'none'; });
  }

  magnify(sx, sy) {
    if (!this.ed.magnifier || !this.img) { this.mag.style.display = 'none'; return; }
    const m = this.mag;
    m.style.display = 'block';
    const W = this.wrap.clientWidth;
    m.style.left = `${sx + 200 > W ? sx - 200 : sx + 20}px`;
    m.style.top = `${Math.max(0, sy - 90)}px`;
    const g = this.magCanvas.getContext('2d');
    const src = this.full || this.img;
    const im = store.project.images[this.idx];
    const k = src.naturalWidth / im.width;
    const [x, y] = this.toImage(sx, sy);
    const R = 22; // source pixels shown on each side of the cursor
    g.imageSmoothingEnabled = false;
    g.fillStyle = '#000'; g.fillRect(0, 0, 180, 180);
    g.drawImage(src, (x + 0.5 - R) * k, (y + 0.5 - R) * k, 2 * R * k, 2 * R * k, 0, 0, 180, 180);
    g.strokeStyle = 'rgba(255,255,255,.8)'; g.beginPath(); g.moveTo(90, 80); g.lineTo(90, 100); g.moveTo(80, 90); g.lineTo(100, 90); g.stroke();
  }

  draw() {
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const W = this.wrap.clientWidth, H = this.wrap.clientHeight;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#111'; g.fillRect(0, 0, W, H);
    if (this.idx < 0 || !store.project.images[this.idx]) return;
    if (!this.fitted && this.img) this.fit();
    const im = store.project.images[this.idx];
    const src = this.full || this.img;
    if (src) {
      g.imageSmoothingEnabled = this.scale < 2;
      g.drawImage(src, -this.ox * this.scale, -this.oy * this.scale, im.width * this.scale, im.height * this.scale);
      this.needFull();
    }
    this.ed.drawMarkers(this, g);
  }
}

export class CpEditor {
  constructor(root) {
    this.root = root;
    this.left = new ImagePane(0, this);
    this.right = new ImagePane(1, this);
    this.sel = -1;
    this.pending = null; // {side, x, y}
    this.autoAdd = true;
    this.finetune = true;
    this.magnifier = true;
    this.linkViews = false;
    this.kind = 'normal';
    this.lineId = 0;
    this.showAll = false;
    this.build();
    store.onChange(() => this.refresh());
    document.addEventListener('keydown', (e) => {
      if (!this.root.offsetParent) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if ((e.key === 'Delete' || e.key === 'Backspace') && this.sel >= 0) { this.deleteSel(); e.preventDefault(); }
      if (e.key === 'Escape') { this.pending = null; this.drawAll(); }
      if (e.key === 'ArrowRight' && e.altKey) this.stepPair(1);
      if (e.key === 'ArrowLeft' && e.altKey) this.stepPair(-1);
    });
  }

  build() {
    const bar = h('div', { class: 'editor-bar' },
      h('span', { class: 'muted' }, 'CP type to add:'),
      select([['normal', 'Normal'], ['vertical', 'Vertical line'], ['horizontal', 'Horizontal line'], ['line', 'Straight line (new)']], this.kind, (v) => {
        this.kind = v;
        if (v === 'line') this.lineId = 3 + Math.max(0, ...store.project.control_points.filter((c) => c.kind === 'line').map((c) => c.line - 2));
      }),
      h('span', { class: 'tb-sep' }),
      checkbox(this.autoAdd, (v) => { this.autoAdd = v; }, 'Auto add'),
      checkbox(this.finetune, (v) => { this.finetune = v; }, 'Fine tune'),
      checkbox(this.magnifier, (v) => { this.magnifier = v; }, 'Magnifier'),
      checkbox(this.linkViews, (v) => { this.linkViews = v; }, 'Link views'),
      h('span', { class: 'tb-sep' }),
      h('button', { class: 'tb-btn', onclick: () => this.generatePair() }, 'Generate CPs for this pair'),
      h('button', { class: 'tb-btn', onclick: () => this.deletePair() }, 'Delete pair CPs'),
      h('button', { class: 'tb-btn', onclick: () => this.stepPair(1), title: 'Next pair with control points (Alt+→)' }, 'Next pair →'),
      h('span', { class: 'tb-spacer' }),
      this.info = h('span', { class: 'muted mono' }));
    this.table = h('div', { class: 'cp-bottom' });
    this.root.append(bar, h('div', { class: 'cp-panes' }, this.left.el, this.right.el), this.table);
  }

  refresh() {
    const p = store.project;
    if (!p) return;
    const n = p.images.length;
    if (!n) { this.left.idx = this.right.idx = -1; this.drawAll(); this.buildTable(); return; }
    if (this.left.idx < 0 || this.left.idx >= n) this.left.setImage(0);
    if (this.right.idx < 0 || this.right.idx >= n) this.right.setImage(Math.min(1, n - 1));
    this.left.buildStrip(); this.right.buildStrip();
    if (this.sel >= p.control_points.length) this.sel = -1;
    this.buildTable();
    this.drawAll();
  }

  select(side, i) {
    (side === 0 ? this.left : this.right).setImage(i);
    this.pending = null;
    this.sel = -1;
    this.buildTable();
    this.drawAll();
  }

  drawAll() { this.left.draw(); this.right.draw(); }

  syncOther(pane) {
    const o = pane === this.left ? this.right : this.left;
    o.scale = pane.scale; o.ox = pane.ox; o.oy = pane.oy; o.draw();
  }

  pairCps() {
    // indices of CPs belonging to the current pair (either orientation) plus same-image line CPs
    const p = store.project;
    const a = this.left.idx, b = this.right.idx;
    const out = [];
    p.control_points.forEach((cp, k) => {
      if ((cp.i === a && cp.j === b) || (cp.i === b && cp.j === a)) out.push(k);
    });
    return out;
  }

  // coordinates of CP k on a given pane, or null
  cpOn(pane, k) {
    const cp = store.project.control_points[k];
    const other = pane === this.left ? this.right : this.left;
    if (cp.i === cp.j) return pane.side === 0 ? [cp.x1, cp.y1] : [cp.x2, cp.y2]; // both points in one image
    if (cp.i === pane.idx && cp.j === other.idx) return [cp.x1, cp.y1];
    if (cp.j === pane.idx && cp.i === other.idx) return [cp.x2, cp.y2];
    return null;
  }

  drawMarkers(pane, g) {
    const p = store.project;
    const ks = this.pairCps();
    for (const k of ks) {
      const xy = this.cpOn(pane, k);
      if (!xy) continue;
      const cp = p.control_points[k];
      const [sx, sy] = pane.toScreen(xy[0], xy[1]);
      const sel = k === this.sel;
      const col = cpColor(k);
      g.lineWidth = sel ? 3 : 2;
      g.strokeStyle = '#000';
      g.beginPath(); g.arc(sx, sy, sel ? 9 : 7, 0, 2 * Math.PI); g.stroke();
      g.strokeStyle = col;
      g.lineWidth = sel ? 2 : 1.5;
      g.beginPath(); g.arc(sx, sy, sel ? 9 : 7, 0, 2 * Math.PI); g.stroke();
      g.beginPath(); g.moveTo(sx - 3, sy); g.lineTo(sx + 3, sy); g.moveTo(sx, sy - 3); g.lineTo(sx, sy + 3); g.stroke();
      const lbl = cp.kind === 'normal' ? String(k) : `${k}${cp.kind[0]}`;
      g.font = '600 11px system-ui';
      g.fillStyle = '#000'; g.fillText(lbl, sx + 11, sy - 7);
      g.fillStyle = sel ? '#fff' : col; g.fillText(lbl, sx + 10, sy - 8);
      if (cp.error > 0 && cp.kind === 'normal') {
        g.fillStyle = errColor(cp.error);
        g.beginPath(); g.arc(sx + 7, sy + 7, 2.5, 0, 2 * Math.PI); g.fill();
      }
    }
    if (this.pending && this.pending.side === pane.side) {
      const [sx, sy] = pane.toScreen(this.pending.x, this.pending.y);
      g.strokeStyle = '#fff'; g.setLineDash([3, 3]); g.lineWidth = 1.5;
      g.beginPath(); g.arc(sx, sy, 10, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]);
    }
    this.info.textContent = `${this.left.idx} ↔ ${this.right.idx}: ${ks.length} control points`;
  }

  hitTest(pane, sx, sy) {
    let best = -1, bd = 12;
    for (const k of this.pairCps()) {
      const xy = this.cpOn(pane, k);
      if (!xy) continue;
      const [x, y] = pane.toScreen(xy[0], xy[1]);
      const d = Math.hypot(x - sx, y - sy);
      if (d < bd) { bd = d; best = k; }
    }
    return best;
  }

  selectCp(k, center = false) {
    this.sel = k;
    this.pending = null;
    if (center && k >= 0) {
      for (const pane of [this.left, this.right]) {
        const xy = this.cpOn(pane, k);
        if (xy) pane.centerOn(xy[0], xy[1], Math.max(pane.scale, 0.6));
      }
    }
    this.table.querySelectorAll('tr').forEach((tr) => tr.classList.toggle('sel', Number(tr.dataset.k) === k));
    this.drawAll();
  }

  async click(pane, x, y) {
    const p = store.project;
    const im = p.images[pane.idx];
    if (!im || x < -0.5 || y < -0.5 || x > im.width - 0.5 || y > im.height - 0.5) return;
    const other = pane === this.left ? this.right : this.left;
    if (this.pending && this.pending.side !== pane.side) {
      // second click completes the control point
      const a = this.pending.side === 0 ? this.pending : { x, y };
      const b = this.pending.side === 0 ? { x, y } : this.pending;
      this.pending = null;
      await this.addCp(this.left.idx, a.x, a.y, this.right.idx, b.x, b.y);
      return;
    }
    this.pending = { side: pane.side, x, y };
    this.sel = -1;
    this.drawAll();
    // only normal CPs between two different images can be predicted; line CPs need two user points
    if (!this.autoAdd || this.kind !== 'normal' || this.left.idx === this.right.idx) return;
    // auto add: predict the matching point from the current alignment, then fine-tune it
    try {
      const pr = await api.post('/api/cps/predict', { i: pane.idx, x, y, j: other.idx });
      if (!pr.valid) { status('Point is outside the other image according to the current alignment — click it manually'); return; }
      let nx = pr.x, ny = pr.y;
      if (this.finetune) {
        const ft = await api.post('/api/cps/finetune', { i: pane.idx, x1: x, y1: y, j: other.idx, x2: nx, y2: ny });
        if (ft.score > 0.6) { nx = ft.x2; ny = ft.y2; } else status(`Fine tune uncertain (correlation ${fmt(ft.score, 2)}) — check the point`);
      }
      this.pending = null;
      const L = pane === this.left ? { x, y } : { x: nx, y: ny };
      const R = pane === this.left ? { x: nx, y: ny } : { x, y };
      await this.addCp(this.left.idx, L.x, L.y, this.right.idx, R.x, R.y);
      other.centerOn(nx, ny);
    } catch (e) { toast(e.message, 'err'); }
  }

  async addCp(i, x1, y1, j, x2, y2) {
    const body = { i, x1, y1, j, x2, y2, kind: this.kind, line: this.kind === 'line' ? this.lineId : 0 };
    const r = await store.post('/api/cps/add', body);
    this.sel = r.index;
    this.buildTable();
    this.drawAll();
  }

  moveCpLocal(pane, k, x, y) {
    const cp = store.project.control_points[k];
    const other = pane === this.left ? this.right : this.left;
    const first = cp.i === cp.j ? pane.side === 0 : (cp.i === pane.idx && cp.j === other.idx);
    if (first) { cp.x1 = x; cp.y1 = y; } else { cp.x2 = x; cp.y2 = y; }
    pane.draw();
  }

  commitMove(k) {
    const cp = store.project.control_points[k];
    store.call('patch', `/api/cps/${k}`, { x1: cp.x1, y1: cp.y1, x2: cp.x2, y2: cp.y2 });
  }

  deleteSel() {
    const k = this.sel;
    this.sel = -1;
    store.post('/api/cps/delete', { indices: [k] });
  }

  deletePair() { store.post('/api/cps/delete', { pair: [this.left.idx, this.right.idx] }); }

  async generatePair() {
    await store.job('Generating control points', '/api/cps/generate', { indices: [this.left.idx, this.right.idx], replace: true });
  }

  stepPair(dir) {
    const p = store.project;
    const pairs = [...new Set(p.control_points.map((c) => `${Math.min(c.i, c.j)},${Math.max(c.i, c.j)}`))].map((s) => s.split(',').map(Number))
      .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (!pairs.length) return;
    const cur = pairs.findIndex(([a, b]) => a === Math.min(this.left.idx, this.right.idx) && b === Math.max(this.left.idx, this.right.idx));
    const nxt = pairs[(cur + dir + pairs.length) % pairs.length];
    this.left.setImage(nxt[0]); this.right.setImage(nxt[1]);
    this.sel = -1; this.buildTable(); this.drawAll();
  }

  buildTable() {
    const p = store.project;
    if (!p) return;
    const ks = this.pairCps();
    const rows = ks.map((k) => {
      const cp = p.control_points[k];
      const tr = h('tr', { 'data-k': k, class: k === this.sel ? 'sel' : '', onclick: () => this.selectCp(k, true) },
        h('td', {}, h('span', { style: { color: cpColor(k) } }, '●'), ` ${k}`),
        h('td', {}, cp.i), h('td', { class: 'mono' }, `${fmt(cp.x1, 1)}, ${fmt(cp.y1, 1)}`),
        h('td', {}, cp.j), h('td', { class: 'mono' }, `${fmt(cp.x2, 1)}, ${fmt(cp.y2, 1)}`),
        h('td', {}, cp.kind + (cp.kind === 'line' ? ` ${cp.line}` : '')),
        h('td', { class: 'mono', style: { color: errColor(cp.error) } }, fmt(cp.error, 2)));
      return tr;
    });
    this.table.replaceChildren(h('table', { class: 'grid' },
      h('thead', {}, h('tr', {}, ['#', 'Image', 'Position', 'Image', 'Position', 'Type', 'Distance (px)'].map((t) => h('th', {}, t)))),
      h('tbody', {}, rows.length ? rows : h('tr', {}, h('td', { colspan: 7, class: 'muted' },
        'No control points for this pair. Click a feature in either image to add one (Auto add places the match), or generate them automatically.')))));
  }
}

// Table of all control points (Control Point Table window)
export function cpTable(onJump) {
  const p = store.project;
  const order = p.control_points.map((c, k) => k).sort((a, b) => p.control_points[b].error - p.control_points[a].error);
  return h('div', { class: 'table-wrap', style: { maxHeight: '60vh' } }, h('table', { class: 'grid' },
    h('thead', {}, h('tr', {}, ['#', 'Images', 'Type', 'Distance'].map((t) => h('th', {}, t)))),
    h('tbody', {}, order.map((k) => {
      const c = p.control_points[k];
      return h('tr', { onclick: () => onJump(k) }, h('td', {}, k), h('td', {}, `${c.i} ↔ ${c.j}`), h('td', {}, c.kind),
        h('td', { class: 'mono', style: { color: errColor(c.error) } }, fmt(c.error, 2)));
    }))));
}
