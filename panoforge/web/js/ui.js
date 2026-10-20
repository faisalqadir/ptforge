// DOM helpers, dialogs, file browser, toasts and progress.
import { api, runJob } from './api.js';

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);
// replaceChildren that skips null/false like h() does
export function fill(root, ...kids) { root.replaceChildren(...kids.flat(Infinity).filter((k) => k !== null && k !== undefined && k !== false)); }
export const fmt = (v, d = 2) => (v === undefined || v === null || Number.isNaN(v)) ? '' : Number(v).toFixed(d);
export const icon = (paths) => {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 24 24');
  s.innerHTML = paths;
  return s;
};

// ------------------------------------------------------------------ toasts
let toastWrap;
export function toast(msg, kind = '') {
  if (!toastWrap) { toastWrap = h('div', { class: 'toast-wrap' }); document.body.append(toastWrap); }
  const t = h('div', { class: `toast ${kind}` }, msg);
  toastWrap.append(t);
  setTimeout(() => t.remove(), kind === 'err' ? 9000 : 4500);
}

export function status(text) {
  const s = document.getElementById('status-text');
  if (s) s.textContent = text;
}

// ------------------------------------------------------------------ modal
export function modal(title, body, buttons = [{ label: 'Close', value: null }]) {
  return new Promise((resolve) => {
    const back = h('div', { class: 'modal-back' });
    const close = (v) => { back.remove(); document.removeEventListener('keydown', onKey); resolve(v); };
    const onKey = (e) => {
      if (e.key === 'Escape') close(null);
      if (e.key === 'Enter' && !(e.target instanceof HTMLTextAreaElement)) {
        const def = buttons.find((b) => b.primary);
        if (def) close(typeof def.value === 'function' ? def.value() : def.value);
      }
    };
    const foot = h('footer', {}, buttons.map((b) => h('button', {
      class: `btn ${b.primary ? 'primary' : ''}`,
      onclick: () => close(typeof b.value === 'function' ? b.value() : b.value),
    }, b.label)));
    back.append(h('div', { class: 'modal' }, h('header', {}, title), h('div', { class: 'body' }, body), foot));
    back.addEventListener('mousedown', (e) => { if (e.target === back) close(null); });
    document.addEventListener('keydown', onKey);
    document.getElementById('overlay-root').append(back);
    const first = back.querySelector('input,select');
    if (first) setTimeout(() => first.focus(), 30);
  });
}

export async function prompt(title, label, value = '') {
  const inp = h('input', { type: 'text', value, style: { width: '100%' } });
  return modal(title, h('div', {}, h('div', { class: 'hint' }, label), inp),
    [{ label: 'Cancel', value: null }, { label: 'OK', primary: true, value: () => inp.value }]);
}

export async function confirmBox(title, text, ok = 'OK') {
  return modal(title, h('p', {}, text), [{ label: 'Cancel', value: false }, { label: ok, primary: true, value: true }]);
}

// ------------------------------------------------------------------ file browser (server side, local disk)
let lastDir = localStorageGet('pf.lastDir') || '~';
function localStorageGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function localStorageSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } }

// File dialogs: the native macOS panel (through the server / desktop app), with the in-app
// browser below as a fallback on systems that have no native dialog.
export async function fileBrowser(opts = {}) {
  const { title = 'Open', filter = 'images', multi = false, save = false, folder = false, defaultName = '' } = opts;
  try {
    const r = await api.post('/api/dialog', { kind: save ? 'save' : folder ? 'folder' : 'open', title, multi, filter, default_name: defaultName });
    if (!r.unsupported) return r.paths && r.paths.length ? r.paths : null;
  } catch (e) { /* fall back to the in-app browser */ }
  return inAppBrowser(opts);
}

function inAppBrowser({ title = 'Open', filter = 'images', multi = false, save = false, defaultName = '' } = {}) {
  return new Promise((resolve) => {
    let cur = lastDir;
    let selected = new Set();
    const pathInput = h('input', { type: 'text', style: { flex: 1 } });
    const nameInput = save ? h('input', { type: 'text', value: defaultName, style: { flex: 1 } }) : null;
    const list = h('div', { class: 'fb-list' });
    let entries = [];
    const load = async (path) => {
      try {
        const r = await api.get(`/api/fs/list?path=${encodeURIComponent(path)}&filter=${filter}`);
        cur = r.path; pathInput.value = r.path; entries = r.entries; selected = new Set();
        list.replaceChildren(
          h('div', { class: 'fb-item', ondblclick: () => load(r.parent) }, h('span', { class: 'ico' }, '↰'), '..'),
          ...r.entries.map((e) => {
            const item = h('div', { class: 'fb-item' }, h('span', { class: 'ico' }, e.dir ? '▸' : '·'), e.name);
            item.onclick = (ev) => {
              if (e.dir) return;
              if (!(multi && (ev.metaKey || ev.ctrlKey || ev.shiftKey))) {
                selected = new Set(); list.querySelectorAll('.sel').forEach((x) => x.classList.remove('sel'));
              }
              selected.add(e.name); item.classList.add('sel');
              if (nameInput) nameInput.value = e.name;
            };
            item.ondblclick = () => { if (e.dir) load(`${r.path}/${e.name}`); else finish([`${r.path}/${e.name}`]); };
            return item;
          }));
      } catch (e) { toast(e.message, 'err'); }
    };
    pathInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') load(pathInput.value); });
    const back = h('div', { class: 'modal-back' });
    const finish = (paths) => {
      back.remove();
      if (paths && paths.length) { lastDir = cur; localStorageSet('pf.lastDir', cur); }
      resolve(paths);
    };
    const selectAll = multi ? h('button', { class: 'btn small', onclick: () => {
      selected = new Set(entries.filter((e) => !e.dir).map((e) => e.name));
      list.querySelectorAll('.fb-item').forEach((x, k) => { if (k > 0 && !entries[k - 1].dir) x.classList.add('sel'); });
    } }, 'Select all') : null;
    const okBtn = h('button', { class: 'btn primary', onclick: () => {
      if (save) finish(nameInput.value ? [`${cur}/${nameInput.value}`] : null);
      else finish([...selected].map((n) => `${cur}/${n}`));
    } }, save ? 'Save' : 'Open');
    back.append(h('div', { class: 'modal', style: { width: '720px' } },
      h('header', {}, title),
      h('div', { class: 'body' },
        h('div', { class: 'row' }, pathInput, h('button', { class: 'btn small', onclick: () => load(pathInput.value) }, 'Go'), selectAll),
        list,
        save ? h('div', { class: 'row' }, h('label', {}, 'File name'), nameInput) : null),
      h('footer', {}, h('button', { class: 'btn', onclick: () => finish(null) }, 'Cancel'), okBtn)));
    document.getElementById('overlay-root').append(back);
    load(cur);
  });
}

// ------------------------------------------------------------------ progress dialog around a job
const STAGE_LABEL = {
  detect: 'Finding features', match: 'Matching image pairs', optimize: 'Optimizing', seams: 'Finding seams',
  coverage: 'Computing coverage', blend: 'Blending', exposure: 'Optimizing exposure', project: 'Project', done: 'Done',
};

export async function withProgress(title, url, body) {
  const label = h('div', {}, 'Starting…');
  const bar = h('div');
  const back = h('div', { class: 'modal-back' });
  let jobId = null;
  back.append(h('div', { class: 'modal progress-box' }, h('header', {}, title),
    h('div', { class: 'body' }, label, h('div', { class: 'bar' }, bar)),
    h('footer', {}, h('button', { class: 'btn', onclick: () => jobId && api.post(`/api/jobs/${jobId}/cancel`) }, 'Cancel'))));
  document.getElementById('overlay-root').append(back);
  const t0 = performance.now();
  try {
    const res = await runJob(url, body, (ev) => {
      jobId = jobId || ev.job;
      const name = STAGE_LABEL[ev.stage] || ev.stage;
      const frac = ev.n ? Math.min(1, (ev.i + 1) / ev.n) : 0;
      label.textContent = `${name}${ev.n > 1 ? ` (${ev.i + 1}/${ev.n})` : ''}${ev.message ? ' — ' + ev.message : ''}`;
      bar.style.width = `${(frac * 100).toFixed(0)}%`;
      status(`${title}: ${name}`);
    });
    status(`${title} finished in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    return res.result;
  } finally {
    back.remove();
  }
}

export function numberInput(value, onchange, { step = 'any', min, max, digits = 3, width } = {}) {
  const inp = h('input', { type: 'number', step, min, max, value: fmt(value, digits) });
  if (width) inp.style.width = width;
  inp.addEventListener('change', () => onchange(parseFloat(inp.value)));
  return inp;
}

export function select(options, value, onchange) {
  const s = h('select', {}, options.map(([v, l]) => h('option', { value: v, selected: String(v) === String(value) }, l)));
  s.addEventListener('change', () => onchange(s.value));
  return s;
}

export function checkbox(checked, onchange, label) {
  const c = h('input', { type: 'checkbox', checked: !!checked });
  c.addEventListener('change', () => onchange(c.checked));
  return label ? h('label', { style: { display: 'inline-flex', gap: '6px', alignItems: 'center', minWidth: 0 } }, c, label) : c;
}
