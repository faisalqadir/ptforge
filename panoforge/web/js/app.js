// PanoForge editor: application shell (menus, toolbar, tabs, shortcuts).
import { api } from './api.js';
import { CpEditor, cpTable } from './cpeditor.js';
import { PanoEditor } from './editor.js';
import { CropEditor, MaskEditor } from './imagetools.js';
import { store } from './store.js';
import * as T from './tabs.js';
import { $, confirmBox, fileBrowser, h, icon, modal, prompt, status, toast } from './ui.js';

try { document.documentElement.dataset.theme = localStorage.getItem('pf.theme') || 'dark'; } catch (e) { /* storage unavailable */ }

// ------------------------------------------------------------------ pages
const PAGES = [
  { group: 'Project' },
  { id: 'assistant', label: 'Project Assistant', render: T.renderAssistant },
  { id: 'sources', label: 'Source Images', render: T.renderSources, badge: (p) => p.images.length },
  { id: 'lens', label: 'Lens Settings', render: T.renderLens },
  { id: 'crop', label: 'Crop', full: true, cls: CropEditor },
  { id: 'mask', label: 'Mask', full: true, cls: MaskEditor },
  { id: 'imageparams', label: 'Image Parameters', render: T.renderImageParams },
  { group: 'Alignment' },
  { id: 'cps', label: 'Control Points', full: true, cls: CpEditor, badge: (p) => p.control_points.length },
  { id: 'optimizer', label: 'Optimizer', render: T.renderOptimizer },
  { id: 'editor', label: 'Panorama Editor', full: true, cls: PanoEditor },
  { group: 'Output' },
  { id: 'exposure', label: 'Exposure / HDR', render: T.renderExposure },
  { id: 'settings', label: 'Project Settings', render: T.renderSettings },
  { id: 'preview', label: 'Preview', render: T.renderPreview, once: true },
  { id: 'metadata', label: 'Metadata', render: T.renderMetadata },
  { id: 'create', label: 'Create Panorama', render: T.renderCreate },
  { id: 'batch', label: 'Batch Stitcher', render: T.renderBatch, once: true },
];
const pageEls = {};
const instances = {};
let active = 'assistant';

function buildPages() {
  const tabs = $('#tabs');
  const content = $('#content');
  for (const pg of PAGES) {
    if (pg.group) { tabs.append(h('div', { class: 'tab-group' }, pg.group)); continue; }
    pg.tabEl = h('div', { class: 'tab', onclick: () => show(pg.id) }, pg.label, pg.badge ? h('span', { class: 'badge' }) : null);
    tabs.append(pg.tabEl);
    const el = h('div', { class: `page ${pg.full ? 'full' : ''}`, id: `page-${pg.id}` });
    content.append(el);
    pageEls[pg.id] = el;
  }
}

function show(id) {
  active = id;
  for (const pg of PAGES) {
    if (pg.group) continue;
    pg.tabEl.classList.toggle('active', pg.id === id);
    pageEls[pg.id].classList.toggle('active', pg.id === id);
  }
  const pg = PAGES.find((x) => x.id === id);
  if (pg.cls && !instances[id]) {
    instances[id] = new pg.cls(pageEls[id]);
    if (store.project) instances[id].refresh?.();
  }
  if (pg.cls && instances[id]) {
    instances[id].refresh?.();
    instances[id].redraw?.();
    instances[id].syncSide?.();
  }
  if (pg.render && (!pg.once || !pg.rendered)) { pg.render(pageEls[id]); pg.rendered = true; }
  try { localStorage.setItem('pf.tab', id); } catch (e) { /* ignore */ }
}

function onProject(p) {
  for (const pg of PAGES) {
    if (pg.badge && pg.tabEl) pg.tabEl.querySelector('.badge').textContent = pg.badge(p);
  }
  const pg = PAGES.find((x) => x.id === active);
  if (pg && pg.render && !pg.once) pg.render(pageEls[active]);
  const name = p.path ? p.path.split('/').pop() : 'Untitled';
  document.title = `${p.dirty ? '• ' : ''}${name} — PanoForge`;
  $('#status-right').textContent = `${p.images.length} images · ${p.control_points.length} control points · ${p.panorama.projection} ${Math.round(p.panorama.hfov)}°×${Math.round(p.panorama.vfov)}°`;
  updateMenus(p);
}

// ------------------------------------------------------------------ file actions
async function newProject() {
  if (store.project?.dirty && !(await confirmBox('New project', 'Discard unsaved changes?', 'Discard'))) return;
  await store.post('/api/project/new');
  show('assistant');
}
async function openProject() {
  const r = await fileBrowser({ title: 'Open project', filter: 'projects' });
  if (!r || !r.length) return;
  await store.post('/api/project/open', { path: r[0] });
  show('assistant');
}
async function saveProject(as = false) {
  let path = store.project.path;
  if (as || !path) {
    const r = await fileBrowser({ title: 'Save project as', filter: 'projects', save: true, defaultName: path ? path.split('/').pop() : 'panorama.pfproj' });
    if (!r) return;
    path = r[0];
  }
  const res = await store.post('/api/project/save', { path });
  toast(`Saved ${res.saved}`, 'ok');
}
async function exportPto() {
  const r = await fileBrowser({ title: 'Export Hugin / Panorama Tools project', filter: 'all', save: true, defaultName: 'panorama.pto' });
  if (r) { await api.post('/api/project/save', { path: r[0].endsWith('.pto') ? r[0] : r[0] + '.pto' }); toast('Exported .pto', 'ok'); }
}

// ------------------------------------------------------------------ menus
let menuDefs;
function menus() {
  const sel = () => [...store.selection];
  return [
    ['File', [
      ['New Project', newProject, '⌘N'], ['Open Project…', openProject, '⌘O'], ['Save Project', () => saveProject(false), '⌘S'], ['Save Project As…', () => saveProject(true), '⇧⌘S'],
      '-', ['Load Images…', T.loadImagesDialog, '⌘L'], '-',
      ['Apply Template…', async () => { const r = await fileBrowser({ title: 'Apply template', filter: 'projects' }); if (r && r.length) store.post('/api/project/apply_template', { path: r[0] }); }],
      ['Export Hugin .pto…', exportPto]]],
    ['Edit', [['Undo', () => store.post('/api/undo'), '⌘Z', () => store.project?.can_undo], ['Redo', () => store.post('/api/redo'), '⇧⌘Z', () => store.project?.can_redo]]],
    ['View', [['Project Assistant', () => show('assistant')], ['Panorama Editor', () => show('editor'), '⌘E'], ['Control Points', () => show('cps'), '⌘K'], ['Preview', () => show('preview')]]],
    ['Images', [
      ['Include selected', () => store.patch({ images: Object.fromEntries(sel().map((i) => [i, { include: true }])) })],
      ['Exclude selected', () => store.patch({ images: Object.fromEntries(sel().map((i) => [i, { include: false }])) })],
      ['Include all', () => store.patch({ images: Object.fromEntries(store.project.images.map((_, i) => [i, { include: true }])) })],
      '-', ['Link selected (bracket stack)', () => store.post('/api/images/link', { indices: sel() })],
      ['Link HDR bracketed exposures (EXIF)', () => store.post('/api/images/link', { auto: true })],
      ['Unlink all', () => store.post('/api/images/link', { indices: store.project.images.map((_, i) => i), unlink: true })],
      '-', ['Remove excluded images', () => store.post('/api/images/remove', { indices: store.project.images.map((im, i) => (im.include ? -1 : i)).filter((i) => i >= 0) })]]],
    ['Control Points', [
      ['Generate control points', () => store.job('Generating control points', '/api/cps/generate', {})],
      ['Generate for overlapping images only', () => store.job('Generating control points', '/api/cps/generate', { prealigned: true })],
      ['Control point table…', () => modal('Control point table', cpTable((k) => { document.querySelector('.modal-back')?.remove(); show('cps'); jumpToCp(k); }))],
      '-', ['Delete worst control points…', async () => {
        const v = await prompt('Delete worst control points', 'Delete control points with a distance larger than (pixels):', '10');
        if (v) { const r = await store.post('/api/cps/delete_worst', { max_error: parseFloat(v) }); toast(`${r.removed} control points removed`, 'ok'); }
      }],
      ['Delete all control points', async () => { if (await confirmBox('Delete all', 'Delete every control point?', 'Delete')) store.post('/api/cps/delete', { all: true }); }]]],
    ['Project', [
      ['Align Images', () => T.alignImages(), '⌘R'], ['Optimize', () => store.job('Optimizing', '/api/optimize', {}), 'F5'],
      ['Initialize and Optimize', () => store.job('Optimizing', '/api/optimize', { reinit: true })],
      ['Optimize exposure / color', () => store.job('Optimizing exposure', '/api/photometric', {})],
      '-', ['Align to grid…', alignToGridDialog], ['Straighten panorama', () => store.post('/api/straighten')], ['Fit panorama', () => store.post('/api/fit', {})],
      '-', ['Create Panorama', async () => { const r = await T.createPanorama(); if (r) T.viewResult(r); }, '⌘B'],
      ['Save and send to Batch Stitcher', () => show('batch')]]],
    ['Tools', [
      ['Batch Stitcher', () => show('batch')],
      ['Convert equirectangular to cube faces…', () => convertDialog('cube')], ['Create little planet…', () => convertDialog('littleplanet')],
      ['Publish to website…', publishDialog], ['View a panorama…', async () => {
        const r = await fileBrowser({ title: 'View panorama', filter: 'images' });
        if (r && r.length) (await import('./viewer.js')).openViewer(`/api/output?path=${encodeURIComponent(r[0])}`, { title: r[0].split('/').pop() });
      }]]],
    ['Help', [['Keyboard shortcuts', shortcutsHelp], ['About PanoForge', () => modal('About PanoForge', h('div', {},
      h('p', {}, 'PanoForge — panorama stitching engine and editor.'),
      h('p', { class: 'muted' }, 'Clean-room implementation: lens model compatible with Panorama Tools parameters, SIFT control points, bundle adjustment, photometric optimization, graph-cut seams, multiband blending, HDR merging and exposure fusion.')))]]],
  ];
}

function buildMenus() {
  const bar = $('#menubar');
  bar.replaceChildren(h('div', { class: 'brand' }, 'PanoForge'));
  menuDefs = menus();
  for (const [title, items] of menuDefs) {
    const box = h('div', { class: 'menu-items' });
    const m = h('div', { class: 'menu' }, h('div', { class: 'menu-title' }, title), box);
    for (const it of items) {
      if (it === '-') { box.append(h('div', { class: 'menu-sep' })); continue; }
      const [label, fn, key, enabled] = it;
      const el = h('div', { class: 'menu-item', onclick: () => { closeMenus(); fn(); } }, h('span', {}, label), key ? h('span', { class: 'key' }, key) : null);
      el._enabled = enabled;
      box.append(el);
    }
    m.querySelector('.menu-title').addEventListener('mousedown', (e) => {
      e.stopPropagation();
      const open = m.classList.contains('open');
      closeMenus();
      if (!open) m.classList.add('open');
    });
    m.querySelector('.menu-title').addEventListener('mouseenter', () => {
      if (document.querySelector('.menu.open') && !m.classList.contains('open')) { closeMenus(); m.classList.add('open'); }
    });
    bar.append(m);
  }
  document.addEventListener('mousedown', closeMenus);
  bar.addEventListener('mousedown', (e) => e.stopPropagation());
}
function closeMenus() { document.querySelectorAll('.menu.open').forEach((m) => m.classList.remove('open')); }
function updateMenus() {
  document.querySelectorAll('.menu-item').forEach((el) => { if (el._enabled) el.classList.toggle('disabled', !el._enabled()); });
}

// ------------------------------------------------------------------ toolbar
const TB = {
  new: '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/>',
  open: '<path d="M3 6h6l2 2h10v11H3z"/>',
  save: '<path d="M5 3h12l2 2v16H5z"/><path d="M8 3v6h8V3M8 21v-7h8v7"/>',
  undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/>',
  redo: '<path d="m15 14 5-5-5-5"/><path d="M20 9H9a5 5 0 0 0 0 10h3"/>',
  images: '<rect x="3" y="5" width="14" height="12" rx="1"/><path d="M7 21h14V9"/><path d="m3 14 4-4 5 5"/>',
  align: '<rect x="3" y="7" width="8" height="10" rx="1"/><rect x="13" y="7" width="8" height="10" rx="1"/><path d="M11 12h2"/>',
  optimize: '<path d="M4 20 10 9l4 6 6-11"/>',
  editor: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
  stitch: '<path d="M4 7h16v10H4z"/><path d="M9 7v10M15 7v10"/>',
};
function buildToolbar() {
  const b = (ic, label, fn, title, cls = '') => h('button', { class: `tb-btn ${cls}`, title: title || label, onclick: fn }, icon(TB[ic]), label);
  $('#toolbar').replaceChildren(
    b('new', '', newProject, 'New project'), b('open', '', openProject, 'Open project'), b('save', '', () => saveProject(false), 'Save project'),
    h('span', { class: 'tb-sep' }), b('undo', '', () => store.post('/api/undo'), 'Undo'), b('redo', '', () => store.post('/api/redo'), 'Redo'),
    h('span', { class: 'tb-sep' }),
    b('images', 'Load images', T.loadImagesDialog),
    b('align', 'Align images', () => T.alignImages()),
    b('optimize', 'Optimize', () => store.job('Optimizing', '/api/optimize', {})),
    b('editor', 'Panorama Editor', () => show('editor')),
    h('span', { class: 'tb-spacer' }),
    b('stitch', 'Create Panorama', async () => { const r = await T.createPanorama(); if (r) T.viewResult(r); }, 'Stitch with the current settings', 'primary'));
}

// ------------------------------------------------------------------ dialogs
async function alignToGridDialog() {
  const p = store.project;
  const f = { columns: p.images.length, rows: 1, yaw0: 0, yaw_step: 360 / Math.max(1, p.images.length), pitch0: 0, pitch_step: -30, order: 'row', snake: false };
  const inp = (k) => { const i = h('input', { type: 'number', value: f[k], step: 'any' }); i.addEventListener('change', () => { f[k] = parseFloat(i.value); }); return i; };
  const body = h('div', { class: 'grid2' },
    h('label', {}, 'Columns × rows'), h('div', { class: 'row' }, inp('columns'), '×', inp('rows')),
    h('label', {}, 'First yaw / spacing'), h('div', { class: 'row' }, inp('yaw0'), inp('yaw_step')),
    h('label', {}, 'First pitch / spacing'), h('div', { class: 'row' }, inp('pitch0'), inp('pitch_step')));
  const ok = await modal('Align to grid', body, [{ label: 'Cancel', value: false }, { label: 'Apply', primary: true, value: true }]);
  if (ok) { await store.post('/api/align_to_grid', f); toast('Images positioned on the grid. Use "Generate control points for overlapping images" next.', 'ok'); }
}

async function convertDialog(kind) {
  const r = await fileBrowser({ title: 'Select an equirectangular panorama', filter: 'images' });
  if (!r || !r.length) return;
  status('Converting…');
  const res = await api.post('/api/convert', { path: r[0], to: kind });
  toast(`Written: ${res.files.map((f) => f.split('/').pop()).join(', ')}`, 'ok');
}

async function publishDialog() {
  const r = await fileBrowser({ title: 'Select an equirectangular panorama to publish', filter: 'images' });
  if (!r || !r.length) return;
  const res = await api.post('/api/publish', { path: r[0], title: (store.project.metadata || {}).title || 'Panorama' });
  toast(`Website folder: ${res.folder}`, 'ok');
}

function shortcutsHelp() {
  const rows = [['⌘O / ⌘S', 'Open / save project'], ['⌘Z / ⇧⌘Z', 'Undo / redo'], ['⌘L', 'Load images'], ['⌘R', 'Align images'], ['F5', 'Optimize'],
    ['⌘E', 'Panorama Editor'], ['⌘K', 'Control points'], ['⌘B', 'Create panorama'],
    ['Editor: drag', 'Move panorama (or image, in individual mode)'], ['Editor: right-drag / Alt-drag', 'Roll'], ['Editor: wheel, Shift-drag', 'Zoom, pan'],
    ['Control points: click', 'Add point (Auto add places the match)'], ['Control points: Delete', 'Delete selected point'], ['Control points: Alt+← / →', 'Previous / next image pair']];
  modal('Keyboard shortcuts', h('table', { class: 'grid' }, h('tbody', {}, rows.map(([k, v]) => h('tr', {}, h('td', { class: 'mono' }, k), h('td', {}, v))))));
}

function jumpToCp(k) {
  const cp = store.project.control_points[k];
  const ed = instances.cps;
  if (!ed || !cp) return;
  ed.left.setImage(cp.i); ed.right.setImage(cp.j);
  setTimeout(() => ed.selectCp(k, true), 300);
}

// ------------------------------------------------------------------ shortcuts
document.addEventListener('keydown', (e) => {
  const mod = e.metaKey || e.ctrlKey;
  const k = e.key.toLowerCase();
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) { if (!(mod && k === 's')) return; }
  if (mod && k === 's') { e.preventDefault(); saveProject(e.shiftKey); }
  else if (mod && k === 'o') { e.preventDefault(); openProject(); }
  else if (mod && k === 'z') { e.preventDefault(); store.post(e.shiftKey ? '/api/redo' : '/api/undo'); }
  else if (mod && k === 'l') { e.preventDefault(); T.loadImagesDialog(); }
  else if (mod && k === 'r') { e.preventDefault(); T.alignImages(); }
  else if (mod && k === 'e') { e.preventDefault(); show('editor'); }
  else if (mod && k === 'k') { e.preventDefault(); show('cps'); }
  else if (mod && k === 'b') { e.preventDefault(); T.createPanorama().then((r) => r && T.viewResult(r)); }
  else if (e.key === 'F5') { e.preventDefault(); store.job('Optimizing', '/api/optimize', {}); }
});
document.addEventListener('pf-tab', (e) => show(e.detail));
window.addEventListener('beforeunload', (e) => { if (store.project?.dirty) { e.preventDefault(); e.returnValue = ''; } });

// hooks for the desktop app (files dropped on the window, errors from native code)
window.pfRefresh = () => store.refresh().then(() => show(active));
window.pfToast = (msg, kind) => toast(msg, kind);
// let a drop anywhere in the window add images (desktop app handles the paths natively)
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => { if (window.pywebview || !e.dataTransfer?.files?.length) e.preventDefault(); });

async function offerRecovery() {
  try {
    const r = await api.get('/api/recovery');
    if (!r.exists || store.project.images.length) return;
    const when = new Date(r.saved * 1000).toLocaleString();
    const ok = await modal('Recover unsaved project?', h('p', {}, `PanoForge was closed with unsaved changes (${r.images} images, ${when}).`),
      [{ label: 'Discard', value: false }, { label: 'Recover', primary: true, value: true }]);
    if (ok) { await store.post('/api/recovery/restore'); toast('Project recovered — remember to save it', 'ok'); show('assistant'); }
    else await api.post('/api/recovery/discard');
  } catch (e) { /* recovery is best effort */ }
}

// ------------------------------------------------------------------ start
buildMenus();
buildToolbar();
buildPages();
store.onChange(onProject);
let start = 'assistant';
try { start = localStorage.getItem('pf.tab') || 'assistant'; } catch (e) { /* ignore */ }
store.refresh().then(() => { show(PAGES.some((p) => p.id === start) ? start : 'assistant'); offerRecovery(); })
  .catch((e) => toast(`Cannot reach the PanoForge server: ${e.message}`, 'err'));
