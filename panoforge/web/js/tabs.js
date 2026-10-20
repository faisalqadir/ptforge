// The main window tabs (everything except the editors).
import { api } from './api.js';
import { PROJ_LABELS, PROJ_MAX } from './geom.js';
import { store } from './store.js';
import { checkbox, confirmBox, fileBrowser, fill, fmt, h, modal, numberInput, prompt, select, status, toast } from './ui.js';
import { openViewer } from './viewer.js';

const P = () => store.project;
const card = (...k) => h('div', { class: 'card' }, ...k);
const row = (label, ...k) => h('div', { class: 'row' }, h('label', {}, label), ...k);
const LENS_TYPES = [['rectilinear', 'Rectilinear (normal lens)'], ['fisheye', 'Fisheye (full frame)'], ['circular_fisheye', 'Circular fisheye'],
  ['cylindrical', 'Cylindrical panorama'], ['equirectangular', 'Equirectangular panorama']];
const FISHEYE_PRESETS = [['0', 'Equidistant (k=0)'], ['0.5', 'Equisolid (k=0.5)'], ['1', 'Orthographic (k=1)'], ['-0.5', 'Stereographic (k=-0.5)']];

export function goTab(name) { document.dispatchEvent(new CustomEvent('pf-tab', { detail: name })); }

// ------------------------------------------------------------------ actions shared by several tabs
export async function loadImagesDialog() {
  const paths = await fileBrowser({ title: 'Load images', filter: 'images', multi: true });
  if (paths && paths.length) {
    status('Reading images…');
    await store.post('/api/images/add', { paths });
    status(`${paths.length} images added`);
  }
}

export async function alignImages(opts = {}) {
  if (!P().images.length) { toast('Load images first', 'err'); return; }
  const s = (P().metadata || {}).align_behavior || {};
  const res = await store.job('Aligning images', '/api/align', { prealigned: !!s.prealigned, photometric: s.exposure !== false, max_cps: s.max_cps || 30, ...opts });
  if (res) {
    const msg = `${res.connected}/${res.n_images} images aligned · ${res.n_cps} control points · mean ${fmt(res.mean_error, 2)} px, max ${fmt(res.max_error, 2)} px`;
    toast(msg, res.unconnected.length ? 'err' : 'ok');
    status(msg);
  }
  return res;
}

export async function createPanorama(extra = {}) {
  const p = P();
  if (!p.images.length) { toast('Nothing to stitch', 'err'); return null; }
  const res = await store.job('Creating panorama', '/api/stitch', extra);
  if (res) {
    toast(`Panorama saved: ${res.path} (${res.width}×${res.height}, ${fmt(res.seconds, 1)} s)`, 'ok');
    lastResult = res;
    document.dispatchEvent(new CustomEvent('pf-result', { detail: res }));
  }
  return res;
}
export let lastResult = null;

export function viewResult(res) {
  const p = P();
  const crop = p.panorama.crop;
  const eq = res.projection === 'equirectangular';
  openViewer(`/api/output?path=${encodeURIComponent(res.path)}&t=${Date.now()}`, {
    projection: res.projection,
    hfov: res.hfov * (crop[2] - crop[0]), vfov: res.vfov * (crop[3] - crop[1]),
    offLon: eq ? ((crop[0] + crop[2]) / 2 - 0.5) * res.hfov : 0, offLat: eq ? (0.5 - (crop[1] + crop[3]) / 2) * res.vfov : 0,
    title: res.path.split('/').pop(),
  });
}

// ------------------------------------------------------------------ Project assistant
export function renderAssistant(root) {
  const p = P();
  const n = p.images.length;
  const aligned = p.control_points.length > 0;
  const ln = p.lenses[0];
  const drop = h('div', { class: 'dropzone' }, '…or drag & drop images here');
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', async (e) => {
    e.preventDefault(); drop.classList.remove('over');
    if (window.pywebview) return; // desktop app: the app adds dropped files by their real paths
    const fd = new FormData();
    for (const f of e.dataTransfer.files) fd.append('files', f, f.name);
    status('Uploading images…');
    const r = await fetch('/api/images/upload', { method: 'POST', body: fd });
    if (!r.ok) { toast(await r.text(), 'err'); return; }
    store.set((await r.json()).project);
  });
  const groups = new Set(p.images.filter((im) => im.link_group >= 0).map((im) => im.link_group));
  const unconnected = p.images.map((im, i) => i).filter((i) => p.images[i].include && p.images[i].n_cps === 0);
  const errs = p.control_points.map((c) => c.error).filter((e) => e > 0);
  const mean = errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : 0;
  const max = errs.length ? Math.max(...errs) : 0;
  const step = (k, state, title, ...body) => h('div', { class: `step ${state}` }, h('div', { class: 'n' }, state === 'done' ? '✓' : k), h('div', {}, h('h3', {}, title), ...body));
  fill(root, 
    h('h1', {}, 'Project Assistant'),
    h('p', { class: 'hint' }, 'Load the images, align them, then create the panorama. Every step can be refined in the other tabs.'),
    h('div', { class: 'steps' },
      step(1, n ? 'done' : 'cur', 'Load images',
        h('div', { class: 'row' }, h('button', { class: 'btn primary', onclick: loadImagesDialog }, n ? 'Add more images…' : 'Load images…'),
          n ? h('button', { class: 'btn', onclick: () => goTab('sources') }, 'Remove or reorder images') : null),
        n ? h('div', {}, h('div', {}, `${n} images · ${p.lenses.length} lens${p.lenses.length > 1 ? 'es' : ''}`),
          ln ? h('div', { class: 'muted' }, `Lens: ${(LENS_TYPES.find((t) => t[0] === ln.lens_type) || [0, ln.lens_type])[1]}, ` +
            `horizontal FoV ${fmt(ln.hfov, 1)}°${ln.focal_mm ? `, ${fmt(ln.focal_mm, 1)} mm` : ''}${ln.crop_factor ? ` (${fmt(ln.crop_factor, 2)}× crop)` : ''}` +
            `${ln.camera ? ` · ${ln.camera}` : ''}`) : null,
          ln && ln.lens_type === 'circular_fisheye' ? h('div', { class: 'muted' }, 'A circular fisheye lens was detected; check the crop circle in the ', h('a', { href: '#', onclick: (e) => { e.preventDefault(); goTab('crop'); } }, 'Crop tab'), '.') : null,
          groups.size ? h('div', { class: 'warn' }, `Bracketed exposures detected: ${groups.size} stacks. HDR mode is ${p.exposure.hdr_mode ? 'on' : 'off'}. `,
            h('a', { href: '#', onclick: (e) => { e.preventDefault(); goTab('exposure'); } }, 'HDR settings')) : null) : drop),
      step(2, !n ? '' : aligned ? 'done' : 'cur', 'Align images',
        h('div', { class: 'row' }, h('button', { class: 'btn primary', disabled: !n, onclick: () => alignImages() }, aligned ? 'Align images again' : 'Align images'),
          aligned ? h('button', { class: 'btn', onclick: () => goTab('editor') }, 'Panorama Editor') : null,
          aligned ? h('button', { class: 'btn', onclick: () => goTab('cps') }, 'Control points') : null),
        aligned ? h('div', {}, `${p.control_points.length} control points · mean distance ${fmt(mean, 2)} px · max ${fmt(max, 2)} px`,
          unconnected.length ? h('div', { class: 'bad' }, `Not all images have control points: ${unconnected.join(', ')}. Add control points manually in the Control Points tab.`) : null,
          max > 3 * Math.max(mean, 1) && max > 20 ? h('div', { class: 'warn' }, 'The project may contain some misplaced control points (see the Control Points table).') : null) : null),
      step(3, aligned ? 'cur' : '', 'Create panorama',
        h('div', { class: 'row' },
          h('label', {}, 'Projection'), select(Object.entries(PROJ_LABELS), p.panorama.projection, (v) => {
            const [mh, mv] = PROJ_MAX[v];
            store.patch({ panorama: { projection: v, hfov: Math.min(p.panorama.hfov, mh), vfov: Math.min(p.panorama.vfov, mv) } });
          }),
          h('span', { class: 'muted' }, `${fmt(p.panorama.hfov, 1)}° × ${fmt(p.panorama.vfov, 1)}°`)),
        h('div', { class: 'row' }, h('button', { class: 'btn primary', disabled: !aligned, onclick: async () => { const r = await createPanorama(); if (r) viewResult(r); } }, 'Create panorama'),
          h('button', { class: 'btn', disabled: !aligned, onclick: () => goTab('create') }, 'Output settings…')))));
}

// ------------------------------------------------------------------ Source images
export function renderSources(root) {
  const p = P();
  const sel = store.selection;
  const selList = () => [...store.selection].sort((a, b) => a - b);
  const bar = h('div', { class: 'row' },
    h('button', { class: 'btn primary', onclick: loadImagesDialog }, 'Add images…'),
    h('button', { class: 'btn danger', disabled: !sel.size, onclick: async () => { if (await confirmBox('Remove images', `Remove ${sel.size} image(s) from the project?`, 'Remove')) { await store.post('/api/images/remove', { indices: selList() }); store.selection = new Set(); } } }, 'Remove'),
    h('button', { class: 'btn', disabled: !sel.size, onclick: () => store.patch({ images: Object.fromEntries(selList().map((i) => [i, { include: !p.images[i].include }])) }) }, 'Include / exclude'),
    h('button', { class: 'btn', disabled: sel.size < 2, onclick: () => store.post('/api/images/link', { indices: selList() }) }, 'Link (bracket stack)'),
    h('button', { class: 'btn', disabled: !sel.size, onclick: () => store.post('/api/images/link', { indices: selList(), unlink: true }) }, 'Unlink'),
    h('button', { class: 'btn', onclick: () => store.post('/api/images/link', { auto: true }) }, 'Link HDR brackets (EXIF)'),
    h('button', { class: 'btn', disabled: !sel.size, onclick: () => store.post('/api/images/assign_lens', { indices: selList() }) }, 'Use separate lens'),
    h('button', { class: 'btn', disabled: sel.size !== 1, onclick: () => move(-1) }, '◀ Move'),
    h('button', { class: 'btn', disabled: sel.size !== 1, onclick: () => move(1) }, 'Move ▶'),
    h('button', { class: 'btn', onclick: () => sortBy('name') }, 'Sort by name'),
    h('button', { class: 'btn', onclick: () => sortBy('time') }, 'Sort by time'));
  async function move(dir) {
    const i = [...sel][0], j = i + dir;
    if (j < 0 || j >= p.images.length) return;
    const order = p.images.map((_, k) => k);
    [order[i], order[j]] = [order[j], order[i]];
    store.selection = new Set([j]);
    await store.post('/api/images/reorder', { order });
  }
  async function sortBy(key) {
    const order = p.images.map((_, k) => k).sort((a, b) => key === 'name' ? p.images[a].name.localeCompare(p.images[b].name)
      : String(p.images[a].exif.datetime || '').localeCompare(String(p.images[b].exif.datetime || '')));
    await store.post('/api/images/reorder', { order });
  }
  const grid = h('div', { class: 'thumbs' }, p.images.map((im, i) => {
    const t = h('div', { class: `thumb ${sel.has(i) ? 'sel' : ''} ${im.include ? '' : 'excluded'}` },
      h('img', { src: store.imgUrl(i, 'thumb', 320), loading: 'lazy' }),
      h('span', { class: 'num' }, i), im.link_group >= 0 ? h('span', { class: 'link' }, `stack ${im.link_group}`) : null,
      h('div', { class: 'cap' }, h('span', { title: im.abspath }, im.name), h('span', { class: 'muted' }, `${im.width}×${im.height}`)),
      h('div', { class: 'cap muted' }, h('span', {}, `lens ${im.lens} · ${im.n_cps} CPs`), h('span', {}, im.exif.exposure_time ? `1/${Math.round(1 / im.exif.exposure_time)}s` : '')));
    t.onclick = (e) => {
      if (e.metaKey || e.ctrlKey) { if (sel.has(i)) sel.delete(i); else sel.add(i); }
      else if (e.shiftKey && sel.size) { const a = Math.min(...sel, i), b = Math.max(...sel, i); for (let k = a; k <= b; k++) sel.add(k); }
      else store.selection = new Set([i]);
      store.current = i;
      renderSources(root);
    };
    t.ondblclick = () => { store.current = i; goTab('imageparams'); };
    return t;
  }));
  fill(root, h('h1', {}, 'Source Images'), h('p', { class: 'hint' }, 'Click to select (⌘/Ctrl or Shift for several). Linked images (bracket stacks) share one position.'),
    bar, p.images.length ? grid : h('div', { class: 'dropzone' }, 'No images yet.'));
}

// ------------------------------------------------------------------ Lens settings
export function renderLens(root) {
  const p = P();
  const cards = p.lenses.map((ln, li) => {
    const users = p.images.map((im, i) => (im.lens === li ? i : -1)).filter((i) => i >= 0);
    const im0 = p.images[users[0]] || { width: 1, height: 1 };
    const patch = (vals) => store.patch({ lenses: { [li]: vals } });
    const fisheye = ln.lens_type === 'fisheye' || ln.lens_type === 'circular_fisheye';
    // focal length <-> field of view through the sensor size (crop factor)
    const sensorW = ln.crop_factor ? 43.2666 / ln.crop_factor * im0.width / Math.hypot(im0.width, im0.height) : 0;
    const k = ln.lens_type === 'rectilinear' ? -1 : ln.fisheye_factor;
    const lr = (th) => (k > 1e-9 ? Math.sin(k * th) / k : k < -1e-9 ? Math.tan(-k * th) / -k : th);
    const lt = (r) => (k > 1e-9 ? Math.asin(Math.min(1, k * r)) / k : k < -1e-9 ? Math.atan(-k * r) / -k : r);
    const fpx = ['cylindrical', 'equirectangular'].includes(ln.lens_type) ? im0.width / 2 / (ln.hfov * Math.PI / 360) : im0.width / 2 / lr(ln.hfov * Math.PI / 360);
    const fmm = sensorW ? fpx * sensorW / im0.width : 0;
    const focal = numberInput(fmm, (v) => {
      if (!sensorW || !(v > 0)) return;
      const f = v / sensorW * im0.width;
      patch({ hfov: 2 * lt(im0.width / 2 / f) * 180 / Math.PI });
    }, { digits: 2, width: '80px' });
    const perImg = new Set(p.optimizer.per_image || []);
    return card(
      h('div', { class: 'row' }, h('strong', {}, `Lens ${li}`), h('span', { class: 'muted' }, `${ln.camera || 'unknown camera'}${ln.lens_name ? ' · ' + ln.lens_name : ''} · images ${users.join(', ')}`)),
      h('div', { class: 'grid2' },
        h('label', {}, 'Lens type'), select(LENS_TYPES, ln.lens_type, (v) => patch({ lens_type: v, ...(v === 'circular_fisheye' ? { crop: { ...ln.crop, mode: 'circle' } } : {}) })),
        h('label', {}, 'Horizontal field of view'), h('div', { class: 'row' }, numberInput(ln.hfov, (v) => patch({ hfov: v }), { digits: 3 }), h('span', { class: 'muted' }, '°')),
        h('label', {}, 'Focal length'), h('div', { class: 'row' }, focal, h('span', { class: 'muted' }, sensorW ? `mm (theoretical ${fmt(ln.focal_mm, 1)} mm EXIF)` : 'mm — set the crop factor to use focal lengths')),
        h('label', {}, 'Crop factor'), h('div', { class: 'row' }, numberInput(ln.crop_factor, (v) => patch({ crop_factor: v }), { digits: 3, width: '80px' }), h('span', { class: 'muted' }, sensorW ? `sensor ${fmt(sensorW, 1)} mm wide` : '')),
        fisheye ? h('label', {}, 'Fisheye factor') : null,
        fisheye ? h('div', { class: 'row' }, numberInput(ln.fisheye_factor, (v) => patch({ fisheye_factor: v }), { digits: 4, step: 0.01 }),
          select([['', 'Preset…'], ...FISHEYE_PRESETS], '', (v) => v !== '' && patch({ fisheye_factor: parseFloat(v) }))) : null,
        h('label', {}, 'Distortion (a / b / c)'), h('div', { class: 'row' }, ['a', 'b', 'c'].map((q) => numberInput(ln[q], (v) => patch({ [q]: v }), { digits: 5, step: 0.001 }))),
        h('label', {}, 'Shift (horizontal / vertical)'), h('div', { class: 'row' }, ['d', 'e'].map((q) => numberInput(ln[q], (v) => patch({ [q]: v }), { digits: 2 })), h('span', { class: 'muted' }, 'px')),
        h('label', {}, 'Shear (horizontal / vertical)'), h('div', { class: 'row' }, ['g', 't'].map((q) => numberInput(ln[q], (v) => patch({ [q]: v }), { digits: 5, step: 0.0001 }))),
        h('label', {}, 'Vignetting'), h('div', { class: 'mono muted' }, `1 ${ln.vig.map((v, n) => `${v >= 0 ? '+' : '−'} ${fmt(Math.abs(v), 3)}·r${n ? (2 * n + 2) : 2}`).join(' ')}`),
        h('label', {}, 'Camera response'), h('div', { class: 'muted' }, ln.response === 'poly' ? `fitted (${ln.response_params.map((c) => fmt(c, 3)).join(', ')})` : ln.response),
        h('label', {}, 'Use individual parameters'), h('div', { class: 'row' }, ['hfov', 'a', 'b', 'c', 'd', 'e'].map((q) => checkbox(perImg.has(q), (on) => {
          const s = new Set(perImg); if (on) s.add(q); else s.delete(q);
          store.patch({ optimizer: { per_image: [...s] } });
        }, q === 'hfov' ? 'FoV' : q)))),
      h('div', { class: 'row' },
        h('button', { class: 'btn small', onclick: () => lensDbDialog(li) }, 'Lens database…'),
        h('button', { class: 'btn small', onclick: async () => {
          const name = await prompt('Save to lens database', 'Name for this lens profile', ln.lens_name || `${ln.camera} ${fmt(ln.focal_mm, 1)}mm`);
          if (name) { await api.post('/api/lensdb/save', { lens: li, name }); toast('Saved to lens database', 'ok'); }
        } }, 'Save to lens database')));
  });
  fill(root, h('h1', {}, 'Lens Settings'),
    h('p', { class: 'hint' }, 'Parameters are shared by all images taken with the same lens. The optimizer refines field of view, distortion and shift.'),
    ...cards);
}

async function lensDbDialog(li) {
  const db = await api.get('/api/lensdb');
  const list = h('div', {}, db.length ? db.map((r) => h('div', { class: 'fb-item', onclick: async () => {
    await store.post('/api/lensdb/apply', { name: r.name, lens: li });
    toast(`Loaded ${r.name}`, 'ok');
    document.querySelector('.modal-back')?.remove();
  } }, h('strong', {}, r.name), h('span', { class: 'muted' }, ` ${r.lens.lens_type}, ${fmt(r.lens.hfov, 1)}°`))) : h('p', { class: 'muted' }, 'The lens database is empty. Optimize a project and save its lens here to reuse it.'));
  await modal('Lens database', list);
}

// ------------------------------------------------------------------ Image parameters
export function renderImageParams(root) {
  const p = P();
  const cols = [['yaw', 'Yaw'], ['pitch', 'Pitch'], ['roll', 'Roll'], ['ev', 'EV'], ['wb_r', 'WB red'], ['wb_b', 'WB blue'], ['flare', 'Flare'],
    ['vp_x', 'VP X'], ['vp_y', 'VP Y'], ['vp_z', 'VP Z']];
  const body = p.images.map((im, i) => h('tr', { class: store.current === i ? 'sel' : '' },
    h('td', {}, i), h('td', { title: im.abspath }, im.name),
    h('td', {}, checkbox(im.include, (v) => store.patch({ images: { [i]: { include: v } } }))),
    h('td', {}, im.link_group >= 0 ? im.link_group : ''),
    ...cols.map(([k]) => h('td', {}, numberInput(im[k], (v) => store.patch({ images: { [i]: { [k]: v } } }), { digits: ['wb_r', 'wb_b'].includes(k) ? 4 : 3 }))),
    h('td', {}, checkbox(im.vp_enabled, (v) => store.patch({ images: { [i]: { vp_enabled: v } } }))),
    h('td', {}, numberInput(im.blend_priority, (v) => store.patch({ images: { [i]: { blend_priority: Math.round(v) } } }), { digits: 0, width: '54px' })),
    h('td', { class: 'mono' }, im.n_cps)));
  fill(root, h('h1', {}, 'Image Parameters'),
    h('p', { class: 'hint' }, 'Positions are in degrees. VP = viewpoint correction (camera offset, for nadir/floor patches). Blend priority: higher wins in overlaps.'),
    h('div', { class: 'table-wrap' }, h('table', { class: 'grid' },
      h('thead', {}, h('tr', {}, ['#', 'Image', 'Incl.', 'Link', ...cols.map((c) => c[1]), 'VP', 'Priority', 'CPs'].map((t) => h('th', {}, t)))),
      h('tbody', {}, body))));
}

// ------------------------------------------------------------------ Optimizer
let advanced = false;
let lastOpt = null;
export function renderOptimizer(root) {
  const p = P();
  const o = p.optimizer;
  const patch = (vals) => store.patch({ optimizer: vals });
  const run = async (body = {}) => {
    const r = await store.job('Optimizing', '/api/optimize', body);
    if (r) { lastOpt = r; toast(`Optimizer: mean ${fmt(r.mean_error, 2)} px, max ${fmt(r.max_error, 2)} px`, 'ok'); }
  };
  const perImg = new Set(o.per_image || []);
  const simple = h('div', { class: 'grid2' },
    h('label', {}, 'Anchor image'), select(p.images.map((im, i) => [i, `${i} — ${im.name}`]), o.anchor, (v) => patch({ anchor: +v })),
    h('label', {}, 'Optimize lens field of view'), checkbox(o.optimize_fov, (v) => patch({ optimize_fov: v })),
    h('label', {}, 'Minimize lens distortion'), select([['none', 'No'], ['medium', 'Medium (b)'], ['heavy', 'Heavy (a, b, c)'], ['heavy_shift', 'Heavy + lens shift']], o.distortion, (v) => patch({ distortion: v })),
    h('label', {}, 'Optimize fisheye factor'), checkbox(o.optimize_fisheye_factor, (v) => patch({ optimize_fisheye_factor: v })),
    h('label', {}, 'Optimize shear'), checkbox(o.optimize_shear, (v) => patch({ optimize_shear: v })),
    h('label', {}, 'Hor / vert control points'), select([['ignore', 'Ignore'], ['second_pass', 'Level in second pass'], ['single_pass', 'Include in single pass']], o.hv_mode, (v) => patch({ hv_mode: v })),
    h('label', {}, 'Viewpoint correction'), checkbox(o.viewpoint, (v) => patch({ viewpoint: v }), 'for images with VP enabled (Image Parameters)'),
    h('label', {}, 'Robust (outlier tolerant)'), checkbox(o.robust, (v) => patch({ robust: v })));
  const adv = advanced ? card(h('strong', {}, 'Per-image lens parameters'), h('p', { class: 'hint' }, 'Checked parameters are optimized individually for every image instead of shared per lens.'),
    h('div', { class: 'row' }, ['hfov', 'a', 'b', 'c', 'd', 'e', 'g', 't'].map((q) => checkbox(perImg.has(q), (on) => {
      const s = new Set(perImg); if (on) s.add(q); else s.delete(q); patch({ per_image: [...s] });
    }, q)))) : null;
  const errsPer = p.images.map((im, i) => {
    const e = p.control_points.filter((c) => c.i === i || c.j === i).map((c) => c.error);
    return [i, e.length, e.length ? e.reduce((a, b) => a + b, 0) / e.length : 0, e.length ? Math.max(...e) : 0];
  });
  fill(root, h('h1', {}, 'Optimizer'),
    h('div', { class: 'row' }, h('div', { class: 'seg' }, h('button', { class: advanced ? '' : 'on', onclick: () => { advanced = false; renderOptimizer(root); } }, 'Simple'),
      h('button', { class: advanced ? 'on' : '', onclick: () => { advanced = true; renderOptimizer(root); } }, 'Advanced'))),
    card(simple), adv,
    h('div', { class: 'row' }, h('button', { class: 'btn primary', onclick: () => run() }, 'Run optimizer'),
      h('button', { class: 'btn', onclick: () => run({ reinit: true }) }, 'Initialize and optimize'),
      h('button', { class: 'btn', onclick: () => store.post('/api/cps/delete_worst', { fraction: 0.1 }) }, 'Delete worst 10% control points'),
      lastOpt ? h('span', { class: 'muted' }, `last run: mean ${fmt(lastOpt.mean_error, 2)} px · max ${fmt(lastOpt.max_error, 2)} px · ${lastOpt.n_vars} variables`) : null),
    h('h2', {}, 'Control point distances per image'),
    h('div', { class: 'table-wrap', style: { maxHeight: '320px' } }, h('table', { class: 'grid' },
      h('thead', {}, h('tr', {}, ['Image', 'Control points', 'Average distance', 'Maximum distance'].map((t) => h('th', {}, t)))),
      h('tbody', {}, errsPer.map(([i, n, a, m]) => h('tr', {}, h('td', {}, `${i} ${p.images[i].name}`), h('td', {}, n),
        h('td', { class: 'mono' }, fmt(a, 2)), h('td', { class: `mono ${m > 10 ? 'bad' : m > 4 ? 'warn' : ''}` }, fmt(m, 2))))))));
}

// ------------------------------------------------------------------ Exposure / HDR
export function renderExposure(root) {
  const p = P();
  const ex = p.exposure;
  const patch = (vals) => store.patch({ exposure: vals });
  const tm = { brightness: 0, boost_shadows: 0.3, reduce_lights: 0.3, compression: 0.5, contrast: 0, saturation: 0, ...(ex.tonemap_params || {}) };
  const slider = (label, key, min, max, step, obj, onset) => {
    const r = h('input', { type: 'range', min, max, step, value: obj[key], style: { width: '220px' } });
    const v = h('span', { class: 'mono' }, fmt(obj[key], 2));
    r.addEventListener('input', () => { v.textContent = fmt(+r.value, 2); });
    r.addEventListener('change', () => onset(+r.value));
    return row(label, r, v);
  };
  const groups = new Set(p.images.filter((im) => im.link_group >= 0).map((im) => im.link_group));
  fill(root, h('h1', {}, 'Exposure / HDR'),
    h('h2', {}, 'HDR'),
    card(
      row('', checkbox(ex.hdr_mode, (v) => patch({ hdr_mode: v }), 'Group bracketed exposures into blend planes (HDR panorama)')),
      h('p', { class: 'hint' }, groups.size ? `The project contains ${groups.size} linked stacks of bracketed exposures.` : 'No linked bracket stacks (use Source Images → Link to group them).'),
      row('Merge using', select([['fusion', 'Exposure fusion'], ['true_hdr', 'True HDR']], ex.hdr_method, (v) => patch({ hdr_method: v }))),
      row('HDR workflow', select([['merge_then_blend', 'Merge bracketed images to HDR, then blend'], ['blend_then_merge', 'Create blend planes first, then merge']], ex.hdr_flow, (v) => patch({ hdr_flow: v }))),
      row('', checkbox(ex.link_brackets, (v) => patch({ link_brackets: v }), 'Bracketed images share their position (linked)')),
      row('', checkbox(ex.deghost, (v) => patch({ deghost: v }), 'Remove ghosts (moving objects) when merging')),
      ex.hdr_method === 'fusion' ? h('div', {},
        slider('Fusion sigma', 'fusion_sigma', 0.05, 0.6, 0.01, ex, (v) => patch({ fusion_sigma: v })),
        slider('Contrast weight', 0, 0, 3, 0.1, ex.fusion_weights, (v) => patch({ fusion_weights: [v, ex.fusion_weights[1], ex.fusion_weights[2]] })),
        slider('Saturation weight', 1, 0, 3, 0.1, ex.fusion_weights, (v) => patch({ fusion_weights: [ex.fusion_weights[0], v, ex.fusion_weights[2]] })),
        slider('Exposure weight', 2, 0, 3, 0.1, ex.fusion_weights, (v) => patch({ fusion_weights: [ex.fusion_weights[0], ex.fusion_weights[1], v] }))) :
        h('div', {}, h('strong', {}, 'Tone mapping (for the LDR output of a true HDR panorama)'),
          ...[['Brightness', 'brightness', -3, 3], ['Boost shadows', 'boost_shadows', 0, 1], ['Reduce lights', 'reduce_lights', 0, 1],
            ['Compression', 'compression', 0, 1], ['Contrast', 'contrast', -1, 1], ['Saturation', 'saturation', -1, 1]].map(([l, k, a, b]) =>
            slider(l, k, a, b, 0.01, tm, (v) => patch({ tonemap_params: { ...tm, [k]: v } }))))),
    h('h2', {}, 'Automatic exposure and color adjustment'),
    card(
      row('Optimize brightness', select([['auto', 'Automatic (fine tune when exposure is known)'], ['full', 'Enabled (full)'], ['finetune', 'Enabled (fine tune only)'], ['off', 'Disabled']], ex.optimize_exposure, (v) => patch({ optimize_exposure: v }))),
      row('Optimize white balance', select([['auto', 'Automatic (when shot with auto WB)'], ['full', 'Enabled'], ['off', 'Disabled']], ex.optimize_wb, (v) => patch({ optimize_wb: v }))),
      row('', checkbox(ex.optimize_vignetting, (v) => patch({ optimize_vignetting: v }), 'Optimize vignetting')),
      row('', checkbox(ex.optimize_flare, (v) => patch({ optimize_flare: v }), 'Optimize lens flare')),
      row('', checkbox(ex.optimize_response, (v) => patch({ optimize_response: v }), 'Generate camera response curve')),
      h('div', { class: 'row' }, h('button', { class: 'btn primary', onclick: async () => {
        const r = await store.job('Optimizing exposure', '/api/photometric', {});
        if (r) toast(`Exposure optimized: residual ${fmt(r.rms_before, 4)} → ${fmt(r.rms_after, 4)}`, 'ok');
      } }, 'Optimize now!'))),
    h('h2', {}, 'Fine tune output'),
    card(slider('Exposure (stops)', 'exposure', -3, 3, 0.05, ex, (v) => patch({ exposure: v })),
      slider('White balance red', 'wb_r', 0.5, 1.5, 0.005, ex, (v) => patch({ wb_r: v })),
      slider('White balance blue', 'wb_b', 0.5, 1.5, 0.005, ex, (v) => patch({ wb_b: v }))));
}

// ------------------------------------------------------------------ Project settings
export function renderSettings(root) {
  const p = P();
  const md = p.metadata || {};
  const ab = { generate: true, prealigned: false, optimize: true, straighten: true, fit: true, choose_projection: true, exposure: true, max_cps: 30, ...(md.align_behavior || {}) };
  const tb = { lens_and_crop: true, image_parameters: true, masks: true, panorama_settings: true, optimizer_settings: true, link_images: 'as_template', camera_curve: 'yes', ...(md.template_behavior || {}) };
  const setAb = (k, v) => store.patch({ metadata: { align_behavior: { ...ab, [k]: v } } });
  const setTb = (k, v) => store.patch({ metadata: { template_behavior: { ...tb, [k]: v } } });
  let theme = 'dark';
  try { theme = localStorage.getItem('pf.theme') || 'dark'; } catch (e) { /* storage unavailable */ }
  fill(root, h('h1', {}, 'Project Settings'),
    h('h2', {}, '“Align images” behavior'),
    card(
      row('', checkbox(ab.generate, (v) => setAb('generate', v), 'Generate control points')),
      row('', checkbox(ab.prealigned, (v) => setAb('prealigned', v), 'Images are already roughly positioned (only search overlapping pairs)')),
      row('', checkbox(ab.exposure, (v) => setAb('exposure', v), 'Perform automatic exposure and color adjustment')),
      row('Control points per pair', numberInput(ab.max_cps, (v) => setAb('max_cps', Math.max(4, Math.round(v))), { digits: 0 }))),
    h('h2', {}, 'Template behavior'),
    card(h('p', { class: 'hint' }, 'When this project is applied as a template, copy:'),
      ...[['lens_and_crop', 'Lens and crop settings'], ['image_parameters', 'Image parameters (positions)'], ['masks', 'Image masks'],
        ['panorama_settings', 'Panorama, HDR and output settings'], ['optimizer_settings', 'Optimizer settings']].map(([k, l]) => row('', checkbox(tb[k], (v) => setTb(k, v), l))),
      row('Link images in target', select([['as_template', 'Same as in this template'], ['auto_exif', 'Automatically from EXIF exposure'], ['keep', "Don't modify"]], tb.link_images, (v) => setTb('link_images', v))),
      h('div', { class: 'row' },
        h('button', { class: 'btn', onclick: async () => {
          const path = await fileBrowser({ title: 'Save as template', filter: 'projects', save: true, defaultName: 'template.pfproj' });
          if (path) { await api.post('/api/project/save_template', { path: path[0], behavior: tb }); toast('Template saved', 'ok'); }
        } }, 'Save as template…'),
        h('button', { class: 'btn', onclick: async () => {
          const path = await fileBrowser({ title: 'Apply template', filter: 'projects' });
          if (path && path.length) await store.post('/api/project/apply_template', { path: path[0] });
        } }, 'Apply template…'))),
    h('h2', {}, 'Appearance'),
    card(row('Theme', select([['dark', 'Dark'], ['light', 'Light']], theme, (v) => {
      try { localStorage.setItem('pf.theme', v); } catch (e) { /* ignore */ }
      document.documentElement.dataset.theme = v;
    }))));
}

// ------------------------------------------------------------------ Preview
export function renderPreview(root) {
  const img = h('img', { style: { maxWidth: '100%', display: 'block', background: '#000', borderRadius: '6px' } });
  let width = 1600, seams = 'voronoi';
  const go = async () => {
    status('Rendering preview…');
    const t0 = performance.now();
    const r = await fetch(`/api/preview?width=${width}&seams=${seams}`);
    if (!r.ok) { toast(await r.text(), 'err'); return; }
    img.src = URL.createObjectURL(await r.blob());
    status(`Preview rendered in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  };
  fill(root, h('h1', {}, 'Preview'),
    h('div', { class: 'row' }, h('label', {}, 'Width'), select([[1200, '1200 px'], [1600, '1600 px'], [2400, '2400 px'], [3200, '3200 px']], width, (v) => { width = +v; }),
      h('label', { style: { minWidth: 0 } }, 'Seams'), select([['voronoi', 'Fast'], ['graphcut', 'Optimal (graph cut)']], seams, (v) => { seams = v; }),
      h('button', { class: 'btn primary', onclick: go }, 'Preview')),
    h('p', { class: 'hint' }, 'Server-side stitch at reduced size, using the same seams and blending as the final panorama.'), img);
}

// ------------------------------------------------------------------ Metadata
export function renderMetadata(root) {
  const md = P().metadata || {};
  const first = (P().images[0] || {}).exif || {};
  const set = (k, v) => store.patch({ metadata: { [k]: v } });
  const txt = (k, ph) => { const i = h('input', { type: 'text', value: md[k] ?? '', placeholder: ph, style: { width: '360px' } }); i.addEventListener('change', () => set(k, i.value)); return i; };
  fill(root, h('h1', {}, 'Metadata'),
    h('h2', {}, 'GPS data'),
    card(row('Latitude', numberInput(md.gps_lat ?? first.gps_lat ?? 0, (v) => set('gps_lat', v), { digits: 6, width: '140px' })),
      row('Longitude', numberInput(md.gps_lon ?? first.gps_lon ?? 0, (v) => set('gps_lon', v), { digits: 6, width: '140px' })),
      first.gps_lat !== undefined ? h('p', { class: 'hint' }, `Source images: ${fmt(first.gps_lat, 6)}, ${fmt(first.gps_lon, 6)}`) : h('p', { class: 'hint' }, 'No location found in the EXIF data.')),
    h('h2', {}, 'Orientation'),
    card(row('Compass angle of the panorama centre', numberInput(md.compass ?? first.gps_heading ?? 0, (v) => set('compass', v), { digits: 1 }), h('span', { class: 'muted' }, '° (0 = north)'))),
    h('h2', {}, 'Description'),
    card(row('Title', txt('title', 'Panorama title')), row('Author', txt('author', '')), row('Copyright', txt('copyright', '')), row('Description', txt('description', ''))));
}

// ------------------------------------------------------------------ Create panorama
export function renderCreate(root) {
  const p = P();
  const o = p.output;
  const opt = p.optimum_size || [0, 0];
  const aspect = opt[0] ? opt[1] / opt[0] : 0.5;
  const crop = p.panorama.crop;
  const cw = (crop[2] - crop[0]), ch = (crop[3] - crop[1]);
  const fullW = p.panorama.width || Math.round(opt[0] * o.size_percent / 100);
  const outW = Math.round(fullW * cw), outH = Math.round(fullW * aspect * ch);
  const patchO = (vals) => store.patch({ output: vals });
  const fmts = [['jpeg', 'JPEG'], ['tiff', 'TIFF'], ['png', 'PNG'], ['psd', 'Photoshop (PSD/PSB)'], ['webp', 'WebP'], ['exr', 'OpenEXR (HDR)'], ['hdr', 'Radiance HDR']];
  const fmtOpts = [];
  if (['jpeg', 'webp'].includes(o.format)) fmtOpts.push(row('Quality', numberInput(o.quality, (v) => patchO({ quality: Math.round(v) }), { digits: 0 })));
  if (['tiff', 'png', 'psd'].includes(o.format)) fmtOpts.push(row('Bit depth', select([[8, '8 bits per channel'], [16, '16 bits per channel']], o.bit_depth, (v) => patchO({ bit_depth: +v }))));
  if (o.format === 'tiff') fmtOpts.push(row('Compression', select([['lzw', 'LZW'], ['zip', 'ZIP (deflate)'], ['packbits', 'PackBits'], ['none', 'None']], o.compression, (v) => patchO({ compression: v }))));
  if (['tiff', 'png', 'psd', 'webp', 'exr'].includes(o.format)) fmtOpts.push(row('', checkbox(o.alpha, (v) => patchO({ alpha: v }), 'Alpha channel (transparency)')));
  const outPath = h('input', { type: 'text', value: o.path || '', placeholder: '(same folder and name as the project)', style: { width: '420px' } });
  outPath.addEventListener('change', () => patchO({ path: outPath.value }));
  const res = lastResult;
  fill(root, h('h1', {}, 'Create Panorama'),
    card(h('div', { class: 'grid2' },
      h('label', {}, 'Size'), h('div', { class: 'row' },
        numberInput(fullW, (v) => store.patch({ panorama: { width: Math.max(16, Math.round(v)) } }), { digits: 0 }), h('span', { class: 'muted' }, '×'),
        numberInput(Math.round(fullW * aspect), (v) => store.patch({ panorama: { width: Math.max(16, Math.round(v / aspect)) } }), { digits: 0 }),
        h('span', { class: 'muted' }, `px (full canvas) → output ${outW} × ${outH} px, ${fmt(outW * outH / 1e6, 1)} MP`),
        h('button', { class: 'btn small', onclick: () => store.patch({ panorama: { width: 0 }, output: { size_percent: 100 } }) }, 'Set optimum size')),
      h('label', {}, 'Scale'), h('div', { class: 'row' }, ['25', '50', '75', '100'].map((s) => h('button', { class: 'btn small', onclick: () => store.patch({ panorama: { width: 0 }, output: { size_percent: +s } }) }, `${s}%`)),
        h('span', { class: 'muted' }, `optimum ${opt[0]} × ${opt[1]}`)),
      h('label', {}, 'File format'), select(fmts, o.format, (v) => patchO({ format: v })),
      h('label', {}, ''), h('div', {}, ...fmtOpts),
      h('label', {}, 'Layers'), select([['blended', 'Blended panorama'], ['layers', 'Individual layers'], ['layers_and_blended', 'Layers + blended (PSD/TIFF)']], o.layers, (v) => patchO({ layers: v })),
      h('label', {}, 'Blending'), select([['multiband', 'Multiband (seamless)'], ['feather', 'Alpha blending (feather)'], ['none', 'None (hard seams)']], o.blender, (v) => patchO({ blender: v })),
      h('label', {}, 'Seams'), select([['graphcut', 'Optimal (graph cut)'], ['voronoi', 'Nearest image centre'], ['none', 'Stacking order']], o.seams, (v) => patchO({ seams: v })),
      h('label', {}, 'Feather'), h('div', { class: 'row' }, h('span', { class: 'muted' }, 'sharp'), (() => {
        const r = h('input', { type: 'range', min: 0, max: 1, step: 0.05, value: o.feather });
        r.addEventListener('change', () => patchO({ feather: +r.value }));
        return r;
      })(), h('span', { class: 'muted' }, 'soft')),
      h('label', {}, 'Interpolator'), select([['lanczos', 'Lanczos (sharp)'], ['bicubic', 'Bicubic'], ['bilinear', 'Bilinear'], ['nearest', 'Nearest neighbour']], o.interpolator, (v) => patchO({ interpolator: v })),
      p.exposure.hdr_mode && p.exposure.hdr_method === 'true_hdr' ? h('label', {}, 'HDR file format') : null,
      p.exposure.hdr_mode && p.exposure.hdr_method === 'true_hdr' ? select([['exr', 'OpenEXR'], ['hdr', 'Radiance .hdr']], o.hdr_format, (v) => patchO({ hdr_format: v })) : null,
      h('label', {}, 'Output file'), h('div', { class: 'row' }, outPath, h('button', { class: 'btn small', onclick: async () => {
        const ext = { jpeg: 'jpg', tiff: 'tif', png: 'png', psd: 'psd', webp: 'webp', exr: 'exr', hdr: 'hdr' }[o.format];
        const r = await fileBrowser({ title: 'Output file', filter: 'all', save: true, defaultName: `panorama.${ext}` });
        if (r) patchO({ path: r[0] });
      } }, 'Browse…')))),
    h('div', { class: 'row' },
      h('button', { class: 'btn primary', onclick: async () => { const r = await createPanorama(); if (r) renderCreate(root); } }, 'Create Panorama'),
      h('button', { class: 'btn', onclick: () => sendToBatch() }, 'Save and send to Batch Stitcher')),
    res ? card(h('strong', {}, 'Last result'), h('div', { class: 'mono' }, res.path), h('div', { class: 'muted' }, `${res.width} × ${res.height} px · ${fmt(res.seconds, 1)} s`),
      ...(res.extra || []).map((f) => h('div', { class: 'mono muted' }, f)),
      h('div', { class: 'row' },
        h('button', { class: 'btn', onclick: () => viewResult(res) }, 'View'),
        h('button', { class: 'btn', onclick: () => api.post('/api/reveal', { path: res.path }) }, 'Show in Finder'),
        res.projection === 'equirectangular' ? h('button', { class: 'btn', onclick: async () => { const r = await api.post('/api/convert', { path: res.path, to: 'cube' }); toast(`Cube faces: ${r.files.length} files`, 'ok'); } }, 'Convert to cube faces') : null,
        res.projection === 'equirectangular' ? h('button', { class: 'btn', onclick: async () => { const r = await api.post('/api/convert', { path: res.path, to: 'littleplanet' }); toast(`Little planet: ${r.files[0]}`, 'ok'); } }, 'Little planet') : null,
        res.projection === 'equirectangular' ? h('button', { class: 'btn', onclick: async () => { const r = await api.post('/api/publish', { path: res.path, title: (P().metadata || {}).title || 'Panorama' }); toast(`Published to ${r.folder}`, 'ok'); } }, 'Publish to website') : null)) : null);
}

async function sendToBatch() {
  let path = P().path;
  if (!path) {
    const r = await fileBrowser({ title: 'Save project', filter: 'projects', save: true, defaultName: 'panorama.pfproj' });
    if (!r) return;
    path = r[0];
  }
  await store.post('/api/project/save', { path });
  const list = batchList();
  if (!list.includes(store.project.path)) list.push(store.project.path);
  saveBatch(list);
  toast('Project saved and added to the Batch Stitcher', 'ok');
}

// ------------------------------------------------------------------ Batch stitcher
const batchList = () => { try { return JSON.parse(localStorage.getItem('pf.batch') || '[]'); } catch (e) { return []; } };
const saveBatch = (l) => { try { localStorage.setItem('pf.batch', JSON.stringify(l)); } catch (e) { /* ignore */ } };
export function renderBatch(root) {
  const list = batchList();
  let align = false;
  const log = h('div', { class: 'mono muted' });
  fill(root, h('h1', {}, 'Batch Stitcher'),
    h('p', { class: 'hint' }, 'Stitch a list of projects one after another, unattended.'),
    h('div', { class: 'row' },
      h('button', { class: 'btn', onclick: async () => { const r = await fileBrowser({ title: 'Add projects', filter: 'projects', multi: true }); if (r) { saveBatch([...new Set([...list, ...r])]); renderBatch(root); } } }, 'Add projects…'),
      h('button', { class: 'btn', onclick: () => { saveBatch([]); renderBatch(root); } }, 'Clear list'),
      checkbox(align, (v) => { align = v; }, 'Align images first'),
      h('button', { class: 'btn primary', disabled: !list.length, onclick: async () => {
        const r = await store.job('Batch stitching', '/api/batch', { paths: list, align });
        if (r) { log.replaceChildren(...r.outputs.map((f) => h('div', {}, `✓ ${f}`))); toast(`Batch finished: ${r.outputs.length} panoramas`, 'ok'); }
      } }, 'Run batch')),
    card(list.length ? h('table', { class: 'grid' }, h('tbody', {}, list.map((f, k) => h('tr', {}, h('td', {}, k + 1), h('td', { class: 'mono' }, f),
      h('td', {}, h('button', { class: 'btn small', onclick: () => { list.splice(k, 1); saveBatch(list); renderBatch(root); } }, 'Remove')))))) : h('p', { class: 'muted' }, 'No projects in the batch list.')),
    log,
    h('h2', {}, 'Batch Builder'),
    card(h('p', { class: 'hint' }, 'Scan a folder of shots, group them into panoramas by capture time and create one project per panorama (optionally from a template).'),
      h('div', { class: 'row' }, h('button', { class: 'btn', onclick: async () => {
        const dir = await fileBrowser({ title: 'Folder with the shots to scan', folder: true });
        if (!dir || !dir.length) return;
        const folder = dir[0].replace(/\/$/, '');
        const gap = parseFloat(await prompt('Batch Builder', 'Maximum seconds between shots of one panorama', '10') || '10');
        const r = await api.post('/api/batch/build', { folder, gap });
        saveBatch([...new Set([...batchList(), ...r.projects])]);
        toast(`${r.projects.length} projects created`, 'ok');
        renderBatch(root);
      } }, 'Build projects from folder…'))));
}
