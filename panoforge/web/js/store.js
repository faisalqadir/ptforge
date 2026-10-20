// Project state shared by all views.  Every server response carries the full project.
import { api, runJob } from './api.js';
import { toast, withProgress } from './ui.js';

const listeners = new Set();
export const store = {
  project: null,
  selection: new Set(), // selected image indices
  current: 0, // "current" image for single-image tabs
  onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); },
  set(project) {
    this.project = project;
    const n = project.images.length;
    this.selection = new Set([...this.selection].filter((i) => i < n));
    if (this.current >= n) this.current = Math.max(0, n - 1);
    for (const fn of listeners) {
      try { fn(project); } catch (e) { console.error(e); }
    }
  },
  async refresh() { this.set(await api.get('/api/project')); },
  async call(method, url, body) {
    try {
      const r = await api[method](url, body);
      if (r && r.project) this.set(r.project);
      return r;
    } catch (e) {
      toast(e.message, 'err');
      throw e;
    }
  },
  post(url, body) { return this.call('post', url, body); },
  patch(body) { return this.call('patch', '/api/project', body); },
  async job(title, url, body) {
    try {
      const res = await withProgress(title, url, body);
      await this.refresh();
      return res;
    } catch (e) {
      toast(`${title} failed: ${e.message}`, 'err');
      await this.refresh();
      return null;
    }
  },
  // URL of an image rendition; changes whenever the file or the server changes (cache safe)
  imgUrl(i, kind = 'thumb', size = 256) {
    const im = this.project.images[i];
    const q = `v=${encodeURIComponent(im.abspath)}&b=${this.project.boot}`;
    return kind === 'full' ? `/api/image/${i}/full?${q}` : `/api/image/${i}/${kind}?size=${size}&${q}`;
  },
  lensOf(i) { return this.project.lenses[this.project.images[i].lens]; },
  lensValue(i, key) {
    const im = this.project.images[i];
    return key in (im.lens_overrides || {}) ? im.lens_overrides[key] : this.project.lenses[im.lens][key];
  },
};

export { runJob };
