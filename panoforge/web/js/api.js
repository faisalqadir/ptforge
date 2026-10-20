// Thin client for the PanoForge server API.

async function request(method, url, body, raw = false) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    if (body instanceof Blob || body instanceof FormData) {
      opt.body = body;
    } else {
      opt.headers['Content-Type'] = 'application/json';
      opt.body = JSON.stringify(body);
    }
  }
  const r = await fetch(url, opt);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = j.detail || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  return raw ? r : r.json();
}

export const api = {
  get: (u) => request('GET', u),
  post: (u, b = {}) => request('POST', u, b),
  patch: (u, b) => request('PATCH', u, b),
  put: (u, b) => request('PUT', u, b),
  raw: (m, u, b) => request(m, u, b, true),
};

// Start a background job and follow its progress (server-sent events).
// onProgress({stage, i, n, message}) ; resolves with the job result.
export function runJob(startUrl, body, onProgress) {
  return api.post(startUrl, body).then(({ job }) => new Promise((resolve, reject) => {
    const es = new EventSource(`/api/jobs/${job}/events`);
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.type === 'progress' && onProgress) onProgress({ ...ev, job });
      if (ev.type === 'done') {
        es.close();
        if (ev.error) reject(new Error(ev.error)); else resolve({ job, result: ev.result });
      }
    };
    es.onerror = () => {
      // fall back to polling if the stream breaks
      es.close();
      const poll = async () => {
        try {
          const s = await api.get(`/api/jobs/${job}`);
          if (s.done) { if (s.error) reject(new Error(s.error)); else resolve({ job, result: s.result }); }
          else { if (s.last && onProgress) onProgress({ ...s.last, job }); setTimeout(poll, 500); }
        } catch (e) { reject(e); }
      };
      poll();
    };
  }));
}

export const cancelJob = (id) => api.post(`/api/jobs/${id}/cancel`);
