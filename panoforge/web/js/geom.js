// JavaScript port of panoforge/geometry.py (must stay numerically identical).
// Vectors are [x, y, z]; matrices are row-major 3x3 arrays of rows.

export const DEG = Math.PI / 180;

export function rotYaw(a) { const c = Math.cos(a), s = Math.sin(a); return [[c, 0, s], [0, 1, 0], [-s, 0, c]]; }
export function rotPitch(a) { const c = Math.cos(a), s = Math.sin(a); return [[1, 0, 0], [0, c, s], [0, -s, c]]; }
export function rotRoll(a) { const c = Math.cos(a), s = Math.sin(a); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; }
export function matMul(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
  return C;
}
export function matVec(A, v) {
  return [A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2], A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2], A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2]];
}
export function transpose(A) { return [[A[0][0], A[1][0], A[2][0]], [A[0][1], A[1][1], A[2][1]], [A[0][2], A[1][2], A[2][2]]]; }
export function rotation(yaw, pitch, roll) { return matMul(matMul(rotYaw(yaw * DEG), rotPitch(pitch * DEG)), rotRoll(roll * DEG)); }
export function matToYpr(R) {
  const sp = Math.max(-1, Math.min(1, R[1][2]));
  const pitch = Math.asin(sp);
  let yaw, roll;
  if (Math.abs(sp) < 0.999999) { yaw = Math.atan2(R[0][2], R[2][2]); roll = Math.atan2(R[1][0], R[1][1]); }
  else { roll = 0; yaw = Math.atan2(-R[2][0], R[0][0]); }
  return [yaw / DEG, pitch / DEG, roll / DEG];
}
export function norm(v) { const n = Math.hypot(v[0], v[1], v[2]) || 1e-300; return [v[0] / n, v[1] / n, v[2] / n]; }
export function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
export function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function axisAngle(axis, ang) {
  const [x, y, z] = norm(axis); const c = Math.cos(ang), s = Math.sin(ang), t = 1 - c;
  return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x], [t * x * z - s * y, t * y * z + s * x, t * z * z + c]];
}

// ------------------------------------------------------------------ lens
export function lensR(th, k) {
  if (k > 1e-9) return Math.sin(k * th) / k;
  if (k < -1e-9) return Math.tan(-k * th) / -k;
  return th;
}
export function lensTheta(r, k) {
  if (k > 1e-9) return Math.asin(Math.max(-1, Math.min(1, k * r))) / k;
  if (k < -1e-9) return Math.atan(-k * r) / -k;
  return r;
}
function lensValid(th, k) {
  if (k > 1e-9) return th < Math.min(Math.PI / (2 * k), Math.PI);
  if (k < -1e-9) return th < (Math.PI / (2 * -k)) * 0.999;
  return th < Math.PI * 0.999;
}

export const RADIAL = new Set(['rectilinear', 'fisheye', 'circular_fisheye']);

export class Camera {
  constructor(img, lens, values) {
    // values: resolved lens parameters (overrides applied)
    this.w = img.width; this.h = img.height;
    this.type = lens.lens_type;
    Object.assign(this, values);
    this.k = this.type === 'rectilinear' ? -1 : this.fisheye_factor;
    this.radial = RADIAL.has(this.type);
    this.normR = Math.min(this.w, this.h) / 2;
    const half = this.hfov * DEG / 2;
    this.f = this.radial ? (this.w / 2) / lensR(half, this.k) : (this.w / 2) / half;
    this.R = rotation(img.yaw, img.pitch, img.roll);
    this.RT = transpose(this.R);
    this.rlim = img.radial_limit || 1e9;
  }
  setRotation(R) { this.R = R; this.RT = transpose(R); }
  // world dir -> [x, y, valid]
  worldToPixel(d) {
    const c = matVec(this.RT, d);
    let xi, yi, ok = true;
    const f = this.f;
    if (this.radial) {
      const rho = Math.hypot(c[0], c[1]);
      const th = Math.atan2(rho, c[2]);
      ok = lensValid(th, this.k);
      const r = lensR(Math.min(th, this.k < 0 ? Math.PI / (2 * -this.k) * 0.999 : th), this.k) * f;
      const s = rho > 1e-12 ? r / rho : f;
      xi = c[0] * s; yi = -c[1] * s;
    } else if (this.type === 'cylindrical') {
      const lon = Math.atan2(c[0], c[2]); const hh = Math.hypot(c[0], c[2]);
      xi = f * lon; yi = -f * c[1] / Math.max(hh, 1e-12); ok = hh > 1e-9;
    } else {
      xi = f * Math.atan2(c[0], c[2]); yi = -f * Math.atan2(c[1], Math.hypot(c[0], c[2]));
    }
    if (this.a || this.b || this.c) {
      const rn = Math.hypot(xi, yi) / this.normR;
      if (rn > this.rlim) ok = false;
      const d0 = 1 - (this.a + this.b + this.c);
      const sc = ((this.a * rn + this.b) * rn + this.c) * rn + d0;
      xi *= sc; yi *= sc;
    }
    const xs = xi + this.g * yi, ys = yi + this.t * xi;
    return [xs + this.w / 2 + this.d - 0.5, ys + this.h / 2 + this.e - 0.5, ok];
  }
  pixelToRay(x, y) {
    let xs = x + 0.5 - this.w / 2 - this.d, ys = y + 0.5 - this.h / 2 - this.e;
    let xd = xs, yd = ys;
    if (this.g || this.t) { const det = 1 - this.g * this.t; xd = (xs - this.g * ys) / det; yd = (ys - this.t * xs) / det; }
    let xi = xd, yi = yd;
    if (this.a || this.b || this.c) {
      const rd = Math.hypot(xd, yd) / this.normR;
      const d0 = 1 - (this.a + this.b + this.c);
      let ru = rd / Math.max(d0, 1e-6);
      for (let k = 0; k < 12; k++) {
        const p = ((this.a * ru + this.b) * ru + this.c) * ru + d0;
        const dp = ((4 * this.a * ru + 3 * this.b) * ru + 2 * this.c) * ru + d0;
        ru -= (ru * p - rd) / (Math.abs(dp) < 1e-9 ? 1e-9 : dp);
      }
      const s = rd > 1e-12 ? ru / rd : 1 / Math.max(d0, 1e-6);
      xi = xd * s; yi = yd * s;
    }
    const f = this.f;
    if (this.radial) {
      const r = Math.hypot(xi, yi) / f; const th = lensTheta(r, this.k); const st = Math.sin(th);
      const rr = Math.max(Math.hypot(xi, yi), 1e-300);
      return norm(r > 1e-12 ? [xi / rr * st, -yi / rr * st, Math.cos(th)] : [xi / f, -yi / f, 1]);
    }
    if (this.type === 'cylindrical') { const lon = xi / f; return norm([Math.sin(lon), -yi / f, Math.cos(lon)]); }
    const lon = xi / f, lat = -yi / f;
    return [Math.sin(lon) * Math.cos(lat), Math.sin(lat), Math.cos(lon) * Math.cos(lat)];
  }
  pixelToWorld(x, y) { return matVec(this.R, this.pixelToRay(x, y)); }
}

export function cameraFor(project, i, Roverride) {
  const img = project.images[i];
  const lens = project.lenses[img.lens];
  const v = {};
  for (const k of ['hfov', 'fisheye_factor', 'a', 'b', 'c', 'd', 'e', 'g', 't']) v[k] = k in (img.lens_overrides || {}) ? img.lens_overrides[k] : lens[k];
  const cam = new Camera(img, lens, v);
  if (Roverride) cam.setRotation(Roverride);
  return cam;
}

// ------------------------------------------------------------------ projections
// Each: fwd(d) -> [u, v, ok], inv(u, v) -> [d, ok], plus extents for fov.
const lonlat = (d) => [Math.atan2(d[0], d[2]), Math.atan2(d[1], Math.hypot(d[0], d[2]))];
const fromLL = (lon, lat) => { const c = Math.cos(lat); return [Math.sin(lon) * c, Math.sin(lat), Math.cos(lon) * c]; };

function azimuthal(g, ginv, thMax) {
  return {
    fwd(d) { const rho = Math.hypot(d[0], d[1]); const th = Math.atan2(rho, d[2]); const r = g(th); const s = rho > 1e-12 ? r / rho : 1; return [d[0] * s, d[1] * s, th < thMax - 1e-6]; },
    inv(u, v) { const r = Math.hypot(u, v); const [th, ok] = ginv(r); const s = r > 1e-12 ? Math.sin(th) / r : 1; return [[u * s, v * s, Math.cos(th)], ok]; },
    uExt(h) { return g(Math.min(h, thMax - 1e-6)); }, vExt(h) { return g(Math.min(h, thMax - 1e-6)); },
  };
}

function panini(p) {
  const dd = p.d ?? 1, vc = p.vcompress ?? 0;
  const vf = (lat) => (1 - vc) * Math.tan(lat) + vc * lat;
  const vinv = (y) => { if (vc <= 0) return Math.atan(y); let lat = vc < 1 ? Math.atan(y) : y; for (let k = 0; k < 8; k++) { const f = vf(lat) - y; const df = (1 - vc) / Math.cos(lat) ** 2 + vc; lat = Math.max(-1.55, Math.min(1.55, lat - f / df)); } return lat; };
  return {
    fwd(d) { const [lon, lat] = lonlat(d); const S = (dd + 1) / (dd + Math.cos(lon)); return [S * Math.sin(lon), S * vf(lat), (dd + Math.cos(lon)) > 1e-6 && Math.abs(lat) < 1.55]; },
    inv(u, v) {
      const k = u * u / (dd + 1) ** 2; const disc = k * k * dd * dd - (k + 1) * (k * dd * dd - 1);
      const cl = (-k * dd + Math.sqrt(Math.max(disc, 0))) / (k + 1);
      const lon = Math.sign(u) * Math.acos(Math.max(-1, Math.min(1, cl)));
      const S = (dd + 1) / (dd + Math.cos(lon));
      return [fromLL(lon, vinv(v / S)), disc >= 0];
    },
    uExt(h) { return (dd + 1) / (dd + Math.cos(h)) * Math.sin(h); }, vExt(h) { return vf(h); },
  };
}

function transverse(base) {
  return {
    fwd(d) { const [u2, v2, ok] = base.fwd([-d[1], d[0], d[2]]); return [v2, -u2, ok]; },
    inv(u, v) { const [d2, ok] = base.inv(-v, u); return [[d2[1], -d2[0], d2[2]], ok]; },
    uExt(h) { return base.vExt(h); }, vExt(h) { return base.uExt(h); },
  };
}

export const PROJ_IDS = {
  equirectangular: 0, cylindrical: 1, mercator: 2, rectilinear: 3, fisheye: 4, stereographic: 5, equisolid: 6, orthographic: 7,
  panini: 8, sinusoidal: 9, architectural: 10, transverse_equirectangular: 11, transverse_cylindrical: 12, transverse_mercator: 13, transverse_panini: 14,
};
export const PROJ_LABELS = {
  equirectangular: 'Equirectangular', cylindrical: 'Cylindrical', mercator: 'Mercator', rectilinear: 'Rectilinear', fisheye: 'Fisheye',
  stereographic: 'Stereographic', equisolid: 'Equisolid (mirror ball)', orthographic: 'Orthographic', panini: 'Panini / Vedutismo',
  sinusoidal: 'Sinusoidal', architectural: 'Architectural', transverse_equirectangular: 'Transverse equirectangular',
  transverse_cylindrical: 'Transverse cylindrical', transverse_mercator: 'Transverse Mercator', transverse_panini: 'Transverse Panini',
};
export const PROJ_MAX = {
  equirectangular: [360, 180], cylindrical: [360, 179], mercator: [360, 179], rectilinear: [179, 179], fisheye: [360, 360],
  stereographic: [360, 360], equisolid: [360, 360], orthographic: [180, 180], panini: [320, 160], sinusoidal: [360, 180],
  architectural: [360, 179], transverse_equirectangular: [180, 360], transverse_cylindrical: [179, 360], transverse_mercator: [179, 360],
  transverse_panini: [160, 320],
};

export function makeProjection(name, params = {}) {
  const P = {
    equirectangular: { fwd: (d) => { const [lo, la] = lonlat(d); return [lo, la, true]; }, inv: (u, v) => [fromLL(u, Math.max(-Math.PI / 2, Math.min(Math.PI / 2, v))), Math.abs(v) <= Math.PI / 2 + 1e-9 && Math.abs(u) <= Math.PI + 1e-9], uExt: (h) => h, vExt: (h) => h },
    cylindrical: { fwd: (d) => { const [lo, la] = lonlat(d); return [lo, Math.tan(la), Math.abs(la) < Math.PI / 2 - 1e-6]; }, inv: (u, v) => [fromLL(u, Math.atan(v)), Math.abs(u) <= Math.PI + 1e-9], uExt: (h) => h, vExt: (h) => Math.tan(h) },
    mercator: { fwd: (d) => { const [lo, la] = lonlat(d); return [lo, Math.asinh(Math.tan(Math.max(-1.55, Math.min(1.55, la)))), Math.abs(la) < 1.55]; }, inv: (u, v) => [fromLL(u, Math.atan(Math.sinh(v))), Math.abs(u) <= Math.PI + 1e-9], uExt: (h) => h, vExt: (h) => Math.asinh(Math.tan(h)) },
    rectilinear: { fwd: (d) => { const ok = d[2] > 1e-6; const z = ok ? d[2] : 1e-6; return [d[0] / z, d[1] / z, ok]; }, inv: (u, v) => [norm([u, v, 1]), true], uExt: Math.tan, vExt: Math.tan },
    fisheye: azimuthal((t) => t, (r) => [r, r <= Math.PI], Math.PI),
    stereographic: azimuthal((t) => 2 * Math.tan(t / 2), (r) => [2 * Math.atan(r / 2), true], Math.PI * 0.999),
    equisolid: azimuthal((t) => 2 * Math.sin(t / 2), (r) => [2 * Math.asin(Math.min(1, r / 2)), r <= 2], Math.PI),
    orthographic: azimuthal(Math.sin, (r) => [Math.asin(Math.min(1, r)), r <= 1], Math.PI / 2),
    sinusoidal: { fwd: (d) => { const [lo, la] = lonlat(d); return [lo * Math.cos(la), la, true]; }, inv: (u, v) => { const cl = Math.cos(v); const lon = cl > 1e-9 ? u / cl : 0; return [fromLL(lon, v), Math.abs(lon) <= Math.PI && Math.abs(v) <= Math.PI / 2]; }, uExt: (h) => h, vExt: (h) => h },
    architectural: { fwd: (d) => { let [lo, la] = lonlat(d); la = Math.max(-1.55, Math.min(1.55, la)); return [lo, la > 0 ? 1.25 * Math.asinh(Math.tan(0.8 * la)) : la, true]; }, inv: (u, v) => { const lat = v > 0 ? Math.atan(Math.sinh(v / 1.25)) / 0.8 : v; return [fromLL(u, lat), Math.abs(u) <= Math.PI && Math.abs(lat) <= Math.PI / 2]; }, uExt: (h) => h, vExt: (h) => (h > 0 ? 1.25 * Math.asinh(Math.tan(0.8 * h)) : h) },
  };
  P.panini = panini(params);
  P.transverse_equirectangular = transverse(P.equirectangular);
  P.transverse_cylindrical = transverse(P.cylindrical);
  P.transverse_mercator = transverse(P.mercator);
  P.transverse_panini = transverse(panini(params));
  const pr = P[name] || P.equirectangular;
  pr.name = name;
  return pr;
}

// Canvas geometry of the panorama: plane extents for the project's hfov / vfov.
export function canvasInfo(project) {
  const pn = project.panorama;
  const P = makeProjection(pn.projection, pn.proj_params || {});
  const uExt = 2 * P.uExt(pn.hfov * DEG / 2);
  const vExt = 2 * P.vExt(pn.vfov * DEG / 2);
  return { P, uExt, vExt, vOff: (pn.v_offset || 0) * vExt, aspect: vExt / uExt };
}
