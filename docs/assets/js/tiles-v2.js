/* Home page tiles: glazed painted tiles.
   Needs window.Painter (tiles-painter.js).
   Cost rules: each tile face is painted once on a CPU canvas; a wall is one canvas;
   a frame redraws only the cells that move (clip + clear per cell, never the whole canvas); DPR <= 1.5. */
(() => {
  'use strict';
  const P = window.Painter;
  const { DPR, C, rng, mix, brush, linePts, quadPts, arcPts, tileShape, glaze, poolEdges, speckle, paintRegion, MOTIF } = P;
  const RM = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const PHONE = () => document.documentElement.clientWidth < 760;

  // ---------- one tile face ----------
  function crackle(ctx, r, S) {
    ctx.save();
    ctx.strokeStyle = '#040b18';
    ctx.lineWidth = 0.6 * DPR;
    const n = 2 + Math.floor(r() * 3);
    for (let k = 0; k < n; k++) {
      ctx.globalAlpha = 0.16 + r() * 0.14;
      let x = r() * S, y = r() * S, a = r() * 6.283;
      ctx.beginPath(); ctx.moveTo(x, y);
      const m = 4 + Math.floor(r() * 5);
      for (let i = 0; i < m; i++) { a += (r() - 0.5) * 1.4; x += Math.cos(a) * S * 0.09; y += Math.sin(a) * S * 0.09; ctx.lineTo(x, y); }
      ctx.stroke();
    }
    ctx.restore();
  }
  function tile(S, seed, bg, draw, rot) {
    const cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const ctx = cv.getContext('2d', { willReadFrequently: true }); // CPU canvas: no GPU texture per face
    const r = rng(seed);
    const shape = tileShape(S, r);
    ctx.save();
    ctx.clip(shape.path);
    glaze(ctx, r, S, bg, bg === C.glaze ? { alpha: 0.42, shade: 0.32 } : { alpha: 0.6, shade: 0.4, dry: 0.4, pool: 0.3, dark: mix(bg, C.ground, 0.6) });
    if (draw) {
      ctx.save();
      if (rot) { ctx.translate(S / 2, S / 2); ctx.rotate(rot * Math.PI / 2); ctx.translate(-S / 2, -S / 2); }
      draw(ctx, r, S);
      ctx.restore();
    }
    poolEdges(ctx, r, shape.edges, bg, S, bg === C.glaze ? 1.1 : 1.35);
    speckle(ctx, r, S, bg);
    crackle(ctx, r, S);
    ctx.restore();
    return cv;
  }
  const quietCache = new Map();
  function quietFace(S, k) {
    const key = S + ':' + (k % 8);
    if (!quietCache.has(key)) quietCache.set(key, tile(S, 9001 + (k % 8) * 31, C.glaze));
    return quietCache.get(key);
  }

  // a tile cut from a wall-sized art layer: the art is painted once for the whole wall,
  // each tile takes its square a little off its neighbour's, over its own glaze
  function sliceTile(S, layer, i, j, seed) {
    const cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    const r = rng(seed);
    ctx.drawImage(quietFace(S, i * 3 + j * 5), 0, 0);
    ctx.globalCompositeOperation = 'source-atop';
    const jx = (r() - 0.5) * S * 0.03, jy = (r() - 0.5) * S * 0.03;
    ctx.drawImage(layer, i * S + jx, j * S + jy, S, S, 0, 0, S, S);
    ctx.globalCompositeOperation = 'source-over';
    crackle(ctx, r, S);
    return cv;
  }

  // ---------- work queue: painting runs in small slices between frames ----------
  const queue = [];
  let painting = 0;
  function pump() {
    const t0 = performance.now();
    while (queue.length && performance.now() - t0 < 10) queue.shift()();
    if (queue.length) requestAnimationFrame(pump);
  }
  function later(fn) { queue.push(fn); if (queue.length === 1) requestAnimationFrame(pump); }

  // ---------- a wall ----------
  const walls = new Set();
  let loopOn = false;
  const easeIO = (x) => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2);
  class Wall {
    constructor(canvas, cols, rows, cssTile) {
      this.cv = canvas; this.cols = cols; this.rows = rows;
      const box = canvas.parentElement;
      box.style.setProperty('--cols', cols); box.style.setProperty('--rows', rows);
      const w = cssTile ? cssTile * cols : box.clientWidth;
      this.S = Math.max(8, Math.round(w / cols * DPR));
      canvas.width = this.S * cols; canvas.height = this.S * rows;
      this.ctx = canvas.getContext('2d');
      this.ctx.fillStyle = C.ground; this.ctx.fillRect(0, 0, canvas.width, canvas.height);
      this.cur = new Array(cols * rows).fill(null);
      this.anim = new Map();
      this.fc = new Map();
      this.dead = false;
      walls.add(this);
    }
    // a resize rebuilds the wall: the old one stops drawing and its pending paint work is skipped
    dispose() { this.dead = true; this.anim.clear(); walls.delete(this); }
    idx(i, j) { return j * this.cols + i; }
    // paint every face of a spec; resolves with the face array
    paint(spec) {
      painting++;
      const S = this.S, faces = new Array(this.cols * this.rows).fill(null);
      return new Promise((done) => {
        for (let j = 0; j < this.rows; j++) for (let i = 0; i < this.cols; i++) later(() => {
          if (this.dead) return;
          const t = spec(i, j, S);
          if (!t) return;
          const k = this.idx(i, j);
          if (t.sig && this.fc.has(t.sig)) { faces[k] = this.fc.get(t.sig); return; } // same face as an earlier state: no repaint, no flip
          if (t.layer) { faces[k] = sliceTile(S, t.layer(S), i, j, i * 977 + j * 131 + 7); if (t.sig) this.fc.set(t.sig, faces[k]); return; }
          if ((t.bg === C.glaze || !t.bg) && !t.draw) { faces[k] = quietFace(S, i * 3 + j * 5); return; }
          faces[k] = tile(S, t.seed ?? (i * 977 + j * 131 + 7), t.bg || C.glaze, t.draw, t.rot);
          if (t.sig) this.fc.set(t.sig, faces[k]);
        });
        later(() => { painting--; done(faces); });
      });
    }
    drawCell(k, face, sx, over) {
      if (this.dead) return;
      const S = this.S, x = (k % this.cols) * S, y = Math.floor(k / this.cols) * S, g = this.ctx;
      g.save();
      g.beginPath(); g.rect(x, y, S, S); g.clip();
      if (!over) { const al = g.globalAlpha; g.globalAlpha = 1; g.fillStyle = C.ground; g.fillRect(x, y, S, S); g.globalAlpha = al; }
      if (face) {
        if (sx === undefined || sx === 1) g.drawImage(face, x, y);
        else { g.translate(x + S / 2, y); g.scale(Math.max(0.001, sx), 1); g.drawImage(face, -S / 2, 0); }
      }
      g.restore();
    }
    quietAll() { this.set(this.cur.map((_, k) => quietFace(this.S, (k % this.cols) * 3 + Math.floor(k / this.cols) * 5))); }
    set(faces) {
      this.anim.clear();
      faces.forEach((f, k) => { this.cur[k] = f; this.drawCell(k, f); });
    }
    // what the cell will show once its queued motion ends
    last(k) { const q = this.anim.get(k); return q && q.length ? q[q.length - 1].to : this.cur[k]; }
    // a tile turns over (flip) or a new layer settles into it (fade).
    // then = true queues after the cell's current motion; otherwise it replaces any queued motion.
    move(k, face, delay, dur, kind, then) {
      if (this.dead) return;
      const now = performance.now();
      let q = this.anim.get(k) || [];
      if (!then) {
        const run = q.find((a) => now >= a.t0);
        q = run ? [run] : [];
      }
      const from = q.length ? q[q.length - 1].to : this.cur[k];
      if (from === face) { this.anim.set(k, q); return; }
      q.push({ kind, from, to: face, t0: now + delay, dur });
      this.anim.set(k, q);
      kick();
    }
    flip(k, face, delay, dur = 560, then) { this.move(k, face, delay, dur, 'flip', then); }
    fade(k, face, delay, dur = 420, then) { this.move(k, face, delay, dur, 'fade', then); }
    tick(now) {
      for (const [k, q] of this.anim) {
        if (!q.length) { this.anim.delete(k); continue; }
        const a = q[0];
        const t = (now - a.t0) / a.dur;
        if (t < 0) continue;
        if (t >= 1) { this.cur[k] = a.to; this.drawCell(k, a.to); q.shift(); if (q.length && q[0].t0 < now) q[0].t0 = now; continue; }
        const e = easeIO(t);
        if (a.kind === 'fade') { this.drawCell(k, a.from); const g = this.ctx; g.globalAlpha = e; this.drawCell(k, a.to, 1, true); g.globalAlpha = 1; }
        else this.drawCell(k, e < 0.5 ? a.from : a.to, Math.abs(Math.cos(Math.PI * e)));
      }
      return this.anim.size > 0;
    }
  }
  function kick() { if (!loopOn) { loopOn = true; requestAnimationFrame(loop); } }
  function loop(now) {
    if (document.hidden) { loopOn = false; return; }
    let any = false;
    for (const w of walls) any = w.tick(now) || any;
    if (any) requestAnimationFrame(loop); else loopOn = false;
  }
  document.addEventListener('visibilitychange', () => { if (!document.hidden) kick(); });

  const inRect = (i, j, R) => R && i >= R[0] && i < R[2] && j >= R[1] && j < R[3];
  const quiet = () => ({ bg: C.glaze });
  const jitter = (k, span) => ((k * 2654435761) >>> 0) % 997 / 997 * span;

  // ---------- hero: the ridgeline, full bleed, in the three tones ----------
  function ridgeLayer(cols, rows, o) {
    o = Object.assign({ n: 13, top: 0.16, span: 0.78, w: 0.05, colour: (d) => mix(C.glaze, C.peri, 0.35 + 0.6 * d) }, o);
    const cache = new Map();
    return (S) => {
      if (cache.has(S)) return cache.get(S);
      const r = rng(1107);
      const cv = document.createElement('canvas');
      cv.width = cols * S; cv.height = rows * S;
      const ctx = cv.getContext('2d', { willReadFrequently: true });
      for (let k = 0; k < o.n; k++) {
        const depth = k / (o.n - 1);
        const base = rows * (o.top + o.span * depth);
        const peaks = [];
        const m = 2 + Math.floor(r() * 3);
        for (let q = 0; q < m; q++) peaks.push({ c: cols * (0.12 + r() * 0.8), h: (0.5 + r() * 1.6) * (0.45 + 0.75 * depth) * (rows / 9), w: 0.35 + r() * 0.9 });
        const pts = [];
        for (let x = -0.5; x <= cols + 0.5; x += 0.06) {
          let y = base + 0.04 * Math.sin(x * 3 + k);
          for (const p of peaks) y -= p.h * Math.exp(-0.5 * ((x - p.c) / p.w) ** 2);
          pts.push([x * S, y * S]);
        }
        // the ground under a nearer ridge hides the ridges behind it
        ctx.globalCompositeOperation = 'destination-out';
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
        for (const p of pts) ctx.lineTo(p[0], p[1]);
        ctx.lineTo(pts[pts.length - 1][0], (rows + 1) * S); ctx.lineTo(pts[0][0], (rows + 1) * S); ctx.closePath();
        ctx.fill();
        ctx.globalCompositeOperation = 'source-over';
        brush(ctx, r, pts, S * o.w, o.colour(depth), { alpha: 0.9, dry: 0.25, shade: 0.12, pool: 0.3, rag: 0, tail: 0 });
      }
      cache.set(S, cv);
      return cv;
    };
  }

  function heroGrid() {
    const sec = document.querySelector('.v2-hero');
    const vw = document.documentElement.clientWidth, vh = sec.clientHeight || innerHeight;
    const Sc = PHONE() ? 56 : Math.max(64, Math.min(112, Math.round(vw / 15)));
    return { sec, Sc, cols: Math.ceil(vw / Sc), rows: Math.ceil(vh / Sc), key: [PHONE(), Sc, Math.ceil(vw / Sc), Math.ceil(vh / Sc)].join() };
  }
  let heroWall = null, heroKey = '';
  // first = true plays the one turn-over; a rebuild after a resize shows the finished wall
  function hero(first) {
    const { sec, Sc, cols, rows, key } = heroGrid();
    if (heroWall) heroWall.dispose();
    heroKey = key;
    // the panel sits above the band the editor covers, and never above the wall's first row
    const span = PHONE() ? 4 : Math.max(3, Math.ceil(220 / Sc)), // rows the name, line and buttons need
      r0 = Math.max(0, (PHONE() ? rows - 8 : rows - 7)), r1 = Math.max(r0 + span, rows - 4);
    const cart = PHONE() ? [0, r0, cols, r1] : [1, r0, Math.min(9, cols - 1), r1];
    const panel = sec.querySelector('.v2-cartouche--hero');
    panel.style.cssText = `--c0:${cart[0]};--c1:${cart[2]};--r0:${cart[1]};--r1:${cart[3]}`;
    const W = heroWall = new Wall(sec.querySelector('canvas'), cols, rows, Sc);
    Object.assign(sec.querySelector('.v2-wall').style, { width: cols * Sc + 'px', height: rows * Sc + 'px' });
    const pal = [C.peri, C.aqua, C.peach];
    const layer = ridgeLayer(cols, rows, { n: 15, top: 0.08, span: 0.88, w: 0.075, colour: (d) => (d > 0.86 ? C.ink : pal[Math.floor(d * 15) % 3]) });
    // the wall is laid plain first; the painted faces turn over together, once
    W.quietAll();
    W.paint((i, j) => (inRect(i, j, cart) ? quiet() : { layer })).then((faces) => {
      if (RM || !first) return W.set(faces);
      faces.forEach((f, k) => W.flip(k, f, jitter(k, 160), 640));
    });
  }

  // ---------- promises: one pinned wall that changes in place ----------
  function layout() {
    return PHONE()
      ? { cols: 8, rows: 6, cart: null,
          causal: { tv: [1, 0], search: [4, 1], holiday: [1, 2], base: [3, 4], sales: [6, 3],
            lit: [[[1, 0], [2, 0], [3, 0], [4, 0], [4, 1]], [[4, 0], [5, 0], [6, 0], [6, 1], [6, 2], [6, 3]], [[4, 1], [5, 1], [6, 1]]],
            calm: [[[1, 2], [2, 2], [3, 2], [4, 2], [4, 1]], [[4, 2], [5, 2], [6, 2]], [[3, 4], [4, 4], [5, 4], [6, 4], [6, 3]]] },
          bayes: { x: 4.6, y: 3, sx: 1.5, sy: 1.15 } }
      : { cols: 16, rows: 6, cart: [0, 4, 6, 6],
          causal: { tv: [1, 0], search: [8, 1], holiday: [1, 2], base: [9, 4], sales: [14, 2],
            lit: [[[1, 0], [8, 0], [8, 1]], [[8, 0], [14, 0], [14, 2]], [[8, 1], [14, 1]]],
            calm: [[[1, 2], [8, 2], [8, 1]], [[8, 2], [14, 2]], [[9, 4], [14, 4], [14, 2]]] },
          bayes: { x: 10.2, y: 2.6, sx: 2.3, sy: 1.15 } };
  }
  // expand [[a],[b],...] corner lists into unit steps
  function walk(pts) {
    const out = [pts[0]];
    for (let q = 1; q < pts.length; q++) {
      let [x, y] = out[out.length - 1];
      const [tx, ty] = pts[q];
      while (x !== tx || y !== ty) { x += Math.sign(tx - x); if (x === tx) y += Math.sign(ty - y); out.push([x, y]); }
    }
    return out;
  }

  function amortizedSpec(L) {
    const cols3 = [C.aqua, C.peach, C.peri];
    return (i, j) => {
      if (inRect(i, j, L.cart)) return quiet();
      const bi = Math.floor(i / 2), bj = Math.floor(j / 2), u = i % 2, v = j % 2;
      const rot = [[2, 1], [3, 0]][u][v];
      if ((bi + bj * 2) % 4 === 3) return { bg: C.ink, draw: (ctx, r, S) => MOTIF.quarter(ctx, r, S, C.glaze), rot: (rot + 2) % 4 };
      const fg = cols3[(bi + bj) % 3];
      return { draw: (ctx, r, S) => MOTIF.quarter(ctx, r, S, fg), rot };
    };
  }

  // Causal: TV drives search and sales; holiday and base demand drive them too but do not move.
  // lit = false paints the graph before the change (all calm); lit = true after it.
  function causalGraph(L) {
    const G = L.causal, cells = new Map();
    const at = (p) => { const k = p[0] + ',' + p[1]; if (!cells.has(k)) cells.set(k, { dirs: [], knot: null, dist: Infinity }); return cells.get(k); };
    const dirOf = (a, b) => (b[0] > a[0] ? 'E' : b[0] < a[0] ? 'W' : b[1] > a[1] ? 'S' : 'N');
    const opp = { E: 'W', W: 'E', N: 'S', S: 'N' };
    const add = (path, lit) => {
      const w = walk(path);
      for (let q = 0; q < w.length - 1; q++) { const d = dirOf(w[q], w[q + 1]); at(w[q]).dirs.push({ d, lit }); at(w[q + 1]).dirs.push({ d: opp[d], lit }); }
    };
    G.lit.forEach((p) => add(p, true));
    G.calm.forEach((p) => add(p, false));
    at(G.tv).knot = { lit: true, R: 0.22 };
    at(G.search).knot = { lit: true, R: 0.17 };
    at(G.holiday).knot = { lit: false, R: 0.2 };
    at(G.base).knot = { lit: false, R: 0.2 };
    at(G.sales).knot = { lit: true, R: 0.3 };
    // distance from TV along lit edges: the order in which the change reaches each tile
    const key = (p) => p[0] + ',' + p[1];
    const adj = new Map();
    G.lit.forEach((p) => { const w = walk(p); for (let q = 0; q < w.length - 1; q++) { const a = key(w[q]); (adj.get(a) || adj.set(a, []).get(a)).push(key(w[q + 1])); } });
    const front = [key(G.tv)]; cells.get(key(G.tv)).dist = 0;
    while (front.length) { const a = front.shift(); for (const b of adj.get(a) || []) { const cb = cells.get(b); if (cb.dist > cells.get(a).dist + 1) { cb.dist = cells.get(a).dist + 1; front.push(b); } } }
    return cells;
  }
  function causalSpec(L, cells, after) {
    const lit = C.peach, calm = mix(C.peri, C.glaze, 0.25), weave = mix(C.peri, C.glaze, 0.78);
    const colOf = (isLit) => (after && isLit ? lit : calm);
    const mid = (S) => ({ N: [S / 2, -S * 0.05], S: [S / 2, S * 1.05], E: [S * 1.05, S / 2], W: [-S * 0.05, S / 2] });
    return (i, j) => {
      const c = cells.get(i + ',' + j);
      if (!c) return inRect(i, j, L.cart) ? quiet() : { sig: 'w' + i + ',' + j, draw: (ctx, r, S) => MOTIF.arcs(ctx, r, S, weave), rot: (i * 7 + j * 3 + ((i * j) % 5)) % 2, seed: i * 31 + j * 17 + 3 };
      const hasLit = c.dirs.some((d) => d.lit) || (c.knot && c.knot.lit);
      return {
        seed: i * 977 + j * 131 + 7,
        sig: 'c' + i + ',' + j + (after && hasLit ? ':lit' : ''),
        draw(ctx, r, S) {
          const M = mid(S), cx = S / 2, cy = S / 2;
          const ds = [...c.dirs].map((x) => ({ d: x.d, col: colOf(x.lit) })).sort((a, b) => (a.col === lit) - (b.col === lit));
          const opp2 = { E: 'W', W: 'E', N: 'S', S: 'N' };
          const paint = (pts, col) => {
            brush(ctx, r, pts, S * 0.22, col, { alpha: 0.9, dry: 0.04, shade: 0.08, pool: 0, rag: 0, tail: 0 });
            brush(ctx, r, pts, S * 0.25, col, { alpha: 0.7, dry: 0.3, shade: 0.16, pool: 0.42, rag: 0, tail: 0.04 });
          };
          const done = new Set();
          for (const a of ds) {
            if (done.has(a)) continue;
            const b = ds.find((x) => x !== a && !done.has(x) && x.col === a.col && x.d === opp2[a.d]);
            if (b) { paint(linePts(M[a.d][0], M[a.d][1], M[b.d][0], M[b.d][1], 0.02, r), a.col); done.add(a); done.add(b); continue; }
            const t = ds.length === 2 ? ds.find((x) => x !== a && !done.has(x) && x.col === a.col) : null;
            if (t) { paint(quadPts(M[a.d][0], M[a.d][1], cx, cy, M[t.d][0], M[t.d][1]), a.col); done.add(a); done.add(t); continue; }
            const ex = cx + (cx - M[a.d][0]) * 0.18, ey = cy + (cy - M[a.d][1]) * 0.18;
            paint(linePts(M[a.d][0], M[a.d][1], ex, ey, 0.03, r), a.col); done.add(a);
          }
          if (c.knot) {
            const R = S * c.knot.R, rim = arcPts(cx, cy, R, 0, Math.PI * 2, r, 0.05);
            const p = new Path2D(); p.moveTo(rim[0][0], rim[0][1]); for (const q of rim) p.lineTo(q[0], q[1]); p.closePath();
            const strokes = []; for (let rr = R * 0.3; rr < R; rr += R * 0.45) strokes.push({ pts: arcPts(cx, cy, rr, r() * 3, r() * 3 + 6.6, r, 0.05), w: R * 0.5 });
            paintRegion(ctx, r, S, p, [rim], colOf(c.knot.lit), strokes);
          }
        },
      };
    };
  }

  // Bayesian: 1,000 dabs, one per posterior draw. upto = how many have landed.
  function bayesDabs(L) {
    const B = L.bayes, r = rng(1000);
    const gauss = () => { let u = 0; while (!u) u = r(); const v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283 * v); };
    const dabs = [];
    for (let k = 0; k < 1000; k++) {
      const gx = gauss(), gy = gauss(), d = Math.hypot(gx, gy);
      dabs.push({ x: B.x + gx * B.sx, y: B.y + gy * B.sy, a: (r() - 0.5) * 0.7, L: 0.22 + r() * 0.16,
        col: d < 1.1 ? C.aqua : d < 1.9 ? mix(C.aqua, C.peri, 0.5) : C.peri, al: d < 1.1 ? 0.75 : d < 1.9 ? 0.6 : 0.45 });
    }
    return dabs;
  }
  function bayesLayer(L, dabs, upto) {
    const cache = new Map();
    return (S) => {
      if (cache.has(S)) return cache.get(S);
      const cv = document.createElement('canvas');
      cv.width = L.cols * S; cv.height = L.rows * S;
      const ctx = cv.getContext('2d', { willReadFrequently: true });
      for (const p of dabs.slice(0, upto)) {
        const dr = rng(Math.round(p.x * 1e4 + p.y * 1e3)); // a dab looks the same at every stage
        const dx = Math.cos(p.a) * p.L * S / 2, dy = Math.sin(p.a) * p.L * S / 2;
        brush(ctx, dr, linePts(p.x * S - dx, p.y * S - dy, p.x * S + dx, p.y * S + dy, 0.2, dr), S * 0.13, p.col,
          { alpha: p.al, dry: 0.35, shade: 0.15, pool: 0.3, rag: 0.05, tail: 0.3 });
      }
      cache.set(S, cv);
      return cv;
    };
  }
  function bayesSpec(L, dabs, upto) {
    const layer = bayesLayer(L, dabs, upto);
    return (i, j) => {
      if (inRect(i, j, L.cart)) return quiet();
      const n = dabs.slice(0, upto).filter((p) => p.x > i - 0.3 && p.x < i + 1.3 && p.y > j - 0.3 && p.y < j + 1.3).length;
      if (!n) return quiet();
      return { layer, sig: 'b' + i + ',' + j + ':' + n };
    };
  }

  function promises() {
    const sec = document.querySelector('[data-pin]');
    const box = sec.querySelector('.v2-wall--promise');
    let L, W, cells, F, phone;
    // (re)builds the wall for the current layout; each promise in F resolves to its faces, painted in the order needed
    function build() {
      if (W) W.dispose();
      phone = PHONE(); L = layout();
      W = new Wall(box.querySelector('canvas'), L.cols, L.rows);
      W.quietAll();
      cells = causalGraph(L); const dabs = bayesDabs(L);
      F = {};
      F.amort = W.paint(amortizedSpec(L));
      F.calm = W.paint(causalSpec(L, cells, false));
      F.lit = W.paint(causalSpec(L, cells, true));
      F.b1 = W.paint(bayesSpec(L, dabs, 250));
      F.b2 = W.paint(bayesSpec(L, dabs, 600));
      F.b3 = W.paint(bayesSpec(L, dabs, 1000));
    }
    build();

    const says = [...sec.querySelectorAll('.v2-say')], marks = [...sec.querySelectorAll('.v2-steps li')];
    let cur = -1, run = 0;
    const all = (faces, span, delay = 0, dur) => faces.forEach((f, k) => W.flip(k, f, delay + jitter(k, span), dur));
    const replayBtn = sec.querySelector('[data-replay]');
    // still = true shows the state's finished picture (after a rebuild)
    async function show(k, still) {
      const me = ++run;
      says.forEach((s, q) => s.classList.toggle('on', q === k));
      if (replayBtn) replayBtn.hidden = RM || k !== 1;
      marks.forEach((s, q) => s.classList.toggle('on', q === k));
      box.classList.toggle('is-causal', k === 1);
      if (k === 0) {
        const f = await F.amort; if (me !== run) return;
        RM || still ? W.set(f) : all(f, 90); // the whole answer turns over at once
      } else if (k === 1) {
        const [calm, lit] = await Promise.all([F.calm, F.lit]); if (me !== run) return;
        if (RM || still) return W.set(lit);
        all(calm, 90);
        propagate(calm, lit, 900);
      } else {
        const [b1, b2, b3] = await Promise.all([F.b1, F.b2, F.b3]); if (me !== run) return;
        if (RM || still) return W.set(b3);
        all(b1, 120);
        b2.forEach((f, q) => W.fade(q, f, 1000 + jitter(q, 300), 500, true));
        b3.forEach((f, q) => W.fade(q, f, 1800 + jitter(q, 300), 500, true));
      }
    }
    // the change starts at TV and reaches only the tiles that depend on it
    function propagate(calm, lit, delay) {
      for (const [key, c] of cells) {
        if (!isFinite(c.dist) && !c.dirs.some((d) => d.lit)) continue;
        const [i, j] = key.split(',').map(Number), q = W.idx(i, j);
        if (lit[q] === calm[q]) continue;
        const d = isFinite(c.dist) ? c.dist : 0;
        W.flip(q, lit[q], delay + d * 85, 480, true);
      }
    }
    // Replays the causal change: a click or tap on the wall, or the keyboard button in the Causal text.
    async function replay() {
      if (cur !== 1 || RM) return;
      const [calm, lit] = await Promise.all([F.calm, F.lit]);
      calm.forEach((f, q) => lit[q] !== f && W.flip(q, f, 0, 300));
      propagate(calm, lit, 250);
    }
    box.addEventListener('click', replay);
    if (replayBtn) replayBtn.addEventListener('click', (e) => { e.stopPropagation(); replay(); });
    // phone and desktop use different grids: rebuild when the layout changes
    onLayout(() => { if (PHONE() === phone) return; build(); if (cur >= 0) show(cur, true); });
    function onScroll() {
      const b = sec.getBoundingClientRect();
      const t = Math.min(0.999, Math.max(0, -b.top / (b.height - innerHeight)));
      const k = Math.floor(t * 3);
      if (k === cur) return;
      cur = k; show(k);
    }
    const forced = new URLSearchParams(location.search).get('promise');
    if (forced !== null) { cur = +forced; show(cur); return; }
    addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // ---------- River 1.0: woven arcs that run as waves; the release in a plain panel ----------
  function waveSpec(cart) {
    const pal = [C.aqua, mix(C.aqua, C.peri, 0.5), C.peri, mix(C.peri, C.glaze, 0.35)];
    return (i, j) => {
      if (inRect(i, j, cart)) return quiet();
      const fg = pal[Math.min(3, Math.abs(j - 2) + ((i * 5 + j) % 7 === 0 ? 1 : 0))];
      return { draw: (ctx, r, S) => MOTIF.arcs(ctx, r, S, fg), rot: (i + j) % 2 };
    };
  }
  function skirtSpec() {
    return (i) => ({ draw: (ctx, r, S) => MOTIF.quarter(ctx, r, S, [C.aqua, C.peri, C.peach][Math.floor(i / 2) % 3]), rot: i % 2 ? 3 : 2 });
  }
  function whenNear(el, fn) {
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); fn(); } }, { rootMargin: '600px 0px' });
    io.observe(el);
  }
  // a wall built when near the viewport, and rebuilt when the phone/desktop layout changes
  function lazyWall(el, make) {
    let W = null, phone;
    const build = () => { if (W) W.dispose(); phone = PHONE(); W = make(phone); };
    whenNear(el, build);
    onLayout(() => { if (W && PHONE() !== phone) build(); });
  }
  function quietWalls() {
    const win = document.querySelector('[data-wall="windows"]');
    lazyWall(win, (ph) => {
      const cols = ph ? 8 : 16, rows = ph ? 6 : 7, cart = ph ? null : [1, 4, 7, 7];
      const W = new Wall(win.querySelector('canvas'), cols, rows);
      W.paint(waveSpec(cart)).then((f) => W.set(f));
      return W;
    });
    const sk = document.querySelector('[data-wall="skirt"]');
    lazyWall(sk, (ph) => { const W = new Wall(sk.querySelector('canvas'), ph ? 12 : 32, 1); W.paint(skirtSpec()).then((f) => W.set(f)); return W; });
  }

  // resize and rotation: debounced; each wall decides whether its grid changed
  const layoutFns = [];
  function onLayout(fn) { layoutFns.push(fn); }
  let layoutTimer = 0;
  addEventListener('resize', () => { clearTimeout(layoutTimer); layoutTimer = setTimeout(() => layoutFns.forEach((fn) => fn()), 200); });
  onLayout(() => { if (heroGrid().key !== heroKey) hero(false); });

  function boot() {
    hero(true);
    promises();
    quietWalls();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
  window.__painted = () => painting === 0 && queue.length === 0;
  window.__wallsDone = () => painting === 0 && queue.length === 0 && ![...walls].some((w) => w.anim.size);
})();
