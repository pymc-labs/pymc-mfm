/* Painted tiles: one pinned wall that changes in place while you scroll.
   Amortized -> Causal -> Bayesian. Self-contained: it only touches .tiles-scene sections.
   Each tile face is painted once, by code, to an offscreen canvas
   (glaze, dry-brush bristles, pigment pooled at the edges, speckle).
   Frames only move those images: flips are transforms, nothing repaints per frame. */
(() => {
  'use strict';

  const DPR = Math.min(2, window.devicePixelRatio || 1);
  const RMQ = matchMedia('(prefers-reduced-motion: reduce)');

  const C = {
    ground: '#07142a',
    glaze: '#132849',
    aqua: '#b4e7dd',
    peach: '#f6ae72',
    peri: '#8fa9ff',
    ink: '#f2f5fa',
  };

  // ---------- small utilities ----------
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const hexCache = new Map();
  function hex(h) {
    let v = hexCache.get(h);
    if (!v) { const n = parseInt(h.slice(1), 16); v = [(n >> 16) & 255, (n >> 8) & 255, n & 255]; hexCache.set(h, v); }
    return v;
  }
  function mix(a, b, t) {
    const A = hex(a), B = hex(b);
    return '#' + A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, '0')).join('');
  }
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
  function easeBack(x) { const s = 1.25, c3 = s + 1; return 1 + c3 * Math.pow(x - 1, 3) + s * Math.pow(x - 1, 2); }

  // the dark tone a pigment pools into
  function poolOf(col) {
    if (col === C.glaze) return '#0a1a33';
    return mix(col, C.ground, 0.38);
  }

  // ---------- the brush ----------
  // pts: polyline in px. Paints it with separate bristles: each bristle
  // wanders, runs dry towards the tail, and the outer bristles carry darker, pooled pigment.
  function brush(ctx, r, pts, w, col, o) {
    o = Object.assign({ alpha: 0.85, dry: 0.3, pool: 0.32, shade: 0.12, rag: 0.06, tail: 0.18, dark: poolOf(col), bristle: 1.15 * DPR }, o);
    // resample by length
    const step = 1.4 * DPR;
    const P = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i], [x1, y1] = pts[i + 1];
      const L = Math.hypot(x1 - x0, y1 - y0);
      const n = Math.max(1, Math.ceil(L / step));
      for (let k = 0; k < n; k++) P.push([x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n]);
    }
    P.push(pts[pts.length - 1]);
    const n = P.length;
    if (n < 3) return;
    const N = P.map((p, k) => {
      const a = P[Math.max(0, k - 2)], b = P[Math.min(n - 1, k + 2)];
      const dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy) || 1;
      return [-dy / l, dx / l];
    });
    const nb = Math.max(5, Math.round(w / o.bristle));
    const g1 = 1.2 + r() * 2.5, g2 = 4 + r() * 5, gp1 = r() * 6.283, gp2 = r() * 6.283;
    const grain = (t) => 0.5 + 0.3 * Math.sin(g1 * t * 6.283 + gp1) + 0.2 * Math.sin(g2 * t * 6.283 + gp2);
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'round';
    for (let b = 0; b < nb; b++) {
      const u = (b + 0.5) / nb - 0.5 + (r() - 0.5) / nb;
      const edge = Math.abs(u) * 2;
      const pooled = edge > 0.8 && r() < 0.8;
      let shade;
      if (pooled) shade = mix(col, o.dark, o.pool * (0.6 + r() * 0.5));
      else if (r() < 0.55) shade = mix(col, o.dark, r() * o.shade);
      else shade = mix(col, '#ffffff', r() * o.shade * 0.5);
      ctx.strokeStyle = shade;
      ctx.lineWidth = (w / nb) * (1.0 + r() * 1.2);
      ctx.globalAlpha = Math.min(1, o.alpha * (0.55 + 0.45 * r()) * (pooled ? 1.2 : 1));
      const t0 = r() * o.rag, t1 = 1 - r() * o.tail;
      const f1 = 0.8 + r() * 2.2, f2 = 3 + r() * 6, p1 = r() * 6.283, p2 = r() * 6.283;
      const thr = o.dry * (0.35 + r() * 0.9);
      const wob = (r() - 0.5) * 0.6 * DPR, wf = 1 + r() * 3, wp = r() * 6.283;
      if (o.soft) {
        // soft dry brush: the pigment thins and returns along the bristle instead of cutting off
        ctx.lineWidth = (w / nb) * (1.7 + r() * 1.3);
        const base = ctx.globalAlpha * 0.85;
        let k0 = 0;
        const pos = (k) => {
          const t = k / (n - 1);
          const sw = 0.84 + 0.16 * Math.sin(Math.PI * t);
          const off = u * w * sw + wob * Math.sin(wf * t * 6.283 + wp);
          return [P[k][0] + N[k][0] * off, P[k][1] + N[k][1] * off];
        };
        const CH = 4;
        for (k0 = 0; k0 < n - 1; k0 += CH) {
          const k1 = Math.min(n - 1, k0 + CH);
          const t = (k0 + k1) / 2 / (n - 1);
          if (t < t0 || t > t1) continue;
          const tail = Math.max(0, (t - 0.45) / 0.55);
          const nz = 0.45 * grain(t) + 0.55 * (0.5 + 0.3 * Math.sin(f1 * t * 6.283 + p1) + 0.2 * Math.sin(f2 * t * 6.283 + p2));
          const a = clamp((nz - thr * (0.2 + tail * 1.3)) * 3.2 + 0.35, 0, 1) * (1 - 0.55 * tail);
          if (a < 0.03) continue;
          ctx.globalAlpha = base * a;
          ctx.beginPath();
          let q = pos(k0); ctx.moveTo(q[0], q[1]);
          for (let k = k0 + 1; k <= k1; k++) { q = pos(k); ctx.lineTo(q[0], q[1]); }
          ctx.stroke();
        }
        continue;
      }
      ctx.beginPath();
      let pen = false;
      for (let k = 0; k < n; k++) {
        const t = k / (n - 1);
        if (t < t0 || t > t1) { pen = false; continue; }
        const tail = Math.max(0, (t - 0.5) / 0.5);
        const nz = 0.45 * grain(t) + 0.55 * (0.5 + 0.3 * Math.sin(f1 * t * 6.283 + p1) + 0.2 * Math.sin(f2 * t * 6.283 + p2));
        if (nz < thr * (0.25 + tail * 1.5)) { pen = false; continue; }
        const sw = 0.84 + 0.16 * Math.sin(Math.PI * t);
        const off = u * w * sw + wob * Math.sin(wf * t * 6.283 + wp);
        const x = P[k][0] + N[k][0] * off, y = P[k][1] + N[k][1] * off;
        if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function linePts(x0, y0, x1, y1, bend, r) {
    const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1;
    const b = (r() - 0.5) * bend * L;
    const cx = mx - dy / L * b, cy = my + dx / L * b;
    const out = [];
    for (let i = 0; i <= 12; i++) {
      const t = i / 12, a = (1 - t) * (1 - t), m = 2 * t * (1 - t), c = t * t;
      out.push([a * x0 + m * cx + c * x1, a * y0 + m * cy + c * y1]);
    }
    return out;
  }
  function quadPts(x0, y0, cx, cy, x1, y1) {
    const out = [];
    for (let i = 0; i <= 16; i++) {
      const t = i / 16, a = (1 - t) * (1 - t), m = 2 * t * (1 - t), c = t * t;
      out.push([a * x0 + m * cx + c * x1, a * y0 + m * cy + c * y1]);
    }
    return out;
  }
  function arcPts(cx, cy, R, a0, a1, r, wob) {
    const out = [], n = 28, ph = r() * 6.283, ph2 = r() * 6.283;
    for (let i = 0; i <= n; i++) {
      const t = i / n, a = a0 + (a1 - a0) * t;
      const rr = R * (1 + wob * (Math.sin(t * 5 + ph) * 0.6 + Math.sin(t * 13 + ph2) * 0.4));
      out.push([cx + Math.cos(a) * rr, cy + Math.sin(a) * rr]);
    }
    return out;
  }

  // ---------- tile body ----------
  // An uneven, softly cornered square. Returns the path and its four edges as polylines.
  function tileShape(S, r) {
    const m = S * 0.014;
    const j = () => (r() - 0.5) * S * 0.022;
    const c = [[m + j(), m + j()], [S - m + j(), m + j()], [S - m + j(), S - m + j()], [m + j(), S - m + j()]];
    const rad = S * (0.05 + r() * 0.05);
    const edges = [];
    const p = new Path2D();
    for (let i = 0; i < 4; i++) {
      const a = c[i], b = c[(i + 1) % 4];
      const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy);
      const ux = dx / L, uy = dy / L, nx = -uy, ny = ux;
      const e = [];
      const ph = r() * 6.283;
      for (let k = 0; k <= 8; k++) {
        const t = k / 8, d = rad + (L - 2 * rad) * t;
        const wv = (k === 0 || k === 8) ? 0 : (Math.sin(t * 7 + ph) * 0.5 + (r() - 0.5)) * S * 0.006;
        e.push([a[0] + ux * d + nx * wv, a[1] + uy * d + ny * wv]);
      }
      edges.push(e);
    }
    p.moveTo(edges[0][0][0], edges[0][0][1]);
    for (let i = 0; i < 4; i++) {
      const e = edges[i];
      for (let k = 1; k < e.length; k++) p.lineTo(e[k][0], e[k][1]);
      const corner = c[(i + 1) % 4], nxt = edges[(i + 1) % 4][0];
      p.quadraticCurveTo(corner[0], corner[1], nxt[0], nxt[1]);
    }
    p.closePath();
    return { path: p, edges };
  }

  // Brushed glaze over the whole tile.
  function glaze(ctx, r, S, col, o) {
    o = Object.assign({ alpha: 0.32, shade: 0.2, dry: 0.55, pool: 0.18, dark: poolOf(col) }, o);
    ctx.fillStyle = col;
    ctx.fillRect(0, 0, S, S);
    const ang = (r() < 0.5 ? 0 : Math.PI / 2) + (r() - 0.5) * 0.35;
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const w = S * (0.2 + r() * 0.1);
    const rows = Math.ceil((S * 1.3) / (w * 0.75));
    for (let i = 0; i < rows; i++) {
      const off = -S * 0.65 + (i + 0.5) * (S * 1.3 / rows) + (r() - 0.5) * w * 0.3;
      const cx = S / 2 - sa * off, cy = S / 2 + ca * off;
      const L = S * 0.8;
      const dir = r() < 0.5 ? 1 : -1;
      brush(ctx, r, linePts(cx - ca * L * dir, cy - sa * L * dir, cx + ca * L * dir, cy + sa * L * dir, 0.08, r), w, col,
        { alpha: o.alpha, shade: o.shade, dry: o.dry, pool: o.pool, dark: o.dark, tail: 0.3 });
    }
  }

  function poolEdges(ctx, r, edges, col, S, strength) {
    const dark = poolOf(col);
    for (const e of edges) {
      brush(ctx, r, e, S * 0.075, dark, { alpha: 0.26 * strength, dry: 0.75, shade: 0.1, pool: 0.2, rag: 0.2, tail: 0.25, dark: dark });
      brush(ctx, r, e, S * 0.022, dark, { alpha: 0.42 * strength, dry: 0.9, shade: 0.1, pool: 0, rag: 0.15, tail: 0.3, dark: dark });
    }
  }

  function speckle(ctx, r, S, col) {
    const n = Math.round(S * S / 220);
    for (let i = 0; i < n; i++) {
      const light = r() < 0.5;
      ctx.fillStyle = light ? mix(col, '#ffffff', 0.35) : poolOf(col);
      ctx.globalAlpha = 0.05 + r() * 0.16;
      const rad = (0.3 + r() * 0.8) * DPR;
      ctx.beginPath();
      ctx.arc(r() * S, r() * S, rad, 0, 6.283);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // A filled region painted with bristles that follow its form, then pooled along its rim.
  function paintRegion(ctx, r, S, path, rim, fg, strokes) {
    ctx.save();
    ctx.clip(path);
    ctx.globalAlpha = 0.94;
    ctx.fillStyle = fg;
    ctx.fill(path);
    ctx.globalAlpha = 1;
    for (const s of strokes) brush(ctx, r, s.pts, s.w, fg, { alpha: 0.5, shade: 0.16, dry: 0.45, pool: 0.25, tail: 0.25 });
    ctx.restore();
    for (const e of rim) {
      brush(ctx, r, e, S * 0.05, poolOf(fg), { alpha: 0.38, dry: 0.6, pool: 0.15, rag: 0.05, tail: 0.12 });
      brush(ctx, r, e, S * 0.016, poolOf(fg), { alpha: 0.55, dry: 0.8, pool: 0, rag: 0.05, tail: 0.2 });
    }
  }

  // ---------- motifs (all painted in the tile's own frame, then rotated by quarter turns at draw time) ----------
  const MOTIF = {
    blank() {},

    // Quarter disc filling the top-left corner. Four of them meet as a circle.
    quarter(ctx, r, S, fg) {
      const R = S * (0.985 + (r() - 0.5) * 0.02);
      const rimPts = arcPts(0, 0, R, 0, Math.PI / 2, r, 0.012);
      const p = new Path2D();
      p.moveTo(0, 0);
      for (const q of rimPts) p.lineTo(q[0], q[1]);
      p.closePath();
      const strokes = [];
      const w = S * 0.2;
      for (let rr = w * 0.6; rr < R; rr += w * 0.72) strokes.push({ pts: arcPts(0, 0, rr, -0.15, Math.PI / 2 + 0.15, r, 0.02), w });
      paintRegion(ctx, r, S, p, [rimPts], fg, strokes);
    },

    // Two quarter-arc bands (a Truchet tile). Neighbours join into continuous loops.
    arcs(ctx, r, S, fg) {
      const w = S * 0.26;
      for (const [cx, cy, a0] of [[0, 0, 0], [S, S, Math.PI]]) {
        const pts = arcPts(cx, cy, S / 2, a0 - 0.12, a0 + Math.PI / 2 + 0.12, r, 0.015);
        brush(ctx, r, pts, w * 0.92, fg, { alpha: 0.8, dry: 0.05, shade: 0.1, pool: 0, rag: 0, tail: 0 });
        brush(ctx, r, pts, w, fg, { alpha: 0.75, dry: 0.4, shade: 0.16, pool: 0.4, rag: 0.02, tail: 0.1 });
      }
    },

    // Half the tile, cut on the diagonal (top-left triangle).
    half(ctx, r, S, fg) {
      const diag = [];
      const ph = r() * 6.283;
      for (let k = 0; k <= 16; k++) {
        const t = k / 16;
        const wv = Math.sin(t * 6 + ph) * S * 0.006 + (r() - 0.5) * S * 0.004;
        diag.push([S * 1.02 * (1 - t) + wv, S * 1.02 * t + wv]);
      }
      const p = new Path2D();
      p.moveTo(-2, -2);
      for (const q of diag) p.lineTo(q[0], q[1]);
      p.closePath();
      const strokes = [];
      const w = S * 0.22;
      for (let d = w * 0.4; d < S * 1.05; d += w * 0.7) strokes.push({ pts: linePts(d + S * 0.05, -S * 0.05, -S * 0.05, d + S * 0.05, 0.05, r), w });
      paintRegion(ctx, r, S, p, [diag], fg, strokes);
    },

    // A painted stem: from the incoming edge to the centre, out to each child edge, a bud where it ends.
    stem(ctx, r, S, fg, spec) {
      const mid = { N: [S / 2, -S * 0.04], S: [S / 2, S * 1.04], E: [S * 1.04, S / 2], W: [-S * 0.04, S / 2] };
      const w = S * 0.27;
      const cx = S / 2 + (r() - 0.5) * S * 0.04, cy = S / 2 + (r() - 0.5) * S * 0.04;
      const from = spec.inDir ? mid[spec.inDir] : null;
      const outs = spec.outDirs;
      const draw = (pts, ww) => {
        brush(ctx, r, pts, ww * 0.9, fg, { alpha: 0.85, dry: 0.08, shade: 0.08, pool: 0, rag: 0, tail: 0.02 });
        brush(ctx, r, pts, ww, fg, { alpha: 0.7, dry: 0.42, shade: 0.16, pool: 0.42, rag: 0.02, tail: 0.12 });
      };
      if (from && outs.length === 1) {
        const to = mid[outs[0]];
        draw(quadPts(from[0], from[1], cx, cy, to[0], to[1]), w);
      } else {
        if (from) draw(linePts(from[0], from[1], cx, cy, 0.06, r), w);
        for (const d of outs) { const to = mid[d]; draw(linePts(cx, cy, to[0], to[1], 0.06, r), w * 0.96); }
      }
      if (!outs.length || !from || outs.length > 1) {
        // a bud (leaf) or a knot (branch point / origin)
        const R = outs.length ? S * 0.17 : S * 0.2;
        const rimPts = arcPts(cx, cy, R, 0, Math.PI * 2, r, 0.05);
        const p = new Path2D();
        p.moveTo(rimPts[0][0], rimPts[0][1]);
        for (const q of rimPts) p.lineTo(q[0], q[1]);
        p.closePath();
        const strokes = [];
        for (let rr = R * 0.3; rr < R; rr += R * 0.45) strokes.push({ pts: arcPts(cx, cy, rr, r() * 3, r() * 3 + 6.6, r, 0.05), w: R * 0.5 });
        paintRegion(ctx, r, S, p, [rimPts], fg, strokes);
      }
    },

    // A loaded brush pulled across most of the tile; the navy shows only at the ragged ends.
    wash(ctx, r, S, fg) {
      const ang = (r() - 0.5) * 0.3;
      const x0 = -S * 0.1 + r() * S * 0.12, x1 = S * (0.88 + r() * 0.2);
      for (let i = 0; i < 3; i++) {
        const y = S * (0.17 + i * 0.33 + (r() - 0.5) * 0.05);
        const pts = linePts(x0 + (r() - 0.5) * S * 0.08, y + ang * S * 0.3, x1 + (r() - 0.5) * S * 0.1, y - ang * S * 0.3, 0.08, r);
        brush(ctx, r, pts, S * 0.42, fg, { alpha: 0.95, dry: 0.12, shade: 0.12, pool: 0.3, rag: 0.0, tail: 0.06 });
      }
      for (let i = 0; i < 2; i++) {
        const y = S * (0.3 + i * 0.4);
        brush(ctx, r, linePts(x0, y, x1, y + (r() - 0.5) * S * 0.1, 0.1, r), S * 0.45, mix(fg, '#ffffff', 0.25),
          { alpha: 0.35, dry: 0.9, shade: 0.1, pool: 0, rag: 0.1, tail: 0.3, soft: true });
      }
    },

    // One or two broad strokes over part of the tile.
    patch(ctx, r, S, fg) {
      const n = r() < 0.6 ? 2 : 1;
      const ang = (r() - 0.5) * 0.5;
      for (let i = 0; i < n; i++) {
        const w = S * (n === 2 ? 0.42 : 0.58);
        const y = n === 2 ? S * (0.3 + i * 0.38 + (r() - 0.5) * 0.06) : S * (0.5 + (r() - 0.5) * 0.12);
        const x0 = -S * 0.08 + r() * S * 0.1, x1 = S * (0.62 + r() * 0.42);
        const pts = linePts(x0, y + ang * S * 0.3, x1, y - ang * S * 0.3, 0.18, r);
        brush(ctx, r, pts, w, fg, { alpha: 0.9, dry: 0.7, shade: 0.2, pool: 0.4, rag: 0.04, tail: 0.25, soft: true, bristle: 0.9 * DPR });
      }
    },

    // A light, dry touch: a swipe, a curved pass or two dabs, varied per tile.
    dry(ctx, r, S, fg) {
      const kind = Math.floor(r() * 3);
      const o = { alpha: 0.75, dry: 1.0, shade: 0.18, pool: 0.3, rag: 0.12, tail: 0.4, soft: true, bristle: 0.8 * DPR };
      if (kind === 0) {
        const y = S * (0.5 + (r() - 0.5) * 0.3), tilt = (r() - 0.5) * S * 0.3;
        brush(ctx, r, linePts(S * 0.02, y + tilt, S * (0.75 + r() * 0.3), y - tilt, 0.2, r), S * (0.32 + r() * 0.14), fg, o);
      } else if (kind === 1) {
        const cx = S * (0.1 + r() * 0.2), cy = S * (1.0 + r() * 0.2), R = S * (0.6 + r() * 0.2);
        brush(ctx, r, arcPts(cx, cy, R, -Math.PI * 0.48, -Math.PI * 0.08, r, 0.03), S * (0.26 + r() * 0.1), fg, o);
      } else {
        const cx = S * (0.95 + r() * 0.2), cy = S * (-0.05 - r() * 0.1), R = S * (0.55 + r() * 0.2);
        brush(ctx, r, arcPts(cx, cy, R, Math.PI * 0.92, Math.PI * 0.55, r, 0.03), S * (0.24 + r() * 0.1), fg, o);
      }
    },
  };

  // Paint one face. spec: { bg, motif, fg, seed, ... }
  function paintFace(px, spec) {
    const S = px;
    const cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const ctx = cv.getContext('2d');
    const r = rng(spec.seed);
    const shape = tileShape(S, r);
    ctx.save();
    ctx.clip(shape.path);
    glaze(ctx, r, S, spec.bg, spec.bg === C.glaze ? { alpha: 0.42, shade: 0.32 } : { alpha: 0.6, shade: 0.4, dry: 0.4, pool: 0.3, dark: mix(spec.bg, C.ground, 0.6) });
    MOTIF[spec.motif || 'blank'](ctx, r, S, spec.fg, spec);
    poolEdges(ctx, r, shape.edges, spec.bg, S, spec.bg === C.glaze ? 1.1 : 1.35);
    speckle(ctx, r, S, spec.bg);
    ctx.restore();
    return cv;
  }


  // ---------- the scene ----------
  const STEP_SCROLL = [1 / 3, 2 / 3];

  class Scene {
    constructor(root) {
      this.root = root;
      this.cv = root.querySelector('.ts-canvas');
      this.ctx = this.cv.getContext('2d');
      this.copies = [...root.querySelectorAll('.ts-copy')];
      this.marks = [...root.querySelectorAll('.ts-steps li')];
      this.faces = new Map();
      this.chapter = -1;
      this.loopAt = Infinity;
      this.visible = true;
      this.rm = RMQ.matches;
    }

    now() { return CLOCK.t; }

    face(spec) {
      const key = this.px + '|' + JSON.stringify(spec);
      let f = this.faces.get(key);
      if (!f) { f = paintFace(this.px, spec); this.faces.set(key, f); }
      return f;
    }

    layout() {
      const w = this.cv.clientWidth, h = this.cv.clientHeight;
      if (!w || !h || (w === this.w && h === this.h)) return false;
      this.w = w; this.h = h;
      this.cv.width = Math.round(w * DPR);
      this.cv.height = Math.round(h * DPR);
      const target = w < 560 ? 44 : 62;
      this.cols = Math.max(6, 2 * Math.round(w / target / 2));
      this.pitch = w / this.cols;
      this.rows = Math.max(6, 2 * Math.floor(h / this.pitch / 2));
      this.oy = (h - this.rows * this.pitch) / 2;
      this.size = this.pitch * 0.905;
      const px = Math.ceil(this.size * DPR * 1.04);
      if (px !== this.px) { this.px = px; this.faces.clear(); }
      const r = rng(1000);
      this.tiles = [];
      for (let j = 0; j < this.rows; j++) {
        for (let i = 0; i < this.cols; i++) {
          this.tiles.push({
            i, j,
            x: (i + 0.5) * this.pitch + (r() - 0.5) * this.pitch * 0.026,
            y: this.oy + (j + 0.5) * this.pitch + (r() - 0.5) * this.pitch * 0.026,
            rot: (r() - 0.5) * 0.022,
            sc: 1 + (r() - 0.5) * 0.024,
            seed: Math.floor(r() * 1e9),
            q: [],
          });
        }
      }
      this.setupAmortized();
      this.ready = { 1: false, 2: false };
      this.gen = (this.gen || 0) + 1;
      const gen = this.gen;
      // paint the later chapters' faces in idle slices so the first one appears at once
      const idle = window.requestIdleCallback || ((f) => setTimeout(f, 16));
      idle(() => { if (gen === this.gen) this.need(1); idle(() => { if (gen === this.gen) this.need(2); }); });
      // show the current chapter's settled state at once (no replay on resize)
      const ch = this.chapter;
      this.chapter = -2;
      this.enter(ch, true);
      return true;
    }

    need(ch) {
      if (ch === 1 && !this.ready[1]) { this.setupCausal(); this.ready[1] = true; }
      if (ch === 2 && !this.ready[2]) { this.setupBayesian(); this.ready[2] = true; }
    }

    // ----- per-tile flips -----
    shown(t, now) {
      if (!t.b) return t.a;
      return (t.d <= 0 || now - t.s >= t.d * 0.5) ? t.b : t.a;
    }
    set(t, f) { t.a = f; t.b = null; t.q.length = 0; }
    clearQueues() { for (const t of this.tiles) t.q.length = 0; }
    queue(t, at, face, dur, axis, ez) {
      t.q.push({ at, face, dur, axis: axis || 'x', ez: ez || 'io' });
      if (t.q.length > 1 && t.q[t.q.length - 2].at > at) t.q.sort((x, y) => x.at - y.at);
    }

    // ----- chapter -1: plain glaze, before the wall is in view -----
    blankFace(t) { return { img: this.face({ bg: C.glaze, motif: 'blank', seed: t.seed % 997 % 14 }), q: Math.floor(t.seed / 7) % 4 }; }

    // ----- 1. Amortized -----
    setupAmortized() {
      const cols = this.cols, rows = this.rows;
      const ringCol = [C.peach, C.aqua, C.peri, C.ink];
      for (const t of this.tiles) {
        t.blank = this.blankFace(t);
        // Pattern A: concentric diamonds of circles and petals around the centre of the wall.
        const u = t.i - cols / 2, v = t.j - rows / 2;
        const bi = Math.floor((u + 1) / 2), bj = Math.floor((v + 1) / 2);
        const di = ((u + 1) % 2 + 2) % 2, dj = ((v + 1) % 2 + 2) % 2;
        const d = Math.abs(bi) + Math.abs(bj);
        const toCentre = [[2, 1], [3, 0]][di][dj];
        const petal = d % 2 === 1;
        const fg = ringCol[d % 4];
        const inv = d % 4 === 3;
        const spec = inv ? { bg: fg, motif: 'quarter', fg: C.glaze, seed: t.seed % 5 }
          : { bg: C.glaze, motif: 'quarter', fg, seed: t.seed % 5 };
        t.patA = { img: this.face(spec), q: petal ? (toCentre + 2) % 4 : toCentre };
        // Pattern B: Truchet arcs, mirrored about both axes, banded by distance from the centre.
        const au = Math.floor(Math.abs(t.i - cols / 2 + 0.5)), av = Math.floor(Math.abs(t.j - rows / 2 + 0.5));
        let o = (au + av) % 2;
        if (t.i >= cols / 2) o = 1 - o;
        if (t.j >= rows / 2) o = 1 - o;
        const band = [C.peach, C.aqua, C.peri, C.aqua][Math.floor((au + av) / 2) % 4];
        t.patB = { img: this.face({ bg: C.glaze, motif: 'arcs', fg: band, seed: t.seed % 4 }), q: o };
        // nearly simultaneous: a tiny ripple out from the centre plus jitter
        const dd = Math.hypot((t.i + 0.5 - cols / 2) / cols, (t.j + 0.5 - rows / 2) / rows);
        t.delayA = dd * 0.16 + ((t.seed % 1000) / 1000) * 0.05;
      }
    }
    flipAll(key, at, dur, ez) {
      for (const t of this.tiles) this.queue(t, at + t.delayA, t[key], dur, 'x', ez);
    }

    // ----- 2. Causal -----
    setupCausal() {
      const W = this;
      const r = rng(77 + W.cols * 31 + W.rows);
      const D = { E: [1, 0], W: [-1, 0], N: [0, -1], S: [0, 1] };
      const OPP = { E: 'W', W: 'E', N: 'S', S: 'N' };
      let best = null;
      for (let attempt = 0; attempt < 40; attempt++) {
        const used = new Set(), nodes = [], roots = [];
        const key = (i, j) => i + ',' + j;
        const nRoots = W.cols < 10 ? 2 : 3;
        for (let k = 0; k < nRoots; k++) {
          const j = Math.floor((k + 0.2 + r() * 0.6) * W.rows / nRoots);
          const i = Math.floor(r() * 2);
          if (used.has(key(i, j))) continue;
          const n = { i, j, inDir: null, outDirs: [], kids: [], parent: null, depth: 0 };
          used.add(key(i, j)); nodes.push(n); roots.push(n);
        }
        let frontier = roots.slice();
        const cap = Math.round(W.cols * W.rows * 0.5);
        while (frontier.length && nodes.length < cap) {
          const next = [];
          for (const n of frontier) {
            const opts = [];
            const straight = (n.run || 0) >= 1 && n.inDir === 'W';
            for (const d of straight ? ['N', 'S'] : ['E', 'N', 'S', 'E']) {
              const i = n.i + D[d][0], j = n.j + D[d][1];
              if (i < 0 || j < 0 || i >= W.cols || j >= W.rows || used.has(key(i, j))) continue;
              opts.push(d);
            }
            if (!opts.length && straight) {
              const i = n.i + 1, j = n.j;
              if (i < W.cols && !used.has(key(i, j))) opts.push('E');
            }
            if (!opts.length) continue;
            const want = r() < 0.3 ? 2 : (r() < 0.9 ? 1 : 0);
            const chosen = [];
            for (let c = 0; c < want && opts.length; c++) {
              const d = opts.splice(Math.floor(r() * opts.length), 1)[0];
              if (chosen.includes(d)) continue;
              const i = n.i + D[d][0], j = n.j + D[d][1];
              if (used.has(key(i, j))) continue;
              chosen.push(d);
              const m = { i, j, inDir: OPP[d], outDirs: [], kids: [], parent: n, depth: n.depth + 1 };
              m.run = (n.inDir === OPP[d]) ? (n.run || 0) + 1 : 0;
              n.outDirs.push(d); n.kids.push(m);
              used.add(key(i, j)); nodes.push(m); next.push(m);
            }
          }
          frontier = next;
        }
        if (!best || nodes.length > best.length) best = nodes;
        if (nodes.length > W.cols * W.rows * 0.42) break;
      }
      const nodes = best;
      const sub = (n) => (n._s = 1 + n.kids.reduce((s, k) => s + sub(k), 0));
      nodes.filter((n) => !n.parent).forEach(sub);
      const leaves = (n) => (n.kids.length ? n.kids.reduce((s, k) => s + leaves(k), 0) : 1);
      const total = W.cols * W.rows;
      let cands = nodes.filter((n) => n.parent && n.depth >= 2 && n._s >= total * 0.06 && n._s <= total * 0.2 && leaves(n) >= 2);
      if (!cands.length) cands = nodes.filter((n) => n.parent && n._s >= 4);
      if (!cands.length) cands = nodes.filter((n) => n.kids.length);
      cands.sort((a, b) => b._s - a._s);
      const isAnc = (a, b) => { for (let n = b; n; n = n.parent) if (n === a) return true; return false; };
      const sources = [];
      for (const c of cands) {
        if (sources.length >= 3) break;
        if (sources.some((s) => isAnc(s, c) || isAnc(c, s))) continue;
        sources.push(c);
      }
      const byKey = new Map(nodes.map((n) => [n.i + ',' + n.j, n]));
      for (const t of W.tiles) {
        const n = byKey.get(t.i + ',' + t.j);
        t.node = n || null;
        if (n) {
          n.tile = t;
          const spec = { motif: 'stem', inDir: n.inDir, outDirs: n.outDirs };
          if (!n.parent) spec.inDir = n.i === 0 ? 'W' : null;
          t.cBase = { img: W.face(Object.assign({ bg: C.glaze, fg: C.peri, seed: t.seed }, spec)), q: 0 };
          t.cAfter = { img: W.face(Object.assign({ bg: C.glaze, fg: C.peach, seed: t.seed }, spec)), q: 0 };
          t.cSrc = { img: W.face(Object.assign({ bg: C.peach, fg: C.glaze, seed: t.seed }, spec)), q: 0 };
          t.cAxis = (n.inDir === 'N' || n.inDir === 'S') ? 'y' : 'x';
        } else {
          t.cBase = t.blank;
        }
      }
      // depth below each source
      W.causal = sources.map((s) => {
        const list = [];
        const walk = (n, d) => { list.push([n.tile, d]); n.kids.forEach((c) => walk(c, d + 1)); };
        walk(s, 0);
        return list;
      });
    }
    causalHeld(k) {
      const m = new Map();
      for (const [t, d] of this.causal[k % this.causal.length] || []) m.set(t, d === 0 ? t.cSrc : t.cAfter);
      return m;
    }
    // one wave from source k starting at T; returns when the wall is back at rest
    scheduleWave(k, T) {
      const list = this.causal[k % this.causal.length] || [];
      let end = T;
      for (const [t, d] of list) {
        const at = T + (d === 0 ? 0 : 0.18 + d * 0.15);
        this.queue(t, at, d === 0 ? t.cSrc : t.cAfter, 0.48, t.cAxis, 'back');
        end = Math.max(end, at + 0.48);
      }
      const back = end + 2.6;
      for (const [t] of list) this.queue(t, back, t.cBase, 0.36, t.cAxis, 'io');
      return back + 0.36;
    }

    // ----- 3. Bayesian -----
    setupBayesian() {
      const F = {};
      for (const c of [C.aqua, C.peri, C.peach, C.ink]) {
        F[c] = {};
        for (const lv of ['full', 'wash', 'patch', 'dry']) {
          F[c][lv] = [0, 1, 2, 3].map((v) => this.face(lv === 'full'
            ? { bg: c, motif: 'blank', seed: 40 + v }
            : { bg: C.glaze, motif: lv, fg: c, seed: 50 + v + (lv === 'dry' ? 10 : lv === 'wash' ? 20 : 0) }));
        }
      }
      this.bayF = F;
      this.bayBlank = [0, 1, 2, 3, 4, 5].map((v) => this.face({ bg: C.glaze, motif: 'blank', seed: v }));
      this.bayCache = new Map();
    }
    baySample(k) {
      let s = this.bayCache.get(k);
      if (s) return s;
      const r = rng(9000 + k * 131 + this.cols);
      const cx = (this.cols - 1) / 2 + (r() - 0.5) * 0.8, cy = (this.rows - 1) / 2 + (r() - 0.5) * 0.6;
      const sx = this.cols * 0.21, sy = this.rows * 0.25;
      const F = this.bayF, pick = (a) => a[Math.floor(r() * a.length)];
      const anyFace = () => {
        if (r() < 0.3) return { img: pick(this.bayBlank), q: Math.floor(r() * 4) };
        const c = pick([C.aqua, C.aqua, C.peri, C.peri, C.peach, C.ink]);
        return { img: pick(F[c][pick(['full', 'wash', 'patch', 'dry'])]), q: Math.floor(r() * 4) };
      };
      s = this.tiles.map((t) => {
        const dx = (t.i - cx) / sx, dy = (t.j - cy) / sy;
        const f = Math.exp(-0.5 * (dx * dx + dy * dy));
        let fin;
        if (r() < Math.min(1, f * 1.25) + 0.02) {
          const v = f * (0.5 + 0.8 * r());
          const lv = v > 0.82 ? 'full' : v > 0.6 ? 'wash' : v > 0.34 ? 'patch' : 'dry';
          const c = f > 0.75 && r() < 0.45 ? C.peach : f > 0.3 ? C.aqua : C.peri;
          fin = { img: pick(F[c][lv]), q: Math.floor(r() * 4) };
        } else {
          fin = { img: pick(this.bayBlank), q: Math.floor(r() * 4) };
        }
        const lock = 1.4 + r() * 1.7 + (1 - f) * 0.5;
        const ev = [];
        let tt = r() * 0.25;
        while (tt < lock) {
          ev.push({ t: tt, f: anyFace() });
          const prog = tt / lock;
          tt += 0.11 + r() * 0.12 + prog * prog * (0.3 + r() * 0.35);
        }
        ev.push({ t: Math.max(lock, tt), f: fin, last: true });
        return ev;
      });
      this.bayCache.set(k, s);
      if (this.bayCache.size > 3) this.bayCache.delete(this.bayCache.keys().next().value);
      return s;
    }
    scheduleSettle(k, T) {
      const s = this.baySample(k);
      this.tiles.forEach((t, n) => {
        for (const e of s[n]) this.queue(t, T + e.t, e.f, e.last ? 0.42 : 0.2, n % 3 === 0 ? 'y' : 'x', e.last ? 'back' : 'io');
      });
    }

    // ----- chapter switch -----
    enter(ch, still) {
      if (ch === this.chapter) return;
      const from = this.chapter;
      this.chapter = ch;
      this.copies.forEach((el, k) => el.classList.toggle('is-on', k === Math.max(0, ch)));
      this.marks.forEach((el, k) => el.classList.toggle('is-on', k === Math.max(0, ch)));
      if (!this.tiles) return;
      this.need(ch);
      const now = this.now();
      this.clearQueues();
      this.loopAt = Infinity;
      const instant = still || this.rm;
      if (ch === -1) {
        if (instant) { for (const t of this.tiles) this.set(t, t.blank); return; }
        for (const t of this.tiles) this.queue(t, now + t.delayA * 0.6, t.blank, 0.36, 'x', 'io');
        return;
      }
      if (ch === 0) {
        this.patKey = 'patA';
        if (instant) { for (const t of this.tiles) this.set(t, t.patA); return; }
        this.flipAll('patA', now, 0.5, 'back');
        this.loopAt = now + 5.6;
        return;
      }
      if (ch === 1) {
        this.waveK = 0;
        const held = from === 2 || instant ? this.causalHeld(0) : null;
        if (instant) {
          for (const t of this.tiles) this.set(t, (held && held.get(t)) || t.cBase);
          return;
        }
        // a calm sweep across the wall into the graph, then the change runs
        for (const t of this.tiles) {
          const at = now + (t.i / this.cols) * 0.55 + (t.j / this.rows) * 0.12 + ((t.seed % 997) / 997) * 0.05;
          this.queue(t, at, (held && held.get(t)) || t.cBase, 0.42, 'x', 'io');
        }
        if (held) {
          // coming back up: arrive at the settled state, then release it and keep going
          const back = now + 3.2;
          for (const [t] of this.causal[0] || []) this.queue(t, back, t.cBase, 0.36, t.cAxis, 'io');
          this.waveK = 1;
          this.loopAt = back + 1.2;
        } else {
          const end = this.scheduleWave(0, now + 1.15);
          this.waveK = 1;
          this.loopAt = end + 0.9;
        }
        return;
      }
      if (ch === 2) {
        this.bayK = 0;
        if (instant) {
          const s = this.baySample(0);
          this.tiles.forEach((t, n) => this.set(t, s[n][s[n].length - 1].f));
          return;
        }
        this.scheduleSettle(0, now);
        this.loopAt = now + 9;
      }
    }

    loop(now) {
      if (now < this.loopAt || this.rm) return;
      if (this.chapter === 0) {
        this.patKey = this.patKey === 'patA' ? 'patB' : 'patA';
        this.flipAll(this.patKey, now, 0.5, 'back');
        this.loopAt = now + 5.6;
      } else if (this.chapter === 1) {
        const end = this.scheduleWave(this.waveK++, now);
        this.loopAt = end + 0.9;
      } else if (this.chapter === 2) {
        this.scheduleSettle(++this.bayK, now);
        this.loopAt = now + 9;
      }
    }

    // ----- drawing -----
    render(now) {
      if (!this.tiles) return;
      const ctx = this.ctx;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1;
      ctx.fillStyle = C.ground;
      ctx.fillRect(0, 0, this.cv.width, this.cv.height);
      for (const t of this.tiles) {
        if (t.q.length && t.q[0].at <= now) {
          t.q.sort((x, y) => x.at - y.at);
          while (t.q.length && t.q[0].at <= now) {
            const e = t.q.shift();
            const cur = this.shown(t, e.at);
            t.a = cur; t.b = e.face; t.s = e.at; t.d = e.dur; t.ax = e.axis; t.ez = e.ez;
          }
        }
        let a = t.a, b = t.b || t.a, p = 0;
        if (t.b) {
          const x = t.d > 0 ? (now - t.s) / t.d : 1;
          if (x >= 1) { t.a = t.b; t.b = null; a = b = t.a; p = 0; }
          else p = x <= 0 ? 0 : t.ez === 'back' ? easeBack(x) : ease(x);
        }
        this.drawTile(t, a, b, p, t.ax);
      }
      ctx.globalAlpha = 1;
    }

    drawTile(t, a, b, p, axis) {
      const ctx = this.ctx;
      const ang = clamp(p, 0, 1.2) * Math.PI;
      const c = Math.cos(ang);
      const f = c >= 0 ? a : b;
      const s = Math.max(0.015, Math.abs(c));
      const lift = 1 + 0.07 * Math.sin(Math.PI * clamp(p, 0, 1));
      const size = this.size * t.sc * lift * 1.04;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.translate(t.x, t.y);
      ctx.rotate(t.rot);
      if (axis === 'y') ctx.scale(1, s); else ctx.scale(s, 1);
      ctx.rotate(f.q * Math.PI / 2);
      ctx.globalAlpha = 0.45 + 0.55 * s;
      ctx.drawImage(f.img, -size / 2, -size / 2, size, size);
    }

    // which chapter the scroll position asks for
    fromScroll() {
      const rect = this.root.getBoundingClientRect();
      const vh = innerHeight;
      if (rect.top > vh * 0.3) return -1;
      const total = Math.max(1, rect.height - vh);
      const p = -rect.top / total;
      return p < STEP_SCROLL[0] ? 0 : p < STEP_SCROLL[1] ? 1 : 2;
    }
  }

  // ---------- run ----------
  // One clock for every scene. It does not run while the page is hidden, so nothing piles up.
  const CLOCK = { t: 0, paused: false };
  const scenes = [];

  function boot(root) {
    if (root.__tilesScene) return root.__tilesScene;
    const s = new Scene(root);
    root.__tilesScene = s;
    scenes.push(s);
    // Paint only when the scene is near the viewport.
    const near = new IntersectionObserver((ents) => {
      if (!ents.some((e) => e.isIntersecting)) return;
      near.disconnect();
      s.layout();
      s.enter(s.fromScroll(), true);
      s.render(s.now());
    }, { rootMargin: '150% 0px' });
    near.observe(root);
    const vis = new IntersectionObserver((ents) => { for (const e of ents) s.visible = e.isIntersecting; });
    vis.observe(root);
    // test hook, kept on the element
    s.pause = (on) => { CLOCK.paused = !!on; };
    return s;
  }

  function onScroll() { for (const s of scenes) if (s.tiles) s.enter(s.fromScroll()); }

  let last = 0;
  function frame(ms) {
    const dt = last ? Math.min(0.1, (ms - last) / 1000) : 0;
    last = ms;
    if (!CLOCK.paused && !document.hidden) CLOCK.t += dt;
    for (const s of scenes) {
      if (!s.tiles || !s.visible || document.hidden) continue;
      s.loop(CLOCK.t);
      s.render(CLOCK.t);
    }
    requestAnimationFrame(frame);
  }

  let rz;
  function onResize() {
    clearTimeout(rz);
    rz = setTimeout(() => { for (const s of scenes) if (s.tiles && s.layout()) s.render(s.now()); }, 150);
  }

  function start() {
    document.querySelectorAll('.tiles-scene').forEach(boot);
    if (start.done) return;
    start.done = true;
    addEventListener('scroll', onScroll, { passive: true });
    addEventListener('resize', onResize);
    RMQ.addEventListener('change', () => {
      for (const s of scenes) { s.rm = RMQ.matches; if (!s.tiles) continue; const ch = s.chapter; s.chapter = -2; s.enter(ch, true); }
    });
    requestAnimationFrame(frame);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
