/* Tile painter for the V2 site: brushed glaze, motifs, pooled edges, speckle. From V1 tiles-scene.js (c8f129b). Shared by tiles-v2.js (home) and tiles-pages.js (inner pages). Exposes window.Painter. */
window.Painter = (() => {
  'use strict';
  const DPR = Math.min(1.5, window.devicePixelRatio || 1);

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

  return { DPR, C, rng, mix, clamp, ease, easeBack, poolOf, brush, linePts, quadPts, arcPts, tileShape, glaze, poolEdges, speckle, paintRegion, MOTIF };
})();
