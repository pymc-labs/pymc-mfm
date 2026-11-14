/* Inner pages: a header band, a skirting row and (Models) wave panels, each painted once into one
   canvas. Nothing repaints per frame and nothing moves, so reduced motion needs no special case.
   tile() and the queue follow v2 (tiles-v2.js); the painter is shared (tiles-painter.js).
   window.__paint() reports the paint cost for the checks. */
(() => {
  'use strict';
  const P = window.Painter;
  if (!P) return;
  const { DPR, C, rng, mix, brush, linePts, tileShape, glaze, poolEdges, speckle, MOTIF } = P;

  // ---------- ceramic finish ----------
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
    const ctx = cv.getContext('2d', { willReadFrequently: true });
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

  // ---------- paint queue: tiles in slices of at most 10 ms between frames ----------
  const SLICE_MS = 10;
  const queue = [];
  const cost = { ms: 0, tiles: 0, start: performance.now(), wallMs: null };
  function pump() {
    const t0 = performance.now();
    while (queue.length && performance.now() - t0 < SLICE_MS) { queue.shift()(); cost.tiles++; }
    cost.ms += performance.now() - t0;
    if (queue.length) requestAnimationFrame(pump);
    else cost.wallMs = performance.now() - cost.start;
  }
  function wall(canvas, cols, rows, spec) {
    const box = canvas.parentElement;
    box.style.setProperty('--cols', cols);
    box.style.setProperty('--rows', rows);
    const S = Math.round(box.clientWidth / cols * DPR);
    canvas.width = S * cols; canvas.height = S * rows;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const wasEmpty = queue.length === 0;
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      queue.push(() => {
        const t = spec(i, j, S);
        if (t) ctx.drawImage(tile(S, t.seed ?? (i * 977 + j * 131 + 7), t.bg || C.glaze, t.draw, t.rot), i * S, j * S);
      });
    }
    if (wasEmpty) requestAnimationFrame(pump);
  }

  // ---------- section motifs ----------
  const hash = (i, j, k = 0) => { let h = (i * 374761393 + j * 668265263 + k * 2147483647) | 0; h = (h ^ (h >>> 13)) * 1274126177; return ((h ^ (h >>> 16)) >>> 0); };
  const quiet = () => ({ bg: C.glaze });

  // Docs: quarter-disc rosettes, four tiles to a circle. Periwinkle and aqua; one peach rosette.
  function rosettes(c0, peach) {
    const cols = [C.peri, C.aqua, C.peri, mix(C.peri, C.aqua, 0.5)];
    return (i, j) => {
      if (i < c0) return quiet();
      const bi = Math.floor((i - c0) / 2), bj = Math.floor(j / 2), u = (i - c0) % 2, v = j % 2;
      const rot = [[2, 1], [3, 0]][u][v];
      const fg = bi === peach ? C.peach : cols[(bi + bj) % cols.length];
      return { draw: (ctx, r, S) => MOTIF.quarter(ctx, r, S, fg), rot };
    };
  }
  // API: woven Truchet arcs; tiles join into loops. Mostly periwinkle, some aqua, one peach.
  function weave(c0, peach) {
    return (i, j) => {
      if (i < c0) return quiet();
      const h = hash(i, j), fg = (i === peach[0] && j === peach[1]) ? C.peach : h % 5 === 0 ? C.aqua : mix(C.peri, C.glaze, 0.12);
      return { draw: (ctx, r, S) => MOTIF.arcs(ctx, r, S, fg), rot: (h >>> 8) % 2 };
    };
  }
  // Blog: painted dabs, short loaded strokes in loose clusters.
  function dabs(c0, peach) {
    return (i, j) => {
      if (i < c0) return quiet();
      return {
        draw(ctx, r, S) {
          const n = 3 + Math.floor(r() * 4);
          for (let k = 0; k < n; k++) {
            const x = r() * S, y = r() * S, a = (r() - 0.5) * 2.2, L = S * (0.32 + r() * 0.3);
            const t = r();
            const col = (i === peach[0] && j === peach[1] && k < 2) ? C.peach : t < 0.5 ? C.peri : t < 0.8 ? C.aqua : mix(C.aqua, C.peri, 0.5);
            const dx = Math.cos(a) * L / 2, dy = Math.sin(a) * L / 2;
            brush(ctx, r, linePts(x - dx, y - dy, x + dx, y + dy, 0.2, r), S * 0.17, col,
              { alpha: 0.6 + r() * 0.3, dry: 0.35, shade: 0.15, pool: 0.3, rag: 0.05, tail: 0.3 });
          }
        },
      };
    };
  }
  // Models and River: woven arcs laid as a wave, aqua first, periwinkle toward the edges.
  function wave(c0, rows) {
    const pal = [C.aqua, mix(C.aqua, C.peri, 0.5), C.peri, mix(C.peri, C.glaze, 0.35)];
    return (i, j) => {
      if (i < c0) return quiet();
      const d = Math.abs(j - (rows - 1) / 2);
      const k = Math.min(3, Math.floor(d * (rows > 2 ? 1 : 0)) + (hash(i, j, 5) % 4 === 0 ? 1 : 0) + (hash(i, j, 6) % 9 === 0 ? 1 : 0));
      return { draw: (ctx, r, S) => MOTIF.arcs(ctx, r, S, pal[k]), rot: j % 2 };
    };
  }
  // Skirting row: the section motif, one tile high.
  function skirtSpec(kind) {
    return (i, j) => {
      const h = hash(i, j, 9);
      const fg = i % 11 === 5 ? C.peach : i % 2 ? C.aqua : C.peri;
      if (kind === 'quarter') return { draw: (ctx, r, S) => MOTIF.quarter(ctx, r, S, fg), rot: i % 2 ? 3 : 2 };
      if (kind === 'arcs' || kind === 'wave') return { draw: (ctx, r, S) => MOTIF.arcs(ctx, r, S, fg === C.peach ? C.peri : fg), rot: (h >>> 8) % 2 };
      return dabs(0, [-1, -1])(i, j);
    };
  }

  // ---------- page wiring ----------
  const SMALL = () => matchMedia('(max-width: 800px)').matches;
  const CARET = '<svg viewBox="0 0 256 256" aria-hidden="true"><path d="M181.66,133.66l-80,80a8,8,0,0,1-11.32-11.32L164.69,128,90.34,53.66a8,8,0,0,1,11.32-11.32l80,80A8,8,0,0,1,181.66,133.66Z"/></svg>';

  // The motif follows the section; the body classes come from each section's metadata.
  function motifFor(band) {
    if (band.dataset.motif && band.dataset.motif !== 'auto') return band.dataset.motif;
    const cl = document.body.classList;
    if (cl.contains('mfm-api') || cl.contains('mfm-ref')) return 'arcs';
    if (cl.contains('mfm-blog')) return 'dabs';
    if (cl.contains('mfm-models-page') || cl.contains('mfm-river-page')) return 'wave';
    return 'quarter';
  }
  // Eyebrow text: the section, then the sidebar group of the current page when there is one.
  function eyebrowFor(kind) {
    const cl = document.body.classList;
    if (cl.contains('mfm-ref')) return ['Python reference'];
    if (cl.contains('mfm-api')) return ['API'];
    if (cl.contains('mfm-blog')) return ['Blog'];
    if (cl.contains('mfm-river-page')) return ['Models', 'Release'];
    if (cl.contains('mfm-models-page')) return ['Models'];
    const active = document.querySelector('#quarto-sidebar .sidebar-link.active');
    const group = active && active.closest('.sidebar-section') && active.closest('.sidebar-item-section')?.querySelector('.sidebar-item-text .menu-text');
    return group ? ['Guide', group.textContent.trim()] : ['Guide'];
  }
  function prepareBand(band) {
    const kind = motifFor(band);
    band.dataset.motif = kind;
    const parts = eyebrowFor(kind);
    const eyebrow = band.querySelector('.mfm-eyebrow');
    if (eyebrow) eyebrow.innerHTML = parts.map((t) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;')).join(' ' + CARET + ' ');
    const title = band.querySelector('h1');
    if (title && title.textContent.length > 26) band.classList.add('is-long');
    return kind;
  }
  // Generated pages (the Python reference) have a first heading but no title block: build the
  // band around that heading.
  function ensureBand() {
    if (document.querySelector('.mfm-band')) return;
    const h1 = document.querySelector('#quarto-document-content h1');
    const page = document.getElementById('quarto-content');
    if (!h1 || !page) return;
    const wrap = document.createElement('div');
    wrap.className = 'mfm-band-wrap';
    wrap.innerHTML = '<div class="mfm-band" data-motif="auto"><canvas aria-hidden="true"></canvas><div class="mfm-cartouche"><p class="mfm-eyebrow"></p></div></div>';
    h1.classList.add('title');
    wrap.querySelector('.mfm-cartouche').append(h1);
    page.before(wrap);
    document.body.classList.add('mfm-band-on');
  }
  function addSkirt() {
    const footer = document.querySelector('footer.footer');
    if (!footer || document.querySelector('.mfm-skirt')) return null;
    const wrap = document.createElement('div');
    wrap.className = 'mfm-skirt-wrap';
    wrap.innerHTML = '<div class="mfm-skirt"><canvas aria-hidden="true"></canvas></div>';
    footer.before(wrap);
    return wrap.querySelector('.mfm-skirt');
  }
  function paintAll() {
    const small = SMALL();
    ensureBand();
    const band = document.querySelector('.mfm-band');
    const skirt = document.querySelector('.mfm-skirt') || addSkirt();
    let kind = 'quarter';
    if (band) {
      kind = prepareBand(band);
      // Desktop: 24 x 2, the title panel fills the left 12 columns. Phone: 12 x 1 of smaller tiles, title below.
      const cols = small ? 12 : 24, rows = small ? 1 : 2, cc = small ? 0 : 12;
      band.style.setProperty('--cc', cc);
      const spec = small ? skirtSpec(kind)
        : kind === 'quarter' ? rosettes(cc, 3)
        : kind === 'wave' ? wave(cc, rows)
        : kind === 'arcs' ? weave(cc, [cc + 8, 1])
        : dabs(cc, [cc + 6, 0]);
      wall(band.querySelector('canvas'), cols, rows, spec);
    }
    if (skirt) wall(skirt.querySelector('canvas'), small ? 14 : 32, 1, skirtSpec(kind));
    // A tile panel stands in for an image: a wave of woven arcs on its own canvas.
    document.querySelectorAll('.mfm-panel').forEach((el) => {
      const size = small ? [8, 4] : [10, 5];
      wall(el.querySelector('canvas'), size[0], size[1], wave(0, size[1]));
    });
  }
  function boot() {
    paintAll();
    // Repaint once if the layout class changes (phone <-> desktop); never per frame.
    let wasSmall = SMALL(), timer;
    addEventListener('resize', () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (SMALL() === wasSmall) return;
        wasSmall = SMALL(); queue.length = 0; paintAll();
      }, 250);
    });
  }

  window.__wallsDone = () => queue.length === 0 && cost.wallMs !== null;
  window.__paint = () => ({ cpuMs: Math.round(cost.ms), wallMs: Math.round(cost.wallMs), tiles: cost.tiles });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
