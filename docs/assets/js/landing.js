/* Landing page motion. The hero canvas draws the library's idea as one picture:
   weekly series (left) enter a network that samples a causal graph per draw (middle)
   and leave as a stacked posterior decomposition (right). The second canvas is an
   illustrative decomposition with context and query windows. Both are decoration
   over static text: with reduced motion each draws one still frame, and nothing
   animates while off screen. */
(() => {
  "use strict";

  const root = document.querySelector(".mfm");
  if (!root) return;
  root.classList.add("mfm-js");

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const COLOR = {
    aqua: [180, 231, 221],
    periwinkle: [159, 170, 226],
    peach: [246, 174, 114],
    violet: [200, 180, 231],
    blue: [74, 158, 222],
    white: [247, 247, 247],
    bg: [7, 20, 42],
  };
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  // Smooth deterministic noise in [0, 1] -------------------------------------
  const hash = (n) => {
    const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
    return s - Math.floor(s);
  };
  const noise = (x) => {
    const i = Math.floor(x);
    const f = x - i;
    const u = f * f * (3 - 2 * f);
    return hash(i) * (1 - u) + hash(i + 1) * u;
  };
  const fbm = (x, seed) =>
    0.6 * noise(x + seed) + 0.28 * noise(2.03 * x + seed * 1.7) + 0.12 * noise(4.1 * x + seed * 3.1);
  const smoothstep = (e0, e1, x) => {
    const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
  };

  // One synthetic world as functions of the week u. The inputs on the left drive the
  // components on the right, so the picture stays internally consistent.
  const spend = {
    tv: (u) => smoothstep(0.42, 0.62, fbm(u * 0.11, 3.0)),
    search: (u) => 0.25 + 0.75 * fbm(u * 0.23, 11.0),
    social: (u) => smoothstep(0.36, 0.7, fbm(u * 0.17, 23.0)),
  };
  const price = (u) => 0.5 + 0.5 * Math.sin((2 * Math.PI * u) / 26 + 0.8);
  const adstock = (f, u, decay, lags) => {
    let total = 0;
    let norm = 0;
    let weight = 1;
    for (let lag = 0; lag < lags; lag++) {
      total += weight * f(u - lag);
      norm += weight;
      weight *= decay;
    }
    return total / norm;
  };
  // Stack order, bottom to top: baseline, price, social, search, tv.
  const STACK = [
    { name: "baseline", color: COLOR.periwinkle },
    { name: "price", color: COLOR.blue },
    { name: "social", color: COLOR.violet },
    { name: "search", color: COLOR.peach },
    { name: "tv", color: COLOR.aqua },
  ];
  const components = (u) => [
    0.4 + 0.05 * Math.sin((2 * Math.PI * u) / 52 + 1.3) + 0.05 * (fbm(u * 0.05, 7.0) - 0.5),
    0.05 * (1 - price(u)),
    0.08 * adstock(spend.social, u, 0.5, 4),
    0.1 * adstock(spend.search, u, 0.3, 3),
    0.17 * adstock(spend.tv, u, 0.7, 6),
  ];
  const sales = (u) => components(u).reduce((a, b) => a + b, 0) + 0.025 * (noise(u * 3.7 + 5) - 0.5);

  // Inputs, top to bottom; they face the output bands in mirrored order.
  const INPUTS = [
    { name: "tv", color: COLOR.aqua, value: spend.tv },
    { name: "search", color: COLOR.peach, value: spend.search },
    { name: "social", color: COLOR.violet, value: spend.social },
    { name: "price", color: COLOR.blue, value: price },
    { name: "sales", color: COLOR.white, value: (u) => (sales(u) - 0.42) / 0.36 },
  ];

  // Canvas plumbing ------------------------------------------------------------
  // Returns the 2D context and calls onResize({w, h}) after layout and on every resize,
  // with the backing store matched to the device pixel ratio.
  function hiDpiCanvas(canvas, onResize) {
    const ctx = canvas.getContext("2d");
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      onResize({ w: rect.width, h: rect.height });
    };
    if ("ResizeObserver" in window) {
      new ResizeObserver(resize).observe(canvas); // also fires once after layout
    } else {
      window.addEventListener("resize", resize);
      requestAnimationFrame(resize);
    }
    return ctx;
  }

  function whenVisible(element, callback, options) {
    if (!("IntersectionObserver" in window)) {
      callback(true);
      return;
    }
    new IntersectionObserver((entries) => {
      for (const entry of entries) callback(entry.isIntersecting);
    }, options).observe(element);
  }

  // Hero -------------------------------------------------------------------------
  function hero(canvas) {
    let layout = null;
    let graphs = { previous: null, current: null, since: 0 };
    let pulses = [];
    let lastSpawn = 0;
    let running = false;
    let raf = 0;

    const ctx = hiDpiCanvas(canvas, (s) => {
      layout = heroLayout(s.w, s.h);
      graphs = { previous: null, current: sampleGraph(layout), since: performance.now() / 1000 - 1 };
      pulses = [];
      if (!running) draw(graphs.since + 1); // still frame: the graph fully faded in
    });

    function heroLayout(w, h) {
      const narrow = w < 700;
      const top = h * 0.08;
      const bottom = h * 0.92;
      const xs = (narrow ? [0.35, 0.5, 0.65] : [0.355, 0.45, 0.55, 0.645]).map((f) => f * w);
      const sizes = narrow ? [5, 6, 5] : [5, 7, 7, 5];
      const layers = sizes.map((n, li) => {
        const gap = Math.min((bottom - top) / (n - 1), narrow ? 34 : 46);
        const mid = (top + bottom) / 2;
        return Array.from({ length: n }, (_, i) => ({ x: xs[li], y: mid + (i - (n - 1) / 2) * gap, layer: li, index: i }));
      });
      const edges = [];
      for (let li = 0; li < layers.length - 1; li++) {
        for (const a of layers[li]) for (const b of layers[li + 1]) edges.push([a, b]);
      }
      return {
        w,
        h,
        top,
        bottom,
        layers,
        edges,
        inEnd: w * (narrow ? 0.27 : 0.29),
        outStart: w * (narrow ? 0.73 : 0.71),
        gapIn: layers[0][1].y - layers[0][0].y,
      };
    }

    // A "graph draw": each output reads one or two inputs through one route of hidden
    // nodes. Successive draws cross-fade, like posterior samples of the causal graph.
    function sampleGraph(L) {
      const nIn = L.layers[0].length;
      const last = L.layers.length - 1;
      const paths = [];
      for (const out of L.layers[last]) {
        const parents = new Set([Math.floor(Math.random() * nIn)]);
        if (Math.random() < 0.55) parents.add(Math.floor(Math.random() * nIn));
        for (const p of parents) {
          const path = [L.layers[0][p]];
          for (let li = 1; li < last; li++) {
            const layer = L.layers[li];
            path.push(layer[Math.floor(Math.random() * layer.length)]);
          }
          path.push(out);
          paths.push(path);
        }
      }
      const edgeKeys = new Set();
      const nodes = new Set();
      for (const path of paths) {
        for (let i = 0; i < path.length - 1; i++) edgeKeys.add(key(path[i], path[i + 1]));
        for (const n of path) nodes.add(n);
      }
      return { paths, edgeKeys, nodes };
    }

    const key = (a, b) => `${a.layer}:${a.index}-${b.layer}:${b.index}`;

    function draw(t) {
      if (!layout) return;
      const L = layout;
      ctx.clearRect(0, 0, L.w, L.h);
      const now = t * 1.5; // weeks per second
      drawInputs(L, now);
      drawNetwork(L, t);
      drawOutputs(L, now);
    }

    function drawInputs(L, now) {
      const weeks = 52;
      const step = Math.max(2, L.inEnd / 140);
      const amp = L.gapIn * 0.42;
      const first = L.layers[0];
      INPUTS.forEach((input, i) => {
        const node = first[i];
        const gradient = ctx.createLinearGradient(0, 0, L.inEnd, 0);
        gradient.addColorStop(0, rgba(input.color, 0));
        gradient.addColorStop(0.45, rgba(input.color, 0.55));
        gradient.addColorStop(1, rgba(input.color, 0.85));
        ctx.strokeStyle = gradient;
        ctx.lineWidth = input.name === "sales" ? 1.4 : 1.1;
        ctx.beginPath();
        let yEnd = node.y;
        for (let x = 0; x <= L.inEnd + 0.01; x += step) {
          const u = now - ((L.inEnd - x) / L.inEnd) * weeks;
          const v = Math.min(1.15, Math.max(-0.15, input.value(u)));
          const y = node.y + amp * (0.5 - v);
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          yEnd = y;
        }
        const span = node.x - L.inEnd;
        ctx.bezierCurveTo(L.inEnd + span * 0.5, yEnd, L.inEnd + span * 0.5, node.y, node.x, node.y);
        ctx.stroke();
      });
    }

    function drawNetwork(L, t) {
      const fade = reduceMotion ? 1 : Math.min(1, Math.max(0, (t - graphs.since) / 0.9));
      // Dense, faint weights.
      ctx.lineWidth = 0.6;
      ctx.strokeStyle = rgba(COLOR.periwinkle, 0.1);
      ctx.beginPath();
      for (const [a, b] of L.edges) {
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
      }
      ctx.stroke();
      // The sampled graph, cross-fading from the previous draw.
      const layersOf = [
        [graphs.previous, 1 - fade],
        [graphs.current, fade],
      ];
      for (const [graph, alpha] of layersOf) {
        if (!graph || alpha <= 0.01) continue;
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = rgba(COLOR.aqua, 0.62 * alpha);
        ctx.beginPath();
        for (const [a, b] of L.edges) {
          if (!graph.edgeKeys.has(key(a, b))) continue;
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
        }
        ctx.stroke();
      }
      // Pulses along the current graph's paths.
      ctx.globalCompositeOperation = "lighter";
      for (const pulse of pulses) {
        const p = (t - pulse.start) / pulse.duration;
        if (p < 0 || p > 1) continue;
        const segments = pulse.path.length - 1;
        const s = Math.min(segments - 1e-6, p * segments);
        const i = Math.floor(s);
        const f = s - i;
        const a = pulse.path[i];
        const b = pulse.path[i + 1];
        const x = a.x + (b.x - a.x) * f;
        const y = a.y + (b.y - a.y) * f;
        const alpha = Math.sin(Math.PI * p);
        ctx.fillStyle = rgba(pulse.color, 0.16 * alpha);
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = rgba(pulse.color, 0.95 * alpha);
        ctx.beginPath();
        ctx.arc(x, y, 1.7, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalCompositeOperation = "source-over";
      // Nodes.
      const last = L.layers.length - 1;
      for (const layer of L.layers) {
        for (const n of layer) {
          const active = graphs.current && graphs.current.nodes.has(n) ? fade : 0;
          const wasActive = graphs.previous && graphs.previous.nodes.has(n) ? 1 - fade : 0;
          const glow = Math.max(active, wasActive);
          const io = n.layer === 0 || n.layer === last;
          const r = io ? 4 : 3.2;
          ctx.fillStyle = rgba(COLOR.bg, 1);
          ctx.strokeStyle = rgba(glow > 0 ? COLOR.aqua : COLOR.periwinkle, 0.35 + 0.55 * glow);
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          if (glow > 0) {
            ctx.fillStyle = rgba(COLOR.aqua, 0.75 * glow);
            ctx.beginPath();
            ctx.arc(n.x, n.y, r * 0.45, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    }

    function drawOutputs(L, now) {
      const weeks = 52;
      const x0 = L.outStart;
      const width = L.w - x0;
      const step = Math.max(2, width / 140);
      // The baseline keeps its shape but not its full level, so the media bands get room.
      const BASE_OFFSET = 0.3;
      const scale = (L.bottom - L.top) / (0.92 - BASE_OFFSET);
      const columns = [];
      for (let x = x0; x <= L.w + 0.01; x += step) {
        const u = now - 3 - ((L.w - x) / width) * weeks;
        const parts = components(u);
        parts[0] -= BASE_OFFSET;
        let acc = 0;
        const bounds = [L.bottom];
        for (const part of parts) {
          acc += part;
          bounds.push(L.bottom - acc * scale);
        }
        columns.push({ x, u, bounds });
      }
      const fadeIn = ctx.createLinearGradient(x0, 0, L.w, 0);
      const baseFill = ctx.createLinearGradient(0, L.top + (L.bottom - L.top) * 0.45, 0, L.bottom);
      baseFill.addColorStop(0, rgba(COLOR.periwinkle, 0.16));
      baseFill.addColorStop(1, rgba(COLOR.periwinkle, 0));
      // Bands: each drawn as a few jittered copies, so edges read as posterior draws.
      const draws = reduceMotion ? 1 : 3;
      STACK.forEach((component, k) => {
        const copies = k === 0 ? 1 : draws;
        for (let d = 0; d < copies; d++) {
          const jitter = (u) => (copies === 1 ? 0 : (fbm(u * 0.35 + d * 9.1, 40 + k * 7) - 0.5) * 5);
          ctx.beginPath();
          columns.forEach((c, i) => {
            const y = c.bounds[k + 1] + jitter(c.u);
            if (i === 0) ctx.moveTo(c.x, y);
            else ctx.lineTo(c.x, y);
          });
          for (let i = columns.length - 1; i >= 0; i--) {
            const c = columns[i];
            const y = k === 0 ? c.bounds[0] : c.bounds[k] + jitter(c.u - 0.37) * 0.6;
            ctx.lineTo(c.x, y);
          }
          ctx.closePath();
          ctx.fillStyle = k === 0 ? baseFill : rgba(component.color, 0.26 / Math.sqrt(copies));
          ctx.fill();
        }
        // Mean edge.
        ctx.beginPath();
        columns.forEach((c, i) => {
          if (i === 0) ctx.moveTo(c.x, c.bounds[k + 1]);
          else ctx.lineTo(c.x, c.bounds[k + 1]);
        });
        ctx.strokeStyle = rgba(component.color, k === 0 ? 0.35 : 0.7);
        ctx.lineWidth = 1;
        ctx.stroke();
      });
      // Output nodes feed their bands.
      const outs = L.layers[L.layers.length - 1];
      const first = columns[0];
      outs.forEach((node, i) => {
        const k = STACK.length - 1 - i; // top node = top band
        const yMid = (first.bounds[k] + first.bounds[k + 1]) / 2;
        const span = x0 - node.x;
        ctx.beginPath();
        ctx.moveTo(node.x, node.y);
        ctx.bezierCurveTo(node.x + span * 0.55, node.y, x0 - span * 0.45, yMid, x0, yMid);
        ctx.strokeStyle = rgba(STACK[k].color, 0.45);
        ctx.lineWidth = 1;
        ctx.stroke();
      });
      // Soft fade at the right edge.
      fadeIn.addColorStop(0.85, rgba(COLOR.bg, 0));
      fadeIn.addColorStop(1, rgba(COLOR.bg, 0.9));
      ctx.fillStyle = fadeIn;
      ctx.fillRect(x0, 0, width, L.h);
    }

    function spawn(t) {
      const graph = graphs.current;
      if (!graph || !graph.paths.length) return;
      const path = graph.paths[Math.floor(Math.random() * graph.paths.length)];
      const input = INPUTS[path[0].index];
      pulses.push({ path, start: t, duration: 1.6 + Math.random() * 0.6, color: input.color });
    }

    const frame = (ms) => {
      const t = ms / 1000;
      if (!layout) {
        raf = requestAnimationFrame(frame);
        return;
      }
      if (t - graphs.since > 2.8) {
        graphs = { previous: graphs.current, current: sampleGraph(layout), since: t };
      }
      if (t - lastSpawn > 0.11) {
        spawn(t);
        lastSpawn = t;
      }
      pulses = pulses.filter((p) => t - p.start < p.duration);
      draw(t);
      raf = requestAnimationFrame(frame);
    };
    const start = () => {
      if (running || reduceMotion) return;
      running = true;
      const t = performance.now() / 1000;
      graphs.since = t - 1;
      raf = requestAnimationFrame(frame);
    };
    const stop = () => {
      running = false;
      cancelAnimationFrame(raf);
    };

    if (reduceMotion) return;
    let onScreen = false;
    whenVisible(canvas, (visible) => {
      onScreen = visible;
      if (visible && !document.hidden) start();
      else stop();
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stop();
      else if (onScreen) start();
    });
  }

  // Illustrative decomposition ------------------------------------------------------
  function decomposition(canvas) {
    const CONTEXT = 52;
    const QUERY = 13;
    const U0 = 140;
    let progress = reduceMotion ? 1 : 0;

    let size = { w: 0, h: 0 };
    const ctx = hiDpiCanvas(canvas, (s) => {
      size = s;
      draw();
    });

    function draw() {
      const w = size.w;
      const h = size.h;
      if (!w || !h) return;
      ctx.clearRect(0, 0, w, h);
      const pad = { l: 4, r: 4, t: 26, b: 22 };
      const n = CONTEXT + QUERY;
      const x = (i) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
      const plotBottom = h - pad.b;
      const scale = (plotBottom - pad.t) / 0.98;
      const y = (v) => plotBottom - v * scale;

      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, pad.l + progress * (w - pad.l - pad.r) + 1, h);
      ctx.clip();

      const rows = Array.from({ length: n }, (_, i) => {
        const parts = components(U0 + i);
        const cum = [0];
        for (const p of parts) cum.push(cum[cum.length - 1] + p);
        return cum;
      });
      const query = (i) => i >= CONTEXT;

      // Stacked mean contributions.
      STACK.forEach((component, k) => {
        ctx.beginPath();
        rows.forEach((cum, i) => (i === 0 ? ctx.moveTo(x(i), y(cum[k + 1])) : ctx.lineTo(x(i), y(cum[k + 1]))));
        for (let i = n - 1; i >= 0; i--) ctx.lineTo(x(i), y(rows[i][k]));
        ctx.closePath();
        ctx.fillStyle = rgba(component.color, k === 0 ? 0.1 : 0.42);
        ctx.fill();
        ctx.beginPath();
        rows.forEach((cum, i) => (i === 0 ? ctx.moveTo(x(i), y(cum[k + 1])) : ctx.lineTo(x(i), y(cum[k + 1]))));
        ctx.strokeStyle = rgba(component.color, 0.85);
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Interval of the total: narrow on the context, wider over the query weeks.
      const half = (i) => (query(i) ? 0.022 + 0.0042 * (i - CONTEXT + 1) : 0.022);
      ctx.beginPath();
      rows.forEach((cum, i) => {
        const top = y(cum[cum.length - 1] + half(i));
        if (i === 0) ctx.moveTo(x(i), top);
        else ctx.lineTo(x(i), top);
      });
      for (let i = n - 1; i >= 0; i--) ctx.lineTo(x(i), y(rows[i][rows[i].length - 1] - half(i)));
      ctx.closePath();
      ctx.fillStyle = rgba(COLOR.white, 0.1);
      ctx.fill();

      // Observed sales on the context weeks only.
      ctx.fillStyle = rgba(COLOR.white, 0.85);
      for (let i = 0; i < CONTEXT; i++) {
        ctx.beginPath();
        ctx.arc(x(i), y(sales(U0 + i)), 1.6, 0, Math.PI * 2);
        ctx.fill();
      }

      // Query shading.
      ctx.fillStyle = rgba(COLOR.bg, 0.28);
      ctx.fillRect(x(CONTEXT - 0.5), pad.t - 18, w - x(CONTEXT - 0.5), plotBottom - pad.t + 18);
      ctx.restore();

      // Frame: baseline, context/query divider and labels.
      ctx.strokeStyle = rgba(COLOR.white, 0.14);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, plotBottom + 0.5);
      ctx.lineTo(w - pad.r, plotBottom + 0.5);
      ctx.stroke();
      const xDiv = x(CONTEXT - 0.5);
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = rgba(COLOR.white, 0.3);
      ctx.beginPath();
      ctx.moveTo(xDiv, pad.t - 18);
      ctx.lineTo(xDiv, plotBottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = '500 10.5px "JetBrains Mono", ui-monospace, monospace';
      ctx.fillStyle = rgba(COLOR.white, 0.5);
      ctx.textBaseline = "top";
      ctx.textAlign = "right";
      ctx.fillText("CONTEXT · SALES SEEN", xDiv - 8, pad.t - 18);
      ctx.textAlign = "left";
      ctx.fillText("QUERY", xDiv + 8, pad.t - 18);
      ctx.textBaseline = "alphabetic";
      ctx.fillStyle = rgba(COLOR.white, 0.35);
      ctx.fillText("weeks", pad.l, h - 4);
    }

    if (reduceMotion) return;
    let started = false;
    whenVisible(
      canvas,
      (visible) => {
        if (!visible || started) return;
        started = true;
        const t0 = performance.now();
        const tick = (ms) => {
          const p = Math.min(1, (ms - t0) / 1600);
          progress = 1 - Math.pow(1 - p, 3);
          draw();
          if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      },
      { threshold: 0.35 },
    );
  }

  // Reveal sections, copy button --------------------------------------------------------
  function reveal() {
    const items = root.querySelectorAll(".mfm-reveal");
    if (reduceMotion || !("IntersectionObserver" in window)) {
      items.forEach((el) => el.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      },
      { threshold: 0.12 },
    );
    items.forEach((el) => observer.observe(el));
  }

  function copyButtons() {
    root.querySelectorAll("[data-copy]").forEach((button) => {
      button.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(button.dataset.copy);
          button.textContent = "Copied";
          button.setAttribute("data-copied", "");
        } catch {
          button.textContent = "Select ↑";
        }
        setTimeout(() => {
          button.textContent = "Copy";
          button.removeAttribute("data-copied");
        }, 1600);
      });
    });
  }

  // Hero background: a WebGL bundle of time-series lines rising to the right ------------
  function waves(canvas, hero) {
    const gl = canvas.getContext("webgl", { alpha: true, antialias: false, premultipliedAlpha: true });
    if (!gl) return;
    const vertex = "attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }";
    const fragment = `
      precision mediump float;
      uniform vec2 uRes;
      uniform float uTime;
      uniform vec2 uMouse;
      void main() {
        vec2 uv = gl_FragCoord.xy / uRes;
        float x = uv.x * uRes.x / uRes.y;
        float px = 1.0 / uRes.y;
        vec3 col = vec3(0.0);
        for (int i = 0; i < 16; i++) {
          float fi = float(i);
          float k = fi / 15.0;
          float spread = 0.003 + 0.03 * smoothstep(0.1, 1.0, uv.x);
          float y = 0.4 + 0.3 * uv.x + (fi - 7.5) * spread
            + 0.05 * sin(x * 0.9 + uTime * 0.10 + fi * 0.17)
            + 0.02 * sin(x * 2.3 - uTime * 0.07 + fi * 0.41)
            + 0.01 * sin(x * 5.0 + uTime * 0.16 + fi) * uv.x
            + (uMouse.y - 0.5) * 0.03 * (k - 0.5) + (uMouse.x - 0.5) * 0.01;
          float d = abs(uv.y - y);
          float core = smoothstep(1.5 * px, 0.0, d);
          float glow = smoothstep(24.0 * px, 0.0, d) * 0.08;
          vec3 c = mix(vec3(0.624, 0.667, 0.886), vec3(0.706, 0.906, 0.867), k);
          col += c * (core * 0.45 + glow);
        }
        float edge = smoothstep(0.12, 0.6, uv.x) * smoothstep(1.0, 0.85, uv.x);
        vec2 q = (uv - vec2(0.5, 0.64)) * vec2(1.0, 1.7);
        float behindTitle = 1.0 - 0.75 * smoothstep(0.45, 0.0, length(q));
        col *= edge * behindTitle * 0.42;
        gl_FragColor = vec4(col, clamp(max(col.r, max(col.g, col.b)), 0.0, 1.0));
      }`;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      return gl.getShaderParameter(shader, gl.COMPILE_STATUS) ? shader : null;
    };
    const vs = compile(gl.VERTEX_SHADER, vertex);
    const fs = compile(gl.FRAGMENT_SHADER, fragment);
    if (!vs || !fs) return;
    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return;
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(program, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const uRes = gl.getUniformLocation(program, "uRes");
    const uTime = gl.getUniformLocation(program, "uTime");
    const uMouse = gl.getUniformLocation(program, "uMouse");
    const mouse = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5 };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width)); // 1x: the lines are soft anyway
      canvas.height = Math.max(1, Math.round(rect.height));
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(uRes, canvas.width, canvas.height);
    };
    const draw = (t) => {
      mouse.x += (mouse.tx - mouse.x) * 0.04;
      mouse.y += (mouse.ty - mouse.y) * 0.04;
      gl.uniform1f(uTime, t);
      gl.uniform2f(uMouse, mouse.x, mouse.y);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    resize();
    if ("ResizeObserver" in window) new ResizeObserver(() => { resize(); draw(performance.now() / 1000); }).observe(canvas);
    draw(40);
    canvas.classList.add("is-ready");
    if (reduceMotion) return;

    hero.addEventListener("pointermove", (event) => {
      const rect = hero.getBoundingClientRect();
      mouse.tx = (event.clientX - rect.left) / rect.width;
      mouse.ty = 1 - (event.clientY - rect.top) / rect.height;
    });
    let raf = 0;
    let onScreen = false;
    const loop = (ms) => {
      draw(40 + ms / 1000);
      raf = requestAnimationFrame(loop);
    };
    const sync = () => {
      cancelAnimationFrame(raf);
      if (onScreen && !document.hidden) raf = requestAnimationFrame(loop);
    };
    whenVisible(canvas, (visible) => {
      onScreen = visible;
      sync();
    });
    document.addEventListener("visibilitychange", sync);
  }

  // Hero intro: words rise into place (GSAP SplitText); the CSS failsafe covers no-JS ----
  function intro() {
    const html = document.documentElement;
    const done = () => html.classList.remove("mfm-intro");
    if (!html.classList.contains("mfm-intro")) return;
    if (reduceMotion || !window.gsap || !window.SplitText) {
      done();
      return;
    }
    gsap.registerPlugin(SplitText);
    const title = root.querySelector(".mfm-hero__title");
    const gradient = title.querySelector(".mfm-grad");
    const split = SplitText.create(title, { type: "words", wordsClass: "mfm-word" });
    if (gradient) {
      // background-clip: text does not reach transformed children; paint each word instead.
      split.words.forEach((word) => gradient.contains(word) && word.classList.add("mfm-grad"));
      gradient.classList.remove("mfm-grad");
    }
    done();
    gsap
      .timeline({ defaults: { ease: "power3.out" } })
      .from(".mfm-kicker", { y: 10, autoAlpha: 0, duration: 0.6 })
      .from(split.words, { y: 34, autoAlpha: 0, filter: "blur(10px)", duration: 1.1, stagger: 0.07 }, "-=0.35")
      .from(".mfm-hero__lede", { y: 14, autoAlpha: 0, duration: 0.9 }, "-=0.75")
      .from(".mfm-hero__actions > *", { y: 10, autoAlpha: 0, duration: 0.7, stagger: 0.08 }, "-=0.6")
      .from(".mfm-hero__figure", { y: 28, autoAlpha: 0, duration: 1.3 }, "-=0.5");
  }

  // How it works: one canvas, three stages driven by a progress p in [0, 1] -------------
  // Wide screens with GSAP: the section pins and scroll scrubs p. Elsewhere p autoplays
  // while the canvas is on screen. The step buttons jump to a stage in either mode.
  function story(section) {
    const canvas = section.querySelector('[data-mfm="story"]');
    const items = [...section.querySelectorAll("[data-step]")];
    const readout = section.querySelector(".mfm-story__readout");
    const track = section.querySelector(".mfm-story__track");
    const WEEKS = 20;
    const STEPS = 16; // FLOW_STEPS_SAMPLE: uniform Euler steps of the sampling grid
    const LOOP = 21;
    const ROW_NODE = ["tv", "search", "social", "price", "sales"]; // INPUTS order
    const NODES = {
      demand: { x: 0.14, y: 0.22, label: "latent demand", color: COLOR.white, latent: true },
      price: { x: 0.14, y: 0.74, label: "price", color: COLOR.blue },
      tv: { x: 0.5, y: 0.2, label: "tv", color: COLOR.aqua },
      search: { x: 0.5, y: 0.47, label: "search", color: COLOR.peach },
      social: { x: 0.5, y: 0.74, label: "social", color: COLOR.violet },
      sales: { x: 0.86, y: 0.47, label: "sales", color: COLOR.white },
    };
    // Candidate edges (from, to, illustrative posterior probability, family).
    const EDGES = [
      ["demand", "tv", 0.55, "dc"],
      ["demand", "search", 0.8, "dc"],
      ["demand", "social", 0.3, "dc"],
      ["demand", "sales", 0.9, "dy"],
      ["demand", "price", 0.35, "dz"],
      ["price", "search", 0.4, "zc"],
      ["price", "tv", 0.15, "zc"],
      ["price", "sales", 0.85, "zy"],
      ["tv", "sales", 0.95, "cy"],
      ["search", "sales", 0.9, "cy"],
      ["social", "sales", 0.6, "cy"],
      ["tv", "search", 0.5, "cc"],
      ["social", "search", 0.25, "cc"],
    ];
    const clamp01 = (v) => Math.min(1, Math.max(0, v));
    const ease = (v) => v * v * (3 - 2 * v);

    let size = { w: 0, h: 0 };
    let particles = [];
    let weeks = [];
    const state = { p: 0 };
    let mode = reduceMotion ? "still" : "auto";
    let t0 = performance.now() / 1000;
    let scrollTween = null;
    let activeStep = -1;
    let lastReadout = "";

    const ctx = hiDpiCanvas(canvas, (s) => {
      size = s;
      build();
      render(performance.now() / 1000);
    });

    // Targets for the flow: per-week stacked components, then particles that land in them.
    function build() {
      weeks = Array.from({ length: WEEKS }, (_, j) => {
        const parts = components(150 + j * 1.5);
        parts[0] -= 0.3;
        const cum = [0];
        for (const part of parts) cum.push(cum[cum.length - 1] + part);
        return cum;
      });
      const n = size.w < 560 ? 420 : 900;
      particles = Array.from({ length: n }, (_, i) => {
        const r1 = hash(i * 1.13 + 0.5) + 1e-6;
        const r2 = hash(i * 2.71 + 1.3);
        const j = Math.floor(hash(i * 3.97 + 2.1) * WEEKS);
        const cum = weeks[j];
        const pick = hash(i * 5.31 + 4.4) * cum[cum.length - 1];
        let k = 0;
        while (k < cum.length - 2 && pick > cum[k + 1]) k++;
        return {
          g1: Math.sqrt(-2 * Math.log(r1)) * Math.cos(2 * Math.PI * r2),
          g2: Math.sqrt(-2 * Math.log(r1)) * Math.sin(2 * Math.PI * r2),
          j,
          jx: hash(i * 7.07 + 3.3) - 0.5,
          k,
          f: hash(i * 9.13 + 6.6),
        };
      });
    }

    const at = (node) => ({ x: node.x * size.w, y: node.y * size.h });

    function drawEncoder(a, morph) {
      const left = size.w * 0.17;
      const right = size.w * 0.96;
      const top = size.h * 0.2;
      const bottom = size.h * 0.82;
      const colW = (right - left) / WEEKS;
      const rowH = (bottom - top) / (INPUTS.length - 1);
      const cell = Math.min(colW * 0.62, rowH * 0.42, 18);
      const cx = (j) => left + (j + 0.5) * colW;
      const cy = (r) => top + r * rowH;
      const q = Math.min(WEEKS - 1, Math.floor(2 + a * (WEEKS - 2.001)));
      const fade = clamp01(1 - morph * 1.8);

      if (fade > 0.01) {
        ctx.font = '500 10px "JetBrains Mono", ui-monospace, monospace';
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillStyle = rgba(COLOR.white, 0.45 * fade);
        INPUTS.forEach((row, r) => ctx.fillText(row.name, left - 10, cy(r)));
        ctx.textAlign = "left";
        ctx.fillText("weeks →", left, bottom + rowH * 0.55);
        // The current week: its column, attention to past weeks (causal), across entities.
        ctx.strokeStyle = rgba(COLOR.aqua, 0.45 * fade);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect?.(cx(q) - cell, top - cell, cell * 2, bottom - top + cell * 2, 8);
        ctx.stroke();
        INPUTS.forEach((row, r) => {
          const y = cy(r) - cell * 0.6;
          for (let k = 0; k < q; k++) {
            const weight = Math.exp(-(q - k) / 4) * (0.45 + 0.55 * hash(r * 31 + k * 7 + q));
            const x0 = cx(k);
            const x1 = cx(q);
            ctx.strokeStyle = rgba(row.color, 0.8 * weight * fade);
            ctx.beginPath();
            ctx.moveTo(x1, y);
            ctx.quadraticCurveTo((x0 + x1) / 2, y - Math.min(rowH * 0.85, (x1 - x0) * 0.32), x0, y);
            ctx.stroke();
          }
        });
        ctx.strokeStyle = rgba(COLOR.white, 0.3 * fade);
        for (let r = 0; r < INPUTS.length - 1; r++) {
          const x = cx(q) + cell * 0.7;
          ctx.beginPath();
          ctx.moveTo(x, cy(r));
          ctx.quadraticCurveTo(x + rowH * 0.35, (cy(r) + cy(r + 1)) / 2, x, cy(r + 1));
          ctx.stroke();
        }
      }
      // Tokens. Future weeks are masked (dim); on the way to stage 2 each row gathers into
      // its node of the graph.
      const m = ease(morph);
      INPUTS.forEach((row, r) => {
        const node = at(NODES[ROW_NODE[r]]);
        for (let j = 0; j < WEEKS; j++) {
          const v = clamp01(row.value(150 + j * 1.5));
          const x = cx(j) + (node.x - cx(j)) * m;
          const y = cy(r) + (node.y - cy(r)) * m;
          const s = cell * (1 - 0.55 * m);
          const alpha = (j > q ? 0.14 : 0.22 + 0.68 * v) * (1 - 0.8 * m);
          ctx.fillStyle = rgba(row.color, alpha);
          ctx.beginPath();
          if (ctx.roundRect) ctx.roundRect(x - s / 2, y - s / 2, s, s, s * 0.28);
          else ctx.rect(x - s / 2, y - s / 2, s, s);
          ctx.fill();
        }
      });
      return `causal attention · week ${q + 1} / ${WEEKS}`;
    }

    function drawGraph(alpha, t, scale = 1, ox = 0, oy = 0, fixedDraw = null) {
      const draw = fixedDraw ?? Math.floor(t / 0.7);
      let on = 0;
      const R = 15;
      ctx.save();
      ctx.translate(ox, oy);
      ctx.scale(scale, scale);
      EDGES.forEach(([from, to, prob], e) => {
        const A = at(NODES[from]);
        const B = at(NODES[to]);
        const dx = B.x - A.x;
        const dy = B.y - A.y;
        const len = Math.hypot(dx, dy);
        const ux = dx / len;
        const uy = dy / len;
        const x0 = A.x + ux * R;
        const y0 = A.y + uy * R;
        const x1 = B.x - ux * (R + 3);
        const y1 = B.y - uy * (R + 3);
        ctx.lineWidth = 1;
        ctx.strokeStyle = rgba(COLOR.periwinkle, (0.06 + 0.16 * prob) * alpha);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        if (hash(draw * 13.7 + e * 3.1) >= prob) return;
        on++;
        ctx.strokeStyle = rgba(COLOR.aqua, 0.85 * alpha);
        ctx.fillStyle = rgba(COLOR.aqua, 0.85 * alpha);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x1 + ux * 3, y1 + uy * 3);
        ctx.lineTo(x1 - ux * 6 - uy * 4, y1 - uy * 6 + ux * 4);
        ctx.lineTo(x1 - ux * 6 + uy * 4, y1 - uy * 6 - ux * 4);
        ctx.closePath();
        ctx.fill();
      });
      ctx.font = '500 10px "JetBrains Mono", ui-monospace, monospace';
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (const node of Object.values(NODES)) {
        const { x, y } = at(node);
        ctx.fillStyle = rgba(COLOR.bg, alpha);
        ctx.strokeStyle = rgba(node.color, 0.8 * alpha);
        ctx.lineWidth = 1.3;
        ctx.setLineDash(node.latent ? [3, 3] : []);
        ctx.beginPath();
        ctx.arc(x, y, R, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = rgba(node.color, 0.7 * alpha);
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fill();
        if (scale > 0.6) {
          ctx.fillStyle = rgba(COLOR.white, 0.55 * alpha);
          ctx.fillText(node.label, x, y + R + 7);
        }
      }
      ctx.restore();
      return { draw, on };
    }

    function drawFlow(c) {
      const left = size.w * 0.42;
      const right = size.w * 0.97;
      const top = size.h * 0.16;
      const bottom = size.h * 0.88;
      const scale = (bottom - top) / 0.62;
      const src = { x: size.w * 0.17, y: size.h * 0.64, sd: size.h * 0.1 };
      const transport = clamp01((c - 0.1) / 0.7);
      const s = transport * STEPS;
      const step = Math.min(STEPS, Math.floor(s));
      // Fixed-grid Euler: each step eases in, so the motion reads as discrete updates.
      const tau = transport >= 1 ? 1 : (step + ease(s - step)) / STEPS;
      const appear = clamp01(c / 0.1);
      const xOf = (j) => left + (j / (WEEKS - 1)) * (right - left);
      STACK.forEach((component, k) => {
        const tint = component.color.map((v, i) => COLOR.white[i] + (v - COLOR.white[i]) * tau);
        ctx.strokeStyle = rgba(tint, 0.18 * appear);
        ctx.fillStyle = rgba(tint, (0.35 + 0.5 * tau) * appear);
        ctx.lineWidth = 1;
        const trails = new Path2D();
        const dots = new Path2D();
        for (const p of particles) {
          if (p.k !== k) continue;
          const cum = weeks[p.j];
          const tx = xOf(p.j + p.jx * 0.8);
          const ty = bottom - (cum[k] + p.f * (cum[k + 1] - cum[k])) * scale;
          const sx = src.x + p.g1 * src.sd * 1.3;
          const sy = src.y + p.g2 * src.sd;
          const x = sx + (tx - sx) * tau;
          const y = sy + (ty - sy) * tau;
          const back = Math.max(0, tau - 0.06);
          trails.moveTo(sx + (tx - sx) * back, sy + (ty - sy) * back);
          trails.lineTo(x, y);
          dots.rect(x - 1, y - 1, 2, 2);
        }
        if (tau > 0 && tau < 1) ctx.stroke(trails);
        ctx.fill(dots);
      });
      // Once landed: the bands the draws filled, and their mean edges.
      const settle = clamp01((c - 0.82) / 0.12);
      if (settle > 0) {
        STACK.forEach((component, k) => {
          const edge = (j, level) => bottom - weeks[j][level] * scale;
          ctx.beginPath();
          weeks.forEach((_, j) => (j ? ctx.lineTo(xOf(j), edge(j, k + 1)) : ctx.moveTo(xOf(j), edge(j, k + 1))));
          for (let j = WEEKS - 1; j >= 0; j--) ctx.lineTo(xOf(j), edge(j, k));
          ctx.closePath();
          ctx.fillStyle = rgba(component.color, (k === 0 ? 0.06 : 0.16) * settle);
          ctx.fill();
          ctx.beginPath();
          weeks.forEach((_, j) => (j ? ctx.lineTo(xOf(j), edge(j, k + 1)) : ctx.moveTo(xOf(j), edge(j, k + 1))));
          ctx.strokeStyle = rgba(component.color, 0.8 * settle);
          ctx.lineWidth = 1.2;
          ctx.stroke();
        });
      }
      ctx.font = '500 10px "JetBrains Mono", ui-monospace, monospace';
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = rgba(COLOR.white, 0.45 * appear * (1 - tau));
      ctx.fillText("noise", src.x, src.y + src.sd * 2.6);
      return transport >= 1 ? "decomposition · posterior draws" : `rectified flow · euler step ${Math.max(1, Math.ceil(s))} / ${STEPS}`;
    }

    function render(t) {
      if (!size.w) return;
      if (mode === "auto") state.p = (((t - t0) % LOOP) + LOOP) % LOOP / LOOP;
      const p = state.p;
      ctx.clearRect(0, 0, size.w, size.h);
      let text;
      let stepIndex;
      if (p < 1 / 3) {
        stepIndex = 0;
        text = drawEncoder(clamp01(p * 3 / 0.85), 0);
      } else if (p < 2 / 3) {
        stepIndex = 1;
        const b = p * 3 - 1;
        if (b < 0.35) drawEncoder(1, clamp01(b / 0.3));
        const g = drawGraph(ease(clamp01((b - 0.15) / 0.3)), t);
        text = b < 0.2 ? "tokens → entities" : `graph draw ${(g.draw % 1000) + 1} · ${g.on} of ${EDGES.length} edges`;
      } else {
        stepIndex = 2;
        const c = p * 3 - 2;
        const shrink = ease(clamp01(c / 0.15));
        const k = 1 - 0.66 * shrink;
        // The decoder conditions on one sampled graph: hold it fixed while the flow runs.
        drawGraph(1 - 0.4 * shrink, t, k, size.w * 0.015 * shrink, size.h * 0.01 * shrink, 16);
        if (shrink > 0.95) {
          ctx.font = '500 10px "JetBrains Mono", ui-monospace, monospace';
          ctx.textAlign = "left";
          ctx.textBaseline = "top";
          ctx.fillStyle = rgba(COLOR.white, 0.4);
          ctx.fillText("conditioned on this graph", size.w * 0.015 + size.w * 0.1, size.h * 0.01 + size.h * 0.3);
        }
        text = drawFlow(c);
      }
      if (stepIndex !== activeStep) {
        activeStep = stepIndex;
        items.forEach((li, i) => li.classList.toggle("is-active", i === stepIndex));
      }
      if (text !== lastReadout) {
        lastReadout = text;
        readout.textContent = text;
      }
      track.style.setProperty("--p", p.toFixed(4));
    }

    if (mode === "still") state.p = 0.16;

    let raf = 0;
    let onScreen = false;
    const loop = (ms) => {
      render(ms / 1000);
      raf = requestAnimationFrame(loop);
    };
    const sync = () => {
      cancelAnimationFrame(raf);
      if (mode !== "still" && onScreen && !document.hidden) raf = requestAnimationFrame(loop);
    };
    whenVisible(section, (visible) => {
      onScreen = visible;
      sync();
    });
    document.addEventListener("visibilitychange", sync);

    if (!reduceMotion && window.gsap && window.ScrollTrigger) {
      gsap.registerPlugin(ScrollTrigger);
      gsap.matchMedia().add("(min-width: 961px) and (min-height: 640px)", () => {
        mode = "scroll";
        state.p = 0;
        section.classList.add("is-pinned");
        scrollTween = gsap.to(state, {
          p: 1,
          ease: "none",
          scrollTrigger: {
            trigger: section,
            start: "top top",
            end: () => `+=${Math.round(window.innerHeight * 2.4)}`,
            pin: true,
            scrub: 0.6,
            invalidateOnRefresh: true,
          },
        });
        return () => {
          mode = "auto";
          scrollTween = null;
          section.classList.remove("is-pinned");
          t0 = performance.now() / 1000;
        };
      });
    }

    const CENTERS = [0.17, 0.53, 0.9];
    items.forEach((li, i) => {
      li.querySelector("button").addEventListener("click", () => {
        if (mode === "scroll" && scrollTween) {
          const st = scrollTween.scrollTrigger;
          window.scrollTo({ top: st.start + CENTERS[i] * (st.end - st.start), behavior: "smooth" });
        } else if (mode === "still") {
          state.p = CENTERS[i];
          render(performance.now() / 1000);
        } else {
          t0 = performance.now() / 1000 - CENTERS[i] * LOOP;
        }
      });
    });
  }

  // A soft light under the pointer on the cards -----------------------------------------
  function spotlight() {
    root.querySelectorAll(".mfm-window, .mfm-plot, .mfm-story__figure").forEach((el) => {
      el.classList.add("mfm-spot");
      el.addEventListener("pointermove", (event) => {
        const rect = el.getBoundingClientRect();
        el.style.setProperty("--mx", `${event.clientX - rect.left}px`);
        el.style.setProperty("--my", `${event.clientY - rect.top}px`);
      });
    });
  }

  const heroSection = root.querySelector(".mfm-hero");
  const wavesCanvas = root.querySelector('[data-mfm="waves"]');
  const heroCanvas = root.querySelector('[data-mfm="hero"]');
  const plotCanvas = root.querySelector('[data-mfm="decomposition"]');
  const storySection = root.querySelector(".mfm-story");
  intro();
  if (wavesCanvas && heroSection) waves(wavesCanvas, heroSection);
  // Canvas text uses the web fonts; draw once they are ready.
  const startCanvases = () => {
    if (heroCanvas) hero(heroCanvas);
    if (plotCanvas) decomposition(plotCanvas);
    if (storySection) story(storySection);
  };
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(startCanvases);
  else startCanvases();
  reveal();
  copyButtons();
  spotlight();
})();
