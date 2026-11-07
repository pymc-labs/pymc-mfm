/* Landing page motion, one scene per section:
   1. Hero: a WebGL fog with a river of light behind the name; the name assembles, then
      dissolves as you scroll away.
   2. Statement: its words light up as they pass.
   3. The editor: a notebook runs its cells when it comes into view ("Run All" replays).
   4. Chapters: the section pins and three words cross-fade in one place.
   Everything is decoration over static HTML. Without GSAP the page keeps its plain
   layout; with reduced motion nothing moves and the notebook shows finished. */
(() => {
  "use strict";

  const root = document.querySelector(".mfm");
  if (!root) return;
  root.classList.add("mfm-js");

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hasGsap = () => Boolean(window.gsap && window.ScrollTrigger);
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  function whenVisible(element, callback, options) {
    if (!("IntersectionObserver" in window)) {
      callback(true);
      return;
    }
    new IntersectionObserver((entries) => {
      for (const entry of entries) callback(entry.isIntersecting);
    }, options).observe(element);
  }

  // 1. Hero fog --------------------------------------------------------------------
  function fog(canvas, hero) {
    const gl = canvas.getContext("webgl", { alpha: false, antialias: false });
    if (!gl) return;
    const vertex = "attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }";
    const fragment = `
      precision highp float;
      uniform vec2 uRes;
      uniform float uTime;
      uniform vec2 uMouse;
      uniform float uFade;
      uniform float uScroll;
      float hash(vec2 p) {
        p = fract(p * vec2(123.34, 456.21));
        p += dot(p, p + 45.32);
        return fract(p.x * p.y);
      }
      float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
                   mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
      }
      float fbm(vec2 p) {
        float v = 0.0;
        float a = 0.5;
        mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
        for (int i = 0; i < 5; i++) { v += a * noise(p); p = m * p; a *= 0.5; }
        return v;
      }
      void main() {
        vec2 p = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
        p += (uMouse - 0.5) * vec2(0.05, 0.03);
        float t = uTime * 0.035;
        vec2 q = vec2(fbm(p * 1.3 + vec2(t, -t * 0.5)), fbm(p * 1.3 + vec2(5.2, 1.3) - t * 0.7));
        vec2 r = vec2(fbm(p * 1.7 + 2.8 * q + vec2(1.7, 9.2) + t),
                      fbm(p * 1.7 + 2.8 * q + vec2(8.3, 2.8) - t));
        float f = fbm(p * 1.1 + 2.2 * r + vec2(t * 2.0, 0.0));
        // A meandering river of light across the lower half.
        float y = -0.26 + 0.07 * sin(p.x * 1.4 + uTime * 0.06)
                + 0.035 * sin(p.x * 3.3 - uTime * 0.045) + (f - 0.5) * 0.24 + uScroll * 0.12;
        float d = p.y - y;
        float river = exp(-d * d / 0.008);
        float haze = exp(-d * d / 0.08);
        vec3 navy = vec3(0.027, 0.078, 0.165);
        vec3 deep = vec3(0.06, 0.14, 0.28);
        vec3 aqua = vec3(0.706, 0.906, 0.867);
        vec3 peri = vec3(0.624, 0.667, 0.886);
        vec3 peach = vec3(0.965, 0.682, 0.447);
        vec3 col = mix(navy, deep, smoothstep(0.25, 0.85, f));
        col += peri * 0.14 * smoothstep(0.55, 0.95, f);
        col += mix(peri, aqua, smoothstep(-0.7, 0.7, p.x)) * (0.2 * haze + 0.42 * river) * (0.5 + 0.7 * f);
        col += peach * river * smoothstep(0.6, 0.9, r.y) * 0.3;
        float vig = smoothstep(1.4, 0.2, length(p * vec2(0.85, 1.15)));
        col *= mix(0.4, 1.0, vig);
        col *= 1.0 - 0.6 * uScroll;
        col = mix(navy * 0.7, col, uFade);
        gl_FragColor = vec4(col, 1.0);
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
    const u = (name) => gl.getUniformLocation(program, name);
    const uRes = u("uRes");
    const uTime = u("uTime");
    const uMouse = u("uMouse");
    const uFade = u("uFade");
    const uScroll = u("uScroll");

    // Fog is soft: render at half resolution and let the browser scale it up.
    const SCALE = 0.5;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * SCALE));
      canvas.height = Math.max(1, Math.round(rect.height * SCALE));
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(uRes, canvas.width, canvas.height);
    };
    const mouse = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5 };
    let start = performance.now();
    const draw = (ms, still = false) => {
      mouse.x += (mouse.tx - mouse.x) * 0.03;
      mouse.y += (mouse.ty - mouse.y) * 0.03;
      const scroll = clamp01(window.scrollY / Math.max(1, hero.offsetHeight));
      gl.uniform1f(uTime, 30 + ms / 1000);
      gl.uniform2f(uMouse, mouse.x, mouse.y);
      gl.uniform1f(uFade, still ? 1 : clamp01((ms - start) / 2600));
      gl.uniform1f(uScroll, scroll);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    resize();
    canvas.classList.add("is-ready");
    if ("ResizeObserver" in window) {
      new ResizeObserver(() => {
        resize();
        draw(performance.now(), reduceMotion);
      }).observe(canvas);
    }
    if (reduceMotion) {
      draw(0, true);
      return;
    }
    hero.addEventListener("pointermove", (event) => {
      const rect = hero.getBoundingClientRect();
      mouse.tx = (event.clientX - rect.left) / rect.width;
      mouse.ty = 1 - (event.clientY - rect.top) / rect.height;
    });
    start = performance.now();
    let raf = 0;
    let onScreen = false;
    const loop = (ms) => {
      draw(ms);
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

  // Hero copy: the name assembles letter by letter, then dissolves on scroll ----------
  function heroCopy(hero) {
    const html = document.documentElement;
    const done = () => html.classList.remove("mfm-intro");
    if (reduceMotion || !hasGsap() || !window.SplitText) {
      done();
      return;
    }
    gsap.registerPlugin(ScrollTrigger, SplitText);
    const title = hero.querySelector(".mfm-hero__title");
    const split = SplitText.create(title, { type: "chars", charsClass: "mfm-char" });
    title.classList.add("is-split");
    done();
    gsap
      .timeline({ defaults: { ease: "power3.out" } })
      .from(".mfm-hero__kicker", { autoAlpha: 0, y: 8, duration: 1.1 }, 0.4)
      .from(split.chars, { autoAlpha: 0, y: 46, filter: "blur(16px)", duration: 1.5, stagger: 0.07 }, 0.55)
      .from(".mfm-hero__line", { autoAlpha: 0, y: 12, filter: "blur(8px)", duration: 1.2 }, 1.25)
      .from(".mfm-hero__actions > *", { autoAlpha: 0, y: 10, duration: 0.9, stagger: 0.1 }, 1.6)
      .from(".mfm-hero__scroll", { autoAlpha: 0, duration: 1.2 }, 2.1);
    gsap.to(".mfm-hero__copy", {
      yPercent: -14,
      autoAlpha: 0,
      filter: "blur(10px)",
      ease: "none",
      scrollTrigger: { trigger: hero, start: "top top", end: "bottom 25%", scrub: true },
    });
    gsap.to(".mfm-hero__scroll", {
      autoAlpha: 0,
      ease: "none",
      scrollTrigger: { trigger: hero, start: "top top", end: "15% top", scrub: true },
    });
  }

  // 2. Statement: words light up as they pass -------------------------------------------
  function statement(element) {
    if (reduceMotion || !hasGsap() || !window.SplitText) return;
    const split = SplitText.create(element, { type: "words", wordsClass: "mfm-word" });
    gsap.fromTo(
      split.words,
      { opacity: 0.14 },
      {
        opacity: 1,
        ease: "none",
        stagger: 0.1,
        scrollTrigger: { trigger: element, start: "top 82%", end: "bottom 48%", scrub: 0.6 },
      },
    );
  }

  // 3. The notebook ----------------------------------------------------------------
  // Illustrative mean contributions over the 13 query weeks (Sep 29 to Dec 22, 2025).
  const AREA_SERIES = [
    { name: "baseline", color: "#9faae2", values: [30.2, 30.6, 30.9, 31.1, 31.6, 32.0, 32.3, 33.1, 34.0, 35.2, 36.4, 37.1, 36.2] },
    { name: "channel:tv", color: "#b4e7dd", values: [0.4, 0.2, 5.8, 10.6, 12.1, 9.4, 5.6, 2.9, 1.2, 7.9, 12.8, 13.4, 8.7] },
    { name: "channel:search", color: "#f6ae72", values: [4.1, 4.3, 4.6, 5.0, 5.2, 4.9, 4.7, 4.8, 5.3, 5.9, 6.4, 6.6, 6.0] },
    { name: "channel:social", color: "#c8b4e7", values: [2.0, 2.2, 2.1, 2.5, 2.9, 2.7, 2.4, 2.3, 2.6, 3.0, 3.3, 3.1, 2.8] },
  ];

  function drawArea(svg) {
    const NS = "http://www.w3.org/2000/svg";
    const el = (tag, attrs, parent = svg) => {
      const node = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
      parent.appendChild(node);
      return node;
    };
    const L = 62;
    const R = 620;
    const T = 14;
    const B = 252;
    const n = AREA_SERIES[0].values.length;
    const yMax = 70; // thousands
    const x = (i) => L + (i / (n - 1)) * (R - L);
    const y = (v) => B - (v / yMax) * (B - T);
    const text = { fill: "#bdbdbd", "font-size": "11", "font-family": "JetBrains Mono, monospace" };

    for (let v = 0; v <= yMax; v += 10) {
      el("line", { x1: L - 4, x2: L, y1: y(v), y2: y(v), stroke: "#8a8a8a" });
      el("text", { ...text, x: L - 8, y: y(v) + 4, "text-anchor": "end" }).textContent = v ? `${v}000` : "0";
    }
    [
      [1, "Oct\n2025"],
      [5, "Nov"],
      [10, "Dec"],
    ].forEach(([i, label]) => {
      el("line", { x1: x(i), x2: x(i), y1: B, y2: B + 4, stroke: "#8a8a8a" });
      label.split("\n").forEach((part, j) => {
        el("text", { ...text, x: x(i), y: B + 17 + j * 13, "text-anchor": "middle" }).textContent = part;
      });
    });
    el("text", { ...text, x: (L + R) / 2, y: 296, "text-anchor": "middle" }).textContent = "time";

    const plot = el("g", { class: "mfm-area__plot" });
    let lower = new Array(n).fill(0);
    for (const series of AREA_SERIES) {
      const upper = lower.map((v, i) => v + series.values[i]);
      const top = upper.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
      const bottom = lower.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).reverse();
      el("polygon", { points: [...top, ...bottom].join(" "), fill: series.color, "fill-opacity": "0.62" }, plot);
      el("polyline", { points: top.join(" "), fill: "none", stroke: series.color, "stroke-width": "1.4" }, plot);
      lower = upper;
    }
    el("rect", { x: L, y: T, width: R - L, height: B - T, fill: "none", stroke: "#8a8a8a" });

    const legend = el("g", { transform: `translate(${L + 12}, ${T + 10})` });
    el("rect", { x: 0, y: 0, width: 132, height: 76, rx: 3, fill: "#1f1f1f", "fill-opacity": "0.85", stroke: "#5a5a5a" }, legend);
    AREA_SERIES.forEach((series, i) => {
      el("rect", { x: 8, y: 9 + i * 16, width: 18, height: 9, fill: series.color, "fill-opacity": "0.75" }, legend);
      el("text", { ...text, x: 32, y: 17 + i * 16, fill: "#d4d4d4" }, legend).textContent = series.name;
    });
  }

  function notebook(ide) {
    const cells = [...ide.querySelectorAll(".mfm-cell")];
    const scroller = ide.querySelector('[data-mfm="cells"]');
    const runAll = ide.querySelector('[data-mfm="run-all"]');
    const cellOf = ide.querySelector('[data-mfm="cell-of"]');
    const area = ide.querySelector('[data-mfm="area"]');
    if (area) drawArea(area);
    if (reduceMotion) return;

    let runId = 0;
    const reset = () => {
      ide.classList.add("is-playing");
      cells.forEach((cell) => cell.classList.remove("is-done", "is-running", "is-focus"));
      scroller.scrollTop = 0;
      cellOf.textContent = `Cell 1 of ${cells.length}`;
    };
    const scrollToCell = (cell, offset = 14) =>
      scroller.scrollTo({ top: Math.max(0, cell.offsetTop - offset), behavior: "smooth" });

    async function play() {
      const id = ++runId;
      const alive = () => id === runId;
      reset();
      await wait(450);
      if (!alive()) return;
      runAll.classList.add("is-pressed");
      await wait(280);
      runAll.classList.remove("is-pressed");
      for (const [i, cell] of cells.entries()) {
        if (!alive()) return;
        cellOf.textContent = `Cell ${i + 1} of ${cells.length}`;
        cell.classList.add("is-running", "is-focus");
        scrollToCell(cell);
        await wait(Number(cell.dataset.duration) || 900);
        if (!alive()) return;
        cell.classList.remove("is-running");
        cell.classList.add("is-done");
        await wait(700);
        cell.classList.remove("is-focus");
      }
      const last = cells[cells.length - 1];
      scroller.scrollTo({
        top: last.offsetTop + last.offsetHeight - scroller.clientHeight + 48,
        behavior: "smooth",
      });
    }

    reset();
    runAll.addEventListener("click", play);
    let started = false;
    whenVisible(
      ide,
      (visible) => {
        if (!visible || started) return;
        started = true;
        play();
      },
      { threshold: 0.4 },
    );
  }

  // 4. Chapters: pin, then cross-fade three words in one place ---------------------------
  function chapters(section) {
    if (reduceMotion || !hasGsap()) return;
    gsap.registerPlugin(ScrollTrigger);
    const items = [...section.querySelectorAll(".mfm-chapter")];
    gsap.matchMedia().add("(min-height: 520px)", () => {
      section.classList.add("is-cinematic");
      gsap.set(items.slice(1), { autoAlpha: 0 });
      const timeline = gsap.timeline({
        defaults: { ease: "power2.out" },
        scrollTrigger: {
          trigger: section,
          start: "top top",
          end: () => `+=${Math.round(window.innerHeight * 2.4)}`,
          pin: true,
          scrub: 0.8,
          invalidateOnRefresh: true,
        },
      });
      items.forEach((item, i) => {
        const parts = item.querySelectorAll(".mfm-chapter__index, .mfm-chapter__word, .mfm-chapter__line");
        if (i > 0) {
          timeline.fromTo(item, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.25 }, i);
          timeline.fromTo(
            parts,
            { y: 50, filter: "blur(12px)" },
            { y: 0, filter: "blur(0px)", duration: 0.45, stagger: 0.06 },
            i,
          );
        }
        if (i < items.length - 1) {
          timeline.to(parts, { y: -50, filter: "blur(12px)", duration: 0.35, stagger: 0.04, ease: "power2.in" }, i + 0.62);
          timeline.to(item, { autoAlpha: 0, duration: 0.3, ease: "power2.in" }, i + 0.7);
        }
      });
      timeline.to({}, { duration: 0.35 }, items.length - 0.35);
      return () => section.classList.remove("is-cinematic");
    });
  }

  // Reveal sections, copy button -------------------------------------------------------
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

  const hero = root.querySelector(".mfm-hero");
  const fogCanvas = root.querySelector('[data-mfm="fog"]');
  const statementText = root.querySelector('[data-mfm="statement"]');
  const ide = root.querySelector('[data-mfm="notebook"]');
  const chapterSection = root.querySelector('[data-mfm="chapters"]');

  if (fogCanvas && hero) fog(fogCanvas, hero);
  if (ide) notebook(ide);
  reveal();
  copyButtons();
  // Text splits measure glyphs: wait for the web fonts.
  const ready = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
  ready.then(() => {
    if (hero) heroCopy(hero);
    if (statementText) statement(statementText);
    if (chapterSection) chapters(chapterSection);
    if (hasGsap()) ScrollTrigger.refresh();
  });
})();
