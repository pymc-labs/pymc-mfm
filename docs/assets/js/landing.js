/* Landing page motion, one scene per section:
   1. Hero: a WebGL ridgeline of drifting distributions behind the name; the name rises
      in on load, and every 7-10 s "PyMC" glitches into "Bayesian" and back. Nothing in the
      hero is tied to scrolling: the copy and the field stay put while the page scrolls.
   2. The editor: a notebook runs its cells when it comes into view ("Run All" replays).
   3. The FAQ: a strip of answer cards that repeats both ways around an active middle card.
   The chapters after the editor (painted tiles) are tiles-scene.js.
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

  // Shared by the title glitch and the field behind it: 0 calm, 1 full glitch.
  const glitch = { level: 0 };

  // 1. Hero field: rows of drifting distributions, a ridgeline plot in motion --------
  function field(canvas, hero) {
    const gl = canvas.getContext("webgl", { alpha: false, antialias: false });
    if (!gl) return;
    const vertex = "attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }";
    const fragment = `
      precision highp float;
      uniform vec2 uRes;
      uniform float uTime;
      uniform vec2 uMouse;
      uniform float uFade;
      uniform float uGlitch;
      uniform vec4 uCopy; // centre and half-size of the hero copy, in 0-1 canvas units
      const int ROWS = 42;
      float hash(float n) { return fract(sin(n) * 43758.5453); }
      float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      // One row: three Gaussian bumps whose centres drift like a wave travelling through
      // the rows. Quieter towards the edges, like a ridgeline plot.
      float ridge(float i, float x, float t) {
        float h = 0.0;
        for (int k = 0; k < 3; k++) {
          float fk = float(k);
          float s = i * 7.13 + fk * 3.71;
          float c = 0.62 * sin(t * (0.08 + 0.035 * fk) - i * 0.17 + fk * 2.1) + 0.3 * (hash(s) - 0.5);
          float w = 0.06 + 0.1 * hash(s + 1.3);
          float a = (0.35 + 0.65 * hash(s + 2.7)) * (0.65 + 0.35 * sin(t * 0.25 + i * 0.37 + fk * 1.7));
          float z = (x - c) / w;
          h += a * exp(-0.5 * z * z);
        }
        float e = x / 0.95;
        return h * exp(-e * e * e * e);
      }
      // Rows run front (bottom, partly below the frame) to back (top); a row's line shows
      // only above every row in front of it, so nearer distributions hide the ones behind.
      vec3 scene(vec2 p, float t) {
        float px = 1.25 / uRes.y;
        vec3 col = vec3(0.0);
        float front = -10.0;
        for (int j = 0; j < ROWS; j++) {
          float i = float(j);
          float depth = i / float(ROWS - 1);
          float scale = 1.0 - 0.45 * depth;
          float base = -0.68 + depth * 0.8;
          if (p.y < base - 2.0 * px) break;              // every later row sits higher
          if (p.y > base + 0.4 * scale) continue;         // this row cannot reach the pixel
          float x = (p.x - (uMouse.x - 0.5) * 0.06 * depth) / scale;
          float y = base + ridge(i, x, t) * 0.22 * scale;
          float visible = smoothstep(front - px, front + px, p.y);
          float d = abs(p.y - y);
          float line = smoothstep(px * 1.4, 0.0, d);
          float glow = smoothstep(px * 12.0, 0.0, d) * 0.12;
          vec3 c = mix(vec3(0.706, 0.906, 0.867), vec3(0.624, 0.667, 0.886), smoothstep(0.0, 0.55, depth));
          c = mix(c, vec3(0.784, 0.706, 0.906), smoothstep(0.55, 1.0, depth));
          col += c * (line * 0.85 + glow) * visible * (1.0 - 0.7 * depth);
          front = max(front, y);
        }
        return col;
      }
      void main() {
        vec2 uv = gl_FragCoord.xy / uRes;
        vec2 p = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
        // Portrait screens see a wider slice, so the rows still read as distributions.
        p.x *= max(1.0, 0.85 * uRes.y / uRes.x);
        float t = uTime;
        vec3 col;
        if (uGlitch > 0.001) {
          // Torn horizontal bands, a colour split and scanlines.
          float band = floor(uv.y * 28.0);
          float tick = floor(t * 20.0);
          float jump = step(0.6, hash2(vec2(band, tick)));
          p.x += (hash2(vec2(band + 3.1, tick)) - 0.5) * 0.14 * uGlitch * jump;
          float off = 0.01 * uGlitch;
          col = vec3(scene(p + vec2(off, 0.0), t).r, scene(p, t).g, scene(p - vec2(off, 0.0), t).b);
          col *= 0.88 + 0.12 * sin(gl_FragCoord.y * 2.2 + t * 40.0);
        } else {
          col = scene(p, t);
        }
        // Rows dissolve as they reach the bottom.
        float bottom = smoothstep(0.0, 0.3, uv.y);
        float sides = smoothstep(0.0, 0.1, uv.x) * smoothstep(1.0, 0.9, uv.x);
        // Quiet behind the copy so it stays legible.
        float behindCopy = length((uv - uCopy.xy) / max(uCopy.zw * 1.2, vec2(0.01)));
        float copy = mix(0.22, 1.0, smoothstep(0.75, 1.35, behindCopy));
        vec3 navy = vec3(0.027, 0.078, 0.165);
        col = navy + col * bottom * sides * copy;
        col = mix(navy, col, uFade);
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
    const uGlitch = u("uGlitch");
    const uCopy = u("uCopy");
    const copy = hero.querySelector(".mfm-hero__copy");

    // Thin lines need full resolution; cap the pixel ratio to bound the cost.
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(uRes, canvas.width, canvas.height);
    };
    // The copy's layout box (offsets ignore the intro and scroll transforms), in the
    // canvas's 0-1 units with y up.
    const copyBox = () => {
      const w = Math.max(1, hero.clientWidth);
      const h = Math.max(1, hero.clientHeight);
      gl.uniform4f(
        uCopy,
        (copy.offsetLeft + copy.offsetWidth / 2) / w,
        1 - (copy.offsetTop + copy.offsetHeight / 2) / h,
        copy.offsetWidth / 2 / w,
        copy.offsetHeight / 2 / h,
      );
    };
    const mouse = { x: 0.5, tx: 0.5 };
    let start = performance.now();
    const draw = (ms, still = false) => {
      mouse.x += (mouse.tx - mouse.x) * 0.03;
      copyBox();
      gl.uniform1f(uTime, 20 + ms / 1000);
      gl.uniform2f(uMouse, mouse.x, 0.5);
      gl.uniform1f(uFade, still ? 1 : clamp01((ms - start) / 2400));
      gl.uniform1f(uGlitch, glitch.level);
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

  // "PyMC" scrambles into "Bayesian", holds, and returns: first after a few seconds, then
  // every 7-10 s while the hero is on screen. The field behind tears and splits its colours
  // for the same moments. It never runs under reduced motion, nor once the reader has
  // scrolled away from the hero.
  const GLITCH_FIRST_MS = 3800;
  const GLITCH_HOLD_MS = 1100;

  function titleGlitch(word, hero) {
    if (reduceMotion) return;
    const FROM = word.textContent;
    const TO = "Bayesian";
    const GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789#%&$@";
    const set = (text) => {
      word.textContent = text;
      word.dataset.text = text;
    };
    const scramble = async (from, to, ms) => {
      const frames = Math.max(1, Math.round(ms / 45));
      for (let f = 1; f <= frames; f++) {
        const progress = f / frames;
        const length = Math.round(from.length + (to.length - from.length) * progress);
        let text = "";
        for (let i = 0; i < length; i++) {
          text += Math.random() < progress && to[i] ? to[i] : GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
        }
        set(text);
        await wait(45);
      }
      set(to);
    };
    const mark = hero.querySelector('[data-mfm="mark"]');
    // The line under the title wraps to the name's width (landing.css): hold that width while
    // the longer word shows, so the line does not reflow.
    const name = hero.querySelector('[data-mfm="name"]');
    const burst = async (from, to, ms) => {
      word.classList.add("is-glitching");
      mark?.classList.add("is-glitching");
      glitch.level = 1;
      await scramble(from, to, ms);
      word.classList.remove("is-glitching");
      mark?.classList.remove("is-glitching");
      glitch.level = 0.15;
      await wait(110);
      glitch.level = 0;
    };
    let onScreen = true;
    whenVisible(hero, (visible) => {
      onScreen = visible;
    });
    const schedule = (delay) =>
      setTimeout(async () => {
        if (onScreen && !document.hidden && window.scrollY < hero.offsetHeight * 0.4) {
          if (name) name.style.width = `${name.getBoundingClientRect().width}px`;
          await burst(FROM, TO, 380);
          await wait(GLITCH_HOLD_MS);
          await burst(TO, FROM, 320);
          if (name) name.style.width = "";
        }
        schedule(7000 + Math.random() * 3000);
      }, delay);
    schedule(GLITCH_FIRST_MS);
  }

  // Hero copy: the words rise out of a blur on load ------------------------------------
  function heroCopy(hero) {
    const html = document.documentElement;
    const done = () => html.classList.remove("mfm-intro");
    if (reduceMotion || !hasGsap()) {
      done();
      return;
    }
    gsap.registerPlugin(ScrollTrigger);
    done();
    gsap
      .timeline({ defaults: { ease: "power3.out" } })
      .from(".mfm-hero__mark", { autoAlpha: 0, y: 14, scale: 0.92, filter: "blur(12px)", duration: 0.8, clearProps: "filter" }, 0)
      .from(".mfm-hero__word", { autoAlpha: 0, y: 46, filter: "blur(16px)", duration: 0.8, stagger: 0.1 }, 0.1)
      .from(".mfm-hero__line", { autoAlpha: 0, y: 12, filter: "blur(8px)", duration: 0.9 }, 0.55)
      .from(".mfm-hero__actions > *", { autoAlpha: 0, y: 10, duration: 0.7, stagger: 0.08 }, 0.8);
  }

  // 2. The notebook ----------------------------------------------------------------
  // Its outputs are static HTML and matplotlib SVGs from a real River 1.0 run; this only
  // plays the run: cells execute in turn, outputs reveal, the editor follows along.
  function notebook(ide) {
    const cells = [...ide.querySelectorAll(".mfm-cell")];
    const scroller = ide.querySelector('[data-mfm="cells"]');
    const runAll = ide.querySelector('[data-mfm="run-all"]');
    const cellOf = ide.querySelector('[data-mfm="cell-of"]');
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

  // 3. FAQ strip ----------------------------------------------------------------------
  // The page holds each card once. A copy of the set goes on each side, hidden from
  // assistive tech and out of the tab order (its links still work with a pointer, since a
  // copy can pass through the middle). The card nearest the middle is the active one
  // (.is-active). Whenever the strip comes to rest on a copy it jumps by exactly one set to
  // the same card among the originals: it never runs out either way, and the card that
  // keyboard focus reaches is the one in view. Arrows step one card and announce the
  // question they land on, a mouse drags the strip, touch and trackpads scroll it natively,
  // and a pointer click or keyboard focus on a side card brings it to the middle; scroll
  // snapping lands every move on a card.
  function faqStrip(track, prev, next) {
    const cards = [...track.children];
    const count = cards.length;
    if (count < 2) return;
    const copies = () =>
      cards.map((card) => {
        const copy = card.cloneNode(true);
        copy.setAttribute("aria-hidden", "true");
        copy.querySelectorAll("a, button").forEach((el) => el.setAttribute("tabindex", "-1"));
        return copy;
      });
    track.prepend(...copies());
    track.append(...copies());
    const all = [...track.children];
    // Names the question the arrows land on; they do not move focus.
    const status = document.createElement("p");
    status.className = "visually-hidden";
    status.setAttribute("aria-live", "polite");
    track.after(status);

    const behavior = reduceMotion ? "auto" : "smooth";
    const step = () => all[1].offsetLeft - all[0].offsetLeft; // card plus gap
    // The scrollLeft that puts card i of the whole strip in the middle, and back.
    const centre = (i) => all[0].offsetLeft + all[0].offsetWidth / 2 - track.clientWidth / 2 + i * step();
    const nearest = (left = track.scrollLeft) =>
      Math.min(all.length - 1, Math.max(0, Math.round((left - centre(0)) / step())));
    const bringIn = (i) => track.scrollTo({ left: centre(i), behavior });

    let active = null;
    const light = () => {
      const card = all[nearest()];
      if (card === active) return;
      if (active) active.classList.remove("is-active");
      card.classList.add("is-active");
      active = card;
    };
    // A move by whole sets changes which copy sits in the middle: light it at once, with
    // no transition to give the swap away.
    const jump = (by) => {
      track.classList.add("is-jumping");
      track.scrollLeft += by;
      light();
      void track.offsetWidth;
      track.classList.remove("is-jumping");
    };
    // At rest the middle card is always one of the originals.
    const settle = () => {
      const i = nearest();
      if (i < count) jump(count * step());
      else if (i >= 2 * count) jump(-count * step());
    };
    jump(centre(count) - track.scrollLeft);

    let target = null; // the card the arrows are taking the strip to
    let held = false; // a finger or the mouse is on it
    let announce = false; // the arrows moved the strip: name the question it lands on
    let rest;
    let frame = 0;
    const atRest = () => {
      clearTimeout(rest);
      rest = setTimeout(() => {
        if (held) return;
        target = null;
        track.classList.remove("is-dragging");
        settle();
        light();
        if (!announce) return;
        announce = false;
        status.textContent = `Question ${all.indexOf(active) - count + 1} of ${count}: ${active.querySelector("h3").textContent}`;
      }, 150);
    };
    track.addEventListener("scroll", () => {
      if (!frame) frame = requestAnimationFrame(() => {
        frame = 0;
        light();
      });
      atRest();
    }, { passive: true });

    const go = (direction) => {
      if (target === null) {
        settle();
        target = nearest();
      } else if (target < count || target >= 2 * count) {
        // Presses faster than the strip comes to rest: move it one set towards the
        // originals (the same cards), so it never runs out of cards mid-flight.
        const by = target < count ? count : -count;
        jump(by * step());
        target += by;
      }
      target += direction;
      announce = true;
      bringIn(target);
    };
    if (prev) prev.addEventListener("click", () => go(-1));
    if (next) next.addEventListener("click", () => go(1));

    // A pointer click on a side card brings it to the middle instead of following a link
    // it blurs. Keyboard activation (detail 0) always follows the focused link.
    track.addEventListener("click", (event) => {
      const card = event.target.closest(".mfm-faq__card");
      if (!card || card === active || event.detail === 0) return;
      event.preventDefault();
      target = null;
      bringIn(all.indexOf(card));
    });
    track.addEventListener("focusin", (event) => {
      const card = event.target.closest(".mfm-faq__card");
      if (card && card !== active) bringIn(all.indexOf(card));
    });

    track.addEventListener("touchstart", () => (held = true), { passive: true });
    for (const type of ["touchend", "touchcancel"]) {
      track.addEventListener(type, () => {
        held = false;
        atRest();
      }, { passive: true });
    }

    // Mouse drag. A short flick still moves one card; the click that ends a drag neither
    // follows a link nor brings a card in.
    let drag = null;
    track.addEventListener("dragstart", (event) => event.preventDefault());
    track.addEventListener("pointerdown", (event) => {
      if (event.pointerType !== "mouse" || event.button !== 0) return;
      drag = { x: event.clientX, left: track.scrollLeft, moved: false };
      held = true;
      target = null;
    });
    window.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const dx = event.clientX - drag.x;
      if (!drag.moved) {
        if (Math.abs(dx) < 5) return;
        drag.moved = true;
        track.classList.add("is-dragging");
        window.getSelection()?.removeAllRanges();
      }
      track.scrollLeft = drag.left - dx;
    });
    const drop = () => {
      if (!drag) return;
      const { left, moved } = drag;
      drag = null;
      held = false;
      if (!moved) {
        atRest(); // a press held through a move still settles the strip
        return;
      }
      const swallow = (event) => {
        event.preventDefault();
        event.stopPropagation();
      };
      track.addEventListener("click", swallow, { capture: true, once: true });
      setTimeout(() => track.removeEventListener("click", swallow, { capture: true }), 0);
      const travelled = track.scrollLeft - left;
      let card = nearest();
      if (card === nearest(left) && Math.abs(travelled) > step() * 0.15) card += Math.sign(travelled);
      bringIn(card);
      atRest();
    };
    window.addEventListener("pointerup", drop);
    window.addEventListener("pointercancel", drop);
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

  // Links to a section of this page (#id) scroll here instead of following
  // the URL: at the host root the page carries a <base> that points its links at the site
  // folder, so "#id" resolves to the site's copy of this page. Quarto also rewrites such
  // links to absolute URLs, so match the resolved address: this page or the base folder.
  const thisPage = (link) => {
    const base = new URL(".", document.baseURI).pathname;
    return [window.location.pathname, base, `${base}index.html`].includes(link.pathname);
  };
  function inPageLinks() {
    root.addEventListener("click", (event) => {
      const link = event.target.closest("a[href]");
      const target = link && link.hash.length > 1 && thisPage(link) && document.getElementById(link.hash.slice(1));
      if (!target) return;
      event.preventDefault();
      target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
      target.focus({ preventScroll: true });
    });
  }

  const hero = root.querySelector(".mfm-hero");
  const fieldCanvas = root.querySelector('[data-mfm="field"]');
  const glitchWord = root.querySelector('[data-mfm="glitch"]');
  const ide = root.querySelector('[data-mfm="notebook"]');

  if (fieldCanvas && hero) field(fieldCanvas, hero);
  if (glitchWord && hero) titleGlitch(glitchWord, hero);
  if (ide) notebook(ide);
  const faq = root.querySelector('[data-mfm="faq"]');
  if (faq) faqStrip(faq, root.querySelector('[data-mfm="faq-prev"]'), root.querySelector('[data-mfm="faq-next"]'));
  reveal();
  inPageLinks();
  // The intro animates laid-out text: wait for the web fonts.
  const ready = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
  ready.then(() => {
    if (hero) heroCopy(hero);
    if (hasGsap()) ScrollTrigger.refresh();
  });
})();
