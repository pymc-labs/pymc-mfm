/* River 1.0 page.
   1. Hero, layers back to front:
      - the river: one still picture of the Orinoco at night (assets/img/river), brought to
        life in WebGL. The water runs towards the viewer and its moonlit path twinkles; the
        leaves in the near corner stir; cloud shadows drift over the forest and thin wisps
        over the sky; the moon's halo breathes and faint rays turn. Every 8 to 14 seconds
        the picture glitches for half a second: wide torn bands, a colour split, scanlines.
        The picture is graded towards the page's colours: navy shadows, cobalt and
        periwinkle midtones, cool white highlights, forest greens leaning to aqua. Mist
        drifts over it, behind the name;
      - faint numbers that glitch in and out among the mist;
      - the name, "River", and its version, in the site's display face;
      - mist in front of everything, the name included: thinner wisps, and now and then a
        thicker bank that rolls across the name and hides it for a moment.
      So the name sits inside the mist: some passes behind it, some in front.
      Everything pauses off screen; reduced motion draws one still frame and no numbers.
      Without WebGL the picture shows as it is.
      The hero fills the first screen under the navbar, which stays clear (no glass, light
      ink) while the hero is under it: body.mfm-nav-clear, site.css.
   2. Article: the charts, numbers and resource lists reveal as they scroll into view.
      Their colours are in model.css (data-series), so they follow light and dark mode. */
(() => {
  "use strict";

  const hero = document.querySelector(".mfm-river__hero");
  if (!hero) return;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
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

  // Shared GLSL: value noise on a lattice wrapped at 289 cells, so float precision holds,
  // its fractal sum, and the mist.
  const NOISE = `
    float hash(vec2 p) {
      p = mod(p, 289.0);
      p = fract(p * vec2(123.34, 456.21));
      p += dot(p, p + 45.32);
      return fract(p.x * p.y);
    }
    float noise(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
      return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
                 mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
    }
    const mat2 ROT = mat2(0.8, -0.6, 0.6, 0.8);
    float fbm5(vec2 p) {
      float v = 0.0; float a = 0.5;
      for (int i = 0; i < 5; i++) { v += a * noise(p); p = ROT * p * 2.03; a *= 0.5; }
      return v;
    }
    // Low wisps drawn out sideways, warped by themselves, drifting with time t; lit by the
    // moon from the upper right. p: centred, in screen heights; uv: 0-1 on the screen.
    // Returns colour, premultiplied, and cover.
    vec4 wisps(vec2 p, vec2 uv, float t, float scale, float speed, float seed, float amount) {
      vec2 q = p * vec2(0.8, 2.4) * scale + vec2(t * speed, seed);
      vec2 w = vec2(fbm5(q + vec2(0.0, t * 0.01)), fbm5(q + vec2(5.2, 1.3)));
      float f = fbm5(q + 1.8 * w);
      float d = smoothstep(0.48, 0.86, f) * (0.5 + 0.5 * smoothstep(0.95, 0.2, uv.y)) * amount;
      vec3 c = vec3(0.62, 0.68, 0.79) * (0.7 + 0.45 * smoothstep(-0.7, 0.9, p.x + p.y)) * (0.75 + 0.35 * f);
      return vec4(c * d, d);
    }`;

  // 1a. The picture, brought to life, with mist behind the name ---------------------------
  const SCENE = `
    precision highp float;
    uniform vec2 uRes;
    uniform float uTime;
    uniform float uFade;
    uniform float uScroll;
    uniform float uGlitch;        // 0 calm, 1 at the height of a glitch
    uniform sampler2D uImage;
    uniform vec2 uImageSize;
    uniform sampler2D uWater;     // where the river is in the picture (1 on the water)
    ${NOISE}

    const vec2 MOON = vec2(0.6865, 0.887);   // the moon in the picture (uv, y up)
    const float HORIZON = 0.58;              // where the forest meets the sky

    // The picture covers the canvas, cropped like object-fit: cover. A narrow screen crops
    // around x = 0.6, which keeps the river and the moon.
    vec2 toPicture(vec2 uv) {
      float r = (uRes.x / uRes.y) / (uImageSize.x / uImageSize.y);
      vec2 s = r > 1.0 ? vec2(1.0, 1.0 / r) : vec2(r, 1.0);
      return (uv - 0.5) * s + 0.5 + (vec2(0.6, 0.5) - 0.5) * (1.0 - s);
    }

    // The page's colours: navy shadows, cobalt then periwinkle in the midtones, cool white
    // highlights; what colour the picture has left leans from green to aqua.
    vec3 grade(vec3 c) {
      float l = dot(c, vec3(0.299, 0.587, 0.114));
      vec3 low = mix(vec3(0.012, 0.035, 0.08), vec3(0.11, 0.19, 0.38), smoothstep(0.0, 0.3, l));
      vec3 high = mix(vec3(0.55, 0.61, 0.85), vec3(0.95, 0.97, 1.0), smoothstep(0.5, 0.9, l));
      vec3 ramp = mix(low, high, smoothstep(0.22, 0.65, l));
      vec3 chroma = c - l;
      chroma = vec3(chroma.r * 0.55, chroma.g * 0.95, chroma.b + chroma.g * 0.45);
      return mix(c, ramp + chroma * 0.9, 0.55);
    }

    void main() {
      vec2 uv = gl_FragCoord.xy / uRes;
      vec2 p = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
      float t = uTime;
      float ia = uImageSize.x / uImageSize.y;
      vec2 iuv = toPicture(uv);

      // The glitch: wide torn bands slide sideways and large blocks jump.
      float g = uGlitch;
      float tick = floor(t * 16.0);
      if (g > 0.0) {
        float h = hash(vec2(floor(uv.y * 12.0), tick));
        iuv.x += (h - 0.5) * 0.18 * g * step(0.55, h);
        vec2 cell = floor(uv * vec2(6.0, 10.0));
        iuv += (vec2(hash(cell + 2.0 + tick), hash(cell + 5.0 + tick)) - 0.5) * 0.12 * g * step(0.86, hash(cell + tick * 1.3));
      }

      // The current: on the water the picture slides a little, towards the viewer, less
      // far off. The leaves in the near corner stir in a breeze.
      float water = texture2D(uWater, iuv).r;
      float near = smoothstep(0.55, 0.05, iuv.y);
      vec2 fq = vec2(iuv.x * ia * 70.0, iuv.y * 30.0 + t * 0.9);
      vec2 shift = (vec2(noise(fq), noise(fq + 31.7)) - 0.5) * 0.004 * water * (0.25 + 0.75 * near);
      float leaf = smoothstep(0.24, 0.04, iuv.x) * smoothstep(0.24, 0.02, iuv.y);
      shift += vec2(sin(t * 1.3 + iuv.y * 40.0), cos(t * 1.1 + iuv.x * 30.0)) * 0.0012 * leaf;
      vec2 q = clamp(iuv + shift, 0.001, 0.999);
      float split = 0.007 * g;
      vec3 col = vec3(texture2D(uImage, q + vec2(split, 0.0)).r, texture2D(uImage, q).g, texture2D(uImage, q - vec2(split, 0.0)).b);

      // The moon's path twinkles as the water runs.
      float lum = dot(col, vec3(0.3, 0.59, 0.11));
      float glint = smoothstep(0.3, 0.75, lum) * water * pow(noise(vec2(iuv.x * ia * 260.0, iuv.y * 120.0 + t * 2.2)), 5.0);
      col += vec3(0.9, 0.95, 1.0) * glint * 1.3;

      // Cloud shadows drift over the forest; thin wisps drift over the sky, brighter near
      // the moon.
      float sky = smoothstep(HORIZON - 0.02, HORIZON + 0.06, iuv.y);
      float shade = smoothstep(0.45, 0.78, fbm5(vec2(iuv.x * ia * 2.0 + t * 0.012, iuv.y * 4.0)));
      col *= 1.0 - 0.2 * shade * (1.0 - sky);
      float moonD = length((iuv - MOON) * vec2(ia, 1.0));
      float wisp = smoothstep(0.5, 0.85, fbm5(vec2(iuv.x * ia * 3.0 - t * 0.01, iuv.y * 9.0) + 4.0));
      col = mix(col, vec3(0.55, 0.6, 0.72) * (0.5 + 0.9 * exp(-moonD * 3.0)), wisp * sky * 0.22);

      // The moon breathes: a soft halo, and faint rays that turn slowly.
      float breathe = 0.85 + 0.15 * sin(t * 0.7);
      float halo = exp(-moonD * moonD * 260.0) * 0.22 + exp(-moonD * 11.0) * 0.07;
      float ang = atan(iuv.y - MOON.y, (iuv.x - MOON.x) * ia);
      float rays = pow(noise(vec2(ang * 6.0 + t * 0.04, 3.0)), 3.0) * exp(-moonD * 9.0) * 0.08;
      col += vec3(0.75, 0.82, 1.0) * (halo + rays) * breathe;

      col = grade(col);

      // Mist behind the name: two slow layers.
      vec4 m1 = wisps(p, uv, t * 0.6, 0.9, 0.012, 3.1, 0.28);
      vec4 m2 = wisps(p, uv, t, 1.4, 0.02, 9.7, 0.3);
      col = m1.rgb + col * (1.0 - m1.a);
      col = m2.rgb + col * (1.0 - m2.a);

      // The glitch, again: scanlines, grain, a colour swap in a few bands.
      if (g > 0.0) {
        col *= 1.0 - 0.3 * g * step(0.5, fract(gl_FragCoord.y * 0.5));
        col += g * 0.12 * (hash(floor(gl_FragCoord.xy * 0.5) + fract(t) * 61.0) - 0.5);
        col = mix(col, col.gbr, 0.4 * g * step(0.8, hash(vec2(floor(uv.y * 6.0), tick))));
      }
      col *= 1.0 - 0.55 * uScroll;
      col = mix(vec3(0.02, 0.05, 0.11), col, uFade);
      col += (hash(gl_FragCoord.xy + fract(t) * 97.0) - 0.5) * (2.0 / 255.0);
      gl_FragColor = vec4(col, 1.0);
    }`;

  // 1b. Mist in front of the name -------------------------------------------------------
  const VEIL = `
    precision highp float;
    uniform vec2 uRes;
    uniform float uTime;
    uniform float uFade;
    uniform float uScroll;
    uniform vec4 uName;           // the name's box in canvas uv: centre, half-size
    ${NOISE}

    void main() {
      vec2 uv = gl_FragCoord.xy / uRes;
      vec2 p = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
      float t = uTime;
      vec4 m = wisps(p, uv, t, 1.9, 0.03, 17.3, 0.3);
      // A thick bank rolls across the name about every 20 seconds, for a few seconds.
      vec2 tc = (uv - uName.xy) / max(uName.zw, vec2(0.01));
      float over = exp(-dot(tc * vec2(0.45, 0.8), tc * vec2(0.45, 0.8)));
      float x = fract((t + 8.0) / 20.0);
      float due = smoothstep(0.0, 0.18, x) * smoothstep(0.45, 0.25, x);
      float roll = smoothstep(0.38, 0.72, fbm5(vec2(p.x * 1.3 - t * 0.07, p.y * 3.0 + 2.0)));
      float bank = clamp(over * due * (0.5 + 0.5 * roll), 0.0, 0.9);
      vec3 col = vec3(0.64, 0.7, 0.8) * (0.8 + 0.25 * roll) * bank + m.rgb * (1.0 - bank);
      float a = bank + m.a * (1.0 - bank);
      float k = uFade * (1.0 - 0.85 * uScroll);
      gl_FragColor = vec4(col * k, a * k);
    }`;

  function layer(canvas, fragment, names, contextOptions) {
    const gl = canvas.getContext("webgl", contextOptions);
    if (!gl) return null;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      return gl.getShaderParameter(shader, gl.COMPILE_STATUS) ? shader : null;
    };
    const vs = compile(gl.VERTEX_SHADER, "attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }");
    const fs = compile(gl.FRAGMENT_SHADER, fragment);
    if (!vs || !fs) return null;
    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null;
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(program, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const uniforms = {};
    for (const name of names) uniforms[name] = gl.getUniformLocation(program, name);
    const state = {
      gl,
      uniforms,
      scale: 1,
      resize() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = Math.max(1, Math.round(rect.width * state.scale));
        canvas.height = Math.max(1, Math.round(rect.height * state.scale));
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.uniform2f(uniforms.uRes, canvas.width, canvas.height);
      },
      draw() {
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      },
    };
    return state;
  }

  // A texture on its own unit from a loaded image: linear and clamped, so any size works.
  function texture(gl, unit, image) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
  }

  const loaded = (img) =>
    img.complete && img.naturalWidth
      ? Promise.resolve(img)
      : new Promise((resolve, reject) => {
          img.addEventListener("load", () => resolve(img), { once: true });
          img.addEventListener("error", reject, { once: true });
        });

  function river(hero) {
    const picture = hero.querySelector('[data-mfm="picture"]');
    const sceneCanvas = hero.querySelector('[data-mfm="river"]');
    const veilCanvas = hero.querySelector('[data-mfm="veil"]');
    const title = hero.querySelector(".mfm-river__title");
    if (!picture || !sceneCanvas) return;
    const scene = layer(sceneCanvas, SCENE, ["uRes", "uTime", "uFade", "uScroll", "uGlitch", "uImage", "uImageSize", "uWater"], {
      alpha: false,
      antialias: false,
      powerPreference: "high-performance",
    });
    // Without WebGL the picture shows as it is.
    if (!scene) return;
    const veil = veilCanvas
      ? layer(veilCanvas, VEIL, ["uRes", "uTime", "uFade", "uScroll", "uName"], { alpha: true, premultipliedAlpha: true, antialias: false })
      : null;
    if (veil) {
      veil.scale = 0.4;
      veilCanvas.classList.add("is-ready");
    }

    const gl = scene.gl;
    const u = scene.uniforms;
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    const water = new Image();
    water.src = picture.dataset.water;
    let ready = false;

    // Where the name is, for the mist bank that rolls across it.
    let name = [0.5, 0.5, 0.25, 0.12];
    const measureName = () => {
      if (!title) return;
      const w = Math.max(1, hero.clientWidth);
      const h = Math.max(1, hero.clientHeight);
      const x = title.offsetLeft + title.offsetWidth / 2;
      const y = title.offsetTop + title.offsetHeight / 2;
      name = [x / w, 1 - y / h, title.offsetWidth / 2 / w, title.offsetHeight / 2 / h];
    };
    // A half-second glitch every 8 to 14 seconds.
    const GLITCH_MS = 550;
    let nextGlitch = performance.now() + 4000 + Math.random() * 4000;
    const glitchAt = (ms) => {
      if (reduceMotion) return 0;
      if (ms > nextGlitch + GLITCH_MS) nextGlitch = ms + 8000 + Math.random() * 6000;
      const k = (ms - nextGlitch) / GLITCH_MS;
      return k < 0 || k > 1 ? 0 : Math.sqrt(Math.sin(Math.PI * k));
    };

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    scene.scale = dpr * (window.innerWidth < 700 ? 0.75 : 1);
    const minScale = 0.5;
    const resize = () => {
      scene.resize();
      if (veil) veil.resize();
      measureName();
    };
    let start = performance.now();
    const draw = (ms, still = false) => {
      if (!ready) return;
      // The clock wraps every 30 minutes, which keeps the noise coordinates small.
      const time = 20 + ((ms / 1000) % 1800);
      const fade = still ? 1 : clamp01((ms - start) / 1500);
      const scroll = clamp01(window.scrollY / Math.max(1, hero.offsetHeight));
      gl.uniform1f(u.uTime, time);
      gl.uniform1f(u.uFade, fade);
      gl.uniform1f(u.uScroll, scroll);
      gl.uniform1f(u.uGlitch, glitchAt(ms));
      scene.draw();
      if (veil) {
        const v = veil.uniforms;
        veil.gl.uniform1f(v.uTime, time);
        veil.gl.uniform1f(v.uFade, fade);
        veil.gl.uniform1f(v.uScroll, scroll);
        veil.gl.uniform4f(v.uName, ...name);
        veil.draw();
      }
    };
    resize();
    if ("ResizeObserver" in window) {
      new ResizeObserver(() => {
        resize();
        draw(performance.now(), reduceMotion);
      }).observe(sceneCanvas);
    }

    let raf = 0;
    let onScreen = false;
    let last = 0;
    let frames = 0;
    let spent = 0;
    const loop = (ms) => {
      // Every 40 frames: if they averaged over 24 ms, render 20% fewer pixels per side.
      if (last) {
        spent += ms - last;
        if (++frames === 40) {
          if (spent / frames > 24 && scene.scale > minScale) {
            scene.scale = Math.max(minScale, scene.scale * 0.8);
            resize();
          }
          frames = 0;
          spent = 0;
        }
      }
      last = ms;
      draw(ms);
      raf = requestAnimationFrame(loop);
    };
    const sync = () => {
      cancelAnimationFrame(raf);
      last = 0;
      frames = 0;
      spent = 0;
      if (ready && !reduceMotion && onScreen && !document.hidden) raf = requestAnimationFrame(loop);
    };

    // The canvas shows once both pictures are textures; until then the <img> under it shows.
    Promise.all([loaded(picture), loaded(water)])
      .then(() => {
        texture(gl, 0, picture);
        texture(gl, 1, water);
        gl.uniform1i(u.uImage, 0);
        gl.uniform1i(u.uWater, 1);
        gl.uniform2f(u.uImageSize, picture.naturalWidth, picture.naturalHeight);
        ready = true;
        start = performance.now();
        sceneCanvas.classList.add("is-ready");
        if (reduceMotion) draw(0, true);
        else sync();
      })
      .catch(() => {});

    if (reduceMotion) return;
    whenVisible(sceneCanvas, (visible) => {
      onScreen = visible;
      sync();
    });
    document.addEventListener("visibilitychange", sync);
  }

  // 1c. Numbers in the mist: now and then a figure from the paper, or a posterior draw,
  // scrambles in, holds, and scrambles out. Faint, and never over the title.
  const FIGURES = ["2,443,583", "200,000", "196,000", "62,500", "0.0295", "0.0241", "0.0263", "1.25 s", "11 / 15", "2,048", "208 wk", "16 steps", "3.40"];
  const posteriorDraw = () => {
    const value = (Math.random() * 0.9).toFixed(4);
    return Math.random() < 0.3 ? `\u2212${value}` : value;
  };
  const scramble = (text, shown) =>
    [...text].map((ch, i) => (/\d/.test(ch) && i >= shown * text.length ? String(Math.floor(Math.random() * 10)) : ch)).join("");

  function numbers(hero) {
    const field = hero.querySelector(".mfm-river__numbers");
    if (!field || reduceMotion) return;
    let onScreen = false;
    whenVisible(hero, (visible) => {
      onScreen = visible;
    });
    const spawn = () => {
      window.setTimeout(spawn, 2200 + Math.random() * 3200);
      if (!onScreen || document.hidden) return;
      const text = Math.random() < 0.5 ? FIGURES[Math.floor(Math.random() * FIGURES.length)] : posteriorDraw();
      let x;
      let y;
      do {
        x = 6 + Math.random() * 84;
        y = 14 + Math.random() * 66;
      } while (Math.abs(x - 50) < 24 && Math.abs(y - 50) < 15);
      const el = document.createElement("span");
      el.className = "mfm-river__number";
      el.style.left = `${x.toFixed(1)}%`;
      el.style.top = `${y.toFixed(1)}%`;
      el.textContent = scramble(text, 0);
      field.appendChild(el);
      const glitch = () => {
        el.classList.remove("is-glitch");
        void el.offsetWidth;
        el.classList.add("is-glitch");
      };
      const IN = 600;
      const HOLD = 1600 + Math.random() * 1200;
      const OUT = 700;
      const born = performance.now();
      requestAnimationFrame(() => {
        el.classList.add("is-on");
        glitch();
      });
      if (Math.random() < 0.5) window.setTimeout(glitch, IN + HOLD * (0.3 + Math.random() * 0.4));
      window.setTimeout(() => {
        el.classList.remove("is-on");
        glitch();
      }, IN + HOLD);
      const tick = () => {
        const t = performance.now() - born;
        if (t < IN) el.textContent = scramble(text, t / IN);
        else if (t < IN + HOLD) el.textContent = text;
        else if (t < IN + HOLD + OUT) el.textContent = scramble(text, 1 - (t - IN - HOLD) / OUT);
        else {
          el.remove();
          return;
        }
        window.setTimeout(tick, 70);
      };
      tick();
    };
    window.setTimeout(spawn, 2500);
  }

  // 2. Scroll: the title recedes; the navbar is clear while the hero is under it ------
  function onScroll(hero) {
    const title = hero.querySelector(".mfm-river__title");
    const hint = hero.querySelector(".mfm-river__scroll");
    const bar = document.getElementById("quarto-header");
    let ticking = false;
    const update = () => {
      ticking = false;
      const barHeight = bar ? bar.offsetHeight : 0;
      document.body.classList.toggle("mfm-nav-clear", hero.getBoundingClientRect().bottom > barHeight);
      const height = Math.max(1, hero.offsetHeight);
      if (reduceMotion || !title) return;
      const p = clamp01(window.scrollY / (height * 0.7));
      title.style.opacity = String(1 - p);
      title.style.transform = `translateY(${(-p * 6).toFixed(2)}vh) scale(${(1 + p * 0.08).toFixed(3)})`;
      title.style.filter = p > 0 ? `blur(${(p * 8).toFixed(1)}px)` : "";
      if (hint) hint.style.opacity = String(1 - Math.min(1, p * 4));
    };
    window.addEventListener("scroll", () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(update);
      }
    }, { passive: true });
    update();
  }

  river(hero);
  numbers(hero);
  onScroll(hero);
})();

/* Reveal the charts, numbers and resource lists as they scroll into view.
   With reduced motion, or without IntersectionObserver, everything shows at once. */
(() => {
  "use strict";

  const items = document.querySelectorAll(".mfm-reveal");
  if (!items.length) return;

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion || !("IntersectionObserver" in window)) {
    items.forEach((el) => el.classList.add("is-visible"));
    return;
  }

  // The bars start collapsed only once scripts run; without them they show in full.
  document.documentElement.classList.add("mfm-js");
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    },
    { threshold: 0.18 },
  );
  items.forEach((el) => observer.observe(el));
})();
