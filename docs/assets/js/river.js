/* River 1.0 page.
   1. Hero: we drift down a wide river into fog. The camera sits low and moves with the
      current; the water carries layered waves and ripples, mirrors the banks and a hidden
      sun, and now and then passes a rock with foam and a wake. Low banks and dark trees
      fade into drifting mist. Rendered near device resolution, scaled down when frames
      run slow, paused off screen; reduced motion draws one still frame.
   2. Navbar: transparent over the hero, filling in as the page scrolls.
   3. Article: sections and charts reveal on scroll; the table of contents follows. */
(() => {
  "use strict";

  const root = document.querySelector(".mfm-river");
  if (!root) return;
  root.classList.add("mfm-js");
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

  // 1. The river ------------------------------------------------------------------------
  const FRAGMENT = `
    precision highp float;
    uniform vec2 uRes;
    uniform float uTime;
    uniform vec2 uMouse;
    uniform float uFade;
    uniform float uScroll;

    const float W = 2.8;      // half-width of the river
    const float FLOW = 1.7;   // speed of the water, world units per second
    const float FOG_DENSITY = 0.05;
    const vec3 SUN = vec3(-0.12, 0.06, 0.991);
    const vec3 SUN_COLOR = vec3(1.0, 0.76, 0.52);
    const vec3 FOG = vec3(0.3, 0.35, 0.43);

    // Value noise on a lattice wrapped at 289 cells, so float precision holds downstream.
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
    // Noise value and its gradient.
    vec3 noised(vec2 x) {
      vec2 i = floor(x);
      vec2 f = fract(x);
      vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
      vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
      float a = hash(i);
      float b = hash(i + vec2(1.0, 0.0));
      float c = hash(i + vec2(0.0, 1.0));
      float d = hash(i + vec2(1.0, 1.0));
      float k1 = b - a;
      float k2 = c - a;
      float k4 = a - b - c + d;
      return vec3(a + k1 * u.x + k2 * u.y + k4 * u.x * u.y, du * vec2(k1 + k4 * u.y, k2 + k4 * u.x));
    }
    const mat2 ROT = mat2(0.8, -0.6, 0.6, 0.8);
    float fbm3(vec2 p) {
      float v = 0.0; float a = 0.5;
      for (int i = 0; i < 3; i++) { v += a * noise(p); p = ROT * p * 2.03; a *= 0.5; }
      return v;
    }
    float fbm6(vec2 p) {
      float v = 0.0; float a = 0.5;
      for (int i = 0; i < 6; i++) { v += a * noise(p); p = ROT * p * 2.03; a *= 0.5; }
      return v;
    }

    // The river's centre line, x as a function of z. The water flows towards +z.
    float riverX(float z) { return 3.2 * sin(z * 0.045) + 1.3 * sin(z * 0.11 + 1.7); }

    // Rocks: at most one per cell of a row of cells along the river, (x, z, radius);
    // radius 0 means the cell has none. Two rows of different lengths break the rhythm.
    vec3 rockIn(float c, float size, float seed) {
      float a = hash(vec2(c, seed));
      if (a < 0.6) return vec3(0.0);
      float b = hash(vec2(seed, c));
      float e = hash(vec2(c + 0.5 * seed, 7.0));
      float z = (c + 0.25 + 0.5 * b) * size;
      float x = riverX(z) + ((a - 0.6) / 0.4 * 2.0 - 1.0) * (W - 0.8);
      return vec3(x, z, 0.1 + 0.32 * e * e);
    }
    // A boulder: a wobbly outline, a flat-topped profile and a broken surface.
    float rockHeight(vec2 p, vec3 rk) {
      if (rk.z <= 0.0) return -9.0;
      vec2 q = (p - rk.xy) / rk.z;
      q.y *= 0.72;
      float d2 = dot(q, q);
      if (d2 >= 1.3) return -9.0;
      float wobble = 0.75 + 0.35 * noise(q / sqrt(d2 + 1e-4) * 1.7 + rk.xy * 5.0);
      d2 /= wobble * wobble;
      if (d2 >= 1.0) return -9.0;
      float bump = 0.75 + 0.35 * noise(p * 4.0 + rk.xy * 3.1) + 0.15 * noise(p * 11.0);
      return rk.z * (pow(1.0 - d2, 0.45) * bump * 1.05 - 0.3);
    }
    float rocks(vec2 p) {
      float h = rockHeight(p, rockIn(floor(p.y / 4.0), 4.0, 11.0));
      return max(h, rockHeight(p, rockIn(floor(p.y / 6.5), 6.5, 37.0)));
    }

    // Land: a bed under the water (y = 0), low banks and soft rises, dark stands of trees.
    float bed(float d) { return -0.6 * (1.0 - smoothstep(W - 1.0, W, d)); }
    float banks(float d, float n) {
      return smoothstep(W - 0.2, W + 4.0, d) * (0.1 + n * 0.35 + n * n * 0.5) + bed(d);
    }
    float forest(vec2 p, float d) {
      return smoothstep(W + 1.0, W + 2.6, d) * smoothstep(0.3, 0.55, noise(p * 0.18 + 3.7));
    }
    float landLo(vec2 p) {
      float d = abs(p.x - riverX(p.y));
      if (d < W - 0.35) return max(bed(d), rocks(p));
      return banks(d, fbm3(p * 0.08)) + forest(p, d) * (0.35 + 0.4 * noise(p * 1.3));
    }
    // Shading only: finer relief on banks, crowns and rocks.
    float landHi(vec2 p) {
      float d = abs(p.x - riverX(p.y));
      if (d < W - 0.35) return max(bed(d), rocks(p) + 0.025 * noise(p * 16.0));
      return banks(d, fbm6(p * 0.08)) + forest(p, d) * (0.35 + 0.4 * noise(p * 1.3) + 0.1 * noise(p * 5.0));
    }

    vec3 fogColor(vec3 rd) {
      float s = max(dot(rd, SUN), 0.0);
      vec3 col = mix(FOG, vec3(0.05, 0.08, 0.14), smoothstep(0.03, 0.45, rd.y));
      return col + SUN_COLOR * (pow(s, 8.0) * 0.45 + pow(s, 60.0) * 0.4);
    }

    vec3 landColor(vec3 p, vec3 rd, float t) {
      float e = 0.012 + 0.0015 * t;
      vec3 n = normalize(vec3(
        landHi(p.xz - vec2(e, 0.0)) - landHi(p.xz + vec2(e, 0.0)),
        2.0 * e,
        landHi(p.xz - vec2(0.0, e)) - landHi(p.xz + vec2(0.0, e))));
      float d = abs(p.x - riverX(p.z));
      float dif = clamp(dot(n, SUN), 0.0, 1.0);
      float amb = 0.5 + 0.5 * n.y;
      if (d < W - 0.3) {
        // Stone: wet and dark at the waterline, dry and lichen-grey above it.
        float dry = smoothstep(0.02, 0.22, p.y);
        vec3 stone = mix(vec3(0.03, 0.032, 0.035),
                         mix(vec3(0.2, 0.19, 0.17), vec3(0.22, 0.24, 0.18), noise(p.xz * 7.0)), dry);
        stone *= 0.7 + 0.5 * noise(p.xz * 23.0);
        float spec = pow(max(dot(n, normalize(SUN - rd)), 0.0), 90.0) * (1.0 - dry);
        return stone * (dif * SUN_COLOR * 1.6 + amb * vec3(0.42, 0.47, 0.56)) + spec * SUN_COLOR * 0.5;
      }
      vec3 grass = mix(vec3(0.07, 0.11, 0.06), vec3(0.12, 0.14, 0.08), noise(p.xz * 0.6));
      vec3 base = mix(grass, vec3(0.02, 0.05, 0.035), forest(p.xz, d));
      base = mix(vec3(0.08, 0.075, 0.065), base, smoothstep(W, W + 0.6, d));
      float rim = pow(clamp(1.0 + dot(rd, n), 0.0, 1.0), 4.0);
      return base * (dif * SUN_COLOR * 1.8 + amb * vec3(0.3, 0.36, 0.48)) + rim * SUN_COLOR * 0.08;
    }

    // Slope of fourteen travelling waves, mostly along the flow, from 2.4 units down to
    // 6 cm. A wave shorter than a few pixels fades out, so the surface stays sharp near
    // the camera and calm far away.
    vec2 waveSlope(vec2 p, float px) {
      vec2 g = vec2(0.0);
      float a = 0.022;
      float f = 2.6;
      for (int i = 0; i < 14; i++) {
        float fi = float(i);
        vec2 d = normalize(vec2(sin(fi * 2.39) * 0.85, 1.0));
        float w = smoothstep(1.5, 4.0, 6.2832 / f / px);
        float ph = f * (dot(d, p) - uTime * FLOW * (0.85 + 0.03 * fi)) + fi * 1.7;
        g += d * (f * a * w * cos(ph));
        f *= 1.32;
        a *= 0.74;
      }
      return g;
    }

    // Fine turbulent ripples carried by the current.
    vec2 microSlope(vec2 p, float px) {
      vec2 g = vec2(0.0);
      vec2 q = p * vec2(5.0, 2.6) - vec2(0.0, uTime * FLOW * 2.6);
      float amp = 0.045;
      float sc = 1.0;
      for (int i = 0; i < 4; i++) {
        float w = smoothstep(1.5, 4.0, 0.2 / sc / px);
        g += noised(q).yz * vec2(5.0, 2.6) * sc * amp * w;
        q = mat2(1.6, -1.2, 1.2, 1.6) * q;
        sc *= 2.0;
        amp *= 0.55;
      }
      return g;
    }

    // The current wraps around a rock: a collar of ripples and foam, and a wake downstream.
    void rockWater(vec3 rk, vec2 p, float size, inout vec2 g, inout float foam) {
      if (rk.z <= 0.0) return;
      vec2 dq = p - rk.xy;
      float r = length(dq * vec2(1.0, 0.72)) / rk.z;
      float collar = smoothstep(2.2, 1.0, r);
      g += normalize(dq + 1e-4) * collar * 0.12 * sin(r * 11.0 - uTime * 6.0);
      foam += smoothstep(1.7, 1.0, r) * 0.7;
      float along = dq.y;
      if (along > 0.0) {
        float across = abs(dq.x);
        float width = rk.z * (0.9 + along * 0.28);
        float tail = exp(-along / (rk.z * 5.0)) * smoothstep(width, width * 0.25, across)
                   * smoothstep(size * 1.2, size * 0.6, along);
        float streak = noise(vec2(across / rk.z * 3.0 + rk.x * 7.0, (along - uTime * FLOW) * 2.2));
        foam += tail * smoothstep(0.4, 0.8, streak) * 0.9;
        g += vec2(sign(dq.x), -0.3) * tail * 0.12 * sin(across * 14.0 / rk.z - along * 3.0);
      }
    }

    vec3 waterColor(vec3 p, vec3 rd, float t) {
      // World size of one pixel here; grazing views stretch it along the view.
      float px = t / (uRes.y * 1.55 * sqrt(max(-rd.y, 0.02)));
      vec2 g = waveSlope(p.xz, px) + microSlope(p.xz, px);
      float foam = 0.0;
      float c1 = floor(p.z / 4.0);
      float c2 = floor(p.z / 6.5);
      rockWater(rockIn(c1, 4.0, 11.0), p.xz, 4.0, g, foam);
      rockWater(rockIn(c1 - 1.0, 4.0, 11.0), p.xz, 4.0, g, foam);
      rockWater(rockIn(c2, 6.5, 37.0), p.xz, 6.5, g, foam);
      rockWater(rockIn(c2 - 1.0, 6.5, 37.0), p.xz, 6.5, g, foam);
      vec3 n = normalize(vec3(-g.x, 1.0, -g.y));

      // Mirror the banks, trees and rocks: march the reflected ray against the land.
      vec3 r = reflect(rd, n);
      r.y = abs(r.y) + 0.001;
      vec3 reflected = fogColor(r);
      float glint = 1.0;
      float rt = 0.2;
      for (int i = 0; i < 28; i++) {
        vec3 rp = p + r * rt;
        float h = rp.y - landLo(rp.xz);
        if (h < 0.006 * rt) {
          float d = abs(rp.x - riverX(rp.z));
          vec3 land = d < W - 0.3 ? vec3(0.03, 0.032, 0.035)
                                  : mix(vec3(0.05, 0.07, 0.05), vec3(0.012, 0.03, 0.022), forest(rp.xz, d));
          reflected = mix(land, fogColor(r), 1.0 - exp(-FOG_DENSITY * (rt + t)));
          glint = 0.0;
          break;
        }
        rt += max(0.04, h * 0.5);
        if (rt > 35.0) break;
      }

      float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, -rd), 0.0), 5.0);
      vec3 body = vec3(0.01, 0.035, 0.04) + vec3(0.0, 0.03, 0.028) * clamp(g.y * 3.0 + 0.5, 0.0, 1.0);
      vec3 col = mix(body, reflected, fresnel);
      // A path of light: glints on facets that face the hidden sun.
      float sp = max(dot(r, SUN), 0.0);
      col += SUN_COLOR * (pow(sp, 900.0) * 6.0 + pow(sp, 90.0) * 0.45) * glint;
      // Foam at the banks and around rocks, in lace carried by the current.
      foam += smoothstep(0.1, 0.0, -landLo(p.xz)) * 0.8;
      float lace = smoothstep(0.35, 0.8, noise(vec2(p.x * 9.0, (p.z - uTime * FLOW) * 4.0)));
      return mix(col, vec3(0.6, 0.64, 0.68) * (0.55 + 0.45 * n.y), clamp(foam * lace, 0.0, 1.0) * 0.75);
    }

    void main() {
      vec2 uv = gl_FragCoord.xy / uRes;
      vec2 s = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
      // Drift with the current: a little slower than the water, rocking gently.
      float z = uTime * 1.1 + uScroll * 4.0;
      float bob = 0.035 * sin(uTime * 0.9) + 0.02 * sin(uTime * 1.7 + 1.0);
      vec3 ro = vec3(riverX(z) + 0.25 * sin(uTime * 0.13), 0.62 + bob + uScroll * 0.6, z);
      vec3 ta = vec3(riverX(z + 8.0) + (uMouse.x - 0.5) * 2.0, 0.32 + (uMouse.y - 0.5) * 0.5 + uScroll * 0.3, z + 8.0);
      vec3 ww = normalize(ta - ro);
      float roll = 0.012 * sin(uTime * 0.6);
      vec3 uu = normalize(cross(vec3(sin(roll), cos(roll), 0.0), ww));
      vec3 vv = cross(ww, uu);
      vec3 rd = normalize(s.x * uu + s.y * vv + 1.55 * ww);

      // March the land down to the water plane; refine a hit by bisection. Steps stay
      // short near the water so that small rocks are not skipped.
      const float FAR = 60.0;
      float tw = rd.y < 0.0 ? -ro.y / rd.y : 1e9;
      float tmax = min(tw, FAR);
      float t = 0.05;
      float tPrev = t;
      bool hit = false;
      for (int i = 0; i < 140; i++) {
        vec3 p = ro + rd * t;
        float h = p.y - landLo(p.xz);
        if (h < 0.001 * t) { hit = true; break; }
        tPrev = t;
        float dt = max(0.01, h * 0.4);
        if (p.y < 0.5) dt = min(dt, max(0.12, t * 0.02));
        t += dt;
        if (t > tmax) break;
      }
      if (hit) {
        float lo = tPrev;
        float hi = t;
        for (int j = 0; j < 5; j++) {
          float m = 0.5 * (lo + hi);
          vec3 p = ro + rd * m;
          if (p.y - landLo(p.xz) < 0.0) hi = m; else lo = m;
        }
        t = hi;
      }

      vec3 col;
      float dist;
      if (hit && t < tw) {
        col = landColor(ro + rd * t, rd, t);
        dist = t;
      } else if (tw < FAR) {
        col = waterColor(ro + rd * tw, rd, tw);
        dist = tw;
      } else {
        col = fogColor(rd);
        dist = 2.0 * FAR;
      }

      // Fog: distance haze, mist that clings to the water, and drifting banks of fog.
      col = mix(col, fogColor(rd), 1.0 - exp(-FOG_DENSITY * dist));
      float py = (ro + rd * min(dist, FAR)).y;
      col = mix(col, fogColor(rd), exp(-max(py, 0.0) * 3.0) * (1.0 - exp(-dist * 0.06)) * 0.45);
      for (int k = 0; k < 4; k++) {
        float fk = float(k);
        float tk = 1.5 + fk * 1.2 + fk * fk * 2.2;
        if (tk > dist) break;
        vec3 fp = ro + rd * tk;
        float dens = smoothstep(0.42, 0.9, fbm3(fp.xz * 0.22 + vec2(uTime * 0.04, -uTime * 0.09) + fk * 3.1));
        col = mix(col, fogColor(rd) * 1.05, dens * exp(-max(fp.y, 0.0)) * 0.3);
      }

      col = 1.0 - exp(-col * 1.5);
      col *= 0.45 + 0.55 * pow(16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.25);
      col *= mix(1.0, 0.72, smoothstep(0.72, 1.0, uv.y));
      col *= 1.0 - 0.55 * uScroll;
      col = mix(vec3(0.02, 0.05, 0.11), col, uFade);
      col += (hash(gl_FragCoord.xy + fract(uTime) * 97.0) - 0.5) * (2.0 / 255.0);
      gl_FragColor = vec4(col, 1.0);
    }`;

  function river(canvas, hero) {
    const gl = canvas.getContext("webgl", { alpha: false, antialias: false, powerPreference: "high-performance" });
    if (!gl) return;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      return gl.getShaderParameter(shader, gl.COMPILE_STATUS) ? shader : null;
    };
    const vs = compile(gl.VERTEX_SHADER, "attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }");
    const fs = compile(gl.FRAGMENT_SHADER, FRAGMENT);
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

    // Start near device resolution for the water's detail; scale down if frames run slow.
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let scale = dpr * (window.innerWidth < 700 ? 0.55 : 0.7);
    const minScale = 0.35;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * scale));
      canvas.height = Math.max(1, Math.round(rect.height * scale));
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(uRes, canvas.width, canvas.height);
    };
    const mouse = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5 };
    let start = performance.now();
    const draw = (ms, still = false) => {
      mouse.x += (mouse.tx - mouse.x) * 0.025;
      mouse.y += (mouse.ty - mouse.y) * 0.025;
      // The journey restarts every 15 minutes, which keeps world coordinates small.
      gl.uniform1f(uTime, 20 + ((ms / 1000) % 900));
      gl.uniform2f(uMouse, mouse.x, mouse.y);
      gl.uniform1f(uFade, still ? 1 : clamp01((ms - start) / 2600));
      gl.uniform1f(uScroll, clamp01(window.scrollY / Math.max(1, hero.offsetHeight)));
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
    let last = 0;
    let frames = 0;
    let spent = 0;
    const loop = (ms) => {
      // Every 40 frames: if they averaged over 24 ms, render 20% fewer pixels per side.
      if (last) {
        spent += ms - last;
        if (++frames === 40) {
          if (spent / frames > 24 && scale > minScale) {
            scale = Math.max(minScale, scale * 0.8);
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
      if (onScreen && !document.hidden) raf = requestAnimationFrame(loop);
    };
    whenVisible(canvas, (visible) => {
      onScreen = visible;
      sync();
    });
    document.addEventListener("visibilitychange", sync);
  }

  // 2. Scroll: the navbar fills in, the title recedes ----------------------------------
  function onScroll(hero) {
    const title = hero.querySelector(".mfm-river__title");
    const hint = hero.querySelector(".mfm-hero__scroll");
    let ticking = false;
    const update = () => {
      ticking = false;
      const height = Math.max(1, hero.offsetHeight);
      document.documentElement.style.setProperty("--mfm-nav-alpha", clamp01(window.scrollY / (height * 0.6)).toFixed(3));
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

  // 3. Article: reveal, charts, table of contents ---------------------------------------
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
      { threshold: 0.18 },
    );
    items.forEach((el) => observer.observe(el));
  }

  function tableOfContents() {
    // Quarto rewrites the hrefs to absolute URLs, so match on the hash.
    const links = [...root.querySelectorAll(".mfm-river__toc a")].filter((a) => a.hash);
    const sections = links.map((a) => document.getElementById(a.hash.slice(1))).filter(Boolean);
    if (!sections.length || !("IntersectionObserver" in window)) return;
    const setActive = (id) => links.forEach((a) => a.classList.toggle("is-active", a.hash === `#${id}`));
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) if (entry.isIntersecting) setActive(entry.target.id);
      },
      { rootMargin: "-25% 0px -65% 0px" },
    );
    sections.forEach((section) => observer.observe(section));
  }

  const hero = root.querySelector(".mfm-river__hero");
  const canvas = root.querySelector('[data-mfm="river"]');
  if (hero && canvas) river(canvas, hero);
  if (hero) onScroll(hero);
  reveal();
  tableOfContents();
})();
