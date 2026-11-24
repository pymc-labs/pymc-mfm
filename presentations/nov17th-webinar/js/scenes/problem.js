/* Problem section scenes (slides 4–8). Every figure is drawn from deterministic synthetic data: the shapes are
   illustrative, not model output. Motion runs only while its slide is active, honours reduced motion and always
   settles on the static final state. */
(() => {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const TAU = Math.PI * 2;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

  const clamp01 = (value) => Math.min(1, Math.max(0, value));
  // Exponential ease-out, close to the deck's --ease-out curve.
  const easeOut = (value) => (value >= 1 ? 1 : 1 - 2 ** (-10 * value));
  const bump = (t, centre, width) => Math.exp(-0.5 * ((t - centre) / width) ** 2);
  const normal = (t, mean, sd) => bump(t, mean, sd) / (sd * Math.sqrt(TAU));
  const total = (values) => values.reduce((sum, value) => sum + value, 0);

  // mulberry32: a small seeded generator, so every load draws exactly the same figure.
  function seeded(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let value = Math.imul(state ^ (state >>> 15), 1 | state);
      value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function line(xs, ys) {
    let path = '';
    for (let i = 0; i < xs.length; i++) path += `${i ? 'L' : 'M'}${xs[i].toFixed(1)} ${ys[i].toFixed(1)}`;
    return path;
  }

  // A closed band: along the upper curve left to right, back along the lower curve.
  function band(xs, upper, lower) {
    let path = line(xs, upper);
    for (let i = xs.length - 1; i >= 0; i--) path += `L${xs[i].toFixed(1)} ${lower[i].toFixed(1)}`;
    return `${path}Z`;
  }

  function svg(name, attributes) {
    const element = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
    return element;
  }

  // Slide 4: y(t) = f1(t) + f2(t). The components draw first (CSS); then the sum assembles from them: it rises
  // from flat to f1, then f2 is added on top, while dashed guides tie each peak of the sum to its component.
  function measureMarketing() {
    const root = document.querySelector('#measure-marketing .pb-sum');
    if (!root) return null;
    const left = 60;
    const right = 940;
    const unit = 150;
    const floors = { y: 232, f1: 466, f2: 692 };
    const ripple = (t, phase) => 0.022 * Math.sin(TAU * 7.3 * t + phase) + 0.014 * Math.sin(TAU * 12.1 * t + 1.7 * phase);
    const curves = {
      f1: (t) => 0.1 + 0.85 * bump(t, 0.22, 0.052) + 0.72 * bump(t, 0.665, 0.062) + ripple(t, 0.8),
      f2: (t) => 0.09 + 0.8 * bump(t, 0.43, 0.048) + 0.86 * bump(t, 0.855, 0.05) + ripple(t, 2.6),
    };
    const peaks = [[0.22, 'f1'], [0.43, 'f2'], [0.665, 'f1'], [0.855, 'f2']];
    const ts = Array.from({ length: 221 }, (_, i) => i / 220);
    const xs = ts.map((t) => left + t * (right - left));
    const values = { f1: ts.map(curves.f1), f2: ts.map(curves.f2) };
    const part = (kind, name) => root.querySelector(`.pb-${kind}.pb-${name}`);
    const area = (ys, floor) => `${line(xs, ys)}L${right} ${floor}L${left} ${floor}Z`;

    const axes = root.querySelector('.pb-axes');
    for (const floor of Object.values(floors)) axes.append(svg('line', { x1: left, x2: right + 6, y1: floor, y2: floor }));
    for (const name of ['f1', 'f2']) {
      const ys = values[name].map((value) => floors[name] - unit * value);
      part('line', name).setAttribute('d', line(xs, ys));
      part('area', name).setAttribute('d', area(ys, floors[name]));
    }
    const links = peaks.map(([t, source]) => {
      const x = (left + t * (right - left)).toFixed(1);
      const from = (floors[source] - unit * curves[source](t)).toFixed(1);
      const guide = svg('line', { class: `pb-${source}`, x1: x, x2: x, y1: from, y2: from });
      const marker = svg('circle', { class: `pb-${source}`, cx: x, cy: floors.y, r: 5.5 });
      root.querySelector('.pb-guides').append(guide);
      root.querySelector('.pb-markers').append(marker);
      return { t, source, guide, marker };
    });

    let frame = 0;
    // shares.f1 and shares.f2 are how much of each component the sum holds; 1 and 1 is the final state.
    function assemble(shares) {
      const ys = ts.map((_, i) => floors.y - unit * (shares.f1 * values.f1[i] + shares.f2 * values.f2[i]));
      part('line', 'y').setAttribute('d', line(xs, ys));
      part('area', 'y').setAttribute('d', area(ys, floors.y));
      for (const { t, source, guide, marker } of links) {
        const top = (floors.y - unit * (shares.f1 * curves.f1(t) + shares.f2 * curves.f2(t))).toFixed(1);
        guide.setAttribute('y2', top);
        marker.setAttribute('cy', top);
        guide.style.opacity = String(shares[source]);
        marker.style.opacity = String(shares[source]);
      }
    }
    function settle() {
      cancelAnimationFrame(frame);
      frame = 0;
      assemble({ f1: 1, f2: 1 });
    }
    settle();
    return {
      id: 'measure-marketing',
      enter() {
        settle();
        if (reducedMotion.matches) return;
        const start = performance.now();
        assemble({ f1: 0, f2: 0 });
        const tick = (now) => {
          const elapsed = now - start;
          assemble({ f1: easeOut(clamp01((elapsed - 1500) / 900)), f2: easeOut(clamp01((elapsed - 2500) / 900)) });
          frame = elapsed < 3400 ? requestAnimationFrame(tick) : 0;
        };
        frame = requestAnimationFrame(tick);
      },
      leave: settle,
    };
  }

  // Slide 5: attention walks once across the five variants, one decision at a time, then every variant stays lit.
  function decisionLandscape() {
    const list = document.querySelector('#decision-landscape .pb-variants');
    if (!list) return null;
    const variants = Array.from(list.children);
    let timer = 0;
    function focus(position) {
      list.classList.toggle('has-focus', position >= 0);
      variants.forEach((variant, index) => variant.classList.toggle('is-focus', index === position));
    }
    function settle() {
      clearTimeout(timer);
      timer = 0;
      focus(-1);
    }
    return {
      id: 'decision-landscape',
      enter() {
        settle();
        if (reducedMotion.matches) return;
        let position = 0;
        const next = () => {
          focus(position < variants.length ? position : -1);
          position += 1;
          timer = position <= variants.length ? setTimeout(next, 2600) : 0;
        };
        timer = setTimeout(next, 1700);
      },
      leave: settle,
    };
  }

  // Slide 6: model versions that fit the same observations equally well. Each fit is the shared shape plus its
  // own smooth deviation, scaled so that R² against the observed series is exactly 0.90.
  function noRightAnswer() {
    const rows = Array.from(document.querySelectorAll('#no-right-answer .pb-version'));
    if (!rows.length) return;
    const width = 760;
    const floor = 124;
    const ceiling = 8;
    const ts = Array.from({ length: 191 }, (_, i) => i / 190);
    const xs = ts.map((t) => t * width);
    const shape = (t) => 0.22 + 0.6 * bump(t, 0.2, 0.045) + 0.38 * bump(t, 0.415, 0.042) + 0.3 * bump(t, 0.63, 0.075) + 0.52 * bump(t, 0.835, 0.045);
    const random = seeded(7);
    const white = ts.map(() => random() - 0.5);
    const noise = white.map((value, i) => 0.05 * ((white[i - 1] ?? value) + value + (white[i + 1] ?? value)) / 3);
    const observed = ts.map((t, i) => shape(t) + noise[i]);
    const mean = total(observed) / observed.length;
    const spread = total(observed.map((value) => (value - mean) ** 2));
    const versions = rows.map((row, index) => {
      const draw = seeded(101 + 37 * index);
      const waves = [1.4, 2.3, 3.6].map((frequency) => [frequency * (0.8 + 0.4 * draw()), TAU * draw(), 0.5 + draw()]);
      const deviation = ts.map((t) => total(waves.map(([frequency, phase, amplitude]) => amplitude * Math.sin(TAU * frequency * t + phase))));
      // Residuals are noise − s·deviation; solve a·s² + b·s + c = 0 so that 1 − SSres/SStot = 0.90.
      const a = total(deviation.map((value) => value * value));
      const b = -2 * total(deviation.map((value, i) => value * noise[i]));
      const c = total(noise.map((value) => value * value)) - 0.1 * spread;
      const scale = (-b + Math.sqrt(b * b - 4 * a * c)) / (2 * a);
      const fit = ts.map((t, i) => shape(t) + scale * deviation[i]);
      const width90 = 0.045 + 0.035 * draw();
      const phase = TAU * draw();
      const half = ts.map((t) => width90 * (1 + 0.45 * Math.sin(TAU * 1.2 * t + phase)));
      return { row, fit, upper: fit.map((value, i) => value + half[i]), lower: fit.map((value, i) => value - half[i]) };
    });
    const high = Math.max(...observed, ...versions.flatMap((version) => version.upper));
    const y = (value) => floor - (value / high) * (floor - ceiling);
    const observedPath = line(xs, observed.map(y));
    for (const { row, fit, upper, lower } of versions) {
      row.querySelector('.pb-observed').setAttribute('d', observedPath);
      row.querySelector('.pb-fit').setAttribute('d', line(xs, fit.map(y)));
      row.querySelector('.pb-band').setAttribute('d', band(xs, upper.map(y), lower.map(y)));
      const axis = row.querySelector('.pb-axis');
      for (const [key, value] of Object.entries({ x1: 0, x2: width, y1: floor + 2, y2: floor + 2 })) axis.setAttribute(key, String(value));
    }
  }

  // Slide 7: the posterior of one estimate given one specification (narrow), and across specifications: a mixture
  // of the posteriors that several reasonable specifications give (wider). Build step 1 morphs the bars in CSS.
  function underestimateUncertainty() {
    const bars = document.querySelector('#underestimate-uncertainty .pb-bars');
    const outline = document.querySelector('#underestimate-uncertainty .pb-density-one');
    const axis = document.querySelector('#underestimate-uncertainty .pb-axis');
    if (!bars || !outline || !axis) return;
    const left = 40;
    const floor = 432;
    const full = 380;
    const pitch = 6;
    const barWidth = 3;
    const count = 147;
    const span = count * pitch;
    const one = (t) => normal(t, 0.46, 0.048);
    const specifications = [[0.3, 0.05, 0.16], [0.39, 0.055, 0.22], [0.46, 0.048, 0.26], [0.56, 0.06, 0.2], [0.67, 0.055, 0.16]];
    const across = (t) => total(specifications.map(([mean, sd, weight]) => weight * normal(t, mean, sd)));
    const random = seeded(23);
    // Multiplicative jitter stands in for the sampling noise of a histogram of posterior draws.
    const heights = Array.from({ length: count }, (_, i) => {
      const t = (i + 0.5) / count;
      return [one(t) * (0.88 + 0.24 * random()), across(t) * (0.88 + 0.24 * random())];
    });
    const tallest = Math.max(...heights.map(([narrow]) => narrow));
    heights.forEach(([narrow, wide], i) => {
      if (Math.max(narrow, wide) / tallest < 0.004) return;
      const rect = svg('rect', { x: (left + i * pitch + (pitch - barWidth) / 2).toFixed(1), y: floor - full, width: barWidth, height: full });
      rect.style.setProperty('--n', (narrow / tallest).toFixed(4));
      rect.style.setProperty('--w', (wide / tallest).toFixed(4));
      bars.append(rect);
    });
    const ts = Array.from({ length: 161 }, (_, i) => 0.46 + 0.2 * (i / 80 - 1));
    outline.setAttribute('d', line(ts.map((t) => left + t * span), ts.map((t) => floor - (full * one(t)) / tallest)));
    for (const [key, value] of Object.entries({ x1: left - 20, x2: left + span + 20, y1: floor + 1, y2: floor + 1 })) axis.setAttribute(key, String(value));
  }

  const scenes = [measureMarketing(), decisionLandscape()].filter(Boolean);
  noRightAnswer();
  underestimateUncertainty();

  let current = '';
  addEventListener('webinar:slide', (event) => {
    const id = event.detail.slideId;
    // Build steps and edited notes dispatch again for the same slide; only a slide change restarts a scene.
    if (id === current) return;
    for (const scene of scenes) if (scene.id === current) scene.leave();
    current = id;
    for (const scene of scenes) if (scene.id === current) scene.enter();
  });
  reducedMotion.addEventListener('change', () => {
    if (!reducedMotion.matches) return;
    for (const scene of scenes) if (scene.id === current) scene.leave();
  });
})();
