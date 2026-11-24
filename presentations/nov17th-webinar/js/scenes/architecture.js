/* Architecture scenes: the signal pulse on the MFM overview (slide 10), the illustrative world
   generator (slide 11) and the signal that climbs the transformer encoder into the flow (slide 12). */
(() => {
  'use strict';

  const mirror = new URLSearchParams(window.location.search).get('mirror') === '1';
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const canMove = () => !mirror && !reducedMotion.matches;

  // A short dash that travels along `path`. Each path starts at distance `data-at` on its scene's
  // signal route, so dashes on separate connectors stay in step with one shared front.
  function comet(path, dash) {
    const length = path.getTotalLength();
    const at = Number(path.dataset.at) || 0;
    path.style.strokeDasharray = `${dash} 100000`;
    return (front) => {
      const head = Math.min(Math.max(front - at, 0), length + dash);
      path.style.strokeDashoffset = String(dash - head);
    };
  }

  /* ---- 10 · A slow pulse along the built inference connectors ---- */
  function overviewScene(slide) {
    const SPEED = 260; // route px per second
    const PERIOD = 6.4; // seconds between pulses
    const DASH = 26;
    let paints = null;
    let frame = 0;
    let origin = 0;

    function tick(now) {
      const seconds = (now - origin) / 1000;
      const front = seconds < 0 ? 0 : (seconds % PERIOD) * SPEED;
      paints.forEach((paint) => paint(front));
      frame = window.requestAnimationFrame(tick);
    }

    function stop() {
      window.cancelAnimationFrame(frame);
      frame = 0;
      slide.classList.remove('is-pulsing');
    }

    function start() {
      stop();
      if (!canMove()) return;
      if (!paints) paints = Array.from(slide.querySelectorAll('.arch-pulse'), (path) => comet(path, DASH));
      origin = performance.now() + 1200;
      slide.classList.add('is-pulsing');
      frame = window.requestAnimationFrame(tick);
    }

    return { enter: start, step() {}, leave: stop, refresh: start };
  }

  /* ---- 11 · Illustrative world generator ---- */
  function worldScene(slide) {
    const POINTS = 52; // weekly values, January to December
    const START = Object.freeze({ trend: 25, seasonality: 24, frequency: 30, volatility: 18, lift: 18 });
    const KEYS = Object.keys(START);
    const NAMES = { trend: 'Trend', seasonality: 'Seasonality', frequency: 'Frequency', volatility: 'Volatility', lift: 'Campaign lift' };
    const RANGES = { trend: [10, 85], seasonality: [5, 70], frequency: [5, 80], volatility: [5, 50], lift: [0, 80] };
    const MONTHS = [['Jan', 0], ['Mar', 2], ['May', 4], ['Jul', 6], ['Sep', 8], ['Dec', 11]];
    const SVG = 'http://www.w3.org/2000/svg';

    const chart = slide.querySelector('.world-chart');
    const svg = chart.querySelector('svg');
    const inputs = new Map(Array.from(slide.querySelectorAll('input[data-param]'), (input) => [input.dataset.param, input]));
    const outputs = new Map(KEYS.map((key) => [key, slide.querySelector(`output[for="${inputs.get(key).id}"]`)]));
    const tagSign = slide.querySelector('.world-tag-sign');
    const tagText = slide.querySelector('.world-tag-text');
    const tagValue = slide.querySelector('.world-tag-value');
    const status = slide.querySelector('.world-status');

    // New world and Reset change every control while focus stays on the button: say what changed.
    function announceWorld(label, values) {
      if (status) status.textContent = `${label}: ${KEYS.map((key) => `${NAMES[key]} ${Math.round(values[key])}%`).join(', ')}.`;
    }

    // Deterministic worlds: world n always draws the same noise, shape and parameters.
    function generator(seed) {
      let state = seed >>> 0;
      return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    }

    function makeWorld(index) {
      const next = generator(Math.imul(index + 1, 0x9e3779b1));
      const noise = new Float64Array(POINTS);
      let previous = 0;
      for (let week = 0; week < POINTS; week += 1) {
        const normal = Math.sqrt(-2 * Math.log(1 - next())) * Math.cos(2 * Math.PI * next());
        previous = 0.55 * previous + 0.835 * normal; // unit-variance AR(1) noise
        noise[week] = previous;
      }
      const world = { noise, phase: next() * 2 * Math.PI, campaign: 0.3 + next() * 0.4, level: 44 + next() * 12 };
      world.params = index === 0 ? { ...START } : Object.fromEntries(KEYS.map((key) => {
        const [low, high] = RANGES[key];
        return [key, Math.round(low + next() * (high - low))];
      }));
      return world;
    }

    // Each slider is a share of its range: trend tilts the year, seasonality and frequency shape the
    // cycle, campaign lift adds a decaying response, volatility scales the world's own noise.
    function series(world, values) {
      const result = new Float64Array(POINTS);
      const slope = ((values.trend - 50) / 50) * 34;
      const amplitude = (values.seasonality / 100) * 34;
      const cycles = 1 + (values.frequency / 100) * 5;
      const sigma = (values.volatility / 100) * 26;
      const lift = (values.lift / 100) * 48;
      for (let week = 0; week < POINTS; week += 1) {
        const share = week / (POINTS - 1);
        const since = (share - world.campaign) / 0.06;
        const response = since > 0 ? since * Math.exp(1 - since) : 0;
        const value = world.level + slope * (share - 0.5)
          + amplitude * Math.sin(2 * Math.PI * cycles * share + world.phase)
          + lift * response + sigma * world.noise[week];
        result[week] = Math.min(98, Math.max(2, value));
      }
      return result;
    }

    const worlds = [makeWorld(0)];
    const startValues = series(worlds[0], START);
    let world = worlds[0];
    let drawn = 0;
    let params = { ...START };
    let shown = startValues.slice();
    let box = null;
    let parts = null;
    let tween = 0;
    let settle = null;
    let demoTimer = 0;

    function node(name, attributes, parent) {
      const element = document.createElementNS(SVG, name);
      for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
      parent.append(element);
      return element;
    }

    const xOf = (share) => box.left + share * (box.right - box.left);
    const yOf = (value) => box.bottom - (value / 100) * (box.bottom - box.top);
    const pathOf = (values) => Array.from(values, (value, week) => `${week ? 'L' : 'M'}${xOf(week / (POINTS - 1)).toFixed(1)} ${yOf(value).toFixed(1)}`).join('');

    // The chart is drawn in layout pixels so its 15px labels never scale down.
    function layout() {
      const width = chart.clientWidth;
      const height = chart.clientHeight;
      if (!width || !height) return false;
      if (box && box.width === width && box.height === height) return true;
      box = { width, height, left: 46, right: width - 16, top: 9, bottom: height - 34 };
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svg.replaceChildren();
      const axes = node('g', { class: 'world-axes' }, svg);
      for (const value of [0, 25, 50, 75, 100]) {
        node('line', { x1: box.left, x2: box.right, y1: yOf(value), y2: yOf(value) }, axes);
        node('text', { x: box.left - 14, y: yOf(value) + 5, 'text-anchor': 'end' }, axes).textContent = String(value);
      }
      for (const [label, month] of MONTHS) {
        node('text', { x: xOf(month / 11), y: height - 8, 'text-anchor': 'middle' }, axes).textContent = label;
      }
      parts = {
        area: node('path', { class: 'world-area' }, svg),
        start: node('path', { class: 'world-start' }, svg),
        line: node('path', { class: 'world-line' }, svg),
        dot: node('circle', { class: 'world-dot', r: 5 }, svg),
      };
      parts.start.setAttribute('d', pathOf(startValues));
      return true;
    }

    function draw() {
      if (!layout()) return;
      const line = pathOf(shown);
      parts.line.setAttribute('d', line);
      parts.area.setAttribute('d', `${line}L${xOf(1).toFixed(1)} ${yOf(0)}L${xOf(0).toFixed(1)} ${yOf(0)}Z`);
      parts.dot.setAttribute('cx', xOf(1).toFixed(1));
      parts.dot.setAttribute('cy', yOf(shown[POINTS - 1]).toFixed(1));
    }

    function render() {
      for (const key of KEYS) {
        const value = Math.round(params[key]);
        const input = inputs.get(key);
        input.value = String(value);
        input.style.setProperty('--fill', String(value / 100));
        input.setAttribute('aria-valuetext', `${value}%`);
        outputs.get(key).textContent = `${value}%`;
      }
      draw();
    }

    function setTag(sign, text, value) {
      tagSign.textContent = sign;
      tagSign.hidden = !sign;
      tagText.textContent = text;
      tagValue.textContent = value;
    }

    // Stop any scripted motion; a running transition completes to a consistent state.
    function cancel() {
      window.clearTimeout(demoTimer);
      demoTimer = 0;
      window.cancelAnimationFrame(tween);
      tween = 0;
      const finish = settle;
      settle = null;
      if (finish) finish();
    }

    function animate(duration, ease, step, finish) {
      const begin = performance.now();
      settle = finish;
      const frame = (now) => {
        const progress = Math.min(1, (now - begin) / duration);
        step(ease(progress));
        if (progress < 1) {
          tween = window.requestAnimationFrame(frame);
        } else {
          tween = 0;
          settle = null;
          finish();
        }
      };
      tween = window.requestAnimationFrame(frame);
    }

    const easeOut = (k) => 1 - (1 - k) ** 3;
    const easeInOut = (k) => (k < 0.5 ? 4 * k ** 3 : 1 - (-2 * k + 2) ** 3 / 2);

    function moveTo(target, values, tag) {
      cancel();
      setTag(...tag);
      world = target;
      const fromParams = { ...params };
      const fromValues = shown.slice();
      const toValues = series(target, values);
      const finish = () => {
        params = { ...values };
        shown = toValues;
        render();
      };
      if (!canMove()) {
        finish();
        return;
      }
      animate(700, easeOut, (k) => {
        for (const key of KEYS) params[key] = fromParams[key] + (values[key] - fromParams[key]) * k;
        shown = fromValues.map((value, week) => value + (toValues[week] - value) * k);
        render();
      }, finish);
    }

    function newWorld() {
      drawn += 1;
      if (!worlds[drawn]) worlds[drawn] = makeWorld(drawn);
      moveTo(worlds[drawn], worlds[drawn].params, ['', 'New world sampled', '']);
      announceWorld(`World ${drawn}`, worlds[drawn].params);
    }

    function reset() {
      drawn = 0;
      moveTo(worlds[0], START, ['', 'Starting values', '']);
      announceWorld('Starting values', START);
    }

    // The entry demonstration: the trend slider sweeps up once and the series responds.
    function demo() {
      demoTimer = 0;
      const from = params.trend;
      const to = 65;
      setTag('+', 'Trend increasing', `${Math.round(from)}%`);
      const show = () => {
        tagValue.textContent = `${Math.round(params.trend)}%`;
        shown = series(world, params);
        render();
      };
      animate(1800, easeInOut, (k) => {
        params.trend = from + (to - from) * k;
        show();
      }, () => {
        params.trend = Math.round(params.trend);
        show();
      });
    }

    function onInput(event) {
      const input = event.currentTarget;
      const key = input.dataset.param;
      const value = Number(input.value);
      cancel();
      const previous = Math.round(params[key]);
      params[key] = value;
      if (value !== previous) {
        const rising = value > previous;
        setTag(rising ? '+' : '−', `${NAMES[key]} ${rising ? 'increasing' : 'decreasing'}`, `${value}%`);
      }
      shown = series(world, params);
      render();
    }

    for (const input of inputs.values()) {
      let pointing = false;
      input.addEventListener('input', onInput);
      input.addEventListener('pointerdown', () => { pointing = true; });
      input.addEventListener('keydown', () => { pointing = false; });
      // After a pointer drag, hand the keyboard back to the deck so the clicker advances slides.
      input.addEventListener('change', () => {
        if (!pointing) return;
        pointing = false;
        input.blur();
      });
    }
    slide.querySelector('.world-new').addEventListener('click', newWorld);
    slide.querySelector('.world-reset').addEventListener('click', reset);
    slide.querySelector('.world-lab').addEventListener('pointerdown', cancel);
    slide.querySelector('.world-lab').addEventListener('keydown', cancel);
    new ResizeObserver(draw).observe(chart);

    function enter() {
      cancel();
      drawn = 0;
      world = worlds[0];
      params = { ...START };
      shown = startValues.slice();
      setTag('', 'Starting values', '');
      render();
      if (canMove()) demoTimer = window.setTimeout(demo, 1100);
    }

    render();
    return { enter, step() {}, leave: cancel, refresh() { if (!canMove()) cancel(); } };
  }

  /* ---- 12 · A signal climbs the encoder, then crosses into the graph-conditional flow ---- */
  function encoderScene(slide) {
    const SPEED = 480; // route px per second
    const DASH = 30;
    const LINGER = 150; // route px a node stays lit after the signal leaves it
    const ENCODER_END = 687; // top of the encoded representations
    const CROSS_START = 660;
    const FLOW_END = 1675; // top of the effects over time
    const lit = Array.from(slide.querySelectorAll('[data-lit]'), (element) => {
      const [from, to] = element.dataset.lit.split(' ').map(Number);
      return { element, from, to };
    });
    let paints = null;
    let frame = 0;
    let run = null;

    function clear() {
      window.cancelAnimationFrame(frame);
      frame = 0;
      run = null;
      slide.classList.remove('is-signalling');
      lit.forEach(({ element }) => element.classList.remove('is-lit'));
    }

    function tick(now) {
      const elapsed = now - run.begin;
      if (elapsed >= 0) {
        const front = run.from + (elapsed / 1000) * SPEED;
        if (front > run.to + LINGER + DASH) {
          clear();
          return;
        }
        paints.forEach((paint) => paint(front));
        lit.forEach(({ element, from, to }) => element.classList.toggle('is-lit', front >= from && front <= to + LINGER));
      }
      frame = window.requestAnimationFrame(tick);
    }

    function play(from, to, delay) {
      clear();
      if (!canMove()) return;
      if (!paints) paints = Array.from(slide.querySelectorAll('.tx-signal'), (path) => comet(path, DASH));
      paints.forEach((paint) => paint(0));
      run = { from, to, begin: performance.now() + delay };
      slide.classList.add('is-signalling');
      frame = window.requestAnimationFrame(tick);
    }

    return {
      enter(step) {
        play(0, step >= 1 ? FLOW_END : ENCODER_END, 900);
      },
      step(step) {
        if (step < 1) clear();
        else if (run) run.to = FLOW_END; // still climbing: carry on into the flow
        else play(CROSS_START, FLOW_END, 300);
      },
      leave: clear,
      refresh() { if (!canMove()) clear(); },
    };
  }

  const scenes = new Map();
  for (const [id, scene] of [['architecture-overview', overviewScene], ['create-worlds', worldScene], ['transformer-encoder', encoderScene]]) {
    const slide = document.getElementById(id);
    if (slide) scenes.set(id, scene(slide));
  }

  let activeId = '';
  let activeStep = 0;
  window.addEventListener('webinar:slide', (event) => {
    const detail = event.detail || {};
    const slide = document.querySelectorAll('#deck > section.slide')[detail.index];
    const id = slide ? slide.id : '';
    const step = Number.isInteger(detail.step) ? detail.step : Infinity;
    if (id === activeId) {
      if (step === activeStep) return;
      activeStep = step;
      const scene = scenes.get(id);
      if (scene) scene.step(step);
      return;
    }
    const previous = scenes.get(activeId);
    if (previous) previous.leave();
    activeId = id;
    activeStep = step;
    const scene = scenes.get(id);
    if (scene) scene.enter(step);
  });
  reducedMotion.addEventListener('change', () => {
    const scene = scenes.get(activeId);
    if (scene) scene.refresh();
  });
})();
