/* Closing scenes. Slide 23 draws a conceptual knowledge frontier (no model data): the learning circle and
   its generated prior-data points, an uneven frontier that grows outward at build step 1, and probes at
   one distance from the circle (step 2). One geometry definition drives the shape, the growth and the probes. */
(() => {
  'use strict';

  const slide = document.getElementById('knowledge-frontier');
  const figure = slide ? slide.querySelector('.kf-figure') : null;
  if (!figure) return;

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const CENTER_X = 470;
  const CENTER_Y = 430;
  const RADIUS = 158;
  const PROBE_RADIUS = RADIUS + 128;
  const BASE_REACH = RADIUS + 30;
  // Frontier lobes: [direction in degrees, extra reach, sharpness]. Directions are clockwise from +x.
  const LOBES = [[-24, 222, 10], [176, 206, 8], [62, 178, 12], [-128, 158, 12], [126, 64, 14]];
  const PROBE_ANGLES = [-82, -26, 20, 62, 112, 176];
  const LEADER_ANGLE = -34;
  const LEADER_TOP = 172;
  const POINT_COUNT = 160;
  const SAMPLES = 120;
  const GROW_MS = 1700;

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const mirror = new URLSearchParams(window.location.search).get('mirror') === '1';
  const band = figure.querySelector('.kf-band');
  const edge = figure.querySelector('.kf-edge');
  const circlePath = `M${CENTER_X - RADIUS} ${CENTER_Y}a${RADIUS} ${RADIUS} 0 1 0 ${RADIUS * 2} 0a${RADIUS} ${RADIUS} 0 1 0 ${-RADIUS * 2} 0Z`;
  let frame = 0;
  let onSlide = false;
  let shownStep = 0;

  const radians = (degrees) => degrees * Math.PI / 180;
  const fixed = (value) => Math.round(value * 10) / 10;
  const easeOut = (t) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));

  function reach(angle) {
    return LOBES.reduce((total, [direction, extra, sharpness]) => (
      total + extra * Math.exp(sharpness * (Math.cos(angle - radians(direction)) - 1))
    ), BASE_REACH);
  }

  function polar(angle, radius) {
    return [CENTER_X + Math.cos(angle) * radius, CENTER_Y + Math.sin(angle) * radius];
  }

  function element(name, attributes) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    return node;
  }

  // A closed Catmull-Rom spline through the samples, written as cubic Béziers.
  function closedSpline(points) {
    const count = points.length;
    let d = `M${fixed(points[0][0])} ${fixed(points[0][1])}`;
    for (let i = 0; i < count; i++) {
      const [x0, y0] = points[(i - 1 + count) % count];
      const [x1, y1] = points[i];
      const [x2, y2] = points[(i + 1) % count];
      const [x3, y3] = points[(i + 2) % count];
      d += `C${fixed(x1 + (x2 - x0) / 6)} ${fixed(y1 + (y2 - y0) / 6)} ${fixed(x2 - (x3 - x1) / 6)} ${fixed(y2 - (y3 - y1) / 6)} ${fixed(x2)} ${fixed(y2)}`;
    }
    return `${d}Z`;
  }

  const samples = Array.from({ length: SAMPLES }, (_, i) => {
    const angle = (i / SAMPLES) * Math.PI * 2;
    return { angle, full: reach(angle) };
  });
  const longest = Math.max(...samples.map((sample) => sample.full)) - RADIUS;

  // Every direction leaves the circle at the same pace; short reaches settle early, long lobes keep going.
  function drawFrontier(elapsed) {
    const points = samples.map(({ angle, full }) => {
      const span = full - RADIUS;
      const duration = GROW_MS * (0.3 + 0.7 * span / longest);
      return polar(angle, RADIUS + span * easeOut(Math.min(1, elapsed / duration)));
    });
    const outline = closedSpline(points);
    edge.setAttribute('d', outline);
    band.setAttribute('d', `${outline}${circlePath}`);
  }

  // Deterministic generated prior data: spread over the circle, clear of the circle's own label.
  function populate() {
    let seed = 23;
    const random = () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const points = document.createDocumentFragment();
    let placed = 0;
    while (placed < POINT_COUNT) {
      const angle = random() * Math.PI * 2;
      const distance = (RADIUS - 12) * Math.sqrt(random());
      const [x, y] = polar(angle, distance);
      const radius = 1.7 + random() * 1.9;
      const opacity = 0.45 + random() * 0.5;
      const delay = 200 + (distance / RADIUS) * 900 + random() * 160;
      if (Math.abs(x - CENTER_X) < 136 && y > CENTER_Y - 42 && y < CENTER_Y + 44) continue;
      const dot = element('circle', { cx: fixed(x), cy: fixed(y), r: fixed(radius), 'fill-opacity': opacity.toFixed(2) });
      dot.style.setProperty('--d', `${Math.round(delay)}ms`);
      points.append(dot);
      placed++;
    }
    figure.querySelector('.kf-points').append(points);
  }

  function placeProbes() {
    const connectors = document.createDocumentFragment();
    const marks = document.createDocumentFragment();
    for (const degrees of PROBE_ANGLES) {
      const angle = radians(degrees);
      const [startX, startY] = polar(angle, RADIUS + 6);
      const [endX, endY] = polar(angle, PROBE_RADIUS - 17);
      const [x, y] = polar(angle, PROBE_RADIUS);
      connectors.append(element('path', { d: `M${fixed(startX)} ${fixed(startY)}L${fixed(endX)} ${fixed(endY)}`, pathLength: 1 }));
      const mark = element('g', { class: `kf-mark ${reach(angle) > PROBE_RADIUS ? 'is-inside' : 'is-beyond'}` });
      mark.append(
        element('circle', { class: 'kf-mark-ring', cx: fixed(x), cy: fixed(y), r: 11 }),
        element('circle', { class: 'kf-mark-dot', cx: fixed(x), cy: fixed(y), r: 4 }),
      );
      marks.append(mark);
    }
    figure.querySelector('.kf-connectors').append(connectors);
    figure.querySelector('.kf-probe-marks').append(marks);
  }

  function placeLeader() {
    const angle = radians(LEADER_ANGLE);
    const [x, y] = polar(angle, reach(angle) + 5);
    figure.querySelector('.kf-leader').setAttribute('d', `M${fixed(x)} ${fixed(y)}V${LEADER_TOP}`);
  }

  function stop() {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
  }

  function settle() {
    stop();
    drawFrontier(Infinity);
  }

  function grow() {
    stop();
    const start = performance.now();
    const tick = (now) => {
      const elapsed = now - start;
      drawFrontier(elapsed);
      frame = elapsed < GROW_MS ? requestAnimationFrame(tick) : 0;
    };
    drawFrontier(0);
    frame = requestAnimationFrame(tick);
  }

  populate();
  placeProbes();
  placeLeader();
  settle();

  addEventListener('webinar:slide', (event) => {
    const detail = event.detail || {};
    if (detail.slideId !== slide.id) {
      onSlide = false;
      shownStep = 0;
      if (frame) settle();
      return;
    }
    const step = Number.isInteger(detail.step) ? detail.step : 0;
    // Grow only on the forward build while the slide is showing; other arrivals show the settled frontier.
    const growing = onSlide && shownStep < 1 && step >= 1 && !reducedMotion.matches && !mirror;
    onSlide = true;
    shownStep = step;
    if (growing) grow();
    else if (step < 1 || !frame) settle();
  });
  reducedMotion.addEventListener('change', () => {
    if (frame) settle();
  });
})();
