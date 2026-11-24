/* A decorative, procedural line field. Each slide chooses it with data-field: contour, ribbon,
   ribbon-low or none (the default). No model data or research results. */
(() => {
  'use strict';
  const canvas = document.getElementById('field');
  const context = canvas && canvas.getContext('2d');
  if (!context) return;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const modes = new Set(['contour', 'ribbon', 'ribbon-low', 'none']);
  let mode = 'none';
  let light = false;
  let frame = 0;
  let last = 0;
  let width = 1920;
  let height = 1080;

  // The stage is scaled with a CSS transform, so its logical size is offsetWidth/offsetHeight.
  function fit() {
    width = canvas.offsetWidth || 1920;
    height = canvas.offsetHeight || 1080;
    const scale = Math.min(devicePixelRatio || 1, 1.5);
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    context.setTransform(scale, 0, 0, scale, 0, 0);
  }

  function drawContour(phase) {
    const centerX = width - 310;
    const centerY = height * 0.68;
    context.lineWidth = 1.4;
    for (let ring = 1; ring <= 27; ring++) {
      context.beginPath();
      const radius = ring * 29;
      for (let point = 0; point <= 160; point++) {
        const angle = point * Math.PI / 80;
        const ripple = 1 + Math.sin(angle * 3 + phase + ring * 0.12) * 0.06;
        const x = centerX + Math.cos(angle) * radius * ripple;
        const y = centerY + Math.sin(angle) * radius * 0.54 * ripple;
        if (point === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.closePath();
      context.strokeStyle = `rgba(180,231,221,${0.28 * (1 - ring / 32)})`;
      context.stroke();
    }
    context.strokeStyle = '#f6ae72';
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(centerX, centerY, 16, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.arc(centerX, centerY, 5, 0, Math.PI * 2);
    context.fillStyle = '#f6ae72';
    context.fill();
  }

  function drawRibbon(phase, low) {
    const lineCount = 40;
    const middle = low ? height * 0.8 : height * 0.58;
    context.lineWidth = 1.4;
    context.strokeStyle = light ? 'rgba(71,100,125,.10)' : 'rgba(180,231,221,.12)';
    for (let line = 0; line < lineCount; line++) {
      const spread = line / lineCount;
      context.beginPath();
      for (let x = -20; x <= width + 20; x += 12) {
        const envelope = low ? Math.sin(Math.max(0, x) / (width * 1.09) * Math.PI) : 1;
        const y = middle + Math.sin(x / 370 + phase + spread * 0.45) * 80
          + (spread - 0.5) * 170 * envelope + Math.sin(x / 1400 + phase) * 55;
        if (x === -20) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
    }
  }

  function draw(time) {
    context.clearRect(0, 0, width, height);
    const phase = reducedMotion.matches ? 0 : time / 14000;
    if (mode === 'contour') drawContour(phase);
    else if (mode === 'ribbon' || mode === 'ribbon-low') drawRibbon(phase, mode === 'ribbon-low');
  }

  function animate(time) {
    frame = 0;
    if (time - last > 45) {
      draw(time);
      last = time;
    }
    if (mode !== 'none' && !reducedMotion.matches && !document.hidden) frame = requestAnimationFrame(animate);
  }

  function restart() {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    draw(performance.now());
    if (mode !== 'none' && !reducedMotion.matches && !document.hidden) frame = requestAnimationFrame(animate);
  }

  addEventListener('webinar:slide', (event) => {
    const slide = document.querySelectorAll('#deck > section.slide')[event.detail.index];
    const requested = slide ? slide.dataset.field : '';
    mode = modes.has(requested) ? requested : 'none';
    light = event.detail.theme === 'light';
    restart();
  });
  new ResizeObserver(() => {
    fit();
    restart();
  }).observe(canvas);
  reducedMotion.addEventListener('change', restart);
  document.addEventListener('visibilitychange', restart);
  fit();
  restart();
})();
