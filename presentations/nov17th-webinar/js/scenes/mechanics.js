/* Mechanics scenes: start backwards (9), decompose in code (13) and model outcome (15).
   Every scene renders from the deck's slide and build step, so fast navigation never drifts.
   The presenter mirror and reduced motion get each step's final state at once. */
(() => {
  'use strict';
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const mirror = new URLSearchParams(location.search).get('mirror') === '1';
  const still = () => mirror || reducedMotion.matches;
  const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - ((-2 * t + 2) ** 3) / 2);

  // Slide 9: at step 0 one token loops through assume, act and measure, then iterates back.
  // Step 1 (CSS) plays the parallel view once; entering it already built shows the final state.
  function startBackwards() {
    const slide = document.getElementById('start-backwards');
    const route = slide && slide.querySelector('.mech-seq-route');
    const token = slide && slide.querySelector('.mech-seq-token');
    const parallel = slide && slide.querySelector('.mech-par');
    if (!route || !token || !parallel) return null;
    const dwell = 520;
    let legs = null;
    let period = 0;
    let active = false;
    let step = -1;
    let frame = 0;
    let origin = 0;

    function place(distance) {
      const point = route.getPointAtLength(distance);
      token.setAttribute('cx', point.x.toFixed(1));
      token.setAttribute('cy', point.y.toFixed(1));
    }

    // Station distances along the route: assume 0, act 250, measure 500, then the iterate arc home.
    function measure() {
      const length = route.getTotalLength();
      legs = [{ from: 0, to: 250, ms: 1000 }, { from: 250, to: 500, ms: 1000 }, { from: 500, to: length, ms: 2100 }];
      period = legs.reduce((total, leg) => total + dwell + leg.ms, 0);
    }

    function distanceAt(elapsed) {
      let time = elapsed % period;
      for (const leg of legs) {
        if (time < dwell) return leg.from;
        time -= dwell;
        if (time < leg.ms) return leg.from + (leg.to - leg.from) * easeInOut(time / leg.ms);
        time -= leg.ms;
      }
      return 0;
    }

    function tick(now) {
      place(distanceAt(now - origin));
      frame = requestAnimationFrame(tick);
    }

    function stop() {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
    }

    return (detail) => {
      const isActive = detail.slideId === slide.id;
      const nextStep = isActive ? detail.step | 0 : -1;
      if (isActive === active && nextStep === step) return;
      const entering = isActive && !active;
      active = isActive;
      step = nextStep;
      stop();
      if (!active) return;
      parallel.classList.toggle('mech-instant', still() || (entering && step > 0));
      if (step > 0) return;
      if (!legs) measure();
      place(0);
      if (still()) return;
      origin = performance.now();
      frame = requestAnimationFrame(tick);
    };
  }

  // Slide 13: step 0 types block 01 on entry; steps 1–3 type the next block. Each render is a pure
  // function of (block, elapsed), so earlier blocks complete and later ones hide on every step change.
  function decompose() {
    const slide = document.getElementById('decompose-in-code');
    const code = slide && slide.querySelector('.mech-code');
    const caret = slide && slide.querySelector('.mech-caret');
    const label = slide && slide.querySelector('.mech-term-step');
    if (!code || !caret || !label) return null;
    const rows = Array.from(code.querySelectorAll('.ln'));
    const items = Array.from(slide.querySelectorAll('.mech-steps li'));
    const blocks = Array.from(code.querySelectorAll('.mech-block'), (block) => {
      const lines = Array.from(block.querySelectorAll('.ln'), (element) => {
        const text = element.textContent;
        return { element, row: rows.indexOf(element), length: text.length, indent: text.length - text.trimStart().length, start: 0 };
      });
      const chars = lines.reduce((total, line) => total + line.length - line.indent, 0);
      const pace = Math.min(1800, Math.max(700, chars * 12)) / (chars + 3 * lines.length);
      let time = 0;
      for (const line of lines) {
        line.start = time;
        time += (line.length - line.indent + 3) * pace;
      }
      return { lines, pace, duration: time, number: block.dataset.number, name: block.dataset.name };
    });
    if (!blocks.length) return null;
    let active = false;
    let step = -1;
    let frame = 0;

    function showLine(line, visible) {
      const partial = visible >= 0 && visible < line.length;
      line.element.classList.toggle('is-hidden', visible < 0);
      line.element.classList.toggle('is-typing', partial);
      if (partial) line.element.style.setProperty('--n', String(visible));
      else line.element.style.removeProperty('--n');
    }

    function caretAt(row, col, idle) {
      caret.style.setProperty('--row', String(row));
      caret.style.setProperty('--col', String(col));
      caret.classList.toggle('is-idle', idle);
    }

    // Blocks 0..last complete, later blocks hidden, caret waiting where the next block starts.
    function rest(last) {
      blocks.forEach((block, position) => {
        for (const line of block.lines) showLine(line, position <= last ? line.length : -1);
      });
      const next = blocks[last + 1];
      if (next) {
        caretAt(next.lines[0].row, 0, true);
      } else {
        const end = blocks[last].lines[blocks[last].lines.length - 1];
        caretAt(end.row, end.length, true);
      }
    }

    function type(position, delay) {
      rest(position - 1);
      const block = blocks[position];
      const begin = performance.now() + delay;
      const tick = (now) => {
        const elapsed = now - begin;
        if (elapsed >= block.duration || still()) {
          frame = 0;
          rest(position);
          return;
        }
        if (elapsed >= 0) {
          let current = block.lines[0];
          let col = 0;
          for (const line of block.lines) {
            const visible = elapsed < line.start ? -1 : Math.min(line.length, line.indent + Math.floor((elapsed - line.start) / block.pace));
            showLine(line, visible);
            if (visible >= 0) {
              current = line;
              col = visible;
            }
          }
          caretAt(current.row, col, false);
        }
        frame = requestAnimationFrame(tick);
      };
      frame = requestAnimationFrame(tick);
    }

    function stop() {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
    }

    function setLabel(position) {
      label.querySelector('.n').textContent = blocks[position].number;
      label.querySelector('.mech-term-step-name').textContent = blocks[position].name;
      items.forEach((item, index) => {
        if (index === position) item.setAttribute('aria-current', 'step');
        else item.removeAttribute('aria-current');
      });
    }

    return (detail) => {
      const isActive = detail.slideId === slide.id;
      const target = isActive ? Math.min(detail.step | 0, blocks.length - 1) : -1;
      // A repeated event for the same step (for example a notes update) must not restart typing.
      if (isActive === active && target === step) return;
      const entering = isActive && !active;
      const previous = step;
      active = isActive;
      step = target;
      stop();
      if (!active) return;
      setLabel(target);
      if (still()) rest(target);
      else if (entering) {
        if (target === 0) type(0, 600);
        else rest(target);
      } else if (target > previous) type(target, 60);
      else rest(target);
    };
  }

  // Slide 15: clicking or focusing a candidate DAG shows the posterior conditional on it.
  function modelOutcome() {
    const slide = document.getElementById('model-outcome');
    const figure = slide && slide.querySelector('.mech-outcome');
    const name = slide && slide.querySelector('.mech-post-dag');
    const buttons = slide ? Array.from(slide.querySelectorAll('.mech-dag')) : [];
    if (!figure || !name || !buttons.length) return null;
    let active = false;

    function select(button) {
      for (const candidate of buttons) candidate.setAttribute('aria-pressed', String(candidate === button));
      figure.dataset.selected = button.dataset.dag;
      name.textContent = button.dataset.name;
    }

    for (const button of buttons) {
      button.addEventListener('click', () => select(button));
      button.addEventListener('focus', () => select(button));
    }

    return (detail) => {
      const isActive = detail.slideId === slide.id;
      // Each visit starts from the most plausible DAG.
      if (isActive && !active) select(buttons[0]);
      active = isActive;
    };
  }

  const scenes = [startBackwards(), decompose(), modelOutcome()].filter(Boolean);
  addEventListener('webinar:slide', (event) => {
    for (const scene of scenes) scene(event.detail);
  });
})();
