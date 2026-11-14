/* Landing page notebook (the home page's other motion is tiles-v2.js):
   the editor runs its cells when it comes into view ("Run All" replays).
   Everything is decoration over static HTML; with reduced motion the notebook shows
   finished. */
(() => {
  "use strict";

  const root = document.querySelector(".mfm");
  if (!root) return;
  root.classList.add("mfm-js");

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  function whenVisible(element, callback, options) {
    if (!("IntersectionObserver" in window)) {
      callback(true);
      return;
    }
    new IntersectionObserver((entries) => {
      for (const entry of entries) callback(entry.isIntersecting);
    }, options).observe(element);
  }

  // 3. The notebook ----------------------------------------------------------------
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

  const ide = root.querySelector('[data-mfm="notebook"]');

  if (ide) notebook(ide);
  reveal();
  inPageLinks();
})();
