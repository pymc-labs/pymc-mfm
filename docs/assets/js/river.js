/* River 1.0 page: reveal the charts, numbers and resource lists as they scroll into view.
   The charts' colours are in model.css (data-series), so they follow light and dark mode.
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
