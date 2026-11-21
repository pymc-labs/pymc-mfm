/* API overview: the sidebar group of the section in view takes the accent colour (scroll-spy).
   Groups are matched by title: the section "Datasets" of the page and the sidebar group "Datasets".
   Loaded by api/_metadata.yml. Pages other than the overview have no `.mktfm-api-overview`. */
(() => {
  const overview = document.querySelector(".mktfm-api-overview");
  if (!overview || !("IntersectionObserver" in window)) return;

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const groups = new Map();
  for (const group of document.querySelectorAll("#quarto-sidebar .sidebar-item-section")) {
    const title = group.querySelector(".menu-text");
    if (title) groups.set(title.textContent.trim(), group);
  }

  const targets = [...overview.querySelectorAll("section.level2")].filter((section) =>
    groups.has(section.querySelector("h2")?.textContent.trim()),
  );
  if (!targets.length) return;

  let current = null;
  const mark = (section) => {
    const group = section ? groups.get(section.querySelector("h2").textContent.trim()) : null;
    if (group === current) return;
    if (current) {
      current.classList.remove("mktfm-spy-current");
      current.querySelector(".sidebar-item-text")?.removeAttribute("aria-current");
    }
    current = group;
    if (!current) return;
    current.classList.add("mktfm-spy-current");
    current.querySelector(".sidebar-item-text")?.setAttribute("aria-current", "location");
    current.scrollIntoView({ block: "nearest", behavior: reduceMotion.matches ? "auto" : "smooth" });
  };

  // A section counts as in view while it crosses a band a quarter of the way down the viewport.
  const visible = new Set();
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) visible.add(entry.target);
        else visible.delete(entry.target);
      }
      mark(targets.find((section) => visible.has(section)) ?? null);
    },
    { rootMargin: "-25% 0px -70% 0px" },
  );
  targets.forEach((section) => observer.observe(section));
})();
