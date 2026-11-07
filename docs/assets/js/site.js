/* Navbar "Models": one click to any model.
   On wide screens with a pointer, hovering "Models" opens its menu and clicking the label
   goes to the Models page; the menu items go straight to each model. Keyboard users open
   the menu with ArrowDown (Bootstrap's own handling). On the collapsed mobile navbar the
   label keeps Bootstrap's tap-to-open, and "All models" is in the menu.
   A menu item whose text ends in " NEW" shows a "New" badge instead. */
(() => {
  "use strict";

  const toggle = document.getElementById("nav-menu-models");
  if (!toggle) return;
  const item = toggle.closest(".dropdown");
  const menu = item && item.querySelector(".dropdown-menu");
  const all = menu && menu.querySelector('a.dropdown-item[href$="models/index.html"]');
  if (!menu || !all) return;

  menu.classList.add("mfm-models-menu");
  menu.querySelectorAll(".dropdown-text").forEach((span) => {
    const match = span.textContent.match(/^(.*\S)\s+NEW$/);
    if (!match) return;
    span.textContent = match[1];
    const badge = document.createElement("span");
    badge.className = "mfm-badge";
    badge.textContent = "New";
    span.after(badge);
  });
  menu.querySelectorAll("a.dropdown-item").forEach((link) => {
    if (link.host === window.location.host) return;
    link.classList.add("is-external");
    link.target = "_blank";
    link.rel = "noopener";
  });

  const wide = window.matchMedia("(min-width: 992px) and (hover: hover)");
  toggle.setAttribute("href", all.getAttribute("href"));

  // Capture phase, so Bootstrap's delegated click handler never sees it.
  toggle.addEventListener(
    "click",
    (event) => {
      if (!wide.matches) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      window.location.href = all.href;
    },
    true,
  );

  const dropdown = () => window.bootstrap && window.bootstrap.Dropdown.getOrCreateInstance(toggle);
  let closing = 0;
  item.addEventListener("mouseenter", () => {
    if (!wide.matches) return;
    window.clearTimeout(closing);
    dropdown()?.show();
    // Bootstrap focuses the toggle on show; a pointer hover should not leave a focus ring.
    toggle.blur();
  });
  item.addEventListener("mouseleave", () => {
    if (!wide.matches) return;
    closing = window.setTimeout(() => dropdown()?.hide(), 180);
  });
})();
