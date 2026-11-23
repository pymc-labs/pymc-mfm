/* Shared behaviour for every page. Each init* function below sets up one piece of site chrome
   and does nothing when its markup is missing:
     initModelsMenu    navbar "Models" menu: hover open, "New" badge
     initConsoleButton marks the last right-hand navbar link with data-mfm="console" (site.css draws it)
     initExternalLinks marks navbar text links that leave the site (PyMC Labs) for their arrow
     initCopyButtons   <button class="mfm-copy" data-copy="TEXT" aria-label="..."> copies TEXT;
                       Quarto's own .code-copy-button copies code and is styled to match
     initCodeLabels    a language label with its logo on every code block (Python, curl, shell, JSON)
     initCodeScroll    marks blocks that scroll sideways, so site.css can fade the edge with more code
     initStackTables   in a ::: {.mfm-stack} div, gives each table cell its column name (data-label),
                       which site.css shows above the cell when the rows stack on a phone
     initTabs          a sliding marker and logos on every Quarto .panel-tabset; clears "Copied" on switch
     initPhoneMenu     below 992px, the navbar panel also lists the current section's sidebar pages
     initMenuSheet     below 992px the menu covers the page: scroll lock while open, Escape closes
     initPhoneToc      below 992px (site.css) a closed "On this page" list under the page title
   Search additions (labels, result paths, the empty message) are in search.js.
   Colours, icons and motion are in site.css; this file only adds classes, attributes and labels. */
(() => {
  "use strict";

  const COPIED_MS = 1600;

  /* Navbar "Models": one click to any model.
     On wide screens with a pointer, hovering "Models" opens its menu and clicking the label
     goes to the Models page; the menu items go straight to each model. Keyboard users open
     the menu with ArrowDown (Bootstrap's own handling). On the collapsed mobile navbar the
     label goes to the Models page as well.
     A menu item whose text ends in " NEW" shows a "New" badge instead. A text-only item
     (which Quarto renders as a header) ending in " SOON" becomes a greyed, disabled entry
     with a "Coming soon" badge: an announced model, with nowhere to go yet. */
  function initModelsMenu() {
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
    menu.querySelectorAll("li.dropdown-header").forEach((header) => {
      const match = header.textContent.trim().match(/^(.*\S)\s+SOON$/);
      if (!match) return;
      const entry = document.createElement("span");
      entry.className = "dropdown-item disabled is-soon";
      entry.setAttribute("aria-disabled", "true");
      const name = document.createElement("span");
      name.className = "dropdown-text";
      name.textContent = match[1];
      const badge = document.createElement("span");
      badge.className = "mfm-badge mfm-badge--soon";
      badge.textContent = "Coming soon";
      entry.append(name, badge);
      header.className = "";
      header.replaceChildren(entry);
    });
    menu.querySelectorAll("a.dropdown-item").forEach((link) => {
      if (link.host === window.location.host) return;
      link.classList.add("is-external");
      link.target = "_blank";
      link.rel = "noopener";
    });

    const wide = window.matchMedia("(min-width: 992px) and (hover: hover)");
    toggle.setAttribute("href", all.getAttribute("href"));

    // Capture phase, so Bootstrap's delegated click handler never sees it. A click or tap
    // always goes to the Models page (on a phone too: one model needs no menu inside the menu).
    toggle.addEventListener(
      "click",
      (event) => {
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
  }

  /* The right-hand navbar list ends with one link, the access page. The hook is this attribute,
     never the link's address, so the address can change without touching the styles. site.css
     draws it as a button with its icon and label at every width. */
  function initConsoleButton() {
    const links = document.querySelectorAll("#navbarCollapse .navbar-nav.ms-auto .nav-link");
    const link = links[links.length - 1];
    if (!link) return;
    link.dataset.mfm = "console";
  }

  /* Navbar text links to another host get .is-external (site.css adds an ArrowUpRight). By host,
     not by the href text: Quarto rewrites the site's own links to full addresses. Get access
     and icon links (GitHub) draw their own icons. */
  function initExternalLinks() {
    document.querySelectorAll("#navbarCollapse .navbar-nav > .nav-item > a.nav-link").forEach((link) => {
      if (link.dataset.mfm || link.querySelector(".bi") || link.host === window.location.host) return;
      link.classList.add("is-external");
    });
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise((resolve, reject) => {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.append(area);
      area.select();
      const done = document.execCommand("copy");
      area.remove();
      done ? resolve() : reject(new Error("copy failed"));
    });
  }

  function resetCopied(button) {
    window.clearTimeout(button.copiedTimer);
    button.classList.remove("is-copied", "code-copy-button-checked");
  }

  function initCopyButtons() {
    document.querySelectorAll("button.code-copy-button").forEach((button) => {
      button.setAttribute("aria-label", "Copy code");
    });
    document.addEventListener("click", (event) => {
      const button = event.target.closest("button.mfm-copy[data-copy]");
      if (!button) return;
      copyText(button.dataset.copy).then(() => {
        button.classList.add("is-copied");
        window.clearTimeout(button.copiedTimer);
        button.copiedTimer = window.setTimeout(() => resetCopied(button), COPIED_MS);
      }, () => {});
    });
  }

  const SHELL_LANGS = new Set(["bash", "sh", "shell", "zsh", "console"]);
  const SKIP_CLASSES = new Set(["sourceCode", "code-with-copy", "cell-code", "numberSource"]);
  const LANG_NAMES = { python: "Python", json: "JSON", yaml: "YAML", toml: "TOML", sql: "SQL", r: "R", html: "HTML", css: "CSS" };

  /* Returns { key, name } for a block, or null when it has no language to show.
     key picks the logo in site.css: python, curl, shell, json, or none. */
  function describeBlock(pre) {
    const lang = [...pre.classList].find((name) => !SKIP_CLASSES.has(name));
    if (!lang) return null;
    if (SHELL_LANGS.has(lang)) {
      const startsWithCurl = /^\s*curl\b/.test(pre.textContent);
      return startsWithCurl ? { key: "curl", name: "curl" } : { key: "shell", name: "Shell" };
    }
    const name = LANG_NAMES[lang] || lang.charAt(0).toUpperCase() + lang.slice(1);
    return { key: lang === "python" || lang === "json" ? lang : "none", name };
  }

  /* Tables that stack on phones (site.css): each body cell gets its column's header text. */
  function initStackTables() {
    document.querySelectorAll(".mfm-stack table").forEach((table) => {
      const names = [...table.querySelectorAll("thead th")].map((th) => th.textContent.trim());
      table.querySelectorAll("tbody tr").forEach((row) => {
        [...row.children].forEach((cell, index) => {
          if (names[index]) cell.dataset.label = names[index];
        });
      });
    });
  }

  function initCodeLabels() {
    document.querySelectorAll("div.sourceCode").forEach((box) => {
      const pre = box.querySelector(":scope > pre");
      if (!pre || box.querySelector(":scope > .mfm-code-label")) return;
      box.classList.add("mfm-code");
      // Inside a tab set the tab already names the language.
      if (box.closest(".panel-tabset")) return;
      const info = describeBlock(pre);
      if (!info) return;
      const label = document.createElement("div");
      label.className = "mfm-code-label";
      label.dataset.lang = info.key;
      const mark = document.createElement("span");
      mark.className = "mfm-lang-mark";
      mark.setAttribute("aria-hidden", "true");
      const name = document.createElement("span");
      name.textContent = info.name;
      label.append(mark, name);
      box.prepend(label);
    });
  }

  /* Long lines scroll sideways. Classes on the block tell site.css where more code lies. A pre that
     scrolls is made focusable, so keyboard users can scroll it too. */
  function initCodeScroll() {
    const update = (box, pre) => {
      const room = pre.scrollWidth - pre.clientWidth;
      box.classList.toggle("can-scroll-left", pre.scrollLeft > 1);
      box.classList.toggle("can-scroll-right", pre.scrollLeft < room - 1);
      if (room > 1) pre.tabIndex = 0;
      else pre.removeAttribute("tabindex");
    };
    document.querySelectorAll("div.sourceCode").forEach((box) => {
      const pre = box.querySelector(":scope > pre");
      if (!pre) return;
      const refresh = () => update(box, pre);
      pre.addEventListener("scroll", refresh, { passive: true });
      if (window.ResizeObserver) new ResizeObserver(refresh).observe(pre);
      refresh();
    });
  }

  const TAB_LANGS = { python: "python", curl: "curl", shell: "shell", bash: "shell", json: "json" };

  function initTabs() {
    document.querySelectorAll(".panel-tabset > ul.nav-tabs").forEach((tabs) => {
      const mark = document.createElement("span");
      mark.className = "mfm-tab-mark";
      mark.setAttribute("aria-hidden", "true");
      tabs.append(mark);
      tabs.querySelectorAll(".nav-link").forEach((link) => {
        const lang = TAB_LANGS[link.textContent.trim().toLowerCase()];
        if (lang) link.dataset.lang = lang;
      });
      const place = () => {
        const active = tabs.querySelector(".nav-link.active");
        if (!active) return;
        mark.style.width = `${active.offsetWidth}px`;
        mark.style.transform = `translateX(${active.offsetLeft}px)`;
      };
      tabs.addEventListener("shown.bs.tab", () => {
        place();
        tabs.closest(".panel-tabset").querySelectorAll(".mfm-copy, .code-copy-button").forEach(resetCopied);
      });
      window.addEventListener("resize", place);
      place();
      // Place the marker before the first paint of its transition, then allow it to slide.
      window.requestAnimationFrame(() => tabs.classList.add("is-ready"));
      if (document.fonts) document.fonts.ready.then(place);
    });
  }

  function initPhoneMenu() {
    const panel = document.getElementById("navbarCollapse");
    const side = document.querySelector("#quarto-sidebar .sidebar-menu-container");
    if (!panel || !side) return;
    const nav = document.createElement("nav");
    nav.className = "mfm-phone-pages";
    nav.setAttribute("aria-label", "Pages in this section");
    const copy = side.cloneNode(true);
    // The clone is a plain list: no duplicated ids, and no Bootstrap toggles that would
    // reach the originals in the desktop sidebar.
    copy.querySelectorAll("[id]").forEach((el) => el.removeAttribute("id"));
    copy.querySelectorAll("[data-bs-toggle], [data-bs-target]").forEach((el) => {
      el.removeAttribute("data-bs-toggle");
      el.removeAttribute("data-bs-target");
    });
    copy.querySelectorAll(".sidebar-item-toggle").forEach((el) => el.remove());
    copy.querySelectorAll(".sidebar-section").forEach((el) => el.classList.add("show"));
    nav.append(copy);
    nav.addEventListener("click", (event) => {
      // A link to this same page (a section anchor) leaves the panel open otherwise.
      const link = event.target.closest("a[href]");
      if (link && link.pathname === window.location.pathname && window.bootstrap) {
        window.bootstrap.Collapse.getOrCreateInstance(panel, { toggle: false }).hide();
      }
    });
    panel.append(nav);

  }

  /* Below 992px the menu is a sheet over the page (site.css): lock the page scroll while it
     is open, and close it on Escape. */
  function initMenuSheet() {
    const panel = document.getElementById("navbarCollapse");
    if (!panel) return;
    const root = document.documentElement;
    panel.addEventListener("show.bs.collapse", () => root.classList.add("mfm-menu-open"));
    panel.addEventListener("hidden.bs.collapse", () => root.classList.remove("mfm-menu-open"));
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || !panel.classList.contains("show") || !window.bootstrap) return;
      window.bootstrap.Collapse.getOrCreateInstance(panel, { toggle: false }).hide();
      document.querySelector(".navbar-toggler")?.focus();
    });
    window.matchMedia("(min-width: 992px)").addEventListener("change", (event) => {
      if (event.matches) root.classList.remove("mfm-menu-open");
    });
  }

  /* The margin list "On this page" is hidden below 992px. Copy it into a closed <details> under
     the page title; site.css shows the copy only below 992px. A page with fewer than two
     entries gets none. */
  function initPhoneToc() {
    let list = document.querySelector("#TOC > ul");
    const main = document.getElementById("quarto-document-content");
    if (!list || !main) return;
    // A reference page has one top entry, the page itself: list its parts.
    const parts = list.children.length === 1 && list.firstElementChild.querySelector(":scope > ul");
    if (parts) list = parts;
    if (list.querySelectorAll("a").length < 2) return;
    // The title block on a guide page; on a reference page the heading is part of the text.
    const title = main.querySelector("#title-block-header .quarto-title") || main.querySelector("h1");
    if (!title) return;
    const toc = document.createElement("details");
    toc.className = "mfm-phone-toc";
    const summary = document.createElement("summary");
    summary.textContent = document.getElementById("toc-title")?.textContent || "On this page";
    const copy = list.cloneNode(true);
    // No duplicated ids, no classes that Quarto's scroll tracking looks for or that fold the
    // nested lists, and no-external, so Quarto does not open these in-page links in a new tab.
    copy.querySelectorAll("[id]").forEach((el) => el.removeAttribute("id"));
    [copy, ...copy.querySelectorAll(".collapse")].forEach((el) => el.classList.remove("collapse"));
    copy.querySelectorAll("a").forEach((a) => {
      a.className = "no-external";
      a.removeAttribute("data-scroll-target");
    });
    toc.append(summary, copy);
    // Close the list after a tap, so the heading is on screen.
    copy.addEventListener("click", (event) => {
      if (event.target.closest("a")) toc.open = false;
    });
    (title.closest("#title-block-header") || title).after(toc);
  }

  [
    initModelsMenu,
    initConsoleButton,
    initExternalLinks,
    initCopyButtons,
    initCodeLabels,
    initCodeScroll,
    initStackTables,
    initTabs,
    initPhoneMenu,
    initMenuSheet,
    initPhoneToc,
  ].forEach((init) => init());
})();
