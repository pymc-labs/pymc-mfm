/* Small additions to Quarto's own search (the overlay behind the navbar's magnifier button;
   no second search UI):
     - Ctrl/Cmd+K also opens it (Quarto handles "/").
     - Each result gets a plain path line ("Guide, Decompose a time series"); API and reference
       rows show the first sentence of the page description instead of the signature.
     - "No results" names the query and links to the Guide and the Python reference.
     - ARIA: the input is a combobox; the sheet's close button is "Close search".
   Paths resolve from Quarto's quarto:offset meta, so the site works under any folder. */
(() => {
  "use strict";

  const search = document.getElementById("quarto-search");
  if (!search) return;

  const SECTION_NAMES = {
    guide: "Guide",
    api: "HTTP API",
    reference: "Python reference",
    models: "Models",
    blog: "Blog",
  };
  const VERB_LINE = /^(GET|POST|PUT|PATCH|DELETE)\s+\//;

  const offset = document.querySelector('meta[name="quarto:offset"]')?.getAttribute("content") || "./";
  const siteRoot = new URL(offset, document.baseURI);

  let pages = null; // site-relative path without hash -> search.json entry

  function loadIndex() {
    fetch(new URL("search.json", siteRoot))
      .then((response) => response.json())
      .then((entries) => {
        pages = new Map();
        for (const entry of entries) {
          if (!entry.href.includes("#")) pages.set(entry.href, entry);
        }
        schedule();
      })
      .catch(() => {});
  }

  /* "guide/decompose.html#section" for any result link, wherever the site is mounted. */
  function pageKey(href) {
    const url = new URL(href, document.baseURI);
    const path = url.pathname.startsWith(siteRoot.pathname)
      ? url.pathname.slice(siteRoot.pathname.length)
      : url.pathname.replace(/^\//, "");
    return path + url.hash;
  }

  function sectionLabel(key) {
    return SECTION_NAMES[key.split("/")[0]] || "Docs";
  }

  function firstSentence(entry) {
    const lines = entry.text.split("\n").map((line) => line.trim()).filter(Boolean);
    const line = lines.find(
      (candidate) =>
        candidate.length > 30 &&
        candidate.includes(" ") &&
        !VERB_LINE.test(candidate) &&
        !candidate.startsWith(entry.title) &&
        !/[,(]$/.test(candidate) &&
        !/^\w+: /.test(candidate),
    );
    if (!line) return "";
    const cut = line.search(/\.(\s|$)/);
    return cut > 0 ? line.slice(0, cut + 1) : line;
  }

  function queryWords() {
    const input = document.querySelector(".aa-Form .aa-Input");
    return (input ? input.value : "").trim().split(/\s+/).filter(Boolean);
  }

  function highlighted(text, words) {
    const fragment = document.createDocumentFragment();
    if (!words.length) {
      fragment.append(text);
      return fragment;
    }
    const escaped = words.map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const pattern = new RegExp(`(${escaped.join("|")})`, "ig");
    text.split(pattern).forEach((part, index) => {
      if (index % 2) {
        const mark = document.createElement("mark");
        mark.className = "search-match";
        mark.textContent = part;
        fragment.append(mark);
      } else if (part) {
        fragment.append(part);
      }
    });
    return fragment;
  }

  /* Preact reuses DOM nodes when results change, so every pass sets the text again. */
  function upsert(host, className, create) {
    let el = host.querySelector(`:scope > .${className}`);
    if (!el) {
      el = create();
      el.className = className;
      host.append(el);
    }
    return el;
  }

  function decorateItem(item) {
    const link = item.querySelector("a.search-result-link");
    if (!link || !pages) {
      item.querySelectorAll(".mfm-search-path, .mfm-search-snippet").forEach((el) => el.remove());
      return;
    }
    const key = pageKey(link.getAttribute("href"));
    const page = pages.get(key.split("#")[0].split("?")[0]);
    const host = item.querySelector(".search-result-container") || link;

    item.querySelectorAll(".mfm-search-path").forEach((el) => el.parentElement !== host && el.remove());
    const label = `${sectionLabel(key)}, ${page ? page.title : ""}`.replace(/, $/, "");
    const path = upsert(host, "mfm-search-path", () => document.createElement("p"));
    if (path.textContent !== label) path.textContent = label;

    const isDocRow = Boolean(item.querySelector(".search-result-container"));
    const sentence = isDocRow && page && /^(api|reference)\//.test(key) ? firstSentence(page) : "";
    const existing = item.querySelector(".mfm-search-snippet");
    if (!sentence) {
      if (existing) existing.remove();
      return;
    }
    const original = item.querySelector(".search-result-text");
    if (!original) return;
    const snippet = existing || document.createElement("p");
    snippet.className = "mfm-search-snippet";
    const stamp = `${sentence}|${queryWords().join(" ")}`;
    if (snippet.dataset.text !== stamp) {
      snippet.dataset.text = stamp;
      snippet.replaceChildren(highlighted(sentence, queryWords()));
    }
    if (original.nextElementSibling !== snippet) original.after(snippet);
  }

  function decorateEmpty() {
    const stale = document.querySelector(".mfm-search-empty");
    const none = document.querySelector(".aa-SourceNoResults");
    // An empty field shows Quarto's hidden "no-query" line; nothing to replace then.
    if (!none || none.querySelector(".no-query")) {
      if (stale) stale.remove();
      return;
    }
    const block = stale || document.createElement("div");
    block.className = "mfm-search-empty";
    const message = `No results for “${queryWords().join(" ")}”.`;
    if (block.dataset.message !== message) {
      block.dataset.message = message;
      const text = document.createElement("p");
      text.textContent = message;
      const guide = document.createElement("a");
      guide.href = new URL("guide/index.html", siteRoot).href;
      guide.textContent = "Guide";
      const reference = document.createElement("a");
      reference.href = new URL("reference/index.html", siteRoot).href;
      reference.textContent = "Python reference";
      block.replaceChildren(text, guide, reference);
    }
    if (none.nextElementSibling !== block) none.after(block);
  }

  function setAttr(el, name, value) {
    if (el && el.getAttribute(name) !== value) el.setAttribute(name, value);
  }

  function fixAria() {
    const wrapper = search.querySelector(".aa-Autocomplete");
    const input = document.querySelector(".aa-Form .aa-Input");
    const list = document.querySelector(".aa-List");
    if (input) {
      input.removeAttribute("aria-labelledby");
      setAttr(input, "aria-label", "Search the docs");
      setAttr(input, "role", "combobox");
      setAttr(input, "aria-haspopup", "listbox");
      setAttr(input, "aria-expanded", list ? "true" : "false");
      if (list) setAttr(input, "aria-controls", list.id);
    }
    if (wrapper) {
      setAttr(wrapper, "role", "presentation");
      wrapper.removeAttribute("aria-expanded");
      wrapper.removeAttribute("aria-haspopup");
      wrapper.removeAttribute("aria-labelledby");
    }
    setAttr(list, "aria-label", "Search results");
    const close = document.querySelector(".aa-DetachedCancelButton");
    setAttr(close, "aria-label", "Close search");
    setAttr(close, "title", "Close search");
    setAttr(search.querySelector(".aa-DetachedSearchButton"), "aria-label", "Search");
  }

  function enhance() {
    document.querySelectorAll(".aa-Item").forEach(decorateItem);
    decorateEmpty();
    fixAria();
  }

  let queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      enhance();
    });
  }

  function isTyping(target) {
    if (!target) return false;
    const tag = target.tagName;
    return (
      target.isContentEditable ||
      tag === "TEXTAREA" ||
      tag === "SELECT" ||
      (tag === "INPUT" && !target.classList.contains("aa-Input"))
    );
  }

  document.addEventListener("keydown", (event) => {
    const isK = event.key === "k" || event.key === "K";
    if (isK && (event.ctrlKey || event.metaKey) && !isTyping(event.target)) {
      event.preventDefault();
      if (window.quartoOpenSearch) window.quartoOpenSearch();
    }
  });

  new MutationObserver(schedule).observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["aria-expanded", "aria-selected", "class"],
  });
  loadIndex();
  schedule();
})();
