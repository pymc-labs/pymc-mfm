(() => {
  "use strict";

  const PREFIX = "pymc-mfm:nov17th-webinar:";
  const STATE_KEY = `${PREFIX}state`;
  const COMMAND_KEY = `${PREFIX}command`;
  const TIMER_KEY = `${PREFIX}timer`;
  const NOTES_PREFIX = `${PREFIX}notes:`;
  const AUDIENCE_NAME = "pymc-mfm-nov17th-webinar-audience";
  const PRESENTER_NAME = "pymc-mfm-nov17th-webinar-presenter";
  const HEARTBEAT_MS = 3000;
  const AUDIENCE_TIMEOUT_MS = 8500;
  const SAVE_DELAY_MS = 300;
  const clientId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  const openedAt = performance.now();
  const notesBySlide = new Map();
  const ui = {
    connectionLabel: document.querySelector("#presenter-connection-label"),
    connectionDetail: document.querySelector("#presenter-connection-detail"),
    position: document.querySelector("#presenter-position"),
    currentTitle: document.querySelector("#presenter-current-title"),
    currentSummary: document.querySelector("#presenter-current-summary"),
    nextPosition: document.querySelector("#presenter-next-position"),
    nextTitle: document.querySelector("#presenter-next-title"),
    nextSummary: document.querySelector("#presenter-next-summary"),
    previewEmpty: document.querySelector("#presenter-preview-empty"),
    previewCaption: document.querySelector("#presenter-preview-caption"),
    notes: document.querySelector("#presenter-notes-input"),
    notesStatus: document.querySelector("#presenter-notes-status"),
    saveNotes: document.querySelector('[data-presenter-action="save-notes"]'),
    previous: document.querySelector('[data-presenter-action="prev"]'),
    next: document.querySelector('[data-presenter-action="next"]'),
    timer: document.querySelector(".presenter-timer"),
    elapsed: document.querySelector("#presenter-elapsed"),
    timerState: document.querySelector("#presenter-timer-state"),
    timerToggle: document.querySelector('[data-presenter-action="timer-toggle"]'),
    timerReset: document.querySelector('[data-presenter-action="timer-reset"]'),
    storageWarning: document.querySelector("#presenter-storage-warning"),
    feedback: document.querySelector("#presenter-feedback"),
  };

  let channel = null;
  let snapshot = null;
  let stateSignature = "";
  let activeIndex = null;
  let pendingNotes = null;
  let saveTimeout = null;
  let audienceWindow = null;
  let lastLiveAt = null;
  let audienceConnected = false;
  let storageUnavailable = false;
  let messageNumber = 0;
  const timer = { elapsedMs: 0, running: false, anchor: performance.now() };

  window.name = PRESENTER_NAME;

  function setText(element, text) {
    if (element.textContent !== text) element.textContent = text;
  }

  function nextMessageId() {
    messageNumber += 1;
    return `${clientId}-${messageNumber}`;
  }

  function parseJSON(value) {
    if (typeof value !== "string") return null;
    try {
      return JSON.parse(value);
    } catch {
      return null;
    }
  }

  function storageFailed() {
    storageUnavailable = true;
    ui.storageWarning.hidden = false;
  }

  function readStored(key) {
    try {
      return parseJSON(localStorage.getItem(key));
    } catch {
      storageFailed();
      return null;
    }
  }

  function writeStored(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch {
      storageFailed();
      return false;
    }
  }

  function isParagraphs(value) {
    return Array.isArray(value) && value.every((paragraph) => typeof paragraph === "string");
  }

  function normalizeState(value) {
    if (!value || value.type !== "state" || !Array.isArray(value.slides) || !value.slides.length) return null;
    if (value.slides.some((slide) => !slide || typeof slide !== "object")) return null;
    const slides = value.slides.map((slide, index) => ({
      title: typeof slide.title === "string" && slide.title ? slide.title : `Slide ${index + 1}`,
      summary: typeof slide.summary === "string" ? slide.summary : "",
      notes: isParagraphs(slide.notes) ? slide.notes : [],
      theme: slide.theme === "light" ? "light" : "dark",
      steps: Number.isSafeInteger(slide.steps) && slide.steps > 0 ? slide.steps : 0,
    }));
    const index = Number.isFinite(value.index) ? Math.max(0, Math.min(slides.length - 1, Math.trunc(value.index))) : 0;
    const step = Number.isFinite(value.step) ? Math.max(0, Math.min(slides[index].steps, Math.trunc(value.step))) : 0;
    // The deck sends the next cue: the next build step of this slide, or the next slide.
    let next = null;
    const cue = value.next;
    if (cue && typeof cue === "object" && Number.isFinite(cue.index) && cue.index >= 0 && cue.index < slides.length) {
      next = {
        index: Math.trunc(cue.index),
        step: Number.isFinite(cue.step) ? Math.max(0, Math.trunc(cue.step)) : 0,
        title: typeof cue.title === "string" && cue.title ? cue.title : slides[Math.trunc(cue.index)].title,
        summary: typeof cue.summary === "string" ? cue.summary : "",
      };
    }
    // A state saved before build steps has no cue at all (unlike `null` at the end): the next slide follows.
    if (cue === undefined && index + 1 < slides.length) {
      next = { index: index + 1, step: 0, title: slides[index + 1].title, summary: slides[index + 1].summary };
    }
    return { type: "state", index, step, total: slides.length, slides, next };
  }

  function send(message) {
    if (channel) {
      try {
        channel.postMessage(message);
        return true;
      } catch {
        channel.close();
        channel = null;
      }
    }
    // A fresh id also makes repeated storage-only requests observable.
    return writeStored(COMMAND_KEY, { ...message, id: message.id || nextMessageId() });
  }

  function requestState() {
    send({ type: "request-state" });
  }

  function updateNavigation() {
    ui.previous.disabled = !audienceConnected || !snapshot || (snapshot.index === 0 && snapshot.step === 0);
    ui.next.disabled = !audienceConnected || !snapshot || !snapshot.next;
  }

  function refreshConnection() {
    const now = performance.now();
    const connected = lastLiveAt !== null && now - lastLiveAt < AUDIENCE_TIMEOUT_MS;
    const connecting = !connected && lastLiveAt === null && now - openedAt < 1500;
    if (connected && !audienceConnected) setText(ui.feedback, "");
    audienceConnected = connected;
    document.body.dataset.audience = connected ? "live" : connecting ? "connecting" : "offline";
    if (connected) {
      setText(ui.connectionLabel, "Audience connected");
      setText(ui.connectionDetail, "The preview follows the audience window.");
      setText(ui.previewCaption, "Live audience slide.");
    } else if (connecting) {
      setText(ui.connectionLabel, "Connecting to audience");
      setText(ui.connectionDetail, "Looking for an open audience window.");
    } else {
      setText(ui.connectionLabel, "No audience detected");
      setText(ui.connectionDetail, snapshot
        ? "Showing your last saved slide. Open the audience deck to navigate."
        : "Open the audience deck to load slides and start presenting.");
      setText(ui.previewCaption, snapshot ? "Last saved audience slide; no audience is connected." : "Waiting for an audience slide.");
    }
    updateNavigation();
  }

  function notesText(index) {
    if (notesBySlide.has(index)) return notesBySlide.get(index);
    const saved = readStored(`${NOTES_PREFIX}${index}`);
    if (isParagraphs(saved)) {
      const text = saved.join("\n\n");
      notesBySlide.set(index, text);
      return text;
    }
    return snapshot.slides[index].notes.join("\n\n");
  }

  function renderNotesStatus() {
    if (activeIndex === null) {
      setText(ui.notesStatus, "Open the audience deck to load notes.");
    } else if (pendingNotes && pendingNotes.index === activeIndex) {
      setText(ui.notesStatus, "Saving changes…");
    } else if (storageUnavailable) {
      setText(ui.notesStatus, "Session only — browser storage is unavailable.");
    } else {
      setText(ui.notesStatus, notesBySlide.has(activeIndex) ? "Saved in this browser" : "Changes save automatically");
    }
  }

  function renderNotes(force = false) {
    ui.notes.disabled = !snapshot;
    ui.saveNotes.disabled = !snapshot;
    if (snapshot && (force || document.activeElement !== ui.notes)) {
      const text = notesText(activeIndex);
      if (ui.notes.value !== text) ui.notes.value = text;
    }
    renderNotesStatus();
  }

  function saveNotes(explicit = false) {
    if (saveTimeout !== null) {
      clearTimeout(saveTimeout);
      saveTimeout = null;
    }
    if (!pendingNotes && explicit && activeIndex !== null) {
      pendingNotes = { index: activeIndex, text: ui.notes.value };
    }
    if (!pendingNotes) return;
    // Keep the slide id with the draft: audience navigation may arrive before debounce.
    const { index, text } = pendingNotes;
    pendingNotes = null;
    notesBySlide.set(index, text);
    // Splitting only at double newlines makes the plain-text round trip lossless.
    const paragraphs = text === "" ? [] : text.split("\n\n");
    const persisted = writeStored(`${NOTES_PREFIX}${index}`, paragraphs);
    send({ type: "notes", index, paragraphs });
    renderNotesStatus();
    if (explicit) {
      setText(ui.feedback, persisted ? "Notes saved in this browser." : "Notes kept for this session; browser storage is unavailable.");
    }
  }

  function receiveState(value, live) {
    const next = normalizeState(value);
    if (!next) return;
    if (live) lastLiveAt = performance.now();
    const signature = JSON.stringify(next);
    if (signature !== stateSignature) {
      const changedSlide = activeIndex !== next.index;
      if (changedSlide) saveNotes();
      snapshot = next;
      activeIndex = next.index;
      stateSignature = signature;
      const current = next.slides[next.index];
      const upcoming = next.next;
      const stepLabel = current.steps ? ` · step ${next.step} of ${current.steps}` : "";
      setText(ui.position, `${next.index + 1} / ${next.total}${stepLabel}`);
      setText(ui.currentTitle, current.title);
      setText(ui.currentSummary, current.summary);
      setText(ui.nextPosition, upcoming ? `${upcoming.index + 1} / ${next.total}${upcoming.step ? ` · step ${upcoming.step}` : ""}` : "");
      setText(ui.nextTitle, upcoming ? upcoming.title : "End of deck");
      setText(ui.nextSummary, upcoming ? upcoming.summary : "This is the last slide.");
      ui.previewEmpty.hidden = true;
      document.title = `Slide ${next.index + 1} of ${next.total} · Presenter — PyMC MFM`;
      renderNotes(changedSlide);
    }
    refreshConnection();
  }

  function navigate(command) {
    saveNotes();
    refreshConnection();
    if (!snapshot || !audienceConnected) {
      setText(ui.feedback, "Open the audience deck before navigating.");
      return;
    }
    if (!send({ type: "command", command, id: nextMessageId() })) {
      setText(ui.feedback, "Cannot reach the audience. Allow site storage or reopen this screen from the audience deck.");
    }
  }

  function findAudienceWindow() {
    if (audienceWindow && !audienceWindow.closed) return audienceWindow;
    try {
      if (!window.opener || window.opener.closed) return null;
      const url = new URL(window.opener.location.href);
      const audienceURL = new URL("index.html", location.href);
      const directory = audienceURL.pathname.slice(0, -"index.html".length);
      if (url.origin === audienceURL.origin && (url.pathname === audienceURL.pathname || url.pathname === directory) && url.searchParams.get("mirror") !== "1") {
        audienceWindow = window.opener;
        return audienceWindow;
      }
    } catch {
      // A separately opened presenter may have an unrelated, cross-origin opener.
    }
    return null;
  }

  function openAudience(event) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    saveNotes();
    try {
      audienceWindow = findAudienceWindow() || window.open("index.html", AUDIENCE_NAME);
      if (!audienceWindow) {
        setText(ui.feedback, "The audience window was blocked. Allow pop-ups, or open the audience link in a new tab.");
        return;
      }
      audienceWindow.focus();
      setText(ui.feedback, audienceConnected ? "Audience window focused." : "Audience window opened. Waiting for its first slide…");
      requestState();
    } catch {
      setText(ui.feedback, "Could not open the audience window. Open the audience link in a new tab.");
    }
  }

  function elapsedAt(now = performance.now()) {
    return timer.elapsedMs + (timer.running ? Math.max(0, now - timer.anchor) : 0);
  }

  function restoreTimer() {
    const saved = readStored(TIMER_KEY);
    if (!saved || saved.version !== 1 || !Number.isFinite(saved.elapsedMs) || saved.elapsedMs < 0 || saved.elapsedMs > Number.MAX_SAFE_INTEGER) return;
    timer.elapsedMs = saved.elapsedMs;
    const now = performance.now();
    timer.anchor = now;
    if (saved.running === true && Number.isFinite(saved.savedAt) && saved.savedAt > 0) {
      // The monotonic epoch bridges reloads without using the adjustable wall clock.
      timer.elapsedMs += Math.max(0, performance.timeOrigin + now - saved.savedAt);
      timer.running = true;
    }
  }

  function persistTimer() {
    const now = performance.now();
    writeStored(TIMER_KEY, {
      version: 1,
      elapsedMs: elapsedAt(now),
      running: timer.running,
      savedAt: performance.timeOrigin + now,
    });
  }

  function renderTimer() {
    const elapsed = elapsedAt();
    const seconds = Math.floor(elapsed / 1000);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor(seconds / 60) % 60;
    const remainingSeconds = seconds % 60;
    const pair = (value) => String(value).padStart(2, "0");
    const display = hours > 0 ? `${pair(hours)}:${pair(minutes)}:${pair(remainingSeconds)}` : `${pair(minutes)}:${pair(remainingSeconds)}`;
    setText(ui.elapsed, display);
    setText(ui.timerState, timer.running ? "Running" : elapsed > 0 ? "Paused" : "Ready");
    setText(ui.timerToggle, timer.running ? "Pause timer" : elapsed > 0 ? "Resume timer" : "Start timer");
    ui.timerReset.disabled = !timer.running && elapsed === 0;
    ui.timer.dataset.running = String(timer.running);
  }

  function toggleTimer() {
    const now = performance.now();
    timer.elapsedMs = elapsedAt(now);
    timer.anchor = now;
    timer.running = !timer.running;
    persistTimer();
    renderTimer();
  }

  function resetTimer() {
    timer.elapsedMs = 0;
    timer.running = false;
    timer.anchor = performance.now();
    persistTimer();
    renderTimer();
  }

  function handleKeyboard(event) {
    if (event.defaultPrevented || event.isComposing) return;
    const target = event.target instanceof Element ? event.target : null;
    const editable = target && target.closest('textarea, input, select, [contenteditable]:not([contenteditable="false"]), [role="textbox"]');
    if ((event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveNotes(true);
      return;
    }
    if (event.key === "Escape" && editable) {
      event.preventDefault();
      if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
      return;
    }
    if (editable || event.metaKey || event.ctrlKey || event.altKey) return;
    if ((event.key === " " || event.key === "Enter") && target && target.closest('button, a, summary, [role="button"]')) return;
    let command = null;
    switch (event.key) {
      case "ArrowLeft":
      case "ArrowUp":
      case "PageUp":
      case "Backspace":
        command = "prev";
        break;
      case "ArrowRight":
      case "ArrowDown":
      case "PageDown":
      case "Enter":
        command = "next";
        break;
      case " ":
        command = event.shiftKey ? "prev" : "next";
        break;
      case "Home":
        command = "first";
        break;
      case "End":
        command = "last";
        break;
      default:
        return;
    }
    event.preventDefault();
    navigate(command);
  }

  try {
    if (typeof BroadcastChannel !== "undefined") {
      channel = new BroadcastChannel(`${PREFIX}sync`);
      channel.addEventListener("message", (event) => receiveState(event.data, true));
    }
  } catch {
    channel = null;
  }

  window.addEventListener("storage", (event) => {
    if (event.key === STATE_KEY && event.newValue !== null) {
      receiveState(parseJSON(event.newValue), true);
    } else if (event.key && event.key.startsWith(NOTES_PREFIX) && event.newValue !== null) {
      const suffix = event.key.slice(NOTES_PREFIX.length);
      if (!/^\d+$/.test(suffix)) return;
      const index = Number(suffix);
      const paragraphs = parseJSON(event.newValue);
      if (!Number.isSafeInteger(index) || (snapshot && index >= snapshot.total) || !isParagraphs(paragraphs)) return;
      if (pendingNotes && pendingNotes.index === index) return;
      notesBySlide.set(index, paragraphs.join("\n\n"));
      if (index === activeIndex) renderNotes();
    }
  });

  ui.notes.addEventListener("input", () => {
    if (activeIndex === null) return;
    pendingNotes = { index: activeIndex, text: ui.notes.value };
    notesBySlide.set(activeIndex, ui.notes.value);
    clearTimeout(saveTimeout);
    saveTimeout = setTimeout(() => saveNotes(), SAVE_DELAY_MS);
    renderNotesStatus();
  });
  ui.notes.addEventListener("blur", () => {
    saveNotes();
    renderNotes(true);
  });
  ui.saveNotes.addEventListener("click", () => saveNotes(true));
  ui.previous.addEventListener("click", () => navigate("prev"));
  ui.next.addEventListener("click", () => navigate("next"));
  ui.timerToggle.addEventListener("click", toggleTimer);
  ui.timerReset.addEventListener("click", resetTimer);
  document.querySelectorAll('[data-presenter-action="open-audience"]').forEach((link) => link.addEventListener("click", openAudience));
  document.addEventListener("keydown", handleKeyboard);
  window.addEventListener("pagehide", () => {
    saveNotes();
    persistTimer();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      saveNotes();
      persistTimer();
    } else {
      requestState();
      refreshConnection();
      renderTimer();
    }
  });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) {
      requestState();
      refreshConnection();
      renderTimer();
    }
  });

  receiveState(readStored(STATE_KEY), false);
  restoreTimer();
  renderTimer();
  renderNotes();
  refreshConnection();
  requestState();
  setTimeout(refreshConnection, 1600);
  setInterval(() => {
    requestState();
    refreshConnection();
  }, HEARTBEAT_MS);
  setInterval(renderTimer, 250);
})();
