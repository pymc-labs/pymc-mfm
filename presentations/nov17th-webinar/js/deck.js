(() => {
  'use strict';

  const PREFIX = 'pymc-mfm:nov17th-webinar:';
  const STATE_KEY = `${PREFIX}state`;
  const COMMAND_KEY = `${PREFIX}command`;
  const NOTES_PREFIX = `${PREFIX}notes:`;
  const PRESENTER_NAME = 'pymc-mfm-nov17th-webinar-presenter';
  const AUDIENCE_NAME = 'pymc-mfm-nov17th-webinar-audience';
  const FOCUSABLE = 'button:not([disabled]), a[href], input:not([disabled]), textarea:not([disabled]), select:not([disabled]), summary, [contenteditable="true"], [tabindex]:not([tabindex="-1"])';
  const INTERACTIVE = 'button, a[href], input, textarea, select, summary, [role="button"], [role="textbox"], [role="slider"], [role="combobox"]';
  const EDITABLE = 'input, textarea, select, [role="textbox"], [role="slider"], [role="combobox"], [role="listbox"]';

  function start() {
    const deck = document.querySelector('#deck');
    const slides = deck ? Array.from(deck.querySelectorAll('section.slide')) : [];
    const status = document.querySelector('#status');
    if (!deck || !slides.length) {
      if (status) status.textContent = 'The presentation could not find its slides.';
      return;
    }

    const mirror = new URLSearchParams(window.location.search).get('mirror') === '1';
    const actions = Array.from(document.querySelectorAll('[data-action]'));
    const count = document.querySelector('#slide-count');
    const progress = document.querySelector('#progress');
    const overview = document.querySelector('#overview');
    const overviewGrid = document.querySelector('#overview-grid');
    const notesPanel = document.querySelector('#notes-panel');
    const notesText = document.querySelector('#notes-text');
    const dialogs = [overview, notesPanel].filter(Boolean);
    const overviewButtons = [];
    const originalNotes = slides.map((slide) => Array.from(slide.querySelectorAll('aside.notes p'), (paragraph) => paragraph.textContent.trim()));
    const stepElements = slides.map((slide) => Array.from(slide.querySelectorAll('[data-step], [data-step-only]')));
    const records = slides.map((slide, index) => ({
      id: slide.id || '',
      title: slide.dataset.title || `Slide ${index + 1}`,
      summary: slide.dataset.summary || '',
      theme: slide.dataset.theme === 'light' ? 'light' : 'dark',
      notes: originalNotes[index].slice(),
      steps: stepElements[index].reduce((highest, element) => Math.max(highest, stepNumber(element)), 0),
    }));
    slides.forEach((slide, position) => slide.classList.toggle('has-steps', records[position].steps > 0));
    const unknownUrlMessage = `Unknown slide URL. Use #/1 through #/${slides.length}, or a slide id.`;

    let index = 0;
    let step = 0;
    let snapshot;
    let channel = null;
    let storageUnavailable = false;
    let statusProblem = '';
    let activeDialog = null;
    let dialogReturnFocus = null;
    let dialogIsolation = [];
    let dialogUsesFallback = false;
    let blackout = false;
    let blackoutReturnFocus = null;
    let blackoutLayer = null;
    let zoomArmed = false;
    let zoomState = null;
    let presenterWindow = null;
    let requestSequence = 0;
    let resizeFrame = 0;
    const seenCommands = new Set();
    const initialDeckInert = deck.inert;
    const originalCursor = { value: deck.style.getPropertyValue('cursor'), priority: deck.style.getPropertyPriority('cursor') };

    if (status) {
      status.setAttribute('role', 'status');
      status.setAttribute('aria-live', 'polite');
      status.setAttribute('aria-atomic', 'true');
      status.textContent = '';
    }
    if (!deck.hasAttribute('tabindex')) deck.tabIndex = -1;
    document.body.classList.toggle('mirror', mirror);
    const announcement = mirror ? null : document.createElement('span');
    if (announcement) {
      announcement.setAttribute('role', 'status');
      announcement.setAttribute('aria-live', 'polite');
      announcement.setAttribute('aria-atomic', 'true');
      Object.assign(announcement.style, {
        position: 'absolute', width: '1px', height: '1px', padding: '0',
        margin: '-1px', overflow: 'hidden', clipPath: 'inset(50%)',
        whiteSpace: 'nowrap', border: '0',
      });
      document.body.append(announcement);
    }

    function announce(message) {
      if (announcement) announcement.textContent = message;
    }

    function refreshStatus() {
      const warning = storageUnavailable
        ? 'Browser storage is unavailable; reload recovery and cross-window sync may be limited.'
        : '';
      if (status) status.textContent = [statusProblem, warning].filter(Boolean).join(' ');
    }

    function showProblem(message) {
      statusProblem = message;
      refreshStatus();
    }

    function clearProblem() {
      statusProblem = '';
      refreshStatus();
    }

    function storageFailed() {
      if (storageUnavailable) return;
      storageUnavailable = true;
      refreshStatus();
    }

    function readStored(key) {
      try {
        return window.localStorage.getItem(key);
      } catch {
        storageFailed();
        return null;
      }
    }

    function writeStored(key, value) {
      try {
        window.localStorage.setItem(key, value);
        return true;
      } catch {
        storageFailed();
        return false;
      }
    }

    function removeStored(key) {
      try {
        window.localStorage.removeItem(key);
      } catch {
        storageFailed();
      }
    }

    function parse(value) {
      if (!value) return null;
      try {
        return JSON.parse(value);
      } catch {
        return null;
      }
    }

    function validParagraphs(value) {
      return Array.isArray(value) && value.every((paragraph) => typeof paragraph === 'string');
    }

    function sameParagraphs(left, right) {
      return left.length === right.length && left.every((paragraph, position) => paragraph === right[position]);
    }

    function clampIndex(value) {
      if (typeof value !== 'number' || !Number.isFinite(value)) return null;
      return Math.min(slides.length - 1, Math.max(0, Math.trunc(value)));
    }

    function stepNumber(element) {
      const raw = element.hasAttribute('data-step-only') ? element.dataset.stepOnly : element.dataset.step;
      return /^[1-9]\d*$/.test(raw || '') ? Number(raw) : 0;
    }

    function clampStep(position, value) {
      if (typeof value !== 'number' || !Number.isFinite(value)) return 0;
      return Math.min(records[position].steps, Math.max(0, Math.trunc(value)));
    }

    function stepCaption(position, target) {
      const element = stepElements[position].find((candidate) => stepNumber(candidate) === target && candidate.dataset.caption);
      return element ? element.dataset.caption : '';
    }

    // `#/N` and `#/N/STEP` address a slide and build step; `#slide-id` addresses a slide (agenda links).
    function hashTarget() {
      const numbered = /^#\/([1-9]\d*)(?:\/(\d+))?\/?$/.exec(window.location.hash);
      if (numbered) {
        const number = Number(numbered[1]);
        if (!Number.isSafeInteger(number)) return null;
        const position = clampIndex(number - 1);
        return { index: position, step: clampStep(position, numbered[2] === undefined ? 0 : Number(numbered[2])) };
      }
      const named = /^#([A-Za-z][\w-]*)$/.exec(window.location.hash);
      const position = named ? records.findIndex((record) => record.id === named[1]) : -1;
      return position < 0 ? null : { index: position, step: 0 };
    }

    function writeHash() {
      // A valid position replaces a mistyped address, so its warning must not outlive it.
      if (statusProblem === unknownUrlMessage) clearProblem();
      const hash = step ? `#/${index + 1}/${step}` : `#/${index + 1}`;
      if (window.location.hash === hash) return;
      try {
        window.history.replaceState(window.history.state, '', hash);
      } catch {
        try {
          window.location.hash = hash;
        } catch {
          showProblem('The slide changed, but this browser could not update its URL.');
        }
      }
    }

    function makeSnapshot() {
      const current = records[index];
      let next = null;
      if (step < current.steps) {
        next = { title: current.title, summary: stepCaption(index, step + 1) || `Build step ${step + 1} of ${current.steps}.`, index, step: step + 1 };
      } else if (records[index + 1]) {
        const following = records[index + 1];
        next = { title: following.title, summary: following.summary, index: index + 1, step: 0 };
      }
      return {
        type: 'state',
        index,
        step,
        steps: current.steps,
        slideId: current.id,
        total: records.length,
        title: current.title,
        summary: current.summary,
        theme: current.theme,
        notes: current.notes.slice(),
        next,
        slides: records.map((record) => ({
          id: record.id,
          title: record.title,
          summary: record.summary,
          notes: record.notes.slice(),
          theme: record.theme,
          steps: record.steps,
        })),
      };
    }

    function post(message) {
      if (!channel) return false;
      try {
        channel.postMessage(message);
        return true;
      } catch {
        try { channel.close(); } catch { /* Storage remains the transport. */ }
        channel = null;
        return false;
      }
    }

    function publishState(reply = false, forceStorageEvent = false) {
      if (mirror) return;
      const sent = post(snapshot);
      const serialized = JSON.stringify(snapshot);
      // An unchanged setItem does not emit a storage event. A fallback reply must.
      if (reply && (!sent || forceStorageEvent) && readStored(STATE_KEY) === serialized) removeStored(STATE_KEY);
      writeStored(STATE_KEY, serialized);
    }

    function requestState() {
      const message = { type: 'request-state' };
      if (post(message)) return;
      writeStored(COMMAND_KEY, JSON.stringify({
        ...message,
        id: `${Date.now()}-${Math.random().toString(36).slice(2)}-${++requestSequence}`,
      }));
    }

    function updateNotesPanel() {
      if (!notesText) return;
      const paragraphs = records[index].notes;
      const elements = (paragraphs.length ? paragraphs : ['No speaker notes for this slide.']).map((text) => {
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        return paragraph;
      });
      notesText.replaceChildren(...elements);
    }

    function fullscreenElement() {
      return document.fullscreenElement || document.webkitFullscreenElement;
    }

    function updateControls() {
      for (const control of actions) {
        const action = control.dataset.action;
        if (action === 'prev' || action === 'next') {
          const disabled = action === 'prev'
            ? index === 0 && step === 0
            : index === slides.length - 1 && step === records[index].steps;
          if ('disabled' in control) control.disabled = disabled;
          control.setAttribute('aria-disabled', String(disabled));
        } else if (action === 'overview' || action === 'notes') {
          const dialog = action === 'overview' ? overview : notesPanel;
          control.setAttribute('aria-expanded', String(Boolean(dialog && activeDialog === dialog)));
          if (dialog) control.setAttribute('aria-controls', dialog.id);
        } else if (action === 'blackout') {
          control.setAttribute('aria-pressed', String(blackout));
        } else if (action === 'fullscreen') {
          const full = Boolean(fullscreenElement());
          control.setAttribute('aria-pressed', String(full));
          control.setAttribute('aria-label', full ? 'Exit fullscreen' : 'Enter fullscreen');
        }
      }
    }

    function render(shouldAnnounce = true) {
      slides.forEach((slide, position) => {
        const active = position === index;
        slide.hidden = !active;
        slide.classList.toggle('is-active', active);
        slide.setAttribute('aria-hidden', String(!active));
        const reached = active ? step : 0;
        slide.dataset.currentStep = String(reached);
        for (const element of stepElements[position]) {
          const target = stepNumber(element);
          element.classList.toggle('is-built', element.hasAttribute('data-step-only') ? target === reached : target <= reached);
        }
      });
      document.body.dataset.theme = records[index].theme;
      if (count) {
        count.textContent = `${index + 1} / ${slides.length}`;
        count.setAttribute('aria-label', `Slide ${index + 1} of ${slides.length}: ${records[index].title}`);
      }
      if (progress) {
        progress.style.width = `${((index + 1) / slides.length) * 100}%`;
        progress.setAttribute('aria-valuemin', '1');
        progress.setAttribute('aria-valuemax', String(slides.length));
        progress.setAttribute('aria-valuenow', String(index + 1));
        progress.setAttribute('aria-valuetext', `Slide ${index + 1} of ${slides.length}`);
      }
      overviewButtons.forEach((button, position) => {
        if (position === index) button.setAttribute('aria-current', 'true');
        else button.removeAttribute('aria-current');
      });
      updateNotesPanel();
      updateControls();
      snapshot = makeSnapshot();
      window.dispatchEvent(new CustomEvent('webinar:slide', { detail: snapshot }));
      if (shouldAnnounce && !count) announce(`Slide ${index + 1} of ${slides.length}: ${records[index].title}.`);
    }

    function focusElement(element) {
      if (element && element !== document.body && element.isConnected && element.getClientRects().length && !element.closest('[hidden], [inert]')) {
        element.focus({ preventScroll: true });
      } else {
        deck.focus({ preventScroll: true });
      }
    }

    function focusableElements(dialog) {
      return Array.from(dialog.querySelectorAll(FOCUSABLE)).filter((element) => (
        element.getClientRects().length > 0 && !element.closest('[hidden], [inert]')
      ));
    }

    function restoreDialogIsolation() {
      for (const entry of dialogIsolation) {
        entry.element.inert = entry.inert;
        if (entry.ariaHidden === null) entry.element.removeAttribute('aria-hidden');
        else entry.element.setAttribute('aria-hidden', entry.ariaHidden);
      }
      dialogIsolation = [];
    }

    function isolateDialog(dialog) {
      let branch = dialog;
      while (branch.parentElement) {
        for (const sibling of branch.parentElement.children) {
          if (sibling === branch || sibling.tagName === 'SCRIPT' || sibling.tagName === 'STYLE') continue;
          dialogIsolation.push({ element: sibling, inert: sibling.inert, ariaHidden: sibling.getAttribute('aria-hidden') });
          sibling.inert = true;
          sibling.setAttribute('aria-hidden', 'true');
        }
        branch = branch.parentElement;
        if (branch === document.body) break;
      }
    }

    function closeDialog(restoreFocus = true) {
      if (!activeDialog) return;
      const dialog = activeDialog;
      const returnFocus = dialogReturnFocus;
      activeDialog = null;
      dialogReturnFocus = null;
      if (typeof dialog.close === 'function' && dialog.open) dialog.close();
      else dialog.removeAttribute('open');
      dialog.hidden = true;
      dialogUsesFallback = false;
      restoreDialogIsolation();
      updateControls();
      if (restoreFocus) focusElement(returnFocus);
    }

    function restoreStyle(property, saved) {
      if (saved.value) deck.style.setProperty(property, saved.value, saved.priority);
      else deck.style.removeProperty(property);
    }

    function setZoomArmed(armed, shouldAnnounce = true) {
      zoomArmed = armed;
      document.body.classList.toggle('is-zoom-armed', armed);
      if (armed) deck.style.cursor = 'zoom-in';
      else if (!zoomState) restoreStyle('cursor', originalCursor);
      if (armed && shouldAnnounce) announce('Zoom ready. Click a point on the slide, or press Z or Escape to cancel.');
    }

    function exitZoom(shouldAnnounce = true) {
      setZoomArmed(false, false);
      if (!zoomState) return;
      restoreStyle('transform', zoomState.transform);
      zoomState = null;
      restoreStyle('cursor', originalCursor);
      document.body.classList.remove('is-zoomed');
      if (shouldAnnounce) announce('Zoom closed.');
    }

    function zoomAt(clientX, clientY) {
      if (blackout || activeDialog || zoomState) return;
      const rectangle = deck.getBoundingClientRect();
      if (!rectangle.width || !rectangle.height) return;
      const computed = window.getComputedStyle(deck);
      const origin = computed.transformOrigin.split(' ');
      const x = ((clientX - rectangle.left) / rectangle.width) * deck.offsetWidth - (parseFloat(origin[0]) || 0);
      const y = ((clientY - rectangle.top) / rectangle.height) * deck.offsetHeight - (parseFloat(origin[1]) || 0);
      zoomState = {
        transform: { value: deck.style.getPropertyValue('transform'), priority: deck.style.getPropertyPriority('transform') },
      };
      setZoomArmed(false, false);
      const transform = computed.transform === 'none' ? '' : computed.transform;
      deck.style.setProperty('transform', `${transform} translate(${x}px, ${y}px) scale(2) translate(${-x}px, ${-y}px)`);
      deck.style.cursor = 'zoom-out';
      document.body.classList.add('is-zoomed');
      announce('Zoomed in. Click, or press Z or Escape, to restore the slide.');
    }

    function setBlackout(blank, shouldAnnounce = true, restoreFocus = true) {
      if (blackout === blank) return;
      if (blank) {
        closeDialog(false);
        exitZoom(false);
        blackoutReturnFocus = document.activeElement;
      }
      blackout = blank;
      document.body.classList.toggle('is-blackout', blank);
      blackoutLayer.hidden = !blank;
      blackoutLayer.style.display = blank ? 'block' : 'none';
      deck.inert = blank || initialDeckInert;
      updateControls();
      if (blank) blackoutLayer.focus({ preventScroll: true });
      else if (restoreFocus) focusElement(blackoutReturnFocus);
      if (shouldAnnounce) announce(blank ? 'Screen blanked. Press B, period or Escape, or click, to restore it.' : 'Presentation restored.');
    }

    function openDialog(dialog) {
      if (!dialog) {
        showProblem('This presentation overlay is unavailable.');
        return;
      }
      if (activeDialog === dialog) {
        closeDialog();
        return;
      }
      if (blackout) setBlackout(false, false);
      exitZoom(false);
      closeDialog(false);
      activeDialog = dialog;
      dialogReturnFocus = document.activeElement;
      dialog.hidden = false;
      dialog.setAttribute('aria-modal', 'true');
      if (!dialog.hasAttribute('aria-label') && !dialog.hasAttribute('aria-labelledby')) {
        dialog.setAttribute('aria-label', dialog === overview ? 'Slide overview' : 'Speaker notes');
      }
      if (!dialog.hasAttribute('tabindex')) dialog.tabIndex = -1;
      try {
        if (typeof dialog.showModal !== 'function') throw new Error('Native dialogs are unavailable.');
        dialog.showModal();
      } catch {
        dialogUsesFallback = true;
        dialog.setAttribute('open', '');
        dialog.setAttribute('role', 'dialog');
        isolateDialog(dialog);
      }
      updateControls();
      const current = dialog === overview ? overviewButtons[index] : null;
      (current || focusableElements(dialog)[0] || dialog).focus({ preventScroll: true });
    }

    function navigate(target, targetStep = 0) {
      const nextIndex = clampIndex(target);
      if (nextIndex === null || mirror) return;
      const nextStep = clampStep(nextIndex, targetStep);
      if (blackout) setBlackout(false, false);
      exitZoom(false);
      closeDialog(false);
      if (nextIndex === index && nextStep === step) {
        writeHash();
        return;
      }
      index = nextIndex;
      step = nextStep;
      writeHash();
      render();
      publishState();
    }

    // Next build step, then the next slide; back to the previous step, then the fully built previous slide.
    function advance() {
      if (step < records[index].steps) navigate(index, step + 1);
      else if (index < slides.length - 1) navigate(index + 1, 0);
    }

    function retreat() {
      if (step > 0) navigate(index, step - 1);
      else if (index > 0) navigate(index - 1, records[index - 1].steps);
    }

    function applyNotes(position, paragraphs, persist) {
      if (!Number.isInteger(position) || position < 0 || position >= records.length || !validParagraphs(paragraphs)) return;
      if (persist) writeStored(`${NOTES_PREFIX}${position}`, JSON.stringify(paragraphs));
      if (sameParagraphs(records[position].notes, paragraphs)) return;
      records[position].notes = paragraphs.slice();
      render(false);
      publishState();
      announce('Speaker notes updated.');
    }

    function receiveState(message) {
      if (!mirror || !message || message.type !== 'state') return;
      const nextIndex = clampIndex(message.index);
      if (nextIndex === null) return;
      const nextStep = clampStep(nextIndex, message.step);
      let changed = nextIndex !== index || nextStep !== step;
      if (Array.isArray(message.slides)) {
        records.forEach((record, position) => {
          const remote = message.slides[position];
          if (remote && validParagraphs(remote.notes) && !sameParagraphs(record.notes, remote.notes)) {
            record.notes = remote.notes.slice();
            changed = true;
          }
        });
      }
      if (validParagraphs(message.notes) && !sameParagraphs(records[nextIndex].notes, message.notes)) {
        records[nextIndex].notes = message.notes.slice();
        changed = true;
      }
      index = nextIndex;
      step = nextStep;
      if (changed || !snapshot) render();
    }

    function receive(message, fromStorage = false) {
      if (!message || typeof message !== 'object') return;
      if (message.type === 'state') {
        receiveState(message);
        return;
      }
      if (mirror) return;
      if (message.type === 'request-state') {
        publishState(true, fromStorage);
      } else if (message.type === 'notes') {
        applyNotes(message.index, message.paragraphs, true);
      } else if (message.type === 'command') {
        if (typeof message.id !== 'string' || !message.id || seenCommands.has(message.id)) return;
        let move;
        switch (message.command) {
          case 'next': move = advance; break;
          case 'prev': move = retreat; break;
          case 'first': move = () => navigate(0); break;
          case 'last': move = () => navigate(slides.length - 1); break;
          case 'goto': {
            const target = clampIndex(message.index);
            if (target === null) return;
            move = () => navigate(target);
            break;
          }
          default: return;
        }
        seenCommands.add(message.id);
        if (seenCommands.size > 128) seenCommands.delete(seenCommands.values().next().value);
        move();
      }
    }

    async function positionPresenter(popup) {
      if (typeof window.getScreenDetails !== 'function' || !navigator.permissions) return;
      try {
        const permission = await navigator.permissions.query({ name: 'window-management' });
        if (permission.state !== 'granted') return;
        const details = await window.getScreenDetails();
        const screen = details.screens.find((candidate) => candidate !== details.currentScreen);
        if (!screen || popup.closed) return;
        const width = Math.min(1100, screen.availWidth);
        const height = Math.min(760, screen.availHeight);
        popup.resizeTo(width, height);
        popup.moveTo(screen.availLeft + Math.round((screen.availWidth - width) / 2), screen.availTop + Math.round((screen.availHeight - height) / 2));
      } catch {
        // Screen permissions and window placement are optional browser capabilities.
      }
    }

    function presenterIsOpen() {
      try {
        return Boolean(presenterWindow && !presenterWindow.closed);
      } catch {
        return false;
      }
    }

    function openPresenter() {
      // Return keyboard focus to the deck before opening the presenter.
      focusElement(deck);
      if (!presenterIsOpen()) {
        const screen = window.screen;
        const width = Math.min(1100, screen.availWidth || 1100);
        const height = Math.min(760, screen.availHeight || 760);
        const left = (screen.availLeft || 0) + Math.max(0, Math.round(((screen.availWidth || width) - width) / 2));
        const top = (screen.availTop || 0) + Math.max(0, Math.round(((screen.availHeight || height) - height) / 2));
        try {
          // An empty URL reattaches to the named presenter after an audience reload.
          presenterWindow = window.open('', PRESENTER_NAME, `popup=yes,width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`);
          if (!presenterWindow) {
            showProblem('The presenter popup was blocked. Allow popups for this site, then press P again.');
            return false;
          }
          const url = new URL('presenter.html', window.location.href);
          let alreadyLoaded = false;
          try {
            const current = new URL(presenterWindow.location.href);
            alreadyLoaded = current.origin === url.origin && current.pathname === url.pathname;
          } catch { /* A different page in the named window must be replaced. */ }
          if (!alreadyLoaded) presenterWindow.location.href = url.href;
          void positionPresenter(presenterWindow);
        } catch {
          presenterWindow = null;
          showProblem('The presenter window could not be opened. Allow popups and press P again.');
          return false;
        }
      }
      try {
        presenterWindow.focus();
      } catch { /* Browsers may leave window focusing to the user. */ }
      clearProblem();
      announce('Presenter opened in a separate window.');
      return true;
    }

    function toggleFullscreen() {
      focusElement(deck);
      if (fullscreenElement()) {
        const exit = document.exitFullscreen || document.webkitExitFullscreen;
        if (!exit) {
          showProblem('Use your browser controls to leave fullscreen.');
          return;
        }
        try {
          Promise.resolve(exit.call(document)).then(clearProblem).catch(() => showProblem('Fullscreen could not be closed. Use your browser controls.'));
        } catch {
          showProblem('Fullscreen could not be closed. Use your browser controls.');
        }
        return;
      }
      const request = document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen;
      if (!request) {
        showProblem('Fullscreen is not available here. Use the browser fullscreen command; P opens the presenter.');
        return;
      }
      function failed() {
        focusElement(deck);
        showProblem('Fullscreen was blocked. Click the audience window and press F again, or use the browser fullscreen command.');
      }
      try {
        // Opening a popup consumes this gesture, so P stays separate from F.
        Promise.resolve(request.call(document.documentElement)).then(() => {
          updateControls();
          clearProblem();
          announce('Fullscreen requested. Press F to leave fullscreen.');
        }).catch(failed);
      } catch {
        failed();
      }
    }

    function isInteractive(target) {
      return target instanceof Element && (target.isContentEditable || Boolean(target.closest(INTERACTIVE)));
    }

    function isEditable(target) {
      return target instanceof Element && (target.isContentEditable || Boolean(target.closest(EDITABLE)));
    }

    function handleDialogKey(event) {
      if (event.key === 'Tab') {
        const elements = focusableElements(activeDialog);
        const first = elements[0];
        const last = elements[elements.length - 1];
        if (!first) {
          event.preventDefault();
          activeDialog.focus({ preventScroll: true });
        } else if (event.shiftKey && (document.activeElement === first || !activeDialog.contains(document.activeElement))) {
          event.preventDefault();
          last.focus({ preventScroll: true });
        } else if (!event.shiftKey && (document.activeElement === last || !activeDialog.contains(document.activeElement))) {
          event.preventDefault();
          first.focus({ preventScroll: true });
        }
        return;
      }
      if (activeDialog !== overview || event.altKey || event.ctrlKey || event.metaKey) return;
      const focused = overviewButtons.indexOf(document.activeElement);
      if (focused < 0) return;
      let destination;
      switch (event.key) {
        case 'ArrowRight': case 'ArrowDown': destination = (focused + 1) % overviewButtons.length; break;
        case 'ArrowLeft': case 'ArrowUp': destination = (focused - 1 + overviewButtons.length) % overviewButtons.length; break;
        case 'Home': destination = 0; break;
        case 'End': destination = overviewButtons.length - 1; break;
        default: return;
      }
      event.preventDefault();
      overviewButtons[destination].focus({ preventScroll: true });
    }

    function handleKey(event) {
      if (event.defaultPrevented || event.isComposing || event.ctrlKey || event.metaKey) return;
      if (event.key === 'Escape') {
        if (isEditable(event.target) && !activeDialog && !blackout && !zoomState && !zoomArmed) return;
        event.preventDefault();
        if (zoomState) exitZoom();
        else if (zoomArmed) setZoomArmed(false);
        else if (blackout) setBlackout(false);
        else if (activeDialog) closeDialog();
        else openDialog(overview);
        return;
      }
      if (activeDialog) {
        handleDialogKey(event);
        return;
      }
      if (blackout && event.key === 'Tab') {
        event.preventDefault();
        blackoutLayer.focus({ preventScroll: true });
        return;
      }
      if (event.altKey || isEditable(event.target)) return;
      const interactive = isInteractive(event.target);
      if (interactive && ['Enter', ' ', 'Spacebar'].includes(event.key)) return;
      let move;
      switch (event.key) {
        case 'ArrowRight': case 'ArrowDown': case 'PageDown': case 'Enter': move = advance; break;
        case ' ': case 'Spacebar': move = event.shiftKey ? retreat : advance; break;
        case 'ArrowLeft': case 'ArrowUp': case 'PageUp': case 'Backspace': move = retreat; break;
        case 'Home': move = () => navigate(0); break;
        case 'End': move = () => navigate(slides.length - 1); break;
        default: break;
      }
      if (move) {
        event.preventDefault();
        move();
        return;
      }
      if (interactive) return;
      const key = event.key.toLowerCase();
      if (!['f', 'p', 'n', 'o', 'b', '.', 'z'].includes(key)) return;
      event.preventDefault();
      if (event.repeat) return;
      switch (key) {
        case 'f': toggleFullscreen(); break;
        case 'p': openPresenter(); break;
        case 'n': openDialog(notesPanel); break;
        case 'o': openDialog(overview); break;
        case 'b': case '.': setBlackout(!blackout); break;
        case 'z':
          if (zoomState) exitZoom();
          else if (!blackout) setZoomArmed(!zoomArmed);
          break;
        default: break;
      }
    }

    function handleAction(event) {
      if (event.defaultPrevented || !(event.target instanceof Element)) return;
      const control = event.target.closest('[data-action]');
      if (!control || control.getAttribute('aria-disabled') === 'true' || control.disabled) return;
      const action = control.dataset.action;
      if (!['prev', 'next', 'fullscreen', 'presenter', 'overview', 'notes', 'blackout'].includes(action)) return;
      event.preventDefault();
      switch (action) {
        case 'prev': retreat(); break;
        case 'next': advance(); break;
        case 'fullscreen': toggleFullscreen(); break;
        case 'presenter': openPresenter(); break;
        case 'overview': openDialog(overview); break;
        case 'notes': openDialog(notesPanel); break;
        case 'blackout': setBlackout(!blackout); break;
        default: break;
      }
    }

    function resize() {
      resizeFrame = 0;
      exitZoom(false);
      const scale = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
      const style = document.documentElement.style;
      style.setProperty('--deck-scale', String(scale));
      style.setProperty('--deck-width', `${window.innerWidth / scale}px`);
      style.setProperty('--deck-height', `${window.innerHeight / scale}px`);
    }

    records.forEach((record, position) => {
      const override = parse(readStored(`${NOTES_PREFIX}${position}`));
      if (validParagraphs(override)) record.notes = override.slice();
    });

    dialogs.forEach((dialog) => {
      dialog.hidden = true;
      dialog.addEventListener('cancel', (event) => {
        event.preventDefault();
        if (activeDialog === dialog) closeDialog();
      });
      dialog.addEventListener('close', () => {
        if (activeDialog === dialog) closeDialog();
      });
      dialog.addEventListener('click', (event) => {
        if (event.target !== dialog || activeDialog !== dialog) return;
        const bounds = dialog.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) closeDialog();
      });
    });

    if (!mirror) {
      try { window.name = AUDIENCE_NAME; } catch { /* Naming is optional in sandboxed browsers. */ }
      if (overviewGrid) {
        const buttons = records.map((record, position) => {
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'overview-item';
          button.dataset.index = String(position);
          button.dataset.theme = record.theme;
          button.setAttribute('aria-label', `Go to slide ${position + 1}: ${record.title}`);
          const number = document.createElement('span');
          number.className = 'overview-number';
          number.textContent = String(position + 1).padStart(2, '0');
          const title = document.createElement('strong');
          title.textContent = record.title;
          const summary = document.createElement('p');
          summary.textContent = record.summary;
          button.append(number, title, summary);
          button.addEventListener('click', () => {
            closeDialog();
            navigate(position);
          });
          overviewButtons.push(button);
          return button;
        });
        overviewGrid.replaceChildren(...buttons);
      }
      blackoutLayer = document.createElement('div');
      blackoutLayer.className = 'blackout-overlay';
      blackoutLayer.hidden = true;
      blackoutLayer.tabIndex = -1;
      blackoutLayer.setAttribute('role', 'group');
      blackoutLayer.setAttribute('aria-label', 'Presentation blanked. Press B, period or Escape, or click, to restore.');
      Object.assign(blackoutLayer.style, { position: 'fixed', inset: '0', background: '#000', zIndex: '2147483646', display: 'none', outline: 'none' });
      document.body.append(blackoutLayer);
      document.addEventListener('keydown', handleKey);
      document.addEventListener('click', handleAction);
      document.addEventListener('click', (event) => {
        if (blackout) {
          event.preventDefault();
          event.stopImmediatePropagation();
          setBlackout(false);
        } else if (zoomState) {
          event.preventDefault();
          event.stopImmediatePropagation();
          exitZoom();
        } else if ((zoomArmed || event.altKey) && event.target instanceof Node && deck.contains(event.target)) {
          event.preventDefault();
          event.stopImmediatePropagation();
          zoomAt(event.clientX, event.clientY);
        }
      }, true);
      document.addEventListener('focusin', (event) => {
        if (blackout && event.target !== blackoutLayer) blackoutLayer.focus({ preventScroll: true });
        else if (activeDialog && dialogUsesFallback && !activeDialog.contains(event.target)) {
          (focusableElements(activeDialog)[0] || activeDialog).focus({ preventScroll: true });
        }
      });
      window.addEventListener('hashchange', () => {
        const target = hashTarget();
        if (target === null) {
          writeHash();
          showProblem(unknownUrlMessage);
        } else {
          navigate(target.index, target.step);
        }
      });
      window.addEventListener('blur', () => {
        if (zoomArmed) setZoomArmed(false, false);
      });
      for (const name of ['fullscreenchange', 'webkitfullscreenchange']) document.addEventListener(name, updateControls);
    } else {
      deck.inert = true;
      actions.forEach((control) => { control.hidden = true; });
      // A mirror is a passive preview, including links inside a slide.
      document.body.style.pointerEvents = 'none';
    }

    try {
      if (typeof window.BroadcastChannel === 'function') {
        channel = new window.BroadcastChannel(`${PREFIX}sync`);
        channel.addEventListener('message', (event) => receive(event.data));
      }
    } catch {
      channel = null;
    }

    window.addEventListener('storage', (event) => {
      if (event.key === STATE_KEY) {
        if (mirror) receiveState(parse(event.newValue));
      } else if (event.key === COMMAND_KEY) {
        if (!mirror) receive(parse(event.newValue), true);
      } else if (!mirror && event.key && event.key.startsWith(NOTES_PREFIX)) {
        const suffix = event.key.slice(NOTES_PREFIX.length);
        if (!/^(0|[1-9]\d*)$/.test(suffix)) return;
        const position = Number(suffix);
        if (!Number.isSafeInteger(position) || position >= records.length) return;
        const paragraphs = event.newValue === null ? originalNotes[position] : parse(event.newValue);
        applyNotes(position, paragraphs, false);
      }
    });
    window.addEventListener('resize', () => {
      if (!resizeFrame) resizeFrame = window.requestAnimationFrame(resize);
    });
    window.addEventListener('pageshow', (event) => {
      if (!event.persisted) return;
      resize();
      if (mirror) requestState();
      else publishState();
    });

    const saved = parse(readStored(STATE_KEY));
    resize();
    if (mirror) {
      if (saved && saved.type === 'state') receiveState(saved);
      if (!snapshot) render(false);
      requestState();
    } else {
      const savedIndex = saved && saved.type === 'state' ? clampIndex(saved.index) : null;
      const target = hashTarget()
        || (savedIndex === null ? { index: 0, step: 0 } : { index: savedIndex, step: clampStep(savedIndex, saved.step) });
      index = target.index;
      step = target.step;
      writeHash();
      render();
      publishState();
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
  else start();
})();
