/* Footer chrome: the current agenda section, a seven-part section track and the tone of colored slides. */
(() => {
  'use strict';
  const label = document.getElementById('deck-section');
  const track = document.querySelector('.section-track');
  const segments = track ? Array.from(track.children) : [];
  const fallback = label ? label.textContent : '';

  addEventListener('webinar:slide', (event) => {
    const slide = document.querySelectorAll('#deck > section.slide')[event.detail.index];
    if (!slide) return;
    const position = Number(slide.dataset.sectionIndex) || 0;
    if (label) label.textContent = slide.dataset.section || fallback;
    if (track) {
      track.hidden = position === 0;
      segments.forEach((segment, offset) => {
        segment.classList.toggle('is-current', offset + 1 === position);
        segment.classList.toggle('is-past', offset + 1 < position);
      });
    }
    const tone = Array.from(slide.classList).find((name) => name.startsWith('tone-'));
    if (tone) document.body.dataset.tone = tone.slice(5);
    else delete document.body.dataset.tone;
  });
})();
