-- Site-wide tags whose paths must be relative to each page, so they work under a project
-- subpath (Quarto only rewrites the paths it knows about):
--   * the favicons (the MFM mark on a navy circle): Quarto's `website.favicon` links the
--     SVG; this adds the .ico for browsers without SVG favicons and the Apple touch icon;
--   * assets/js/site.js, the navbar, footer, code block and tab behaviour shared by every page;
--   * assets/js/search.js, the additions to Quarto's search (path lines, empty message, panel);
--   * preloads for the two fonts every page needs (text and page titles), so titles do not jump.
function Meta(meta)
  local root = quarto.project.offset
  quarto.doc.include_text(
    "in-header",
    string.format(
      '<link rel="icon" href="%s/assets/favicon/favicon.ico" sizes="any">\n'
        .. '<link rel="apple-touch-icon" href="%s/assets/favicon/apple-touch-icon.png">\n'
        .. '<link rel="preload" href="%s/assets/fonts/ibm-plex-sans-latin-400-500.woff2" as="font" type="font/woff2" crossorigin>\n'
        .. '<link rel="preload" href="%s/assets/fonts/archivo-latin-wdth125-400.woff2" as="font" type="font/woff2" crossorigin>',
      root,
      root,
      root,
      root
    )
  )
  quarto.doc.include_text(
    "after-body",
    string.format(
      '<script src="%s/assets/js/site.js" defer></script>\n'
        .. '<script src="%s/assets/js/search.js" defer></script>',
      root,
      root
    )
  )
  return meta
end
