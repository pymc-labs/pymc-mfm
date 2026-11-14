-- Site-wide tags whose paths must be relative to each page, so they work under a project
-- subpath (Quarto only rewrites the paths it knows about):
--   * the favicons (the MFM mark on a navy circle): Quarto's `website.favicon` links the
--     SVG; this adds the .ico for browsers without SVG favicons and the Apple touch icon;
--   * assets/js/site.js, the navbar, footer, code block and tab behaviour shared by every page;
--   * assets/js/search.js, the additions to Quarto's search (path lines, empty message, panel);
--   * preloads for the two fonts every page needs (text and page titles), so titles do not jump;
--   * inner pages only (every page but the home page, index.qmd at the project root):
--     tiles-pages.css, and tiles-painter.js + tiles-pages.js, which paint the header band
--     and the skirting row once into canvases.
function Meta(meta)
  local root = quarto.project.offset
  -- Quarto does not pass `body-classes` to filters, so the home page is the one page at the
  -- project root (its offset is "."); every page in a folder is an inner page.
  local inner = root ~= "."

  local head = string.format(
    '<link rel="icon" href="%s/assets/favicon/favicon.ico" sizes="any">\n'
      .. '<link rel="apple-touch-icon" href="%s/assets/favicon/apple-touch-icon.png">\n'
      .. '<link rel="preload" href="%s/assets/fonts/ibm-plex-sans-latin-400-500.woff2" as="font" type="font/woff2" crossorigin>\n'
      .. '<link rel="preload" href="%s/assets/fonts/archivo-latin-wdth125-400.woff2" as="font" type="font/woff2" crossorigin>',
    root,
    root,
    root,
    root
  )
  if inner then
    head = head .. string.format('\n<link rel="stylesheet" href="%s/assets/styles/tiles-pages.css">', root)
  end
  quarto.doc.include_text("in-header", head)

  local body = string.format(
    '<script src="%s/assets/js/site.js" defer></script>\n'
      .. '<script src="%s/assets/js/search.js" defer></script>',
    root,
    root
  )
  if inner then
    body = body
      .. string.format(
        '\n<script src="%s/assets/js/tiles-painter.js" defer></script>\n'
          .. '<script src="%s/assets/js/tiles-pages.js" defer></script>',
        root,
        root
      )
  end
  quarto.doc.include_text("after-body", body)
  return meta
end
