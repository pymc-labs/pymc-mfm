-- Site-wide tags whose paths must be relative to each page, so they work under a project
-- subpath (Quarto only rewrites the paths it knows about):
--   * the favicon set from the MFM logo package: Quarto's `website.favicon` links the SVG;
--     this adds the .ico for browsers without SVG favicons and the Apple touch icon;
--   * assets/js/site.js, the navbar behaviour shared by every page.
function Meta(meta)
  local root = quarto.project.offset
  quarto.doc.include_text(
    "in-header",
    string.format(
      '<link rel="icon" href="%s/assets/favicon/favicon.ico" sizes="any">\n'
        .. '<link rel="apple-touch-icon" href="%s/assets/favicon/apple-touch-icon.png">',
      root,
      root
    )
  )
  quarto.doc.include_text(
    "after-body",
    string.format('<script src="%s/assets/js/site.js" defer></script>', root)
  )
  return meta
end
