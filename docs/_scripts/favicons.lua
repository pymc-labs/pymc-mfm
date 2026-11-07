-- The favicon set from the MFM logo package. Quarto's `website.favicon` links the SVG;
-- this adds the .ico for browsers without SVG favicons and the Apple touch icon, with
-- paths relative to each page so they work under a project subpath.
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
  return meta
end
