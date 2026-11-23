-- Pilot profile only (_quarto-pilot.yml). The Python reference is one page, so a link to
-- the page of one class or function ("../reference/client.Client.qmd") goes to that
-- object's anchor on reference/index.qmd. A link that has its own #anchor keeps it.
function Link(el)
  local prefix, name, fragment = el.target:match("^(.-)reference/(client%.[%w_.]-)%.qmd(#?.*)$")
  if not name then
    return nil
  end
  if fragment == "" then
    fragment = "#pymc_mfm." .. name
  end
  el.target = prefix .. "reference/index.qmd" .. fragment
  return el
end
