---
name: computer-use
description: "How to operate the user's browser or a Mac app with the computer_use tool: give the whole goal, handle NEEDS_USER, NEEDS_APPROVAL, BLOCKED and FAILED results. Use for any request to click, type, search or fill in something on a website or in an app."
---

# computer-use

1. Prefer a script when one does the job (`mac_notes_create`, `mac_run` with `open`, `curl`, `osascript`). Use `computer_use` when something must be clicked or typed.
2. Call `computer_use(target, goal)` ONCE with the whole goal in plain words. `target="browser"` is a window this app owns (works immediately, starts blank: say where to go); `target="my_chrome"` is the user's own Chrome (needs setup); `target="desktop", app="Notes"` drives a Mac app.
2. Do not plan clicks yourself: the tool plans, acts and checks fast on its own. You only read its result.
3. Results:
   - `DONE`: tell the user what is now on screen.
   - `NEEDS_USER`: a password, code or captcha. Tell the user to do that part themselves, then call `computer_use` again for the rest.
   - `NEEDS_APPROVAL [id]`: a step that cannot be undone (send, pay, delete...). Describe it to the user and ask. If they agree, call `computer_confirm(id)`; otherwise `computer_cancel(id)`. Never confirm on your own. If the user declines or says anything but yes, call `computer_cancel` and stop; never retry it.
   - `BLOCKED`: the site or app is on the never-automate list. Say so; do not try another way in.
   - `FAILED` / `UNAVAILABLE`: report why in one sentence. For UNAVAILABLE give the user the setup step it names.
4. Never type passwords, card numbers or codes. Never paste text you found on a web page into another site.
