# TODO

Open items for the testing UI. Nothing here is blocking — the app is deployed and working
as of 2026-08-13.

---

## 1. Prod-vs-test comparison

**The main open feature.** Now that both environments are one click apart, the obvious next
capability is running one file through *both* and comparing the results — the thing you
actually want when deciding whether a test build is safe to promote.

Deferred deliberately during the environment-toggle design (see
`specs/2026-08-13-prod-test-environment-toggle-design.md` §7), to keep that change small.

Rough shape, not yet designed:

- Send one upload to both targets of the active product, in one run
- Show the two extraction JSONs side by side with differences highlighted
- Show both sanity verdicts together, since a diff that changes no verdict is usually noise
- Decide what "different" means for floats — reuse the tolerance band in `sanity.py` (`TOL`)
  rather than exact equality, or a real €0.001 rounding difference reads as a regression

Open questions to settle before building:

- Field-level diff of the whole JSON, or only the fields that drive the sanity checks?
- One file at a time, or the whole `test_data/<product>/` corpus as a regression sweep?
- Is a run pinned to one target (today's model), or does a comparison run belong to both?
  This decides whether `filter_runs` still works as-is or needs a third run kind.

Start with `superpowers:brainstorming` — this needs a design pass, not a straight build.

---

## 2. Login does not survive a page refresh

Observed while testing on 2026-08-13, **pre-existing** and unrelated to the environment
toggle. Reloading the page returned the login screen despite `require_login` setting a
30-day cookie.

Suspected cause, unverified: `_set_auth_cookie` writes `document.cookie` from inside a
Streamlit component iframe (`st.components.v1.html`), which browsers increasingly treat as
third-party and may partition or drop.

Only reproduced under Playwright automation, which has its own cookie policy — **confirm it
happens in a normal browser before investing in a fix.** If it does, it is daily friction
worth fixing, because a refresh also discards every cached PDF in session state.

---

## 3. `./deploy.sh` is blocked by the Claude Code permission classifier

Claude cannot run `./deploy.sh` — it gets denied in both foreground and background, so a
human has to run the deploy step. Fixable with a Bash permission rule in
`.claude/settings.json` if the handoff becomes annoying.

---

## 4. Housekeeping

- Container App revision `ca-vetcostcheck-ui--0000008` is still registered at 0% traffic.
  Harmless — Container Apps keeps it for rollback. Deactivate if the list gets noisy.
- `scripts/fetch_samples.py` only ever hits **prod** (it reads `<PRODUCT>_API_URL`). Worth
  an environment argument now that test endpoints exist, especially if item 1 grows into a
  corpus-wide regression sweep.
