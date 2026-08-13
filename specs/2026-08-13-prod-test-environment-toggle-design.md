# Prod / Test Environment Toggle — Design Spec

**Date:** 2026-08-13
**Scope:** UI-only (`app.py`, tests, deployment config). No API changes.

---

## 1. Purpose

Each of the three products (VetCostCheck, BPS, Sanierer) now has **two** deployed
extraction APIs: a production instance and a test instance. The UI currently knows only
about the production instance of each product.

Add an **environment toggle** so the tester can point the UI at either instance of the
selected product, without ever being unsure which one a given result came from.

The two APIs are functionally identical — the test instances expose the same surface
(`/upload`, `/process`, `/job/{job_id}`, `/healthz`, `/ready`) and return the same result
shape. **Switching environment is purely a change of base URL and API key.** No request,
polling, parsing, or sanity-check logic changes.

---

## 2. Target model: a target is (product, environment)

Today `app.py` holds a module-level `PRODUCTS` dict built from `os.getenv` at import time.
This is untestable (env is read once, at import) and assumes a single environment.

Replace it with a resolver:

```python
PRODUCT_PREFIX = {
    "VetCostCheck": "VETCOSTCHECK",
    "BPS": "BPS",
    "Sanierer": "SANIERER",
}
ENVIRONMENTS = ("Test", "Prod")
DEFAULT_PRODUCT = "VetCostCheck"
DEFAULT_ENVIRONMENT = "Test"

def resolve_target(product: str, env: str, environ=None) -> dict:
    """Return {"base_url": str, "api_key": str} for a (product, environment) pair."""
```

`environ` defaults to `os.environ` and exists so tests can inject a dict. The function is
pure with respect to that mapping.

### Variable names

These match what is already present in `.env`:

| | Prod | Test |
|---|---|---|
| URL | `<PREFIX>_API_URL` | `<PREFIX>_TEST_API_URL` |
| Key | `<PREFIX>_API_KEY` | `<PREFIX>_TEST_API_KEY` |

### Fallbacks

- **VetCostCheck + Prod** falls back to the legacy `API_BASE_URL` / `API_KEY` if
  `VETCOSTCHECK_API_URL` / `VETCOSTCHECK_API_KEY` are unset. This fallback must be kept:
  the deployed Container App currently sets only the legacy pair for VetCostCheck, so
  removing it breaks production.
- **No other fallback exists.** In particular, a Test target never falls back to a Prod
  variable — silently sending test traffic to a production API is the one failure mode
  this feature must not have. An unset test variable yields an empty string, which the UI
  surfaces (§3) rather than hiding.
- A missing variable resolves to `""`, never `None`, so the sidebar text inputs and
  `api_headers()` behave as they do today.

---

## 3. UI changes

### 3.1 Sidebar selector

A second `st.segmented_control` labelled **Environment** with options `Test` / `Prod`,
placed directly beneath the existing Product control and above the divider. It carries the
same guard as the product control: single-select segmented controls can be cleared by
clicking the active chip, so a `None` value falls back to `DEFAULT_ENVIRONMENT`.

Default is **Test** — this is a test console, and reaching production should be a
deliberate act. This changes existing behaviour, where the UI opened on the production
VetCostCheck endpoint.

### 3.2 Making Prod unmistakable

Because production is now one click from the default, it must be visually obvious:

- When `Prod` is selected, the sidebar renders a warning badge (e.g.
  `⚠️ PROD — live endpoint`) directly under the Environment control.
- The page title always carries the environment, uppercased:
  `Invoice Extraction – Test Console · BPS · PROD` and
  `Invoice Extraction – Test Console · BPS · TEST`.
- `Test` renders no sidebar badge; the title is the only marker.

If a selected target has an empty `base_url` or `api_key`, the sidebar shows an explicit
error naming the missing environment variable, so an unconfigured test endpoint is
diagnosed rather than producing an opaque HTTP failure.

### 3.3 Per-target editable fields

The sidebar `API_BASE_URL` / `API_KEY` text inputs are currently keyed per product
(`api_base_url_{product}`). They become keyed per **(product, environment)**
(`api_base_url_{product}_{env}`), so switching environment never shows the other
environment's URL or key.

Note the consequence, verified in the browser: Streamlit discards widget state for keys not
rendered in a run, so a *manual* edit to one of these fields is lost once you switch away
and back — the field re-initialises from the resolved environment variable. That is the
safe direction to fail (you always see the target you actually selected), and manual edits
are a debugging convenience, not the normal path. Persisting them is deliberately not
implemented.

### 3.4 API Docs page

The docs page already receives the resolved base URL and therefore follows the toggle with
no change. Add a caption naming the active product and environment, so it is clear which
spec is on screen.

---

## 4. Run tagging and filtering

This is the part that matters most for trustworthiness. Runs are currently tagged with
`product` and the Inspector filters on it. Without an equivalent environment tag, a prod
run and a test run of the same file sit in the same list, indistinguishable.

- `add_run(files, product, env)` stores `env` alongside `product` in the run dict.
- `inspector_panel(product, env)` filters runs on **both** fields.
- The empty state names both: `No BPS runs in Test yet. Upload and process files above.`

Runs are not migrated or reinterpreted: session state is per-session and lost on refresh,
so no backwards compatibility concern exists. A run recorded before this change cannot
exist in a session running this code.

---

## 5. Deployment

The deployed Container App (`ca-vetcostcheck-ui`, resource group `rg-3c-invoice`) currently
sets, for production only:

```
API_BASE_URL, API_KEY (secret: api-key)
BPS_API_URL, BPS_API_KEY (secret: bps-api-key)
SANIERER_API_URL, SANIERER_API_KEY (secret: sanierer-api-key)
UI_USERNAME, UI_PASSWORD (secret: ui-password)
```

Six new variables are required — three plain URLs and three keys, the latter as Container
App secrets following the existing `<product>-api-key` naming convention:

```
VETCOSTCHECK_TEST_API_URL   VETCOSTCHECK_TEST_API_KEY  (secret: vetcostcheck-test-api-key)
BPS_TEST_API_URL            BPS_TEST_API_KEY           (secret: bps-test-api-key)
SANIERER_TEST_API_URL       SANIERER_TEST_API_KEY      (secret: sanierer-test-api-key)
```

The exact `az containerapp secret set` / `az containerapp update --set-env-vars` commands
are documented in `CLAUDE.md`. **They are not executed as part of implementation** — the
operator runs them, then deploys via `./deploy.sh`.

Values already exist locally in `.env` (gitignored) and are the source of truth for the
URLs. Note the deployment consequence of §2: until the test variables are set in Azure,
selecting `Test` in the deployed UI shows the §3.2 configuration error. That is the
intended behaviour, and is strictly better than the alternative of falling through to prod.

---

## 6. Testing

`resolve_target` is pure over an injected mapping, so it is unit-testable without Streamlit.
New tests live in `tests/test_targets.py`, alongside the existing `tests/test_sanity.py`:

1. Each of the six (product, environment) pairs resolves to its own URL and key.
2. VetCostCheck + Prod falls back to `API_BASE_URL` / `API_KEY` when the
   `VETCOSTCHECK_*` pair is absent.
3. VetCostCheck + Prod prefers `VETCOSTCHECK_API_URL` over the legacy `API_BASE_URL` when
   both are set.
4. A Test target with no variables set resolves to empty strings and **does not** return
   the corresponding Prod values.
5. An unknown product name raises rather than silently returning an empty target.

Existing `tests/test_sanity.py` must continue to pass unchanged — the sanity evaluator is
untouched by this work.

---

## 7. Out of scope

- Comparing or diffing prod against test results (explicitly deferred).
- Running both environments in a single run.
- Any change to `sanity.py`, the polling loop, the upload/process calls, or the result
  rendering.
- Executing the Azure configuration commands.
- Per-environment polling intervals or timeouts — these stay global.
