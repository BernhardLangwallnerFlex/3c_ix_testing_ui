# Prod / Test Environment Toggle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the tester switch the UI between each product's production and test extraction API, without ever being able to confuse results from the two.

**Architecture:** A new pure module `targets.py` owns the notion of a *target* — a (product, environment) pair — resolving it to a base URL and API key from environment variables, and deciding which stored runs belong to it. `app.py` gains a second sidebar segmented control and threads the selected environment through the same paths that already carry `product`. No request, polling, parsing, or sanity-check logic changes: the two APIs are functionally identical.

**Tech Stack:** Python 3.12, Streamlit 1.51.0, pytest ≥ 8. Tests run from the repo root with `.venv/bin/python -m pytest`.

**Spec:** `specs/2026-08-13-prod-test-environment-toggle-design.md`

## Global Constraints

- **A Test target must never fall back to a Prod environment variable.** Silently sending test traffic to a production API is the one failure mode this feature must not have. An unset test variable resolves to `""`.
- **VetCostCheck + Prod must keep its legacy fallback** to `API_BASE_URL` / `API_KEY`. The deployed Container App sets only that legacy pair for VetCostCheck; removing the fallback breaks production.
- **Default environment is `Test`.** Default product stays `VetCostCheck`.
- **No API/backend changes.** `sanity.py` is not touched.
- **All 28 existing tests in `tests/test_sanity.py` must keep passing**, unchanged.
- **Do not execute the Azure configuration commands.** They are documented for the operator to run.
- Environment variable names, exactly: `<PREFIX>_API_URL` / `<PREFIX>_API_KEY` for Prod and `<PREFIX>_TEST_API_URL` / `<PREFIX>_TEST_API_KEY` for Test, where `<PREFIX>` is `VETCOSTCHECK`, `BPS`, or `SANIERER`.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `targets.py` | **create** | Pure target model: product/environment constants, `resolve_target`, `target_env_vars`, `filter_runs`. No Streamlit, no network. Mirrors the existing `sanity.py` pattern of a pure module beside `app.py`. |
| `tests/test_targets.py` | **create** | Unit tests for `targets.py`. |
| `app.py` | modify | Sidebar environment control, per-target widget keys, PROD badge, config error, title, run tagging, inspector filtering. |
| `CLAUDE.md` | modify | Document the environment toggle and the Azure variables the operator must set. |

---

### Task 1: Target resolution

Creates `targets.py` with the constants and the URL/key resolver. Nothing in `app.py` changes yet — this task is self-contained and fully unit-tested.

**Files:**
- Create: `targets.py`
- Create: `tests/test_targets.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `PRODUCT_PREFIX: Dict[str, str]` — display name → env-var prefix
  - `PRODUCTS: List[str]` — display names, in sidebar order
  - `ENVIRONMENTS: Tuple[str, str]` — `("Test", "Prod")`
  - `DEFAULT_PRODUCT: str` — `"VetCostCheck"`
  - `DEFAULT_ENVIRONMENT: str` — `"Test"`
  - `target_env_vars(product: str, env: str) -> Tuple[str, str]` — `(url_var_name, key_var_name)`
  - `resolve_target(product: str, env: str, environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]` — `{"base_url": str, "api_key": str}`, both always `str`, `""` when unset

- [ ] **Step 1: Write the failing tests**

Create `tests/test_targets.py`:

```python
"""Tests for target resolution (specs/2026-08-13-prod-test-environment-toggle-design.md)."""
import pytest

from targets import (
    DEFAULT_ENVIRONMENT,
    DEFAULT_PRODUCT,
    ENVIRONMENTS,
    PRODUCTS,
    resolve_target,
    target_env_vars,
)

# A fully-populated environment: all six targets distinct.
FULL_ENV = {
    "VETCOSTCHECK_API_URL": "https://vcc.prod",
    "VETCOSTCHECK_API_KEY": "vcc-prod-key",
    "BPS_API_URL": "https://bps.prod",
    "BPS_API_KEY": "bps-prod-key",
    "SANIERER_API_URL": "https://san.prod",
    "SANIERER_API_KEY": "san-prod-key",
    "VETCOSTCHECK_TEST_API_URL": "https://vcc.test",
    "VETCOSTCHECK_TEST_API_KEY": "vcc-test-key",
    "BPS_TEST_API_URL": "https://bps.test",
    "BPS_TEST_API_KEY": "bps-test-key",
    "SANIERER_TEST_API_URL": "https://san.test",
    "SANIERER_TEST_API_KEY": "san-test-key",
}


def test_defaults_are_the_agreed_ones():
    assert DEFAULT_PRODUCT == "VetCostCheck"
    assert DEFAULT_ENVIRONMENT == "Test"
    assert ENVIRONMENTS == ("Test", "Prod")
    assert PRODUCTS == ["VetCostCheck", "BPS", "Sanierer"]


@pytest.mark.parametrize(
    "product,env,url,key",
    [
        ("VetCostCheck", "Prod", "https://vcc.prod", "vcc-prod-key"),
        ("VetCostCheck", "Test", "https://vcc.test", "vcc-test-key"),
        ("BPS", "Prod", "https://bps.prod", "bps-prod-key"),
        ("BPS", "Test", "https://bps.test", "bps-test-key"),
        ("Sanierer", "Prod", "https://san.prod", "san-prod-key"),
        ("Sanierer", "Test", "https://san.test", "san-test-key"),
    ],
)
def test_each_pair_resolves_to_its_own_target(product, env, url, key):
    assert resolve_target(product, env, FULL_ENV) == {"base_url": url, "api_key": key}


def test_vetcostcheck_prod_falls_back_to_legacy_vars():
    environ = {"API_BASE_URL": "https://legacy", "API_KEY": "legacy-key"}
    assert resolve_target("VetCostCheck", "Prod", environ) == {
        "base_url": "https://legacy",
        "api_key": "legacy-key",
    }


def test_vetcostcheck_prod_prefers_specific_vars_over_legacy():
    environ = dict(FULL_ENV, API_BASE_URL="https://legacy", API_KEY="legacy-key")
    assert resolve_target("VetCostCheck", "Prod", environ) == {
        "base_url": "https://vcc.prod",
        "api_key": "vcc-prod-key",
    }


def test_legacy_fallback_does_not_apply_to_other_products():
    environ = {"API_BASE_URL": "https://legacy", "API_KEY": "legacy-key"}
    assert resolve_target("BPS", "Prod", environ) == {"base_url": "", "api_key": ""}


@pytest.mark.parametrize("product", ["VetCostCheck", "BPS", "Sanierer"])
def test_test_target_never_falls_back_to_prod(product):
    """The one failure mode this feature must not have."""
    environ = {
        "VETCOSTCHECK_API_URL": "https://vcc.prod",
        "VETCOSTCHECK_API_KEY": "vcc-prod-key",
        "BPS_API_URL": "https://bps.prod",
        "BPS_API_KEY": "bps-prod-key",
        "SANIERER_API_URL": "https://san.prod",
        "SANIERER_API_KEY": "san-prod-key",
        "API_BASE_URL": "https://legacy",
        "API_KEY": "legacy-key",
    }
    assert resolve_target(product, "Test", environ) == {"base_url": "", "api_key": ""}


def test_missing_variables_resolve_to_empty_strings_not_none():
    target = resolve_target("BPS", "Test", {})
    assert target == {"base_url": "", "api_key": ""}
    assert isinstance(target["base_url"], str)
    assert isinstance(target["api_key"], str)


def test_unknown_product_raises():
    with pytest.raises(ValueError):
        resolve_target("Nonesuch", "Test", FULL_ENV)


def test_unknown_environment_raises():
    with pytest.raises(ValueError):
        resolve_target("BPS", "Staging", FULL_ENV)


@pytest.mark.parametrize(
    "product,env,expected",
    [
        ("VetCostCheck", "Prod", ("VETCOSTCHECK_API_URL", "VETCOSTCHECK_API_KEY")),
        ("VetCostCheck", "Test", ("VETCOSTCHECK_TEST_API_URL", "VETCOSTCHECK_TEST_API_KEY")),
        ("BPS", "Test", ("BPS_TEST_API_URL", "BPS_TEST_API_KEY")),
        ("Sanierer", "Prod", ("SANIERER_API_URL", "SANIERER_API_KEY")),
    ],
)
def test_target_env_vars_names(product, env, expected):
    assert target_env_vars(product, env) == expected


def test_resolve_target_reads_os_environ_by_default(monkeypatch):
    monkeypatch.setenv("BPS_TEST_API_URL", "https://from-os-environ")
    monkeypatch.setenv("BPS_TEST_API_KEY", "k")
    assert resolve_target("BPS", "Test")["base_url"] == "https://from-os-environ"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_targets.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'targets'`

- [ ] **Step 3: Write the implementation**

Create `targets.py`:

```python
"""API targets: a target is a (product, environment) pair.

Pure module — no Streamlit, no network, no I/O beyond reading a mapping.
See specs/2026-08-13-prod-test-environment-toggle-design.md.
"""
import os
from typing import Dict, List, Mapping, Optional, Tuple

# Display name -> environment-variable prefix.
PRODUCT_PREFIX: Dict[str, str] = {
    "VetCostCheck": "VETCOSTCHECK",
    "BPS": "BPS",
    "Sanierer": "SANIERER",
}
PRODUCTS: List[str] = list(PRODUCT_PREFIX)

ENVIRONMENTS: Tuple[str, str] = ("Test", "Prod")

DEFAULT_PRODUCT = "VetCostCheck"
DEFAULT_ENVIRONMENT = "Test"

# VetCostCheck production predates the per-product variables, and the deployed
# Container App still sets only this legacy pair. Test targets deliberately have
# no fallback of any kind — see spec §2.
_LEGACY_PROD_VARS: Dict[str, Tuple[str, str]] = {
    "VetCostCheck": ("API_BASE_URL", "API_KEY"),
}


def _validate(product: str, env: str) -> None:
    if product not in PRODUCT_PREFIX:
        raise ValueError(f"Unknown product: {product!r}")
    if env not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {env!r}")


def target_env_vars(product: str, env: str) -> Tuple[str, str]:
    """Names of the (url, key) environment variables backing this target."""
    _validate(product, env)
    prefix = PRODUCT_PREFIX[product]
    infix = "_TEST" if env == "Test" else ""
    return f"{prefix}{infix}_API_URL", f"{prefix}{infix}_API_KEY"


def resolve_target(
    product: str, env: str, environ: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Resolve a (product, environment) pair to {"base_url", "api_key"}.

    Missing variables yield "" — never None, and never a value belonging to a
    different environment.
    """
    _validate(product, env)
    environ = os.environ if environ is None else environ

    url_var, key_var = target_env_vars(product, env)
    base_url = environ.get(url_var, "")
    api_key = environ.get(key_var, "")

    if env == "Prod" and product in _LEGACY_PROD_VARS:
        legacy_url, legacy_key = _LEGACY_PROD_VARS[product]
        base_url = base_url or environ.get(legacy_url, "")
        api_key = api_key or environ.get(legacy_key, "")

    return {"base_url": base_url, "api_key": api_key}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_targets.py -q`
Expected: all pass (21 tests).

Then confirm nothing else broke: `.venv/bin/python -m pytest -q` — expected 49 passed.

- [ ] **Step 5: Commit**

```bash
git add targets.py tests/test_targets.py
git commit -m "Add (product, environment) target resolution"
```

---

### Task 2: Run scoping

Adds the run filter that keeps prod and test results apart. Still pure, still no `app.py` change.

**Files:**
- Modify: `targets.py` (append)
- Modify: `tests/test_targets.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1 beyond living in the same module.
- Produces: `filter_runs(runs: List[dict], product: str, env: str) -> List[dict]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_targets.py` (and add `filter_runs` to the existing `from targets import (...)` block):

```python
def make_run(run_id, product, env=None):
    run = {"run_id": run_id, "product": product, "files": []}
    if env is not None:
        run["env"] = env
    return run


def test_filter_runs_matches_both_product_and_environment():
    runs = [
        make_run("a", "BPS", "Test"),
        make_run("b", "BPS", "Prod"),
        make_run("c", "VetCostCheck", "Test"),
    ]
    assert [r["run_id"] for r in filter_runs(runs, "BPS", "Test")] == ["a"]


def test_filter_runs_preserves_order():
    runs = [
        make_run("a", "BPS", "Test"),
        make_run("b", "BPS", "Prod"),
        make_run("c", "BPS", "Test"),
    ]
    assert [r["run_id"] for r in filter_runs(runs, "BPS", "Test")] == ["a", "c"]


def test_filter_runs_excludes_runs_with_no_environment():
    runs = [make_run("a", "BPS")]
    assert filter_runs(runs, "BPS", "Test") == []


def test_filter_runs_on_empty_list():
    assert filter_runs([], "BPS", "Test") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_targets.py -q`
Expected: `ImportError: cannot import name 'filter_runs' from 'targets'`

- [ ] **Step 3: Write the implementation**

Append to `targets.py`:

```python
def filter_runs(runs: List[dict], product: str, env: str) -> List[dict]:
    """Runs belonging to this target, in their existing order.

    A run must match on both fields: a prod run and a test run of the same file
    are otherwise indistinguishable in the Inspector.
    """
    return [r for r in runs if r.get("product") == product and r.get("env") == env]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_targets.py -q`
Expected: all pass (25 tests).

- [ ] **Step 5: Commit**

```bash
git add targets.py tests/test_targets.py
git commit -m "Scope runs to a (product, environment) target"
```

---

### Task 3: Environment selector in the UI

Replaces the module-level `PRODUCTS` dict in `app.py` with the resolver, adds the Environment control, the PROD badge, the missing-configuration error, and the environment in the page title.

`app.py` has no automated test coverage (it is a Streamlit script), so verification is an import smoke check plus a short manual pass. Do not skip the manual pass.

**Files:**
- Modify: `app.py` — lines 23–28 (constants), 38–55 (`PRODUCTS`/`DEFAULT_PRODUCT`/`APP_VERSION`), 235–263 (`sidebar_config`), 668–726 (`main`)

**Interfaces:**
- Consumes: `PRODUCTS`, `ENVIRONMENTS`, `DEFAULT_PRODUCT`, `DEFAULT_ENVIRONMENT`, `resolve_target`, `target_env_vars` from `targets`.
- Produces: `sidebar_config(product: str, env: str) -> Tuple[str, str, float, int]` (unchanged return shape: `api_base_url, api_key, poll_interval, job_timeout`).

- [ ] **Step 1: Replace the config block**

In `app.py`, delete the `DEFAULT_API_BASE_URL` and `DEFAULT_API_KEY` lines (23–24) and the whole `# Products` block including the `PRODUCTS` dict and `DEFAULT_PRODUCT` (lines 31–52). Add the import alongside `import sanity` (line 17):

```python
import sanity
from targets import (
    DEFAULT_ENVIRONMENT,
    DEFAULT_PRODUCT,
    ENVIRONMENTS,
    PRODUCTS,
    filter_runs,
    resolve_target,
    target_env_vars,
)
```

Keep the remaining config constants exactly as they are:

```python
DEFAULT_UI_USERNAME = os.getenv("UI_USERNAME", "admin")
DEFAULT_UI_PASSWORD = os.getenv("UI_PASSWORD", "")
DEFAULT_POLL_INTERVAL = float(os.getenv("POLL_INTERVAL_SECONDS", "1.0"))
DEFAULT_JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT_SECONDS", "600"))  # 10 minutes
```

Bump the version marker:

```python
APP_VERSION = "ui-v5-prod-test-toggle"
```

> Note the deliberate behaviour change: the old `PRODUCTS` dict carried hard-coded fallback URLs (`http://localhost:8000`, `changeme123`, and the three prod hostnames). Those are gone. An unconfigured target now resolves to empty strings and is reported in the sidebar (Step 2) instead of silently pointing somewhere.

- [ ] **Step 2: Take the environment in `sidebar_config`**

Replace the head of `sidebar_config` (through the two `text_input` calls) with:

```python
def sidebar_config(product: str, env: str):
    target = resolve_target(product, env)
    st.sidebar.header("⚙️ API Configuration")
    # Key the fields per (product, environment) so each of the six targets keeps
    # its own editable URL/key and switching never shows a stale value.
    api_base_url = st.sidebar.text_input(
        "API_BASE_URL", value=target["base_url"], key=f"api_base_url_{product}_{env}"
    )
    api_key = st.sidebar.text_input(
        "API_KEY (X-Api-Key)", value=target["api_key"], type="password",
        key=f"api_key_{product}_{env}",
    )

    # Report an unconfigured target by name rather than letting it fail as an
    # opaque HTTP error. Checks the effective values, so pasting one clears it.
    url_var, key_var = target_env_vars(product, env)
    missing = [v for v, val in ((url_var, api_base_url), (key_var, api_key)) if not val]
    if missing:
        st.sidebar.error(f"{product} · {env} is not configured — set {' and '.join(missing)}")
```

The rest of the function (Polling header, key-length caption, Clear/Logout buttons, `return`) stays exactly as it is.

- [ ] **Step 3: Add the Environment control in `main`**

In `main()`, after the existing product `segmented_control` and its `None` guard, and *before* `st.sidebar.divider()`:

```python
    env = st.sidebar.segmented_control(
        "Environment", options=list(ENVIRONMENTS), default=DEFAULT_ENVIRONMENT
    )
    if env is None:  # single-select can be cleared by clicking the active chip
        env = DEFAULT_ENVIRONMENT
    if env == "Prod":
        st.sidebar.warning("⚠️ PROD — live endpoint")
```

Change the product control to source its options from the shared constant:

```python
    product = st.sidebar.segmented_control(
        "Product", options=PRODUCTS, default=DEFAULT_PRODUCT
    )
```

- [ ] **Step 4: Thread `env` through the page bodies**

Update the call site and title in `main()`:

```python
    api_base_url, api_key, poll_interval, job_timeout = sidebar_config(product, env)

    if page == "Invoice Processing":
        st.title(f"Invoice Extraction – Test Console · {product} · {env.upper()}")
        st.caption("Uploads are cached in this Streamlit session. Refreshing the page will lose cached PDFs.")
        upload_and_process_run(api_base_url, api_key, poll_interval, job_timeout, product, env)
        st.divider()
        inspector_panel(product, env)
    else:
        docs_page(api_base_url, product, env)
```

`upload_and_process_run`, `inspector_panel`, and `docs_page` gain their new parameters in Tasks 4 and 5. To keep this task independently runnable, add the parameters to their signatures now, unused:

```python
def upload_and_process_run(api_base_url: str, api_key: str, poll_interval: float, job_timeout: int, product: str, env: str):
def inspector_panel(product: str, env: str):
def docs_page(api_base_url: str, product: str, env: str):
```

- [ ] **Step 5: Verify it imports and the suite still passes**

Run: `.venv/bin/python -c "import app; print(app.APP_VERSION)"`
Expected: `ui-v5-prod-test-toggle`, no traceback.

Run: `.venv/bin/python -m pytest -q`
Expected: 53 passed (28 sanity + 25 targets).

- [ ] **Step 6: Verify manually**

Run: `.venv/bin/python -m streamlit run app.py`

Confirm, in order:
1. The app opens on **Test** with the title `… · VetCostCheck · TEST`.
2. The sidebar shows the VetCostCheck test URL and a non-empty key.
3. Clicking **Prod** shows the `⚠️ PROD — live endpoint` badge, swaps the title to `· PROD`, and swaps the URL to the production host.
4. Switching product to BPS keeps the environment, and the URL changes to the BPS host for that environment.
5. Editing the URL field and switching environment shows the *other* target's URL, never the edit. (Switching back re-initialises the field from the environment variable — Streamlit discards state for unrendered keys. This is expected; see spec §3.3.)
6. Uploading and processing one PDF against Test still works end to end. (Verified as part of Task 4's manual pass, which processes runs in both environments — no need to run the same extraction twice.)

Stop the server when done.

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "Add prod/test environment selector to the sidebar"
```

---

### Task 4: Tag and filter runs by environment

Without this, a prod run and a test run of the same file share one list in the Inspector.

**Files:**
- Modify: `app.py` — `add_run` (175–185), `upload_and_process_run` (266, `add_run` call at 304), `inspector_panel` (515–521)

**Interfaces:**
- Consumes: `filter_runs` from `targets` (Task 2), already imported in Task 3.
- Produces: run dicts carrying an `"env"` key.

- [ ] **Step 1: Record the environment on each run**

```python
def add_run(files: List[FileJob], product: str, env: str) -> str:
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    st.session_state["runs"].insert(0, {
        "run_id": run_id,
        "created_at": now_ts(),
        "product": product,
        "env": env,
        "files": files,
    })
    st.session_state["selected_run_id"] = run_id
    st.session_state["selected_file_key"] = files[0].file_key if files else None
    return run_id
```

In `upload_and_process_run`, update the call:

```python
    run_id = add_run(file_jobs, product, env)
```

- [ ] **Step 2: Filter the Inspector on both fields**

In `inspector_panel`, replace the product-only comprehension:

```python
    # Only show runs belonging to the active product AND environment.
    product_runs = filter_runs(st.session_state["runs"], product, env)
    if not product_runs:
        st.info(f"No {product} runs in {env} yet. Upload and process files above.")
        return
```

The rest of `inspector_panel` is unchanged.

- [ ] **Step 3: Verify it imports and the suite still passes**

Run: `.venv/bin/python -c "import app"` — expected: no traceback.
Run: `.venv/bin/python -m pytest -q` — expected: 50 passed.

- [ ] **Step 4: Verify manually**

Run: `.venv/bin/python -m streamlit run app.py`

1. Process a file against **Test**. The Inspector shows the run.
2. Switch to **Prod** without processing anything. The Inspector shows `No VetCostCheck runs in Prod yet.` — the test run must **not** appear.
3. Switch back to **Test**. The run reappears.
4. Process a file against **Prod**, then toggle between environments and confirm each list holds only its own run.

Stop the server when done.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "Keep prod and test runs separate in the Inspector"
```

---

### Task 5: Label the docs page and document the deployment

**Files:**
- Modify: `app.py` — `docs_page` (626–627)
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `docs_page(api_base_url, product, env)` signature from Task 3.
- Produces: nothing consumed downstream.

- [ ] **Step 1: Name the target on the docs page**

In `docs_page`, directly after `st.title("API Documentation")`:

```python
    st.caption(f"{product} · {env} — {api_base_url}")
```

- [ ] **Step 2: Update CLAUDE.md**

In the **Running Locally** section, replace the environment-variable list with:

```markdown
Configuration is via environment variables (see `.env` for the full list):
- `UI_USERNAME` / `UI_PASSWORD` — basic login credentials
- `POLL_INTERVAL_SECONDS`, `JOB_TIMEOUT_SECONDS` — polling behavior
- Per-target API endpoints, one pair per (product, environment):

  | Product | Prod | Test |
  |---|---|---|
  | VetCostCheck | `VETCOSTCHECK_API_URL` / `VETCOSTCHECK_API_KEY` | `VETCOSTCHECK_TEST_API_URL` / `VETCOSTCHECK_TEST_API_KEY` |
  | BPS | `BPS_API_URL` / `BPS_API_KEY` | `BPS_TEST_API_URL` / `BPS_TEST_API_KEY` |
  | Sanierer | `SANIERER_API_URL` / `SANIERER_API_KEY` | `SANIERER_TEST_API_URL` / `SANIERER_TEST_API_KEY` |

  VetCostCheck **prod** falls back to the legacy `API_BASE_URL` / `API_KEY` if its
  own pair is unset. Test targets have no fallback: an unset test variable is
  reported in the sidebar rather than silently resolving to the prod endpoint.
```

In the **Architecture** section, after the description of the two pages, add:

```markdown
Two sidebar controls select the API target: **Product** (VetCostCheck / BPS / Sanierer)
and **Environment** (Test / Prod, defaulting to Test). Together they resolve to a base URL
and API key via `targets.resolve_target`. Runs are tagged with both, and the Inspector only
shows runs matching the active target — a prod run and a test run of the same file never
share a list. Prod is marked with a sidebar warning badge and in the page title.
```

In the **Docker / Deployment** section, after the `deploy.sh` paragraph, add:

```markdown
### Container App environment variables

The app reads its targets from Container App env vars. Production keys are stored as
secrets. The test targets must be added once, before `Test` works in the deployed UI
(until then the sidebar reports them as unconfigured):

```bash
az containerapp secret set \
  --name ca-vetcostcheck-ui --resource-group rg-3c-invoice \
  --secrets \
    vetcostcheck-test-api-key="<VETCOSTCHECK_TEST_API_KEY>" \
    bps-test-api-key="<BPS_TEST_API_KEY>" \
    sanierer-test-api-key="<SANIERER_TEST_API_KEY>"

az containerapp update \
  --name ca-vetcostcheck-ui --resource-group rg-3c-invoice \
  --set-env-vars \
    VETCOSTCHECK_TEST_API_URL="https://3cvetcostcheck-test.flex-capital-scale.com" \
    BPS_TEST_API_URL="https://3cbps-test.flex-capital-scale.com" \
    SANIERER_TEST_API_URL="https://3csanierer-test.flex-capital-scale.com" \
    VETCOSTCHECK_TEST_API_KEY=secretref:vetcostcheck-test-api-key \
    BPS_TEST_API_KEY=secretref:bps-test-api-key \
    SANIERER_TEST_API_KEY=secretref:sanierer-test-api-key
```

Key values are in the local `.env` (gitignored). VetCostCheck prod needs no new variable —
it still resolves through the legacy `API_BASE_URL` / `API_KEY` pair already set there.
```

- [ ] **Step 3: Verify**

Run: `.venv/bin/python -c "import app"` — expected: no traceback.
Run: `.venv/bin/python -m pytest -q` — expected: 50 passed.
Run: `.venv/bin/python -m streamlit run app.py`, open **API Docs**, confirm the caption names the active product, environment, and URL, and that switching either control updates both the caption and the rendered spec. Stop the server.

- [ ] **Step 4: Commit**

```bash
git add app.py CLAUDE.md
git commit -m "Label the docs page with its target; document env vars and Azure config"
```

---

## Done when

- `.venv/bin/python -m pytest -q` reports 53 passed.
- The UI opens on Test, marks Prod unmistakably, and keeps the two environments' runs apart.
- `CLAUDE.md` documents the variables and the exact Azure commands — **not run** by the implementer.
