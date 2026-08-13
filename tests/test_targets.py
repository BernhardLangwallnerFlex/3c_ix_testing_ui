"""Tests for target resolution (specs/2026-08-13-prod-test-environment-toggle-design.md)."""
import pytest

from targets import (
    DEFAULT_ENVIRONMENT,
    DEFAULT_PRODUCT,
    ENVIRONMENTS,
    PRODUCTS,
    filter_runs,
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


# --------------------------------------------------------------------------
# run scoping
# --------------------------------------------------------------------------
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
