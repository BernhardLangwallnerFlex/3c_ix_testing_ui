"""Behavioral tests for the sanity-check evaluator (specs/2026-05-30-ui-sanity-checks-design.md).

Tests drive the public API `evaluate(result) -> SanityReport`. They use small
hand-built results for each rule and real captured fixtures for end-to-end shape.
"""
import json
from pathlib import Path

import pytest

from sanity import evaluate

FIXTURES = Path(__file__).resolve().parent.parent / "test_data" / "results"


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def item(qty=1.0, unit=10.0, line=10.0, tax_rate=None, discount=None, name="x"):
    return {
        "qty": qty,
        "unitPriceNet": unit,
        "lineTotalNet": line,
        "taxRate": tax_rate,
        "discount": discount,
        "name": name,
    }


def subdoc(items=None, net=None, rate=None, tax_amount=None, gross=None,
           discount=None, number="N-1", currency="EUR"):
    return {
        "type": "invoice",
        "currency": currency,
        "number": number,
        "items": items if items is not None else [],
        "totals": {
            "net": net,
            "tax": {"rate": rate, "amount": tax_amount},
            "gross": gross,
            "discount": discount,
        },
        "warnings": [],
    }


def result(*subdocs):
    return {"number_of_subdocuments": len(subdocs), "subdocuments": list(subdocs)}


def checks_by_id(subdoc_report):
    return {c.id: c for c in subdoc_report.checks}


# --------------------------------------------------------------------------
# clean / happy path
# --------------------------------------------------------------------------
def test_clean_invoice_all_pass():
    s = subdoc(
        items=[item(2, 10, 20), item(3, 10, 30)],
        net=50.0, rate=19.0, tax_amount=9.5, gross=59.5,
    )
    rep = evaluate(result(s))
    assert rep.verdict == "pass"
    assert rep.subdoc_count == 1
    c = checks_by_id(rep.subdocs[0])
    assert c["lineItems"].verdict == "pass"
    assert c["itemsSumNet"].verdict == "pass"
    assert c["tax"].verdict == "pass"
    assert c["gross"].verdict == "pass"


def test_subdoc_has_exactly_four_checks_in_order():
    rep = evaluate(result(subdoc(items=[item()], net=10.0, rate=19.0, tax_amount=1.9, gross=11.9)))
    ids = [c.id for c in rep.subdocs[0].checks]
    assert ids == ["lineItems", "itemsSumNet", "tax", "gross"]


# --------------------------------------------------------------------------
# Check 4 — gross
# --------------------------------------------------------------------------
def test_broken_gross_fails_and_rolls_up():
    s = subdoc(items=[item(1, 50, 50)], net=50.0, rate=19.0, tax_amount=9.5, gross=170.0)
    rep = evaluate(result(s))
    c = checks_by_id(rep.subdocs[0])
    assert c["gross"].verdict == "fail"
    assert c["tax"].verdict == "pass"
    assert rep.verdict == "fail"


def test_gross_discount_fallback_passes():
    # direct net+tax = 119 != gross 109; with totals.discount 10 -> 109 matches
    s = subdoc(items=[item(1, 100, 100)], net=100.0, rate=19.0, tax_amount=19.0,
               gross=109.0, discount=10.0)
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["gross"].verdict == "pass"


# --------------------------------------------------------------------------
# Check 3 — tax
# --------------------------------------------------------------------------
def test_mixed_vat_downgrades_fail_to_warn():
    # blended check would fail, but item taxRates differ from totals rate -> warn
    s = subdoc(
        items=[item(1, 100, 100, tax_rate=7.0), item(1, 100, 100, tax_rate=19.0)],
        net=200.0, rate=19.0, tax_amount=26.0, gross=226.0,  # 200*19% = 38 != 26 -> would fail
    )
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["tax"].verdict == "warn"
    assert "mixed" in (c["tax"].note or "").lower()


def test_mixed_vat_passing_tax_stays_pass():
    s = subdoc(
        items=[item(1, 100, 100, tax_rate=7.0), item(1, 100, 100, tax_rate=19.0)],
        net=200.0, rate=19.0, tax_amount=38.0, gross=238.0,  # exact -> pass despite mixed
    )
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["tax"].verdict == "pass"


# --------------------------------------------------------------------------
# Check 2 — items sum to net
# --------------------------------------------------------------------------
def test_items_sum_discount_fallback_passes():
    # Σ items = 100, net = 90; with totals.discount 10 -> matches
    s = subdoc(items=[item(1, 100, 100)], net=90.0, rate=19.0, tax_amount=17.1,
               gross=107.1, discount=10.0)
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["itemsSumNet"].verdict == "pass"


def test_items_sum_incomplete_caps_at_warn():
    # one line missing lineTotalNet; remaining sum equals net exactly but must cap at warn
    items = [item(1, 50, 50), item(1, 50, None)]
    s = subdoc(items=items, net=50.0, rate=19.0, tax_amount=9.5, gross=59.5)
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["itemsSumNet"].verdict == "warn"
    assert "1/2" in (c["itemsSumNet"].note or "")


# --------------------------------------------------------------------------
# Check 1 — line-item math (capped at warn, our decision)
# --------------------------------------------------------------------------
def test_line_items_clean_pass_with_count_note():
    s = subdoc(items=[item(2, 10, 20), item(3, 10, 30)], net=50.0, rate=19.0,
               tax_amount=9.5, gross=59.5)
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["lineItems"].verdict == "pass"
    assert "2/2" in (c["lineItems"].note or "")


def test_line_item_mismatch_caps_at_warn_never_fail():
    # qty*unit grossly != lineTotal, but sum still equals net -> warn, never fail
    items = [item(1, 371, 371), item(1, 371, 3.71)]  # second line wildly off
    s = subdoc(items=items, net=374.71, rate=19.0, tax_amount=71.19, gross=445.90)
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["lineItems"].verdict == "warn"   # NOT "fail"
    assert c["itemsSumNet"].verdict == "pass"


def test_negative_line_with_matching_signed_product_passes():
    # qty=-1 unit=50 line=-50 -> signed product matches
    items = [item(2, 50, 100), item(-1, 50, -50)]
    s = subdoc(items=items, net=50.0, rate=19.0, tax_amount=9.5, gross=59.5)
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["lineItems"].verdict == "pass"


def test_line_discount_percentage_interpretation_passes():
    # qty*unit*(1 - discount/100): 1*100*(1-10/100)=90
    items = [item(1, 100, 90, discount=10.0)]
    s = subdoc(items=items, net=90.0, rate=19.0, tax_amount=17.1, gross=107.1)
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["lineItems"].verdict == "pass"


# --------------------------------------------------------------------------
# skipped / defensive
# --------------------------------------------------------------------------
def test_null_net_skips_dependent_checks():
    s = subdoc(items=[item(1, 50, 50)], net=None, rate=19.0, tax_amount=9.5, gross=59.5)
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["itemsSumNet"].verdict == "skipped"
    assert c["tax"].verdict == "skipped"
    assert c["gross"].verdict == "skipped"
    assert c["lineItems"].verdict == "pass"  # line math independent of totals
    assert "net" in (c["tax"].note or "").lower()


def test_missing_keys_never_crash():
    # totally empty subdoc -> all skipped, no exception
    rep = evaluate(result({"items": None, "totals": None}))
    assert rep.subdocs[0].verdict == "skipped"


def test_zero_is_a_real_value_not_skipped():
    # net=0, tax.amount=0, gross=0, one zero line -> all consistent, pass (not skipped)
    s = subdoc(items=[item(1, 0, 0)], net=0.0, rate=19.0, tax_amount=0.0, gross=0.0)
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["tax"].verdict == "pass"
    assert c["gross"].verdict == "pass"


# --------------------------------------------------------------------------
# tolerance band (§5)
# --------------------------------------------------------------------------
def test_small_delta_within_pass_band():
    # gross off by 0.01 on ~119 -> within max(0.02, 0.1%) -> pass
    s = subdoc(items=[item(1, 100, 100)], net=100.0, rate=19.0, tax_amount=19.0, gross=119.01)
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["gross"].verdict == "pass"


def test_medium_delta_is_warn():
    # gross off by 0.50 on 119 -> beyond pass(0.12) within warn(1.19) -> warn
    s = subdoc(items=[item(1, 100, 100)], net=100.0, rate=19.0, tax_amount=19.0, gross=119.50)
    c = checks_by_id(evaluate(result(s)).subdocs[0])
    assert c["gross"].verdict == "warn"


# --------------------------------------------------------------------------
# rollup (§7)
# --------------------------------------------------------------------------
def test_multi_subdoc_rollup_is_worst():
    good = subdoc(items=[item(1, 50, 50)], net=50.0, rate=19.0, tax_amount=9.5, gross=59.5)
    bad = subdoc(items=[item(1, 50, 50)], net=50.0, rate=19.0, tax_amount=9.5, gross=200.0)
    rep = evaluate(result(good, bad))
    assert rep.verdict == "fail"
    assert rep.subdocs[0].verdict == "pass"
    assert rep.subdocs[1].verdict == "fail"


def test_empty_result_is_skipped():
    rep = evaluate(result())
    assert rep.verdict == "skipped"
    assert rep.subdoc_count == 0


def test_subdoc_verdict_ignores_skipped():
    # net null -> 3 skipped + lineItems pass -> subdoc verdict pass (not skipped)
    s = subdoc(items=[item(1, 50, 50)], net=None, rate=None, tax_amount=None, gross=None)
    assert evaluate(result(s)).subdocs[0].verdict == "pass"


# --------------------------------------------------------------------------
# real fixtures
# --------------------------------------------------------------------------
def load(product, name):
    return json.loads((FIXTURES / product / f"{name}.json").read_text())


@pytest.mark.parametrize("product,name", [
    ("bps", "BPS_2"),
    ("bps", "BPS_4"),
    ("sanierer", "LO_Rechnung"),
    ("sanierer", "Verkaufsrechnung_42613847"),
    ("vetcostcheck", "testrechnung_01_bulldogge"),
])
def test_fixture_clean_docs_pass(product, name):
    rep = evaluate(load(product, name))
    assert rep.verdict == "pass", f"{product}/{name} expected pass, got {rep.verdict}"


def test_fixture_ar26076770_lineitems_warn_others_pass():
    rep = evaluate(load("sanierer", "AR26076770"))
    c = checks_by_id(rep.subdocs[0])
    assert c["lineItems"].verdict == "warn"     # capped, real qty/unit noise
    assert c["itemsSumNet"].verdict == "pass"
    assert c["tax"].verdict == "pass"
    assert c["gross"].verdict == "pass"
    assert rep.verdict == "warn"


def test_fixture_vcc_katze_lineitems_capped_to_warn():
    rep = evaluate(load("vetcostcheck", "testrechnung_03_katze"))
    assert checks_by_id(rep.subdocs[0])["lineItems"].verdict == "warn"
    assert rep.verdict == "warn"


def test_fixture_vcc_multidoc_rollup_ignores_skipped_prescription():
    # 4 subdocs: 3 invoices pass, 1 prescription (net=null) fully skipped.
    rep = evaluate(load("vetcostcheck", "VCC_Viele_Dokumente"))
    assert rep.subdoc_count == 4
    assert [sd.verdict for sd in rep.subdocs[:3]] == ["pass", "pass", "pass"]
    assert rep.subdocs[3].verdict == "skipped"   # prescription, no numbers
    assert rep.verdict == "pass"                 # skipped doesn't drag the rollup down
