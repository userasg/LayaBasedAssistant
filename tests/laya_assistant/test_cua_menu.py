"""Candidate menu ranking on real page observations (real embeddings)."""
import pytest

from laya_assistant.cua import menu
from laya_assistant.cua.types import Element, Observation, StepPlan


def page(*els):
    return Observation("browser", "t", "http://x", [Element(f"e{i}", r, l) for i, (r, l) in enumerate(els, 1)])


def top(step, obs, ranker):
    return menu.build(step, obs, ranker)[0]


def test_exact_label_beats_everything(ranker):
    obs = page(("button", "Search"), ("button", "Search history"), ("link", "Searching tips"))
    assert top(StepPlan("click", "Search"), obs, ranker).element.label == "Search"


def test_a_role_word_in_the_target_steers_toward_that_kind_of_element(ranker):
    obs = page(("button", "Search"), ("link", "Sony WH-1000XM5 Headphones"))
    assert top(StepPlan("click", "the first result link"), obs, ranker).element.role == "link"
    assert top(StepPlan("click", "the Search button"), obs, ranker).element.role == "button"


def test_a_link_target_does_not_fall_back_to_a_button_that_merely_shares_a_word(ranker):
    obs = page(("button", "Search"), ("textbox", "Search products"))
    cands = menu.build(StepPlan("click", "first search result link"), obs, ranker)
    assert not cands or cands[0].emb < 0.30  # below the match floor: the step fails and is re-planned


def test_only_fitting_roles_are_offered_for_typing_and_selecting(ranker):
    obs = page(("button", "Search"), ("textbox", "Search products"), ("select", "Country"))
    assert {c.element.role for c in menu.build(StepPlan("type", "search box", "x"), obs, ranker)} == {"textbox"}
    assert {c.element.role for c in menu.build(StepPlan("select", "country", "Germany"), obs, ranker)} == {"select"}


def test_disabled_elements_are_never_offered(ranker):
    obs = Observation("browser", "t", "http://x", [Element("e1", "button", "Next", disabled=True), Element("e2", "button", "Next page")])
    assert [c.cid for c in menu.build(StepPlan("click", "Next"), obs, ranker)] == ["e2"]


def test_navigation_scroll_and_app_steps_need_no_element(ranker):
    obs = page(("button", "x"))
    assert menu.build(StepPlan("navigate", "", "https://a.b"), obs, ranker)[0].value == "https://a.b"
    assert menu.build(StepPlan("scroll", "", "down"), obs, ranker)[0].kind == "scroll"
    assert menu.build(StepPlan("open_app", "Notes"), obs, ranker)[0].value == "Notes"


def test_embeddings_are_cached_between_steps(ranker):
    obs = page(("button", "Continue to payment"), ("button", "Go back"))
    menu.build(StepPlan("click", "continue"), obs, ranker)
    before = len(ranker._cache)
    menu.build(StepPlan("click", "go back"), obs, ranker)
    assert len(ranker._cache) == before  # same elements: nothing re-embedded
