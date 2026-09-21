import json

import pytest

from laya_assistant.cua.policy import DomainPolicy, host_of, is_secret_field, validate
from laya_assistant.cua.types import Candidate, Element, Observation


def obs(url="http://127.0.0.1:1/x.html", text="", app="", source="browser"):
    return Observation(source, "t", url, [], text=text, app=app)


def cand(kind="click", label="Continue", **kw):
    return Candidate("e1", kind, Element("e1", kw.pop("role", "button"), label, **kw), kw.get("value"))


@pytest.fixture
def dom(tmp_path):
    return DomainPolicy(tmp_path / "policy.json", ask_unknown=True)


@pytest.mark.parametrize("url", ["https://www.chase.com/login", "https://accounts.google.com/signin", "https://my.1password.com/", "https://www.paypal.com/x"])
def test_banking_payment_and_password_domains_are_blocked_outright(dom, url):
    assert dom.check(url).action == "block"


def test_unknown_domains_ask_once_then_are_remembered(dom):
    assert dom.check("https://example.org/a").action == "ask"
    dom.approve("https://example.org/a")
    assert dom.check("https://example.org/b").action == "allow"
    assert dom.check("http://localhost:8501").action == "allow"


def test_a_lookalike_domain_is_not_confused_with_a_blocked_one(dom):
    assert dom.check("https://notchase.com/").action == "ask"  # not chase.com, and not silently allowed either
    assert dom.check("https://secure.chase.com/").action == "block"


def test_the_users_policy_file_extends_the_lists(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"deny": ["internal.example"], "allow": ["docs.python.org"]}))
    d = DomainPolicy(p)
    assert d.check("https://internal.example/").action == "block" and d.check("https://docs.python.org/3/").action == "allow"


@pytest.mark.parametrize("label,typ,secret", [("Password", "password", True), ("Email", "email", False), ("Card number", "", True),
                                              ("Verification code", "", True), ("Search products", "", False), ("Your name", "", False)])
def test_secret_fields(label, typ, secret):
    assert is_secret_field(Element("e", "textbox", label, type=typ)) is secret


def test_typing_into_a_password_hands_over_to_the_user(dom):
    v = validate("type", cand("type", "Password", role="textbox", type="password", value="x"), obs(), dom)
    assert v.action == "handover" and "password" in v.reason


def test_card_numbers_are_never_typed_by_the_agent(dom):
    assert validate("type", cand("type", "Notes", role="textbox", value="4111 1111 1111 1111"), obs(), dom).action == "handover"


@pytest.mark.parametrize("label", ["Send message", "Place order", "Delete account", "Yes, delete everything", "Pay now", "Sign out", "Publish", "Transfer funds"])
def test_irreversible_labels_always_ask(dom, label):
    assert validate("click", cand("click", label), obs(), dom).action == "ask"


@pytest.mark.parametrize("label", ["Next", "Search", "Cancel", "Accept all cookies", "Export data", "Sign in", "Rename workspace"])
def test_ordinary_buttons_are_allowed(dom, label):
    assert validate("click", cand("click", label), obs(), dom).action == "allow"


def test_laya_irreversible_score_can_raise_an_ask_but_never_lower_one(dom):
    assert validate("click", cand("click", "Next"), obs(), dom, laya_irreversible=0.8).action == "ask"
    assert validate("click", cand("click", "Delete account"), obs(), dom, laya_irreversible=0.0).action == "ask"


def test_a_captcha_hands_over(dom):
    assert validate("click", cand("click", "Subscribe"), obs(text="please complete the captcha"), dom).action == "handover"


def test_blocked_domains_stop_everything_even_ordinary_clicks(dom):
    assert validate("click", cand("click", "Next"), obs(url="https://www.paypal.com/x"), dom).action == "block"


def test_first_visit_asks_before_acting_on_a_new_site(dom):
    assert validate("click", cand("click", "Next"), obs(url="https://example.org/"), dom).action == "ask"


def test_unapproved_desktop_apps_ask(dom):
    assert validate("click", cand("click", "OK"), obs(source="desktop", url=None, app="Notes"), dom, allowed_apps={"Finder"}).action == "ask"
    assert validate("click", cand("click", "OK"), obs(source="desktop", url=None, app="Finder"), dom, allowed_apps={"Finder"}).action == "allow"


def test_host_of():
    assert host_of("https://A.B.com:8080/x") == "a.b.com" and host_of(None) == ""


@pytest.mark.parametrize("dest,action", [("https://www.paypal.com/", "block"), ("https://accounts.google.com/", "block"),
                                         ("https://www.google.com/", "ask"), ("https://example.org/x", "ask")])
def test_navigating_is_judged_by_where_it_goes_not_where_it_starts(dom, dest, action):
    c = Candidate("navigate", "navigate", None, dest)
    assert validate("navigate", c, obs(url="http://127.0.0.1:1/x.html"), dom).action == action  # started on an approved page


def test_navigating_to_an_approved_site_or_localhost_is_allowed(dom):
    dom.approve("https://example.org/")
    assert validate("navigate", Candidate("navigate", "navigate", None, "https://example.org/y"), obs(), dom).action == "allow"
    assert validate("navigate", Candidate("navigate", "navigate", None, "http://localhost:8501/"), obs(), dom).action == "allow"


def test_the_agent_owned_browser_skips_first_visit_asks_but_never_the_deny_list(tmp_path):
    own = DomainPolicy(tmp_path / "p.json", ask_unknown=False)
    assert own.check("https://www.example.org/").action == "allow"   # no personal logins in that browser
    assert own.check("https://www.paypal.com/").action == "block"    # blocked sites stay blocked
    assert DomainPolicy(tmp_path / "p2.json", ask_unknown=True).check("https://www.example.org/").action == "ask"  # asks when you turn it on


def test_new_sites_do_not_ask_by_default_but_the_deny_list_still_blocks(tmp_path):
    from laya_assistant import approvals

    assert approvals.current.new_sites is False
    d = DomainPolicy(tmp_path / "p3.json")  # follows the Autonomy setting
    assert d.check("https://www.example.org/").action == "allow" and d.check("https://www.chase.com/").action == "block"
    approvals.current.new_sites = True
    try:
        assert d.check("https://www.example.org/").action == "ask"
    finally:
        approvals.current.new_sites = False
