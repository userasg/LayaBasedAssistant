"""Observation and action script, in a real Chromium on local pages (no models involved)."""
import pytest

from laya_assistant.cua.playwright_driver import PlaywrightDriver
from laya_assistant.cua.types import Action


@pytest.fixture
def drv(page, site_url):
    d = PlaywrightDriver(page)
    return d, site_url


def find(obs, label, role=None):
    return next(e for e in obs.elements if label.lower() in e.label.lower() and (role is None or e.role == role))


def test_observe_lists_visible_interactive_elements_with_roles_and_labels(drv):
    d, url = drv
    d.page.goto(url + "/index.html")
    obs = d.observe()
    assert obs.title == "Acme Shop - Home" and obs.host == "127.0.0.1"
    roles = {(e.role, e.label) for e in obs.elements}
    assert ("textbox", "Search products") in roles and ("button", "Search") in roles and ("link", "Contact us") in roles
    assert all(e.id.startswith("e") for e in obs.elements)


def test_element_ids_are_stable_across_observations(drv):
    d, url = drv
    d.page.goto(url + "/index.html")
    a = {e.label: e.id for e in d.observe().elements}
    b = {e.label: e.id for e in d.observe().elements}
    assert a == b


def test_password_values_are_masked_and_typing_into_them_is_refused_by_the_page_script(drv):
    d, url = drv
    d.page.goto(url + "/login.html")
    pw = find(d.observe(), "password", "textbox")
    assert pw.type == "password"
    res = d.act(Action("type", id=pw.id, value="hunter2"))
    assert res == {"ok": False, "error": "password_field"}
    assert find(d.observe(), "password", "textbox").value == ""


def test_type_click_navigation_and_stable_ids_work_end_to_end(drv):
    d, url = drv
    d.page.goto(url + "/index.html")
    obs = d.observe()
    assert d.act(Action("type", id=find(obs, "Search products", "textbox").id, value="sony"))["ok"]
    assert find(d.observe(), "Search products", "textbox").value == "sony"
    d.act(Action("click", id=find(d.observe(), "Search", "button").id))
    after = d.observe()
    assert "results.html" in after.url and any("Sony WH-1000XM5" in e.label for e in after.elements)


def test_select_check_and_radio_actions(drv):
    d, url = drv
    d.page.goto(url + "/form.html")
    obs = d.observe()
    assert d.act(Action("select", id=find(obs, "country").id, value="Germany"))["ok"]
    assert d.act(Action("check", id=find(obs, "agree").id, value="true"))["checked"] is True
    after = d.observe()
    assert find(after, "country").value == "Germany" and find(after, "agree").checked


def test_a_dialog_is_reported(drv):
    d, url = drv
    d.page.goto(url + "/danger.html")
    assert d.observe().dialogs == 0
    d.act(Action("click", id=find(d.observe(), "Delete account").id))
    assert d.observe().dialogs == 1


def test_missing_elements_return_an_error_not_an_exception(drv):
    d, url = drv
    d.page.goto(url + "/index.html")
    assert d.act(Action("click", id="e999")) == {"ok": False, "error": "element_not_found"}


def test_fingerprint_changes_when_the_page_changes(drv):
    d, url = drv
    d.page.goto(url + "/wizard.html")
    before = d.observe().fingerprint
    d.act(Action("click", id=find(d.observe(), "Next").id))
    assert d.observe().fingerprint != before
