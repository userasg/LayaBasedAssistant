"""The mic component is a page of JavaScript that no Streamlit test harness runs, so a typo in it (an unescaped quote once) silently leaves the
Talk button dead. This loads the real page in real Chromium and drives the messages Streamlit sends it."""
from pathlib import Path

COMPONENT = Path(__file__).parents[2] / "src" / "laya_assistant" / "voice_component" / "index.html"


def render(page, **args):
    page.evaluate("a => window.postMessage({ type: 'streamlit:render', args: a, theme: { textColor: '#fafafa', primaryColor: '#00aa88' } }, '*')", args)
    page.wait_for_timeout(100)


def test_the_script_loads_without_errors_and_the_button_is_wired(page):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(COMPONENT.as_uri())
    page.wait_for_timeout(200)
    assert errors == []
    assert page.evaluate("typeof document.getElementById('btn').onclick") == "function"
    assert page.inner_text("#btn") == "🎙 Talk" and "open Notes" in page.inner_text("#hint")


def test_it_follows_the_app_state_and_theme(page):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(COMPONENT.as_uri())
    render(page, disabled=True, port=8765)
    assert page.is_disabled("#btn") and "Busy" in page.inner_text("#hint")
    render(page, disabled=False, port=8765)  # this is the line that once had a syntax error in its string
    assert not page.is_disabled("#btn") and "open Notes" in page.inner_text("#hint")
    assert page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--fg')").strip() == "#fafafa"  # the app's theme, not the OS's
    assert errors == []


def test_pressing_talk_without_a_voice_server_says_so_instead_of_hanging(page):
    page.goto(COMPONENT.as_uri())
    render(page, disabled=False, port=1)  # nothing listens on this port
    page.click("#btn")
    page.wait_for_function("document.getElementById('hint').textContent.startsWith('Could not start')", timeout=5000)
    assert page.inner_text("#btn") == "🎙 Talk" and not page.is_disabled("#btn")  # and it can be tried again
