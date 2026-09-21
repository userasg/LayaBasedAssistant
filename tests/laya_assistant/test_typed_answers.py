import pytest

from laya_assistant.session import decisions_from_text


@pytest.mark.parametrize("text", ["yes", "Yes.", "y", "ok", "OK!", "go ahead", "do it", "looks good", "  sure  ", "approve", "lgtm"])
def test_plain_yes_approves_every_pending_action(text):
    assert decisions_from_text(text, 2) == [{"type": "approve"}, {"type": "approve"}]


@pytest.mark.parametrize("text", ["no", "No.", "n", "nope", "cancel", "stop", "don't", "never mind", "reject"])
def test_plain_no_rejects_every_pending_action(text):
    assert decisions_from_text(text, 1) == [{"type": "reject"}]


@pytest.mark.parametrize("text", ["yes, but skip step 2", "no wait, use the other site", "actually make it three steps", "what does step 2 do?", "ok but only for the first file"])
def test_anything_else_is_the_users_reply_never_misread_as_a_yes(text):
    d = decisions_from_text(text, 1)
    assert d == [{"type": "respond", "message": text.strip()}]


def test_one_answer_covers_all_pending_actions():
    assert len(decisions_from_text("yes", 3)) == 3
