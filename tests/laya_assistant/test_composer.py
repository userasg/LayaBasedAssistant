"""What happens to transcribed speech. A pure decision, tested on its own (the mic itself cannot be driven from a test)."""
import pytest

from laya_assistant.ui.composer import route_speech


@pytest.mark.parametrize("heard,done,locked,auto,expected", [
    ("open notes and write a shopping list", [], False, True, "send"),  # auto-send on, idle, a real sentence
    ("open notes and write a shopping list", [], False, False, "review"),  # the default: you check it first
    ("yes", [], False, True, "review"),  # a lone word is more likely noise than a request
    ("write the list", ["open Notes"], False, True, "send"),  # a short remainder after something already done is fine
    ("yes please go ahead", [], True, True, "review"),  # a question is waiting (or a turn is running): never unreviewed
    ("no", [], True, True, "review"),
    ("", [], False, True, "nothing"),
    ("", ["open Notes"], False, True, "nothing"),  # everything you said was a quick command and already ran
])
def test_speech_is_sent_unreviewed_only_when_it_is_safe_to(heard, done, locked, auto, expected):
    assert route_speech(heard, done, locked, auto) == expected
