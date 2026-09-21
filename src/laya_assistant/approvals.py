"""What is allowed to ask you, in one place. The default is to ask about as little as possible:

  * irreversible things (sending, paying, deleting, posting) and dangerous commands always stop for you: that is not a setting;
  * plans are shown as they are made but not held for approval (`plan`);
  * shell commands on your Mac run unless they look risky (`host_commands`: "risky" | "always_ask" | "never_ask");
  * first visits to new sites in your own Chrome do not ask (`new_sites`); banks, payments and password sites stay blocked;
  * app windows come forward while the assistant works and this chat is hidden until it is done (`windows`); off = everything in the background.
Each is a switch in the Autonomy panel. A process-wide object: this is a single-user local app, and switches apply immediately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Approvals:
    plan: bool = False
    host_commands: str = "risky"
    new_sites: bool = False
    windows: bool = True  # show the app windows (and hide this chat) while a task runs; False: work in the background and stay in the chat


current = Approvals()

# Shell commands that change or remove things, or reach outside the machine's usual reversible operations.
RISKY_HOST = re.compile(
    r"\b(rm|rmdir|mv|cp -f|chmod|chown|kill|killall|pkill|dd|mkfs|shutdown|reboot|halt|diskutil|launchctl|defaults\s+(write|delete)|"
    r"crontab|git\s+(push|reset|clean)|npm\s+publish|brew\s+(uninstall|remove)|pip\s+uninstall|trash)\b"
    r"|osascript\b.*\b(delete|remove|do shell script|keystroke|empty trash|shut down|restart)\b"
    r"|(^|[^>&])>(?!>)\s*[~/\w]|\btee\b|\bcurl\b.*\s-(o|O)\b|\bsudo\b", re.I)


def host_command_needs_approval(command: str, mode: str | None = None) -> bool:
    mode = mode or current.host_commands
    if mode == "never_ask":
        return False
    if mode == "always_ask":
        return True
    return bool(RISKY_HOST.search(command))
