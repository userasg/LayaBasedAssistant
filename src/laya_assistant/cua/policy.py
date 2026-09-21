"""What the computer-use agent is allowed to do. Exact rules live in code; Laya adds a second opinion, never the first.

Real logins are in play (your Chrome), so the defaults are strict:
  * banking, payments, password managers and account-security pages are denied outright;
  * the first visit to any other domain asks you (localhost is pre-approved);
  * the agent never types a credential: password-like fields hand control to you;
  * irreversible steps (send, pay, buy, delete, post, publish, transfer, cancel...) always ask.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .. import approvals, config
from .types import Candidate, Element

DENY_DOMAINS = (
    # banking and payments
    "chase.com", "bankofamerica.com", "wellsfargo.com", "citi.com", "capitalone.com", "usbank.com", "americanexpress.com",
    "paypal.com", "venmo.com", "cash.app", "zellepay.com", "wise.com", "revolut.com", "monzo.com", "hsbc.com", "barclays.co.uk",
    "lloydsbank.com", "natwest.com", "santander.co.uk", "coinbase.com", "binance.com", "kraken.com", "robinhood.com",
    "fidelity.com", "schwab.com", "vanguard.com", "stripe.com",
    # password managers and identity
    "1password.com", "bitwarden.com", "lastpass.com", "dashlane.com", "keepersecurity.com", "okta.com", "auth0.com",
    "accounts.google.com", "myaccount.google.com", "appleid.apple.com", "login.microsoftonline.com", "account.microsoft.com",
)
PRE_APPROVED = ("localhost", "127.0.0.1", "::1")

IRREVERSIBLE_LABEL = re.compile(
    r"\b(send|submit|place order|buy|purchase|pay|checkout|check out|confirm (?:payment|order|purchase)|delete|remove|erase|"
    r"deactivate|close account|cancel (?:subscription|account|order)|unsubscribe|post|publish|transfer|withdraw|sign out|log out|"
    r"logout|yes,? delete|empty trash|format|reset)\b", re.I)
SECRET_LABEL = re.compile(r"pass(?:word|code|phrase)|\bpin\b|\bcvv\b|\bcvc\b|card number|security code|\bssn\b|social security|one-time|verification code|2fa|otp|secret", re.I)
CARD_LIKE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
CAPTCHA = re.compile(r"captcha|i'?m not a robot|verify you are human", re.I)


def host_of(url: str | None) -> str:
    return (urlparse(url or "").hostname or "").lower()


def _matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


@dataclass
class Verdict:
    action: str  # allow | ask | handover | block
    reason: str = ""


class DomainPolicy:
    """Deny list, per-session approvals, and the user's own additions in ~/.laya_assistant/policy.json."""

    def __init__(self, path: Path | None = None, ask_unknown: bool | None = None):
        self.ask_unknown = ask_unknown  # False for the agent-owned browser: no personal logins there, so only the deny list applies
        self.approved: set[str] = set(PRE_APPROVED)
        self.extra_deny: set[str] = set()
        self.always_allow: set[str] = set()
        p = path or (config.HOME / "policy.json")
        if p.exists():
            data = json.loads(p.read_text())
            self.extra_deny = set(data.get("deny", []))
            self.always_allow = set(data.get("allow", []))

    def check(self, url: str | None) -> Verdict:
        h = host_of(url)
        if not h:
            return Verdict("allow")
        if any(_matches(h, d) for d in (*DENY_DOMAINS, *self.extra_deny)):
            return Verdict("block", f"{h} is on the never-automate list (banking, payments, passwords or account security)")
        if h in self.approved or any(_matches(h, d) for d in self.always_allow):
            return Verdict("allow")
        ask = approvals.current.new_sites if self.ask_unknown is None else self.ask_unknown
        if not ask:
            return Verdict("allow")
        return Verdict("ask", f"first visit to {h} in this session")

    def approve(self, url: str | None) -> None:
        self.approved.add(host_of(url))


def is_secret_field(el: Element | None) -> bool:
    return bool(el) and (el.type == "password" or bool(SECRET_LABEL.search(el.label)))


def validate(step_kind: str, cand: Candidate, obs, domain: DomainPolicy, laya_irreversible: float = 0.0,
             irreversible_at: float = 0.6, allowed_apps: set[str] | None = None) -> Verdict:
    """Decide whether a chosen candidate may run. Order: hard blocks, handover to the user, then asks."""
    if cand.kind == "navigate":
        # the DESTINATION is what matters: checking only the page we are on let a plan walk to any site unasked
        dest = domain.check(cand.value)
        if dest.action != "allow":
            return dest
    if obs.source == "browser":
        v = domain.check(obs.url)
        if v.action == "block":
            return v
    if obs.source == "desktop" and allowed_apps is not None and obs.app not in allowed_apps:
        return Verdict("ask", f"{obs.app} has not been approved for this session")
    el = cand.element
    if cand.kind == "type":
        if is_secret_field(el):
            return Verdict("handover", "this is a password or secret field: you type it, not me")
        if cand.value and CARD_LIKE.search(cand.value):
            return Verdict("handover", "that looks like a card number: you enter it, not me")
    if CAPTCHA.search(obs.text or "") or (el and CAPTCHA.search(el.label)):
        return Verdict("handover", "a captcha needs a human")
    if v := (domain.check(obs.url) if obs.source == "browser" else None):
        if v.action == "ask":
            return v
    if cand.kind in ("click", "press_enter", "check") and el and IRREVERSIBLE_LABEL.search(el.label):
        return Verdict("ask", f"'{el.label}' looks irreversible")
    # Laya's irreversible score is noisy on short labels (it rated a calculator's "=" 0.72). In a native app the label rules above still catch
    # Delete / Send / Buy / Pay, so Laya alone must be very sure there; on the web it keeps the normal threshold.
    if getattr(obs, "source", "") == "desktop":
        irreversible_at = max(irreversible_at, 0.9)
    if cand.kind == "click" and laya_irreversible >= irreversible_at:
        return Verdict("ask", f"Laya rates this step irreversible (p={laya_irreversible:.2f})")
    return Verdict("allow")
