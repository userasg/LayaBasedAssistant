from laya_assistant.cua import authority as A


def decision(step, laya, chosen, conf):
    return {"step": step, "laya_choice": laya, "chosen": chosen, "laya_conf": conf}


def outcome(step, ok):
    return {"kind": "outcome", "step": step, "verified": ok}


def rows(n_right, n_wrong, conf):
    out = []
    for i in range(n_right):
        out += [decision(f"r{i}", "e1", "e1", conf), outcome(f"r{i}", True)]
    for i in range(n_wrong):
        out += [decision(f"w{i}", "e2", "e1", conf), outcome(f"w{i}", True)]  # Laya disagreed with the final pick
    return out


def test_decisions_are_paired_with_their_outcomes_and_unmatched_rows_are_ignored():
    r = [decision("a", "e1", "e1", 0.9), outcome("a", True), decision("b", "e1", "e1", 0.9), outcome("zzz", True), {"step": "c"}]
    assert [d["step"] for d in A.paired(r)] == ["a"]


def test_accuracy_is_reported_per_confidence_band():
    rep = A.accuracy_report(rows(9, 1, 0.95) + rows(3, 3, 0.7))
    assert rep["bands"][(0.90, 1.01)] == {"n": 10, "correct": 9, "accuracy": 0.9}
    assert rep["bands"][(0.60, 0.75)]["accuracy"] == 0.5


def test_a_failed_verification_counts_against_laya_even_when_it_matched():
    r = [decision("a", "e1", "e1", 0.95), outcome("a", False)]
    assert A.accuracy_report(r)["bands"][(0.90, 1.01)]["correct"] == 0


def test_laya_earns_control_only_with_enough_samples_at_high_accuracy():
    assert A.min_confidence_for_authority(rows(29, 0, 0.95)) is None  # right every time, but only 29 samples
    assert A.min_confidence_for_authority(rows(30, 0, 0.95)) == 0.90
    assert A.min_confidence_for_authority(rows(25, 5, 0.95)) is None  # 30 samples but only 83%
    assert A.min_confidence_for_authority([]) is None


def test_authority_extends_down_a_band_when_the_evidence_supports_it():
    assert A.min_confidence_for_authority(rows(40, 0, 0.95) + rows(40, 0, 0.8)) == 0.75


def test_the_summary_says_what_is_missing():
    assert "no recorded" in A.summary_line([])
    assert "needs 30" in A.summary_line(rows(5, 1, 0.95))
    assert "earned control" in A.summary_line(rows(30, 0, 0.95))


def test_a_real_loop_run_produces_rows_the_authority_can_read(page, site_url, laya_model, ranker, tmp_path):
    from laya_assistant.cua.loop import Loop
    from laya_assistant.cua.playwright_driver import PlaywrightDriver
    from laya_assistant.cua.policy import DomainPolicy
    from laya_assistant.cua.recorder import StepRecorder
    from laya_assistant.cua.types import StepPlan

    rec = StepRecorder()
    page.goto(site_url + "/console.html")
    Loop(PlaywrightDriver(page), laya_model, ranker, DomainPolicy(tmp_path / "p.json"), rec).run([StepPlan("click", "Settings")])
    assert any(r.get("kind") == "outcome" and r["verified"] for r in rec.rows)  # the click was verified and recorded
    assert A.accuracy_report(rec.rows)["n"] >= 0  # readable whether or not this step was ambiguous enough to ask Laya
