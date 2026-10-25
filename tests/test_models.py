import pytest
from pydantic import ValidationError
from decision_journal.models import Decision, Challenge, Outcome


def test_confidence_above_1_is_rejected():
    with pytest.raises(ValidationError):
        Decision(title="Take the job", context="Two offers",
                 choice="Job A", reasoning="Better team",
                 predicted_outcome="I'll be happy there", confidence=1.5)


def make_decision(**overrides) -> Decision:
    fields = {
        "title": "Take the job",
        "context": "Offer from X",
        "choice": "Accept",
        "reasoning": "More growth",
        "predicted_outcome": "Happier in 6 months",
        "confidence": 0.8,
    }
    return Decision(**(fields | overrides))


def test_new_decision_is_unresolved():
    decision = make_decision()

    assert not decision.is_resolved
    assert decision.brier_score is None


@pytest.mark.parametrize(
    ("confidence", "correct", "expected"),
    [
        (0.8, True, 0.04),
        (0.8, False, 0.64),
        (1.0, True, 0.0),
        (0.5, False, 0.25),
    ],
)
def test_brier_score(confidence, correct, expected):
    decision = make_decision(confidence=confidence)
    decision.outcome = Outcome(what_happened="...", prediction_correct=correct)

    assert decision.is_resolved
    assert decision.brier_score == pytest.approx(expected)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_must_be_between_0_and_1(confidence):
    with pytest.raises(ValidationError):
        make_decision(confidence=confidence)


def test_decisions_get_unique_ids():
    assert make_decision().id != make_decision().id


def test_round_trips_through_json():
    decision = make_decision(challenges=[Challenge(question="What if the team is bad?")])

    assert Decision.model_validate_json(decision.model_dump_json()) == decision
