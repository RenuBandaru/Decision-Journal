import pytest
from pydantic import ValidationError

from decision_journal.models import Challenge, Decision, Outcome


def make_decision(**overrides) -> Decision:
    """Build a valid Decision, overriding only the fields a test cares about."""
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
    """A decision with no outcome is unresolved and has no Brier score."""
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
    """The Brier score is the squared gap between confidence and the actual result."""
    decision = make_decision(confidence=confidence)
    decision.outcome = Outcome(what_happened="...", prediction_correct=correct)

    assert decision.is_resolved
    assert decision.brier_score == pytest.approx(expected)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_must_be_between_0_and_1(confidence):
    """Confidence values outside 0-1 fail validation."""
    with pytest.raises(ValidationError):
        make_decision(confidence=confidence)


def test_decisions_get_unique_ids():
    """Each new decision gets its own ID."""
    assert make_decision().id != make_decision().id


def test_round_trips_through_json():
    """A decision, including its challenges, survives saving to JSON and loading back."""
    decision = make_decision(challenges=[Challenge(question="What if the team is bad?")])

    assert Decision.model_validate_json(decision.model_dump_json()) == decision


def test_empty_title_is_rejected():
    """A decision must have a non-empty title."""
    with pytest.raises(ValidationError):
        make_decision(title="")
