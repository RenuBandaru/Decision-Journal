"""Core data models for the decision journal."""

from datetime import UTC, date, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


class Challenge(BaseModel):
    """A question or counterargument the AI raised about a decision."""

    question: str
    response: str | None = None
    created_at: datetime = Field(default_factory=_now)


class Outcome(BaseModel):
    """What actually happened, recorded when the decision is reviewed."""

    what_happened: str
    prediction_correct: bool
    notes: str | None = None
    reviewed_at: datetime = Field(default_factory=_now)


class Decision(BaseModel):
    """A single journal entry: the choice made and the prediction behind it."""

    id: UUID = Field(default_factory=uuid4)
    title: str
    context: str
    options: list[str] = Field(default_factory=list)
    choice: str
    reasoning: str
    predicted_outcome: str
    confidence: float = Field(ge=0.0, le=1.0, description="Probability the prediction comes true")
    review_by: date | None = None
    challenges: list[Challenge] = Field(default_factory=list)
    outcome: Outcome | None = None
    created_at: datetime = Field(default_factory=_now)

    @property
    def is_resolved(self) -> bool:
        return self.outcome is not None

    @property
    def brier_score(self) -> float | None:
        """Squared error between confidence and result (0 is perfect, 1 is worst)."""
        if self.outcome is None:
            return None
        actual = 1.0 if self.outcome.prediction_correct else 0.0
        return (self.confidence - actual) ** 2
