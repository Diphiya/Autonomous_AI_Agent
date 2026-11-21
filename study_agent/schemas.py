"""Structured inputs/outputs for the router and every tool.

Every LLM call in the system returns one of these Pydantic models. Validators encode
the *semantic* rules (not just types) so that an LLM response that is well-formed JSON
but logically wrong (e.g. a 4-week plan with 3 weeks, an MCQ whose answer is "E") is
rejected and retried instead of being shown to the student.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Student profile
# --------------------------------------------------------------------------- #
class Level(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class ProfileUpdate(BaseModel):
    """Facts about the student extracted from the latest message. All optional:
    only fields the student actually mentioned should be filled."""

    goal: Optional[str] = Field(None, description="Overall learning goal, e.g. 'pass my ML exam'")
    subject: Optional[str] = Field(None, description="Main subject, e.g. 'machine learning'")
    topics: Optional[list[str]] = Field(None, description="Specific topics the student mentioned")
    level: Optional[Level] = None
    duration_weeks: Optional[int] = Field(None, ge=1, le=52)
    hours_per_week: Optional[float] = Field(None, gt=0, le=80)
    learning_style: Optional[str] = Field(None, description="e.g. 'examples first', 'visual', 'concise'")


class StudentProfile(BaseModel):
    goal: Optional[str] = None
    subject: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    level: Level = Level.beginner
    level_confirmed: bool = False
    duration_weeks: Optional[int] = None
    hours_per_week: Optional[float] = None
    learning_style: Optional[str] = None

    def merge(self, upd: ProfileUpdate | None) -> "StudentProfile":
        """Deterministic merge: new non-empty values overwrite, topics are unioned."""
        if upd is None:
            return self
        data = self.model_dump()
        for key, value in upd.model_dump(exclude_none=True).items():
            if key == "topics":
                seen = {t.lower() for t in data["topics"]}
                data["topics"] += [t for t in value if t and t.lower() not in seen]
            elif key == "level":
                data["level"] = value
                data["level_confirmed"] = True
            else:
                data[key] = value
        return StudentProfile(**data)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
class Action(str, Enum):
    create_study_plan = "create_study_plan"
    explain_topic = "explain_topic"
    generate_quiz = "generate_quiz"
    evaluate_answers = "evaluate_answers"
    ask_clarification = "ask_clarification"
    respond = "respond"  # finish the turn (answer directly / summarise tool results)


TOOL_ACTIONS = {
    Action.create_study_plan,
    Action.explain_topic,
    Action.generate_quiz,
    Action.evaluate_answers,
}


class RouterDecision(BaseModel):
    """The agent's decision for the next step of the current turn."""

    reasoning: str = Field(..., description="One or two sentences: why this action, given the state.")
    action: Action
    topic: Optional[str] = Field(None, description="Topic for explain_topic / generate_quiz")
    num_questions: Optional[int] = Field(None, ge=1, le=10)
    difficulty: Optional[Level] = None
    student_answers: Optional[list[str]] = Field(
        None, description="For evaluate_answers: the student's answers in question order"
    )
    clarification_question: Optional[str] = Field(None, description="Required when action=ask_clarification")
    pending_action: Optional[Action] = Field(
        None, description="For ask_clarification: the action you intend to take once the student answers"
    )
    message: Optional[str] = Field(
        None, description="For respond: short text to the student (summary, small talk, next step)"
    )
    profile_updates: Optional[ProfileUpdate] = None

    @model_validator(mode="after")
    def _check_required_fields(self):
        if self.action == Action.ask_clarification and not (self.clarification_question or "").strip():
            raise ValueError("ask_clarification requires a non-empty clarification_question")
        return self


# --------------------------------------------------------------------------- #
# Tool: study planner
# --------------------------------------------------------------------------- #
class StudyPlanInput(BaseModel):
    goal: str
    subject: str
    topics: list[str] = Field(default_factory=list)
    level: Level = Level.beginner
    duration_weeks: int = Field(..., ge=1, le=52)
    hours_per_week: float = Field(..., gt=0, le=80)
    learning_style: Optional[str] = None
    weak_concepts: list[str] = Field(default_factory=list)


class StudyWeek(BaseModel):
    week: int = Field(..., ge=1)
    focus: str
    topics: list[str] = Field(..., min_length=1)
    activities: list[str] = Field(..., min_length=1)
    estimated_hours: float = Field(..., gt=0)
    checkpoint: str = Field(..., description="How the student checks they've mastered the week")


class StudyPlan(BaseModel):
    goal: str
    duration_weeks: int
    hours_per_week: float
    weeks: list[StudyWeek]
    tips: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self):
        if len(self.weeks) != self.duration_weeks:
            raise ValueError(f"plan has {len(self.weeks)} weeks but duration_weeks={self.duration_weeks}")
        if [w.week for w in self.weeks] != list(range(1, self.duration_weeks + 1)):
            raise ValueError("weeks must be numbered 1..duration_weeks in order")
        for w in self.weeks:
            if w.estimated_hours > self.hours_per_week * 1.25:
                raise ValueError(
                    f"week {w.week} needs {w.estimated_hours}h, exceeds the student's "
                    f"{self.hours_per_week}h/week budget"
                )
        return self


# --------------------------------------------------------------------------- #
# Tool: explainer
# --------------------------------------------------------------------------- #
class ExplainInput(BaseModel):
    topic: str
    level: Level = Level.beginner
    subject: Optional[str] = None
    learning_style: Optional[str] = None
    focus_on: list[str] = Field(default_factory=list, description="Misconceptions/weak spots to address")


class Explanation(BaseModel):
    topic: str
    summary: str = Field(..., description="One-sentence summary")
    explanation: str
    analogy: Optional[str] = None
    example: str
    key_points: list[str] = Field(..., min_length=2, max_length=7)
    common_mistakes: list[str] = Field(default_factory=list)
    check_question: str = Field(..., description="A quick self-check question")


# --------------------------------------------------------------------------- #
# Tool: quiz generator
# --------------------------------------------------------------------------- #
class QuizInput(BaseModel):
    topic: str
    level: Level = Level.beginner
    num_questions: int = Field(4, ge=1, le=10)
    question_types: list[Literal["mcq", "short"]] = Field(default_factory=lambda: ["mcq", "short"])
    focus_on: list[str] = Field(default_factory=list)


class QuizQuestion(BaseModel):
    id: int = Field(..., ge=1)
    type: Literal["mcq", "short"]
    question: str
    options: Optional[list[str]] = Field(None, description="Exactly 4 options for mcq, labelled implicitly A-D")
    correct_option: Optional[Literal["A", "B", "C", "D"]] = None
    reference_answer: str = Field(..., description="Model answer / explanation of the correct option")
    concept: str = Field(..., description="The concept this question tests")

    @model_validator(mode="after")
    def _mcq_shape(self):
        if self.type == "mcq":
            if not self.options or len(self.options) != 4:
                raise ValueError(f"question {self.id}: mcq needs exactly 4 options")
            if self.correct_option is None:
                raise ValueError(f"question {self.id}: mcq needs correct_option A-D")
            if len({o.strip().lower() for o in self.options}) != 4:
                raise ValueError(f"question {self.id}: mcq options must be distinct")
        return self


class Quiz(BaseModel):
    topic: str
    level: Level
    questions: list[QuizQuestion] = Field(..., min_length=1)

    @field_validator("questions")
    @classmethod
    def _unique_sequential_ids(cls, qs: list[QuizQuestion]):
        if [q.id for q in qs] != list(range(1, len(qs) + 1)):
            raise ValueError("question ids must be 1..n in order")
        return qs


# --------------------------------------------------------------------------- #
# Tool: answer evaluator
# --------------------------------------------------------------------------- #
class QuestionResult(BaseModel):
    id: int
    correct: bool
    score: float = Field(..., ge=0, le=1)
    feedback: str
    concept: str


class ShortAnswerGrade(BaseModel):
    """What the LLM returns when grading ONE short answer."""

    score: float = Field(..., ge=0, le=1)
    feedback: str
    missing_points: list[str] = Field(default_factory=list)


class Evaluation(BaseModel):
    topic: str
    results: list[QuestionResult]
    total_score: float = Field(..., ge=0, le=1)
    weak_concepts: list[str] = Field(default_factory=list)
    summary_feedback: str
