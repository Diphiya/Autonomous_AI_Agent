# Evaluation report

## 1. What was measured and how

| Layer | Method | Artifact |
|---|---|---|
| Correctness of components | 59 pytest tests: schemas and semantic validators, retry/repair, fault injector, every policy guard, answer parsing, evaluator, end-to-end multi-turn flows, persistence, API | `tests/` |
| Multi-step behaviour | 5 scripted scenarios, 20 turns, with full decision traces | `docs/DEMO_SCENARIOS.md`, `scripts/run_scenarios.py` |
| Decision quality | 23 labelled cases in 7 categories; raw-LLM vs system accuracy, tool-sequence match, clarification precision/recall, consistency | `eval/decision_cases.json`, `scripts/eval_decisions.py`, `docs/eval_results_<model>.md` |
| Failure handling | Faults injected at the LLM boundary (invalid output, timeout, 5xx) in scenarios, tests and eval cases | `FaultInjector`, `POST /api/sessions/{id}/faults` |

> **Important caveat.** All numbers below were produced with the offline **mock LLM**, which is rule-based and was written alongside the test cases. They prove that the **workflow** (routing, state, guards, recovery, persistence) behaves as designed and reproducibly. They are **not** evidence of a real model's decision quality. To get those, run:
> ```bash
> LLM_MODEL=openai:gpt-4o-mini python scripts/eval_decisions.py --repeats 3
> LLM_MODEL=openai:gpt-4o-mini python scripts/run_scenarios.py --out docs/DEMO_SCENARIOS_real.md --json ""
> ```
> These write `docs/eval_results_<model>.md` with the same metrics.

## 2. Results

### Tests
`59 passed in ~1.5 s`.

### Multi-step scenarios (`DEMO_SCENARIOS.md`)

| ID | Scenario | Actions the agent selected (and why) | Result |
|---|---|---|---|
| S1 | Plan → quiz on week 1 → wrong answers → retry | `create_study_plan` (all prerequisites known) → `generate_quiz` ("week 1" resolved from the plan) → **deterministic** `evaluate_answers` (pure answer set) → `explain_topic` (**added by the agent** because the score was 0% < 60%, focused on the missed concepts) → new quiz → 75% → `respond` with no remediation | ✅ adapts to intermediate results; weak concepts carried across turns |
| S2 | "Help me study" → "machine learning" → "a plan please, I have 3 weeks" → "explain the first topic" | `ask_clarification` ×2 (nothing actionable yet; then subject known but not the kind of help) → plan **policy-override**: clarification budget used up, so plan with stated default 5 h/week and tell the student → "first topic" resolved from the plan | ✅ never re-asks known facts; can't get stuck asking questions |
| S3 | Faults: quiz invalid once; explanation times out ×3; quiz invalid ×6; router 503 once | Repair-retry succeeds on attempt 2 → explanation fails after 3 attempts → `recover` → honest message → quiz: full attempt fails → **degraded retry** fails → honest message → router 503 retried transparently → next turn works | ✅ no crash, no lost state, no retry loop |
| S4 | "Explain gradient descent and then give me 3 practice questions" → "I think 1. B 2) b 3. c" | `explain_topic` → router sees it is done and the quiz is still pending → `generate_quiz(n=3)` → `respond`; answers in mixed formats parsed → fast-path grading 100% | ✅ tool chaining in one turn; robust parsing |
| S5 | Grade with no quiz; week 5 of a 2-week plan; faulty router repeating itself | **policy-override** → respond with an offer; **policy-override** → precise clarification ("Your plan only has 2 weeks"); duplicate call blocked after the first explanation | ✅ incorrect decisions caught deterministically |

Latency with the mock is 4–20 ms per turn (pure orchestration overhead). Injected timeouts add backoff time (S3: 162 ms). With a real model, expect roughly one LLM call per router step plus one per tool: about 2–3 sequential calls for a typical turn.

### Decision-quality harness (mock)

| Metric | Value |
|---|---|
| System first-action accuracy | 100% (23/23) |
| **Raw LLM** first-action accuracy | 87% (20/23) |
| Tool-sequence exact match | 100% |
| Clarification precision / recall | 100% / 100% |
| Policy overrides per turn | 0.13 |
| Consistency across 2 repeats | 100% |

The 3 raw misses are the *invalid-request* cases (D18–D20): the router proposed grading with no quiz, or a quiz on a week that doesn't exist, and the policy corrected all three. With a real model, the **gap between raw and system accuracy** is the key diagnostic. A large gap means the prompt or model should improve. Overrides concentrated in one category point to the rule the model doesn't follow.

## 3. Known limitations

1. **No real-model numbers in this submission.** All reported metrics are mock-based (see caveat). Run the eval with a real model before drawing conclusions about decision quality.
2. **Wrong-but-valid decisions aren't caught.** The policy catches impossible or redundant actions, but not "explained when they wanted a quiz". Mitigation today: visible state, easy correction next turn. Future: confidence field + ask when low, and student-correction tracking.
3. **Content correctness isn't verified.** Explanations and questions come from the LLM without retrieval grounding. A model can produce a subtly wrong fact or an ambiguous MCQ. Schema validation checks structure, not truth.
4. **Short-answer grading is LLM judgement.** It can be lenient or strict. The mock's keyword grader is only a placeholder.
5. **Latency is serial.** Each router step is a sequential LLM call. There is no streaming of partial results.
6. **Timeouts don't cancel work.** A timed-out tool thread finishes in the background; its result is discarded but the tokens are still spent.
7. **Single-process persistence.** `MemorySaver` is lost on restart; SQLite suits one instance. Multi-instance deployment needs `PostgresSaver`, plus auth so that `session_id` isn't the only access control.
8. **Plan editing is coarse.** "Make week 2 lighter" regenerates the plan instead of patching it.
9. **Mock language understanding is regex-based.** Demos use phrasings it understands; that is the point of a mock, but it shouldn't be read as the system's language ability.

## 4. Possible improvements

- **Ground content in course material** (RAG over the student's syllabus or notes) and cite sources in explanations and questions.
- **Self-check for quizzes:** a second, cheap LLM pass (or the same model in a verifier role) answers its own questions; drop questions where its answer disagrees with the key.
- **Confidence-aware routing:** add `confidence` to `RouterDecision` and ask a clarification when it is low; calibrate the threshold on the eval set.
- **Cheaper, faster routing:** a small model or a fine-tuned classifier for routing, a larger model for content; cache explanations per (topic, level).
- **Spaced repetition:** schedule reviews of weak concepts by date, driven by `quiz_history`.
- **Observability:** export traces to LangSmith or OpenTelemetry; dashboard override rate, failure rate, tokens per turn.
- **Streaming responses** over SSE so plans and explanations appear progressively.
- **Larger eval set:** paraphrases, other languages, adversarial inputs, and LLM-as-judge scoring of content quality.
