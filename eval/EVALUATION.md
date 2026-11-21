# Evaluation report

## 1. What was measured and how

| Layer | Method | Artifact |
|---|---|---|
| Component correctness | 59 pytest tests: schemas and semantic validators, retry/repair, fault injector, every policy guard, answer parsing, evaluator, end-to-end multi-turn flows, persistence, API, and regression tests for bugs found with the real model | `tests/` |
| Multi-step behaviour, real model | Manual sessions in the web UI with **Claude Haiku (`anthropic:claude-haiku-5-5`)** | §2.1 |
| Multi-step behaviour, reproducible | 5 scripted scenarios, 20 turns, with full decision traces (offline mock) | `docs/DEMO_SCENARIOS.md` |
| Decision quality | 23 labelled cases in 7 categories, **run against Claude Haiku with 3 repeats** (69 conversations) and against the mock | `eval/decision_cases.json`, `scripts/eval_decisions.py` |
| Failure handling | Faults injected at the LLM boundary (invalid output, timeout, HTTP 503), plus real validation failures observed from Claude | `FaultInjector`, §2.3 |

**Two LLM backends, two purposes.** The offline **mock** is deterministic. It checks that the *workflow* (routing, state, guards, recovery) behaves as designed, reproducibly and without an API key. It says nothing about decision quality because it was written alongside the test cases. **Claude Haiku** provides the real evidence of decision quality and content quality.

---

## 2. Results with Claude Haiku

### 2.1 Manual multi-step scenario (web UI)

| Turn | Student | Agent behaviour (from the decision trace) |
|---|---|---|
| 1 | "I'm a beginner… machine learning for my exam… 4 weeks, 6 hours a week. Make me a study plan." | `create_study_plan` (all prerequisites known). It produced a concrete 4-week plan with measurable weekly checkpoints. It **did not invent the exam syllabus**: a tip tells the student to check their course outline. 11.8 s. |
| 2 | "Quiz me on week 2" | `generate_quiz`. "Week 2" was resolved from the stored plan (deterministic). 5 questions: 3 MCQ + 2 short answer. |
| 3 | Mostly wrong answers | Graded at 20%. Weak concepts recorded. (The run before the fix exposed bug B1, §2.4.) |
| 4 | "Quiz me again on the same topic" | A new quiz **focused on the weak concepts** (MSE, sigmoid, precision). Claude's closing message says so explicitly. |
| 5 | Correct answers | **100%**, so **no remediation step**. The agent suggests moving on to week 3. |

So the same tool can lead to a different next step depending on the intermediate result: low score → remediation, high score → progress. Short answers were graded on meaning rather than keywords, with specific feedback (e.g. it noted that `w·x + b` was not mentioned, without deducting marks).

### 2.2 Decision-quality evaluation (23 cases × 3 repeats)

Raw output: `docs/eval_results_anthropic_claude_haiku_5_5_run1.json` (harness v1).

| Metric | Reported by harness v1 | After reviewing every case | Mock (for comparison) |
|---|---|---|---|
| First-action accuracy (system) | 100% | **100%** (23/23) | 100% |
| First-action accuracy (**raw LLM**, before policy) | 100% | **100%** | 87% |
| Tool-sequence match | 87% (20/23) | **20/20 assessable** (3 cases not assessable, see below) | 100% |
| Clarification recall | 100% | **100%** | 100% |
| Clarification precision | 86% | **100%** (formula bug, see below) | 100% |
| Policy overrides per turn | 0 | **0** | 0.13 |
| Consistency over 3 repeats | 96% (22/23) | **96%** | 100% |
| Median latency per turn | 5.7 s | 5.7 s | 0.006 s |

**Review of the four reported misses.** All of them came from the harness, not from Claude:

| Case | Finding | Classification |
|---|---|---|
| D12 "Quiz me on week 2" | Correct tool. The case also checked that the topic contained "overfitting", which is true only in the mock's plan; Claude's plans order topics differently. | Label tied to mock content |
| D17 answers `1. B 2. C 3. B …` | These answers were correct for the *mock's* quiz. Against Claude's own quiz they were largely wrong, and Claude graded them **then added an explanation**: the intended adaptive behaviour. | Label tied to mock content (Claude's behaviour consistent with design) |
| D16 answers `1. A 2. A 3. A …` | Graded and responded without remediation. Whether that is right depends on the score against Claude's quiz, which harness v1 didn't record. | Not assessable with v1 |
| Clarification precision | The extra clarification was D18 ("Check my answers" with no quiz), where a question is listed as correct. The v1 formula counted any clarification outside clarify-only cases as unneeded. | Metric bug |

**Fix (harness v2, in this repository).** D16/D17 now check *adaptive* behaviour independently of the model: graded, and `explain_topic` follows **if and only if** the score is < 60% (the score is recorded). D12 compares the resolved topic with week 2 of the plan that was *actually generated*. Clarification precision counts a question as correct wherever one is listed as acceptable. Re-run with `python scripts/eval_decisions.py --repeats 3` to get v2 numbers for Claude.

**Interpretation.**
- Claude Haiku made the right routing decision in every case, *including* the invalid-request cases (grading with no quiz, a nonexistent plan week, a plan week with no plan), which the mock got wrong and the policy had to correct. With this model the **policy layer was never triggered (0 overrides)**. It remains necessary as a safety net: bug B1 (§2.4) is a real case where Claude *did* propose an invalid action, and a regression test now covers it.
- **Latency** depends on the kind of turn: about 1.2–1.9 s for clarifications and small talk (one LLM call), 4–9 s for one tool, 10–12 s for study plans (long structured output), and 13–14 s for two-tool turns.
- **Consistency:** 22 of 23 cases produced the same action sequence in all three repeats. v1 didn't record which case varied; v2 lists inconsistent cases with the sequences observed.

### 2.3 Failure handling observed with the real model

In addition to the injected faults (all handled as designed in all 3 repeats), the evaluation produced **two genuine invalid outputs from Claude**. Both were rejected by Pydantic validation and repaired on the next attempt:

1. An `Explanation` with the required `example` field missing.
2. A `StudyPlan` where week 3 had `estimated_hours = 0.0` (violates `> 0`).

Without validation, both would have reached the student: one as a broken explanation, one as a plan with an empty week. This supports decision D2 (semantic validation + repair-retry).

A **provider configuration error** was also handled without a crash. The first live run failed because the model rejects the `temperature` parameter. The agent returned a friendly fallback, and the decision trace showed the exact provider error, which made it a 5-minute diagnosis. `temperature` is now only sent when configured.

### 2.4 Bugs found with the real model (all fixed, with regression tests)

| # | Bug | Root cause | Fix |
|---|---|---|---|
| B1 | After grading a partially answered quiz, the student was told "There's no open quiz to grade" | The router prompt referred to `tool_calls_this_turn`, but the state summary sent the field as `tool_calls`, so Claude didn't see that grading was already done and proposed it again. The precondition guard then fired with a misleading message. | Field renamed; explicit `turn_progress` line ("answers HAVE BEEN GRADED… score below 60%: rule 3 applies"); repeated grading in a turn is now treated as a duplicate. Tests: `test_regrading_in_same_turn_is_a_duplicate_not_no_quiz`, `test_partial_answers_go_through_router_and_regrade_is_blocked` |
| B2 | `1*1 + 2*1` rendered as "11 + 21" in the UI | The UI's Markdown renderer treated `*` as italics | Italics only between word boundaries |
| B3 | Every request failed with HTTP 400 | `temperature` always sent; rejected by the model | Only sent if `LLM_TEMPERATURE` is set. Test: `test_temperature_not_sent_when_unset` |

None of these were visible with the mock. B1 in particular is the kind of error only a real model produces: a confident, well-formed but wrong decision. It shows why the evaluation used both backends.

---

## 3. Results with the offline mock (reproducible)

`59 passed` in about 1.5 s. The five scripted scenarios in `DEMO_SCENARIOS.md` all behave as designed:

- **S1** plan → quiz → low score → remediation → retry
- **S2** clarifications and the clarification budget
- **S3** repair-retry, degraded quiz fallback, graceful failure, router retry
- **S4** two tools chained from one message, plus messy answer formats
- **S5** policy overrides: no quiz, week out of range, duplicate-call loop

Decision harness (v2): 100% system accuracy and **87% raw** accuracy. The 3 raw misses (D18–D20) were corrected by the policy layer, which demonstrates the guards working.

---

## 4. Content-quality observations (Claude Haiku)

| Observation | Example | Impact |
|---|---|---|
| A reference answer contradicted itself | Q5 (mean/median): "rises to 14.1 approximately: sum 131/8 = 16.375, so mean rises to 16.375" | A student could learn a wrong number. Validation checks structure, not truth |
| Imprecise feedback text | Q3: "0.67 is recall-like (40/60…) reversed"; 40/60 *is* the recall | Minor confusion |
| Questions repeated across quizzes | The sigmoid output-range question appeared almost unchanged in two consecutive quizzes | Less practice value |
| Closing message repeats the plan | After the plan, the summary restated each week's topics | Verbose reply |
| Mixed answer formats handled well | `1. A 2. A 3. A 4. not sure` against a 5-question quiz: missing answers marked "not answered"; "not sure" flagged as unreadable with the correct option shown | Positive |

---

## 5. Known limitations

1. **Content correctness isn't verified** (§4). Schema validation guarantees structure, not facts. The self-contradicting reference answer is a real example.
2. **Wrong-but-valid decisions aren't caught** by the policy (e.g. explaining when the student wanted a quiz). They're limited by visible state and easy correction in the next turn. B1 shows such decisions do happen.
3. **Latency** of 5–14 s per turn with a real model. Calls are sequential and responses aren't streamed.
4. **No memory of past questions,** so quizzes can repeat themselves.
5. **Short-answer grading is LLM judgement** and can vary.
6. **Timeouts don't cancel work.** A timed-out call keeps running in the background and still uses tokens.
7. **Single-process persistence.** Memory or SQLite only; multi-instance deployment needs a Postgres checkpointer and authentication.
8. **Plan editing is coarse.** "Make week 2 lighter" regenerates the whole plan.
9. **Evaluation set is small** (23 cases, English only, one real model). v1 labels were partly tied to mock content (fixed in v2).

## 6. Improvements

- **Verify quiz content:** a second, cheaper LLM pass answers each question independently; drop questions where its answer disagrees with the key or the reference answer is inconsistent. This targets the §4 issues directly.
- **Avoid repeated questions:** pass recent question texts from `quiz_history` to the quiz tool.
- **Ground content** in the student's syllabus or notes (RAG) and cite sources.
- **Shorter closing messages:** instruct the router not to restate tool output, only the next step.
- **Streaming** (SSE) so long plans appear progressively. Route with a smaller or faster model and generate content with a stronger one.
- **Confidence-aware routing:** a `confidence` field, with a clarification below a calibrated threshold.
- **Larger eval set:** paraphrases, German as well as English, adversarial inputs, several models, and LLM-as-judge scoring of content quality.
