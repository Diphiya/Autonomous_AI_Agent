# Reasoning and technical challenges

## 1. Why is an agentic workflow appropriate here?

What a student needs next depends on information that only shows up *during* the interaction:

- **The next step depends on intermediate results.** Whether to move on or re-explain depends on the quiz score, which doesn't exist until the student answers. In S1 a 0% score makes the agent add an explanation step that nobody asked for explicitly. With 75% it doesn't.
- **Requests vary in shape and size.** "Explain X and give me 3 questions" needs two tools in order. "Quiz me on week 2" needs a lookup in earlier state. "Help me study" needs a question back. A fixed pipeline would need a hand-written branch for every combination.
- **Information arrives incrementally.** The student supplies subject, timeframe and hours across several turns. The agent must remember what is still missing (`pending_action`) and resume the task when it has enough.
- **Failures need a decision.** When a tool fails, the right move (retry simpler, explain the failure, or do something else useful) depends on context.

Still, the agent is **bounded**: a small set of typed tools, a deterministic policy, and hard limits. The problem needs flexible routing, not open-ended autonomy.

## 2. Deterministic logic vs LLM reasoning

Rule of thumb: **if a correct answer can be computed, compute it. Use the LLM only where you need language understanding or generation.**

| Deterministic | Why | LLM | Why |
|---|---|---|---|
| MCQ grading, scores, weak-concept tracking | Exact comparison; must be 100% correct and repeatable | Choosing the next action from free text | Needs language understanding and use of context |
| Answer parsing and the fast path | Regular format; saves an LLM call | Extracting profile facts ("about 6 h a week, total beginner") | Too many phrasings for regex |
| Limits, loop and duplicate blocking, preconditions | Safety properties must hold *even when the LLM is wrong* | Generating plans, explanations, questions | Open-ended content |
| "week N" / "it" resolution against stored state | The ground truth is in state; an LLM could hallucinate it | Grading free-text answers | Judging meaning, not keywords |
| Validation (week count, hour budget, MCQ shape) | Checkable invariants | Short summaries / next-step suggestions | Natural phrasing |
| Rendering; hiding the answer key | Must never leak, so no LLM in that path | | |

## 3. How does the system select the next action?

A **propose → dispose** loop (`router` node):

1. The LLM sees a structured state summary, recent history, the message, and the router prompt's rules. It returns a `RouterDecision` validated by Pydantic. A clarification without a question, for example, is rejected and repaired.
2. `apply_policy` checks it in a fixed order: limits → clarification budget → failed/duplicate blocks → per-action preconditions and argument resolution. It either executes the action as proposed, fills in its arguments, or overrides it with a recorded reason.
3. Conditional edges route to the chosen node. After each tool, control returns to the router, which sees the updated `tool_calls` log and decides whether more work is needed or the turn is done.
4. One shortcut skips the LLM: a message that is *only* a complete answer set goes straight to grading.

Every decision is traced with its `source` (`llm`, `deterministic`, `policy-override`, `recovery`, `tool`) and reasoning, so each step in a demo can be explained.

## 4. How are state and conversational context maintained?

- **Persistence:** LangGraph checkpointer keyed by `thread_id = session_id`. `MemorySaver` by default, or SQLite via `CHECKPOINT_DB`, which survives restarts (tested in `test_sqlite_checkpointer_persists_across_restarts`). Sessions are isolated.
- **Two scopes:** session state (profile, plan, active quiz, history, weak concepts, last topic, pending action) persists. Turn state (steps, tool log, outputs, trace) is reset by `ingest`, so the limits are per-turn.
- **Two context channels for the LLM:** a *structured* summary for facts that must be exact, plus a *windowed* message history for conversational nuance. The structured part keeps prompts bounded as conversations grow and prevents "forgetting" the plan after many turns.
- **Deterministic merges:** profile updates from the LLM are merged by code. Scalars overwrite and topics are unioned, so a bad extraction can't wipe the profile.

## 5. How does it avoid unnecessary tool calls and infinite loops?

Layered defences, from cheapest to last resort:

1. **Prompt rules:** don't repeat a successful tool, don't retry a failed one, respond when done.
2. **Visible tool log:** the router sees exactly what already ran this turn.
3. **Duplicate block:** same action + same normalised topic already succeeded → `respond`. Demonstrated in S5 with a simulated faulty router that keeps proposing the same call.
4. **Failed-tool block:** at most one explicit, degraded retry, owned by `recover` rather than the LLM.
5. **Tool-call cap (4) and step cap (6) per turn:** stop an agent that keeps proposing *different* calls. Tested with `LoopingLLM`.
6. **Clarification budget (2):** prevents the polite loop of endlessly asking questions.
7. **Fast path:** grading pure answer messages costs no routing call at all.
8. **LangGraph `recursion_limit`:** backstop that the service catches and turns into a friendly reply.

## 6. How does it handle incorrect decisions and tool failures?

**Incorrect decisions**

- *Impossible actions* (grading with no quiz, week 5 of a 2-week plan) are caught by preconditions and turned into a helpful reply or a precise clarification (S5).
- *Missing arguments* are filled from state (last topic, plan week) or turned into a question.
- *Low-value repetition* is blocked by the duplicate and limit guards.
- *Wrong-but-valid choices* (for example, explaining when the student wanted a quiz) can't be caught deterministically. They're limited by the router prompt, the visible state, and an easy correction path: the student says "no, a quiz", and the next turn has the full context. They are **measured** by the eval harness (§8).

**Tool failures**

- Invalid output → repair-retry with the validation error included. Transient errors → backoff.
- Persistent failure → `recover`: a degraded retry (shorter MCQ-only quiz) or an honest note ("I couldn't generate the explanation right now… nothing was lost"). Session state is untouched and the next turn works (S3).
- Partial failure → partial result. If short-answer grading fails, MCQ grades are still returned.
- Router failure → deterministic fallback reply.
- Every failure is reproducible on demand through fault injection, with the mock or a real LLM.

## 7. Agentic vs conventional sequential workflow

| | Sequential pipeline (e.g. profile → plan → quiz → grade) | This agentic workflow |
|---|---|---|
| Control flow | Fixed order in code | Chosen per step by router + policy |
| Variable requests ("explain then quiz", "quiz me on week 2", "help me study") | Every path hand-coded or forced into the fixed order | Handled by the same loop |
| Adapting to results | Only branches you wrote in advance | Router sees the score and adds a remediation step |
| Missing info | Rigid form or fails | Asks targeted questions, resumes the pending task |
| Predictability / testability | High: same input, same path | Lower; recovered through the policy layer, structured outputs, traces and eval cases |
| Cost / latency | Fewer LLM calls | One routing call per step (≈2–3 per turn). The fast path and deterministic rendering cut this |
| Failure modes | Simple | Loops, wrong tool choice → explicit guards and limits needed |

**Conclusion:** a sequential flow is better for a fixed, single-purpose task (for example, "generate a quiz from this PDF"). For an open-ended conversational tutor, the agentic approach is worth it, *as long as* the deterministic parts stay deterministic. This design is a hybrid: agentic routing on top of deterministic guardrails and tools.

## 8. How could decision-making quality be evaluated?

Implemented in `scripts/eval_decisions.py` with 23 labelled cases in `eval/decision_cases.json`:

- **First-action accuracy:** executed action ∈ acceptable set, measured **twice**: *raw LLM proposal* and *system after policy*. The gap shows how much the guardrails compensate. Mock run: 87% raw → 100% system.
- **Tool-sequence exact match** for the whole turn (catches missing or extra steps in multi-step requests).
- **Clarification precision/recall:** asked when needed, not when the information was already there.
- **Efficiency:** tool calls per turn, policy-override rate, latency.
- **Consistency:** the same case run *k* times (`--repeats`) should yield the same action sequence.
- **Failure categories:** cases with injected faults must still reach the expected outcome.

Beyond this harness: an LLM-as-judge rubric for content quality (plan feasibility, explanation correctness, question ambiguity), human review of a sample of traces, and online signals such as how often students correct the agent ("no, I meant…"), quiz score improvement after remediation, and task completion rate.
