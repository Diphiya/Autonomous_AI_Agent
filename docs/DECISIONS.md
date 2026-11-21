# Technical decision log

Each entry gives the decision, the alternatives considered, the reasons for the choice, and its consequences.

---

### D1. Single LLM router + deterministic policy, instead of free-form tool calling (ReAct) or a multi-agent setup

- **Decision:** One router node returns a structured `RouterDecision`. A deterministic `apply_policy` validates or overrides it before execution.
- **Alternatives:** (a) a prebuilt ReAct agent with native tool calling (`create_react_agent`); (b) a supervisor + specialist sub-agents; (c) a fixed sequential pipeline.
- **Why:** In ReAct, the guarantees we need (no grading without a quiz, no repeated calls, per-turn caps, correct "week N" resolution) would live in prompts, which are only suggestions. With an explicit decision object, those rules are code, and every override can be logged and tested. Multi-agent adds coordination cost and latency without a clear benefit for four small tools. A fixed pipeline can't handle the variety of requests (see REASONING §7).
- **Consequences:** It's easy to explain why each action happened (decision trace) and the guards have unit tests. The cost is a little more code than a prebuilt agent, and new tools need a policy entry.

### D2. Pydantic structured outputs with semantic validators and a repair-retry loop

- **Decision:** Every LLM call goes through `structured_call()`, which returns a validated model or raises a typed error. Validators enforce semantics, not just types: plan week count matches the request, each week fits the hour budget, MCQs have 4 distinct options and a valid key, a clarification includes its question. On failure, the validation error is sent back to the model ("repair prompt") and the call is retried.
- **Alternatives:** Free-text output + regex parsing; JSON mode without validation; blind retries.
- **Why:** "Valid JSON" is not "usable output". Most real LLM failures are well-formed but logically wrong. Feeding the exact error back fixes most of them in one retry, while a blind retry just rolls the dice again.
- **Consequences:** Invalid output never reaches the student or the state. One retry layer (no nested tool-level retries) keeps the attempt count predictable: `1 + LLM_MAX_RETRIES`.

### D3. Hybrid grading: MCQs graded by code, short answers by the LLM

- **Decision:** The quiz generator returns an answer key. MCQ answers are normalised (`b`, `B)`, `option b`, or the option text) and compared in code. Only free-text answers go to an LLM grader, one call per answer.
- **Alternatives:** Grade everything with one LLM call.
- **Why:** MCQ grading is exact; an LLM there only adds cost, latency and a chance of error. Per-question grading also allows **partial failure**: if a short-answer call fails, the MCQ results survive.
- **Consequences:** Grading is reproducible and mostly free. The answer key must be kept server-side, so the public state strips it and rendering never shows it (tested).

### D4. Two kinds of state + LangGraph checkpointer (in-memory default, SQLite optional)

- **Decision:** Session state (profile, plan, quiz, history, weak concepts, pending action) is persisted per `thread_id`. Turn state is reset each message. The router gets a compact structured summary plus a short message window.
- **Alternatives:** Send the full chat history as the only memory; an external DB with custom code; a vector store of past turns.
- **Why:** Facts the agent acts on (plan weeks, the quiz answer key) must be exact and must not depend on the LLM rereading old messages. Structured state keeps prompts small as conversations grow. The checkpointer gives persistence and session isolation for free, and SQLite makes it survive restarts with one env var.
- **Consequences:** Adding a field requires a schema change (an explicit, reviewable step). For multi-instance production deployment, swap in the Postgres checkpointer (see EVALUATION limitations).

### D5. Explicit execution limits and a clarification budget, enforced in code

- **Decision:** Per turn: ≤6 router steps and ≤4 tool calls. Failed tools aren't retried by the LLM. Duplicates are blocked. ≤2 consecutive clarifications, then proceed with stated defaults (never an invented subject) or offer options. LangGraph recursion limit as a backstop.
- **Alternatives:** Rely on the prompt and the recursion limit only.
- **Why:** Loops are the classic failure of agents and are expensive. Limits must hold even when the model misbehaves. Tests simulate both a router that repeats itself and one that keeps finding new work. The clarification budget handles a subtler loop: an over-cautious agent that keeps asking.
- **Consequences:** On rare legitimate large requests the agent stops early and says so. Students can always continue in the next turn. The defaults are always stated to the student so they can change them.

### D6. Recovery as a graph node, with a degraded-retry strategy

- **Decision:** A failed tool routes to `recover`. For quizzes it retries once with a simpler request (≤3 MCQ-only questions). Otherwise it adds an honest note and returns to the router, which can't call the failed tool again.
- **Alternatives:** Raise an error to the user; let the LLM decide whether to retry.
- **Why:** A simpler request is less likely to hit the same failure (fewer tokens, simpler schema), and a smaller result beats none. Keeping retry decisions out of the LLM prevents retry loops.
- **Consequences:** Behaviour under failure is predictable and testable (S3 and `test_persistent_failure_...`).

### D7. Provider-agnostic LLM layer + a deterministic offline mock

- **Decision:** Real models via LangChain `init_chat_model("provider:model")` + `with_structured_output`. A rule-based `MockLLM` implements the same interface, and a `FaultInjector` hooks into `structured_call` for any model.
- **Alternatives:** Hard-code one provider; test only against a live API.
- **Why:** Reviewers can run everything with no key and no cost. Tests are fast (≈1.5 s for 59) and deterministic. Failure demos are reproducible on demand rather than waiting for a real outage. The same scenario and eval scripts run unchanged against a real model.
- **Consequences:** Mock results show that the *workflow* is correct, **not** that the model makes good decisions. This is stated in every generated report. Real-model numbers come from re-running the eval with `LLM_MODEL` set.

### D8. Deterministic rendering instead of an LLM "final answer" call

- **Decision:** `respond` renders tool outputs to Markdown in code and adds the router's one-line message.
- **Alternatives:** A final LLM call to write the whole reply.
- **Why:** It saves one LLM call per turn (latency and cost), makes the answer key impossible to leak, and stops the content from being changed after validation.
- **Consequences:** Replies are a bit more templated. The router's `message` field adds the conversational touch.
