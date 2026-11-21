"""Evaluate decision-making quality on a labelled set of cases.

    python scripts/eval_decisions.py                               # offline mock
    LLM_MODEL=openai:gpt-4o-mini python scripts/eval_decisions.py  # real model
    python scripts/eval_decisions.py --repeats 3                   # measure consistency

Metrics (see docs/EVALUATION.md for definitions):
  * system first-action accuracy  — executed action matches the label
  * raw LLM first-action accuracy — what the LLM *proposed*, before the policy layer
  * tool-sequence exact match     — the whole turn's successful tool calls match
  * clarification precision / recall
  * policy override rate, avg tool calls & LLM routing calls per turn, latency
  * consistency across repeats (same executed action sequence)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study_agent.agent import StudyAssistant  # noqa: E402
from study_agent.config import get_settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def first_decision(trace):
    for t in trace:
        if t["node"] in ("router", "ingest"):
            m = re.search(r"LLM proposed (\w+) →", t.get("detail") or "")
            proposed = m.group(1) if m else (t["decision"] if t["source"] in ("llm", "deterministic") else None)
            return t["decision"], proposed, t["source"]
    return None, None, None


def run_case(bot: StudyAssistant, case: dict, sid: str) -> dict:
    for m in case.get("setup", []):
        bot.chat(m, sid)
    for task, kinds in (case.get("faults") or {}).items():
        bot.inject_faults(sid, task, kinds)
    t0 = time.perf_counter()
    r = bot.chat(case["message"], sid)
    latency = time.perf_counter() - t0
    executed, proposed, source = first_decision(r.trace)
    tools = [c["action"] for c in r.tool_calls if c["status"] == "ok"]
    ok_first = executed in case["expect_first"]
    score = None
    for c in r.tool_calls:
        m = re.search(r"score (\d+)%", c.get("summary") or "")
        if c["action"] == "evaluate_answers" and m:
            score = int(m.group(1)) / 100
    if case.get("adaptive"):
        # Correct = graded, then explain_topic if and only if the score was below 60%.
        ok_tools = bool(tools) and tools[0] == "evaluate_answers" and score is not None \
            and (("explain_topic" in tools) == (score < 0.6))
    else:
        ok_tools = tools == case["expect_tools"]
    ok_args = True
    for key, want in (case.get("expect_arg") or {}).items():
        k = key.rstrip("~")
        got = " ".join(str(c["args"].get(k, "")) for c in r.tool_calls)
        ok_args &= want.lower() in got.lower()
    if case.get("expect_arg_plan_week"):
        plan = bot.public_state(sid).get("study_plan") or {}
        n = case["expect_arg_plan_week"]
        want = ", ".join(plan["weeks"][n - 1]["topics"]).lower() if plan.get("weeks") and len(plan["weeks"]) >= n else None
        got = [str(c["args"].get("topic", "")).lower() for c in r.tool_calls]
        ok_args &= want is not None and want in got
    if case.get("expect_reply~"):
        ok_args &= case["expect_reply~"].lower() in r.reply.lower()
    return {
        "id": case["id"], "category": case["category"], "message": case["message"],
        "expected_first": case["expect_first"], "executed_first": executed, "llm_proposed_first": proposed,
        "first_source": source, "expected_tools": case["expect_tools"], "tools": tools,
        "pass_first": ok_first, "pass_tools": ok_tools and ok_args,
        "raw_llm_correct": proposed in case["expect_first"] if proposed else None,
        "overrides": sum(1 for t in r.trace if t["source"] == "policy-override"),
        "eval_score": score,
        "tool_calls": len(r.tool_calls), "latency_s": round(latency, 3),
        "action_sequence": [t["decision"] for t in r.trace if t["node"] in ("router", "ingest")],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "eval" / "decision_cases.json"))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cases = json.loads(Path(a.cases).read_text())
    settings = get_settings()
    bot = StudyAssistant(settings)
    model = bot.llm.name
    runs: list[list[dict]] = []
    for rep in range(a.repeats):
        runs.append([run_case(bot, c, f"eval-{rep}-{c['id']}") for c in cases])
    rows = runs[0]

    n = len(rows)
    # recall: of the cases where ONLY a question is right, how many got one;
    # precision: of the questions asked, how many were acceptable (a question is listed as correct for that case).
    clar_expected = {r["id"] for r in rows if r["expected_first"] == ["ask_clarification"]}
    clar_made = {r["id"] for r in rows if r["executed_first"] == "ask_clarification"}
    clar_ok = {r["id"] for r in rows if r["executed_first"] == "ask_clarification" and "ask_clarification" in r["expected_first"]}
    raw = [r["raw_llm_correct"] for r in rows if r["raw_llm_correct"] is not None]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    consistency = None
    if a.repeats > 1:
        for i in range(n):
            rows[i]["consistent"] = len({tuple(run[i]["action_sequence"]) for run in runs}) == 1
            rows[i]["sequences_seen"] = sorted({" > ".join(run[i]["action_sequence"]) for run in runs})
        consistency = sum(r["consistent"] for r in rows) / n

    m = {
        "model": model, "cases": n, "repeats": a.repeats,
        "system_first_action_accuracy": sum(r["pass_first"] for r in rows) / n,
        "raw_llm_first_action_accuracy": (sum(raw) / len(raw)) if raw else None,
        "tool_sequence_exact_match": sum(r["pass_tools"] for r in rows) / n,
        "clarification_precision": (len(clar_ok) / len(clar_made)) if clar_made else None,
        "clarification_recall": (len(clar_expected & clar_made) / len(clar_expected)) if clar_expected else None,
        "policy_override_rate_per_turn": sum(r["overrides"] for r in rows) / n,
        "avg_tool_calls_per_turn": sum(r["tool_calls"] for r in rows) / n,
        "median_latency_s": statistics.median(r["latency_s"] for r in rows),
        "consistency_across_repeats": consistency,
    }

    safe = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    out = Path(a.out or ROOT / "docs" / f"eval_results_{safe}.md")
    lines = [f"# Decision-quality evaluation — `{model}`\n", f"{n} labelled cases, {a.repeats} repeat(s). "
             "Generated by `scripts/eval_decisions.py`.\n", "## Summary\n", "| Metric | Value |", "|---|---|"]
    for k, v in m.items():
        if k in ("model", "cases", "repeats"):
            continue
        lines.append(f"| {k.replace('_', ' ')} | {v:.0%} |" if isinstance(v, float) and k != "median_latency_s" and "per_turn" not in k
                     else f"| {k.replace('_', ' ')} | {v if not isinstance(v, float) else round(v, 3)} |")
    lines += ["\n## By category\n", "| Category | Cases | First action | Tool sequence |", "|---|---|---|---|"]
    for cat, rs in by_cat.items():
        lines.append(f"| {cat} | {len(rs)} | {sum(r['pass_first'] for r in rs)}/{len(rs)} | {sum(r['pass_tools'] for r in rs)}/{len(rs)} |")
    lines += ["\n## Per case\n", "| ID | Message | Expected first | LLM proposed | Executed (source) | Tools | ✓ |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        ok = "✅" if r["pass_first"] and r["pass_tools"] else ("🟡" if r["pass_first"] else "❌")
        lines.append(f"| {r['id']} | {r['message'][:60]} | {'/'.join(r['expected_first'])} | {r['llm_proposed_first']} | "
                     f"{r['executed_first']} ({r['first_source']}) | {', '.join(r['tools']) or '—'} | {ok} |")
    flaky = [r for r in rows if r.get("consistent") is False]
    if flaky:
        lines += ["\n## Inconsistent across repeats\n", "| ID | Message | Sequences seen |", "|---|---|---|"]
        lines += [f"| {r['id']} | {r['message'][:60]} | {' / '.join(r['sequences_seen'])} |" for r in flaky]
    if model == "mock":
        lines.append("\n> The mock LLM is rule-based and was written alongside these cases, so its scores validate the "
                     "harness and the deterministic layers — they are **not** evidence of LLM decision quality. Run "
                     "with a real `LLM_MODEL` for meaningful numbers.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"metrics": m, "rows": rows}, indent=2))
    print(json.dumps(m, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
