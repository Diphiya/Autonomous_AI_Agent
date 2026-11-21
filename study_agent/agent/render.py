"""Deterministic Markdown rendering of tool outputs.

Rendering is plain code, not an LLM call: it is instant, free, and guarantees the quiz
answer key is never leaked to the student.
"""
from __future__ import annotations

from typing import Any

LETTERS = "ABCD"


def render_plan(p: dict[str, Any]) -> str:
    lines = [f"### 📅 Study plan — {p['goal']}", f"*{p['duration_weeks']} weeks · ~{p['hours_per_week']:g} h/week*", ""]
    for w in p["weeks"]:
        lines.append(f"**Week {w['week']} — {w['focus']}** (~{w['estimated_hours']:g} h)")
        lines.append(f"- Topics: {', '.join(w['topics'])}")
        for a in w["activities"]:
            lines.append(f"- {a}")
        lines.append(f"- ✅ Checkpoint: {w['checkpoint']}")
        lines.append("")
    if p.get("tips"):
        lines.append("**Tips:** " + " ".join(p["tips"]))
    return "\n".join(lines).strip()


def render_explanation(e: dict[str, Any]) -> str:
    lines = [f"### 💡 {e['topic'].title()}", f"**In one line:** {e['summary']}", "", e["explanation"]]
    if e.get("analogy"):
        lines += ["", f"**Analogy:** {e['analogy']}"]
    lines += ["", f"**Example:** {e['example']}", "", "**Key points:**"] + [f"- {k}" for k in e["key_points"]]
    if e.get("common_mistakes"):
        lines += ["", "**Common mistakes:**"] + [f"- {m}" for m in e["common_mistakes"]]
    lines += ["", f"**Check yourself:** {e['check_question']}"]
    return "\n".join(lines)


def render_quiz(q: dict[str, Any]) -> str:
    lines = [f"### 📝 Quiz — {q['topic']} ({q['level']})", ""]
    for item in q["questions"]:
        lines.append(f"**{item['id']}. {item['question']}**" + ("  *(short answer)*" if item["type"] == "short" else ""))
        if item["type"] == "mcq":
            for i, opt in enumerate(item["options"]):
                lines.append(f"   {LETTERS[i]}) {opt}")
        lines.append("")
    return "\n".join(lines).strip()  # NOTE: correct_option / reference_answer intentionally not rendered


def render_evaluation(ev: dict[str, Any]) -> str:
    lines = [f"### 📊 Results — {ev['topic']}: **{ev['total_score']:.0%}**", ""]
    for r in ev["results"]:
        mark = "✅" if r["correct"] else ("🟡" if r["score"] > 0 else "❌")
        lines.append(f"{mark} **Q{r['id']}** ({r['concept']}): {r['feedback']}")
    lines += ["", ev["summary_feedback"]]
    if ev.get("weak_concepts"):
        lines.append(f"**To review:** {', '.join(ev['weak_concepts'])}")
    return "\n".join(lines)


RENDERERS = {
    "plan": render_plan,
    "explanation": render_explanation,
    "quiz": render_quiz,
    "evaluation": render_evaluation,
}


def render_block(block: dict[str, Any]) -> str:
    return RENDERERS[block["kind"]](block["data"])
