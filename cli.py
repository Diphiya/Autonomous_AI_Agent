"""Interactive terminal client.

    python cli.py                 # uses LLM_MODEL from env/.env (default: mock)
    python cli.py --trace         # also print the decision trace after each turn
Commands: /state  /fault <task> <kind,...>  /new  /quit
"""
from __future__ import annotations

import argparse
import json
import uuid

from study_agent.agent import StudyAssistant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="store_true", help="print decision trace after each reply")
    ap.add_argument("--session", default=None)
    args = ap.parse_args()

    bot = StudyAssistant()
    sid = args.session or uuid.uuid4().hex[:8]
    print(f"AI Study Assistant (LLM: {bot.llm.name}, session: {sid}). Type /quit to exit.\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/new":
            sid = uuid.uuid4().hex[:8]
            print(f"(new session {sid})")
            continue
        if msg == "/state":
            print(json.dumps(bot.public_state(sid), indent=2))
            continue
        if msg.startswith("/fault"):
            _, task, kinds = msg.split(maxsplit=2)
            print("queued:", bot.inject_faults(sid, task, kinds.split(",")))
            continue
        r = bot.chat(msg, sid)
        print(f"\nassistant> {r.reply}\n")
        if args.trace:
            for t in r.trace:
                print(f"  [{t['step']}] {t['node']} -> {t['decision']} ({t['source']}): {t['reasoning']}")
                if t.get("detail"):
                    print(f"        {t['detail']}")
            print(f"  ({r.latency_ms} ms)\n")


if __name__ == "__main__":
    main()
