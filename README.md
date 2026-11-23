📚 AI Study Assistant — an agentic workflow with LangGraph
Project overview
An AI study assistant that understands a student's goals and decides for itself what to do next: build a personalised study plan, explain a topic at the right level, generate a quiz, grade the answers, or ask a clarifying question when a request is ambiguous.
It keeps context across turns, adapts its workflow to intermediate results (for example, a low quiz score triggers a focused re-explanation), and recovers from failed tool calls and invalid LLM output.
How it works: an LLM router proposes the next action as a structured decision. A deterministic policy layer then checks that proposal (execution limits, loop prevention, preconditions) before anything runs. Each step is recorded in a decision trace, so every action can be explained. The workflow is built with LangGraph and served through a FastAPI backend with a small web UI.
---
Dependencies
Python 3.10 or newer (3.11 or 3.12 recommended)
An LLM API key for real use, e.g. Anthropic (Claude) or OpenAI. Optional: an offline mock LLM runs the whole system without a key.
Python packages (installed from `requirements.txt`):
Package	Purpose
`langgraph`, `langgraph-checkpoint-sqlite`	Agent workflow (state graph, conditional routing, session memory)
`langchain`, `langchain-core`	Provider-independent LLM interface and structured output
`langchain-anthropic`, `langchain-openai`	Connectors for Claude and OpenAI models
`pydantic`	Validated, structured inputs and outputs for every tool
`fastapi`, `uvicorn`	REST API and web server
`python-dotenv`	Loads settings from `.env`
`pytest`, `httpx`	Automated tests
---
Setup instructions
1. Get the code
```bash
git clone https://github.com/Diphiya/Autonomous_AI_Agent.git
cd Autonomous_AI_Agent
```
2. Create and activate a virtual environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```
On Windows, if activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.
3. Install the dependencies
```bash
pip install -r requirements.txt
```
4. Configure the LLM
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```
Open `.env` and choose a model:
```
# Claude (Anthropic)
LLM_MODEL=anthropic:claude-haiku-5-5
LLM_TEMPERATURE=
ANTHROPIC_API_KEY=sk-ant-...

# or OpenAI
# LLM_MODEL=openai:gpt-4o-mini
# OPENAI_API_KEY=sk-...

# or no key at all (offline, deterministic mock LLM)
# LLM_MODEL=mock
```
Leave `LLM_TEMPERATURE` empty for newer Claude models, which reject the parameter. Never commit `.env`; it is already excluded by `.gitignore`.
---
Execution steps
Start the server
```bash
uvicorn study_agent.api.app:app --reload
```
Web UI: http://localhost:8000. The chat is on the left; the decision trace for each turn is on the right.
API documentation (Swagger): http://localhost:8000/docs
Stop the server with `Ctrl + C`.
Try these messages in the UI
`Make me a study plan for machine learning: 4 weeks, 6 hours a week, I'm a beginner`
`Quiz me on week 1`, then answer, e.g. `1. A 2. B 3. C 4. ...`
`Help me study` (ambiguous, so the assistant asks a clarifying question)
`Explain gradient descent and then give me 3 practice questions` (two tools in one turn)
The Failure injection buttons in the UI make the next LLM call fail on purpose, to demonstrate retries and recovery.
Terminal chat (alternative to the UI)
```bash
python cli.py --trace
```
Run the tests
```bash
pytest -q
```
Expected result: `61 passed`.
Run the demo scenarios and the decision-quality evaluation (optional)
```bash
python scripts/run_scenarios.py                 # 5 multi-step scenarios with decision traces
python scripts/eval_decisions.py --repeats 3    # 23 labelled cases, 3 runs each
```
Both scripts use the model set in `.env` and write their reports as local files.
