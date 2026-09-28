# langGraph: Research Assistant

A web research assistant built step by step with [LangGraph](https://github.com/langchain-ai/langgraph). It searches the web, writes an answer with sources, checks every claim against the search results, and asks you to approve it.

```
START → router ──(small talk)──→ chat → END
           └──(question)──→ [ research ] → human approval ──(approve)──→ END
                                 ↑                  └──(feedback)──┘
[ research ] = planner → parallel searches → agent ⇄ web_search → grounded reviewer
```

- **Router**: small talk gets a quick reply; real questions go to research.
- **Planner + parallel search**: 2-3 queries run at the same time (Tavily, DuckDuckGo as fallback).
- **Agent**: writes the answer and can search more.
- **Grounded reviewer**: quotes a source sentence for every claim; code checks the quotes and scores the answer, and unsupported claims go back to the agent.
- **Human approval**: the graph pauses (`interrupt`) until you approve or send feedback.
- **Memory**: conversations are saved in `memory.db` (SQLite) and survive restarts.

| Step | Model (→ fallback) |
|---|---|
| router, planner, chat | Gemini 3.5 Flash-Lite → Groq gpt-oss-20b |
| agent | Groq gpt-oss-120b → Gemini 3.5 Flash-Lite |
| reviewer | Groq gpt-oss-120b → Gemini 3.8 Flash → Gemini 3.5 Flash-Lite |
| no API keys | local qwen2.5:3b via Ollama |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Free API keys (set as environment variables, e.g. `setx GOOGLE_API_KEY "..."` on Windows, then open a new terminal):

| Variable | Get it at | Used for |
|---|---|---|
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey | Gemini |
| `GROQ_API_KEY` | https://console.groq.com/keys | gpt-oss |
| `TAVILY_API_KEY` | https://app.tavily.com | web search |

All three are optional; without them it falls back to DuckDuckGo and a local Ollama model.

## Run

```bash
streamlit run app.py                  # web UI at http://localhost:8501
python research_assistant.py cricket  # terminal chat; "cricket" = conversation name
python tests.py                       # checks for the plain-code logic (no model calls)
```

## Learning log

[LEARNING_LOG.md](LEARNING_LOG.md) explains all 20 steps: what was built, why, what the tests showed, and the alternatives.
