import logging
import os
import re
import sqlite3
import sys
from datetime import date
from typing import Annotated, Literal

from ddgs import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, RetryPolicy, Send, interrupt
from pydantic import BaseModel, Field
from tavily import TavilyClient

# Step 1: a graph with a single node that asks the model a question.
# Step 2: give the model a web search tool and let it loop: agent -> tools -> agent.
# Step 3: a system prompt with rules, and stream each step so we see what it does.
# Step 4: a reviewer node checks the answer against the search results and can send it back.
# Step 5: memory. A checkpointer saves the state after every step, per thread_id, so follow-up questions work.
# Step 6: a judge node scores the final answer (1-5) with structured output. It grades; it doesn't send back.
# Step 7: a planner writes 2-3 search queries, and a search node runs once per query, all at the same time.
# Step 8: human approval. The graph pauses with interrupt(); you approve the answer or send it back with feedback.
# Step 9: a router sends small talk and simple questions to a quick chat node, and only real questions to research.
# Step 10: memory saved to a SQLite file (memory.db), so conversations survive restarts. Pick one by name.
# Step 11: the research part (planner -> search -> agent <-> tools -> reviewer) is its own graph, used as one node.
# Step 12: a stronger judge. With GOOGLE_API_KEY set, every node (judge included) uses Gemini online;
#          without it, everything runs on the local Ollama model.
# Step 13: web UI in app.py (Streamlit).
# Step 14: judge grounding. The judge quotes the evidence for each claim; code checks the quotes and computes the scores.
# Step 15: Tavily search (with TAVILY_API_KEY), DuckDuckGo as the fallback.
# Step 16: the grounded judge IS the reviewer: one model call fewer, and the agent is told exactly which claims lack support.
# Step 17: shorter history. Models get earlier turns only as question -> final answer (last 3), not every search result.
# Step 18: retry on errors. Nodes that call a model retry up to 3 times on hiccups (RetryPolicy).
# Step 19: Groq gpt-oss-120b (with GROQ_API_KEY) for the agent and reviewer; Gemini is the fallback.
# Step 20: router, planner and chat fall back to Groq gpt-oss-20b when Gemini Flash-Lite fails.

# The model's own knowledge stops around its training date, so we tell it today's date.
# Without this, 'latest' means 'latest the model remembers' (years ago).
TODAY = date.today().isoformat()
YEAR = date.today().year

ROUTER = (
    "Classify the user's latest message. Answer 'research' unless it is ONLY a greeting, thanks, "
    "small talk or simple arithmetic. When unsure, answer 'research'.\n"
    "Examples:\n"
    "'hi' -> chat\n'thanks!' -> chat\n'how are you?' -> chat\n'what is 2+2' -> chat\n"
    "'who won latest t20 worldcup' -> research\n'price of bitcoin' -> research\n"
    "'who is the PM of India' -> research\n'what is LangGraph' -> research"
)

CHAT = "You are a friendly research assistant. Reply briefly. If the user asks for facts, suggest they ask as a research question."

PLANNER = (
    f"Today is {TODAY}. You plan web research. Write 2 or 3 short, different web search queries that together "
    "answer the user's latest question. Use the chat history to make follow-up questions specific. "
    f"For 'latest', 'current' or 'recent' questions, put the year in the queries, e.g. 'T20 World Cup {YEAR} winner' "
    f"and 'T20 World Cup {YEAR - 1} winner'."
)

SYSTEM = (
    f"Today is {TODAY}. You are a research assistant. Your own knowledge is out of date, so trust the "
    "research results given below over what you remember. For 'latest' questions, compare the dates in the "
    "results and answer with the most recent event. "
    "If they are not enough, use the web_search tool to search again with a different query. "
    "Only state facts that appear in the search results, and include the year for each fact. "
    "End with a 'Sources:' list of the URLs you used."
)

JUDGE = (
    f"Today is {TODAY}. You are a strict judge grading a research answer. Use only the search results as the truth, "
    "not your own knowledge. List every factual claim in the answer (names, numbers, dates, results). For each, "
    "copy word for word the sentence from the search results that supports it, or leave the quote empty if none "
    "does. Never write a quote yourself: your quotes are checked against the search results by code. "
    "Then score relevance from 1 (bad) to 5 (excellent); for 'latest' questions, an older event scores low."
)

MAX_REVIEWS = 2  # stop after this many reviews so a picky reviewer can't loop forever


def should_retry(error: Exception) -> bool:
    """Retry hiccups: network errors, server errors, per-minute rate limits, and a model reply that didn't fit
    the schema (a pydantic ValueError, which LangGraph's default rule would not retry).
    Don't retry what waiting a few seconds can't fix."""
    if "PerDay" in str(error):  # Gemini's daily quota is used up
        return False
    return not isinstance(error, (KeyError, TypeError, AttributeError, NameError))  # bugs in our code: fail at once


# Used on every node that calls a model. A retry re-runs just that node, from the same saved state.
RETRY = RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=3.0, retry_on=should_retry)  # waits ~2s, then ~6s


def add_or_reset(old: list, new: list | None) -> list:
    """Reducer for `research`: parallel search nodes each add their results; None clears the list.
    Skips items already there, because the research subgraph hands its whole list back to the parent."""
    return [] if new is None else old + [x for x in new if x not in old]


# State = the messages plus the current question, its plan and research, and the reviewer/judge results.
class State(MessagesState):
    question: str
    route: str
    queries: list[str]
    research: Annotated[list[str], add_or_reset]  # the reducer decides how parallel writes combine
    reviews: int
    verdict: str
    scores: dict


class Route(BaseModel):
    route: Literal["chat", "research"]  # Literal = the model can only pick one of these two


class Plan(BaseModel):
    queries: list[str] = Field(description="2 or 3 short web search queries.")


# The judge must reply in exactly this shape; with_structured_output turns it into a Python object.
class Claim(BaseModel):
    claim: str = Field(description="One factual claim made in the answer.")
    quote: str = Field(description="The exact sentence from the search results that supports it, copied word for word. Empty if none does.")


class Score(BaseModel):
    # claims come first so the model checks the evidence before it scores anything.
    claims: list[Claim] = Field(description="Every factual claim in the answer, each with its supporting quote.")
    relevance: int = Field(ge=1, le=5, description="Does it directly answer the question?")
    reason: str = Field(description="One sentence explaining the verdict.")


tavily = TavilyClient() if os.environ.get("TAVILY_API_KEY") else None  # the client reads the key itself
SEARCH_NAME = "tavily" if tavily else "duckduckgo (set TAVILY_API_KEY to use Tavily)"


def search_tavily(query: str) -> list[tuple]:
    # basic depth = 1 credit. `content` is a cleaned extract of the page, much longer than a DuckDuckGo snippet.
    results = tavily.search(query, max_results=5, search_depth="basic")["results"]
    return [(r["title"], r["url"], r.get("published_date"), r["content"]) for r in results]


def search_ddg(query: str) -> list[tuple]:
    return [(r["title"], r["href"], None, r["body"]) for r in DDGS().text(query, max_results=5)]


@tool
def web_search(query: str) -> str:
    """Search the web and return the top results with title, URL and page text."""
    error = None
    for search in ([search_tavily] if tavily else []) + [search_ddg]:  # DuckDuckGo is the fallback
        try:
            rows = search(query)
        except Exception as e:  # out of credits, rate limits or network errors shouldn't crash the whole graph
            error = e
            continue
        return "\n\n".join(
            f"{title}\n{url}\n" + (f"Published: {published}\n" if published else "") + text[:MAX_RESULT_CHARS]
            for title, url, published, text in rows
        ) or "No results."
    return f"Search failed: {error}"


# Each result's page text is cut to this length. Keeps the agent and reviewer inputs small enough for
# Groq's free 8,000 tokens/minute (about 3 queries x 5 results x 800 chars = ~3,000 tokens).
MAX_RESULT_CHARS = 800


def name(m) -> str:
    return getattr(m, "model_name", None) or getattr(m, "model", "?")  # Groq calls it model_name, the others model


def chain(models: list, wrap):
    """wrap(first model), falling back to wrap(next model) in order if it fails."""
    first, *rest = [wrap(m) for m in models]
    return first.with_fallbacks(rest) if rest else first


# When Gemini takes over from Groq it correctly drops Groq's hidden reasoning, but logs a warning each time.
logging.getLogger("langchain_google_genai.chat_models").setLevel(logging.ERROR)

if os.environ.get("GOOGLE_API_KEY"):
    # Online, on Google's servers. Free-tier quotas are per model and small (gemini-3.8-flash: 20 requests/day),
    # so the many cheap calls use Flash-Lite. Change them with the GEMINI_FAST_MODEL and GEMINI_MODEL env vars.
    # No temperature: these Gemini models use fixed sampling and would only print a warning.
    # max_retries=1 (default 6): on a quota error, move on to the fallback within seconds, not a minute.
    model = ChatGoogleGenerativeAI(model=os.environ.get("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite"), max_retries=1)
    # max_retries=0: when its quota is used up, fail at once so the next fallback takes over (no long waits).
    gemini_strong = ChatGoogleGenerativeAI(model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"), max_retries=0)
else:
    model = gemini_strong = ChatOllama(model="qwen2.5:3b", temperature=0)

if os.environ.get("GROQ_API_KEY"):
    # Groq free tier: 1,000 requests/day per model (vs 20 for gemini-3.8-flash), very fast, but only 8,000
    # tokens/minute. So it gets the two steps where quality matters most: the agent and the reviewer.
    # reasoning_effort="low": gpt-oss thinks before it answers; less thinking = fewer tokens and faster.
    # max_retries=1: on a 429 (token limit), fall back to Gemini quickly instead of waiting out the minute.
    groq = ChatGroq(model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"), reasoning_effort="low", max_retries=1)
    # The small gpt-oss has its own separate 1,000/day quota: a backup for the cheap steps (router, planner, chat).
    groq_small = ChatGroq(model=os.environ.get("GROQ_SMALL_MODEL", "openai/gpt-oss-20b"), reasoning_effort="low", max_retries=1)
    agent_models, reviewer_models, fast_models = [groq, model], [groq, gemini_strong, model], [model, groq_small]
else:
    agent_models, reviewer_models, fast_models = [model], [gemini_strong, model], [model]

MODEL_NAME = " | ".join(f"{step}: {' → '.join(map(name, models))}" for step, models in
                        [("agent", agent_models), ("reviewer", reviewer_models), ("router, planner, chat", fast_models)])
# with_fallbacks: if a model fails (quota used up, token limit, no internet), the next one takes over.
llm = chain(agent_models, lambda m: m.bind_tools([web_search]))  # the agent can search
judge_llm = chain(reviewer_models, lambda m: m.with_structured_output(Score))
router_llm = chain(fast_models, lambda m: m.with_structured_output(Route))
planner_llm = chain(fast_models, lambda m: m.with_structured_output(Plan))
chat_llm = chain(fast_models, lambda m: m)


MAX_TURNS = 3  # earlier question/answer pairs sent to the model; older ones stay saved but aren't sent


def split_turns(messages: list) -> tuple[list, list]:
    """(earlier turns, current turn). The current turn starts at the user's latest real question."""
    starts = [i for i, m in enumerate(messages) if m.type == "human" and m.name is None]
    cut = starts[-1] if starts else 0
    return messages[:cut], messages[cut:]


def compact(messages: list) -> list:
    """Earlier turns as plain question -> final answer pairs: no search results, tool calls,
    rejected drafts or reviewer notes, and only the last MAX_TURNS pairs."""
    pairs = []
    for m in messages:
        if m.type == "human" and m.name is None:
            pairs.append(m)
        elif m.type == "ai" and not m.tool_calls and m.text:
            if pairs and pairs[-1].type == "ai":
                pairs[-1] = m  # a later draft replaces the rejected one
            else:
                pairs.append(m)
    pairs = pairs[-2 * MAX_TURNS:]
    while pairs and pairs[0].type == "ai":  # chat models expect the conversation to start with the user
        pairs.pop(0)
    return pairs


def history(state: State) -> list:
    """For the agent: short earlier turns + everything from this turn (its searches and feedback)."""
    earlier, current = split_turns(state["messages"])
    return compact(earlier) + current


def brief(state: State) -> list:
    """For router, planner and chat: short earlier turns + this turn's question and user feedback only."""
    earlier, current = split_turns(state["messages"])
    return compact(earlier) + [m for m in current if m.type == "human" and m.name != "reviewer"]


def router(state: State) -> dict:
    return {"route": router_llm.invoke([SystemMessage(ROUTER)] + brief(state)).route}


def chat(state: State) -> dict:
    return {"messages": [chat_llm.invoke([SystemMessage(CHAT)] + brief(state))]}


def planner(state: State) -> dict:
    plan = planner_llm.invoke([SystemMessage(PLANNER)] + brief(state))
    return {"queries": plan.queries[:3] or [state["question"]]}  # never empty, or no search would run


def fan_out(state: State) -> list[Send]:
    # One Send per query = one "search" node run per query, all in parallel.
    return [Send("search", {"query": q}) for q in state["queries"]]


def search(item: dict) -> dict:
    # Gets only {"query": ...} from Send, not the whole state.
    return {"research": [f"Query: {item['query']}\n{web_search.invoke(item['query'])}"]}


def agent(state: State) -> dict:
    # The system prompt and research are added on every call but not saved in messages.
    context = SYSTEM + "\n\nResearch results:\n\n" + "\n\n---\n\n".join(state["research"])
    return {"messages": [llm.invoke([SystemMessage(context)] + history(state))]}


def search_text(state: State) -> str:
    """All search results for this question: the parallel research plus this turn's extra tool searches."""
    _, current = split_turns(state["messages"])
    return "\n\n".join(state["research"] + [m.text for m in current if m.type == "tool"])


def evidence(state: State) -> str:
    """Question + search results + the latest answer, for the reviewer and judge."""
    return f"Question: {state['question']}\n\nSearch results:\n{search_text(state)}\n\nAnswer:\n{state['messages'][-1].text}"


# Look-alike characters models and web pages mix up: curly quotes, and the many kinds of dash/hyphen
# (gpt-oss writes "back‑to‑back" with non-breaking hyphens U+2011).
LOOKALIKES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "‐": "-", "‑": "-", "–": "-", "—": "-", " ": " "})


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(LOOKALIKES)).strip(" \"'.").lower()


def is_grounded(quote: str, results: str) -> bool:
    """True if the quote really is in the search results. A '...' in the quote may skip text between parts."""
    # ponytail: exact match after normalizing spaces/case, so a lightly reworded quote fails.
    # Switch to difflib fuzzy matching if too many honest quotes get rejected.
    parts = [normalize(p) for p in re.split(r"\.\.\.|…", quote) if p.strip()]
    return sum(map(len, parts)) >= 20 and all(p in results for p in parts)  # 20+ chars: "Australia" alone proves nothing


def cited_urls(answer: str) -> list[str]:
    """URLs in the answer, without surrounding markdown or brackets like (), [], <> or gpt-oss's 【】."""
    return [u.rstrip(".,/") for u in re.findall(r"https?://[^\s()\[\]<>\"'【】]+", answer)]


def score_1_to_5(good: int, total: int) -> int:
    return 1 + round(4 * good / total) if total else 1


def grade(state: State) -> dict:
    # Grounding: the model must back each claim with a quote, and code, not the model, checks the quotes
    # and computes accuracy and sources. The model only scores relevance, which has no hard evidence.
    results = search_text(state)
    out = judge_llm.invoke([SystemMessage(JUDGE), HumanMessage(evidence(state))])
    claims = [{"claim": c.claim, "quote": c.quote, "grounded": is_grounded(c.quote, normalize(results))} for c in out.claims]
    cited = set(cited_urls(state["messages"][-1].text))
    found = [u for u in cited if u in results]
    return {
        "accuracy": score_1_to_5(sum(c["grounded"] for c in claims), len(claims)),
        "relevance": out.relevance,
        "sources": score_1_to_5(len(found), len(cited)),
        "reason": out.reason,
        "claims": claims,
    }


def reviewer(state: State) -> dict:
    # The grounded judge is the reviewer: one model call both checks the answer and scores it.
    reviews = state.get("reviews", 0) + 1
    # Plain code check, no model call. It looks for URLs, not the word "Sources:", because models format
    # that heading differently (gpt-oss writes "**Sources**").
    if not cited_urls(state["messages"][-1].text):
        problems, scores = ["The answer cites no source URLs. End it with a 'Sources:' list of the URLs you used."], {}
    else:
        scores = grade(state)
        problems = [f"Not supported by the search results: {c['claim']}" for c in scores["claims"] if not c["grounded"]]
        if scores["relevance"] < 3:
            problems.append(f"Doesn't answer the question well: {scores['reason']}")
    update = {"reviews": reviews, "scores": scores, "verdict": "REVISE: " + "; ".join(problems) if problems else "APPROVED"}
    if not problems or reviews >= MAX_REVIEWS:
        return update
    # Feedback goes into the chat as a new user message, so the agent sees exactly what to fix and tries again.
    feedback = ("Fix these problems and write the full answer again. For each unsupported claim, either remove it "
                "or use web_search to find a source that states it:\n- " + "\n- ".join(problems))
    return update | {"messages": [HumanMessage(feedback, name="reviewer")]}


def human(state: State) -> dict:
    # interrupt() saves the state and stops the graph here. It continues when we call
    # graph.stream(Command(resume=reply)), and `reply` becomes the return value of interrupt().
    reply = interrupt("Approve the answer? Enter = yes, or type what to change")
    if not reply:
        return {}
    # reviews and research reset so the redo (planner onwards) starts clean with your feedback.
    return {"reviews": 0, "research": None, "messages": [HumanMessage(f"The user wants changes. Research this and write the full answer again, doing this:\n{reply}", name="feedback")]}


def after_review(state: State) -> str:
    return "agent" if state["messages"][-1].type == "human" else END  # END of the research subgraph


def after_human(state: State) -> str:
    return "research" if state["messages"][-1].type == "human" else END


# Research subgraph: everything needed to produce a reviewed answer. It shares the parent's State,
# so it reads and writes the same keys (messages, question, research, ...) with no conversion.
research_builder = StateGraph(State)
research_builder.add_node("planner", planner, retry_policy=RETRY)
research_builder.add_node("search", search)  # web_search already catches its own errors
research_builder.add_node("agent", agent, retry_policy=RETRY)
research_builder.add_node("tools", ToolNode([web_search]))
research_builder.add_node("reviewer", reviewer, retry_policy=RETRY)
research_builder.add_edge(START, "planner")
research_builder.add_conditional_edges("planner", fan_out, ["search"])
research_builder.add_edge("search", "agent")  # agent waits until every parallel search has finished
# tools_condition returns "tools" or END; the map sends END to the reviewer instead.
research_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "reviewer"})
research_builder.add_edge("tools", "agent")
research_builder.add_conditional_edges("reviewer", after_review, ["agent", END])
research_graph = research_builder.compile()  # no checkpointer: it uses the parent's automatically

# Main graph: routing, grading and approval. The whole research subgraph is the single node "research".
builder = StateGraph(State)
builder.add_node("router", router, retry_policy=RETRY)
builder.add_node("chat", chat, retry_policy=RETRY)
builder.add_node("research", research_graph)  # a compiled graph can be added like any node function
builder.add_node("human", human)
builder.add_edge(START, "router")
builder.add_conditional_edges("router", lambda s: s["route"], {"chat": "chat", "research": "research"})
builder.add_edge("chat", END)  # small talk skips search, review and approval
builder.add_edge("research", "human")  # the reviewer inside research already graded the answer
builder.add_conditional_edges("human", after_human, ["research", END])
# check_same_thread=False: parallel search nodes run in other threads and share this connection.
memory = SqliteSaver(sqlite3.connect("memory.db", check_same_thread=False))
memory.setup()  # create the tables now; otherwise a brand-new memory.db has none until the first save
graph = builder.compile(checkpointer=memory)


def describe(node: str, update: dict) -> str:
    """One node's update as readable text. Shared by the terminal (print) and the web UI (app.py)."""
    if node == "router":
        return f"[router] {update['route']}"
    if node == "chat":
        msg = update["messages"][-1]
        return f"[chat: {msg.response_metadata.get('model_name', '?')}] {msg.text}"
    if node == "planner":
        return f"[planner] queries: {update['queries']}"
    if node == "search":
        text = update["research"][0]
        return f"[search] {text.splitlines()[0]} -> {text.count('http')} results"
    if node == "reviewer":
        lines = [f"[reviewer] {update['verdict']} (review {update['reviews']}/{MAX_REVIEWS})"]
        if s := update["scores"]:
            lines.append(f"    accuracy {s['accuracy']}/5, relevance {s['relevance']}/5, sources {s['sources']}/5: {s['reason']}")
            lines += [f"    {'✅' if c['grounded'] else '❌'} {c['claim']}" for c in s["claims"]]
        return "\n".join(lines)
    if node == "human":
        return "[human] approved" if not update else "[human] sent back with your feedback"
    msg = update["messages"][-1]
    if msg.type == "tool":
        return f"[tools] got {msg.text.count('http')} results"
    by = msg.response_metadata.get("model_name", "?")  # which model really answered (shows fallbacks)
    if msg.tool_calls:
        return f"[agent: {by}] searching: {msg.tool_calls[0]['args'].get('query')}"
    return f"[agent: {by}] answer:\n{msg.text}"


def steps(graph_input, config: dict):
    """Run the graph and yield (inside_subgraph, text) for each step. Check pending_question() afterwards."""
    # stream(..., "updates") yields {node_name: what_it_returned} after each node finishes.
    # subgraphs=True also streams the nodes inside the research subgraph, as (namespace, step) pairs.
    for namespace, step in graph.stream(graph_input, config, stream_mode="updates", subgraphs=True):
        for node, update in step.items():
            # "research" is the subgraph's final output (its inner steps were already shown);
            # "__interrupt__" means the graph paused at interrupt().
            if node not in ("research", "__interrupt__"):
                yield bool(namespace), describe(node, update or {})


def pending_question(config: dict) -> str | None:
    """The interrupt() question if this conversation is paused waiting for approval, else None."""
    paused = graph.get_state(config).interrupts
    return paused[0].value if paused else None


def new_question(question: str) -> dict:
    # Only the new message is sent; the checkpointer adds it to the saved history.
    # reviews = 0 and research = None reset them, so each question starts clean.
    return {"messages": [HumanMessage(question)], "question": question, "reviews": 0, "research": None}


def run(graph_input, config: dict) -> str | None:
    """Terminal version: print each step. Returns the interrupt question if it paused, else None."""
    for inside, text in steps(graph_input, config):
        print(("  " if inside else "") + text)  # indent subgraph steps
    return pending_question(config)


if __name__ == "__main__":
    # Conversation name from the command line, e.g. `python research_assistant.py cricket`.
    # Same name = same conversation, even after closing the program.
    thread = sys.argv[1] if len(sys.argv) > 1 else "chat-1"
    config = {"configurable": {"thread_id": thread}}
    saved = graph.get_state(config).values.get("messages", [])
    print(f"Model: {MODEL_NAME}\nSearch: {SEARCH_NAME}")
    print(f"Conversation '{thread}': {len(saved)} saved messages" + (f", last question: {graph.get_state(config).values['question']}" if saved else ""))
    while question := input("\nYou (empty to quit): ").strip():
        paused = run(new_question(question), config)
        while paused:
            reply = input(f"\n[human] {paused}: ").strip()
            paused = run(Command(resume=reply), config)  # continue from where it paused