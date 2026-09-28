"""Checks for the plain-code logic (no model calls). Run: python tests.py"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from research_assistant import MAX_TURNS, brief, cited_urls, compact, history, is_grounded, normalize, score_1_to_5

# ---------- Grounding (step 14) ----------
results = normalize("2023 Cricket World Cup final\nhttps://x.org\nAustralia beat India by six wickets in the final on 19 November 2023.")

assert is_grounded("Australia beat India by six wickets in the final", results)            # exact
assert is_grounded("australia  beat India by SIX wickets\nin the final.", results)         # spaces/case/newline
assert is_grounded("Australia beat India ... on 19 November 2023", results)                # "..." skips text
assert not is_grounded("India beat Australia by six wickets in the final", results)        # made-up quote
assert not is_grounded("Australia", results)                                               # too short to prove anything
assert not is_grounded("", results)                                                        # no quote
page = normalize("Indian captain Suryakumar Yadav's dismissal for a duck ... a back-to-back win")
assert is_grounded("Indian captain Suryakumar Yadav’s dismissal for a duck", page)        # curly vs straight apostrophe
assert is_grounded("a back‑to‑back win", page) is False                                   # too short (under 20 chars)...
assert is_grounded("Yadav’s dismissal for a duck ... a back‑to‑back win", page)           # ...but hyphen U+2011 matches -

assert score_1_to_5(3, 3) == 5 and score_1_to_5(0, 3) == 1 and score_1_to_5(2, 4) == 3 and score_1_to_5(0, 0) == 1

# URLs in the formats different models write (step 19): plain, markdown link, gpt-oss 【】 brackets
answer = ("**Sources**\n- https://a.org/x.\n- [Wiki](https://en.wikipedia.org/wiki/2026_Men%27s_T20_World_Cup)\n"
          "- title (2026)【https://b.com/list.html】")
assert cited_urls(answer) == ["https://a.org/x", "https://en.wikipedia.org/wiki/2026_Men%27s_T20_World_Cup", "https://b.com/list.html"]
assert cited_urls("no links here") == []

# ---------- Shorter history (step 17) ----------
big = "search result text " * 500  # ~10k characters, like a real page of results


def research_turn(n: int) -> list:
    """One full research turn as it is saved: question, tool call, results, rejected draft, reviewer note, final answer."""
    return [
        HumanMessage(f"question {n}"),
        AIMessage("", tool_calls=[{"name": "web_search", "args": {"query": f"q{n}"}, "id": f"c{n}"}]),
        ToolMessage(big, tool_call_id=f"c{n}"),
        AIMessage(f"draft {n}"),
        HumanMessage("Fix these problems...", name="reviewer"),
        AIMessage(f"final answer {n}"),
    ]


saved = sum((research_turn(n) for n in range(1, 6)), []) + [HumanMessage("question 6"), AIMessage("", tool_calls=[{"name": "web_search", "args": {"query": "q6"}, "id": "c6"}]), ToolMessage(big, tool_call_id="c6")]
state = {"messages": saved}

old = compact(saved[:-3])
assert [m.text for m in old] == ["question 3", "final answer 3", "question 4", "final answer 4", "question 5", "final answer 5"]
assert len(old) == 2 * MAX_TURNS

sent = history(state)  # agent: short earlier turns + this turn in full (it needs its own search results)
assert sent[:6] == old and [m.text for m in sent[6:]] == ["question 6", "", big]

assert [m.text for m in brief(state)] == [m.text for m in old] + ["question 6"]  # router/planner: no search results at all

before, after = sum(len(m.text) for m in saved), sum(len(m.text) for m in sent)
print(f"agent input: {before:,} -> {after:,} characters ({100 - 100 * after // before}% smaller)")

# ---------- Retry on errors (step 18) ----------
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from research_assistant import RETRY, should_retry

try:
    class Needs(BaseModel):
        n: int
    Needs(n="not a number")
except ValidationError as e:
    bad_reply = e  # what a model reply that doesn't fit the schema raises

assert should_retry(ConnectionError("network hiccup")) and should_retry(bad_reply)
assert should_retry(Exception("429 RESOURCE_EXHAUSTED ... GenerateRequestsPerMinute"))    # per-minute limit: wait helps
assert not should_retry(Exception("429 RESOURCE_EXHAUSTED ... GenerateRequestsPerDay"))   # daily quota: it doesn't
assert not should_retry(KeyError("scores"))                                               # our bug: fail at once

attempts = []


class S(TypedDict):
    attempts: int


def flaky(state: S) -> dict:
    attempts.append(1)
    if len(attempts) == 1:
        raise ConnectionError("network hiccup")  # fails the first time only
    return {"attempts": len(attempts)}


g = StateGraph(S)
g.add_node("flaky", flaky, retry_policy=RETRY)
g.add_edge(START, "flaky")
g.add_edge("flaky", END)
assert g.compile().invoke({"attempts": 0}) == {"attempts": 2}  # failed once, retried, succeeded
print("retry: node failed once, retried, succeeded")
print("all checks passed")
