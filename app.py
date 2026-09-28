"""Web UI for the research assistant. Run with:  streamlit run app.py

Streamlit re-runs this whole file on every click or message, top to bottom.
All real state lives in the LangGraph checkpointer (memory.db), so each run just reads it and draws it.
"""
import streamlit as st
from langgraph.types import Command

import research_assistant as ra

st.set_page_config(page_title="Research Assistant", page_icon="🔎", layout="centered")


def conversations() -> list[str]:
    rows = ra.memory.conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
    return sorted(r[0] for r in rows)


def start_new() -> None:
    if name := st.session_state.new_name.strip():
        st.session_state.thread = name
    st.session_state.new_name = ""


def delete_current() -> None:
    ra.memory.delete_thread(st.session_state.thread)
    st.session_state.pop("thread")
    st.session_state.pop("last_steps", None)


# ---------- Sidebar: model and conversations ----------
threads = conversations()
st.session_state.setdefault("thread", threads[0] if threads else "chat-1")

with st.sidebar:
    st.title("🔎 Research Assistant")
    st.caption(f"Model: **{ra.MODEL_NAME}**  \nSearch: **{ra.SEARCH_NAME}**")
    st.text_input("New conversation", key="new_name", placeholder="name, e.g. cricket")
    # Not disabled while empty: the name only "commits" on blur, so a disabled button would swallow the first click.
    st.button("➕ Start", on_click=start_new, use_container_width=True)
    options = sorted(set(threads) | {st.session_state.thread})
    st.radio("Conversations", options, key="thread")
    st.button("🗑️ Delete this conversation", on_click=delete_current, use_container_width=True)
    with st.expander("How it works"):
        st.markdown(
            "1. **Router**: small talk → quick reply, questions → research\n"
            "2. **Planner**: writes 2-3 search queries\n"
            "3. **Search**: runs them in parallel (Tavily, or DuckDuckGo)\n"
            "4. **Agent**: writes the answer, can search more\n"
            "5. **Reviewer**: quotes a source for every claim; code checks the quotes, "
            "scores the answer, and sends unsupported claims back to the agent\n"
            "6. **You**: approve, or send it back with feedback"
        )

config = {"configurable": {"thread_id": st.session_state.thread}}


def run_graph(graph_input) -> None:
    """Run the graph with a live step list, keep the steps for later, then redraw the page."""
    lines, error = [], None
    with st.status("Working…", expanded=True) as status:
        try:
            for inside, text in ra.steps(graph_input, config):
                line = ("↳ " if inside else "") + text
                lines.append(line)
                status.markdown(line)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:  # e.g. Gemini quota used up or no internet; progress so far is saved
            status.update(label="Failed", state="error")
            error = e
    if error:  # shown outside the status box, which may be collapsed
        if "RESOURCE_EXHAUSTED" in str(error):
            st.error("Gemini's free quota is used up for now. Wait a minute (or until tomorrow for the daily limit) and ask again.")
        else:
            st.error(f"Something went wrong: {error}")
        return
    st.session_state.last_steps = lines
    st.rerun()


# ---------- Chat history ----------
# Show your questions and feedback, and only the final answer per question (drafts the reviewer
# rejected, tool calls and reviewer messages stay hidden).
turns = []
for m in ra.graph.get_state(config).values.get("messages", []):
    if m.type == "human" and m.name != "reviewer":
        text = "✏️ " + m.text.split("doing this:\n", 1)[-1] if m.name == "feedback" else m.text
        turns.append(["user", text])
    elif m.type == "ai" and not m.tool_calls and m.text:
        if turns and turns[-1][0] == "assistant":
            turns[-1][1] = m.text  # a later draft replaces the rejected one
        else:
            turns.append(["assistant", m.text])

if not turns:
    st.markdown("### Ask me anything 👋")
    st.caption("I search the web, check my answer, score it, and let you approve it before it's final.")

for role, text in turns:
    with st.chat_message(role, avatar="🧑" if role == "user" else "🔎"):
        st.markdown(text)

if st.session_state.get("last_steps"):
    with st.expander("🧭 Steps behind the last answer"):
        st.markdown("\n\n".join(st.session_state.last_steps))

# ---------- Approval panel (the graph is paused at interrupt()) ----------
waiting = ra.pending_question(config)
if waiting:
    scores = ra.graph.get_state(config).values.get("scores") or {}
    with st.container(border=True):
        st.markdown("#### ✅ Review this answer")
        if scores:
            a, r, s = st.columns(3)
            a.metric("Accuracy", f"{scores['accuracy']}/5")
            r.metric("Relevance", f"{scores['relevance']}/5")
            s.metric("Sources", f"{scores['sources']}/5")
            st.caption(f"Reviewer: {scores['reason']}")
            claims = scores.get("claims", [])
            with st.expander(f"🔍 Grounding: {sum(c['grounded'] for c in claims)}/{len(claims)} claims found in the sources"):
                for c in claims:
                    st.markdown(f"{'✅' if c['grounded'] else '❌'} **{c['claim']}**")
                    st.caption(f"“{c['quote']}”" if c["quote"] else "No supporting quote given")
        if st.button("👍 Approve", type="primary", use_container_width=True):
            run_graph(Command(resume=""))
        feedback = st.text_input("…or tell it what to change", placeholder="e.g. also add who was player of the match")
        if st.button("↩️ Send back", use_container_width=True) and feedback.strip():
            with st.chat_message("user", avatar="🧑"):
                st.markdown("✏️ " + feedback)
            run_graph(Command(resume=feedback.strip()))

# ---------- New question ----------
if question := st.chat_input("Approve or send back the answer first" if waiting else "Ask a question…", disabled=bool(waiting)):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    run_graph(ra.new_question(question))
