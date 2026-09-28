 # LangGraph Research Assistant: Learning Log

## Step 1: One model node

```
START → agent → END
```

**What:** A graph with one node (`agent`) that sends the chat messages to the model and adds its reply.

**Why:**
- `StateGraph` is the base of every LangGraph app: nodes (functions) + edges (order).
- `MessagesState` keeps the conversation as a message list; each node's new messages are appended, not overwritten.
- `ChatOllama` runs a free local model (qwen2.5:3b), so no API key is needed.

**Alternatives:**
- Call `llm.invoke(question)` directly with no graph. That's simpler for one call, but you can't add loops or steps later.
- Write your own state (`TypedDict`) instead of `MessagesState`, like `hello_graph.py` does.
- Use a cloud model (`ChatAnthropic`, `ChatOpenAI`): faster and smarter, but needs an API key.

## Step 2: Web search tool + loop

```
START → agent ──(tool call)──→ tools
          ↑                      │
          └──────────────────────┘
          └──(plain answer)──→ END
```

**What:**
- `@tool web_search`: a Python function the model is allowed to call. The docstring tells the model what it does.
- `llm.bind_tools([web_search])` shows the model the tool's name, arguments and description.
- `ToolNode` runs whatever tool the model asked for and adds the result as a message.
- `tools_condition` checks the model's last reply: if it asked for a tool, go to `tools`; if not, go to END.
- `tools → agent` sends the search results back so the model can search again or answer.

**Why:** A model only knows its training data. Tools let it fetch fresh facts, and the loop lets it decide by itself how many searches it needs.

**Alternatives:**
- `create_react_agent(llm, [web_search])` from `langgraph.prebuilt` builds this same graph in one line. It's quicker but hides how it works.
- Always search first, then answer (`START → search → agent → END`). That's simpler and works with weak models, but it always searches exactly once.
- Write your own router function instead of `tools_condition`, like `keep_going` in `hello_graph.py`.
- Tavily search instead of DuckDuckGo: cleaner results, but needs an API key.

## Step 3: System prompt + live progress

**What:**
- `SYSTEM` holds the rules: always search, retry on unclear results, only state facts from the results, list sources.
- `agent` puts a `SystemMessage(SYSTEM)` before the chat on every call. It isn't saved in state, so it never gets duplicated.
- `graph.stream(..., stream_mode="updates")` returns each node's output as soon as the node finishes, instead of waiting for the end like `invoke`.

**Why:** Without rules the model answers however it wants. Without streaming you can't see what it searched for or why it got something wrong.

**What we saw:** It searched once ("latest cricket worldcup winner") and answered "India, 2011". That's out of date, and it skipped the Sources list. A small model often ignores rules, which is why step 4 adds a reviewer.

**Alternatives:**
- Put the system prompt in the initial state instead. That's simpler, but it gets saved in memory and needs care once we add memory in step 5.
- Other stream modes: `"values"` (the full state after each step), `"messages"` (the model's reply token by token, like ChatGPT typing), `"debug"` (everything).
- LangSmith tracing: a web dashboard that shows every step. Needs an account and API key.

## Step 4: Reviewer node

```
START → agent ⇄ tools
          │
     (answer ready)
          ↓
       reviewer ──(REVISE, feedback added as a message)──→ agent
          └──(APPROVED or limit reached)──→ END
```

**What:**
- Custom `State(MessagesState)` adds `reviews` (a counter) and `verdict` (the reviewer's last reply) to the messages.
- `reviewer` first runs a plain-code check for a `Sources:` list. If that passes, a second model call checks the answer against the search results and replies `APPROVED` or `REVISE: ...`.
- On `REVISE`, the feedback is added to the chat as a user message, so the agent sees it and tries again.
- `tools_condition` with a path map `{"tools": "tools", END: "reviewer"}` reuses the prebuilt router but sends the "done" case to the reviewer instead of ending.
- `MAX_REVIEWS = 2` limits the loop so it can't run forever.

**Why:** A model checks better than it writes in one go (this pattern is called reflection or self-critique). Some rules are simple enough for plain code, which is free and never wrong.

**What we saw:** The code check caught the missing Sources list, the agent added one, and the model reviewer approved on review 2. But "India, 2011" is still wrong: Australia won the 2023 ODI World Cup. The 3B reviewer can't spot that the answer isn't the *latest* result. Model size is now the bottleneck, not the graph.

**Alternatives:**
- Structured output (`model.with_structured_output(Review)`, a Pydantic model with `approved: bool, feedback: str`) instead of parsing the text "APPROVED". More reliable on bigger models.
- A bigger or cloud model for the reviewer only: a cheap writer paired with a strong checker.
- A human reviewer: `interrupt()` pauses the graph so you approve the answer yourself (human-in-the-loop). Needs a checkpointer (step 5).
- `Command(goto="agent", update=...)` returned from the node: the node itself picks the next step, with no separate `after_review` function.

## Step 5: Memory (checkpointer + chat loop)

**What:**
- `builder.compile(checkpointer=InMemorySaver())`: after every node, LangGraph saves the whole state.
- `config = {"configurable": {"thread_id": "chat-1"}}`: the saved state is filed under this ID. The same ID continues the conversation; a new ID starts a fresh one.
- Each turn sends only the new question. The saved history is loaded and the new message is appended to it.
- The state gets a `question` field (the reviewer checks against the current question, not the first one ever asked), and `reviews` is reset to 0 each turn.
- A `while input()` loop makes it a chat. An empty line quits.
- Prompt tweak: "search even for follow-up questions". Without it, the agent reused old results and couldn't answer "who was the captain".

**Why:** Without memory every run starts blank, so "that team" means nothing. The checkpointer is also what makes human approval (`interrupt()`), undo, and "time travel" possible later.

**What we saw:**
- Memory works: "who was the captain of that team" became the search "2023 cricket world cup winner captain".
- The agent is still unreliable: one run said Australia won (correct), and another said India won and then "Rohit Sharma" as captain, consistent with its own wrong answer. The reviewer approved both. Same bottleneck as step 4: the 3B model.

**Alternatives:**
- `SqliteSaver` (`pip install langgraph-checkpoint-sqlite`): saves to a file, so memory survives restarts. `PostgresSaver` for real apps.
- A `Store` (`InMemoryStore`) for long-term memory *across* threads, such as user preferences. A checkpointer only remembers within one thread.
- Trim or summarize old messages (`trim_messages`, or a summary node) once the chat gets long. Small models have short context windows.
- No checkpointer: pass the whole message list back in yourself each turn. That works, but you manage the history by hand.

## Step 6: Judge node (LLM-as-a-judge)

```
START → agent ⇄ tools
          ↓
       reviewer ──(REVISE)──→ agent
          ↓ (APPROVED / limit)
        judge → END
```

**What:**
- `Score` (a Pydantic model) defines the exact reply shape: `accuracy`, `relevance` and `sources` (each 1-5) plus a `reason`.
- `model.with_structured_output(Score)` makes the model reply in that shape and returns a Python object, so there's no text parsing like the reviewer's "APPROVED".
- `judge` runs once after the reviewer is done and saves the scores in `state["scores"]`.
- `evidence()` builds the question + search results + answer text, shared by the reviewer and the judge.
- `add_conditional_edges("reviewer", after_review, ["agent", "judge"])`: the list tells LangGraph the possible targets, so the graph drawing is correct.

**Reviewer vs judge:** The reviewer is a gatekeeper inside the loop (pass or fail, can send the answer back). The judge is a grader at the end (numbers and a reason, never sends it back). Scores let you compare runs, prompts or models.

**What we saw:** Answer "Australia, 19 Nov 2023" (correct), judged 5/5/5. But the Sources list had page titles and no URLs, so a strict judge would give less than 5 for sources. A 3B judge is lenient.

**Alternatives:**
- A stronger model as the judge (e.g. `ollama pull qwen2.5:7b`, or a cloud model). The judge matters most, so give it the best model.
- Make the judge a gate too: if a score is < 3, route back to `agent` with the reason.
- Run the judge offline on a fixed list of test questions (an eval set) instead of on every chat turn. That's how teams compare prompts or models.
- Plain-code checks inside the judge (e.g. count the `http` links) for anything that doesn't need a model.

## Step 7: Planner + parallel search (fan-out / fan-in)

```
            ┌→ search (query 1) ─┐
START → planner → search (query 2) ─┼→ agent ⇄ tools
            └→ search (query 3) ─┘      ↓
                                     reviewer ──(REVISE)──→ agent
                                        ↓
                                      judge → END
```

**What:**
- `planner` uses structured output (`Plan` with `queries: list[str]`) to write 2-3 different search queries.
- `fan_out` returns one `Send("search", {"query": q})` per query. **`Send`** starts a copy of the `search` node for each item, all running at the same time (fan-out).
- `search` receives only `{"query": ...}`, not the whole state. That's how each parallel copy gets its own input.
- `add_edge("search", "agent")`: the agent runs once, after **all** searches finish (fan-in).
- `research: Annotated[list[str], add_or_reset]`: a **reducer**. Three searches write to `research` at the same time; the reducer says to *append*, not overwrite. Passing `None` clears it for the next question.
- The agent gets the research in its system prompt and can still use `web_search` if that's not enough.
- `web_search` catches errors, so one failed search doesn't crash the graph.
- Reviewer feedback now says "write the full answer again". Before, the agent replied with only the Sources list.

**Why:** One vague search often misses things. Several focused queries give better coverage, and running them in parallel means the wait is for the slowest search, not all three added up.

**What we saw:**
- The searches finished in a different order than planned, which shows they ran in parallel.
- Second run: "Australia, Pat Cummins", with 5 URL sources, approved, and judged 5/5/5. That's correct.
- First run: the judge said "Steve Smith" was captain and still gave 5/5. The 3B judge makes things up, so don't trust its scores blindly.

**Alternatives:**
- Fixed fan-out without `Send`: add 3 nodes by hand and edges `planner → search1/2/3 → agent`. Simpler, but the number of searches is fixed in the code.
- No planner: let the agent call `web_search` several times in one reply (parallel tool calls). The `ToolNode` already runs them in parallel, but small models rarely do this.
- Plain Python parallelism (`concurrent.futures.ThreadPoolExecutor`) inside one node. It works, but the searches become invisible to the graph (no per-search streaming or retries).
- A built-in reducer like `operator.add` if you never need to reset the list.

## Step 8: Human approval (`interrupt`)

```
... → reviewer → judge → human ──(Enter)──→ END
                           └──(your feedback)──→ agent → ... → human (asks again)
```

**What:**
- `human` node calls `interrupt("Approve the answer? ...")`. The graph **saves its state and stops** at that line.
- In `run()`, the stream yields `"__interrupt__"`. We print the question and read your reply with `input()`.
- `graph.stream(Command(resume=reply), config)` continues from the pause, and `reply` becomes the return value of `interrupt()`.
- An empty reply (Enter) approves → END. Any text is added to the chat as feedback → `agent` rewrites the answer (with a fresh review count) → reviewer → judge → `human` asks again.
- The printing code moved into `show()` and `run()` so it can be reused for the first run and every resume.

**Why:** Some decisions need a person, like publishing, sending or paying. `interrupt` only works because of the checkpointer (step 5): the paused state is saved under the `thread_id`, so it could even resume hours later or from a different process (with `SqliteSaver`).

**Important detail:** On resume, the `human` node runs **from the start again**, and `interrupt()` returns the reply instead of pausing. So never put side effects (like sending an email) *before* `interrupt()` in a node.

**What we saw:** The feedback "also say who was player of the match" sent the answer back correctly. But the agent didn't search; it just said "not in the results". Fix: the feedback message now tells it to use `web_search` if the research doesn't cover it.

**Alternatives:**
- `compile(interrupt_before=["judge"])`: pause before a node without changing its code. Simpler, but it can't ask a question or get a reply value.
- Let the human edit the state directly: `graph.update_state(config, {...})`, then resume.
- Put the approval before the answer is shown (e.g. approve the planner's queries before searching).

## Step 9: Router

```
START → router ──(chat)──→ chat → END
           └──(research)──→ planner → search ×3 → agent ... → human → END
```

**What:**
- `Route` uses `Literal["chat", "research"]`, so structured output can only return one of those two words.
- The `router` node saves the decision in `state["route"]`. The conditional edge `lambda s: s["route"]` reads it, and the map `{"chat": "chat", "research": "planner"}` picks the next node.
- `chat` node: a plain model reply with no search, review, judge or approval.

**Why:** "hi, how are you?" doesn't need 3 web searches, 4 model calls and your approval. Routing saves time and avoids nonsense searches. It's the most common pattern in real LangGraph apps (e.g. a support bot routing to billing / tech / general).

**What we saw:**
- "hi, how are you?" → `chat` → instant friendly reply.
- "who was player of the match in the 2023 cricket world cup final" → `research` → "Travis Head, 137 runs". That's correct, with 5 sources, approved and judged 5/5/5.

**Fix after real use:** "who won latest t20 worldcup" was routed to `chat`. The 3B model misread vague category descriptions. The new prompt makes `research` the default ("chat only if it's ONLY a greeting / thanks / small talk / arithmetic; when unsure, research") and adds examples (few-shot prompting). Tested on 8 messages: 6 correct. The 2 misses ("how are you doing", "5 times 7") now go to research, which is the safe mistake: slower, but you still get an answer.

**Alternatives:**
- Route with plain code (keywords, or a question mark plus length) instead of a model. It's free and instant, but easily fooled.
- Put the model call directly inside the conditional-edge function instead of a node. That's shorter, but the decision isn't saved in state or shown in the stream.
- More routes: e.g. `"math"` → a calculator tool, or `"followup"` → answer from the chat history without searching.
- `Command(goto=...)` returned from the router node: routing without a separate edge function.

## Fix: "latest" gave a 10-year-old answer

**Problem:** "who won latest t20 world cup" → "Australia, 2016". That's wrong twice: West Indies won in 2016, and it's years out of date.

**Cause:** The model doesn't know today's date. Its knowledge stops around its training date, so "latest" meant "latest it remembers", and the planner searched without a year.

**Fix:** `TODAY` and `YEAR` (from `date.today()`) are now in every prompt:
- Planner: "for 'latest' questions, put the year in the queries, e.g. 'T20 World Cup {YEAR} winner'".
- Agent: "your knowledge is out of date, trust the research results; for 'latest', pick the most recent event".
- Reviewer and judge: told today's date, so they can spot an old answer.

**Status:** Tested during step 11. The planner searched "T20 World Cup 2025 winner" and "T20 World Cup 2026 winner", and the answer was "India, 2026 T20 World Cup". That matches the search results (India beat New Zealand, 8 March 2026).

## Step 10: Memory that survives restarts (SQLite)

**What:**
- `InMemorySaver()` → `SqliteSaver(sqlite3.connect("memory.db", check_same_thread=False))`. Every checkpoint is now written to the file `memory.db` in the project folder.
- `check_same_thread=False`: the parallel `search` nodes run in other threads and use the same database connection.
- Conversation name from the command line: `python research_assistant.py cricket`. The name is the `thread_id`: the same name continues that conversation, a new name starts a fresh one. The default is `chat-1`.
- On start, `graph.get_state(config)` reads the saved state and prints how many messages that conversation has.

**Why:** RAM memory disappears when the program exits. With a file you can close the program and continue tomorrow. A question paused at the human-approval step (`interrupt`) also stays paused in the file.

**What we saw:** Run 1 with `demo`: "hi" → 0 saved messages before, 2 after. Run 2 with `demo`: "Conversation 'demo': 2 saved messages, last question: hi". The memory survived the restart.
 
**Alternatives:**
- `PostgresSaver` (`langgraph-checkpoint-postgres`): for real apps with many users or servers.
- `graph.get_state_history(config)`: list every saved checkpoint, then resume from an older one ("time travel" / undo).
- A `Store` for facts about the user shared across all conversations (a checkpointer is per conversation).
- To forget everything, delete `memory.db`.

## Step 11: Subgraph

Main graph:
START → router ──(chat)──→ chat → END
           └──(research)──→ [ research ] → judge → human ──(Enter)──→ END
                                ↑                    └──(feedback)──┘

Inside [ research ] (its own graph):
START → planner → search ×3 → agent ⇄ tools → reviewer ──(REVISE)──→ agent
                                                  └──(APPROVED)──→ END
```

**What:**
- `research_builder` is a separate `StateGraph` with planner, search, agent, tools and reviewer. Compiled into `research_graph`.
- `builder.add_node("research", research_graph)`: a compiled graph is added exactly like a node function.
- Both graphs use the same `State`, so the subgraph reads and writes the same keys with no conversion.
- The subgraph has no checkpointer of its own. It automatically uses the parent's (`memory.db`).
- The reviewer's "done" now goes to the subgraph's `END` (which returns to the parent), not directly to `judge`.
- Human feedback now goes back to `research`. That re-runs the whole subgraph from the planner, so it plans new searches for your feedback. `research` is reset to `None` first.
- `graph.stream(..., subgraphs=True)` also streams nodes inside the subgraph. Inner steps are printed indented.
- `graph.get_graph(xray=True)` draws the inside of subgraphs.
- The `add_or_reset` reducer now skips duplicates. When the subgraph finishes, it hands its whole `research` list back to the parent, and without the check every result would appear twice.

**What we saw:** Inner steps print indented under `[router] research`, then `[judge]` and approval run in the main graph. The parent state had 2 research items for 2 searches, so the duplicate check works.

**Why:**
- Organization: the main graph reads as 5 steps (router, chat, research, judge, human) instead of 9 nodes.
- Reuse: `research_graph` can be used in another app, or tested alone (`research_graph.invoke(...)`).
- Teams: each subgraph can be built and tested by different people. This is how "multi-agent" systems are made: each agent is a subgraph.

**Alternatives:**
- Different state for the subgraph: call it inside a normal node function (`def research(state): out = research_graph.invoke({...}); return {...}`) and convert the keys yourself. Needed when parent and child don't share keys.
- Keep one flat graph. That's fine for small apps; subgraphs pay off as graphs grow.
- Multi-agent libraries built on this idea: `langgraph-supervisor` (a boss agent picks which sub-agent works) and `langgraph-swarm` (agents hand off to each other).

## Step 12: Stronger judge → the whole assistant online (Gemini)
                                                                                  
**Problem:** The 3B judge approved wrong answers ("Steve Smith", "India 2023"). A 7B local model needs about 5 GB of RAM (only about 1 GB was free), and running models locally makes the laptop CPU hot.

**What:**
- `ChatGoogleGenerativeAI(model="gemini-3.8-flash")` from `langchain-google-genai`. It's a free-tier Gemini model, changeable with the `GEMINI_MODEL` env var.
- If `GOOGLE_API_KEY` is set, **every node** (router, planner, agent, reviewer, judge) uses Gemini online. If not, everything uses local `qwen2.5:3b` as before. The program prints which one at start.
- The graph code didn't change at all. Only `model = ...` changed. This is the point of LangChain chat models: they're interchangeable (`bind_tools`, `with_structured_output` and `invoke` work the same).
- `judge_llm = ....with_fallbacks([local judge])`: if the online judge fails (no internet, rate limit), the local model grades instead of the run crashing.
- `.content` → `.text` everywhere. Newer Gemini models can return content as a list of parts; `.text` always gives a plain string.

**Tested:** With a fake key, Gemini was rejected, the fallback kicked in, and the local judge correctly scored a wrong answer ("India won 2023", made-up URL) 1/1/1. The local-only path still works.

**Online run (real key):** "who won latest t20 world cup" → the planner used 3 queries, and the agent searched 3 more times by itself (dates, final, women's event). That's something the 3B model never did. The answer covered both the men's (India, 8 March 2026, beat NZ by 96 runs) and women's (Australia, July 2026) tournaments with 4 sources, and was judged 5/5/5. Judge test on a wrong answer ("India won 2023" + a made-up URL): accuracy 1, sources 1, with the reason "contradicts the search results… cites a fabricated source". The strong judge catches what the 3B one missed.

**Alternatives:**
- Mix models per node: a cheap/local model for the router and chat, and a strong one for the agent and judge. You'd just create two `model` objects.
- Other providers: `ChatAnthropic` (Claude), `ChatOpenAI`, `ChatGroq` (very fast, free tier). Same code, different class and key.
- `init_chat_model("google_genai:gemini-3.8-flash")`: pick the provider from one string, handy for switching via config.
- Ollama cloud models: the same `ChatOllama` code, but the model runs on Ollama's servers.

**Update: quota problem.** The free tier gave `gemini-3.8-flash` only **20 requests per day**, and a research question uses 5-8. Quotas are counted per model, so now:
- `model` = **`gemini-3.5-flash-lite`** for the many cheap calls (router, planner, agent, reviewer, chat). Change it with `GEMINI_FAST_MODEL`.
- `strong` = **`gemini-3.8-flash`** for the judge only (1 call per question). Change it with `GEMINI_MODEL`. It uses `max_retries=0`, so when its quota runs out it fails at once and the judge falls back to Flash-Lite instead of waiting.
- See your real limits at https://aistudio.google.com/rate-limit.

## Step 13: Web UI (Streamlit)

Run: `streamlit run app.py`, then open http://localhost:8501.

**What:**
- `app.py`: a chat page with a conversations sidebar (new, switch, delete), live steps while it works, and an approval panel with the judge's 3 scores, **Approve** and **Send back** buttons.
- Streamlit re-runs the whole script on every click. All real state lives in the LangGraph checkpointer (`memory.db`), so each re-run just reads it (`graph.get_state(config)`) and draws it. The UI and the terminal version share the same conversations.
- `graph.get_state(config).interrupts` tells whether the graph is paused at `interrupt()`. If it is, the approval panel shows, and the buttons call `Command(resume="")` (approve) or `Command(resume=feedback)`.
- Changes in `research_assistant.py`:
  - `show()` became `describe()`, which returns text instead of printing.
  - `steps()` yields each step, `pending_question()` checks for a pause, and `new_question()` builds the input. The terminal and the UI both use them.
  - Internal messages get a `name` (`"reviewer"`, `"feedback"`), so the UI can hide the reviewer's notes and show your feedback as ✏️.
- The chat shows only the final answer per question. Rejected drafts and tool calls stay in the "Steps behind the last answer" expander.

**What we saw:** "who won the 2023 cricket world cup and who was player of the match" → the reviewer caught an unsupported claim ("scored a century"), the agent searched the scorecard, and the answer was "Australia, 6 wickets, 19 Nov 2023, Travis Head 137 runs". The judge scored 5/5/5, and approval came from the button.

**Bugs found while testing:**
- Buttons disabled until the text box had a value swallowed the first click (Streamlit only commits typed text on blur). The buttons are now always enabled and ignore empty input.
- The error message was hidden inside the collapsed status box. It's now shown below it, with a friendly message for quota errors.

**Alternatives:**
- **Gradio** (`gr.ChatInterface`): similar and simple, good for demos and Hugging Face Spaces.
- **Chainlit**: made for LLM chat apps, with built-in step display.
- **LangGraph Studio / `langgraph dev`**: an official visual debugger that draws the graph and lets you step through it. It's for developers, not end users.
- **FastAPI + React**: a full custom web app. Much more work, total control.

## Step 14: Judge grounding

**Problem:** The judge just *said* "accuracy 5/5", and nothing checked it. The 3B judge once invented "Steve Smith was captain" and still gave 5/5.

**What:**
- The `Score` output changed: `claims` (a list of `Claim` with `claim` and `quote`) comes **first**, then `relevance` and `reason`. The model has to find evidence before it scores anything.
- The prompt says: copy the supporting sentence **word for word** from the search results, or leave it empty. It also warns that "your quotes are checked by code".
- `is_grounded(quote, results)`: plain code checks that the quote really is in the search results (ignoring case and spaces; `...` may skip text; it must be 20+ characters, because "Australia" alone proves nothing).
- **Code computes the scores**, not the model:
  - accuracy = share of claims with a real quote → 1-5
  - sources = share of cited URLs that appear in the search results → 1-5
  - relevance is still the model's opinion (there's no hard evidence for it).
- `search_text()` is shared by `evidence()` and `judge()`.
- The terminal shows ✅/❌ per claim. The web UI has a "🔍 Grounding: 3/3 claims found in the sources" expander with each claim and its quote.
- `test_grounding.py` (now `tests.py`): checks the grounding code with no model calls (`python tests.py`).

**What we saw:**
- Wrong answer ("India won, Kohli player of the match", fake URL) → ❌❌❌, accuracy 1, sources 1.
- Right answer ("Australia by six wickets, 19 Nov 2023, Head 137") → ✅✅✅, 5/5/5.

**Why:** A model can hallucinate while judging, too. A quote can be checked by code, a score can't. This is the difference between "trust me" and "show me".

**Limit:** It's an exact match, so a quote the model slightly reworded counts as ❌. If that happens often, switch to fuzzy matching with `difflib` (marked `ponytail:` in the code).

**Alternatives:**
- **Gemini "Grounding with Google Search"**: the judge checks facts against a live Google search instead of our DuckDuckGo results. Stronger, but it has its own quota and verifies against different sources than the agent used.
- **NLI / fact-checking models**: a small classifier that says whether sentence A supports sentence B. More forgiving of rewording than exact match.
- **Citations inside the answer**: make the *agent* put a quote next to every fact, so grounding is checked while writing, not only when judging.

## Step 15: Tavily search

**What:**
- `TavilyClient()` is used when `TAVILY_API_KEY` is set. `search_depth="basic"` costs 1 credit; the free plan has 1,000 per month.
- `web_search` tries the search engines in order, `[search_tavily, search_ddg]`. If Tavily fails (no credits, network), it falls back to DuckDuckGo instead of failing the step.
- Each result is now title + URL + `Published:` date (when Tavily knows it) + page text. The date helps with "latest" questions.
- The program and the UI show which search is active (`SEARCH_NAME`).

**Tested:** With a fake key, Tavily was rejected and DuckDuckGo results came back. **A real Tavily run is pending your key** (https://app.tavily.com, then `setx TAVILY_API_KEY "tvly-..."` and a new terminal).

## Step 16: The grounded judge becomes the reviewer (efficiency)

```
Before:  agent → reviewer (model: APPROVED / REVISE) → ... → judge (model: claims + quotes) → human
After:   agent → reviewer (model: claims + quotes, code checks them) → human
```

**What:**
- The old reviewer asked a model for "APPROVED / REVISE". The judge asked another model for grounded claims. Both judged the same answer.
- Now `reviewer` calls `grade()` (the grounded check from step 14) once. It approves if every claim has a real quote and relevance is at least 3.
- If not, the feedback lists **exactly** the unsupported claims: "Not supported by the search results: X. Remove it or use web_search to find a source." The old feedback was a vague "REVISE: ...".
- The separate `judge` node and the `REVIEW` prompt are deleted. The main graph is now `router → research → human`.
- The scores are saved by the reviewer, so the approval panel still shows them with the ✅/❌ claims.
- `temperature=0` removed for Gemini. These models use fixed sampling and only printed a warning.

**Why:**
- Efficient: 1 model call fewer per question (about 5 instead of 6-7), which matters with 20-per-day quotas.
- Proper: the loop now fixes the *specific* unsupported facts, and the check that approves the answer is the same check you see.

**What we saw:** "who won the 2023 world cup and who was player of the match" → 2 parallel searches + 1 extra agent search → approved on review 1 with 4/4 claims ✅ (Australia, six wickets, sixth title, Head 137 off 120). That took 5 model calls in total.

## Step 17: Shorter history

**Problem:** Every model call got `state["messages"]`, the **whole** saved conversation: every old search result (thousands of characters each), tool calls, rejected drafts and reviewer notes. The input grows with every question, which is slower and costs more tokens. Worse, old search results can confuse the model on a new question.

**What:**
- The checkpointer still **saves everything** (nothing is deleted); only what we *send* to the model changes.
- `split_turns(messages)`: splits into earlier turns and the current turn. The current turn starts at the user's latest real question (a human message with no `name`).
- `compact(earlier)`: keeps only the **question → final answer** pairs of the last `MAX_TURNS = 3` turns. It drops search results, tool calls, rejected drafts and reviewer notes, and never starts with an AI message (Gemini wants the user first).
- `history(state)`: for the **agent**, compact earlier turns + the current turn **in full** (it needs its own searches and the reviewer's feedback).
- `brief(state)`: for the **router, planner and chat**, compact earlier turns + only this turn's question and user feedback. They never need search results.
- `search_text()` (the evidence the reviewer checks) now uses only this turn's tool searches, not old turns.
- `tests.py` (renamed from `test_grounding.py`): grounding checks + history checks, no model calls.

**What we saw:**
- `tests.py`, fake 6-question conversation: agent input **57,270 → 9,582 characters (84% smaller)**. The router and planner get no search results at all.
- Live: "who won the 2023 cricket world cup" → Australia. Then "who was the captain of **that team**" → planner searched "2023 Cricket World Cup winning captain Australia" → "Pat Cummins" ✅. Follow-ups still work. The router and planner got 3 messages (666 characters) instead of the full 2,913.

**Alternatives:**
- `trim_messages(messages, max_tokens=..., strategy="last")` from LangChain: cuts by token count instead of by turns. Simpler, but it can cut in the middle of a tool call / tool result pair.
- **Summarize** old turns: a node that writes a running summary of the conversation once it gets long. It keeps more context, but costs an extra model call.
- `RemoveMessage` to actually **delete** old messages from state. That saves storage, but you lose the full record.

## Step 18: Retry on errors (`RetryPolicy`)

**Problem:** One network hiccup, Gemini server error, per-minute rate limit, or a model reply that didn't fit our schema failed the **whole question**.

**What:**
- `RETRY = RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=3.0, retry_on=should_retry)`: up to 3 tries, waiting about 2s and then about 6s (with a little random jitter so retries don't all hit at once).
- `add_node(..., retry_policy=RETRY)` on every node that calls a model: router, chat, planner, agent, reviewer. `search` and `tools` don't need it, because `web_search` already catches its own errors.
- A retry re-runs **only that node**, from the same saved state. The work before it isn't repeated.
- `should_retry(error)`, our own rule:
  - **retry:** network errors, server errors, per-minute rate limits, and a reply that didn't fit the Pydantic schema (a `ValueError`; **LangGraph's default rule would not retry this**).
  - **don't retry:** Gemini's *daily* quota (`PerDay` in the error), because waiting seconds can't fix it, and bugs in our code (`KeyError`, `TypeError`, ...), which should fail at once so we see them.
- `tests.py`: `should_retry` cases + a tiny graph whose node fails once with a network error. It retries and succeeds (`attempts == 2`).

**Tested:** `tests.py` passes (node failed once, retried, succeeded). Live "hi" → router → chat works with the policies attached.

**Layers of protection now:**
1. Inside the model client: `ChatGoogleGenerativeAI` retries its own HTTP calls (6 times by default; 0 for the judge model, so its fallback kicks in fast).
2. `with_fallbacks`: the strong judge → Flash-Lite.
3. `RetryPolicy`: re-runs the whole node (step 18).
4. The UI shows a clear error if everything still fails, and the saved state lets you just ask again.

**Alternatives:**
- LangGraph's default `retry_on`: only network and 5xx errors. Safer against loops, but misses bad model replies.
- Retry inside each node with `tenacity` (`@retry`): more control, more code.
- `error_handler=` on `add_node` (new in LangGraph 1.2): run a node when a node fails, e.g. to write "sorry, try again" into the chat instead of raising.
- `timeout=` on `add_node`: a time limit per attempt. It only works with **async** nodes, and ours are sync.

## Step 19: Groq (`gpt-oss-120b`) for the agent and reviewer

**Which model does what now:**

| Step | Model (→ fallback if it fails) |
|---|---|
| agent | Groq `openai/gpt-oss-120b` → `gemini-3.5-flash-lite` |
| reviewer | Groq `openai/gpt-oss-120b` → `gemini-3.8-flash` → `gemini-3.5-flash-lite` |
| router, planner, chat | `gemini-3.5-flash-lite` |
| no keys at all | local `qwen2.5:3b` for everything |

**What:**
- `ChatGroq(model="openai/gpt-oss-120b", reasoning_effort="low", max_retries=1)`. Groq's free tier gives 1,000 requests/day per model (vs 20 for Gemini Flash), is very fast, and needs no card. Change the model with `GROQ_MODEL`.
- `reasoning_effort="low"`: gpt-oss "thinks" before answering, and less thinking means fewer tokens and a faster reply.
- `max_retries=1`: Groq's free limit is only **8,000 tokens per minute**. On a 429 we switch to Gemini at once instead of waiting out the minute.
- `chain(models, wrap)`: builds "first model, then the next, then the next" with `with_fallbacks`, for tools (`bind_tools`) or structured output.
- `MAX_RESULT_CHARS = 800`: each search result's text is cut to 800 characters, so an agent/reviewer call stays around 3,000 tokens (under Groq's limit).
- The steps show which model really answered (`[agent: openai/gpt-oss-120b]`), taken from `response_metadata["model_name"]`, so fallbacks are visible.

**Problems found by testing, and fixed:**
1. **gpt-oss writes `**Sources**`, not `Sources:`**. The old text check rejected good answers and skipped grounding. Now the check is "does the answer cite a URL?" (`cited_urls`).
2. **gpt-oss wraps links as `【https://…】`**. The URL pattern now strips `【】` along with `()[]<>`.
3. **Look-alike characters broke grounding.** gpt-oss writes curly `’` and non-breaking hyphens (`back‑to‑back`), while pages use `'` and `-`. A correct quote was marked ❌. `normalize()` now maps these look-alikes (`LOOKALIKES`) before matching.
4. The Gemini fallback logged "Dropping reasoning block from provider 'groq'" each time. That's correct behaviour (Gemini can't use Groq's hidden reasoning), so the warning is silenced.
- `tests.py` has new checks for 1-3.

**What we saw:**
- "who won the latest t20 world cup" → India, beat New Zealand, 8 March 2026 → approved on review 1, all ✅, **10-15 seconds** (Gemini-only took 20-40s).
- "who was the captain of that team" → Suryakumar Yadav → approved on review 1, ✅ after the look-alike fix (❌ ❌ before it).
- Once, Groq failed mid-question (token limit) and `[agent: gemini-3.5-flash-lite]` took over. The fallback works.

**Alternatives:**
- `openai/gpt-oss-20b` or `qwen/qwen3.8-27b` on Groq: separate 1,000/day quotas, so they could serve as more fallbacks.
- OpenRouter `:free` models: 50 requests/day in total (1,000 after buying $10 of credit).
- Cerebras: very fast, but the no-card free tier ended in Aug 2026.

## Step 20: Fallback for router, planner and chat

**Problem:** These three used only Gemini Flash-Lite. If its daily quota ran out, every question stopped at the router, so even the Groq agent couldn't help.

**What:**
- `groq_small = ChatGroq(model="openai/gpt-oss-20b", reasoning_effort="low", max_retries=1)`: a smaller gpt-oss with its **own separate** 1,000/day quota, so it doesn't eat into the agent's 120b quota. Change it with `GROQ_SMALL_MODEL`.
- `fast_models = [Flash-Lite, gpt-oss-20b]`. `router_llm`, `planner_llm` and the new `chat_llm` are all built with `chain(fast_models, ...)`, the same helper as the agent and reviewer.
- Flash-Lite `max_retries` 6 → 1: with a fallback available, a quota error now switches within seconds instead of retrying for up to a minute.
- The chat step shows which model answered (`[chat: openai/gpt-oss-20b]`).

**Tested:** Flash-Lite was forced to fail (`GEMINI_FAST_MODEL=gemini-does-not-exist`):
- "hi" → router ✓ → `[chat: openai/gpt-oss-20b]` reply in 3s.
- "who won the 2023 cricket world cup" → router ✓ → planner ✓ (2 queries) → agent gpt-oss-120b → approved, ✅ → 8s.

**Every step now has a fallback:**

| Step | Main → fallbacks |
|---|---|
| router, planner, chat | Gemini 3.5 Flash-Lite → Groq gpt-oss-20b |
| search | Tavily → DuckDuckGo |
| agent | Groq gpt-oss-120b → Gemini 3.5 Flash-Lite |
| reviewer | Groq gpt-oss-120b → Gemini 3.8 Flash → Gemini 3.5 Flash-Lite |

Plus `RetryPolicy` (3 tries per step) around all of them.

## Note: Tavily (explained before step 15)

A search API built for AI agents, a drop-in replacement for DuckDuckGo in `web_search`.
- **Better than DDG:** it returns cleaned page content (not just a 1-2 line snippet), can fetch the full page text (`include_raw_content`), can prefer recent news (`topic="news"`, `time_range`), and filters sites (`include_domains`). It also doesn't get rate-blocked like scraping DuckDuckGo.
- **Cost:** free plan with 1,000 credits per month and no card. A basic search costs 1 credit and an advanced one 2. Our planner makes 2-3 searches per question, so roughly 300-500 questions a month.
- **Change needed:** `pip install langchain-tavily`, set `TAVILY_API_KEY`, and replace the body of `web_search` (or use the ready-made `TavilySearch` tool). The graph doesn't change.
- **Grounding bonus:** longer page text means more real sentences the judge can quote, so fewer honest claims get ❌.
