# 💍 Destination Wedding Planner Agent

> A **multi-agent system** that plans your destination wedding end-to-end — flights, venues, and a curated playlist — from a single sentence.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.3%2B-informational)](https://python.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0%2B-informational)](https://langchain-ai.github.io/langgraph)
[![MCP](https://img.shields.io/badge/MCP-Kiwi%20Travel-blue)](https://mcp.kiwi.com)
[![Groq](https://img.shields.io/badge/Groq-free%20tier-orange)](https://console.groq.com)
[![Tavily](https://img.shields.io/badge/Tavily-search%20API-green)](https://tavily.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## What It Does

You type one sentence:

> *"I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre"*

The coordinator agent:
1. **Extracts** origin, destination, guest count, and genre
2. **Saves them to shared state** so every sub-agent can access them
3. **Delegates in parallel** to three specialist sub-agents
4. **Compiles** all results into a complete, formatted wedding plan

### The Dream Team

| Agent | Tool | Speciality |
|-------|------|-----------|
| 🎯 **Coordinator** | `update_state` + calls all 3 | Gathers details, delegates, compiles the plan |
| ✈ **Travel Agent** | Kiwi Travel MCP (live flights) | Finds real economy flights, optimised for price and timing |
| 🏛 **Venue Agent** | Tavily web search | Finds venues matching location and guest capacity |
| 🎵 **Playlist Agent** | SQL (Chinook music DB) | Curates a genre-matching playlist with duration and cost |

### Sample output

```
You: I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre

Coordinator:
  ✈ FLIGHTS (London → Paris)
  ─────────────────────────────────────────────────
  • LHR → CDG: from ~£65 one-way economy
    Best season: late May / early June
  • LGW → ORY: from ~£45 (budget option)
  Tip: block-book group flights 12 months ahead

  🏛 VENUES (Paris, 100 guests)
  ─────────────────────────────────────────────────
  1. Hôtel Rochechouart (9th)  — €8k–€12k  ⭐ 4.7
  2. Château de Vaux-le-Vicomte — €15k–€25k ⭐ 4.9
  3. La Salle Pleyel (17th)    — €10k–€18k  ⭐ 4.6

  🎵 JAZZ PLAYLIST (20 tracks, ~1h 45min, $19.80)
  ─────────────────────────────────────────────────
  All Blues · Miles Davis · 5:37 · $0.99
  So What   · Miles Davis · 9:22 · $0.99
  Take Five · Dave Brubeck · 5:24 · $0.99
  ...
```

---

## Architecture

```
User: "London → Paris, 100 guests, jazz"
             │
             ▼
      ┌──────────────────────────────────────────┐
      │           Coordinator Agent              │
      │  tools: update_state, search_flights,    │
      │          search_venues, suggest_playlist  │
      │  state_schema: WeddingState              │
      └─────────────────┬────────────────────────┘
                        │
           ┌────────────┤  Step 1: update_state
           │            │  Command.update → WeddingState
           │            │
           │  ┌─────────▼──────────────────────────┐
           │  │  WeddingState (shared, per thread)  │
           │  │  origin · destination               │
           │  │  guest_count · genre                │
           │  └────────────────────────────────────┘
           │
           ├──► Step 2: search_flights(runtime)
           │        reads state["origin"] + state["destination"]
           │        ┌──────────────────────────────┐
           │        │  Travel Agent                │
           │        │  tools: [Kiwi Travel MCP]    │
           │        └──────────────────────────────┘
           │
           ├──► Step 3: search_venues(runtime)
           │        reads state["destination"] + state["guest_count"]
           │        ┌──────────────────────────────┐
           │        │  Venue Agent                 │
           │        │  tools: [web_search]         │
           │        └──────────────────────────────┘
           │
           └──► Step 4: suggest_playlist(runtime)
                    reads state["genre"]
                    ┌──────────────────────────────┐
                    │  Playlist Agent              │
                    │  tools: [query_playlist_db]  │
                    └──────────────────────────────┘
             │
             ▼
      Coordinator compiles → Final wedding plan
```

### Message flow inside one coordinator turn

| Step | Message type | Content |
|------|-------------|---------|
| 1 | `HumanMessage` | User's wedding request |
| 2 | `AIMessage` | `tool_calls: [update_state(...)]` |
| 3 | `ToolMessage` | "State saved: London → Paris, 100 guests, jazz" |
| 4 | `AIMessage` | `tool_calls: [search_flights()]` |
| 5 | `ToolMessage` | Travel agent's full flight results (entire sub-agent run hidden here) |
| 6 | `AIMessage` | `tool_calls: [search_venues()]` |
| 7 | `ToolMessage` | Venue agent's results |
| 8 | `AIMessage` | `tool_calls: [suggest_playlist()]` |
| 9 | `ToolMessage` | Playlist agent's results |
| 10 | `AIMessage` | Final compiled wedding plan |

---

## Key Concepts Implemented

| Concept | Where | Why it matters |
|---------|-------|----------------|
| `create_agent()` × 4 | `coordinator.py` | One coordinator + 3 sub-agents, each focused and independent |
| **Supervisor + sub-agent pattern** | `coordinator.py` | Core multi-agent architecture — orchestrator delegates via tool calls |
| **`@tool` wrapping an agent** | `search_flights` etc. | The trick: any sub-agent is just a tool from the coordinator's view |
| `WeddingState(AgentState)` | `coordinator.py` | Shared mutable state — 4 fields persisted per `thread_id` |
| `Command(update={...})` | `update_state` tool | LangGraph primitive: writes to state from inside a tool |
| `runtime.state["field"]` | Sub-agent tools | Reads live state values inside a tool — no direct arg passing needed |
| `MultiServerMCPClient` | `coordinator.py` | Connects to Kiwi Travel MCP server over `streamable_http` |
| `RetryMCPInterceptor` | `coordinator.py` | Retries transient MCP failures; surfaces errors gracefully |
| `ainvoke` (async) | `search_flights` | Required when a sub-agent uses async tools (MCP) |
| `recursion_limit=40` | `config` dict | Multi-agent chains need more steps than the default budget |
| LangSmith tracing | `config["tags"]` | Reveals the full nested call tree — essential for multi-agent debugging |

---

## Project Structure

```
wedding-planner-agent/
├── coordinator.py    # Full multi-agent system — run as CLI
├── demo.ipynb        # Step-by-step notebook with explanations and outputs
├── data/
│   └── Chinook.db    # SQLite music database (tracks, artists, prices)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/marehman-exe/wedding-planner-agent.git
cd wedding-planner-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Set up API keys

```bash
cp .env.example .env   # macOS/Linux
copy .env.example .env  # Windows
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

| Key | Cost | Where to get it |
|-----|------|-----------------|
| `GROQ_API_KEY` | Free | [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | Free tier | [tavily.com](https://tavily.com) |

> **Kiwi Travel MCP** (`https://mcp.kiwi.com`) is a free public server — no API key required.

### 4. Run

**CLI:**

```bash
python coordinator.py
```

```
Destination Wedding Planner  |  type 'quit' to exit
Example: I'm from London and I'd like a wedding in Paris for 100 guests, jazz genre
----------------------------------------------------------------------

You: I'm from London and I'd like a wedding in Paris for 100 guests, jazz genre
[Planning your wedding — this may take 1–2 minutes...]

Coordinator:
  ✈ Flights, 🏛 Venues, 🎵 Playlist all compiled below...
```

**Jupyter notebook (step-by-step):**

```bash
jupyter lab
# Open demo.ipynb
```

> ⏱️ **Expect 1–3 minutes per run.** The coordinator runs 4 tool calls in sequence, and each sub-agent may itself make multiple calls (Kiwi MCP retries, Tavily web searches, SQL queries).

---

## How Shared State Works

This is what makes multi-agent coordination clean — sub-agents don't need to be told the wedding details directly:

```python
# 1. Coordinator tells update_state what it learned from the user
@tool
def update_state(origin, destination, guest_count, genre, runtime: ToolRuntime) -> Command:
    return Command(update={
        "origin": origin,          # ← written into WeddingState
        "destination": destination,
        "guest_count": guest_count,
        "genre": genre,
        "messages": [...],
    })

# 2. Sub-agent tools read from the same state — no args needed
@tool
def search_flights(runtime: ToolRuntime) -> str:
    origin      = runtime.state["origin"]       # ← read from WeddingState
    destination = runtime.state["destination"]
    return travel_agent.ainvoke(...)

# 3. The WeddingState class ties it all together
class WeddingState(AgentState):
    origin: str
    destination: str
    guest_count: str
    genre: str
```

The `InMemorySaver` checkpointer (used internally by the coordinator) persists this state for the duration of the run, scoped to the `thread_id`.

---

## What I Learned Building This

This project was built as the capstone of **Module 2: Advanced Agent** from [LangChain Academy's Introduction to LangChain (Python)](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python).

### Concepts applied end-to-end

**1. Multi-agent architecture (Supervisor pattern)**  
Breaking a complex task into specialised sub-agents is the practical solution to context-window overload. The coordinator stays lean — it only sees 4 tool calls, not MCP schemas, SQL tables, or web search results.

**2. Sub-agents as tools**  
The elegance of LangChain's multi-agent design is that wrapping a `create_agent()` graph inside a `@tool` requires almost no extra code. Tool calling is the universal interface.

**3. Shared state via `WeddingState`**  
`Command(update={...})` and `runtime.state["field"]` work together to create a clean shared-memory contract. The coordinator writes once; all sub-agents read independently.

**4. MCP (Model Context Protocol)**  
Using a community MCP server (Kiwi Travel) unlocked real-time flight search without building or maintaining an API integration. The `RetryMCPInterceptor` made it production-stable.

**5. Async agents**  
The MCP client is async-only (`await client.get_tools()`). Any agent that uses async tools must itself be invoked with `ainvoke` — and any tool that calls such an agent must be an `async def`.

**6. LangSmith tracing for multi-agent systems**  
The outer message list is deceptive — `ToolMessage` 5 looks like a short string, but it contains an entire sub-agent run (multiple MCP calls, retries, reasoning). LangSmith shows the full tree.

---

## Tech Stack

| Library | Version | Role |
|---------|---------|------|
| `langchain` | ≥ 1.3 | Agent framework, tool decorator, message types |
| `langgraph` | ≥ 1.0 | State machine, `create_agent()`, `InMemorySaver`, `Command` |
| `langchain-mcp-adapters` | ≥ 0.1 | `MultiServerMCPClient` — connects agents to MCP servers |
| `mcp` | ≥ 1.21 | MCP client/server protocol types |
| `langchain-community` | ≥ 0.4 | `SQLDatabase` for Chinook DB queries |
| `langchain-groq` | ≥ 1.1 | Groq provider integration |
| `tavily` | ≥ 0.7 | Real-time web search |
| `python-dotenv` | ≥ 1.2 | Load API keys from `.env` |

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built as part of the [Introduction to LangChain — Python](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python) course by [LangChain Academy](https://academy.langchain.com).*
