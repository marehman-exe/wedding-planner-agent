"""
Destination Wedding Planner Agent
----------------------------------
A multi-agent system that plans your destination wedding end-to-end.
Tell it your origin, destination, guest count, and music genre in one message —
the coordinator gathers the details and delegates to three specialist sub-agents:

  ✈  Travel Agent    — finds real flights via the Kiwi Travel MCP server
  🏛  Venue Agent     — searches the web for venues matching your location & capacity
  🎵  Playlist Agent  — queries a music database and curates a genre-matching playlist

Features:
  • Input validation  — guest count must be a positive integer; origin/destination required
  • Follow-up questions — coordinator asks for missing details instead of guessing
  • Structured output — consistent sections for flights, venues, and playlist
  • Sample-data fallback — clearly labelled example results when a service is unavailable
  • Improved error messages — each failure names the service and suggests next steps

Built with: LangChain · LangGraph · Tavily · MCP (Kiwi Travel) · SQLite · Groq
Author: Muhammad Ali Rehman (marehman-exe)
Course: Introduction to LangChain – Python (LangChain Academy, Module 2)
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from typing import Dict, Any

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.utilities import SQLDatabase
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent
from tavily import TavilyClient

# ── MCP error handling ────────────────────────────────────────────────────────
# Kiwi's MCP server can return transient errors (-32603).
# This interceptor retries those and converts all failures to graceful strings
# so the agent can adjust and try again instead of crashing.

RETRYABLE_MCP_CODES = {-32603}

class RetryMCPInterceptor:
    """Retry transient MCP errors; surface all failures as readable messages."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                print(f"[MCP] {request.name} error (code {exc.error.code}, "
                      f"attempt {attempt + 1}/{self.max_retries}): {exc}")
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text",
                                             text=f"Tool call failed (non-retryable): {exc}")],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
                print(f"[MCP] {request.name} exception "
                      f"(attempt {attempt + 1}/{self.max_retries}): {exc}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        return CallToolResult(
            content=[TextContent(type="text",
                                 text=f"Tool failed after {self.max_retries} retries: {last_error}")],
            isError=False,
        )

# ── Input validation ──────────────────────────────────────────────────────────

REQUIRED_FIELDS = ("origin", "destination", "guest_count", "genre")

def validate_inputs(
    origin: str,
    destination: str,
    guest_count: str,
    genre: str,
) -> list[str]:
    """Return a list of human-readable error strings.
    An empty list means all inputs are valid.
    """
    errors: list[str] = []

    if not origin or not origin.strip():
        errors.append("Origin city is required (e.g. 'London').")
    if not destination or not destination.strip():
        errors.append("Destination city is required (e.g. 'Paris').")
    if not genre or not genre.strip():
        errors.append("Music genre is required (e.g. 'jazz').")

    # guest_count must parse as a positive integer
    if not guest_count or not guest_count.strip():
        errors.append("Guest count is required (e.g. '100').")
    else:
        cleaned = guest_count.strip()
        try:
            n = int(cleaned)
            if n <= 0:
                errors.append(
                    f"Guest count must be a positive whole number (got '{cleaned}')."
                )
        except ValueError:
            errors.append(
                f"Guest count must be a whole number (got '{cleaned}'). "
                "Example: '80' or '120'."
            )

    return errors


# ── Shared state ──────────────────────────────────────────────────────────────
# WeddingState extends AgentState so the coordinator and sub-agents all share
# the wedding fields — persisted per thread_id by InMemorySaver.
#
# missing_fields  — fields the coordinator still needs to ask for
# validation_errors — human-readable errors from validate_inputs()

class WeddingState(AgentState):
    origin: str            # e.g. "London"
    destination: str       # e.g. "Paris"
    guest_count: str       # e.g. "100"
    genre: str             # e.g. "jazz"
    missing_fields: list   # fields still needed before planning can proceed
    validation_errors: list  # errors to show the user before retrying

# ── Tools: web search + SQL ───────────────────────────────────────────────────

tavily_client = TavilyClient()

@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. Track your search count with search_number
    (starting at 1) and max_search_number on every call. Use plain text only."""
    if search_number > max_search_number:
        return {"message": "Search limit reached. Summarise findings and provide final answer."}
    try:
        return tavily_client.search(query)
    except Exception as e:
        return {"error": str(e)}

db = SQLDatabase.from_uri("sqlite:///data/Chinook.db")

@tool
def query_playlist_db(query: str) -> str:
    """Query the Chinook music database for playlist information (SQLite).
    Discover the schema first before writing data queries."""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"

# ── Sub-agents ────────────────────────────────────────────────────────────────
# Each sub-agent is created inside an async function because the travel agent
# needs to await the MCP client to fetch its tools.

async def build_agents():
    """Build all sub-agents and the coordinator. Returns the coordinator."""

    # Travel agent — uses live Kiwi Travel MCP server for real flight data
    mcp_client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com",
            }
        },
        tool_interceptors=[RetryMCPInterceptor()],
    )
    mcp_tools = await mcp_client.get_tools()

    travel_agent = create_agent(
        model="groq:openai/gpt-oss-120b",
        tools=mcp_tools,
        system_prompt="""
        You are a travel agent specialising in group wedding travel.
        Search for the best one-way economy flights from the given origin to the destination.
        Optimise for: lowest price → shortest duration → best wedding-season timing.
        Search only for one ticket as a representative price.
        Make multiple searches if needed to narrow down options.
        If an MCP tool call fails, retry it. Return your top 2–3 shortlisted flight options.
        Do NOT ask follow-up questions.
        """,
    )

    # Venue agent — uses Tavily web search
    venue_agent = create_agent(
        model="groq:openai/gpt-oss-120b",
        tools=[web_search],
        system_prompt="""
        You are a venue specialist. Search the web for wedding venues at the given destination
        that match the guest count. Optimise for: lowest price → exact capacity → best reviews.
        Make up to 12 web searches; count every call. After 12 searches, summarise what you found.
        Do NOT ask follow-up questions.
        """,
    )

    # Playlist agent — queries Chinook SQLite music database
    playlist_agent = create_agent(
        model="groq:openai/gpt-oss-120b",
        tools=[query_playlist_db],
        system_prompt="""
        You are a wedding playlist specialist. Query the Chinook music database to curate
        a playlist that matches the requested genre. Discover the schema first, then query
        for matching tracks. Calculate total duration and cost. If a query errors, fix and retry.
        Return a formatted playlist with track names, artists, duration, and total cost.
        """,
    )

    # ── Coordinator tools ─────────────────────────────────────────────────────
    # Each sub-agent is wrapped as a @tool so the coordinator can call it.
    # The coordinator reads wedding details from the shared WeddingState.

    # Sample data — returned when a live service is unavailable.
    # Clearly labelled as estimates so users know to verify with providers.
    _SAMPLE_FLIGHTS = (
        "[SAMPLE DATA — live flight search unavailable]\n"
        "These are illustrative examples only. Verify with a travel provider.\n\n"
        "• London Heathrow (LHR) → Paris CDG: ~£60–80 economy, ~1h 15min\n"
        "• London Gatwick (LGW) → Paris Orly (ORY): ~£45–65 economy, ~1h 10min\n"
        "Tip: book group flights 6–12 months in advance for best rates."
    )
    _SAMPLE_VENUES = (
        "[SAMPLE DATA — live venue search unavailable]\n"
        "These are illustrative examples only. Verify with venues directly.\n\n"
        "• Hôtel Rochechouart, Paris — capacity ~120, est. €8,000–€12,000\n"
        "• La Salle Pleyel, Paris    — capacity ~100, est. €10,000–€18,000\n"
        "• Château de Vaux-le-Vicomte — capacity ~150, est. €15,000–€25,000"
    )
    _SAMPLE_PLAYLIST = (
        "[SAMPLE DATA — music database query unavailable]\n"
        "These are illustrative jazz tracks only.\n\n"
        "• So What — Miles Davis, 9:22\n"
        "• Take Five — Dave Brubeck, 5:24\n"
        "• All Blues — Miles Davis, 5:37\n"
        "• Autumn Leaves — Bill Evans, 5:52\n"
        "Est. total: ~26 min for 4 tracks"
    )

    @tool
    def ask_followup(question: str, missing: str, runtime: ToolRuntime) -> Command:
        """Ask the user a follow-up question when required details are missing.
        Use this instead of guessing. Provide a clear, friendly question.
        'missing' is a comma-separated list of the field names still needed."""
        missing_list = [f.strip() for f in missing.split(",") if f.strip()]
        return Command(
            update={
                "missing_fields": missing_list,
                "messages": [ToolMessage(
                    question,
                    tool_call_id=runtime.tool_call_id,
                )],
            }
        )

    @tool
    def update_state(
        origin: str,
        destination: str,
        guest_count: str,
        genre: str,
        runtime: ToolRuntime,
    ) -> Command:
        """Save the wedding details (origin, destination, guest_count, genre) to shared state.
        Validates all four fields before saving. If validation fails, the errors are saved
        to state and returned so the coordinator can ask the user to fix them.
        Call this FIRST, before delegating to any sub-agent. Call it alone — no parallel calls."""
        # Validate before writing — never store bad values in state
        errors = validate_inputs(origin, destination, guest_count, genre)
        if errors:
            error_text = (
                "The following details need to be corrected before planning can continue:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )
            return Command(
                update={
                    "validation_errors": errors,
                    "messages": [ToolMessage(
                        error_text,
                        tool_call_id=runtime.tool_call_id,
                    )],
                }
            )
        return Command(
            update={
                "origin": origin.strip(),
                "destination": destination.strip(),
                "guest_count": guest_count.strip(),
                "genre": genre.strip(),
                "missing_fields": [],
                "validation_errors": [],
                "messages": [ToolMessage(
                    f"Details saved: {origin.strip()} → {destination.strip()}, "
                    f"{guest_count.strip()} guests, {genre.strip()} music. "
                    "Ready to start planning.",
                    tool_call_id=runtime.tool_call_id,
                )],
            }
        )

    @tool
    async def search_flights(runtime: ToolRuntime) -> str:
        """Delegate to the travel sub-agent to find flights using the wedding origin and destination."""
        origin = runtime.state.get("origin", "")
        destination = runtime.state.get("destination", "")
        if not origin or not destination:
            return (
                "[Flight Search — Missing Info]\n"
                "Origin or destination is not set. "
                "Please provide both before searching for flights."
            )
        try:
            response = await travel_agent.ainvoke({
                "messages": [HumanMessage(
                    content=f"Find flights from {origin} to {destination}"
                )]
            })
            result = response["messages"][-1].content
            # Tag the section header for consistent output formatting
            return f"✈ FLIGHTS ({origin} → {destination})\n{'-'*50}\n{result}"
        except Exception as exc:
            # Named failure — tells the user which service broke and what to do
            return (
                f"✈ FLIGHTS — Service Unavailable\n"
                f"The flight search service could not be reached ({exc}).\n"
                f"You can retry, or use the sample data below:\n\n{_SAMPLE_FLIGHTS}"
            )

    @tool
    def search_venues(runtime: ToolRuntime) -> str:
        """Delegate to the venue sub-agent to find venues at the destination for the guest count."""
        destination = runtime.state.get("destination", "")
        guest_count = runtime.state.get("guest_count", "")
        if not destination or not guest_count:
            return (
                "[Venue Search — Missing Info]\n"
                "Destination or guest count is not set. "
                "Please provide both before searching for venues."
            )
        try:
            response = venue_agent.invoke({
                "messages": [HumanMessage(
                    content=f"Find wedding venues in {destination} for {guest_count} guests"
                )]
            })
            result = response["messages"][-1].content
            return f"🏛 VENUES ({destination}, {guest_count} guests)\n{'-'*50}\n{result}"
        except Exception as exc:
            return (
                f"🏛 VENUES — Service Unavailable\n"
                f"The venue search service could not be reached ({exc}).\n"
                f"You can retry, or use the sample data below:\n\n{_SAMPLE_VENUES}"
            )

    @tool
    def suggest_playlist(runtime: ToolRuntime) -> str:
        """Delegate to the playlist sub-agent to curate a playlist for the wedding genre."""
        genre = runtime.state.get("genre", "")
        if not genre:
            return (
                "[Playlist — Missing Info]\n"
                "Music genre is not set. Please provide a genre before building a playlist."
            )
        try:
            response = playlist_agent.invoke({
                "messages": [HumanMessage(
                    content=f"Find {genre} tracks for a wedding playlist"
                )]
            })
            result = response["messages"][-1].content
            return f"🎵 PLAYLIST ({genre})\n{'-'*50}\n{result}"
        except Exception as exc:
            return (
                f"🎵 PLAYLIST — Service Unavailable\n"
                f"The music database could not be reached ({exc}).\n"
                f"You can retry, or use the sample data below:\n\n{_SAMPLE_PLAYLIST}"
            )

    # ── Coordinator ───────────────────────────────────────────────────────────

    coordinator = create_agent(
        model="groq:openai/gpt-oss-120b",
        tools=[ask_followup, update_state, search_flights, search_venues, suggest_playlist],
        state_schema=WeddingState,
        system_prompt="""
        You are a wedding coordinator managing a dream team of specialists.

        Your workflow — follow this order strictly:

        STEP 1 — Check for required details.
        You need four pieces of information: origin city, destination city,
        guest count (a positive whole number), and music genre.
        If any are missing from the user's message, call ask_followup with a clear,
        friendly question listing exactly what is needed. Then wait for the user's reply.

        STEP 2 — Save the details.
        Once you have all four values, call update_state. If update_state returns
        validation errors, relay them to the user and ask them to correct the details.
        Do not proceed to planning until update_state confirms the details are saved.

        STEP 3 — Run the specialists.
        Call search_flights, search_venues, and suggest_playlist.

        STEP 4 — Compile the plan.
        Format the final answer with these sections in order:
          SUMMARY    — one sentence with destination, guest count, and genre
          ✈ FLIGHTS  — shortlisted flight options
          🏛 VENUES  — shortlisted venue options
          🎵 PLAYLIST — curated track list with total duration and cost
        If a section shows "Service Unavailable", include it with the sample data
        and note that the user should verify with providers directly.
        """,
    )

    return coordinator

# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    coordinator = await build_agents()

    # Each session gets its own thread_id so planning runs stay separate.
    # Using a fixed ID here keeps the conversation continuous across turns.
    config = {
        "tags": ["wedding-planner"],
        "recursion_limit": 40,  # multi-agent chains can be deep
        "configurable": {"thread_id": "wedding-session-1"},
    }

    print("Destination Wedding Planner  |  type 'quit' to exit")
    print("Required: origin city, destination city, guest count, music genre")
    print("Example : I'm from London, wedding in Paris for 100 guests, jazz genre")
    print("-" * 70)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        print("\n[Working on your wedding plan — this may take 1–2 minutes...]\n")

        response = await coordinator.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        # If the coordinator asked a follow-up question, surface it clearly
        missing = response.get("missing_fields") or []
        val_errors = response.get("validation_errors") or []

        if val_errors:
            print("Coordinator: Please correct the following before we continue:")
            for err in val_errors:
                print(f"  • {err}")
        elif missing:
            # The last message is the follow-up question — print it as-is
            print(f"Coordinator: {response['messages'][-1].content}")
        else:
            print(f"Coordinator:\n{response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
