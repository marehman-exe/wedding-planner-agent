"""
Destination Wedding Planner Agent
----------------------------------
A multi-agent system that plans your destination wedding end-to-end.
Tell it your origin, destination, guest count, and music genre in one message —
the coordinator gathers the details and delegates to three specialist sub-agents:

  ✈  Travel Agent    — finds real flights via the Kiwi Travel MCP server
  🏛  Venue Agent     — searches the web for venues matching your location & capacity
  🎵  Playlist Agent  — queries a music database and curates a genre-matching playlist

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

# ── Shared state ──────────────────────────────────────────────────────────────
# WeddingState extends AgentState so the coordinator and sub-agents all share
# the same four wedding fields — persisted per thread_id by InMemorySaver.

class WeddingState(AgentState):
    origin: str         # e.g. "London"
    destination: str    # e.g. "Paris"
    guest_count: str    # e.g. "100"
    genre: str          # e.g. "jazz"

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

    @tool
    def update_state(
        origin: str,
        destination: str,
        guest_count: str,
        genre: str,
        runtime: ToolRuntime,
    ) -> Command:
        """Save the wedding details (origin, destination, guest_count, genre) to shared state.
        Call this FIRST, before delegating to any sub-agent. Call it alone — no parallel calls."""
        return Command(
            update={
                "origin": origin,
                "destination": destination,
                "guest_count": guest_count,
                "genre": genre,
                "messages": [ToolMessage(
                    f"State updated: {origin} → {destination}, "
                    f"{guest_count} guests, {genre} music",
                    tool_call_id=runtime.tool_call_id,
                )],
            }
        )

    @tool
    async def search_flights(runtime: ToolRuntime) -> str:
        """Delegate to the travel sub-agent to find flights using the wedding origin and destination."""
        origin = runtime.state["origin"]
        destination = runtime.state["destination"]
        response = await travel_agent.ainvoke({
            "messages": [HumanMessage(content=f"Find flights from {origin} to {destination}")]
        })
        return response["messages"][-1].content

    @tool
    def search_venues(runtime: ToolRuntime) -> str:
        """Delegate to the venue sub-agent to find venues at the destination for the guest count."""
        destination = runtime.state["destination"]
        guest_count = runtime.state["guest_count"]
        response = venue_agent.invoke({
            "messages": [HumanMessage(
                content=f"Find wedding venues in {destination} for {guest_count} guests"
            )]
        })
        return response["messages"][-1].content

    @tool
    def suggest_playlist(runtime: ToolRuntime) -> str:
        """Delegate to the playlist sub-agent to curate a playlist for the wedding genre."""
        genre = runtime.state["genre"]
        response = playlist_agent.invoke({
            "messages": [HumanMessage(content=f"Find {genre} tracks for a wedding playlist")]
        })
        return response["messages"][-1].content

    # ── Coordinator ───────────────────────────────────────────────────────────

    coordinator = create_agent(
        model="groq:openai/gpt-oss-120b",
        tools=[update_state, search_flights, search_venues, suggest_playlist],
        state_schema=WeddingState,
        system_prompt="""
        You are a wedding coordinator managing a dream team of specialists.

        Your workflow — follow this order strictly:
        1. Extract origin, destination, guest_count, and genre from the user's message.
        2. Call update_state with all four values. Wait for it to complete before proceeding.
        3. Call search_flights, search_venues, and suggest_playlist (can run after state is set).
        4. Compile all results into a clear, well-formatted wedding plan for the user.

        Do NOT ask follow-up questions. Work with the information provided.
        """,
    )

    return coordinator

# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    coordinator = await build_agents()
    config = {
        "tags": ["wedding-planner"],
        "recursion_limit": 40,  # multi-agent chains can be deep
    }

    print("Destination Wedding Planner  |  type 'quit' to exit\n")
    print("Example: I'm from London and I'd like a wedding in Paris for 100 guests, jazz genre")
    print("-" * 70)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        print("\n[Planning your wedding — this may take 1–2 minutes...]\n")

        response = await coordinator.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        print(f"Coordinator:\n{response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
