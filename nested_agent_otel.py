"""
This is a demo of tracing a nested agent with OpenTelemetry, including
- token usages tracking
- tool calls tracing
- llm calls tracing

to use this, you need to docker compose up the otel stack first:
cd docker && docker-compose up -d

then you can run the script:
uv run python nested_agent_otel.py

you can view the traces at: http://localhost:16686 (Jaeger)
you can view the metrics at: http://localhost:3000 (Grafana)
# dashboard is already there.

"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# Import OpenTelemetry middleware
from otel_middleware import setup_otel_tracing, OtelMiddleware


# ============================================================
# Define Tools
# ============================================================


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        A string describing the current weather.
    """
    # Simulated weather data
    weather_data = {
        "new york": "Sunny, 72°F (22°C), humidity 45%",
        "london": "Cloudy, 59°F (15°C), humidity 78%",
        "tokyo": "Rainy, 68°F (20°C), humidity 85%",
        "paris": "Partly cloudy, 64°F (18°C), humidity 62%",
        "sydney": "Clear, 77°F (25°C), humidity 55%",
    }
    city_lower = city.lower()
    return weather_data.get(
        city_lower,
        f"Weather data not available for {city}. Available cities: {', '.join(weather_data.keys())}",
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2 * 3").

    Returns:
        The result of the calculation.
    """
    try:
        # Safe evaluation of mathematical expressions
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters. Only numbers and basic operators (+, -, *, /, ()) are allowed."

        result = eval(expression)  # Note: In production, use a safer evaluator
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def search_knowledge(query: str) -> str:
    """Search the internal knowledge base for information.

    Args:
        query: The search query.

    Returns:
        Relevant information from the knowledge base.
    """
    # Simulated knowledge base
    knowledge = {
        "python": "Python is a high-level, interpreted programming language known for its readability and versatility.",
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "opentelemetry": "OpenTelemetry is an observability framework for creating and collecting telemetry data.",
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value

    return f"No specific information found for '{query}'. Try searching for: {', '.join(knowledge.keys())}"


@tool
def get_time(timezone: str = "UTC") -> str:
    """Get the current time in a specific timezone.

    Args:
        timezone: The timezone to get time for (e.g., "UTC", "EST", "PST").

    Returns:
        The current time in the specified timezone.
    """
    from datetime import datetime, timedelta

    # Simplified timezone handling
    offsets = {
        "utc": 0,
        "est": -5,
        "pst": -8,
        "cet": 1,
        "jst": 9,
    }

    tz_lower = timezone.lower()
    offset = offsets.get(tz_lower, 0)

    now = datetime.utcnow() + timedelta(hours=offset)
    return f"Current time in {timezone.upper()}: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# ============================================================
# Create Subagent Tools (Agents as Tools)
# ============================================================


def create_weather_agent_tool(model: str, otel_middleware: OtelMiddleware):
    """Create a weather subagent wrapped as a tool."""

    # Create the subagent
    weather_agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="""You are a weather assistant. Your job is to:
1. Help users get weather information for cities
2. Provide helpful context about the weather conditions
3. Suggest activities based on the weather

Always use the get_weather tool to fetch accurate weather data.
Be concise in your responses.""",
        middleware=[otel_middleware],
        name="weather_agent",
    )

    @tool
    def ask_weather_agent(query: str) -> str:
        """Ask the weather assistant about weather conditions in any city.

        Use this tool when the user asks about weather, temperature, or climate.

        Args:
            query: The weather-related question to ask.

        Returns:
            The weather assistant's response.
        """
        result = weather_agent.invoke({"messages": [HumanMessage(content=query)]})
        # Extract the last AI message content
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if (
                hasattr(msg, "content")
                and msg.content
                and not hasattr(msg, "tool_calls")
            ):
                return msg.content
        return "No response from weather agent."

    return ask_weather_agent


def create_research_agent_tool(model: str, otel_middleware: OtelMiddleware):
    """Create a research subagent wrapped as a tool."""

    research_agent = create_agent(
        model=model,
        tools=[search_knowledge],
        system_prompt="""You are a research assistant. Your job is to:
1. Search the knowledge base for relevant information
2. Provide clear and accurate answers
3. Explain technical concepts in simple terms

Use the search_knowledge tool to find information.
Be concise and informative.""",
        middleware=[otel_middleware],
        name="research_agent",
    )

    @tool
    def ask_research_agent(query: str) -> str:
        """Ask the research assistant about programming, frameworks, or technical topics.

        Use this tool when the user asks about technical concepts, programming languages,
        or frameworks like Python, LangChain, LangGraph, or OpenTelemetry.

        Args:
            query: The research question to ask.

        Returns:
            The research assistant's response.
        """
        result = research_agent.invoke({"messages": [HumanMessage(content=query)]})
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if (
                hasattr(msg, "content")
                and msg.content
                and not hasattr(msg, "tool_calls")
            ):
                return msg.content
        return "No response from research agent."

    return ask_research_agent


def create_utility_agent_tool(model: str, otel_middleware: OtelMiddleware):
    """Create a utility subagent wrapped as a tool."""

    utility_agent = create_agent(
        model=model,
        tools=[calculate, get_time],
        system_prompt="""You are a utility assistant. Your job is to:
1. Perform mathematical calculations using the calculate tool
2. Provide current time in different timezones using the get_time tool
3. Help with numerical and time-related tasks

Be precise and helpful.""",
        middleware=[otel_middleware],
        name="utility_agent",
    )

    @tool
    def ask_utility_agent(query: str) -> str:
        """Ask the utility assistant for calculations or time queries.

        Use this tool when the user needs mathematical calculations or wants to know
        the current time in a specific timezone.

        Args:
            query: The utility-related question (math calculation or time query).

        Returns:
            The utility assistant's response.
        """
        result = utility_agent.invoke({"messages": [HumanMessage(content=query)]})
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if (
                hasattr(msg, "content")
                and msg.content
                and not hasattr(msg, "tool_calls")
            ):
                return msg.content
        return "No response from utility agent."

    return ask_utility_agent


# ============================================================
# Create Main Orchestrating Agent
# ============================================================


def create_main_agent(model: str = "gpt-4o-mini"):
    """Create the main orchestrating agent with subagent tools.

    Uses a SINGLE shared OtelMiddleware instance for all agents (main and sub-agents).
    This ensures proper context propagation and creates a unified trace hierarchy
    where all operations are nested under a root span.
    """

    # Create a SINGLE shared OTel middleware for ALL agents
    # This enables automatic context propagation between main agent and sub-agents
    shared_middleware = OtelMiddleware(
        trace_content=True,
        service_name="langchain-agents",  # Unified service name
    )

    # Create subagent tools - all using the SAME middleware instance
    weather_tool = create_weather_agent_tool(model, shared_middleware)
    research_tool = create_research_agent_tool(model, shared_middleware)
    utility_tool = create_utility_agent_tool(model, shared_middleware)

    # Create the main agent with subagent tools - also using the SAME middleware
    main_agent = create_agent(
        model=model,
        tools=[weather_tool, research_tool, utility_tool],
        system_prompt="""You are a helpful personal assistant that coordinates with specialized sub-agents.

Your capabilities include:
- Weather information: Use ask_weather_agent for weather-related questions
- Research and knowledge: Use ask_research_agent for technical/knowledge questions
- Calculations and time: Use ask_utility_agent for math or time queries

When a user asks a question:
1. Determine which subagent tool is best suited to handle the request
2. Call the appropriate tool to delegate the task
3. Present the response to the user

If a query spans multiple domains, you can call multiple tools.
Always be helpful and provide complete answers.""",
        middleware=[shared_middleware],
        name="main_orchestrator",
    )

    return main_agent, shared_middleware


# ============================================================
# Main Execution
# ============================================================


if __name__ == "__main__":
    # Initialize OpenTelemetry tracing
    # Connect to OTEL collector running in Docker (port 4319 maps to internal 4317)
    setup_otel_tracing(
        service_name="nested-agent-demo",
        otlp_endpoint="localhost:4319",
        enable_console_export=False,  # Set to True for debugging
    )

    print("=" * 80)
    print("Nested Agent with OpenTelemetry Tracing Demo")
    print("=" * 80)
    print("\nThis demo shows:")
    print("  - Main agent that orchestrates subagents")
    print("  - Weather Agent (get_weather tool)")
    print("  - Research Agent (search_knowledge tool)")
    print("  - Utility Agent (calculate, get_time tools)")
    print("  - All traced with OtelMiddleware")
    print("\nMake sure the OTEL stack is running:")
    print("  cd docker && docker-compose up -d")
    print("  View traces at: http://localhost:16686 (Jaeger)")
    print("=" * 80)

    # Create the main agent with shared middleware
    agent, middleware = create_main_agent()

    # Example queries that will trigger different subagents
    queries = [
        "What's the weather like in Tokyo?",
        "Calculate 15 * 7 + 23",
        "What is LangGraph?",
        "What time is it in PST?",
    ]

    print("\n" + "=" * 80)
    print("Running example queries...")
    print("=" * 80)

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 60)

        try:
            # Use middleware's invoke_traced method for automatic root span creation
            # This ensures all operations are nested under a single trace
            result = middleware.invoke_traced(
                agent,
                {"messages": [("user", query)]},
                agent_name="main_orchestrator",
            )

            # Extract and print the final response
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    # Skip tool messages
                    if hasattr(msg, "tool_call_id"):
                        continue
                    # Skip messages with tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        continue
                    print(f"Response: {msg.content}")
                    break
        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

        print("-" * 60)

    print("\n" + "=" * 80)
    print("Demo complete! Check Jaeger for traces: http://localhost:16686")
    print("=" * 80)
