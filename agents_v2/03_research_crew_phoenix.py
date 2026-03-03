"""
Research CrewAI + Phoenix Arize Observability Demo

Demonstrates:
1. CrewAI sequential process with LangChain tools
2. Phoenix Arize observability (excellent for tool tracking)
3. RAG-like workflow visualization
4. Tool execution monitoring

ВАЖЛИВО: Потребує встановлення Phoenix:
    pip install arize-phoenix openinference-instrumentation-crewai

SETUP:
1. Terminal 1: python -m phoenix.server.main serve
2. Terminal 2: python 03_research_crew_phoenix.py
3. Open: http://localhost:6006

PHOENIX BENEFITS FOR CREWAI:
- Visualizes tool execution flow
- Tracks agent interactions
- Monitors RAG-like patterns
- Local deployment (no cloud required)
- Excellent for development/debugging

CrewAI Version: 1.4.0+
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain.tools import tool

load_dotenv()

# Verify API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# ============================================================================
# PHOENIX ARIZE SETUP - Open-source observability
# ============================================================================

PHOENIX_AVAILABLE = False
try:
    from phoenix.otel import register
    from openinference.instrumentation.crewai import CrewAIInstrumentor

    # Register Phoenix tracer
    tracer_provider = register(
        project_name="crewai-research-demo",
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"),
    )

    # Instrument CrewAI
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

    PHOENIX_AVAILABLE = True

    print("=" * 70)
    print("🔥 PHOENIX ARIZE OBSERVABILITY ENABLED")
    print("=" * 70)
    print("✅ Phoenix instrumentation active for CrewAI")
    print(f"📊 Project: crewai-research-demo")
    print(f"🌐 Endpoint: {os.getenv('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:6006/v1/traces')}")
    print("🖥️  UI: http://localhost:6006")
    print()
    print("⚠️  Make sure Phoenix server is running:")
    print("   Terminal 1: python -m phoenix.server.main serve")
    print("=" * 70 + "\n")

except ImportError:
    print("⚠️  Phoenix Arize not installed. Install with:")
    print("   pip install arize-phoenix openinference-instrumentation-crewai")
    print("   Continuing without Phoenix...\n")
except Exception as e:
    print(f"⚠️  Phoenix setup failed: {e}")
    print("Continuing without Phoenix...\n")

# Check if LangSmith is also enabled
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("=" * 70)
    print("✅ LANGSMITH ALSO ENABLED")
    print("=" * 70)
    print(f"📊 Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    print("🔍 Dual observability: LangSmith + Phoenix")
    print("=" * 70 + "\n")


# ============================================================================
# CUSTOM TOOLS - Will be tracked by Phoenix
# ============================================================================

@tool
def search_documentation(query: str) -> str:
    """
    Search through documentation for relevant information.

    Args:
        query: Search query

    Returns:
        Relevant documentation excerpts
    """
    # Simulated documentation search
    docs = {
        "langchain": "LangChain 1.0 is a framework for developing applications powered by language models. Key features: create_agent API, StateGraph, callbacks, and production-ready patterns.",
        "crewai": "CrewAI is a framework for orchestrating role-playing autonomous AI agents. Key features: hierarchical process, sequential process, task delegation, and collaborative workflows.",
        "phoenix": "Phoenix Arize is an open-source observability platform for LLM applications. Key features: trace visualization, embedding analysis, RAG monitoring, and local deployment.",
        "langfuse": "LangFuse is an open-source LLM engineering platform. Key features: prompt management, production monitoring, analytics, and both cloud and self-hosted options."
    }

    results = []
    query_lower = query.lower()

    for topic, content in docs.items():
        if topic in query_lower or any(word in content.lower() for word in query_lower.split()):
            results.append(f"[{topic.upper()}] {content}")

    if not results:
        return "No relevant documentation found. Try broader search terms."

    return "\n\n".join(results)


@tool
def analyze_data(data: str) -> str:
    """
    Analyze data and extract key insights.

    Args:
        data: Data to analyze

    Returns:
        Analysis results with key insights
    """
    # Simulated data analysis
    word_count = len(data.split())
    has_langchain = "langchain" in data.lower()
    has_crewai = "crewai" in data.lower()
    has_observability = any(word in data.lower() for word in ["phoenix", "langfuse", "langsmith", "observability"])

    analysis = f"""Data Analysis Results:
- Word count: {word_count}
- Contains LangChain info: {'Yes' if has_langchain else 'No'}
- Contains CrewAI info: {'Yes' if has_crewai else 'No'}
- Contains observability info: {'Yes' if has_observability else 'No'}

Key Insight: {'This data covers multiple AI frameworks and observability tools.' if word_count > 20 else 'Limited data provided.'}
"""
    return analysis


@tool
def calculate_metrics(text: str) -> str:
    """
    Calculate various metrics from text.

    Args:
        text: Text to analyze

    Returns:
        Calculated metrics
    """
    sentences = text.split('.')
    words = text.split()

    metrics = f"""Text Metrics:
- Total words: {len(words)}
- Total sentences: {len(sentences)}
- Average words per sentence: {len(words) / max(len(sentences), 1):.1f}
- Complexity score: {'High' if len(words) / max(len(sentences), 1) > 15 else 'Medium' if len(words) / max(len(sentences), 1) > 10 else 'Low'}
"""
    return metrics


# ============================================================================
# CREATE RESEARCH CREW WITH TOOLS
# ============================================================================

def create_research_crew_with_tools():
    """
    Create a research crew with custom tools.

    Phoenix will track:
    - Tool invocations
    - Agent interactions
    - Data flow through the crew
    """

    print("=" * 70)
    print("🤖 CREATING RESEARCH CREW WITH PHOENIX OBSERVABILITY")
    print("=" * 70 + "\n")

    # Agent 1: Data Researcher (with search tool)
    researcher = Agent(
        role="Senior Data Researcher",
        goal="Research comprehensive information about {topic} using available tools",
        backstory=(
            "You are an expert researcher with access to documentation search tools. "
            "You know how to find relevant information and organize it effectively. "
            "You always use your search_documentation tool to find accurate information."
        ),
        tools=[search_documentation],
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    # Agent 2: Data Analyst (with analysis tools)
    analyst = Agent(
        role="Data Analyst",
        goal="Analyze research data and extract key insights about {topic}",
        backstory=(
            "You are a skilled analyst who processes research data to extract "
            "meaningful insights. You use analyze_data and calculate_metrics tools "
            "to provide data-driven conclusions."
        ),
        tools=[analyze_data, calculate_metrics],
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    # Agent 3: Report Writer (no tools, synthesizes results)
    writer = Agent(
        role="Technical Report Writer",
        goal="Compile research and analysis into a comprehensive report about {topic}",
        backstory=(
            "You are a technical writer who excels at creating clear, structured "
            "reports. You synthesize information from researchers and analysts into "
            "cohesive, professional documents."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    # Define tasks
    research_task = Task(
        description=(
            "Research {topic} using your documentation search tool. "
            "Find comprehensive information including:\n"
            "- Key features and capabilities\n"
            "- Use cases and applications\n"
            "- Comparison with alternatives\n"
            "- Best practices\n\n"
            "Use search_documentation tool to find relevant information."
        ),
        expected_output=(
            "Detailed research findings with 8-10 bullet points covering "
            "all aspects of {topic} based on documentation search."
        ),
        agent=researcher
    )

    analysis_task = Task(
        description=(
            "Analyze the research findings about {topic}. Use your tools to:\n"
            "- Analyze the data using analyze_data tool\n"
            "- Calculate metrics using calculate_metrics tool\n"
            "- Extract key insights and patterns\n"
            "- Identify important trends\n\n"
            "Provide data-driven analysis."
        ),
        expected_output=(
            "Comprehensive analysis of research findings with metrics, "
            "insights, and data-driven conclusions about {topic}."
        ),
        agent=analyst
    )

    writing_task = Task(
        description=(
            "Create a comprehensive report about {topic} based on research and analysis. "
            "Structure:\n"
            "- Executive Summary\n"
            "- Key Findings (from research)\n"
            "- Data Analysis (from analyst)\n"
            "- Conclusions and Recommendations\n\n"
            "Make it professional and well-structured."
        ),
        expected_output=(
            "Professional report about {topic} with executive summary, "
            "findings, analysis, and recommendations. 500-700 words."
        ),
        agent=writer
    )

    print("Agents created:")
    print(f"  • {researcher.role} (with search_documentation tool)")
    print(f"  • {analyst.role} (with analyze_data, calculate_metrics tools)")
    print(f"  • {writer.role} (synthesis, no tools)")
    print()

    print("Tasks defined:")
    print(f"  1. Research task → {researcher.role}")
    print(f"  2. Analysis task → {analyst.role}")
    print(f"  3. Writing task → {writer.role}")
    print()

    # Create sequential crew
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,  # Execute tasks in order
        verbose=True
    )

    print("✅ Sequential research crew created")
    print("   Process: Sequential execution (research → analysis → writing)")
    print()

    if PHOENIX_AVAILABLE:
        print("🔥 Phoenix will track:")
        print("   • Tool executions (search_documentation, analyze_data, calculate_metrics)")
        print("   • Agent LLM calls")
        print("   • Data flow between agents")
        print("   • Complete execution timeline")
        print()

    return crew


def run_research_crew_with_phoenix(topic: str):
    """
    Run the research crew with Phoenix observability.
    """

    crew = create_research_crew_with_tools()

    print("=" * 70)
    print(f"🚀 STARTING RESEARCH CREW: {topic}")
    print("=" * 70 + "\n")

    if PHOENIX_AVAILABLE:
        print("🔥 Phoenix observability active")
        print("   View real-time traces at: http://localhost:6006")
        print()

    # Start crew execution
    inputs = {"topic": topic}

    print("📝 Input:")
    print(f"   Topic: {topic}")
    print("\n" + "-" * 70)
    print("EXECUTION LOG (Sequential: Research → Analysis → Writing):")
    print("-" * 70 + "\n")

    try:
        result = crew.kickoff(inputs=inputs)

        print("\n" + "=" * 70)
        print("✅ CREW EXECUTION COMPLETED")
        print("=" * 70 + "\n")

        print("📋 FINAL REPORT:")
        print("-" * 70)
        print(result)
        print("-" * 70 + "\n")

        if PHOENIX_AVAILABLE:
            print("=" * 70)
            print("🔥 PHOENIX OBSERVABILITY - NEXT STEPS")
            print("=" * 70)
            print()
            print("Open Phoenix UI: http://localhost:6006")
            print()
            print("You can explore:")
            print("  • Traces tab: See complete execution flow")
            print("  • Tools: View all tool invocations")
            print("  • Agents: Track agent-specific LLM calls")
            print("  • Timeline: Understand execution sequence")
            print("  • Metrics: Analyze performance and costs")
            print()

        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            print("📊 Also check LangSmith:")
            print("   https://smith.langchain.com")
            print()

        return result

    except Exception as e:
        print(f"\n❌ ERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("🎯 Research CrewAI + Phoenix Arize Observability Demo")
    print("=" * 70)
    print()
    print("This demo shows:")
    print("  ✅ CrewAI sequential process with tools")
    print("  ✅ Phoenix Arize observability")
    print("  ✅ Tool execution tracking")
    print("  ✅ RAG-like workflow visualization")
    print()
    print("Observability stack:")
    print(f"  • Phoenix:   {'✅ Active' if PHOENIX_AVAILABLE else '❌ Not installed'}")
    print(f"  • LangSmith: {'✅ Active' if os.getenv('LANGCHAIN_TRACING_V2') == 'true' else '❌ Disabled'}")
    print()
    print("=" * 70 + "\n")

    if not PHOENIX_AVAILABLE:
        print("⚠️  PHOENIX SETUP REQUIRED")
        print("=" * 70)
        print()
        print("To enable Phoenix observability:")
        print("  1. Install: pip install arize-phoenix openinference-instrumentation-crewai")
        print("  2. Terminal 1: python -m phoenix.server.main serve")
        print("  3. Terminal 2: python 03_research_crew_phoenix.py")
        print("  4. Open: http://localhost:6006")
        print()
        print("=" * 70 + "\n")

        response = input("Continue without Phoenix? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            exit(0)
        print()
    else:
        print("=" * 70)
        print("⚠️  IMPORTANT: Make sure Phoenix server is running!")
        print("=" * 70)
        print()
        print("In a separate terminal, run:")
        print("  python -m phoenix.server.main serve")
        print()
        print("Then access Phoenix UI at: http://localhost:6006")
        print("=" * 70 + "\n")

        input("Press Enter when Phoenix server is ready...\n")

    # Run the crew with a sample topic
    topic = "Observability Platforms for AI Agents (Phoenix, LangFuse, LangSmith)"

    result = run_research_crew_with_phoenix(topic)

    if result:
        print("\n" + "=" * 70)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)

        if PHOENIX_AVAILABLE:
            print("\n💡 Next steps:")
            print("   1. Open Phoenix UI: http://localhost:6006")
            print("   2. Explore the Traces tab")
            print("   3. Click on individual spans to see details")
            print("   4. View tool executions and their inputs/outputs")
            print("   5. Analyze the complete workflow timeline")
            print()
    else:
        print("\n❌ Demo encountered errors")
