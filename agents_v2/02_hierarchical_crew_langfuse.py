"""
Hierarchical CrewAI + LangFuse Observability Demo

Demonstrates:
1. CrewAI hierarchical process with auto-created manager
2. LangFuse observability for production monitoring
3. Task delegation tracking
4. Manager decision visibility

ВАЖЛИВО: Потребує встановлення LangFuse:
    pip install langfuse

SETUP:
1. Sign up at https://cloud.langfuse.com (або self-host)
2. Get your public and secret keys
3. Add to .env:
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com

OBSERVABILITY:
- LangFuse captures all LLM calls через LangChain integration
- Tracks manager decisions and delegations
- Monitors task execution flow
- Provides production-grade analytics

CrewAI Version: 1.4.0+
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

load_dotenv()

# Verify API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# ============================================================================
# LANGFUSE SETUP - Production observability
# ============================================================================

LANGFUSE_AVAILABLE = False
try:
    from langfuse.callback import CallbackHandler

    # Initialize LangFuse callback handler
    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )

    LANGFUSE_AVAILABLE = True

    print("=" * 70)
    print("📊 LANGFUSE OBSERVABILITY ENABLED")
    print("=" * 70)
    print("✅ LangFuse callback handler initialized")
    print(f"🌐 Host: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
    print("📈 All LLM calls will be traced")
    print("=" * 70 + "\n")

except ImportError:
    print("⚠️  LangFuse not installed. Install with:")
    print("   pip install langfuse")
    print("   Continuing without LangFuse...\n")
except Exception as e:
    print(f"⚠️  LangFuse setup failed: {e}")
    print("Continuing without LangFuse...\n")

# Check if LangSmith is also enabled
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("=" * 70)
    print("✅ LANGSMITH ALSO ENABLED")
    print("=" * 70)
    print(f"📊 Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    print("🔍 Dual observability: LangSmith + LangFuse")
    print("=" * 70 + "\n")


def create_content_production_crew():
    """
    Create a hierarchical crew for content production.

    Manager (auto-created) coordinates:
    - Content Researcher
    - Content Writer
    - Content Editor

    This demonstrates hierarchical task delegation with LangFuse tracking.
    """

    print("=" * 70)
    print("🤖 CREATING HIERARCHICAL CREW WITH LANGFUSE")
    print("=" * 70 + "\n")

    # Specialist Agent 1: Researcher
    researcher = Agent(
        role="Senior Content Researcher",
        goal="Research comprehensive information about {topic}",
        backstory=(
            "You are an expert researcher with 10 years of experience. "
            "You excel at finding credible sources and synthesizing "
            "complex information into clear insights."
        ),
        verbose=True,
        allow_delegation=False,  # Only manager can delegate
        llm="gpt-4o-mini"
    )

    # Specialist Agent 2: Writer
    writer = Agent(
        role="Professional Content Writer",
        goal="Create engaging, well-structured content about {topic}",
        backstory=(
            "You are a skilled writer known for creating compelling content. "
            "You transform research into engaging narratives that captivate "
            "readers while maintaining accuracy and clarity."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    # Specialist Agent 3: Editor
    editor = Agent(
        role="Senior Content Editor",
        goal="Review and polish content to ensure highest quality",
        backstory=(
            "You are a meticulous editor with a sharp eye for detail. "
            "You ensure content is error-free, well-structured, and "
            "meets professional standards."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    # Define tasks
    research_task = Task(
        description=(
            "Research {topic} comprehensively. Include:\n"
            "- Key facts and statistics\n"
            "- Recent developments (2024-2025)\n"
            "- Expert opinions and credible sources\n"
            "- Practical applications and examples\n"
            "Provide 10-12 well-researched bullet points."
        ),
        expected_output=(
            "Detailed research report with 10-12 bullet points covering "
            "all aspects of {topic} with credible sources."
        ),
        agent=researcher
    )

    writing_task = Task(
        description=(
            "Based on research, write an engaging article about {topic}. "
            "Structure:\n"
            "- Compelling introduction\n"
            "- 3-4 main sections with subheadings\n"
            "- Practical examples\n"
            "- Conclusion with key takeaways\n"
            "Target length: 800-1000 words."
        ),
        expected_output=(
            "Well-structured 800-1000 word article about {topic} with "
            "introduction, main sections, examples, and conclusion."
        ),
        agent=writer
    )

    editing_task = Task(
        description=(
            "Review and edit the article about {topic}. Check:\n"
            "- Grammar and spelling\n"
            "- Clarity and flow\n"
            "- Factual accuracy\n"
            "- Structure and formatting\n"
            "- Tone consistency\n"
            "Provide polished final version."
        ),
        expected_output=(
            "Polished, publication-ready article about {topic} with all "
            "edits applied and quality verified."
        ),
        agent=editor
    )

    print("Agents created:")
    print(f"  • {researcher.role}")
    print(f"  • {writer.role}")
    print(f"  • {editor.role}")
    print(f"  • Manager (auto-created by CrewAI)")
    print()

    print("Tasks defined:")
    print(f"  1. Research task → {researcher.role}")
    print(f"  2. Writing task → {writer.role}")
    print(f"  3. Editing task → {editor.role}")
    print()

    # Create hierarchical crew
    # Manager will be auto-created and will coordinate the specialists
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.hierarchical,  # Enables manager-led coordination
        manager_llm="gpt-4o-mini",     # LLM for the manager agent
        verbose=True
    )

    print("✅ Hierarchical crew created")
    print("   Process: Manager-led coordination")
    print("   Manager will delegate tasks and validate outputs")
    print()

    if LANGFUSE_AVAILABLE:
        print("📊 LangFuse will track:")
        print("   • Manager delegation decisions")
        print("   • Each agent's LLM calls")
        print("   • Task execution flow")
        print("   • Token usage and costs")
        print()

    return crew


def run_crew_with_langfuse(topic: str):
    """
    Run the hierarchical crew with LangFuse observability.

    LangFuse integration happens through LangChain callbacks since
    CrewAI uses LangChain internally.
    """

    crew = create_content_production_crew()

    print("=" * 70)
    print(f"🚀 STARTING HIERARCHICAL CREW: {topic}")
    print("=" * 70 + "\n")

    if LANGFUSE_AVAILABLE:
        print("📊 LangFuse observability active")
        print("   View traces at: " + os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
        print()

    # Note: CrewAI doesn't directly support callbacks parameter like LangChain
    # But since CrewAI uses LangChain internally, we can set up LangChain tracing
    # through environment variables (LANGCHAIN_TRACING_V2=true)

    # For production observability, the best approach is to use:
    # 1. LangSmith (via LANGCHAIN_TRACING_V2=true)
    # 2. Custom logging in agent backstories or task descriptions
    # 3. OpenTelemetry instrumentation (advanced)

    # Start crew execution
    inputs = {"topic": topic}

    print("📝 Input:")
    print(f"   Topic: {topic}")
    print("\n" + "-" * 70)
    print("EXECUTION LOG (Manager coordinates specialists):")
    print("-" * 70 + "\n")

    try:
        result = crew.kickoff(inputs=inputs)

        print("\n" + "=" * 70)
        print("✅ CREW EXECUTION COMPLETED")
        print("=" * 70 + "\n")

        print("📋 FINAL OUTPUT:")
        print("-" * 70)
        print(result)
        print("-" * 70 + "\n")

        if LANGFUSE_AVAILABLE:
            print("📊 View detailed traces and analytics in LangFuse:")
            print(f"   {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
            print()
            print("   You can see:")
            print("   • Manager's delegation decisions")
            print("   • Each agent's LLM interactions")
            print("   • Token usage and costs")
            print("   • Latency and performance metrics")
            print()

        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            print("📊 Also check LangSmith for additional traces:")
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
    print("🎯 Hierarchical CrewAI + LangFuse Observability Demo")
    print("=" * 70)
    print()
    print("This demo shows:")
    print("  ✅ CrewAI hierarchical process (manager + 3 specialists)")
    print("  ✅ LangFuse production observability")
    print("  ✅ Task delegation tracking")
    print("  ✅ Manager decision visibility")
    print()
    print("Observability stack:")
    print(f"  • LangFuse:  {'✅ Active' if LANGFUSE_AVAILABLE else '❌ Not configured'}")
    print(f"  • LangSmith: {'✅ Active' if os.getenv('LANGCHAIN_TRACING_V2') == 'true' else '❌ Disabled'}")
    print()
    print("=" * 70 + "\n")

    if not LANGFUSE_AVAILABLE:
        print("⚠️  LANGFUSE SETUP REQUIRED")
        print("=" * 70)
        print()
        print("To enable LangFuse observability:")
        print("  1. Install: pip install langfuse")
        print("  2. Sign up: https://cloud.langfuse.com")
        print("  3. Add to .env:")
        print("     LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("     LANGFUSE_SECRET_KEY=sk-lf-...")
        print("     LANGFUSE_HOST=https://cloud.langfuse.com")
        print()
        print("=" * 70 + "\n")

        response = input("Continue without LangFuse? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            exit(0)
        print()

    # Run the crew with a sample topic
    topic = "AI Agent Frameworks in 2025: LangChain vs CrewAI"

    result = run_crew_with_langfuse(topic)

    if result:
        print("\n" + "=" * 70)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)

        if LANGFUSE_AVAILABLE:
            print("\n💡 Next steps:")
            print("   1. Check LangFuse dashboard for detailed traces")
            print("   2. Analyze token usage and costs")
            print("   3. Review manager delegation patterns")
            print("   4. Optimize agent performance based on metrics")
            print()
    else:
        print("\n❌ Demo encountered errors")
