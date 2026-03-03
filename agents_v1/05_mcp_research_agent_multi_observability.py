#!/usr/bin/env python3
"""
MCP Research Agent з потрійною observability інтеграцією.

Покроковий аналіз теми (MCP Sequential Thinking) + реальний веб-пошук (DuckDuckGo),
з трейсингом у LangSmith, Phoenix Arize та LangFuse одночасно.

Запуск:
    python 05_mcp_research_agent_multi_observability.py --observability langsmith phoenix langfuse
    python 05_mcp_research_agent_multi_observability.py --topic "RAG optimization" --steps 3
"""

import os
import sys
import argparse
import warnings
from typing import Dict, List, Any

# duckduckgo_search перейменували в ddgs, але ddgs зависає на macOS — тримаємо старий пакет
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*renamed.*ddgs.*")

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MCP Research Agent with configurable observability",
    )
    parser.add_argument(
        "--observability", nargs="+",
        choices=["langsmith", "phoenix", "langfuse"],
        default=["langsmith"],
        help="observability platforms to enable (default: langsmith)",
    )
    parser.add_argument(
        "--topic",
        default="LangChain agent patterns for production systems",
        help="research topic",
    )
    parser.add_argument(
        "--steps", type=int, default=5, choices=range(1, 6),
        help="MCP thinking steps (1-5, default: 5)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Observability — кожна платформа ініціалізується незалежно
# ---------------------------------------------------------------------------

def setup_observability(platforms: List[str]) -> tuple[Dict[str, bool], Any]:
    """Повертає (status_dict, langfuse_handler)."""
    status = {"langsmith": False, "phoenix": False, "langfuse": False}
    langfuse_handler = None

    print("\n--- Observability ---")

    if "langsmith" in platforms:
        if not os.getenv("LANGCHAIN_API_KEY"):
            sys.exit("Error: LANGCHAIN_API_KEY not set")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", "mcp-research-agent")
        status["langsmith"] = True
        print(f"  LangSmith:  ON  (project: {os.environ['LANGCHAIN_PROJECT']})")

    if "phoenix" in platforms:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
        project = os.getenv("PHOENIX_PROJECT_NAME", "mcp-research-agent")
        tp = register(project_name=project, endpoint=endpoint)
        LangChainInstrumentor().instrument(tracer_provider=tp)
        status["phoenix"] = True
        print(f"  Phoenix:    ON  (ui: http://localhost:6006)")

    if "langfuse" in platforms:
        from langfuse.langchain import CallbackHandler

        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            sys.exit("Error: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
        # LangFuse 3.x підтримує LANGFUSE_HOST і LANGFUSE_BASE_URL, але пріоритет у HOST
        host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        if not os.getenv("LANGFUSE_HOST") and os.getenv("LANGFUSE_BASE_URL"):
            os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL")
        langfuse_handler = CallbackHandler()
        status["langfuse"] = True
        print(f"  LangFuse:   ON  ({host})")

    inactive = [k for k, v in status.items() if not v]
    if inactive:
        print(f"  Inactive:   {', '.join(inactive)}")
    print()

    return status, langfuse_handler


# ---------------------------------------------------------------------------
# LangChain imports (після observability, щоб Phoenix встиг проінструментувати)
# ---------------------------------------------------------------------------

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


# ---------------------------------------------------------------------------
# Інструменти — реальний веб-пошук через DuckDuckGo
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for articles, papers and documentation on a topic."""
    from duckduckgo_search import DDGS

    results = DDGS().text(query, max_results=5)
    if not results:
        return f"No results for: {query}"

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['href']}")
        lines.append(f"   {r['body']}")
        lines.append("")
    return "\n".join(lines)


@tool
def news_search(query: str) -> str:
    """Search recent news articles on a topic. Good for trends and current events."""
    from duckduckgo_search import DDGS

    results = DDGS().news(query, max_results=5)
    if not results:
        return f"No news for: {query}"

    lines = []
    for i, r in enumerate(results, 1):
        source = r.get("source", "")
        date = r.get("date", "")
        url = r.get("url", r.get("href", ""))
        lines.append(f"{i}. [{source}] {r['title']}")
        if date:
            lines.append(f"   {date}")
        lines.append(f"   {url}")
        lines.append(f"   {r['body']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP Sequential Thinking — покроковий аналіз теми через LLM
#
# В продакшені підключається до MCP сервера (@anthropic/mcp-sequential-thinking).
# Тут — локальна імплементація з тим самим підходом: кожен крок бачить попередні.
# ---------------------------------------------------------------------------

class MCPSequentialThinking:

    FOCUS_AREAS = [
        "Key concepts and definitions",
        "Main themes and patterns",
        "Critical analysis — contradictions and trade-offs",
        "Synthesis of insights",
        "Conclusions and practical implications",
    ]

    STEP_PROMPT = (
        "You are analyzing: {topic}\n\n"
        "Step {step}/{total}. Focus: {focus}\n\n"
        "Previous thoughts:\n{previous}\n\n"
        "Provide concise analysis for this step (2-3 sentences):"
    )

    SYNTHESIS_PROMPT = (
        "Based on these sequential thoughts about '{topic}':\n\n"
        "{numbered_thoughts}\n\n"
        "Provide a final synthesis (3-4 sentences):"
    )

    def __init__(self, llm):
        self.llm = llm

    def run(self, topic: str, num_steps: int = 5) -> Dict[str, Any]:
        thoughts = []

        print(f"--- MCP Sequential Thinking ({num_steps} steps) ---\n")

        for i, focus in enumerate(self.FOCUS_AREAS[:num_steps], 1):
            previous = "\n".join(
                f"  Step {j}: {t}" for j, t in enumerate(thoughts, 1)
            ) or "(none)"

            prompt = self.STEP_PROMPT.format(
                topic=topic, step=i, total=num_steps,
                focus=focus, previous=previous,
            )
            response = self.llm.invoke(prompt)
            thoughts.append(response.content)

            print(f"  [{i}/{num_steps}] {focus}")
            print(f"  {response.content}\n")

        # Фінальний синтез на основі всіх кроків
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(thoughts, 1))
        synthesis = self.llm.invoke(
            self.SYNTHESIS_PROMPT.format(topic=topic, numbered_thoughts=numbered)
        ).content

        print(f"  Synthesis: {synthesis}\n")

        return {"topic": topic, "thoughts": thoughts, "synthesis": synthesis}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a research analyst with access to real web search.

Workflow:
1. Use web_search to find articles, papers and docs on the topic
2. Use news_search to find recent news and trends
3. Analyze and synthesize everything into a structured report

Always cite sources with URLs. Be specific, avoid generic statements."""


def create_research_agent(langfuse_handler=None):
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0.7)

    agent = create_agent(
        model=model,
        tools=[web_search, news_search],
        system_prompt=SYSTEM_PROMPT,
    )

    if langfuse_handler:
        agent = agent.with_config({"callbacks": [langfuse_handler]})

    return agent, llm


# ---------------------------------------------------------------------------
# Pipeline: MCP thinking -> agent research
# ---------------------------------------------------------------------------

def run_research(topic: str, num_steps: int, obs_status: Dict, langfuse_handler):
    agent, llm = create_research_agent(langfuse_handler)

    # Фаза 1: структурований аналіз — агент "думає" покроково перед пошуком
    mcp = MCPSequentialThinking(llm)
    thinking = mcp.run(topic, num_steps=num_steps)

    # Фаза 2: агент шукає реальну інформацію, маючи контекст з фази 1
    print("--- Research Agent ---\n")

    query = (
        f"Research topic: {topic}\n\n"
        f"Prior analysis context:\n{thinking['synthesis']}\n\n"
        "Search for real articles, news and data. Provide a structured report with sources."
    )
    result = agent.invoke({"messages": [("user", query)]})
    print(result["messages"][-1].content)

    # Лінки на дашборди де можна подивитись трейси
    print("\n--- Traces ---")
    if obs_status["langsmith"]:
        print("  LangSmith: https://smith.langchain.com")
    if obs_status["phoenix"]:
        print("  Phoenix:   http://localhost:6006")
    if obs_status["langfuse"]:
        print(f"  LangFuse:  {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
    print()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY not set")

    try:
        status, lf_handler = setup_observability(args.observability)
        run_research(args.topic, args.steps, status, lf_handler)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
