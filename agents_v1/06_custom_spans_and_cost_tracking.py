#!/usr/bin/env python3
"""
Custom OpenTelemetry Spans + Token/Cost Tracking

Демонструє:
1. Ручні OpenTelemetry spans поряд з auto-instrumented (Phoenix/LangSmith)
2. Реальний cost tracking через usage_metadata з AIMessage
3. Nested spans для кожної фази research pipeline (search → analysis → synthesis)

Запуск:
    python 06_custom_spans_and_cost_tracking.py --topic "AI agent observability"
    python 06_custom_spans_and_cost_tracking.py --observability phoenix langsmith
    python 06_custom_spans_and_cost_tracking.py --topic "RAG optimization" --observability phoenix
"""

import os
import sys
import argparse
import warnings
from typing import Dict, List, Any
from dataclasses import dataclass, field

# duckduckgo_search перейменували в ddgs, але ddgs зависає на macOS — тримаємо старий пакет
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*renamed.*ddgs.*")

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Research Agent with custom OpenTelemetry spans and cost tracking",
    )
    parser.add_argument(
        "--observability", nargs="+",
        choices=["langsmith", "phoenix"],
        default=["langsmith"],
        help="observability platforms to enable (default: langsmith)",
    )
    parser.add_argument(
        "--topic",
        default="AI agent observability best practices",
        help="research topic",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Observability — Phoenix + LangSmith + OpenTelemetry tracer
# ---------------------------------------------------------------------------

# OpenTelemetry доступний завжди (входить у Phoenix або standalone)
try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


def setup_observability(platforms: List[str]) -> Dict[str, bool]:
    """Ініціалізує платформи, повертає статус."""
    status = {"langsmith": False, "phoenix": False}

    print("\n--- Observability ---")

    if "langsmith" in platforms:
        if not os.getenv("LANGCHAIN_API_KEY"):
            sys.exit("Error: LANGCHAIN_API_KEY not set")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", "custom-spans-cost-tracking")
        status["langsmith"] = True
        print(f"  LangSmith:  ON  (project: {os.environ['LANGCHAIN_PROJECT']})")

    if "phoenix" in platforms:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
        project = os.getenv("PHOENIX_PROJECT_NAME", "custom-spans-cost-tracking")
        tp = register(project_name=project, endpoint=endpoint)
        LangChainInstrumentor().instrument(tracer_provider=tp)
        status["phoenix"] = True
        print(f"  Phoenix:    ON  (ui: http://localhost:6006)")

    inactive = [k for k, v in status.items() if not v]
    if inactive:
        print(f"  Inactive:   {', '.join(inactive)}")
    print()

    return status


# ---------------------------------------------------------------------------
# LangChain imports (після observability, щоб Phoenix проінструментував)
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
# CostTracker — витягує usage_metadata з AIMessage і рахує вартість
# ---------------------------------------------------------------------------

# Ціни за 1M токенів (USD) — актуальні на 2025
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


@dataclass
class PhaseUsage:
    """Використання токенів для однієї фази pipeline."""
    name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CostTracker:
    """
    Трекер вартості — витягує usage_metadata з AIMessage.

    usage_metadata — це dict з ключами:
      input_tokens, output_tokens, total_tokens
    Доступний в AIMessage, коли LangChain отримує відповідь від провайдера.
    """
    model: str = "gpt-4o-mini"
    phases: Dict[str, PhaseUsage] = field(default_factory=dict)

    def _get_pricing(self) -> Dict[str, float]:
        """Повертає pricing для поточної моделі."""
        # Шукаємо точний матч або найближчий
        for key, pricing in MODEL_PRICING.items():
            if key in self.model:
                return pricing
        # Fallback на gpt-4o-mini pricing
        return MODEL_PRICING["gpt-4o-mini"]

    def track_messages(self, phase_name: str, messages: list):
        """
        Витягує usage_metadata з AIMessage у списку повідомлень.

        Args:
            phase_name: назва фази (search, analysis, synthesis)
            messages: список повідомлень з result["messages"]
        """
        from langchain_core.messages import AIMessage

        phase = self.phases.get(phase_name, PhaseUsage(name=phase_name))

        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            usage = getattr(msg, "usage_metadata", None)
            if not usage:
                continue

            input_t = usage.get("input_tokens", 0)
            output_t = usage.get("output_tokens", 0)
            total_t = usage.get("total_tokens", input_t + output_t)

            phase.input_tokens += input_t
            phase.output_tokens += output_t
            phase.total_tokens += total_t

        # Рахуємо вартість
        pricing = self._get_pricing()
        phase.cost_usd = (
            phase.input_tokens * pricing["input"] / 1_000_000
            + phase.output_tokens * pricing["output"] / 1_000_000
        )

        self.phases[phase_name] = phase

    def print_report(self):
        """Друкує таблицю вартості per-phase і total."""
        print("\n" + "=" * 70)
        print("COST REPORT")
        print("=" * 70)
        print(f"  Model: {self.model}")
        pricing = self._get_pricing()
        print(f"  Pricing: ${pricing['input']}/1M input, ${pricing['output']}/1M output")
        print()
        print(f"  {'Phase':<20} {'Input':>10} {'Output':>10} {'Total':>10} {'Cost':>12}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

        total_in = 0
        total_out = 0
        total_all = 0
        total_cost = 0.0

        for phase in self.phases.values():
            print(
                f"  {phase.name:<20} {phase.input_tokens:>10,} "
                f"{phase.output_tokens:>10,} {phase.total_tokens:>10,} "
                f"${phase.cost_usd:>10.6f}"
            )
            total_in += phase.input_tokens
            total_out += phase.output_tokens
            total_all += phase.total_tokens
            total_cost += phase.cost_usd

        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
        print(
            f"  {'TOTAL':<20} {total_in:>10,} {total_out:>10,} "
            f"{total_all:>10,} ${total_cost:>10.6f}"
        )
        print()


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


def create_research_agent():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    agent = create_agent(
        model=model,
        tools=[web_search, news_search],
        system_prompt=SYSTEM_PROMPT,
    )
    return agent, model


# ---------------------------------------------------------------------------
# Pipeline з custom spans
# ---------------------------------------------------------------------------

def run_research_with_spans(topic: str, obs_status: Dict[str, bool]):
    """
    Research pipeline де кожна фаза обгорнута у custom OpenTelemetry span.

    Структура spans:
      research_pipeline (root)
        ├── phase_search      — агент шукає інформацію
        ├── phase_analysis    — агент аналізує знайдене
        └── phase_synthesis   — агент синтезує звіт
    """
    agent, model_name = create_research_agent()
    cost = CostTracker(model=model_name)

    # Отримуємо OpenTelemetry tracer для ручних spans
    if OTEL_AVAILABLE:
        tracer = trace.get_tracer("research-agent")
    else:
        tracer = None

    # --- Головна обгортка ---
    def _run():
        # Фаза 1: Search — агент шукає інформацію по темі
        print("--- Phase 1: Search ---\n")
        search_query = (
            f"Search for the latest articles, papers and news about: {topic}\n"
            "Find at least 5 diverse sources. Return raw search results."
        )

        if tracer:
            with tracer.start_as_current_span("phase_search") as span:
                span.set_attribute("phase", "search")
                span.set_attribute("research.topic", topic)
                result1 = agent.invoke({"messages": [("user", search_query)]})
                span.set_attribute("messages.count", len(result1["messages"]))
        else:
            result1 = agent.invoke({"messages": [("user", search_query)]})

        cost.track_messages("search", result1["messages"])
        search_context = result1["messages"][-1].content
        print(f"  Search done: {len(search_context)} chars\n")

        # Фаза 2: Analysis — аналіз знайденої інформації
        print("--- Phase 2: Analysis ---\n")
        analysis_query = (
            f"Based on these search results about '{topic}':\n\n"
            f"{search_context[:3000]}\n\n"
            "Provide a structured analysis: key themes, patterns, contradictions, "
            "and areas that need more investigation."
        )

        if tracer:
            with tracer.start_as_current_span("phase_analysis") as span:
                span.set_attribute("phase", "analysis")
                span.set_attribute("input.length", len(analysis_query))
                result2 = agent.invoke({"messages": [("user", analysis_query)]})
                span.set_attribute("messages.count", len(result2["messages"]))
        else:
            result2 = agent.invoke({"messages": [("user", analysis_query)]})

        cost.track_messages("analysis", result2["messages"])
        analysis_context = result2["messages"][-1].content
        print(f"  Analysis done: {len(analysis_context)} chars\n")

        # Фаза 3: Synthesis — фінальний звіт
        print("--- Phase 3: Synthesis ---\n")
        synthesis_query = (
            f"Create a final research report on '{topic}'.\n\n"
            f"Analysis:\n{analysis_context[:3000]}\n\n"
            "Structure: Executive Summary, Key Findings (3-5), "
            "Trends, Recommendations, Sources."
        )

        if tracer:
            with tracer.start_as_current_span("phase_synthesis") as span:
                span.set_attribute("phase", "synthesis")
                result3 = agent.invoke({"messages": [("user", synthesis_query)]})
                span.set_attribute("messages.count", len(result3["messages"]))
        else:
            result3 = agent.invoke({"messages": [("user", synthesis_query)]})

        cost.track_messages("synthesis", result3["messages"])
        report = result3["messages"][-1].content

        return report

    # Запускаємо все під root span
    if tracer:
        with tracer.start_as_current_span("research_pipeline") as root:
            root.set_attribute("research.topic", topic)
            root.set_attribute("research.model", model_name)
            report = _run()
            root.set_attribute("report.length", len(report))
    else:
        report = _run()

    # Виводимо результати
    print("=" * 70)
    print("RESEARCH REPORT")
    print("=" * 70)
    print(report)

    # Таблиця вартості
    cost.print_report()

    # Лінки на дашборди
    print("--- Traces ---")
    if obs_status["langsmith"]:
        print("  LangSmith: https://smith.langchain.com")
    if obs_status["phoenix"]:
        print("  Phoenix:   http://localhost:6006")
        print("  (подивись custom spans 'research_pipeline' > 'phase_*' поряд з auto-instrumented)")
    print()

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY not set")

    try:
        status = setup_observability(args.observability)
        run_research_with_spans(args.topic, status)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
