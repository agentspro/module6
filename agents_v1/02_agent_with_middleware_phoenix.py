"""
АГЕНТ З CALLBACKS + PHOENIX ARIZE - Multi-Platform Observability Demo

Демонструє одночасне використання:
1. LangChain 1.0 BaseCallbackHandler (логування, security, токени)
2. LangSmith трейсинг (автоматичний)
3. Phoenix Arize observability (RAG/embedding візуалізація)

ВАЖЛИВО: Потребує встановлення Phoenix:
    pip install arize-phoenix openinference-instrumentation-langchain

ЗАПУСК:
1. Термінал 1: python -m phoenix.server.main serve
2. Термінал 2: python 02_agent_with_middleware_phoenix.py
3. Phoenix UI: http://localhost:6006
4. LangSmith: https://smith.langchain.com

OBSERVABILITY STACK:
- LangSmith: Production tracing, cost tracking
- Phoenix: Development debugging, detailed traces
- Custom Callbacks: Business logic monitoring
"""

import os
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

# ============================================================================
# PHOENIX ARIZE SETUP - Open-source observability
# ============================================================================

# Check if Phoenix libraries are installed
PHOENIX_AVAILABLE = False
try:
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    PHOENIX_AVAILABLE = True
except ImportError:
    print("⚠️  Phoenix Arize not installed. Install with:")
    print("   pip install arize-phoenix openinference-instrumentation-langchain")
    print("   Continuing without Phoenix...\n")

# Initialize Phoenix if available
if PHOENIX_AVAILABLE:
    try:
        # Register Phoenix tracer
        tracer_provider = register(
            project_name="langchain-callbacks-demo",
            endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"),
        )

        # Instrument LangChain
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        print("=" * 70)
        print("🔥 PHOENIX ARIZE OBSERVABILITY ENABLED")
        print("=" * 70)
        print("✅ Phoenix instrumentation active")
        print(f"📊 Project: langchain-callbacks-demo")
        print(f"🌐 Endpoint: {os.getenv('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:6006/v1/traces')}")
        print("🖥️  UI: http://localhost:6006")
        print("\n⚠️  Make sure Phoenix server is running:")
        print("   Terminal 1: python -m phoenix.server.main serve")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"⚠️  Phoenix setup failed: {e}")
        print("Continuing without Phoenix...\n")
        PHOENIX_AVAILABLE = False

# ============================================================================
# LANGSMITH VERIFICATION
# ============================================================================

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("=" * 70)
    print("✅ LANGSMITH TRACING ENABLED")
    print("=" * 70)
    print(f"📊 Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    print("🔍 All LangChain operations will be traced")
    print("🌐 Dashboard: https://smith.langchain.com")
    print("=" * 70 + "\n")
else:
    print("⚠️  LangSmith not enabled. Add to .env:")
    print("   LANGCHAIN_TRACING_V2=true")
    print("   LANGCHAIN_API_KEY=ls__your-key\n")

# Print observability stack summary
print("=" * 70)
print("📊 OBSERVABILITY STACK SUMMARY")
print("=" * 70)
print(f"1. LangSmith:        {'✅ Active' if os.getenv('LANGCHAIN_TRACING_V2') == 'true' else '❌ Disabled'}")
print(f"2. Phoenix Arize:    {'✅ Active' if PHOENIX_AVAILABLE else '❌ Not installed'}")
print("3. Custom Callbacks: ✅ Active (Logging, Security, Tokens)")
print("=" * 70 + "\n")


# ============================================================================
# TOOLS
# ============================================================================

@tool
def get_stock_price(symbol: str) -> str:
    """Get real-time stock price using yfinance API."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")

        if data.empty:
            return f"No data found for symbol {symbol}"

        current_price = data['Close'].iloc[-1]
        return f"${current_price:.2f}"

    except Exception as e:
        return f"Error fetching price for {symbol}: {str(e)}"


@tool
def send_notification(message: str, recipient: str) -> str:
    """
    Send notification to user. This is a HIGH-RISK action.

    Args:
        message: Notification message
        recipient: Recipient email or ID
    """
    return f"✅ Notification sent to {recipient}: {message}"


@tool
def execute_trade(symbol: str, quantity: int, action: str) -> str:
    """
    Execute a trade. HIGH-RISK action.

    Args:
        symbol: Stock symbol
        quantity: Number of shares
        action: 'buy' or 'sell'
    """
    return f"⚠️  Would execute {action} {quantity} shares of {symbol}"


# ============================================================================
# CUSTOM CALLBACK HANDLERS - LangChain 1.0 ОФІЦІЙНИЙ API
# ============================================================================

class LoggingCallback(BaseCallbackHandler):
    """
    Офіційний LangChain 1.0 Callback Handler для детального логування

    Працює разом з Phoenix Arize - Phoenix отримує structured traces,
    а цей callback логує human-readable output.
    """

    def __init__(self):
        super().__init__()
        self.llm_calls = 0
        self.tool_calls = 0
        self.logs = []

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Викликається ПЕРЕД кожним викликом LLM"""
        self.llm_calls += 1

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "llm_start",
            "call_number": self.llm_calls,
            "prompt_length": len(prompts[0]) if prompts else 0
        }
        self.logs.append(log_entry)

        print(f"\n{'='*60}")
        print(f"📝 LOGGING CALLBACK: LLM Call #{self.llm_calls} Started")
        print(f"⏰ Time: {log_entry['timestamp']}")
        print(f"📏 Prompt length: {log_entry['prompt_length']} chars")
        if PHOENIX_AVAILABLE:
            print(f"🔥 Phoenix: Trace captured at http://localhost:6006")
        print(f"{'='*60}\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Викликається ПІСЛЯ кожного виклику LLM"""

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "llm_end",
            "call_number": self.llm_calls,
            "generations": len(response.generations)
        }
        self.logs.append(log_entry)

        print(f"\n{'='*60}")
        print(f"✅ LOGGING CALLBACK: LLM Call #{self.llm_calls} Completed")
        print(f"⏰ Time: {log_entry['timestamp']}")
        print(f"📊 Generations: {log_entry['generations']}")
        print(f"{'='*60}\n")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any
    ) -> None:
        """Викликається ПЕРЕД кожним викликом tool"""
        self.tool_calls += 1

        tool_name = serialized.get("name", "unknown")

        print(f"\n{'='*60}")
        print(f"🔧 TOOL CALL #{self.tool_calls}: {tool_name}")
        print(f"📥 Input: {input_str}")
        print(f"{'='*60}\n")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Викликається ПІСЛЯ кожного виклику tool"""

        print(f"\n{'='*60}")
        print(f"✅ TOOL COMPLETED")
        print(f"📤 Output: {output[:100]}...")
        print(f"{'='*60}\n")

    def get_stats(self):
        """Повертає статистику викликів"""
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "total_logs": len(self.logs)
        }


class SecurityCallback(BaseCallbackHandler):
    """
    Callback для перехоплення та блокування небезпечних дій
    """

    def __init__(self):
        super().__init__()
        self.high_risk_tools = ["execute_trade", "send_notification"]
        self.blocked_calls = 0

    def on_agent_action(
        self,
        action: AgentAction,
        **kwargs: Any
    ) -> None:
        """Перехоплює дії агента перед виконанням"""

        tool_name = action.tool

        if tool_name in self.high_risk_tools:
            self.blocked_calls += 1

            print(f"\n{'='*60}")
            print(f"🔒 SECURITY CALLBACK: HIGH-RISK ACTION DETECTED")
            print(f"⚠️  Tool: {tool_name}")
            print(f"📋 Input: {action.tool_input}")
            print(f"🚫 This would be blocked in production")
            print(f"   Total blocked: {self.blocked_calls}")
            print(f"{'='*60}\n")

    def get_stats(self):
        """Повертає статистику блокувань"""
        return {
            "blocked_calls": self.blocked_calls,
            "high_risk_tools": self.high_risk_tools
        }


class TokenCountCallback(BaseCallbackHandler):
    """
    Callback для підрахунку використаних токенів
    """

    def __init__(self, max_tokens: int = 10000):
        super().__init__()
        self.max_tokens = max_tokens
        self.total_tokens = 0
        self.calls_over_limit = 0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Оцінює кількість токенів перед викликом"""

        # Приблизна оцінка токенів (1 токен ≈ 4 символи для англійської мови)
        estimated_tokens = sum(len(p) // 4 for p in prompts)
        self.total_tokens += estimated_tokens

        print(f"\n{'='*60}")
        print(f"📊 TOKEN COUNTER CALLBACK:")
        print(f"   Estimated input tokens: ~{estimated_tokens}")
        print(f"   Total tokens used: {self.total_tokens}")
        print(f"   Max allowed: {self.max_tokens}")

        if self.total_tokens > self.max_tokens:
            self.calls_over_limit += 1
            print(f"   ⚠️  WARNING: Approaching token limit!")

        print(f"{'='*60}\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Враховує токени у відповіді"""

        # Якщо є token_usage в llm_output
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            if "total_tokens" in usage:
                actual_tokens = usage["total_tokens"]
                print(f"   📈 Actual tokens used: {actual_tokens}")

    def get_stats(self):
        """Повертає статистику використання"""
        return {
            "total_tokens": self.total_tokens,
            "calls_over_limit": self.calls_over_limit,
            "max_tokens": self.max_tokens
        }


# ============================================================================
# СТВОРЕННЯ АГЕНТА З CALLBACKS + PHOENIX
# ============================================================================

def create_agent_with_observability():
    """
    Створює агента з повним observability stack:
    1. Custom Callbacks (business logic)
    2. LangSmith (production tracing)
    3. Phoenix Arize (development debugging)

    Всі три рівні працюють одночасно без конфліктів!
    """
    print("=" * 70)
    print("🤖 АГЕНТ З MULTI-PLATFORM OBSERVABILITY")
    print("=" * 70 + "\n")

    # Створюємо callback instances
    logging_cb = LoggingCallback()
    security_cb = SecurityCallback()
    token_cb = TokenCountCallback(max_tokens=10000)

    # Tools
    tools = [get_stock_price, send_notification, execute_trade]

    print("Available tools:")
    for tool_item in tools:
        risk = " (HIGH-RISK)" if tool_item.name in security_cb.high_risk_tools else ""
        print(f"  • {tool_item.name}{risk}")
    print()

    print("Observability layers:")
    print("  1. Custom Callbacks (Logging, Security, Tokens)")
    print("  2. LangSmith (automatic tracing)")
    print("  3. Phoenix Arize (structured traces)")
    print()

    # Створюємо агента з LangChain 1.0 API
    # Phoenix автоматично інструментує всі LangChain операції
    agent = create_agent(
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="""You are a helpful financial assistant with access to tools.

IMPORTANT: When considering high-risk actions like execute_trade or send_notification, always explain why you would use them.

Think step-by-step and use tools when needed to answer questions accurately."""
    )

    return agent, logging_cb, security_cb, token_cb


# ============================================================================
# ТЕСТУВАННЯ АГЕНТА З MULTI-PLATFORM OBSERVABILITY
# ============================================================================

def test_agent_with_observability():
    """Тестує агента з повним observability stack"""

    agent, logging_cb, security_cb, token_cb = create_agent_with_observability()

    test_queries = [
        {
            "query": "What's the current price of AAPL stock?",
            "description": "Safe query - all observability platforms capture this",
            "expected": "get_stock_price tool call"
        },
        {
            "query": "Get TSLA price and send me notification about it",
            "description": "Contains HIGH-RISK tool - security callback + Phoenix trace",
            "expected": "SecurityCallback + Phoenix span for risky action"
        }
    ]

    for i, test in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"TEST {i}: {test['description']}")
        print("=" * 70)
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")
        print("-" * 70 + "\n")

        try:
            # LangChain 1.0 invoke з callbacks
            # Phoenix автоматично захоплює всі операції через instrumentation
            result = agent.invoke({
                "messages": [{"role": "user", "content": test["query"]}]
            }, config={"callbacks": [logging_cb, security_cb, token_cb]})

            # Extract output from messages
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                output = last_message.content if hasattr(last_message, "content") else str(last_message)
            else:
                output = str(result)

            print("\n" + "-" * 70)
            print("📋 RESULT:")
            print("-" * 70)
            print(f"Output: {output}\n")

            if PHOENIX_AVAILABLE:
                print("🔥 Check Phoenix UI for detailed traces: http://localhost:6006")
            if os.getenv("LANGCHAIN_TRACING_V2") == "true":
                print("📊 Check LangSmith for production traces: https://smith.langchain.com")

        except Exception as e:
            print(f"\n❌ ERROR: {e}\n")
            import traceback
            traceback.print_exc()

        input("\n⏸️  Press Enter to continue to next test...\n")

    # Виводимо статистику всіх observability layers
    print("\n" + "=" * 70)
    print("📊 MULTI-PLATFORM OBSERVABILITY STATISTICS")
    print("=" * 70 + "\n")

    print("Custom Callbacks:")
    print("-" * 40)
    logging_stats = logging_cb.get_stats()
    print(f"  LLM calls: {logging_stats['llm_calls']}")
    print(f"  Tool calls: {logging_stats['tool_calls']}")
    print(f"  Total logs: {logging_stats['total_logs']}")
    print()

    security_stats = security_cb.get_stats()
    print(f"  High-risk detections: {security_stats['blocked_calls']}")
    print(f"  Monitored tools: {', '.join(security_stats['high_risk_tools'])}")
    print()

    token_stats = token_cb.get_stats()
    print(f"  Total tokens: {token_stats['total_tokens']}")
    print(f"  Calls over limit: {token_stats['calls_over_limit']}")
    print()

    print("External Platforms:")
    print("-" * 40)
    if PHOENIX_AVAILABLE:
        print("  🔥 Phoenix Arize: http://localhost:6006")
        print("     → Detailed traces, embeddings, spans")
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("  📊 LangSmith: https://smith.langchain.com")
        print("     → Production monitoring, costs, analytics")
    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("🎯 Multi-Platform Observability Demo")
    print("=" * 70)
    print()
    print("This demo shows LangChain agent with 3 observability layers:")
    print()
    print("  1️⃣  Custom Callbacks (business logic)")
    print("     ✅ Logging (on_llm_start, on_llm_end)")
    print("     ✅ Security (on_agent_action)")
    print("     ✅ Token counting (cost control)")
    print()
    print("  2️⃣  LangSmith (production monitoring)")
    print("     ✅ Automatic tracing")
    print("     ✅ Cost tracking")
    print("     ✅ Performance analytics")
    print()
    print("  3️⃣  Phoenix Arize (development/debugging)")
    print("     ✅ Structured traces")
    print("     ✅ Embedding visualization")
    print("     ✅ Local deployment")
    print()
    print("All three work together without conflicts!")
    print("=" * 70 + "\n")

    # Перевірка Phoenix server
    if PHOENIX_AVAILABLE:
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

    # Перевірка API ключів
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found in environment!")
        print("Please set it in .env file")
        exit(1)

    try:
        test_agent_with_observability()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 70)
        print("\n💡 View traces in:")
        if PHOENIX_AVAILABLE:
            print("   🔥 Phoenix: http://localhost:6006")
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            print("   📊 LangSmith: https://smith.langchain.com")
        print()

    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
