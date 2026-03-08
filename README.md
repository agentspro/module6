# Multi-Agent AI Systems з Observability: LangChain 1.0, LangGraph 1.0 та CrewAI

Цей репозиторій містить практичні приклади мультиагентних систем з інтеграцією observability платформ (LangSmith, Phoenix Arize, LangFuse), evaluation та datasets.

## Структура репозиторію

```
module6/
├── agents_v1/          # LangChain 1.0 + LangGraph 1.0 + Observability & Evaluation
│   ├── 01_basic_agent.py
│   ├── 02_agent_with_middleware.py
│   ├── 02_agent_with_middleware_phoenix.py          # Phoenix tracing
│   ├── 03_rag_agent_langgraph.py
│   ├── 04_multiagent_langgraph.py
│   ├── 05_mcp_research_agent_multi_observability.py # LangSmith + Phoenix + LangFuse
│   ├── 05_mcp_research_agent_evaluation.py          # DeepEval metrics
│   ├── 05_mcp_research_agent_evaluation_with_guardrails.py  # DeepEval + Guardrails AI
│   ├── 05_mcp_research_agent_langsmith_eval.py      # LangSmith evaluation
│   ├── 06_custom_spans_and_cost_tracking.py         # OpenTelemetry spans + cost tracking
│   ├── dataset_manager.py                           # Dataset export/import/generation
│   ├── datasets/
│   ├── requirements.txt
│   └── .env.example
│
└── agents_v2/          # CrewAI Framework + Observability
    ├── 01_basic_crew.py
    ├── 02_hierarchical_crew.py
    ├── 02_hierarchical_crew_langfuse.py             # LangFuse tracing
    ├── 03_research_crew_with_tools.py
    ├── 03_research_crew_phoenix.py                  # Phoenix tracing
    ├── 04_memory_enabled_crew.py
    ├── requirements.txt
    └── .env.example
```

## Що нового порівняно з Module 5

Module 6 базується на агентах з Module 5 і додає:
- **Observability** — інтеграція з LangSmith, Phoenix Arize та LangFuse
- **Evaluation** — DeepEval метрики якості, Guardrails AI safety, LangSmith evaluation
- **Custom spans** — ручні OpenTelemetry spans + token/cost tracking
- **Datasets** — CLI інструмент для роботи з evaluation datasets

---

## Observability платформи

Підтримуються три платформи, які можна використовувати одночасно:

| Платформа | Тип | Найкраще для | Скрипти |
|-----------|-----|--------------|---------|
| **LangSmith** | Commercial | LangChain/LangGraph нативна інтеграція | Працює через env vars автоматично |
| **Phoenix Arize** | Open-source (локальний) | Development, RAG visualization, tool tracking | `02_*_phoenix.py`, `03_*_phoenix.py` |
| **LangFuse** | Open-source/Cloud | Production monitoring, analytics | `02_*_langfuse.py`, `05_*_multi_observability.py` |

### LangSmith (нульова конфігурація коду)

Увімкнений через env vars — працює для всіх скриптів автоматично:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your-key-here
LANGCHAIN_PROJECT=your-project
```

Реєстрація: https://smith.langchain.com

### Phoenix Arize (локальний сервер)

```bash
pip install arize-phoenix openinference-instrumentation-langchain  # для agents_v1
pip install arize-phoenix openinference-instrumentation-crewai     # для agents_v2

# Запустити сервер
python -m phoenix.server.main serve

# UI на http://localhost:6006
```

### LangFuse (cloud або self-hosted)

```bash
pip install langfuse

# Додати до .env:
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Реєстрація: https://cloud.langfuse.com

---

## agents_v1: LangChain 1.0 & LangGraph 1.0

### Базові агенти (з Module 5)

1. **Basic Agent** (`01`) — базовий агент з `create_agent()` API
2. **Middleware Agent** (`02`) — logging, security, token limit middleware
3. **RAG Agent** (`03`) — Agentic RAG з LangGraph StateGraph + FAISS
4. **Multi-Agent** (`04`) — Supervisor Pattern з спеціалізованими агентами

### Observability скрипти

5. **Multi Observability** (`05_mcp_research_agent_multi_observability.py`) — MCP research agent з потрійним трейсингом (LangSmith + Phoenix + LangFuse одночасно)
6. **Custom Spans** (`06_custom_spans_and_cost_tracking.py`) — ручні OpenTelemetry spans + реальний token/cost tracking
7. **Phoenix Middleware** (`02_agent_with_middleware_phoenix.py`) — middleware agent з Phoenix tracing

### Evaluation скрипти

8. **DeepEval** (`05_mcp_research_agent_evaluation.py`) — answer relevancy, faithfulness, contextual relevancy, custom G-Eval
9. **Guardrails** (`05_mcp_research_agent_evaluation_with_guardrails.py`) — DeepEval + Guardrails AI (PII detection, toxicity, format validation)
10. **LangSmith Eval** (`05_mcp_research_agent_langsmith_eval.py`) — нативна LangSmith evaluation

### CLI аргументи

Скрипти 05 і 06 підтримують CLI:

```bash
python 05_mcp_research_agent_multi_observability.py --observability langsmith phoenix langfuse
python 05_mcp_research_agent_multi_observability.py --topic "RAG optimization" --steps 3
python 06_custom_spans_and_cost_tracking.py --observability phoenix langsmith --topic "AI agents"
```

### Dataset інструменти

```bash
python dataset_manager.py --action export    # Експорт з LangSmith
python dataset_manager.py --action import    # Імпорт dataset
python dataset_manager.py --action generate  # Генерація LLM-ом
python dataset_manager.py --action template  # Шаблон для ручного створення
```

---

## agents_v2: CrewAI Framework

### Базові crews (з Module 5)

1. **Basic Crew** (`01`) — sequential crew з 3 агентами
2. **Hierarchical Crew** (`02`) — ієрархічна структура з авто-менеджером
3. **Research Crew** (`03`) — tools integration (FileRead, custom tools)
4. **Memory Crew** (`04`) — персистентна пам'ять між сесіями

### Observability варіанти

5. **LangFuse** (`02_hierarchical_crew_langfuse.py`) — hierarchical crew з LangFuse моніторингом
6. **Phoenix** (`03_research_crew_phoenix.py`) — research crew з Phoenix tool tracking

---

## Швидкий старт

### 1. Клонуйте та оберіть фреймворк

```bash
git clone https://github.com/agentspro/module6.git
cd module6
cd agents_v1  # або cd agents_v2
```

### 2. Створіть venv та встановіть залежності

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

**ВАЖЛИВО:** Python 3.11 або 3.12 (не 3.14 — ламає Pydantic).

### 3. Налаштуйте API ключі

```bash
cp .env.example .env
# Відредагуйте .env
```

### 4. Запустіть базові агенти

```bash
# agents_v1
python 01_basic_agent.py
python 02_agent_with_middleware.py
python 03_rag_agent_langgraph.py
python 04_multiagent_langgraph.py

# agents_v2
python 01_basic_crew.py
python 02_hierarchical_crew.py
python 03_research_crew_with_tools.py
python 04_memory_enabled_crew.py
```

### 5. Запустіть observability демо

```bash
# agents_v1 — потрійна observability
python 05_mcp_research_agent_multi_observability.py --observability langsmith phoenix langfuse

# agents_v1 — custom spans + cost tracking
python 06_custom_spans_and_cost_tracking.py --observability phoenix

# agents_v2 — LangFuse для hierarchical crew
python 02_hierarchical_crew_langfuse.py

# agents_v2 — Phoenix для research crew
python 03_research_crew_phoenix.py
```

---

## Конфігурація API ключів

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# LangSmith (рекомендовано)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your-key-here
LANGCHAIN_PROJECT=your-project

# Phoenix Arize (опціонально)
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces

# LangFuse (опціонально)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Tools (потрібно для агентів з інструментами)
OPENWEATHERMAP_API_KEY=your-key-here

# Evaluation (опціонально)
GUARDRAILS_API_KEY=your-key-here
```

---

## Порівняння фреймворків

### LangChain/LangGraph (agents_v1)

**Переваги:**
- Повний контроль над потоком виконання
- Потужна система middleware
- Нативна інтеграція з LangSmith
- Детальний state management

**Використовуйте коли:**
- Потрібен детальний контроль над логікою
- Складні графи станів з умовними переходами
- Критична observability та evaluation

### CrewAI (agents_v2)

**Переваги:**
- Швидка розробка з простим API
- Вбудована рольова модель та делегація
- Hierarchical process out-of-the-box
- Природна колаборація між агентами

**Використовуйте коли:**
- Потрібна швидка розробка multi-agent систем
- Чітка рольова структура команди
- Focus на бізнес-логіку

---

## Вимоги

- Python >= 3.10, < 3.14 (рекомендовано 3.11 або 3.12)
- OpenAI API ключ
- (Опціонально) LangSmith, Phoenix Arize, LangFuse для observability
- (Опціонально) DeepEval, Guardrails AI для evaluation

---

## Ресурси

### Observability
- [LangSmith](https://smith.langchain.com/)
- [Phoenix Arize](https://phoenix.arize.com/)
- [LangFuse](https://langfuse.com/)

### Evaluation
- [DeepEval](https://docs.confident-ai.com/)
- [Guardrails AI](https://www.guardrailsai.com/)

### Фреймворки
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI Documentation](https://docs.crewai.com/)

---

## Ліцензія

MIT License

---

## Автор

sanyaden <alex.denysyuk@gmail.com>
