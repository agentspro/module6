# Evaluation Pipeline: Tracing, Datasets, Evaluation Runs, Guardrails

Покроковий навчальний посібник для демонстрації на курсі.

---

## Зміст

1. [Tracing — що бачить агент під капотом](#1-tracing)
2. [Datasets — єдине джерело правди](#2-datasets)
3. [Evaluation Runs — автоматична оцінка якості](#3-evaluation-runs)
4. [Guardrails AI — safety layer](#4-guardrails-ai)
5. [Як все пов'язано](#5-як-все-повязано)

---

## 1. Tracing

### Що це

Tracing — автоматичний запис **кожної** операції агента: LLM виклики, tool calls, час виконання, токени, вартість. Без змін у коді.

### Як працює

```
User Query
└── Agent Run (trace)
    ├── LLM Call #1 (планування)        150 tokens, 1.2s
    ├── Tool: search_topic("AI agents")  0.3s
    ├── LLM Call #2 (аналіз)            450 tokens, 1.8s
    ├── Tool: analyze_data("AI agents")  0.1s
    └── LLM Call #3 (фінальна відповідь) 800 tokens, 2.1s

Total: 3 LLM calls, 2 tool calls, 5.5s, ~$0.003
```

Кожен вузол розкривається — видно точний input/output, кількість токенів, latency.

### Як увімкнути (LangSmith)

Tracing вмикається **тільки** через env vars — жодних змін в коді:

```bash
# .env
LANGCHAIN_TRACING_V2=true        # вмикає tracing
LANGCHAIN_API_KEY=lsv2_pt_...    # API ключ LangSmith
LANGCHAIN_PROJECT=my-project     # назва проєкту (групує traces)
```

Після цього **кожен** виклик LangChain (agent, chain, LLM) автоматично записується.

### Демонстрація

**Скрипт:** `05_mcp_research_agent_multi_observability.py`

```bash
# Запуск з LangSmith tracing
python 05_mcp_research_agent_multi_observability.py --observability langsmith --topic "AI agents"

# Запуск з трьома платформами одночасно
python 05_mcp_research_agent_multi_observability.py --observability langsmith phoenix langfuse
```

**Що показати в UI:**
1. Відкрити https://smith.langchain.com
2. Зліва вибрати проєкт `mcp-research-agent`
3. Клікнути на trace — показати дерево викликів
4. Розкрити LLM Call — показати input/output, tokens, latency
5. Розкрити Tool Call — показати що агент передав і що отримав

**Скрипт з OpenTelemetry spans:** `06_custom_spans_and_cost_tracking.py`

```bash
python 06_custom_spans_and_cost_tracking.py --topic "RAG optimization" --observability langsmith
```

Цей скрипт додає **кастомні spans** поверх автоматичних — можна групувати операції (search phase, analysis phase, synthesis phase) і рахувати вартість.

### Три платформи порівняння

| Платформа | Тип | Tracing | Datasets | Evaluation |
|-----------|-----|---------|----------|------------|
| **LangSmith** | Cloud | env vars, zero code | API | Built-in |
| **Phoenix Arize** | Local (`:6006`) | OpenTelemetry | --- | --- |
| **LangFuse** | Cloud/Self-host | Callback handler | API | API |

Phoenix — тільки tracing. LangSmith та LangFuse — tracing **+** datasets **+** evaluation.

---

## 2. Datasets

### Що це

Dataset — колекція пар `input` / `expected_output` для тестування агента. Одне джерело правди для всіх eval-скриптів.

### Навіщо

- **Без dataset:** кожен eval-скрипт хардкодить свої test cases — дублювання, неконсистентність
- **З dataset:** один JSON файл → всі скрипти читають з нього → змінив в одному місці — працює скрізь

### Формат

```json
{
  "name": "mcp-research-agent-eval",
  "description": "Evaluation dataset for MCP Research Agent",
  "examples": [
    {
      "inputs": {"question": "Які тренди в AI agent architectures?"},
      "outputs": {"expected": "Multi-agent systems, LangGraph, tool-use agents..."},
      "metadata": {"category": "trend-analysis", "difficulty": "medium"}
    }
  ]
}
```

Ключові поля:
- `inputs.question` — що подаємо агенту
- `outputs.expected` — очікувана відповідь (reference для порівняння)
- `metadata` — категорія, складність (для фільтрації та аналізу)

### Демонстрація

**Файл:** `datasets/eval_dataset.json` — unified dataset з 12 test cases.

```bash
# Подивитися вміст
cat datasets/eval_dataset.json | python -m json.tool

# Статистика по категоріях
python -c "
import json
with open('datasets/eval_dataset.json') as f:
    d = json.load(f)
for e in d['examples']:
    cat = e['metadata']['category']
    diff = e['metadata']['difficulty']
    q = e['inputs']['question'][:60]
    print(f'  [{cat:>17}] [{diff:>6}] {q}...')
"
```

### Завантаження в LangSmith / LangFuse

**Скрипт:** `upload_dataset.py`

```bash
# Завантажити тільки в LangSmith
python upload_dataset.py --platform langsmith

# Завантажити в обидві платформи
python upload_dataset.py --platform langsmith langfuse

# Перезавантажити (видалити + створити заново)
python upload_dataset.py --platform langsmith --recreate

# Завантажити свій файл
python upload_dataset.py --platform langsmith --file datasets/my_dataset.json
```

**Що показати в UI після завантаження:**
1. LangSmith: Datasets & Testing → `mcp-research-agent-eval` → видно всі 12 examples
2. LangFuse: Datasets → `mcp-research-agent-eval` → видно items з metadata

### Генерація datasets

**Скрипт:** `dataset_manager.py`

```bash
# Згенерувати dataset через LLM
python dataset_manager.py --action generate --topic "RAG systems" --count 10

# Експорт існуючого dataset з LangSmith
python dataset_manager.py --action export --name "mcp-research-agent-eval"

# Імпорт в LangSmith з файлу
python dataset_manager.py --action import --file datasets/eval_dataset.json

# Створити шаблон для ручного заповнення
python dataset_manager.py --action template --type research
```

### Різниця: Dataset vs Traces

```
TRACES (всі 3 платформи):
  Що агент зробив → запис факту
  "Агент відповів X на питання Y за 3.2 секунди"
  Для: дебагінг, моніторинг, оптимізація

DATASETS (LangSmith, LangFuse):
  Що агент ПОВИНЕН зробити → еталон
  "На питання Y правильна відповідь — Z"
  Для: тестування, regression, порівняння версій

EVALUATION RUNS (LangSmith, LangFuse):
  Порівняння traces з datasets → оцінка
  "Агент відповів X, еталон Z, score = 0.85"
  Для: якість, CI/CD gates, прийняття рішень
```

---

## 3. Evaluation Runs

### Що це

Evaluation Run — прогін агента через dataset + автоматична оцінка кожної відповіді через evaluators (метрики якості).

### Як працює

```
Dataset Example ──→ Agent ──→ Actual Output
     │                             │
     │         ┌───────────────────┘
     ▼         ▼
  EVALUATORS
  ├── Relevancy:   0.92  ✅
  ├── Helpfulness: 0.85  ✅
  ├── Faithfulness: 0.78 ✅
  └── Correctness: 0.45  ❌  (threshold: 0.7)

Repeat × N examples → Aggregate → Pass/Fail
```

Це як unit tests, але для LLM:
- Unit test: `assertEqual(output, expected)` — exact match
- Evaluation: `score(output, expected) >= threshold` — fuzzy match через метрики

### Типи evaluators

**1. Rule-based (детерміністичні):**
- Довжина відповіді (100-2000 символів)
- Наявність ключових слів
- Формат (JSON, markdown, тощо)

**2. LLM-as-Judge:**
- GPT оцінює релевантність (0.0 — 1.0)
- GPT оцінює корисність
- GPT перевіряє фактичну точність

**3. DeepEval метрики (спеціалізовані):**
- Answer Relevancy — чи відповідь релевантна питанню
- Faithfulness — чи відповідь вірна контексту (не галюцинує)
- Contextual Relevancy — чи контекст релевантний питанню
- G-Eval — кастомні критерії оцінки

### Демонстрація: DeepEval

**Скрипт:** `05_mcp_research_agent_evaluation.py`

```bash
python 05_mcp_research_agent_evaluation.py
```

Що робить скрипт:
1. Читає dataset з `datasets/eval_dataset.json` (12 examples)
2. Прогоняє агента по кожному питанню → отримує `actual_output`
3. Збирає `retrieval_context` з tool calls
4. Оцінює через 5 метрик:
   - Answer Relevancy (threshold 0.7)
   - Faithfulness (threshold 0.7)
   - Contextual Relevancy (threshold 0.6)
   - Research Quality — кастомний G-Eval
   - Tool Usage — кастомний G-Eval

**Що показати:**
- Для кожного test case — scores по всіх метриках
- Зведена таблиця pass/fail
- Reason чому метрика не пройшла (DeepEval пояснює)

### Демонстрація: LangSmith Evaluation

**Скрипт:** `05_mcp_research_agent_langsmith_eval.py`

```bash
python 05_mcp_research_agent_langsmith_eval.py
```

Що робить скрипт:
1. Перевіряє чи dataset `mcp-research-agent-eval` існує в LangSmith (якщо ні — завантажує з JSON)
2. Визначає `research_agent_predict` — функцію яка приймає input і повертає output
3. Створює evaluators: relevance (LLM judge), helpfulness (LLM judge), length (rule-based)
4. Викликає `langsmith.evaluation.evaluate()` — прогоняє агента через dataset

**Що показати в LangSmith UI:**
1. Datasets & Testing → `mcp-research-agent-eval`
2. Experiment run → видно scores для кожного example
3. Клікнути на example → видно trace + evaluator результати
4. Порівняти два runs — видно regression/improvement

### Порівняння DeepEval vs LangSmith evaluation

| Аспект | DeepEval | LangSmith evaluate() |
|--------|----------|---------------------|
| Dataset | Читає з JSON файлу | Читає з LangSmith platform |
| Evaluators | Спеціалізовані метрики (Faithfulness, etc.) | Кастомні функції |
| UI | Confident AI (опціонально) | LangSmith dashboard |
| Trace linking | Ні | Так (кожен eval прив'язаний до trace) |
| Порівняння runs | Обмежено | Вбудовано в UI |
| Залежності | `pip install deepeval` | Вже є з `langsmith` |

---

## 4. Guardrails AI

### Що це

Guardrails AI — фреймворк для валідації output LLM перед поверненням користувачу. Safety layer.

### Навіщо

LLM може згенерувати:
- Персональні дані (PII) — email, телефон, адреса
- Токсичний контент — образи, дискримінація
- Галюцинації — вигадані факти
- Невідповідний формат — JSON замість тексту

Guardrails ловить це **до** повернення користувачу.

### Архітектура

```
User Query ──→ Agent ──→ Raw Output
                              │
                       ┌──────▼──────┐
                       │  GUARDRAILS │
                       │             │
                       │ ✓ No PII    │
                       │ ✓ No toxic  │
                       │ ✓ On topic  │
                       │ ✓ Length OK │
                       └──────┬──────┘
                              │
                         Pass? ─┐
                        /       \
                      Yes        No
                       │          │
                  Return      Fix / Block /
                  to user     Retry (reask)
```

### Встановлення

```bash
# Основний пакет
pip install guardrails-ai

# Встановити конкретні validators з Hub
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/detect_pii

# API ключ для Hub (вже в .env)
export GUARDRAILS_API_KEY=eyJhbG...
```

### Як працює Guard

```python
from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage, DetectPII

# Створити guard з validators
guard = Guard().use_many(
    ToxicLanguage(
        threshold=0.5,
        validation_method="sentence",
        on_fail=OnFailAction.EXCEPTION     # блокувати повністю
    ),
    DetectPII(
        pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
        on_fail=OnFailAction.FIX           # замінити на [REDACTED]
    ),
)

# Валідація тексту
result = guard.validate("Contact john@example.com for details")
print(result.validated_output)   # "Contact [REDACTED] for details"
print(result.validation_passed)  # True (бо FIX, не EXCEPTION)
```

### OnFailAction варіанти

| Action | Що робить | Коли використовувати |
|--------|-----------|---------------------|
| `EXCEPTION` | Raise помилку, блокує повністю | Toxicity, критичні порушення |
| `FIX` | Автоматично виправити | PII (замінити на [REDACTED]) |
| `REASK` | Попросити LLM спробувати ще раз | Невірний формат |
| `NOOP` | Нічого, тільки логувати | Monitoring, збір статистики |

### Демонстрація

**Скрипт:** `05_mcp_research_agent_evaluation_with_guardrails.py`

```bash
python 05_mcp_research_agent_evaluation_with_guardrails.py
```

Що робить скрипт:
1. Читає dataset з `datasets/eval_dataset.json`
2. Прогоняє `GuardedResearchAgent` — агент з вбудованою валідацією
3. Кожна відповідь проходить через guardrails перевірки:
   - Довжина (50–5000 символів)
   - Toxic keywords
   - Змістовність (мінімум 3 речення)
4. Потім — DeepEval метрики (як в Part 3)
5. Зведений звіт: guardrails pass rate + DeepEval scores

**Що показати:**
- Guardrails Validation Summary — скільки пройшло / скільки ні
- Violations та Warnings для кожного test case
- Комбінація: safety (guardrails) + quality (DeepEval)

### Guardrails в production pipeline

```python
# В реальному коді агента:

agent = create_agent(model="gpt-4o-mini", tools=tools, system_prompt="...")
guard = Guard().use_many(ToxicLanguage(...), DetectPII(...))

def safe_research(query: str) -> str:
    """Агент з safety layer."""
    result = agent.invoke({"messages": [("user", query)]})
    raw_output = result["messages"][-1].content

    try:
        validated = guard.validate(raw_output)
        return validated.validated_output  # очищений вихід
    except Exception:
        return "Вибачте, не можу надати таку відповідь."
```

---

## 5. Як все пов'язано

### Повний pipeline

```
                    DEV TIME                          PRODUCTION
            ┌─────────────────────┐          ┌─────────────────────┐
            │                     │          │                     │
            │  1. Dataset         │          │  4. Guardrails      │
            │     (JSON файл)     │          │     (safety layer)  │
            │         │           │          │         │           │
            │         ▼           │          │         ▼           │
            │  2. Evaluation Run  │  deploy  │  5. Tracing         │
            │     (DeepEval /     │ ───────→ │     (LangSmith)     │
            │      LangSmith)     │          │         │           │
            │         │           │          │         ▼           │
            │         ▼           │          │  6. Monitoring      │
            │  3. Pass? ──→ Fix   │          │     (dashboards)    │
            │     Yes ──→ Deploy  │          │                     │
            └─────────────────────┘          └─────────────────────┘
```

### Скрипти по кроках

| Крок | Скрипт | Що робить |
|------|--------|-----------|
| Tracing demo | `05_mcp_research_agent_multi_observability.py` | Запуск агента з записом traces |
| Custom spans | `06_custom_spans_and_cost_tracking.py` | Кастомні OpenTelemetry spans + вартість |
| Dataset upload | `upload_dataset.py` | Завантаження dataset в LangSmith/LangFuse |
| Dataset generate | `dataset_manager.py` | Генерація/експорт/імпорт datasets |
| DeepEval | `05_mcp_research_agent_evaluation.py` | Evaluation через DeepEval метрики |
| LangSmith eval | `05_mcp_research_agent_langsmith_eval.py` | Evaluation через LangSmith evaluate() |
| Guardrails eval | `05_mcp_research_agent_evaluation_with_guardrails.py` | DeepEval + Guardrails validation |

### Рекомендований порядок демонстрації

```bash
# 1. Показати tracing — запустити агента, відкрити LangSmith UI
python 05_mcp_research_agent_multi_observability.py --observability langsmith

# 2. Показати dataset — що в JSON файлі, завантажити в LangSmith
cat datasets/eval_dataset.json | python -m json.tool
python upload_dataset.py --platform langsmith

# 3. Показати evaluation — запустити LangSmith eval, відкрити результати
python 05_mcp_research_agent_langsmith_eval.py

# 4. Показати DeepEval — інший підхід до evaluation
python 05_mcp_research_agent_evaluation.py

# 5. Показати Guardrails — safety + evaluation
python 05_mcp_research_agent_evaluation_with_guardrails.py
```

### Що відкрити в браузері

- **LangSmith:** https://smith.langchain.com
  - Projects → traces (Part 1)
  - Datasets & Testing → dataset + evaluation runs (Part 2, 3)
- **LangFuse:** https://cloud.langfuse.com
  - Traces → traces (Part 1)
  - Datasets → dataset items (Part 2)
- **Phoenix:** http://localhost:6006 (якщо запущено)
  - Traces → тільки traces (Part 1)
