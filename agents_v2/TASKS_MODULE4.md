# Практичні завдання: CrewAI + MCP (Model Context Protocol)

Базовий репозиторій: https://github.com/agentspro/module4

## Огляд проекту

У цьому проекті ви працюватимете з:
- **CrewAI** - фреймворк для multi-agent систем
- **MCP (Model Context Protocol)** - стандарт для підключення зовнішніх інструментів
- **Паралельні агенти** - одночасний пошук з різних джерел
- **Sequential Thinking** - структуроване покрокове мислення

---

## Завдання 1: Розширення пошукових джерел

### 1.1 Додати агента для пошуку в Guardian
**Файл:** `parallel_agent_with_mcp.py`

**Що зробити:**
1. У функції `create_search_agents()` додати четвертого агента: `guardian_agent`
2. Агент має шукати новини на `site:theguardian.com`
3. Додати відповідну задачу `guardian_task`
4. Оновити `analysis_task` щоб враховувати 4 джерела замість 3
5. Оновити `synthesis_task` для обробки 4 джерел

**Очікуваний результат:**
```python
guardian_agent = Agent(
    role='Guardian News Researcher',
    goal='Знайти та проаналізувати останні новини з Guardian',
    backstory='Ти експерт з пошуку та аналізу новин Guardian...',
    tools=[search_tool],
    verbose=True,
    allow_delegation=False
)

# У return додати guardian_agent
return bbc_agent, cnn_agent, reuters_agent, guardian_agent
```

**Критерії перевірки:**
- ✅ Guardian агент створений
- ✅ Задача для Guardian додана
- ✅ Crew включає всіх 4 агентів
- ✅ Аналіз враховує всі 4 джерела

---

### 1.2 Додати агента для пошуку в соцмережах
**Що зробити:**
1. Створити `social_media_agent` для пошуку в Twitter/X
2. Пошуковий запит: `site:twitter.com {topic}`
3. Агент має фокусуватися на trending topics та viral posts
4. Додати у фінальний звіт секцію "Social Media Sentiment"

**Підказка:**
```python
social_agent = Agent(
    role='Social Media Researcher',
    goal='Знайти та проаналізувати обговорення теми в соцмережах',
    backstory='Ти експерт з аналізу соціальних медіа...',
    ...
)
```

---

## Завдання 2: Покращення MCP Sequential Thinking

### 2.1 Збільшити кількість кроків аналізу
**Файл:** `parallel_agent_with_mcp.py`

**Що змінити:**
1. У `analysis_description` змінити кількість кроків з 5 на 7
2. Додати нові кроки:
   - Крок 6: "Перевіряю факти та достовірність джерел"
   - Крок 7: "Оцінюю довгострокові наслідки"
3. Оновити всі `thoughtNumber` та `totalThoughts`

**Приклад:**
```python
Крок 6 - thought: "Перевіряю факти та достовірність джерел: [твій аналіз]"
        thoughtNumber: 6, totalThoughts: 7, nextThoughtNeeded: true

Крок 7 - thought: "Оцінюю довгострокові наслідки: [твій аналіз]"
        thoughtNumber: 7, totalThoughts: 7, nextThoughtNeeded: false
```

---

### 2.2 Додати валідацію кроків
**Що зробити:**
1. Після кожного непарного кроку (1, 3, 5) додати перевірку
2. Якщо аналіз недостатній - додати додатковий крок
3. Використати поля `isRevision: true` та `revisesThought: N`

**Приклад:**
```python
Крок 3 - thought: "Виділяю протиріччя..."
        thoughtNumber: 3, totalThoughts: 7, nextThoughtNeeded: true

Крок 3.1 - thought: "Переглядаю попередній аналіз протиріч, знайшов більше деталей..."
          thoughtNumber: 4, totalThoughts: 8, nextThoughtNeeded: true,
          isRevision: true, revisesThought: 3
```

---

## Завдання 3: Робота з іншими MCP серверами

### 3.1 Додати Filesystem MCP для збереження звітів
**Файл:** новий `save_reports_agent.py`

**Що зробити:**
1. Створити агента з Filesystem MCP tools
2. Агент має зберігати фінальні звіти у файли
3. Формат імені: `report_{topic}_{timestamp}.md`
4. Створити директорію `/reports` якщо її немає
5. Додати метадані на початку файлу (дата, тема, джерела)

**Структура:**
```python
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

def create_file_saver_agent():
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "./reports"  # дозволена директорія
        ]
    )

    with MCPServerAdapter(server_params, connect_timeout=60) as mcp_tools:
        agent = Agent(
            role='Report Saver',
            goal='Зберігати звіти у файли з правильним форматуванням',
            tools=mcp_tools,
            ...
        )
        # ваш код
```

**Критерії:**
- ✅ Агент створений з Filesystem MCP
- ✅ Звіти зберігаються у `/reports`
- ✅ Формат імені правильний
- ✅ Метадані присутні

---

### 3.2 Додати Memory MCP для збереження історії аналізів
**Файл:** новий `memory_agent.py`

**Що зробити:**
1. Підключити Memory MCP server
2. Створити агента який зберігає:
   - Entity: кожна тема аналізу
   - Entity: кожне джерело (BBC, CNN, etc.)
   - Relations: тема → джерела що використовувались
   - Observations: ключові висновки
3. Додати функцію `get_analysis_history(topic)` - отримати всі попередні аналізи теми
4. Інтегрувати з основним агентом

**Приклад структури:**
```python
# Entities
create_entities([
    {"name": "AI_breakthrough", "entityType": "topic", "observations": [...]},
    {"name": "BBC", "entityType": "source", "observations": [...]},
])

# Relations
create_relations([
    {"from": "AI_breakthrough", "to": "BBC", "relationType": "analyzed_in"}
])
```

---

### 3.3 Додати Git MCP для аналізу змін у коді
**Файл:** новий `code_analysis_agent.py`

**Що зробити:**
1. Підключити Git MCP server до поточного репозиторію
2. Створити агента який аналізує:
   - Останні 5 комітів
   - Які файли найчастіше змінюються
   - Які розробники найактивніші
3. Генерувати звіт про активність проекту
4. Додати рекомендації щодо code review

---

## Завдання 4: Покращення архітектури

### 4.1 Додати Coordinator Agent
**Файл:** `parallel_agent_with_mcp.py`

**Що зробити:**
1. Створити `CoordinatorAgent` який керує процесом
2. Coordinator приймає рішення:
   - Чи потрібно більше джерел?
   - Чи достатньо інформації?
   - Чи потрібна додаткова перевірка?
3. Може динамічно додавати задачі
4. Використовує `allow_delegation=True`

**Структура:**
```python
coordinator = Agent(
    role='Analysis Coordinator',
    goal='Координувати процес аналізу та забезпечити якість',
    backstory='Ти досвідчений менеджер проектів...',
    allow_delegation=True,  # може делегувати іншим агентам
    verbose=True
)
```

---

### 4.2 Додати Fact Checker Agent
**Що зробити:**
1. Створити агента для перевірки фактів
2. Має перевіряти claims з аналізу через додатковий пошук
3. Відмічати claims як:
   - ✅ Verified (знайдено в 3+ джерелах)
   - ⚠️ Unverified (знайдено в 1-2 джерелах)
   - ❌ Disputed (суперечливі дані)
4. Додати у фінальний звіт секцію "Fact Check Summary"

---

## Завдання 5: Metrics та Monitoring

### 5.1 Додати збір метрик
**Файл:** новий `metrics.py`

**Що зробити:**
1. Створити клас `AnalysisMetrics`:
   ```python
   class AnalysisMetrics:
       def __init__(self):
           self.start_time = None
           self.end_time = None
           self.agents_used = []
           self.mcp_calls = 0
           self.api_calls = 0
           self.errors = []
   ```
2. Логувати всі виклики MCP tools
3. Вимірювати час кожного агента
4. Підраховувати вартість API calls
5. Зберігати метрики в JSON файл

---

### 5.2 Створити dashboard
**Файл:** новий `dashboard.py`

**Що зробити:**
1. Прочитати всі JSON файли з метриками
2. Вивести статистику:
   - Середній час аналізу
   - Найпопулярніші теми
   - Найчастіше використовувані агенти
   - Загальна вартість
3. Використати rich library для красивого виводу в консоль

**Приклад:**
```python
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="Analysis Statistics")
table.add_column("Metric", style="cyan")
table.add_column("Value", style="magenta")
# ... додати рядки
console.print(table)
```

---

## Завдання 6: Нові типи аналізу

### 6.1 Sentiment Analysis Agent
**Файл:** новий `sentiment_agent.py`

**Що зробити:**
1. Створити агента для аналізу тональності новин
2. Класифікувати кожне джерело:
   - Positive
   - Neutral
   - Negative
3. Побудувати загальний sentiment score (-1.0 до 1.0)
4. Додати візуалізацію у звіт

---

### 6.2 Trend Prediction Agent
**Що зробити:**
1. Агент аналізує тренди на основі історичних даних
2. Використовує Memory MCP для доступу до минулих аналізів
3. Прогнозує:
   - Чи буде тема популярнішою?
   - Які нові аспекти можуть з'явитися?
   - Рекомендації коли переглянути тему знову
4. Додає confidence score до прогнозів

---

## Завдання 7: Error Handling та Resilience

### 7.1 Додати retry механізм
**Файл:** `parallel_agent_with_mcp.py`

**Що зробити:**
1. Обгорнути `search_news()` у retry decorator
2. Максимум 3 спроби
3. Exponential backoff: 1s, 2s, 4s
4. Логувати кожну спробу
5. Якщо всі спроби failed - повернути "Source unavailable"

**Приклад:**
```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
            return wrapper
    return decorator

@retry(max_attempts=3, delay=1)
@tool("Brave News Search")
def search_news(query: str) -> str:
    ...
```

---

### 7.2 Graceful degradation
**Що зробити:**
1. Якщо один з агентів (BBC/CNN/Reuters) падає - продовжити з іншими
2. Якщо MCP сервер недоступний - використати звичайний аналіз без Sequential Thinking
3. Додати fallback на Google Search якщо Brave API не працює
4. У фінальному звіті відмічати які джерела недоступні

---

## Завдання 8: Тестування

### 8.1 Unit тести
**Файл:** `tests/test_agents.py`

**Що зробити:**
1. Тести для `search_news()`:
   ```python
   def test_search_news_success():
       result = search_news("AI")
       assert "результат" in result.lower()
       assert len(result) > 0
   ```
2. Тести для створення агентів
3. Мок Brave API для швидких тестів
4. Coverage мінімум 70%

---

### 8.2 Integration тести
**Файл:** `tests/test_integration.py`

**Що зробити:**
1. Тест повного flow (але з обмеженою темою)
2. Перевірка MCP підключення
3. Перевірка що всі агенти запускаються
4. Перевірка формату фінального звіту

---

## Завдання 9: Створення власного MCP сервера

### 9.1 Custom MCP Server для аналізу PDF
**Файл:** новий `mcp_servers/pdf_analyzer/`

**Що створити:**
1. MCP сервер який може:
   - Читати PDF файли
   - Витягувати текст
   - Знаходити ключові слова
   - Рахувати статистику (слова, сторінки)
2. Інтегрувати з CrewAI агентом
3. Використати для аналізу research papers

**Структура:**
```
mcp_servers/
└── pdf_analyzer/
    ├── index.js          # MCP server
    ├── package.json
    └── README.md
```

**package.json:**
```json
{
  "name": "mcp-pdf-analyzer",
  "version": "1.0.0",
  "type": "module",
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.0.0",
    "pdf-parse": "^1.1.1"
  }
}
```

---

### 9.2 Custom MCP Server для Database
**Що створити:**
1. MCP сервер для роботи з SQLite
2. Tools:
   - `query_database(sql)` - виконати SELECT
   - `get_schema()` - отримати схему БД
   - `get_table_stats(table)` - статистика таблиці
3. Створити агента який може відповідати на питання про дані

---

## Завдання 10: Deployment

### 10.1 Docker контейнеризація
**Файли:** `Dockerfile`, `docker-compose.yml`

**Що зробити:**
1. Dockerfile з Python та Node.js
2. docker-compose з сервісами:
   - app (основний агент)
   - postgres (для збереження результатів)
   - redis (для кешування API calls)
3. Volume для `/reports`
4. Environment variables через .env

**Dockerfile:**
```dockerfile
FROM node:18-slim AS node-base
FROM python:3.11-slim

# Копіюємо Node.js з першого образу
COPY --from=node-base /usr/local/bin/node /usr/local/bin/
COPY --from=node-base /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "parallel_agent_with_mcp.py"]
```

---

### 10.2 API endpoint
**Файл:** новий `api.py`

**Що зробити:**
1. FastAPI сервер з endpoint:
   ```
   POST /analyze
   {
     "topic": "AI breakthrough",
     "sources": ["bbc", "cnn", "reuters"],
     "use_mcp": true
   }
   ```
2. Async обробка запитів
3. WebSocket для streaming результатів
4. Rate limiting (10 requests/minute)
5. API key authentication

---

## Бонусні завдання ⭐

### Б1. Multi-language support
- Додати підтримку аналізу новин українською, російською, англійською
- Автоматичне визначення мови теми
- Пошук у відповідних джерелах

### Б2. Visualization
- Створити HTML звіт з графіками
- Sentiment timeline
- Source distribution pie chart
- Word cloud з ключових слів

### Б3. Telegram Bot
- Бот який приймає тему для аналізу
- Відправляє прогрес в реальному часі
- Фінальний звіт у вигляді файлу
- Команди: /analyze, /history, /stats

### Б4. CI/CD Pipeline
- GitHub Actions для запуску тестів
- Автоматичний build Docker image
- Deploy на cloud (Heroku/Railway/Render)
- Автоматичне оновлення при push

---

## Як здавати

1. Fork репозиторію https://github.com/agentspro/module4
2. Створити гілку для кожного завдання: `task-1-1`, `task-2-1`, etc.
3. Додати коментарі в код що саме змінили
4. Оновити README з описом змін
5. Запустити тести
6. Створити Pull Request

**Формат коміту:**
```
Task 1.1: Add Guardian news agent

- Added guardian_agent for searching Guardian news
- Created guardian_task for the agent
- Updated analysis to include 4 sources
- Updated synthesis to process 4 sources
- Tested with topic "climate change"
```

---

## Оцінювання

**Завдання 1-2:** Базовий рівень (по 5 балів)
**Завдання 3-4:** Середній рівень (по 10 балів)
**Завдання 5-7:** Складний рівень (по 15 балів)
**Завдання 8-9:** Експертний рівень (по 20 балів)
**Завдання 10:** Production рівень (30 балів)
**Бонусні:** по 10 балів

**Максимум: 250 балів + 40 бонусних**

**Оцінки:**
- 220+ балів: Відмінно
- 180-219: Добре
- 140-179: Задовільно
- <140: Потребує доопрацювання

---

## Корисні ресурси

- [CrewAI Documentation](https://docs.crewai.com/)
- [MCP Documentation](https://modelcontextprotocol.io/)
- [MCP Servers Repository](https://github.com/modelcontextprotocol/servers)
- [Brave Search API](https://brave.com/search/api/)
- [CrewAI Examples](https://github.com/crewAIInc/crewAI-examples)

**Успіхів!** 🚀
