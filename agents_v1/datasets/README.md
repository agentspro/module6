# 📁 Datasets Directory - Готові та згенеровані датасети

Ця директорія містить датасети у форматі JSON для evaluation AI агентів.

## ⚠️ Важливо про інструменти

LangSmith SDK має **офіційні методи** для роботи з датасетами:
- `client.upload_csv()` - імпорт з CSV ✅
- `client.upload_dataframe()` - імпорт з pandas DataFrame ✅

Наш `dataset_manager.py` додає функції, **яких немає в офіційному SDK**:
- Експорт в JSON (для бекапів) ⚠️
- Імпорт з JSON (для міграції) ⚠️
- LLM-генерація через код (SDK тільки UI) ⚠️

**Для CSV/DataFrame використовуйте офіційний SDK!**

Детальне порівняння: `../DATASET_TOOLS_COMPARISON.md`

---

## 📂 Структура

```
datasets/
├── README.md                              # Ця інструкція
├── research_agent_dataset.json            # Експорт поточного датасету
├── generated_langchain_example.json       # Приклад LLM-генерації
├── template_research_agent.json           # Шаблон для research агентів
├── template_qa_agent.json                 # Шаблон для Q&A агентів
└── (ваші датасети...)                     # Додайте свої!
```

---

## 🎯 Формат датасету

### Структура JSON файлу:

```json
{
  "name": "dataset-name",
  "description": "What this dataset tests",
  "created_at": "2025-11-17 ...",
  "examples_count": 4,
  "examples": [
    {
      "inputs": {
        "question": "Your test question"
      },
      "outputs": {
        "expected": "Expected answer or behavior"
      },
      "metadata": {
        "category": "type-of-question",
        "difficulty": "easy|medium|hard"
      }
    }
  ]
}
```

---

## 🔄 Як працювати з датасетами

### 1. Експорт існуючого датасету з LangSmith

```bash
python dataset_manager.py --action export \
  --name mcp-research-agent-eval-dataset \
  --output datasets/my_dataset.json
```

**Результат:** JSON файл з усіма examples

---

### 2. Імпорт датасету в LangSmith

```bash
python dataset_manager.py --action import \
  --input datasets/template_research_agent.json \
  --name my-imported-dataset
```

**Результат:** Новий датасет створено в LangSmith

---

### 3. Генерація датасету з LLM (автоматично!)

```bash
python dataset_manager.py --action generate \
  --topic "Python programming" \
  --count 10 \
  --name python-qa-dataset
```

**Що відбувається:**
- LLM створює 10 питань про Python
- LLM генерує очікувані відповіді
- Автоматично додає metadata (difficulty, category)
- Зберігає в LangSmith

**Приклад згенерованого питання:**
```json
{
  "question": "How can you use LangChain to create a chatbot...",
  "expected": "A good answer should explain the process...",
  "difficulty": "medium",
  "category": "practical"
}
```

---

### 4. Створення з готового шаблону

```bash
python dataset_manager.py --action template \
  --type research \
  --name my-research-dataset
```

**Доступні шаблони:**
- `research` - Для research агентів (search, analyze, synthesize)
- `qa` - Для Q&A агентів (definitions, explanations)
- `analysis` - Для аналітичних агентів (data analysis, predictions)
- `code` - Для code generation агентів

---

## 📚 Готові файли в цій директорії

### 1. `research_agent_dataset.json`

**Опис:** Реальний датасет з нашого MCP Research Agent

**Використання:**
```bash
# Імпортувати як новий датасет
python dataset_manager.py --action import \
  --input datasets/research_agent_dataset.json \
  --name my-version-of-research
```

**Що містить:**
- 4 test cases для research агента
- Питання про: trends, analysis, recommendations, comparisons
- Reference answers з описом очікуваної поведінки

---

### 2. `generated_langchain_example.json`

**Опис:** Приклад автоматично згенерованого датасету про LangChain

**Цікаво:**
- Питання створені LLM
- Expected answers також згенеровані
- Автоматична категоризація (factual/practical)
- Metadata з difficulty та category

**Як повторити:**
```bash
python dataset_manager.py --action generate \
  --topic "LangChain framework" \
  --count 5 \
  --name my-langchain-dataset
```

---

### 3. `template_research_agent.json`

**Опис:** Шаблон для створення research датасетів

**Використання:**
1. Відкрийте файл
2. Замініть `[YOUR TOPIC]`, `[FRAMEWORK A]` на ваші значення
3. Додайте 10-15 ваших examples
4. Імпортуйте:

```bash
python dataset_manager.py --action import \
  --input datasets/template_research_agent.json \
  --name customized-research-dataset
```

**Категорії в шаблоні:**
- trend-analysis
- comparison
- market-analysis
- recommendations
- multi-step

---

### 4. `template_qa_agent.json`

**Опис:** Шаблон для Q&A агентів

**Типи питань:**
- Definition: "What is X?"
- Explanation: "How does X work?"
- Recommendation: "Why use X?"
- When-to-use: "When is X appropriate?"
- Pros-cons: "Pros and cons of X?"
- Troubleshooting: "How to fix X?"

---

## 🛠️ Практичні сценарії

### Сценарій 1: Створити датасет для вашого агента

```bash
# Крок 1: Генерація з LLM
python dataset_manager.py --action generate \
  --topic "Your agent topic" \
  --count 15 \
  --name initial-dataset

# Крок 2: Експорт для редагування
python dataset_manager.py --action export \
  --name initial-dataset \
  --output datasets/my_dataset.json

# Крок 3: Редагуйте my_dataset.json вручну
#   - Покращте питання
#   - Уточніть expected answers
#   - Додайте metadata

# Крок 4: Імпорт назад
python dataset_manager.py --action import \
  --input datasets/my_dataset.json \
  --name final-dataset
```

---

### Сценарій 2: Розширення існуючого датасету

```bash
# Крок 1: Експорт існуючого
python dataset_manager.py --action export \
  --name existing-dataset \
  --output datasets/existing.json

# Крок 2: Генерація додаткових examples
python dataset_manager.py --action generate \
  --topic "Same topic" \
  --count 10 \
  --name additional-examples

# Крок 3: Експорт додаткових
python dataset_manager.py --action export \
  --name additional-examples \
  --output datasets/additional.json

# Крок 4: Вручну об'єднайте existing.json + additional.json

# Крок 5: Імпорт об'єднаного
python dataset_manager.py --action import \
  --input datasets/combined.json \
  --name expanded-dataset
```

---

### Сценарій 3: Створення версій датасету

```bash
# Baseline
python dataset_manager.py --action template \
  --type research \
  --name research-dataset-v1

# Додайте examples через LangSmith UI або скриптами

# Version 2: Більше складних кейсів
python dataset_manager.py --action generate \
  --topic "Advanced research scenarios" \
  --count 20 \
  --name research-dataset-v2

# Version 3: З edge cases
# Вручну створіть edge_cases.json
python dataset_manager.py --action import \
  --input datasets/edge_cases.json \
  --name research-dataset-v3
```

---

## 💡 Поради по створенню якісних датасетів

### 1. Різноманітність важливіша за кількість

```
✅ DO: 10 різних типів питань
❌ DON'T: 50 варіацій одного питання
```

### 2. Покривайте весь спектр складності

```json
{
  "easy": ["basic definitions", "simple queries"],
  "medium": ["analysis", "comparisons"],
  "hard": ["multi-step", "edge cases", "complex synthesis"]
}
```

### 3. Realistic expected answers

```
✅ DO: "Should discuss X, Y, and provide Z. Include examples."
❌ DON'T: "The exact answer is: [very specific text]"
```

### 4. Metadata допомагає організації

```json
{
  "metadata": {
    "category": "comparison",
    "difficulty": "hard",
    "requires_tools": ["search", "analyze"],
    "added_date": "2025-11-17",
    "why_important": "Tests multi-tool coordination"
  }
}
```

### 5. Версіонуйте датасети

```
dataset-v1-baseline.json
dataset-v2-with-edge-cases.json
dataset-v3-production-ready.json
```

---

## 🔄 Workflow: Від шаблону до production

```
1. Почніть з шаблону
   ↓
2. Генеруйте базові examples з LLM
   ↓
3. Експортуйте та редагуйте вручну
   ↓
4. Додайте edge cases
   ↓
5. Додайте domain-specific питання
   ↓
6. Тестуйте агента
   ↓
7. Додавайте failed cases до датасету
   ↓
8. Повторюйте 6-7 до досягнення quality targets
```

---

## 📊 Рекомендовані розміри датасетів

| Етап проекту | Кількість examples | Джерело |
|--------------|-------------------|---------|
| **Початок** | 5-10 | Шаблон + ручні |
| **Розробка** | 15-30 | LLM generation + ручні |
| **Testing** | 30-50 | Додати edge cases |
| **Production** | 50-100+ | Continuous: додавати failures |

---

## 🚀 Швидкий старт

```bash
# 1. Почніть з генерації
python dataset_manager.py --action generate \
  --topic "Your domain" \
  --count 10 \
  --name quick-start-dataset

# 2. Експортуйте
python dataset_manager.py --action export \
  --name quick-start-dataset \
  --output datasets/my_dataset.json

# 3. Редагуйте my_dataset.json

# 4. Імпортуйте назад
python dataset_manager.py --action import \
  --input datasets/my_dataset.json \
  --name final-dataset

# 5. Запустіть evaluation
python 05_mcp_research_agent_langsmith_eval.py
```

---

## 📖 Додаткові ресурси

- **Dataset Manager:** `../dataset_manager.py`
- **Evaluation Guide:** `../DATASET_GUIDE_FOR_STUDENTS.md`
- **Practice Script:** `../dataset_practice.py`
- **Cheatsheet:** `../DATASET_CHEATSHEET.md`

---

**Готові датасети = швидший старт evaluation! 🎉**
