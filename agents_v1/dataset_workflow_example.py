#!/usr/bin/env python3
"""
Приклад комбінованого workflow: Офіційний SDK + dataset_manager.py
===================================================================

Цей скрипт показує як ефективно комбінувати:
1. Офіційний LangSmith SDK (для CSV/DataFrame)
2. dataset_manager.py (для експорту, LLM-генерації)

Сценарій:
---------
1. Маємо CSV з базовими питаннями
2. Імпортуємо через офіційний SDK (швидко і стабільно)
3. Додаємо LLM-генерацію для розширення
4. Експортуємо для бекапу та версіонування
5. Запускаємо evaluation

Author: Claude Code
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("LANGCHAIN_API_KEY"):
    print("❌ LANGCHAIN_API_KEY not found in .env")
    sys.exit(1)

from langsmith import Client
from langchain_openai import ChatOpenAI

# ============================================================================
# КРОК 1: Імпорт базових даних з CSV через ОФІЦІЙНИЙ SDK
# ============================================================================

def step1_import_csv_official():
    """
    ✅ ОФІЦІЙНИЙ SDK - найкраще для CSV/DataFrame

    Переваги:
    - Швидко (один метод)
    - Стабільно (офіційна підтримка)
    - Оптимізовано
    """

    print("\n" + "="*70)
    print("КРОК 1: Імпорт CSV через офіційний SDK")
    print("="*70 + "\n")

    client = Client()

    # Створимо приклад CSV якщо немає
    csv_file = "example_questions.csv"

    if not os.path.exists(csv_file):
        print("📝 Створюємо приклад CSV файлу...")

        df = pd.DataFrame({
            'question': [
                'What is LangSmith?',
                'How to create a dataset?',
                'What are evaluation metrics?'
            ],
            'expected_answer': [
                'LangSmith is a platform for developing, monitoring, and testing LLM applications',
                'Use client.create_dataset() or upload CSV/DataFrame',
                'Metrics include relevance, helpfulness, accuracy, and completeness'
            ],
            'difficulty': ['easy', 'medium', 'medium']
        })

        df.to_csv(csv_file, index=False)
        print(f"✅ Створено {csv_file}\n")

    # ✅ ОФІЦІЙНИЙ МЕТОД SDK
    print("📤 Завантаження CSV в LangSmith...")

    dataset = client.upload_csv(
        csv_file=csv_file,
        input_keys=['question'],           # Колонка для inputs
        output_keys=['expected_answer'],   # Колонка для outputs
        name="combined-workflow-dataset",
        description="Example of combined workflow: Official SDK + custom tools",
        data_type="kv"
    )

    print(f"✅ Датасет створено: {dataset.name}")
    print(f"   ID: {dataset.id}")
    print(f"   Examples: {dataset.example_count}")

    # Додати metadata з CSV (якщо потрібно)
    print("\n📝 Додавання metadata до examples...")

    examples = list(client.list_examples(dataset_id=dataset.id))
    df = pd.read_csv(csv_file)

    for i, example in enumerate(examples):
        if i < len(df):
            client.update_example(
                example_id=example.id,
                metadata={"difficulty": df.iloc[i]['difficulty']}
            )

    print(f"✅ Metadata додано до {len(examples)} examples")

    return dataset.name

# ============================================================================
# КРОК 2: Розширення датасету через LLM-генерацію (CUSTOM)
# ============================================================================

def step2_expand_with_llm(dataset_name: str):
    """
    ⚠️ CUSTOM ФУНКЦІЯ - офіційний SDK не має LLM-генерації через код

    Використовуємо dataset_manager.py functionality
    """

    print("\n" + "="*70)
    print("КРОК 2: Розширення через LLM-генерацію (custom)")
    print("="*70 + "\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Пропускаємо LLM-генерацію (OPENAI_API_KEY не знайдено)")
        return

    client = Client()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Знайти датасет
    datasets = list(client.list_datasets())
    dataset = next((d for d in datasets if d.name == dataset_name), None)

    if not dataset:
        print(f"❌ Dataset '{dataset_name}' not found")
        return

    print(f"🤖 Генерація додаткових examples з LLM...")
    print(f"   Тема: LangSmith evaluation best practices")
    print(f"   Кількість: 3 examples\n")

    # Генерація 3 додаткових examples
    topic = "LangSmith evaluation best practices"

    for i in range(3):
        print(f"Generating example {i+1}/3...", end=" ")

        prompt = f"""Generate a test case about {topic}.

Create a realistic question and expected answer for evaluating an AI agent.

Question should be:
- Specific and clear
- About {topic}
- Suitable for testing

Expected answer should describe what a good answer would contain (2-3 sentences).

Output ONLY valid JSON:
{{
    "question": "Your question here",
    "expected": "Description of expected answer",
    "difficulty": "easy|medium|hard",
    "category": "best-practices"
}}"""

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            import json
            data = json.loads(content)

            # ✅ ОФІЦІЙНИЙ МЕТОД SDK для додавання
            client.create_example(
                inputs={"question": data["question"]},
                outputs={"expected_answer": data["expected"]},
                metadata={
                    "difficulty": data.get("difficulty", "medium"),
                    "category": data.get("category", "best-practices"),
                    "generated": True,
                    "source": "llm"
                },
                dataset_id=dataset.id
            )

            print("✅")

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    print(f"\n✅ Додано 3 LLM-генерованих examples до датасету")

# ============================================================================
# КРОК 3: Експорт для бекапу (CUSTOM)
# ============================================================================

def step3_export_for_backup(dataset_name: str):
    """
    ⚠️ CUSTOM ФУНКЦІЯ - офіційний SDK не має експорту

    Використовуємо dataset_manager.py functionality
    """

    print("\n" + "="*70)
    print("КРОК 3: Експорт для бекапу (custom)")
    print("="*70 + "\n")

    client = Client()

    # Знайти датасет
    datasets = list(client.list_datasets())  # ✅ Офіційний метод
    dataset = next((d for d in datasets if d.name == dataset_name), None)

    if not dataset:
        print(f"❌ Dataset '{dataset_name}' not found")
        return None

    # Експорт (custom функціональність)
    examples = list(client.list_examples(dataset_name=dataset_name))  # ✅ Офіційний метод

    import json
    from datetime import datetime

    export_data = {
        "name": dataset.name,
        "description": dataset.description or "",
        "created_at": str(dataset.created_at),
        "examples_count": len(examples),
        "examples": []
    }

    for example in examples:
        export_data["examples"].append({
            "inputs": example.inputs,
            "outputs": example.outputs,
            "metadata": example.metadata or {}
        })

    # Зберегти
    os.makedirs("backups", exist_ok=True)
    backup_file = f"backups/{dataset_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Експортовано в: {backup_file}")
    print(f"   Examples: {len(examples)}")
    print(f"   Size: {os.path.getsize(backup_file)} bytes")

    return backup_file

# ============================================================================
# КРОК 4: Додавання з pandas DataFrame (ОФІЦІЙНИЙ SDK)
# ============================================================================

def step4_add_from_dataframe(dataset_name: str):
    """
    ✅ ОФІЦІЙНИЙ SDK - найкраще для pandas

    Показує як додати дані з DataFrame до існуючого датасету
    """

    print("\n" + "="*70)
    print("КРОК 4: Додавання даних з pandas DataFrame (офіційний SDK)")
    print("="*70 + "\n")

    client = Client()

    # Знайти датасет
    datasets = list(client.list_datasets())
    dataset = next((d for d in datasets if d.name == dataset_name), None)

    if not dataset:
        print(f"❌ Dataset '{dataset_name}' not found")
        return

    # Створити DataFrame з додатковими даними
    df = pd.DataFrame({
        'question': [
            'How to version datasets?',
            'Best practices for evaluation?'
        ],
        'expected_answer': [
            'Export datasets to JSON and store in git for version control',
            'Use diverse examples, cover edge cases, and iteratively improve based on failures'
        ],
        'difficulty': ['medium', 'hard'],
        'source': ['manual', 'manual']
    })

    print(f"📊 Додаємо {len(df)} examples з DataFrame...")

    # ✅ ОФІЦІЙНИЙ МЕТОД SDK - додавання до існуючого датасету
    for i, row in df.iterrows():
        client.create_example(
            inputs={"question": row['question']},
            outputs={"expected_answer": row['expected_answer']},
            metadata={
                "difficulty": row['difficulty'],
                "source": row['source']
            },
            dataset_id=dataset.id
        )
        print(f"   ✅ Added example {i+1}/{len(df)}")

    print(f"\n✅ Додано {len(df)} examples з DataFrame")

# ============================================================================
# КРОК 5: Перегляд фінального датасету
# ============================================================================

def step5_view_final_dataset(dataset_name: str):
    """
    Показати фінальний стан датасету
    """

    print("\n" + "="*70)
    print("КРОК 5: Фінальний стан датасету")
    print("="*70 + "\n")

    client = Client()

    # ✅ ОФІЦІЙНІ МЕТОДИ SDK
    datasets = list(client.list_datasets())
    dataset = next((d for d in datasets if d.name == dataset_name), None)

    if not dataset:
        print(f"❌ Dataset '{dataset_name}' not found")
        return

    examples = list(client.list_examples(dataset_name=dataset_name))

    print(f"📊 Dataset: {dataset.name}")
    print(f"   Description: {dataset.description}")
    print(f"   Total examples: {len(examples)}")

    # Статистика по джерелах
    sources = {}
    difficulties = {}

    for ex in examples:
        source = ex.metadata.get('source', 'csv') if ex.metadata else 'csv'
        diff = ex.metadata.get('difficulty', 'unknown') if ex.metadata else 'unknown'

        sources[source] = sources.get(source, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1

    print("\n📈 Statistics:")
    print(f"   By source:")
    for source, count in sources.items():
        print(f"      - {source}: {count}")

    print(f"\n   By difficulty:")
    for diff, count in difficulties.items():
        print(f"      - {diff}: {count}")

    print("\n💡 Датасет готовий для evaluation!")

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """
    Комбінований workflow:

    1. CSV → Офіційний SDK (швидко, стабільно)
    2. LLM-генерація → Custom код (автоматизація)
    3. Експорт → Custom код (бекап, версіонування)
    4. DataFrame → Офіційний SDK (додаткові дані)
    5. Перегляд → Офіційний SDK
    """

    print("\n" + "="*70)
    print("🚀 КОМБІНОВАНИЙ WORKFLOW: Офіційний SDK + Custom інструменти")
    print("="*70)

    try:
        # Крок 1: Імпорт з CSV (офіційний SDK)
        dataset_name = step1_import_csv_official()

        # Крок 2: LLM-генерація (custom)
        step2_expand_with_llm(dataset_name)

        # Крок 3: Експорт для бекапу (custom)
        backup_file = step3_export_for_backup(dataset_name)

        # Крок 4: Додавання з DataFrame (офіційний SDK)
        step4_add_from_dataframe(dataset_name)

        # Крок 5: Перегляд результату
        step5_view_final_dataset(dataset_name)

        # Фінальний summary
        print("\n" + "="*70)
        print("✅ WORKFLOW ЗАВЕРШЕНО")
        print("="*70 + "\n")

        print("📊 Що було зроблено:")
        print("   1. ✅ Імпортовано CSV через офіційний SDK")
        print("   2. ✅ Додано LLM-генерацію (custom)")
        print("   3. ✅ Експортовано бекап (custom)")
        print("   4. ✅ Додано DataFrame через офіційний SDK")

        if backup_file:
            print(f"\n💾 Бекап збережено: {backup_file}")

        print(f"\n🎯 Датасет готовий: {dataset_name}")
        print("\n📝 Наступні кроки:")
        print("   - Запустіть evaluation: python 05_mcp_research_agent_langsmith_eval.py")
        print("   - Перегляньте результати в LangSmith UI")
        print("   - Покращте датасет на основі failures")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(0)
