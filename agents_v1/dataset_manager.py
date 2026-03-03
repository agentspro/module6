#!/usr/bin/env python3
"""
Dataset Manager - Експорт, Імпорт, Генерація датасетів
======================================================

Цей скрипт показує як:
1. Експортувати існуючі датасети в JSON (CUSTOM - немає в офіційному SDK)
2. Імпортувати датасети з JSON (CUSTOM - SDK має тільки CSV/DataFrame)
3. Генерувати датасети автоматично з LLM (CUSTOM - SDK має тільки UI)
4. Створювати датасети з шаблонів (CUSTOM - немає в SDK)

ВАЖЛИВО:
--------
LangSmith SDK має офіційні методи для:
- client.upload_csv() - імпорт з CSV
- client.upload_dataframe() - імпорт з pandas DataFrame
- client.create_dataset() - створення датасету
- client.create_examples() - додавання examples

Цей скрипт доповнює SDK функціями, яких там немає:
- Експорт датасетів в JSON (для бекапів та версіонування)
- Імпорт з JSON (для міграції та відновлення)
- Автоматична генерація через LLM (програмний доступ)
- Шаблони для швидкого старту

Для CSV/DataFrame краще використовувати офіційний SDK!

Використання:
------------
# Експорт існуючого датасету
python dataset_manager.py --action export --name mcp-research-agent-eval-dataset --output dataset.json

# Імпорт датасету з JSON
python dataset_manager.py --action import --input dataset.json --name new-dataset

# Генерація нового датасету з LLM
python dataset_manager.py --action generate --topic "Python programming" --count 10

# Створення з шаблону
python dataset_manager.py --action template --type research --name my-research-dataset

Author: Claude Code
Див. також: DATASET_TOOLS_COMPARISON.md для порівняння з офіційним SDK
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("LANGCHAIN_API_KEY"):
    print("❌ LANGCHAIN_API_KEY not found in .env")
    sys.exit(1)

from langsmith import Client
from langchain_openai import ChatOpenAI

# ============================================================================
# EXPORT - Експорт датасету в JSON
# ============================================================================

def export_dataset(client: Client, dataset_name: str, output_file: str):
    """
    Експортує датасет в JSON файл.

    ⚠️ CUSTOM ФУНКЦІЯ - офіційний SDK не має експорту в JSON!

    Офіційний SDK має:
    - client.list_datasets() ✅ (використовуємо)
    - client.list_examples() ✅ (використовуємо)
    Але НЕМАЄ методу експорту в файл.

    Формат:
    {
        "name": "dataset-name",
        "description": "...",
        "created_at": "...",
        "examples": [...]
    }
    """

    print(f"\n📤 EXPORTING DATASET: {dataset_name}\n")

    # Знайти датасет (використовуємо офіційний SDK)
    try:
        datasets = list(client.list_datasets())  # ✅ Офіційний метод SDK
        dataset = next((d for d in datasets if d.name == dataset_name), None)

        if not dataset:
            print(f"❌ Dataset '{dataset_name}' not found!")
            print("\nAvailable datasets:")
            for ds in datasets:
                print(f"  - {ds.name}")
            return

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Отримати всі examples (використовуємо офіційний SDK)
    examples = list(client.list_examples(dataset_name=dataset_name))  # ✅ Офіційний метод SDK

    # Підготувати дані для експорту
    export_data = {
        "name": dataset.name,
        "description": dataset.description or "",
        "created_at": str(dataset.created_at),
        "examples_count": len(examples),
        "examples": []
    }

    # Додати кожен example
    for example in examples:
        export_data["examples"].append({
            "inputs": example.inputs,
            "outputs": example.outputs,
            "metadata": example.metadata or {}
        })

    # Зберегти в JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Exported {len(examples)} examples to: {output_file}")
    print(f"\n📄 File structure:")
    print(f"   Name: {export_data['name']}")
    print(f"   Examples: {export_data['examples_count']}")
    print(f"   Size: {os.path.getsize(output_file)} bytes\n")

# ============================================================================
# IMPORT - Імпорт датасету з JSON
# ============================================================================

def import_dataset(client: Client, input_file: str, new_name: str = None):
    """
    Імпортує датасет з JSON файлу.

    ⚠️ CUSTOM ФУНКЦІЯ - офіційний SDK має тільки CSV/DataFrame!

    Офіційний SDK має:
    - client.upload_csv() ✅ (для CSV файлів)
    - client.upload_dataframe() ✅ (для pandas DataFrame)
    Але НЕМАЄ upload_json()!

    Для CSV/DataFrame краще використовувати офіційні методи:

    ```python
    # Для CSV:
    dataset = client.upload_csv(
        csv_file='data.csv',
        input_keys=['question'],
        output_keys=['answer'],
        name="my-dataset"
    )

    # Для DataFrame:
    import pandas as pd
    df = pd.read_csv('data.csv')
    dataset = client.upload_dataframe(
        df=df,
        input_keys=['question'],
        output_keys=['answer'],
        name="my-dataset"
    )
    ```
    """

    print(f"\n📥 IMPORTING DATASET from: {input_file}\n")

    # Читання JSON
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    # Назва датасету
    dataset_name = new_name or data.get("name", f"imported-dataset-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    # Перевірка чи існує
    try:
        existing = list(client.list_datasets())
        if any(d.name == dataset_name for d in existing):
            print(f"⚠️  Dataset '{dataset_name}' already exists!")
            overwrite = input("Overwrite? (yes/no): ")
            if overwrite.lower() not in ['yes', 'y']:
                print("❌ Import cancelled")
                return
    except:
        pass

    # Створення датасету (офіційний SDK)
    dataset = client.create_dataset(  # ✅ Офіційний метод SDK
        dataset_name=dataset_name,
        description=data.get("description", f"Imported from {input_file}")
    )

    print(f"✅ Created dataset: {dataset_name}")

    # Додавання examples (офіційний SDK)
    examples = data.get("examples", [])
    print(f"📝 Importing {len(examples)} examples...")

    for i, ex in enumerate(examples, 1):
        client.create_example(  # ✅ Офіційний метод SDK
            inputs=ex.get("inputs", {}),
            outputs=ex.get("outputs", {}),
            metadata=ex.get("metadata", {}),
            dataset_id=dataset.id
        )
        print(f"   ✅ Added example {i}/{len(examples)}")

    print(f"\n✅ Import completed! Dataset: {dataset_name}\n")

# ============================================================================
# GENERATE - Автоматична генерація датасету з LLM
# ============================================================================

def generate_dataset(client: Client, topic: str, count: int, dataset_name: str = None):
    """
    Генерує датасет автоматично використовуючи LLM.

    ⚠️ CUSTOM ФУНКЦІЯ - офіційний SDK має генерацію тільки через UI!

    LangSmith UI має кнопку "Generate synthetic examples", але SDK не має
    програмного API для цього. Ця функція дозволяє генерувати через код,
    що корисно для:
    - Автоматизації в CI/CD
    - Batch generation
    - Кастомні промпти для генерації

    Використовує:
    - client.create_dataset() ✅ (офіційний SDK)
    - client.create_example() ✅ (офіційний SDK)
    - ChatOpenAI для генерації контенту (custom логіка)
    """

    print(f"\n🤖 GENERATING DATASET with LLM\n")
    print(f"Topic: {topic}")
    print(f"Count: {count} examples\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required for generation")
        return

    # Ініціалізація LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Назва датасету
    if not dataset_name:
        safe_topic = topic.replace(" ", "-").lower()[:30]
        dataset_name = f"generated-{safe_topic}-{datetime.now().strftime('%Y%m%d')}"

    # Створення датасету (офіційний SDK)
    dataset = client.create_dataset(  # ✅ Офіційний метод SDK
        dataset_name=dataset_name,
        description=f"Auto-generated dataset about {topic} using LLM"
    )

    print(f"✅ Created dataset: {dataset_name}\n")
    print("🔄 Generating examples with LLM...\n")

    # Генерація examples
    for i in range(count):
        print(f"Generating example {i+1}/{count}...", end=" ")

        # Промпт для LLM
        generation_prompt = f"""Generate a test case for evaluating an AI agent about {topic}.

Create a realistic question and expected answer.

Question should be:
- Specific and clear
- Relevant to {topic}
- Suitable for testing AI agent capabilities

Expected answer should be:
- Describe what good answer would contain
- Not exact words, but general expectations
- 2-3 sentences

Output ONLY valid JSON in this format:
{{
    "question": "Your question here",
    "expected": "Description of expected answer",
    "difficulty": "easy|medium|hard",
    "category": "factual|analytical|practical"
}}"""

        try:
            # Генерація
            response = llm.invoke(generation_prompt)
            content = response.content.strip()

            # Парсинг JSON
            # Видалити markdown code blocks якщо є
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            example_data = json.loads(content)

            # Додавання до датасету (офіційний SDK)
            client.create_example(  # ✅ Офіційний метод SDK
                inputs={"question": example_data["question"]},
                outputs={"expected": example_data["expected"]},
                metadata={
                    "difficulty": example_data.get("difficulty", "medium"),
                    "category": example_data.get("category", "general"),
                    "generated": True,
                    "topic": topic
                },
                dataset_id=dataset.id
            )

            print("✅")

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    print(f"\n✅ Generated {count} examples in dataset: {dataset_name}\n")

# ============================================================================
# TEMPLATE - Створення з шаблону
# ============================================================================

def create_from_template(client: Client, template_type: str, dataset_name: str):
    """
    Створює датасет з готового шаблону.

    ⚠️ CUSTOM ФУНКЦІЯ - офіційний SDK не має шаблонів!

    Використовує офіційні методи SDK:
    - client.create_dataset() ✅
    - client.create_example() ✅

    Але додає бібліотеку готових шаблонів для різних типів агентів.

    Шаблони:
    - research: Для research агентів
    - qa: Для Q&A агентів
    - analysis: Для аналітичних агентів
    - code: Для code generation агентів
    """

    print(f"\n📋 CREATING FROM TEMPLATE: {template_type}\n")

    # Шаблони
    templates = {
        "research": {
            "description": "Template for research agents with search and analysis",
            "examples": [
                {
                    "inputs": {"question": "What are the latest trends in AI?"},
                    "outputs": {"expected": "Should discuss recent developments, trends, and future directions in AI"},
                    "metadata": {"category": "trend-analysis", "difficulty": "medium"}
                },
                {
                    "inputs": {"question": "Compare framework X vs framework Y"},
                    "outputs": {"expected": "Should provide balanced comparison covering features, pros/cons, use cases"},
                    "metadata": {"category": "comparison", "difficulty": "hard"}
                },
                {
                    "inputs": {"question": "Analyze the market for technology Z"},
                    "outputs": {"expected": "Should include market size, growth, key players, trends"},
                    "metadata": {"category": "market-analysis", "difficulty": "hard"}
                }
            ]
        },
        "qa": {
            "description": "Template for Q&A agents",
            "examples": [
                {
                    "inputs": {"question": "What is X?"},
                    "outputs": {"expected": "Clear definition of X with examples"},
                    "metadata": {"category": "definition", "difficulty": "easy"}
                },
                {
                    "inputs": {"question": "How does X work?"},
                    "outputs": {"expected": "Step-by-step explanation of how X works"},
                    "metadata": {"category": "explanation", "difficulty": "medium"}
                },
                {
                    "inputs": {"question": "Why should I use X?"},
                    "outputs": {"expected": "Benefits, use cases, and when X is appropriate"},
                    "metadata": {"category": "recommendation", "difficulty": "medium"}
                }
            ]
        },
        "analysis": {
            "description": "Template for analytical agents",
            "examples": [
                {
                    "inputs": {"question": "Analyze data for X"},
                    "outputs": {"expected": "Statistical analysis, trends, patterns, insights"},
                    "metadata": {"category": "data-analysis", "difficulty": "hard"}
                },
                {
                    "inputs": {"question": "What are pros and cons of X?"},
                    "outputs": {"expected": "Balanced analysis of advantages and disadvantages"},
                    "metadata": {"category": "pros-cons", "difficulty": "medium"}
                },
                {
                    "inputs": {"question": "Predict future trends for X"},
                    "outputs": {"expected": "Data-driven predictions with reasoning"},
                    "metadata": {"category": "prediction", "difficulty": "hard"}
                }
            ]
        },
        "code": {
            "description": "Template for code generation agents",
            "examples": [
                {
                    "inputs": {"question": "Write a function to X"},
                    "outputs": {"expected": "Working code with proper syntax, error handling, documentation"},
                    "metadata": {"category": "code-generation", "difficulty": "medium"}
                },
                {
                    "inputs": {"question": "Debug this code: [code snippet]"},
                    "outputs": {"expected": "Identification of bugs and corrected code"},
                    "metadata": {"category": "debugging", "difficulty": "hard"}
                },
                {
                    "inputs": {"question": "Explain how this code works"},
                    "outputs": {"expected": "Clear explanation of code logic and flow"},
                    "metadata": {"category": "explanation", "difficulty": "medium"}
                }
            ]
        }
    }

    if template_type not in templates:
        print(f"❌ Unknown template type: {template_type}")
        print(f"\nAvailable templates: {', '.join(templates.keys())}")
        return

    template = templates[template_type]

    # Створення датасету (офіційний SDK)
    dataset = client.create_dataset(  # ✅ Офіційний метод SDK
        dataset_name=dataset_name,
        description=template["description"]
    )

    print(f"✅ Created dataset: {dataset_name}")
    print(f"   Type: {template_type}")
    print(f"   Description: {template['description']}\n")

    # Додавання examples з шаблону
    print(f"📝 Adding {len(template['examples'])} examples from template...")

    for i, ex in enumerate(template['examples'], 1):
        client.create_example(  # ✅ Офіційний метод SDK
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            metadata=ex["metadata"],
            dataset_id=dataset.id
        )
        print(f"   ✅ Added example {i}: {ex['inputs']['question'][:50]}...")

    print(f"\n✅ Template dataset created: {dataset_name}")
    print(f"\n💡 Customize by adding more examples specific to your use case\n")

# ============================================================================
# VIEW - Перегляд датасету
# ============================================================================

def view_dataset_details(client: Client, dataset_name: str):
    """Показує детальну інформацію про датасет."""

    print(f"\n👀 VIEWING DATASET: {dataset_name}\n")

    # Знайти датасет
    try:
        datasets = list(client.list_datasets())
        dataset = next((d for d in datasets if d.name == dataset_name), None)

        if not dataset:
            print(f"❌ Dataset '{dataset_name}' not found!")
            return

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Інформація про датасет
    print(f"Name: {dataset.name}")
    print(f"ID: {dataset.id}")
    print(f"Description: {dataset.description or 'N/A'}")
    print(f"Created: {dataset.created_at}\n")

    # Examples
    examples = list(client.list_examples(dataset_name=dataset_name))
    print(f"Examples: {len(examples)}\n")

    if examples:
        print("="*70)
        for i, ex in enumerate(examples, 1):
            print(f"\n[Example #{i}]")
            print(f"ID: {ex.id}")
            print(f"Input: {json.dumps(ex.inputs, indent=2, ensure_ascii=False)}")
            print(f"Output: {json.dumps(ex.outputs, indent=2, ensure_ascii=False)}")
            if ex.metadata:
                print(f"Metadata: {json.dumps(ex.metadata, indent=2, ensure_ascii=False)}")
            print("="*70)

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Manager - Export, Import, Generate datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Export existing dataset
  python dataset_manager.py --action export --name mcp-research-agent-eval-dataset --output my_dataset.json

  # Import dataset from JSON
  python dataset_manager.py --action import --input my_dataset.json --name imported-dataset

  # Generate dataset with LLM
  python dataset_manager.py --action generate --topic "Python programming" --count 5 --name python-qa-dataset

  # Create from template
  python dataset_manager.py --action template --type research --name my-research-dataset

  # View dataset
  python dataset_manager.py --action view --name my-dataset
        """
    )

    parser.add_argument('--action', required=True,
                       choices=['export', 'import', 'generate', 'template', 'view'],
                       help='Action to perform')

    parser.add_argument('--name', help='Dataset name')
    parser.add_argument('--output', help='Output file for export')
    parser.add_argument('--input', help='Input file for import')
    parser.add_argument('--topic', help='Topic for generation')
    parser.add_argument('--count', type=int, default=5, help='Number of examples to generate')
    parser.add_argument('--type', help='Template type (research/qa/analysis/code)')

    args = parser.parse_args()

    # Initialize client
    print("\n🔧 Initializing LangSmith client...")
    client = Client()
    print("✅ Connected\n")

    # Execute action
    if args.action == 'export':
        if not args.name or not args.output:
            print("❌ --name and --output required for export")
            return
        export_dataset(client, args.name, args.output)

    elif args.action == 'import':
        if not args.input:
            print("❌ --input required for import")
            return
        import_dataset(client, args.input, args.name)

    elif args.action == 'generate':
        if not args.topic:
            print("❌ --topic required for generation")
            return
        generate_dataset(client, args.topic, args.count, args.name)

    elif args.action == 'template':
        if not args.type or not args.name:
            print("❌ --type and --name required for template")
            return
        create_from_template(client, args.type, args.name)

    elif args.action == 'view':
        if not args.name:
            print("❌ --name required for view")
            return
        view_dataset_details(client, args.name)

    print("✅ Done!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
