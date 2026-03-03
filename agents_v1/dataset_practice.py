#!/usr/bin/env python3
"""
Практичний скрипт для студентів: Робота з датасетами
====================================================

Цей скрипт демонструє:
1. Як створити датасет
2. Як додати examples
3. Як оновити examples
4. Як видалити examples
5. Як переглянути датасет

Використання:
------------
python dataset_practice.py --action [create|add|update|view|delete]

Приклади:
--------
# Створити новий датасет
python dataset_practice.py --action create --name my-test-dataset

# Додати приклад
python dataset_practice.py --action add --name my-test-dataset

# Переглянути всі приклади
python dataset_practice.py --action view --name my-test-dataset

# Оновити приклад
python dataset_practice.py --action update --name my-test-dataset --example-id xxx

# Видалити приклад
python dataset_practice.py --action delete --name my-test-dataset --example-id xxx

Author: Claude Code для студентів
"""

import os
import sys
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Check API keys
if not os.getenv("LANGCHAIN_API_KEY"):
    print("❌ ERROR: LANGCHAIN_API_KEY not found in .env")
    print("Отримайте ключ на: https://smith.langchain.com")
    sys.exit(1)

from langsmith import Client

# ============================================================================
# ФУНКЦІЇ ДЛЯ РОБОТИ З ДАТАСЕТОМ
# ============================================================================

def create_dataset(client: Client, dataset_name: str):
    """
    Створює новий датасет.

    Для студентів:
    - Датасет = колекція тестових прикладів
    - Один датасет для одного агента/проекту
    - Можна створювати версії: my-dataset-v1, my-dataset-v2
    """

    print("\n" + "="*70)
    print("📊 СТВОРЕННЯ НОВОГО ДАТАСЕТУ")
    print("="*70 + "\n")

    # Перевірка чи існує
    try:
        existing = list(client.list_datasets())
        for ds in existing:
            if ds.name == dataset_name:
                print(f"⚠️  Dataset '{dataset_name}' вже існує!")
                print(f"   Використайте інше ім'я або видаліть існуючий")
                return
    except Exception as e:
        print(f"Warning: {e}")

    # Створення
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=f"Practice dataset created by student for learning evaluation"
    )

    print(f"✅ Створено датасет: '{dataset_name}'")
    print(f"   ID: {dataset.id}")
    print(f"\n📍 Переглянути: https://smith.langchain.com")
    print(f"   → Datasets & Testing → '{dataset_name}'\n")


def add_example(client: Client, dataset_name: str):
    """
    Додає новий приклад до датасету.

    Для студентів:
    - Example = один тестовий кейс
    - Складається з: input (питання) + output (очікувана відповідь)
    - Можна додавати необмежену кількість
    """

    print("\n" + "="*70)
    print("➕ ДОДАВАННЯ НОВОГО ПРИКЛАДУ")
    print("="*70 + "\n")

    # Знайти датасет
    try:
        datasets = list(client.list_datasets())
        dataset = next((d for d in datasets if d.name == dataset_name), None)

        if not dataset:
            print(f"❌ Dataset '{dataset_name}' не знайдено!")
            print(f"   Створіть його спочатку: --action create")
            return

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Інтерактивне введення
    print("Введіть дані для нового прикладу:\n")

    question = input("📝 Питання (input): ")
    if not question:
        print("❌ Питання не може бути порожнім!")
        return

    expected_answer = input("✅ Очікувана відповідь (output): ")
    if not expected_answer:
        print("❌ Відповідь не може бути порожньою!")
        return

    # Опціональні метадані
    print("\n💡 Опціонально - додати метадані (Enter щоб пропустити):")
    category = input("   Категорія (basic/complex/edge-case): ") or "general"
    difficulty = input("   Складність (easy/medium/hard): ") or "medium"

    # Додавання
    example = client.create_example(
        inputs={"question": question},
        outputs={"expected": expected_answer},
        metadata={
            "category": category,
            "difficulty": difficulty,
            "added_by": "student"
        },
        dataset_id=dataset.id
    )

    print(f"\n✅ Приклад додано!")
    print(f"   ID: {example.id}")
    print(f"   Question: {question[:50]}...")
    print(f"   Expected: {expected_answer[:50]}...")
    print(f"\n📍 Переглянути: https://smith.langchain.com\n")


def view_dataset(client: Client, dataset_name: str):
    """
    Показує всі приклади з датасету.

    Для студентів:
    - Перегляд всього, що є в датасеті
    - Можна побачити IDs для оновлення/видалення
    """

    print("\n" + "="*70)
    print("👀 ПЕРЕГЛЯД ДАТАСЕТУ")
    print("="*70 + "\n")

    # Знайти датасет
    try:
        datasets = list(client.list_datasets())
        dataset = next((d for d in datasets if d.name == dataset_name), None)

        if not dataset:
            print(f"❌ Dataset '{dataset_name}' не знайдено!")
            return

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print(f"Dataset: {dataset_name}")
    print(f"ID: {dataset.id}")
    print(f"Created: {dataset.created_at}\n")

    # Отримати приклади
    examples = list(client.list_examples(dataset_name=dataset_name))

    if not examples:
        print("📭 Датасет порожній. Додайте приклади: --action add")
        return

    print(f"📚 Знайдено {len(examples)} прикладів:\n")
    print("="*70)

    for i, example in enumerate(examples, 1):
        print(f"\n[Example #{i}]")
        print(f"ID: {example.id}")
        print(f"─────────────────────────────────────")

        # Input
        question = example.inputs.get('question', 'N/A')
        print(f"📝 Question: {question}")

        # Output
        expected = example.outputs.get('expected', 'N/A')
        print(f"✅ Expected: {expected}")

        # Metadata
        if example.metadata:
            print(f"🏷️  Metadata: {example.metadata}")

        print("="*70)

    print(f"\n💡 Для оновлення/видалення використовуйте ID")
    print(f"   Приклад: --action update --example-id {examples[0].id}\n")


def update_example(client: Client, dataset_name: str, example_id: str):
    """
    Оновлює існуючий приклад.

    Для студентів:
    - Коли виявили помилку в reference answer
    - Коли з'явилась нова інформація
    - Коли треба покращити якість тесту
    """

    print("\n" + "="*70)
    print("✏️  ОНОВЛЕННЯ ПРИКЛАДУ")
    print("="*70 + "\n")

    if not example_id:
        print("❌ Вкажіть --example-id для оновлення")
        print("   Подивіться IDs: --action view")
        return

    # Знайти example
    try:
        examples = list(client.list_examples(dataset_name=dataset_name))
        example = next((e for e in examples if str(e.id) == example_id), None)

        if not example:
            print(f"❌ Example з ID '{example_id}' не знайдено!")
            return

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Показати поточний стан
    print("Поточний стан:")
    print(f"Question: {example.inputs.get('question')}")
    print(f"Expected: {example.outputs.get('expected')}\n")

    # Нові дані
    print("Введіть нові дані (Enter щоб залишити без змін):\n")

    new_question = input("📝 Нове питання: ")
    new_expected = input("✅ Нова відповідь: ")

    # Підготовка update
    updated_inputs = example.inputs.copy()
    updated_outputs = example.outputs.copy()

    if new_question:
        updated_inputs['question'] = new_question
    if new_expected:
        updated_outputs['expected'] = new_expected

    # Оновлення
    client.update_example(
        example_id=example_id,
        inputs=updated_inputs,
        outputs=updated_outputs
    )

    print(f"\n✅ Приклад оновлено!")
    print(f"   ID: {example_id}\n")


def delete_example(client: Client, dataset_name: str, example_id: str):
    """
    Видаляє приклад з датасету.

    Для студентів:
    - Коли приклад більше не релевантний
    - Коли є дублікати
    - Коли тест занадто легкий/складний
    """

    print("\n" + "="*70)
    print("🗑️  ВИДАЛЕННЯ ПРИКЛАДУ")
    print("="*70 + "\n")

    if not example_id:
        print("❌ Вкажіть --example-id для видалення")
        print("   Подивіться IDs: --action view")
        return

    # Знайти example
    try:
        examples = list(client.list_examples(dataset_name=dataset_name))
        example = next((e for e in examples if str(e.id) == example_id), None)

        if not example:
            print(f"❌ Example з ID '{example_id}' не знайдено!")
            return

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Показати що буде видалено
    print("⚠️  Буде видалено:")
    print(f"Question: {example.inputs.get('question')}")
    print(f"Expected: {example.outputs.get('expected')}\n")

    # Підтвердження
    confirm = input("Підтвердіть видалення (yes/no): ")

    if confirm.lower() not in ['yes', 'y']:
        print("❌ Видалення скасовано")
        return

    # Видалення
    client.delete_example(example_id=example_id)

    print(f"\n✅ Приклад видалено!")
    print(f"   ID: {example_id}\n")


def list_all_datasets(client: Client):
    """Показує всі доступні датасети."""

    print("\n" + "="*70)
    print("📚 ВСІ ДОСТУПНІ ДАТАСЕТИ")
    print("="*70 + "\n")

    try:
        datasets = list(client.list_datasets())

        if not datasets:
            print("📭 Датасетів не знайдено")
            print("   Створіть новий: --action create")
            return

        print(f"Знайдено {len(datasets)} датасетів:\n")

        for i, dataset in enumerate(datasets, 1):
            print(f"{i}. {dataset.name}")
            print(f"   ID: {dataset.id}")
            print(f"   Created: {dataset.created_at}")

            # Кількість прикладів
            examples = list(client.list_examples(dataset_id=dataset.id))
            print(f"   Examples: {len(examples)}")
            print()

    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Практичний скрипт для роботи з LangSmith датасетами",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:

  # Переглянути всі датасети
  python dataset_practice.py --action list

  # Створити новий датасет
  python dataset_practice.py --action create --name my-test-dataset

  # Додати приклад
  python dataset_practice.py --action add --name my-test-dataset

  # Переглянути приклади
  python dataset_practice.py --action view --name my-test-dataset

  # Оновити приклад
  python dataset_practice.py --action update --name my-test-dataset --example-id xxx

  # Видалити приклад
  python dataset_practice.py --action delete --name my-test-dataset --example-id xxx

Для навчання:
  1. Створіть тестовий датасет
  2. Додайте 3-5 прикладів
  3. Спробуйте оновити/видалити
  4. Перегляньте в LangSmith UI
        """
    )

    parser.add_argument(
        '--action',
        required=True,
        choices=['create', 'add', 'view', 'update', 'delete', 'list'],
        help='Дія для виконання'
    )

    parser.add_argument(
        '--name',
        help='Назва датасету'
    )

    parser.add_argument(
        '--example-id',
        help='ID прикладу для оновлення/видалення'
    )

    args = parser.parse_args()

    # Initialize client
    print("\n🔧 Ініціалізація LangSmith client...")
    client = Client()
    print("✅ З'єднано з LangSmith\n")

    # Execute action
    if args.action == 'list':
        list_all_datasets(client)

    elif args.action == 'create':
        if not args.name:
            print("❌ Вкажіть --name для створення датасету")
            return
        create_dataset(client, args.name)

    elif args.action == 'add':
        if not args.name:
            print("❌ Вкажіть --name датасету")
            return
        add_example(client, args.name)

    elif args.action == 'view':
        if not args.name:
            print("❌ Вкажіть --name датасету")
            return
        view_dataset(client, args.name)

    elif args.action == 'update':
        if not args.name:
            print("❌ Вкажіть --name датасету")
            return
        update_example(client, args.name, args.example_id)

    elif args.action == 'delete':
        if not args.name:
            print("❌ Вкажіть --name датасету")
            return
        delete_example(client, args.name, args.example_id)

    print("\n" + "="*70)
    print("✅ ГОТОВО!")
    print("="*70)
    print("\n💡 Переглянути результати: https://smith.langchain.com")
    print("   → Datasets & Testing\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Перервано користувачем")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
