#!/usr/bin/env python3
"""
Upload Unified Dataset to LangSmith та LangFuse
================================================

Завантажує єдиний JSON-датасет на платформи evaluation.

Використання:
    python upload_dataset.py --platform langsmith langfuse
    python upload_dataset.py --platform langsmith --file datasets/eval_dataset.json
    python upload_dataset.py --platform langfuse --recreate

Параметри:
    --platform   Платформи для завантаження (langsmith, langfuse)
    --file       Шлях до JSON файлу (за замовчуванням: datasets/eval_dataset.json)
    --recreate   Видалити існуючий dataset і створити заново

Примітка: Phoenix Arize не підтримує datasets напряму — працює тільки з traces.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# ЗАВАНТАЖЕННЯ ДАТАСЕТУ З ФАЙЛУ
# ============================================================================

def load_dataset(file_path: str) -> dict:
    """Завантажити dataset з JSON файлу."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ Файл не знайдено: {file_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = ["name", "examples"]
    for key in required_keys:
        if key not in data:
            print(f"❌ Відсутній ключ '{key}' у JSON файлі")
            sys.exit(1)

    print(f"✅ Завантажено dataset: {data['name']}")
    print(f"   Кількість прикладів: {len(data['examples'])}")
    if data.get("description"):
        print(f"   Опис: {data['description']}")
    print()

    return data

# ============================================================================
# LANGSMITH UPLOAD
# ============================================================================

def upload_to_langsmith(data: dict, recreate: bool = False):
    """Завантажити dataset в LangSmith."""
    print("=" * 60)
    print("📤 ЗАВАНТАЖЕННЯ В LANGSMITH")
    print("=" * 60 + "\n")

    # Перевірка API ключів
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("❌ LANGCHAIN_API_KEY не знайдено в .env")
        print("   Отримайте ключ: https://smith.langchain.com")
        return False

    try:
        from langsmith import Client
    except ImportError:
        print("❌ langsmith не встановлено: pip install langsmith")
        return False

    client = Client()
    dataset_name = data["name"]

    # Перевірка існуючого dataset
    existing = None
    try:
        for ds in client.list_datasets():
            if ds.name == dataset_name:
                existing = ds
                break
    except Exception as e:
        print(f"⚠️  Помилка перевірки існуючих datasets: {e}")

    if existing:
        if recreate:
            print(f"🗑️  Видаляємо існуючий dataset '{dataset_name}'...")
            client.delete_dataset(dataset_id=existing.id)
            existing = None
        else:
            print(f"✅ Dataset '{dataset_name}' вже існує (ID: {existing.id})")
            print("   Використовуйте --recreate для перезавантаження")
            return True

    # Створення dataset
    print(f"📊 Створюємо dataset: {dataset_name}")
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=data.get("description", "")
    )

    # Додавання прикладів
    for i, example in enumerate(data["examples"], 1):
        client.create_example(
            inputs=example["inputs"],
            outputs=example["outputs"],
            metadata=example.get("metadata", {}),
            dataset_id=dataset.id
        )
        q = example["inputs"]["question"][:50]
        print(f"   ✅ [{i}/{len(data['examples'])}] {q}...")

    print(f"\n✅ Завантажено {len(data['examples'])} прикладів в LangSmith")
    print(f"   Перегляд: https://smith.langchain.com → Datasets & Testing → {dataset_name}\n")
    return True

# ============================================================================
# LANGFUSE UPLOAD
# ============================================================================

def upload_to_langfuse(data: dict, recreate: bool = False):
    """Завантажити dataset в LangFuse."""
    print("=" * 60)
    print("📤 ЗАВАНТАЖЕННЯ В LANGFUSE")
    print("=" * 60 + "\n")

    # Перевірка API ключів
    required_keys = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        print(f"❌ Відсутні змінні: {', '.join(missing)}")
        print("   Налаштуйте LANGFUSE_PUBLIC_KEY та LANGFUSE_SECRET_KEY в .env")
        return False

    try:
        from langfuse import Langfuse
    except ImportError:
        print("❌ langfuse не встановлено: pip install langfuse")
        return False

    langfuse = Langfuse()
    dataset_name = data["name"]

    # Перевірка існуючого dataset
    existing = None
    try:
        existing = langfuse.get_dataset(dataset_name)
    except Exception:
        pass  # Dataset не існує

    if existing and not recreate:
        print(f"✅ Dataset '{dataset_name}' вже існує в LangFuse")
        print("   Використовуйте --recreate для перезавантаження")
        return True

    # Створення dataset (LangFuse створює або повертає існуючий)
    if recreate and existing:
        print(f"⚠️  LangFuse не підтримує видалення datasets через API")
        print(f"   Додаємо нові items до існуючого dataset...")

    print(f"📊 Створюємо dataset: {dataset_name}")
    dataset = langfuse.create_dataset(
        name=dataset_name,
        description=data.get("description", "")
    )

    # Додавання items
    for i, example in enumerate(data["examples"], 1):
        dataset.create_item(
            input=example["inputs"],
            expected_output=example["outputs"],
            metadata=example.get("metadata", {})
        )
        q = example["inputs"]["question"][:50]
        print(f"   ✅ [{i}/{len(data['examples'])}] {q}...")

    langfuse.flush()

    print(f"\n✅ Завантажено {len(data['examples'])} прикладів в LangFuse")
    print(f"   Перегляд: LangFuse UI → Datasets → {dataset_name}\n")
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Завантаження eval dataset в LangSmith та/або LangFuse"
    )
    parser.add_argument(
        "--platform",
        nargs="+",
        choices=["langsmith", "langfuse"],
        default=["langsmith"],
        help="Платформи для завантаження (за замовчуванням: langsmith)"
    )
    parser.add_argument(
        "--file",
        default="datasets/eval_dataset.json",
        help="Шлях до JSON файлу з dataset (за замовчуванням: datasets/eval_dataset.json)"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Видалити існуючий dataset і створити заново"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("📦 UPLOAD DATASET — Unified Evaluation Dataset")
    print("=" * 60 + "\n")

    # Завантажити JSON
    data = load_dataset(args.file)

    results = {}

    # Завантаження на платформи
    if "langsmith" in args.platform:
        results["langsmith"] = upload_to_langsmith(data, args.recreate)

    if "langfuse" in args.platform:
        results["langfuse"] = upload_to_langfuse(data, args.recreate)

    # Підсумок
    print("=" * 60)
    print("📋 ПІДСУМОК")
    print("=" * 60 + "\n")

    for platform, success in results.items():
        status = "✅ Успішно" if success else "❌ Помилка"
        print(f"   {platform}: {status}")

    print()

    if all(results.values()):
        print("✅ Всі завантаження завершено успішно!\n")
    else:
        print("⚠️  Деякі завантаження не вдалися. Перевірте логи вище.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
