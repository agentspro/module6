#!/bin/bash
# setup_phoenix_venv.sh
#
# Автоматичне створення окремого Python 3.13 venv для Phoenix Arize
# Phoenix підтримує тільки Python 3.9-3.13 (не 3.14)

set -e  # Exit on error

echo "🔧 Phoenix Arize Setup for Python 3.13"
echo "========================================"
echo ""

# Перевірка наявності Python 3.13
if ! command -v python3.13 &> /dev/null; then
    echo "❌ Python 3.13 не знайдено на вашій системі"
    echo ""
    echo "Встановіть Python 3.13:"
    echo "  macOS:  brew install python@3.13"
    echo "  Ubuntu: sudo apt install python3.13 python3.13-venv"
    exit 1
fi

echo "✅ Python 3.13 знайдено: $(which python3.13)"
echo ""

# Створення venv
VENV_NAME="venv_phoenix"

if [ -d "$VENV_NAME" ]; then
    echo "⚠️  venv_phoenix вже існує"
    read -p "Видалити та пересворити? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Видалення старого venv..."
        rm -rf "$VENV_NAME"
    else
        echo "❌ Скасовано"
        exit 0
    fi
fi

echo "📦 Створення Python 3.13 venv..."
python3.13 -m venv "$VENV_NAME"

echo "✅ venv створено: $VENV_NAME"
echo ""

# Активація venv
echo "🔌 Активація venv..."
source "$VENV_NAME/bin/activate"

# Оновлення pip
echo "⬆️  Оновлення pip..."
pip install --upgrade pip --quiet

# Встановлення базових залежностей
echo "📚 Встановлення базових залежностей..."
pip install -r requirements.txt --quiet

# Встановлення Phoenix
echo "🐦 Встановлення Phoenix Arize..."
pip install arize-phoenix openinference-instrumentation-langchain --quiet

# Встановлення LangFuse (опціонально)
echo "📊 Встановлення LangFuse..."
pip install langfuse --quiet

echo ""
echo "✅ Налаштування завершено!"
echo ""
echo "📝 Наступні кроки:"
echo ""
echo "1️⃣  Активуйте новий venv:"
echo "    source venv_phoenix/bin/activate"
echo ""
echo "2️⃣  Запустіть Phoenix server (Термінал 1):"
echo "    python -m phoenix.server.main serve"
echo ""
echo "3️⃣  Запустіть агента з Phoenix (Термінал 2):"
echo "    source venv_phoenix/bin/activate"
echo "    python 05_mcp_research_agent_multi_observability.py --observability phoenix"
echo ""
echo "4️⃣  Або з усіма трьома платформами:"
echo "    python 05_mcp_research_agent_multi_observability.py --observability langsmith phoenix langfuse"
echo ""
echo "🌐 Phoenix Dashboard: http://localhost:6006"
echo ""
echo "💡 Для повернення до Python 3.14 venv:"
echo "    deactivate"
echo "    source venv/bin/activate"
echo ""
