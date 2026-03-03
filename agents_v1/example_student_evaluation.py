#!/usr/bin/env python3
"""
Приклад для студентів: Як створити evaluation з нуля
=====================================================

Цей файл показує покроково як:
1. Створити свій датасет
2. Додати test cases
3. Створити простих evaluators
4. Запустити evaluation
5. Переглянути результати

Використовуйте цей файл як шаблон для вашого проекту!

Author: Claude Code для студентів
License: Educational use
"""

import os
from dotenv import load_dotenv

# Завантажити environment variables
load_dotenv()

# ============================================================================
# ПЕРЕВІРКА НАЛАШТУВАНЬ
# ============================================================================

# Студенти: Переконайтеся що ці ключі є в вашому .env файлі!
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Додайте OPENAI_API_KEY в .env файл!")
    exit(1)

if not os.getenv("LANGCHAIN_API_KEY"):
    print("❌ Додайте LANGCHAIN_API_KEY в .env файл!")
    print("Отримайте на: https://smith.langchain.com")
    exit(1)

# Увімкнути LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "student-evaluation-project"

# ============================================================================
# IMPORTS
# ============================================================================

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# ============================================================================
# КРОК 1: СТВОРІТЬ СВОГО АГЕНТА
# ============================================================================

# Студенти: Це простий приклад. Замініть на вашого агента!

@tool
def calculate(expression: str) -> str:
    """
    Обчислює математичний вираз.

    Приклади:
    - "2 + 2" → "4"
    - "10 * 5" → "50"
    """
    try:
        result = eval(expression)  # В production використовуйте safe eval!
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_info(topic: str) -> str:
    """
    Повертає інформацію про тему.

    Студенти: Це заглушка. В реальному агенті тут буде API call.
    """
    info = {
        "python": "Python is a programming language created by Guido van Rossum in 1991.",
        "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
        "langchain": "LangChain is a framework for developing applications powered by LLMs."
    }

    return info.get(topic.lower(), f"No information available about {topic}")


def create_my_agent():
    """
    Створює агента з tools.

    Студенти: Налаштуйте system prompt під ваш use case!
    """
    tools = [calculate, get_info]

    agent = create_agent(
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="""You are a helpful assistant with access to tools.

When asked a question:
1. Think about which tool to use
2. Use the tool to get information
3. Provide a clear, concise answer

Always use tools when available instead of guessing."""
    )

    return agent


def my_agent_predict(inputs: dict) -> dict:
    """
    Функція для evaluation - отримує input, повертає output.

    LangSmith викликає цю функцію для кожного test case.
    """
    agent = create_my_agent()
    question = inputs["question"]

    # Викликаємо агента
    result = agent.invoke({"messages": [("user", question)]})

    # Витягуємо відповідь
    answer = result["messages"][-1].content

    return {"output": answer}


# ============================================================================
# КРОК 2: СТВОРІТЬ DATASET З TEST CASES
# ============================================================================

def create_my_dataset(client: Client):
    """
    Створює датасет з тестовими прикладами.

    Студенти: Змініть examples під ваш use case!
    """

    dataset_name = "student-evaluation-dataset"

    print("\n📊 CREATING DATASET\n")

    # Перевірка чи вже існує
    try:
        datasets = list(client.list_datasets())
        for ds in datasets:
            if ds.name == dataset_name:
                print(f"✅ Dataset '{dataset_name}' already exists\n")
                return dataset_name
    except:
        pass

    # Створення нового
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Student evaluation dataset - learning example"
    )

    # Студенти: Додайте ваші test cases тут!
    test_cases = [
        {
            # Test Case 1: Математичний запит
            "question": "What is 15 multiplied by 7?",
            "expected": "Should use calculate tool and return 105"
        },
        {
            # Test Case 2: Інформаційний запит
            "question": "Tell me about Python programming language",
            "expected": "Should use get_info tool and provide information about Python"
        },
        {
            # Test Case 3: Комбінований запит
            "question": "What is 20 + 30 and also tell me about AI",
            "expected": "Should use both calculate (result 50) and get_info (AI definition)"
        }
    ]

    # Додаємо examples до dataset
    for i, test_case in enumerate(test_cases, 1):
        client.create_example(
            inputs={"question": test_case["question"]},
            outputs={"expected": test_case["expected"]},
            dataset_id=dataset.id
        )
        print(f"✅ Added test case {i}: {test_case['question'][:50]}...")

    print(f"\n✅ Created dataset with {len(test_cases)} examples\n")

    return dataset_name


# ============================================================================
# КРОК 3: СТВОРІТЬ EVALUATORS
# ============================================================================

def create_my_evaluators():
    """
    Створює evaluators для оцінки якості відповідей.

    Студенти: Додайте власні evaluators під ваші потреби!
    """

    print("🔧 CREATING EVALUATORS\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Evaluator 1: Чи використав агент tools?
    def tool_usage_evaluator(run: Run, example: Example) -> dict:
        """
        Перевіряє чи агент використав tools.

        Студенти: Це важливо - агент має використовувати tools замість guessing!
        """
        # Перевіряємо trace чи були tool calls
        # В реальному коді ви б перевіряли run.child_runs
        answer = run.outputs["output"].lower()

        # Простий check: чи згадуються числа/факти що потребують tools?
        question = example.inputs["question"]

        if "multiply" in question or "+" in question:
            # Має бути число в відповіді
            has_calculation = any(char.isdigit() for char in answer)
            score = 1.0 if has_calculation else 0.0
        elif "tell me about" in question:
            # Має бути інформація
            score = 1.0 if len(answer) > 50 else 0.5
        else:
            score = 0.7  # Default

        return {
            "key": "tool_usage",
            "score": score,
            "comment": "Agent used tools appropriately" if score >= 0.7 else "Agent might not have used tools"
        }

    # Evaluator 2: Чи правильна відповідь?
    def correctness_evaluator(run: Run, example: Example) -> dict:
        """
        Оцінює правильність відповіді через LLM.

        Студенти: LLM-based evaluation корисний коли немає точної правильної відповіді.
        """
        question = example.inputs["question"]
        answer = run.outputs["output"]
        expected = example.outputs["expected"]

        prompt = f"""Question: {question}

Agent's Answer: {answer}

Expected behavior: {expected}

Is the agent's answer correct and appropriate?
Rate from 0.0 (completely wrong) to 1.0 (perfect).
Respond with ONLY a number between 0.0 and 1.0."""

        try:
            response = llm.invoke(prompt)
            score = float(response.content.strip())
            return {"key": "correctness", "score": max(0.0, min(1.0, score))}
        except:
            return {"key": "correctness", "score": 0.5}

    # Evaluator 3: Чи достатньо детальна відповідь?
    def detail_evaluator(run: Run, example: Example) -> dict:
        """
        Перевіряє чи відповідь достатньо детальна.

        Студенти: Занадто короткі відповіді часто неповні.
        """
        answer = run.outputs["output"]
        length = len(answer)

        # Оцінка на базі довжини
        if length < 20:
            score = 0.3  # Занадто коротко
        elif length < 50:
            score = 0.6  # Коротко
        elif length < 200:
            score = 1.0  # Добре
        else:
            score = 0.9  # Можливо занадто багато

        return {
            "key": "detail",
            "score": score,
            "comment": f"Answer length: {length} characters"
        }

    evaluators = [tool_usage_evaluator, correctness_evaluator, detail_evaluator]

    print("✅ Created 3 evaluators:")
    print("   1. Tool Usage - чи використовує агент tools")
    print("   2. Correctness - чи правильна відповідь")
    print("   3. Detail - чи достатньо детальна відповідь\n")

    return evaluators


# ============================================================================
# КРОК 4: ЗАПУСТІТЬ EVALUATION
# ============================================================================

def run_my_evaluation(client: Client, dataset_name: str, evaluators: list):
    """
    Запускає evaluation на вашому агенті.

    Студенти: Це core функція - тут відбувається magic!
    """

    print("="*70)
    print("🚀 RUNNING EVALUATION")
    print("="*70 + "\n")

    print(f"📊 Dataset: {dataset_name}")
    print(f"🔧 Evaluators: {len(evaluators)}")
    print(f"🎯 Target: my_agent_predict function\n")

    print("⏳ Running... (це може зайняти 1-2 хвилини)\n")

    # Запуск evaluation
    results = evaluate(
        my_agent_predict,                    # Ваша функція для тестування
        data=dataset_name,                    # Ваш dataset
        evaluators=evaluators,                # Ваші evaluators
        experiment_prefix="student-eval",     # Prefix для experiment name
        description="Student evaluation example - learning how to test AI agents"
    )

    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETED!")
    print("="*70 + "\n")

    return results


# ============================================================================
# КРОК 5: ПЕРЕГЛЯНЬТЕ РЕЗУЛЬТАТИ
# ============================================================================

def show_results_instructions():
    """
    Показує де переглянути результати.

    Студенти: LangSmith UI найкращий спосіб аналізувати results!
    """

    print("="*70)
    print("📈 HOW TO VIEW RESULTS")
    print("="*70 + "\n")

    print("🌐 Open LangSmith Dashboard:")
    print("   https://smith.langchain.com\n")

    print("📍 Navigate to:")
    print("   1. Click 'Datasets & Testing' in sidebar")
    print("   2. Find dataset: 'student-evaluation-dataset'")
    print("   3. Click on latest experiment run\n")

    print("👀 What you'll see:")
    print("   • Aggregate scores for each evaluator")
    print("   • Results for each test case")
    print("   • Full traces showing agent's steps")
    print("   • Comparison with previous runs\n")

    print("🔍 For failed tests:")
    print("   • Click on the test case")
    print("   • Click 'View Trace'")
    print("   • See exactly what agent did")
    print("   • Understand why it failed\n")

    print("💡 Next steps:")
    print("   1. Review all results")
    print("   2. Identify patterns in failures")
    print("   3. Improve your agent")
    print("   4. Re-run evaluation")
    print("   5. Compare before/after\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Головна функція - виконує весь workflow.

    Студенти: Запустіть цей файл і подивіться що відбувається!
    """

    print("\n" + "="*70)
    print("🎓 STUDENT EVALUATION EXAMPLE")
    print("   Learning how to test AI agents with LangSmith")
    print("="*70 + "\n")

    # Ініціалізація LangSmith client
    print("🔧 Initializing LangSmith client...")
    client = Client()
    print("✅ Connected to LangSmith\n")

    # Крок 1: Створити dataset
    dataset_name = create_my_dataset(client)

    # Крок 2: Створити evaluators
    evaluators = create_my_evaluators()

    # Крок 3: Запустити evaluation
    results = run_my_evaluation(client, dataset_name, evaluators)

    # Крок 4: Показати як переглянути results
    show_results_instructions()

    print("="*70)
    print("✅ DONE! Check LangSmith for detailed results")
    print("="*70 + "\n")

    print("💡 Tips for students:")
    print("   • Modify test_cases to match your agent")
    print("   • Create custom evaluators for your use case")
    print("   • Run evaluation after each improvement")
    print("   • Use traces to debug failures")
    print("   • Compare results over time\n")

    print("📚 For more help:")
    print("   • Read: DATASET_GUIDE_FOR_STUDENTS.md")
    print("   • Practice: dataset_practice.py")
    print("   • Reference: DATASET_CHEATSHEET.md\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
