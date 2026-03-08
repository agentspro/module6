#!/usr/bin/env python3
"""
DeepEval Evaluation for MCP Research Agent
==========================================

This script demonstrates comprehensive evaluation of the MCP Research Agent using DeepEval.

Evaluation Metrics:
------------------
1. Answer Relevancy: How relevant is the agent's answer to the question
2. Faithfulness: Does the answer stay true to the retrieved context
3. Contextual Relevancy: Is the retrieved context relevant to the question
4. Correctness: Overall quality of the response
5. Custom G-Eval: Task-specific evaluation criteria

Requirements:
-------------
pip install deepeval langchain langchain-openai python-dotenv

Setup:
------
1. Set OPENAI_API_KEY in .env
2. (Optional) Set DEEPEVAL_API_KEY for Confident AI integration
3. Run: python 05_mcp_research_agent_evaluation.py

Author: Claude Code
License: MIT
"""

import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not found in environment")
    print("Please set OPENAI_API_KEY in your .env file")
    sys.exit(1)

# ============================================================================
# DEEPEVAL IMPORTS
# ============================================================================

try:
    from deepeval import evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRelevancyMetric,
        GEval
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.dataset import EvaluationDataset
except ImportError:
    print("❌ ERROR: DeepEval not installed")
    print("Install with: pip install deepeval")
    sys.exit(1)

# ============================================================================
# IMPORT RESEARCH AGENT
# ============================================================================

# Import the research agent components
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# Copy tools from original file
@tool
def search_topic(query: str) -> str:
    """Search for information about a topic."""
    results = f"""Search Results for: {query}

📚 Found 3 relevant sources:

1. Academic Paper: "Recent Advances in {query}"
   - Published: 2024
   - Key finding: Significant progress in practical applications

2. Industry Report: "{query} Market Analysis"
   - Published: Q4 2024
   - Key finding: 40% YoY growth in adoption

3. Technical Blog: "Understanding {query}"
   - Published: Nov 2024
   - Key finding: Best practices and implementation patterns
"""
    return results

@tool
def analyze_data(topic: str) -> str:
    """Analyze data related to a topic."""
    analysis = f"""Data Analysis for: {topic}

📊 Key Metrics:
- Trend: ↗️ Upward (35% growth)
- Adoption: 67% of enterprises
- Satisfaction: 8.2/10

📈 Statistical Insights:
- Mean value: 42.5
- Std deviation: 12.3
- Correlation: 0.78 (strong positive)

🎯 Recommendations:
1. Focus on scalability improvements
2. Address integration challenges
3. Invest in training programs
"""
    return analysis

@tool
def synthesize_findings(findings: str) -> str:
    """Synthesize research findings into actionable insights."""
    synthesis = f"""Synthesis of Findings:

🎯 Key Takeaways:
1. Strong market momentum and adoption
2. Clear value proposition validated by data
3. Implementation challenges are well-documented

💡 Strategic Recommendations:
- Prioritize scalability and integration
- Develop comprehensive training materials
- Monitor emerging best practices

📝 Action Items:
1. Conduct deeper technical evaluation
2. Create implementation roadmap
3. Establish success metrics
"""
    return synthesis

# ============================================================================
# RESEARCH AGENT WRAPPER FOR EVALUATION
# ============================================================================

class ResearchAgentEvaluator:
    """Wrapper for research agent to facilitate evaluation."""

    def __init__(self):
        """Initialize research agent."""
        tools = [search_topic, analyze_data, synthesize_findings]

        self.agent = create_agent(
            model="gpt-4o-mini",
            tools=tools,
            system_prompt="""You are an expert research analyst with access to powerful tools.

You have access to:
- search_topic: Search for information about a topic
- analyze_data: Analyze data related to a topic
- synthesize_findings: Synthesize research findings into actionable insights

Use the appropriate tools for each request and provide thorough, comprehensive research analysis."""
        )

    def research(self, query: str) -> Dict[str, Any]:
        """
        Execute research query and return structured result.

        Args:
            query: Research question

        Returns:
            Dictionary with answer, context, and metadata
        """
        # Invoke agent
        result = self.agent.invoke({"messages": [("user", query)]})

        # Extract answer
        final_message = result["messages"][-1]
        answer = final_message.content

        # Extract context from tool calls (if any)
        contexts = []
        for msg in result["messages"]:
            if hasattr(msg, "content") and "Search Results" in str(msg.content):
                contexts.append(str(msg.content))
            elif hasattr(msg, "content") and "Data Analysis" in str(msg.content):
                contexts.append(str(msg.content))
            elif hasattr(msg, "content") and "Synthesis of Findings" in str(msg.content):
                contexts.append(str(msg.content))

        return {
            "answer": answer,
            "contexts": contexts,
            "raw_result": result
        }

# ============================================================================
# EVALUATION TEST CASES
# ============================================================================

def load_dataset_examples() -> List[Dict[str, Any]]:
    """
    Завантажити test cases з unified JSON dataset.

    Returns:
        Список прикладів з dataset
    """
    import json

    dataset_file = os.path.join(os.path.dirname(__file__), "datasets", "eval_dataset.json")

    if not os.path.exists(dataset_file):
        print(f"❌ Файл не знайдено: {dataset_file}")
        print("   Переконайтесь що datasets/eval_dataset.json існує")
        sys.exit(1)

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✅ Завантажено {len(data['examples'])} прикладів з {dataset_file}")
    return data["examples"]


def create_test_cases() -> List[LLMTestCase]:
    """
    Створити evaluation test cases з unified dataset.
    Читає питання з JSON, прогоняє агента для отримання actual_output та retrieval_context.

    Returns:
        List of LLMTestCase objects
    """

    print("\n" + "="*70)
    print("🧪 СТВОРЕННЯ EVALUATION TEST CASES З UNIFIED DATASET")
    print("="*70 + "\n")

    examples = load_dataset_examples()
    agent = ResearchAgentEvaluator()
    test_cases = []

    for i, example in enumerate(examples, 1):
        query = example["inputs"]["question"]
        expected = example["outputs"]["expected"]
        category = example.get("metadata", {}).get("category", "unknown")

        print(f"📝 Test Case {i}/{len(examples)}: [{category}] {query[:60]}...")
        result = agent.research(query)

        test_case = LLMTestCase(
            input=query,
            actual_output=result["answer"],
            retrieval_context=result["contexts"],
            expected_output=expected
        )
        test_cases.append(test_case)
        print(f"   ✅ Створено test case з {len(result['contexts'])} context chunks\n")

    print(f"✅ Створено {len(test_cases)} test cases\n")

    return test_cases

# ============================================================================
# EVALUATION METRICS SETUP
# ============================================================================

def setup_metrics() -> List:
    """
    Setup DeepEval metrics for evaluation.

    Returns:
        List of metric objects
    """

    print("="*70)
    print("📊 SETTING UP EVALUATION METRICS")
    print("="*70 + "\n")

    metrics = []

    # 1. Answer Relevancy Metric
    print("1️⃣  Answer Relevancy Metric")
    print("   Measures: How relevant is the answer to the question")
    print("   Threshold: ≥ 0.7 (70%)\n")
    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True
    )
    metrics.append(answer_relevancy)

    # 2. Faithfulness Metric
    print("2️⃣  Faithfulness Metric")
    print("   Measures: Does the answer stay true to retrieved context")
    print("   Threshold: ≥ 0.7 (70%)\n")
    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True
    )
    metrics.append(faithfulness)

    # 3. Contextual Relevancy Metric
    print("3️⃣  Contextual Relevancy Metric")
    print("   Measures: Is the retrieved context relevant to the question")
    print("   Threshold: ≥ 0.6 (60%)\n")
    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.6,
        model="gpt-4o-mini",
        include_reason=True
    )
    metrics.append(contextual_relevancy)

    # 4. Custom G-Eval Metric for Research Quality
    print("4️⃣  Custom G-Eval: Research Quality")
    print("   Measures: Overall research quality and completeness")
    print("   Threshold: ≥ 0.7 (70%)\n")
    research_quality = GEval(
        name="Research Quality",
        criteria="Evaluate the research quality based on: 1) Comprehensiveness of analysis, 2) Use of appropriate tools, 3) Clear actionable insights, 4) Proper synthesis of information",
        evaluation_steps=[
            "Check if multiple relevant sources were consulted",
            "Verify that data analysis was performed when appropriate",
            "Assess the clarity and actionability of recommendations",
            "Evaluate the logical flow and synthesis of information"
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT
        ],
        threshold=0.7,
        model="gpt-4o-mini"
    )
    metrics.append(research_quality)

    # 5. Custom G-Eval Metric for Tool Usage
    print("5️⃣  Custom G-Eval: Tool Usage Effectiveness")
    print("   Measures: How effectively the agent uses available tools")
    print("   Threshold: ≥ 0.7 (70%)\n")
    tool_usage = GEval(
        name="Tool Usage",
        criteria="Evaluate how effectively the agent uses available tools: 1) Appropriate tool selection, 2) Proper sequencing of tool calls, 3) Integration of tool outputs into final answer",
        evaluation_steps=[
            "Check if the right tools were selected for the task",
            "Verify tools were used in a logical sequence",
            "Assess how well tool outputs were integrated into the final answer"
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT
        ],
        threshold=0.7,
        model="gpt-4o-mini"
    )
    metrics.append(tool_usage)

    print(f"✅ Configured {len(metrics)} evaluation metrics\n")

    return metrics

# ============================================================================
# RUN EVALUATION
# ============================================================================

def run_evaluation():
    """Execute comprehensive evaluation of research agent."""

    print("\n" + "="*70)
    print("🎯 DEEPEVAL EVALUATION FOR MCP RESEARCH AGENT")
    print("="*70 + "\n")

    print("This evaluation assesses:")
    print("  ✓ Answer relevancy to questions")
    print("  ✓ Faithfulness to retrieved context")
    print("  ✓ Context relevancy to questions")
    print("  ✓ Research quality and completeness")
    print("  ✓ Tool usage effectiveness\n")

    # Create test cases
    test_cases = create_test_cases()

    # Setup metrics
    metrics = setup_metrics()

    # Run evaluation
    print("="*70)
    print("🚀 RUNNING EVALUATION")
    print("="*70 + "\n")

    print("⏳ Evaluating test cases... (this may take a few minutes)\n")

    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    # Display summary
    print("\n" + "="*70)
    print("📈 EVALUATION SUMMARY")
    print("="*70 + "\n")

    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Total Metrics: {len(metrics)}")
    print(f"\nOverall Results:")

    # Calculate pass rates for each metric
    metric_names = [
        "Answer Relevancy",
        "Faithfulness",
        "Contextual Relevancy",
        "Research Quality",
        "Tool Usage"
    ]

    for i, metric_name in enumerate(metric_names):
        passed = sum(1 for tc in test_cases if hasattr(tc, f'metric_{i}') and getattr(tc, f'metric_{i}', 0) >= metrics[i].threshold)
        total = len(test_cases)
        pass_rate = (passed / total * 100) if total > 0 else 0
        print(f"  {metric_name}: {passed}/{total} passed ({pass_rate:.1f}%)")

    print("\n" + "="*70)
    print("💡 NEXT STEPS")
    print("="*70 + "\n")

    print("1. Review detailed results above")
    print("2. Check metrics that didn't meet threshold")
    print("3. Analyze reasons for failures (included in detailed output)")
    print("4. Improve agent prompts or tools based on feedback")
    print("5. Re-run evaluation to measure improvements\n")

    # Optional: Integration with Confident AI
    if os.getenv("DEEPEVAL_API_KEY"):
        print("="*70)
        print("☁️  CONFIDENT AI INTEGRATION")
        print("="*70 + "\n")
        print("✅ Results will be uploaded to Confident AI")
        print("   View at: https://app.confident-ai.com\n")
    else:
        print("="*70)
        print("💡 TIP: CONFIDENT AI INTEGRATION")
        print("="*70 + "\n")
        print("Set DEEPEVAL_API_KEY to upload results to Confident AI")
        print("Get your key at: https://app.confident-ai.com\n")

    print("="*70)
    print("✅ EVALUATION COMPLETED")
    print("="*70 + "\n")

    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        results = run_evaluation()

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
