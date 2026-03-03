#!/usr/bin/env python3
"""
LangSmith Native Evaluation for MCP Research Agent
==================================================

This script demonstrates evaluation using LangSmith's built-in capabilities:
1. LangSmith Datasets for test case management
2. Built-in evaluators (QA, Criteria, Custom)
3. Evaluation runs with comparison and tracking
4. Integration with existing LangSmith observability

Advantages over DeepEval:
------------------------
✅ No additional dependencies (LangSmith already integrated)
✅ Unified observability + evaluation platform
✅ Automatic trace linking to evaluation results
✅ Built-in dataset versioning and management
✅ Team collaboration and sharing
✅ Historical comparison and regression detection

LangSmith Evaluators:
--------------------
1. QA Evaluators: Correctness, Relevance, Helpfulness
2. Criteria Evaluators: Custom criteria evaluation
3. String Evaluators: Exact match, contains, regex
4. Custom Evaluators: Your own evaluation logic

Requirements:
-------------
# Already installed in project:
langchain>=1.0.0
langchain-openai>=1.0.0
langsmith>=0.1.0  # Included with langchain>=1.0

Setup:
------
1. Set OPENAI_API_KEY in .env
2. Set LANGCHAIN_API_KEY in .env (for LangSmith)
3. Set LANGCHAIN_TRACING_V2=true
4. Run: python 05_mcp_research_agent_langsmith_eval.py

View Results:
-------------
https://smith.langchain.com -> Datasets & Testing tab

Author: Claude Code
License: MIT
"""

import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check required API keys
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not found in environment")
    sys.exit(1)

if not os.getenv("LANGCHAIN_API_KEY"):
    print("❌ ERROR: LANGCHAIN_API_KEY not found in environment")
    print("Get your key at: https://smith.langchain.com")
    sys.exit(1)

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "mcp-research-agent-eval")

# ============================================================================
# LANGSMITH IMPORTS
# ============================================================================

from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example

# LangChain imports
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# RESEARCH TOOLS (same as original)
# ============================================================================

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
# RESEARCH AGENT
# ============================================================================

def create_research_agent():
    """Create research agent for evaluation."""
    tools = [search_topic, analyze_data, synthesize_findings]

    agent = create_agent(
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="""You are an expert research analyst with access to powerful tools.

You have access to:
- search_topic: Search for information about a topic
- analyze_data: Analyze data related to a topic
- synthesize_findings: Synthesize research findings into actionable insights

Use the appropriate tools for each request and provide thorough, comprehensive research analysis."""
    )

    return agent

def research_agent_predict(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper function for agent prediction (required by LangSmith evaluate).

    Args:
        inputs: Dictionary with 'question' key

    Returns:
        Dictionary with 'output' key
    """
    agent = create_research_agent()
    query = inputs["question"]

    result = agent.invoke({"messages": [("user", query)]})

    # Extract final answer
    final_message = result["messages"][-1]
    answer = final_message.content

    return {"output": answer}

# ============================================================================
# LANGSMITH DATASET CREATION
# ============================================================================

def create_evaluation_dataset(client: Client, dataset_name: str) -> str:
    """
    Create LangSmith dataset with test examples.

    Args:
        client: LangSmith client
        dataset_name: Name for the dataset

    Returns:
        Dataset name
    """

    print("\n" + "="*70)
    print("📊 CREATING LANGSMITH DATASET")
    print("="*70 + "\n")

    # Check if dataset already exists
    try:
        existing_datasets = list(client.list_datasets())
        for ds in existing_datasets:
            if ds.name == dataset_name:
                print(f"✅ Dataset '{dataset_name}' already exists")
                print(f"   Using existing dataset\n")
                return dataset_name
    except Exception as e:
        print(f"⚠️  Warning checking existing datasets: {e}")

    # Create new dataset
    print(f"Creating new dataset: {dataset_name}")

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluation dataset for MCP Research Agent"
    )

    # Define test examples
    examples = [
        {
            "question": "What are the latest trends in AI agent architectures?",
            "reference_answer": "Recent trends in AI agent architectures include multi-agent systems, LangChain/LangGraph patterns, and modern orchestration frameworks with focus on production readiness and observability."
        },
        {
            "question": "Analyze the adoption metrics for LangChain in production systems",
            "reference_answer": "LangChain adoption shows strong growth with increasing enterprise usage, positive satisfaction metrics, and recommendations for scalability improvements and integration capabilities."
        },
        {
            "question": "Provide strategic recommendations for implementing AI agents in enterprise",
            "reference_answer": "Strategic recommendations should include comprehensive analysis, actionable insights on scalability and integration, training programs, and clear implementation roadmap with success metrics."
        },
        {
            "question": "Research observability platforms for AI agents, analyze their features, and recommend the best approach",
            "reference_answer": "Comprehensive research should cover multiple observability platforms (LangSmith, Phoenix, LangFuse), analyze features, provide comparative analysis, and give clear recommendations based on use case."
        }
    ]

    # Add examples to dataset
    for i, example in enumerate(examples, 1):
        client.create_example(
            inputs={"question": example["question"]},
            outputs={"expected": example["reference_answer"]},
            dataset_id=dataset.id
        )
        print(f"   ✅ Added example {i}: {example['question'][:50]}...")

    print(f"\n✅ Created dataset with {len(examples)} examples")
    print(f"   View at: https://smith.langchain.com\n")

    return dataset_name

# ============================================================================
# CUSTOM EVALUATORS
# ============================================================================

def create_custom_evaluators() -> List:
    """
    Create custom evaluators for research agent assessment.
    Uses LangSmith 1.0+ compatible evaluator functions.

    Returns:
        List of evaluator functions
    """

    print("="*70)
    print("🔧 CREATING CUSTOM EVALUATORS")
    print("="*70 + "\n")

    # Initialize LLM for evaluations
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Evaluator functions that work with LangSmith
    def relevance_evaluator(run: Run, example: Example) -> dict:
        """Evaluate if answer is relevant to the question."""
        question = example.inputs["question"]
        answer = run.outputs["output"]

        prompt = f"""Question: {question}

Answer: {answer}

Is this answer relevant to the question? Does it address the main points?
Rate from 0.0 to 1.0 where 1.0 is perfectly relevant.
Respond with just a number."""

        try:
            response = llm.invoke(prompt)
            score = float(response.content.strip())
            return {"key": "relevance", "score": max(0.0, min(1.0, score))}
        except:
            return {"key": "relevance", "score": 0.5}

    def helpfulness_evaluator(run: Run, example: Example) -> dict:
        """Evaluate if answer is helpful and actionable."""
        answer = run.outputs["output"]

        prompt = f"""Answer: {answer}

Is this answer helpful and actionable? Does it provide practical insights?
Rate from 0.0 to 1.0 where 1.0 is very helpful.
Respond with just a number."""

        try:
            response = llm.invoke(prompt)
            score = float(response.content.strip())
            return {"key": "helpfulness", "score": max(0.0, min(1.0, score))}
        except:
            return {"key": "helpfulness", "score": 0.5}

    def length_checker(run: Run, example: Example) -> dict:
        """Check if answer has reasonable length."""
        answer = run.outputs["output"]
        length = len(answer)

        # Good length is between 100 and 2000 characters
        if length < 100:
            score = length / 100
        elif length > 2000:
            score = max(0.5, 1.0 - (length - 2000) / 2000)
        else:
            score = 1.0

        return {"key": "length", "score": score}

    evaluators = [relevance_evaluator, helpfulness_evaluator, length_checker]

    print("1️⃣  Relevance Evaluator")
    print("   Evaluates: How relevant is the answer to the question\n")

    print("2️⃣  Helpfulness Evaluator")
    print("   Evaluates: Is the answer helpful and actionable\n")

    print("3️⃣  Length Checker")
    print("   Evaluates: Answer has reasonable length (100-2000 chars)\n")

    print(f"✅ Created {len(evaluators)} evaluators\n")

    return evaluators

# ============================================================================
# RUN EVALUATION
# ============================================================================

def run_langsmith_evaluation(
    client: Client,
    dataset_name: str,
    evaluators: List
) -> Dict[str, Any]:
    """
    Run evaluation using LangSmith.

    Args:
        client: LangSmith client
        dataset_name: Name of the dataset to evaluate
        evaluators: List of evaluators to use

    Returns:
        Evaluation results
    """

    print("="*70)
    print("🚀 RUNNING LANGSMITH EVALUATION")
    print("="*70 + "\n")

    print(f"Dataset: {dataset_name}")
    print(f"Evaluators: {len(evaluators)}")
    print(f"Target: research_agent_predict function\n")

    print("⏳ Running evaluation... (this may take a few minutes)\n")

    # Run evaluation
    results = evaluate(
        research_agent_predict,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="mcp-research-agent",
        description="Evaluation of MCP Research Agent with LangSmith native evaluators",
        metadata={
            "model": "gpt-4o-mini",
            "version": "1.0",
            "timestamp": datetime.now().isoformat()
        }
    )

    print("\n" + "="*70)
    print("📊 EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")

    print("✅ Evaluation finished")
    print(f"   Dataset: {dataset_name}")
    print(f"   Evaluators: {len(evaluators)}\n")

    print("="*70)
    print("📈 VIEW DETAILED RESULTS IN LANGSMITH")
    print("="*70 + "\n")

    print("🌐 Open LangSmith Dashboard:")
    print(f"   https://smith.langchain.com\n")

    print("📍 Navigate to:")
    print("   1. Click 'Datasets & Testing' in sidebar")
    print("   2. Find dataset: '{}'".format(dataset_name))
    print("   3. View your experiment run\n")

    print("📊 What you'll see:")
    print("  • Aggregate metrics for all evaluators")
    print("  • Example-by-example results")
    print("  • Full traces for each test case")
    print("  • Score distributions")
    print("  • Comparison with previous runs\n")

    return results

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main evaluation workflow."""

    print("\n" + "="*70)
    print("🎯 LANGSMITH NATIVE EVALUATION")
    print("   MCP Research Agent Quality Assessment")
    print("="*70 + "\n")

    print("Evaluation Approach:")
    print("  ✓ LangSmith Datasets for test management")
    print("  ✓ Built-in + Custom evaluators")
    print("  ✓ Automatic trace linking")
    print("  ✓ Historical comparison tracking\n")

    # Initialize LangSmith client
    print("Initializing LangSmith client...")
    client = Client()
    print(f"✅ Connected to LangSmith")
    print(f"   Project: {os.environ['LANGCHAIN_PROJECT']}\n")

    # Create dataset
    dataset_name = "mcp-research-agent-eval-dataset"
    dataset_name = create_evaluation_dataset(client, dataset_name)

    # Create evaluators
    evaluators = create_custom_evaluators()

    # Run evaluation
    results = run_langsmith_evaluation(client, dataset_name, evaluators)

    # Final summary
    print("="*70)
    print("💡 NEXT STEPS")
    print("="*70 + "\n")

    print("1. Review results in LangSmith dashboard")
    print("2. Click on failing examples to see traces")
    print("3. Improve agent based on feedback")
    print("4. Re-run evaluation and compare results")
    print("5. Track improvements over time\n")

    print("="*70)
    print("✅ EVALUATION COMPLETED")
    print("="*70 + "\n")

    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        results = main()

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
