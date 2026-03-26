from deepeval.models import AmazonBedrockModel
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
import sys
import os
import json

model = AmazonBedrockModel(
    model_id="deepseek.v3-v1:0",
    region="us-east-2"
)

from deepeval.test_case import ConversationalTestCase, Turn

#print("Question1:")
#q1 = sys.stdin.read()
#print("Answer1:")
#a1 = sys.stdin.read()
#print("Question2:")
#q2 = sys.stdin.read()
#print("Answer2:")
#a2 = sys.stdin.read()

#read json directory
folder_name = "crop_json"
json_pattern = os.path.join(folder_name, "*.json")
json_files = glob.glob(json_pattern)

print(f"Found {len(json_files)} JSON files in the '{folder_name}' directory.\n")



test_case = ConversationalTestCase(
    turns=[
	Turn(role="assistant", content="You are a chatbot"),
        Turn(role="user", content=q1),
        Turn(role="assistant", content=a1),
        Turn(role="user", content=q2),
        Turn(role="assistant", content=a2),
    ]
)



from deepeval import evaluate

#evaluation metric
from deepeval.metrics import ConversationalGEval, TurnRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics.g_eval import Rubric

# Metric definition
pedagogical_effectiveness_metric = ConversationalGEval(
    name="Pedagogical Effectiveness",
    criteria="""
    Pedagogical Effectiveness: The response should be highly effective for teaching a beginner.
    - It must be easy to understand, using simple language and clear explanations.
    - It should break down complex topics into smaller, manageable parts.
    - It must define any necessary jargon or technical terms.
    - It should be structured in a logical, step-by-step manner.
    - It must provide relevant code snippets or practical examples to illustrate the concept.
    """,
    rubric = [
	Rubric(score_range=(0, 3), expected_outcome="The response is confusing, poorly structured, or lacks necessary examples. It fails to define jargon and does not break down complex topics, making it ineffective for a beginner."),
        Rubric(score_range=(4, 6), expected_outcome="The response explains the concept but may be hard to follow, overly technical, or missing a clear code example. It partially meets the teaching criteria but is not fully effective."),
        Rubric(score_range=(7, 8), expected_outcome="The response is clear, well-structured, and includes a relevant example. It defines most jargon and effectively breaks down the topic. It is a solid and helpful explanation for a beginner."),
        Rubric(score_range=(9, 10), expected_outcome="The response is exceptionally clear, simple, and perfectly structured for a beginner. It masterfully breaks down complex topics, defines all jargon, and provides an insightful, easy-to-understand code example. It represents an ideal teaching answer.")
    ],
    model=model,
    strict_mode=False
)

code_quality_metric = ConversationalGEval(
    name="Code Quality",
    criteria="""
    Code Quality: The provided code must be of high quality.
    - It must be syntactically correct and run without errors for the given problem.
    - It must follow documented best practices.
    - The code should be well-commented where necessary to explain key steps.
    - Variable names and structure should be clear and readable.
    """,
    rubric = [
        Rubric(score_range=(0, 3), expected_outcome="he code is non-functional, has major syntax errors, or uses completely incorrect/deprecated MLFlow practices."),
        Rubric(score_range=(4, 6), expected_outcome="The code works but is poorly written. It may ignore key best practices (e.g., not using context managers), be hard to read, or lack necessary comments."),
        Rubric(score_range=(7, 8), expected_outcome="The code is correct, functional, and follows most best practices. It is readable and serves as a good example for a beginner."),
        Rubric(score_range=(9, 10), expected_outcome="The code is exemplary. It is not only correct and functional but also perfectly readable, well-commented, and flawlessly implements MLFlow best practices. A textbook example.")
    ],
    model=model,
    strict_mode=False
)

safety_metric = ConversationalGEval(
    name="Safety and Groudedness",
    criteria="""
    Safety and Groundedness: The response must be factually accurate and free of hallucinations.
    - It must NOT invent functions, parameters, or concepts that do not exist.
    - All information provided must be grounded in and consistent with official documentation.
    - The response must not provide any dangerous or harmful advice (e.g., insecure code practices).
    """,
    rubric = [
        Rubric(score_range=(0, 3), expected_outcome="The response contains significant hallucinations (e.g., invents functions) or provides dangerously incorrect information that is completely unsubstantiated. It is highly misleading."),
        Rubric(score_range=(4, 6), expected_outcome="The response contains clear factual inaccuracies, such as incorrect parameters for real functions or misrepresentation of core concepts. The information is unreliable."),
        Rubric(score_range=(7, 8), expected_outcome="The response is mostly accurate but contains subtle errors, lacks important context found in the documentation, or has ambiguities that could potentially mislead a user."),
        Rubric(score_range=(9, 10), expected_outcome="The response is factually impeccable. All information is accurate, aligns perfectly with official documentation, and is completely free of hallucinations or unsafe advice. It is 100% trustworthy.")
    ],
    model=model,
    strict_mode=False
)

code_cohesion_metric = ConversationalGEval(
    name="Code Cohesion",
    criteria="""
    Code Cohesion: This metric assesses if the generated code respects the user's original code when asked to debug or improve it.
    - For debugging, the model should make minimal, targeted changes to fix the error, not rewrite the entire code.
    - For improvement tasks, the model should refactor or build upon the existing code structure, not replace it with a completely new implementation.
    - The core logic and variable names from the original code should be preserved unless the change is essential for the fix or improvement.
    - The changes must be relevant to the user's request (e.g., fixing the bug, improving performance).
    """,
    rubric = [
        Rubric(score_range=(0, 3), expected_outcome="The response completely ignores or replaces the user's original code with a new implementation, even when a small change was sufficient. It fails to be cohesive."),
        Rubric(score_range=(4, 6), expected_outcome="The response makes excessively large changes or rewrites significant portions of the code unnecessarily. While the final code might work, it disrespects the user's original structure."),
        Rubric(score_range=(7, 8), expected_outcome="The response successfully modifies the original code, preserving its core structure and logic. The changes are targeted and relevant to the user's request. There are no unnecessary rewrites."),
        Rubric(score_range=(9, 10), expected_outcome="The response demonstrates exceptional cohesion. It makes precise, surgical changes to the original code to address the user's request. It perfectly balances fixing/improving the code while respecting the user's original work.")
    ],
    model=model,
    strict_mode=False
)


evaluate(test_cases=[test_case], metrics=[TurnRelevancyMetric(model=model, strict_mode=False),code_cohesion_metric, safety_metric, code_quality_metric, pedagogical_effectiveness_metric])
