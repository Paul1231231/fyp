from deepeval.models import AmazonBedrockModel
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
import sys
import os
import json
import glob
import csv
import statistics

model = AmazonBedrockModel(
    model_id="deepseek.v3-v1:0",
    region="us-east-2"
)

from deepeval.test_case import ConversationalTestCase, Turn
#read json directory
folder_name = "crop_json/category_crop_4"
json_pattern = os.path.join(folder_name, "*.json")
json_files = glob.glob(json_pattern)

print(f"Found {len(json_files)} JSON files in the '{folder_name}' directory.\n")


from deepeval import evaluate

#evaluation metric
from deepeval.metrics import ConversationalGEval, TurnRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics.g_eval import Rubric
from deepeval.dataset import EvaluationDataset


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
    model=model,
    strict_mode=False,
    threshold=0.7
)

memorizing_ability_metric = ConversationalGEval(
    name="Memorizing ability",
    criteria="""
    Evaluate whether the assistant accurately remembers and uses important information from prior turns.

    Fail if ANY of the following occur:
    1) Missed required info: The response omits any important user-provided constraint, preference, fact, or request that is necessary to fully answer the current turn.
    2) Partial answer: The response answers only part of a multi-part question/request.
    3) Memory contradiction: The response conflicts with previously stated facts in the conversation.
    4) Unjustified assumptions: The response invents details not supported by prior turns or the current user message.
    5) Wrong prioritization: The response focuses on minor details while ignoring key requested items.

    Pass only if ALL are true:
    - Includes all critical prior-turn details needed for this turn.
    - Fully addresses every explicit sub-question/sub-task in the current turn.
    - Is consistent with conversation history.
    - Does not fabricate missing details.

    Scoring guidance:
    - If any required item is missing or any sub-question is unanswered => FAIL.
    - If response is relevant but incomplete => FAIL.
    - Only complete, consistent, non-fabricated answers => PASS.
    """,
    model=model,
    strict_mode=False,
    threshold=0.7
)

my_metrics=[
    TurnRelevancyMetric(model=model, strict_mode=False),
    code_cohesion_metric,
#    safety_metric,
    memorizing_ability_metric,
#    pedagogical_effectiveness_metric
]

all_test_cases = []
for file_path in json_files:
    print(f"--- Processing {file_path} ---")

    # read json file
    with open(file_path, 'r', encoding='utf-8') as file:
        conversation = json.load(file)

    # get the conversation
    q1 = conversation[0]['content']
    a1 = conversation[1]['content']
    q2 = conversation[2]['content']
    a2 = conversation[3]['content']
    test_case = ConversationalTestCase(
        turns=[
            Turn(role="assistant", content="You are a chatbot"),
            Turn(role="user", content=q1),
            Turn(role="assistant", content=a1),
            Turn(role="user", content=q2),
            Turn(role="assistant", content=a2),
        ]
    )

    all_test_cases.append(test_case)


results = evaluate(test_cases=all_test_cases,metrics=my_metrics)
print(results)


