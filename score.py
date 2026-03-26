import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def read_multiline_input(prompt):
    print(prompt);
    input = sys.stdin.read()
    return input

model_a_output = read_multiline_input("Enter the first LLM's prediction:")

baseline = read_multiline_input("Enter the baseline:")



question = read_multiline_input("Question: ")

prompt = f"""
You are an impartial judge evaluating the quality of AI response compared to a gold-standard baseline.

[Question]: {question}
[Model A Response]: {model_a_output}
[Baseline]: {baseline}

Evaluate both models based on:
1. Accuracy (compared to baseline)
2. Completeness
3. Clarity

Give each model a score out of 10 and provide a brief reasoning.
Output the result strictly in JSON format:
{{
    "Accuracy": "",
    "Completeness": "",
    "Clarity": ""
}}
"""

tokenizer = AutoTokenizer.from_pretrained("gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained("gpt-oss-20b-new", device_map="auto", offload_folder="offload", load_in_4bit=True)
eos_token_id = model.config.eos_token_id
print(eos_token_id)
messages = [{"role": "assistant", "content": prompt}]
inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
).to(model.device)
print(model.device)
outputs = model.generate(**inputs, max_new_tokens=500, eos_token_id=eos_token_id)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
print(response)
