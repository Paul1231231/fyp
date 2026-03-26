# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from datasets import load_dataset
import os
import json

#initialize output dir
output_dir = "crop_json"
os.makedirs(output_dir, exist_ok=True)

def summarize(question, answer, model, tokenizer):
    question_answer = [{"role": "assistant", "content": "From the following chat history, extract only useful information for future conversation. Remove all explanations, commentary, or descriptive text, and keep only the actionable or useful content."}]
    text = "Question: " + question + "\n" + "Answer: " + answer
    question_answer.append({"role": "user", "content": text})
    inputs = tokenizer.apply_chat_template(
        question_answer,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=2560, eos_token_id = eos_token_id)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    response = response.split("<|channel|>final<|message|>")[1].split("<|return|>")[0]
    tokens = len(tokenizer.encode(response))
    return response, tokens

#import dataset
dataset = load_dataset("HuggingFaceH4/mt_bench_prompts")


tokenizer = AutoTokenizer.from_pretrained("gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained("gpt-oss-20b-new", device_map="auto", offload_folder="offload", load_in_8bit=True)
eos_token_id = model.config.eos_token_id
print(eos_token_id)

cropped_ratio = []
for i in range(40,50):
    n = len(dataset["train"][i]["prompt"])
    messages = [{"role": "assistant", "content": "You are a chatbot."}]
    conversation = []
    for j in range(2):
        #append the prompt to conversation(for output file) and message(for llm)
        user_input = dataset["train"][i]["prompt"][j]
        messages.append({"role": "user", "content": user_input})
        conversation.append({"role": "user", "content": user_input})

	#generating output
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=5000, eos_token_id=eos_token_id)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        response = response.split("<|channel|>final<|message|>")[1].split("<|return|>")[0]
        conversation.append({"role": "assistant", "content": response})
        tokens = len(tokenizer.encode(response))
	#get suppress ratio and feed cropped message to llm as reference
        response, crop_token = summarize(user_input, response, model, tokenizer)
        ratio = crop_token/tokens
        messages.pop()
        messages.append({"role": "assistant", "content": response})
        cropped_ratio.append(ratio)

    #output into json file
    file_name = f"question_{i}.json"
    filepath = os.path.join(output_dir, file_name)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation, f, indent=4)
    print(f"Done question {i}")
#output the ratio as a csv
import pandas as pd
df = pd.DataFrame(cropped_ratio, columns=["crop_ratio"])
df.to_csv('crop_5.csv')
