# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained("gpt-oss-20b-new", device_map="auto", offload_folder="offload", load_in_8bit=True)
eos_token_id = model.config.eos_token_id
print(eos_token_id)
messages = [{"role": "assistant", "content": "You are a useful assistant in teaching machine learning"}]


inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)
print(model.device)
print("\nChat with the model! Type 'exit' or 'quit' to end the conversation.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break

    messages.append({"role": "user", "content": user_input})
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1280, eos_token_id = eos_token_id)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    response = response.split("<|channel|>final<|message|>")[1].split("<|return|>")[0]
    print(f"Model: {response}")
    messages.append({"role": "assistant", "content": response})
