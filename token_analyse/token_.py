from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

response  = sys.stdin.read()
tokenizer = AutoTokenizer.from_pretrained("gpt-oss-20b")
tokens = tokenizer.encode(response)
print("Number of token", len(tokens))
