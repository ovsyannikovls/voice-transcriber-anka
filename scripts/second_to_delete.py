from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
model_path = os.path.join(ROOT_DIR, "models", "base", "qwen2_2B")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    local_files_only=True
)

if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

gen_kwargs = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": model.config.pad_token_id
}

while True:
    prompt = input("You: ")
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, **gen_kwargs)
    print("AI:", tokenizer.decode(output[0], skip_special_tokens=True))