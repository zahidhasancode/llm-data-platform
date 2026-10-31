"""Load the base model with its adapter on CPU, then serve it."""

from __future__ import annotations

import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.finetune.prompting import messages
from src.finetune.sqlcheck import extract_sql
from src.serving.app import create_app

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "/models/adapter")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()


@torch.no_grad()
def generate(context: str, question: str) -> str:
    text = tokenizer.apply_chat_template(
        messages(context, question), tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(text, return_tensors="pt")
    output = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return extract_sql(tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True))


app = create_app(generate, os.environ.get("MODEL_VERSION", "unknown"))
