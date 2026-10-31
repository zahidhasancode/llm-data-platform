"""Score the base model and the fine-tuned model on the same held-out test split.

Both use the same 4-bit weights, prompt and greedy decoding, so the only difference is the adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.finetune.dataset import dataset_hash, load_split
from src.finetune.prompting import messages
from src.finetune.scoring import score


def load_model(base: str, adapter: str | None):
    compute = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map={"": 0}, torch_dtype=compute
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.no_grad()
def generate(model, tok, conversations: list, batch: int, max_new: int) -> list[str]:
    outputs = []
    for i in range(0, len(conversations), batch):
        texts = [
            tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations[i : i + batch]
        ]
        encoded = tok(texts, return_tensors="pt", padding=True).to(model.device)
        generated = model.generate(
            **encoded, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id
        )
        outputs += tok.batch_decode(
            generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="outputs/adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=256)
    args = parser.parse_args()

    rows = load_split(args.data, "test")
    conversations = [messages(r["context"], r["question"]) for r in rows]
    report = {
        "dataset_hash": dataset_hash(args.data),
        "base_model": args.model,
        "decoding": "greedy",
        "gpu": torch.cuda.get_device_name(0),
        "conditions": {},
    }
    Path("results").mkdir(exist_ok=True)
    for name, adapter in (("base", None), ("qlora", args.adapter)):
        model, tok = load_model(args.model, adapter)
        metrics, predictions = score(rows, generate(model, tok, conversations, args.batch, args.max_new))
        report["conditions"][name] = metrics
        with open(f"results/predictions_{name}.jsonl", "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(name, json.dumps(metrics), flush=True)
        del model
        torch.cuda.empty_cache()
    Path("results/metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
