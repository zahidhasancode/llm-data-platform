"""QLoRA fine-tuning of a small instruct model for text-to-SQL.

Runs on one 16 GB GPU (Kaggle T4): 4-bit NF4 base weights, LoRA adapters on all
linear layers, loss only on the assistant's SQL (not on the prompt).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from src.finetune.dataset import dataset_hash, load_split
from src.finetune.prompting import training_example


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", required=True, help="dataset version folder from the Spark job")
    ap.add_argument("--out", default="outputs/adapter")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    train = Dataset.from_list([training_example(r) for r in load_split(a.data, "train")])
    validation = Dataset.from_list([training_example(r) for r in load_split(a.data, "validation")])

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    fp16 = not torch.cuda.is_bf16_supported()  # T4: no bf16, so train in fp16
    compute = torch.float16 if fp16 else torch.bfloat16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=bnb, device_map={"": 0}, torch_dtype=compute, attn_implementation="sdpa"
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=a.rank,
        lora_alpha=2 * a.rank,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    # Train only on the answer: everything up to the assistant header is masked out of the loss.
    collator = DataCollatorForCompletionOnlyLM(response_template="<|im_start|>assistant\n", tokenizer=tok)

    cfg = SFTConfig(
        output_dir="outputs/checkpoints",
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        fp16=fp16,
        bf16=not fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_seq_length=a.max_len,
        packing=False,
        report_to="none",
        seed=a.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train,
        eval_dataset=validation,
        peft_config=lora,
        data_collator=collator,
        processing_class=tok,
    )
    trainer.model.print_trainable_parameters()

    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    out = Path(a.out)
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)

    Path("results").mkdir(exist_ok=True)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    info = {
        "base_model": a.model,
        "dataset_hash": dataset_hash(a.data),
        "train_examples": len(train),
        "val_examples": len(validation),
        "epochs": a.epochs,
        "learning_rate": a.lr,
        "lora_rank": a.rank,
        "effective_batch": a.batch * a.grad_accum,
        "max_len": a.max_len,
        "precision": "fp16" if fp16 else "bf16",
        "gpu": torch.cuda.get_device_name(0),
        "trainable_params": trainable,
        "train_minutes": round(minutes, 1),
        "log_history": trainer.state.log_history,
    }
    json.dump(info, open("results/train_log.json", "w"), indent=1)
    print(f"done in {minutes:.1f} min -> {out}")


if __name__ == "__main__":
    main()
