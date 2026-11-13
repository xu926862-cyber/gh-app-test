#!/usr/bin/env python
# QLoRA fine-tune of a small Qwen2.5-Coder model on repo Q&A data.
# Designed for a free Colab T4 (16 GB VRAM); CPU is not practical for 3B.
#
# Usage:
#   pip install -r requirements.txt
#   python train_lora.py                      # defaults: Qwen2.5-Coder-3B-Instruct
#   BASE_MODEL=Qwen/Qwen2.5-Coder-0.5B-Instruct python train_lora.py   # tiny smoke test
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", "data/sample.jsonl")
OUT_DIR = os.environ.get("OUT_DIR", "out/lora-repo-bot")
EPOCHS = int(os.environ.get("EPOCHS", "3"))
MAX_LEN = int(os.environ.get("MAX_LEN", "1024"))

# --- data ---
rows = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
print("examples:", len(rows))

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def to_text(r):
    return tokenizer.apply_chat_template(
        r["messages"], tokenize=False, add_generation_prompt=False
    )


ds = Dataset.from_list([{"text": to_text(r)} for r in rows])

# --- model (4-bit QLoRA) ---
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# --- train ---
cfg = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    max_length=MAX_LEN,
    dataset_text_field="text",
    report_to="none",
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)
trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("saved adapter to", OUT_DIR)
