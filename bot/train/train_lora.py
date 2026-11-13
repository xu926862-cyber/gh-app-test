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
# Qwen ships bfloat16 weights; the fp16 GradScaler on T4 cannot unscale bf16
# gradients. transformers 5 renamed torch_dtype -> dtype and ignores the old
# kwarg, so pick the right one for the installed version.
import transformers

_dtype_kw = (
    {"dtype": torch.float16}
    if int(transformers.__version__.split(".")[0]) >= 5
    else {"torch_dtype": torch.float16}
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    **_dtype_kw,
)


def force_amp_safe(model):
    """No bf16 may reach the fp16 GradScaler.

    Accelerate's GradScaler.unscale_ calls
    _amp_foreach_non_finite_check_and_unscale_cuda, which raises
    NotImplementedError for BFloat16. Normally the dtype kwarg above keeps the
    model in fp16, but the kwarg name changed between transformers 4 and 5 and
    is silently ignored when wrong. Cast whatever is left: frozen weights to
    fp16, trainable ones (LoRA) to fp32 -- both are scaler-safe.
    """
    fixed = 0
    for _, p in model.named_parameters():
        if p.__class__.__name__ in ("Params4bit", "Int8Params"):
            continue  # quantized storage, cast via bnb_4bit_compute_dtype instead
        if p.dtype != torch.bfloat16:
            continue
        p.data = p.data.to(torch.float32 if p.requires_grad else torch.float16)
        fixed += 1
    if fixed:
        print(f"dtype guard: cast {fixed} bf16 tensors (trainable->fp32, frozen->fp16)")


force_amp_safe(model)
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
force_amp_safe(model)  # adapters may come out in the base model's dtype
model.print_trainable_parameters()
print(
    "trainable dtypes:",
    sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
)

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
    # T4 is compute capability 7.5: the default chunked_nll loss uses BF16
    # tensor cores (needs 8.0+) and crashes. Use the plain NLL path + fp16.
    loss_type="nll",
    bf16=False,
    fp16=True,
    use_liger_kernel=False,
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)
trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("saved adapter to", OUT_DIR)
