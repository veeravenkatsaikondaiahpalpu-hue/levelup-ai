"""
kaggle_train.py
===============
LevelUp QLoRA fine-tuning — optimised for Kaggle P100 (16GB VRAM).

Paste this entire file into a Kaggle notebook code cell and run.
Dataset input path: /kaggle/input/levelup-finetune-data/
"""

# ── Step 1: Install dependencies ──────────────────────────────────────────────
import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip("transformers>=4.40")
pip("peft>=0.10")
pip("trl>=1.2")
pip("bitsandbytes>=0.43")
pip("datasets>=2.18")
pip("accelerate>=0.28")
pip("huggingface_hub")

print("✅ Packages installed")

# ── Step 2: Imports ────────────────────────────────────────────────────────────
import os, json, torch
from pathlib import Path
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

print(f"✅ PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Step 3: Config ─────────────────────────────────────────────────────────────
MODEL_NAME   = "unsloth/Llama-3.2-3B-Instruct"
HF_TOKEN     = ""   # ← paste your HF token here if needed: "hf_xxxx"

TRAIN_FILE   = "/kaggle/input/levelup-finetune-data/finetune_train_v2.jsonl"
VAL_FILE     = "/kaggle/input/levelup-finetune-data/finetune_val_v2.jsonl"
OUTPUT_DIR   = "/kaggle/working/levelup-qlora-checkpoints"
FINAL_DIR    = "/kaggle/working/levelup-qlora-final"

# P100 16GB optimised settings
BATCH_SIZE      = 2       # P100 has 16GB — can do batch=2 comfortably
GRAD_ACCUM      = 8       # effective batch = 16
MAX_LENGTH      = 512
NUM_EPOCHS      = 3
LEARNING_RATE   = 2e-4
LORA_R          = 16
LORA_ALPHA      = 32

# ── Step 4: Login to HuggingFace (optional) ────────────────────────────────────
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✅ HuggingFace logged in")

# ── Step 5: Load Dataset ───────────────────────────────────────────────────────
def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

print("\nLoading dataset...")
train_samples = load_jsonl(TRAIN_FILE)
val_samples   = load_jsonl(VAL_FILE)

train_ds = Dataset.from_list(train_samples)
val_ds   = Dataset.from_list(val_samples)

print(f"  Train: {len(train_ds):,} samples")
print(f"  Val  : {len(val_ds):,} samples")

# ── Step 6: Load Model + Tokenizer ─────────────────────────────────────────────
print(f"\nLoading model: {MODEL_NAME}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=False,
    attn_implementation="eager",
    token=HF_TOKEN if HF_TOKEN else None,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=False,
    token=HF_TOKEN if HF_TOKEN else None,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# ── Step 7: Apply LoRA ─────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Step 8: Training Config ────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    run_name="levelup-qlora-kaggle",

    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    optim="paged_adamw_32bit",
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    weight_decay=0.01,
    max_grad_norm=0.3,

    bf16=True,
    fp16=False,

    max_length=MAX_LENGTH,
    packing=False,

    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    logging_steps=50,
    logging_first_step=True,
    report_to="none",

    dataset_text_field="text",
    dataloader_num_workers=2,
    seed=42,
)

# ── Step 9: Train ──────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=sft_config,
)

print(f"\n🚀 Starting training...")
print(f"   Effective batch size : {BATCH_SIZE * GRAD_ACCUM}")
print(f"   Train samples        : {len(train_ds):,}")
print(f"   Val samples          : {len(val_ds):,}")
print(f"   Output               : {OUTPUT_DIR}")

trainer.train()

# ── Step 10: Save Final Adapter ────────────────────────────────────────────────
print(f"\n💾 Saving final adapter -> {FINAL_DIR}")
trainer.model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print("\n✅ Training complete!")
print(f"   Adapter saved to: {FINAL_DIR}")
print(f"\n📥 Download your adapter:")
print(f"   Go to Kaggle Output tab → download 'levelup-qlora-final' folder")
