"""
train.py
========
QLoRA fine-tuning for LevelUp AI chatbot using TRL + PEFT on LLaMA 3.

Architecture:
  - Base model  : meta-llama/Meta-Llama-3-8B-Instruct  (configurable)
  - Quantisation: 4-bit NF4 (BitsAndBytes) to fit on a single GPU
  - LoRA adapters: r=16, alpha=32, targeting all attention + MLP projections
  - Trainer     : TRL SFTTrainer with sequence packing for efficiency

Usage:
  # Full dataset (all builds)
  python -m chatbot.fine_tuning.train

  # Single build only (faster per-build adapter)
  python -m chatbot.fine_tuning.train --build TITAN

  # Quick smoke test (100 steps)
  python -m chatbot.fine_tuning.train --smoke_test

  # Resume from checkpoint
  python -m chatbot.fine_tuning.train --resume_from models/checkpoints/levelup-qlora/checkpoint-500

Output:
  models/checkpoints/{run_name}/          <- training checkpoints
  models/final/{run_name}/                <- merged LoRA adapter (ready for inference)
"""

import argparse
import os
from pathlib import Path

# ── Load .env and login to HuggingFace ────────────────────────────────────────
_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
if _ROOT_ENV.exists():
    for _line in _ROOT_ENV.read_text(encoding="utf-8").splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login as _hf_login
    _hf_login(token=_hf_token, add_to_git_credential=False)
    print(f"[HF] Logged in with token (len={len(_hf_token)})")

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
CHECKPOINTS  = ROOT / "models" / "checkpoints"
FINAL_MODELS = ROOT / "models" / "final"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
FINAL_MODELS.mkdir(parents=True, exist_ok=True)


# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"   # fits in 8GB VRAM; swap for 8B on cloud GPU

# LoRA targets for LLaMA 3 — all attention projections + feed-forward gate/up/down
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantisation config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # nested quantisation saves ~0.4 GB
    )


def get_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05) -> LoraConfig:
    """LoRA adapter config."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )


# ── Model + Tokenizer loading ──────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """
    Load the base model with optional 4-bit quantisation and the tokenizer.
    Returns (model, tokenizer).
    """
    print(f"\nLoading model: {model_name}")

    hf_token   = os.environ.get("HF_TOKEN")
    bnb_config = get_bnb_config() if use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",                # auto-places layers across available GPUs
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="eager",      # use "flash_attention_2" if installed
        token=hf_token,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False,
                                              token=hf_token)

    # LLaMA 3 doesn't have a pad token by default — use EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"   # required for causal LM training

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    model_name:    str   = DEFAULT_MODEL,
    build_filter:  str   = None,
    run_name:      str   = None,
    smoke_test:    bool  = False,
    resume_from:   str   = None,
    # Hyperparameters
    num_epochs:         int   = 3,
    per_device_batch:   int   = 4,
    gradient_accum:     int   = 4,
    learning_rate:      float = 2e-4,
    max_seq_length:     int   = 2048,
    lora_r:             int   = 16,
    lora_alpha:         int   = 32,
    lora_dropout:       float = 0.05,
    warmup_ratio:       float = 0.05,
    weight_decay:       float = 0.01,
    save_steps:         int   = 200,
    eval_steps:         int   = 200,
    logging_steps:      int   = 25,
):
    # ── Dataset ───────────────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(ROOT))
    from chatbot.fine_tuning.dataset_prep import load_levelup_dataset

    train_ds, val_ds = load_levelup_dataset(
        build_filter=build_filter,
        max_train_samples=200 if smoke_test else None,
        max_val_samples=50  if smoke_test else None,
    )

    if smoke_test:
        num_epochs   = 1
        save_steps   = 50
        eval_steps   = 50
        logging_steps = 10
        print("\n[SMOKE TEST MODE] Running 1 epoch on 200 samples")

    # ── Run name ──────────────────────────────────────────────────────────────
    if run_name is None:
        suffix = f"-{build_filter.lower()}" if build_filter else "-all"
        run_name = f"levelup-qlora{suffix}"

    output_dir = str(CHECKPOINTS / run_name)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Apply LoRA
    lora_config = get_lora_config(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── SFT Config ────────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=run_name,

        # Training schedule
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accum,
        gradient_checkpointing=True,            # saves GPU memory
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Optimiser
        optim="paged_adamw_32bit",              # paged AdamW for QLoRA
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=0.3,

        # Precision
        bf16=True,                              # bfloat16 mixed precision
        fp16=False,

        # Sequence packing — packs multiple short samples into one 2048-token window
        # massively improves GPU utilisation on short Q&A data
        max_length=max_seq_length,
        packing=True,

        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Logging
        logging_steps=logging_steps,
        logging_first_step=True,
        report_to="none",                       # set to "wandb" if you use W&B

        # Dataset column
        dataset_text_field="text",

        # Misc
        dataloader_num_workers=0,               # Windows compatibility
        seed=42,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
    )

    print(f"\nStarting training: {run_name}")
    print(f"  Effective batch size : {per_device_batch * gradient_accum}")
    print(f"  Train samples        : {len(train_ds):,}")
    print(f"  Val samples          : {len(val_ds):,}")
    print(f"  Output               : {output_dir}")

    # Resume from checkpoint if specified
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save final adapter ────────────────────────────────────────────────────
    final_path = str(FINAL_MODELS / run_name)
    print(f"\nSaving final adapter -> {final_path}")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print("\nTraining complete!")
    print(f"  Checkpoint dir : {output_dir}")
    print(f"  Final adapter  : {final_path}")
    print(f"\nTo run inference:")
    print(f"  python -m chatbot.inference --adapter {final_path} --build {build_filter or 'ORACLE'}")

    return trainer


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LevelUp QLoRA fine-tuning")
    p.add_argument("--model",        default=DEFAULT_MODEL,
                   help="HuggingFace model ID or local path")
    p.add_argument("--build",        default=None,
                   choices=["TITAN", "ORACLE", "PHANTOM", "SAGE", "MUSE", "EMPIRE", "GG"],
                   help="Train on one build only (default: all builds)")
    p.add_argument("--run_name",     default=None,
                   help="Experiment name for checkpoints (auto-generated if not set)")
    p.add_argument("--smoke_test",   action="store_true",
                   help="Quick 1-epoch test on 200 samples to verify pipeline")
    p.add_argument("--resume_from",  default=None,
                   help="Resume from a checkpoint directory")
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--max_seq_len",  type=int,   default=2048)
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model,
        build_filter=args.build,
        run_name=args.run_name,
        smoke_test=args.smoke_test,
        resume_from=args.resume_from,
        num_epochs=args.epochs,
        per_device_batch=args.batch_size,
        gradient_accum=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
