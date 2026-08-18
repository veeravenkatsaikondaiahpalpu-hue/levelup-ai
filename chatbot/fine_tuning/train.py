"""
QLoRA fine-tuning for the LevelUp AI chatbot — runs on any single GPU with 8GB+ VRAM.

How it works:
  - Base model   : unsloth/Llama-3.2-3B-Instruct (small enough for an 8GB GPU)
  - Quantisation : 4-bit NF4 via BitsAndBytes — cuts VRAM usage roughly in half
  - LoRA adapters: r=16, alpha=32 on all attention + MLP projections (~1-2% of params trainable)
  - Trainer      : TRL SFTTrainer (handles the LLaMA 3 chat template automatically)

How to run:
  # Train on all 7 builds at once
  python -m chatbot.fine_tuning.train

  # Train only one build (much faster, good for per-persona experiments)
  python -m chatbot.fine_tuning.train --build TITAN

  # Quick sanity check — 1 epoch on 200 samples to make sure the pipeline works
  python -m chatbot.fine_tuning.train --smoke_test

  # Resume from a saved checkpoint if training was interrupted
  python -m chatbot.fine_tuning.train --resume_from models/checkpoints/levelup-qlora/checkpoint-500

Where things get saved:
  models/checkpoints/{run_name}/    ← checkpoints every N steps (keeps last 3)
  models/final/{run_name}/          ← final adapter weights, ready to load for inference
"""

import argparse
import os
from pathlib import Path

# Load .env and login to HuggingFace
_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
if _ROOT_ENV.exists():
    for _line in _ROOT_ENV.read_text(encoding="utf-8").splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token and os.environ.get("HF_HUB_OFFLINE", "0") != "1":
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
        print(f"[HF] Logged in with token (len={len(_hf_token)})")
    except Exception as _e:
        print(f"[HF] Login skipped ({_e.__class__.__name__}) — using cached model")

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# Paths
ROOT         = Path(__file__).resolve().parents[2]
CHECKPOINTS  = ROOT / "models" / "checkpoints"
FINAL_MODELS = ROOT / "models" / "final"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
FINAL_MODELS.mkdir(parents=True, exist_ok=True)


# Configuration

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"  # 3B fits comfortably in 8GB VRAM; use 8B on a cloud A100

# We target every attention and feed-forward projection in LLaMA 3.
# Hitting all 7 modules gives the adapter broad control over the model's style.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # MLP feed-forward
]


def get_bnb_config() -> BitsAndBytesConfig:
    """
    4-bit NF4 quantisation — the 'Q' in QLoRA.
    Keeps the frozen base model at ~2GB instead of ~6GB, leaving room for gradients.
    double_quant squeezes out another ~0.4GB by quantising the quantisation constants.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05) -> LoraConfig:
    """
    LoRA adapter config.
    r=16 means each weight update is factored into two matrices of rank 16.
    alpha/r = 2.0 — moderate scaling, neither too timid nor too aggressive.
    """
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )


# Model + Tokenizer loading

def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """
    Load the base model (optionally 4-bit quantised) plus its tokenizer.
    Returns (model, tokenizer) ready for LoRA application.
    """
    print(f"\nLoading base model: {model_name}")

    hf_token   = os.environ.get("HF_TOKEN")
    bnb_config = get_bnb_config() if use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",           # let HuggingFace decide how to spread across GPUs
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="eager", # swap to "flash_attention_2" for ~30% speedup if installed
        token=hf_token,
    )

    if use_4bit:
        # Required step before attaching LoRA to a quantised model
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False,
                                              token=hf_token)

    # LLaMA 3 ships without a pad token — we reuse EOS, which is standard practice
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # must be right for causal LM training

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# Training

def train(
    model_name:    str   = DEFAULT_MODEL,
    build_filter:  str   = None,
    run_name:      str   = None,
    smoke_test:    bool  = False,
    resume_from:   str   = None,
    # Hyperparameters
    num_epochs:         int   = 3,
    per_device_batch:   int   = 1,
    gradient_accum:     int   = 16,
    learning_rate:      float = 2e-4,
    max_seq_length:     int   = 512,
    lora_r:             int   = 16,
    lora_alpha:         int   = 32,
    lora_dropout:       float = 0.05,
    warmup_ratio:       float = 0.05,
    weight_decay:       float = 0.01,
    save_steps:         int   = 200,
    eval_steps:         int   = 200,
    logging_steps:      int   = 25,
):
    # Dataset
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

    # Run name
    if run_name is None:
        suffix = f"-{build_filter.lower()}" if build_filter else "-all"
        run_name = f"levelup-qlora{suffix}"

    output_dir = str(CHECKPOINTS / run_name)

    # Model
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Apply LoRA
    lora_config = get_lora_config(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # SFT Config
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
        optim="paged_adamw_32bit",              # offloads optimizer states to CPU RAM — saves ~4GB VRAM
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=0.3,

        # bfloat16 mixed precision — stable range on modern GPUs, no overflow issues
        bf16=True,
        fp16=False,

        # 512 tokens covers ~95% of our training samples; packing off to avoid OOM on 8GB
        max_length=max_seq_length,
        packing=False,

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

    # Trainer
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

    # Save final adapter
    final_path = str(FINAL_MODELS / run_name)
    print(f"\nSaving final adapter -> {final_path}")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print("\nTraining complete!")
    print(f"  Checkpoint dir : {output_dir}")
    print(f"  Final adapter  : {final_path}")
    print("\nTo run inference:")
    print(f"  python -m chatbot.inference --adapter {final_path} --build {build_filter or 'ORACLE'}")

    return trainer


# CLI entry point

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
    p.add_argument("--batch_size",   type=int,   default=1)
    p.add_argument("--grad_accum",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--max_seq_len",  type=int,   default=512)
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
