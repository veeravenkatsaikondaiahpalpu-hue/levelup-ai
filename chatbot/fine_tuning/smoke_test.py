"""
smoke_test.py
=============
Verifies the full QLoRA training pipeline using TinyLlama-1.1B
(open-access, no HuggingFace token required, fits on any GPU).

This tests:
  [1] Dataset loading + LLaMA 3 formatting
  [2] Model + tokenizer loading with 4-bit quantisation
  [3] LoRA adapter attachment
  [4] SFTTrainer 20 steps without crashing
  [5] Adapter save + reload

When this passes, the exact same pipeline will work with
meta-llama/Meta-Llama-3-8B-Instruct once you have your HF token.

Usage:
    python -m chatbot.fine_tuning.smoke_test
"""

import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

# ── Config ─────────────────────────────────────────────────────────────────────
SMOKE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT_DIR       = str(ROOT / "models" / "checkpoints" / "smoke-test")
FINAL_DIR     = str(ROOT / "models" / "final" / "smoke-test")
NUM_STEPS     = 20
MAX_SEQ_LEN   = 512
BATCH_SIZE    = 2
GRAD_ACCUM    = 2

PASS  = "[PASS]"
FAIL  = "[FAIL]"
STEP  = "  -->"


def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def main():
    print("\n" + "="*55)
    print("  LevelUp QLoRA Smoke Test")
    print(f"  Model : {SMOKE_MODEL}")
    print(f"  GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("="*55)

    results = {}

    # ── [1] Dataset ────────────────────────────────────────────────────────────
    section("[1] Dataset loading + formatting")
    try:
        from chatbot.fine_tuning.dataset_prep import load_levelup_dataset
        train_ds, val_ds = load_levelup_dataset(
            build_filter="ORACLE",
            max_train_samples=60,
            max_val_samples=20,
        )
        sample = train_ds[0]["text"]
        has_bos    = "<|begin_of_text|>" in sample
        has_eot    = "<|eot_id|>" in sample
        has_system = "<|start_header_id|>system<|end_header_id|>" in sample
        has_user   = "<|start_header_id|>user<|end_header_id|>" in sample
        has_asst   = "<|start_header_id|>assistant<|end_header_id|>" in sample
        print(f"  Train samples   : {len(train_ds)}")
        print(f"  Val samples     : {len(val_ds)}")
        print(f"  BOS token       : {PASS if has_bos   else FAIL}")
        print(f"  EOT token       : {PASS if has_eot   else FAIL}")
        print(f"  System header   : {PASS if has_system else FAIL}")
        print(f"  User header     : {PASS if has_user   else FAIL}")
        print(f"  Assistant header: {PASS if has_asst   else FAIL}")
        assert all([has_bos, has_eot, has_system, has_user, has_asst]), "Format check failed"
        print(f"\n  {PASS} Dataset OK")
        results["dataset"] = True
    except Exception as e:
        print(f"\n  {FAIL} Dataset FAILED: {e}")
        results["dataset"] = False
        raise

    # ── [2] Tokenizer ──────────────────────────────────────────────────────────
    section("[2] Tokenizer loading")
    try:
        tokenizer = AutoTokenizer.from_pretrained(SMOKE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        enc = tokenizer(sample[:200], return_tensors="pt")
        print(f"  Vocab size  : {tokenizer.vocab_size:,}")
        print(f"  Test encode : {enc['input_ids'].shape[1]} tokens")
        print(f"  {PASS} Tokenizer OK")
        results["tokenizer"] = True
    except Exception as e:
        print(f"  {FAIL} Tokenizer FAILED: {e}")
        results["tokenizer"] = False
        raise

    # ── [3] Model + 4-bit quant ────────────────────────────────────────────────
    section("[3] Model loading (4-bit QLoRA)")
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            SMOKE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        model = prepare_model_for_kbit_training(model)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Parameters  : {total/1e9:.2f}B")
        mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"  VRAM used   : {mem:.2f} GB")
        print(f"  {PASS} Model loaded OK")
        results["model"] = True
    except Exception as e:
        print(f"  {FAIL} Model load FAILED: {e}")
        results["model"] = False
        raise

    # ── [4] LoRA adapter ───────────────────────────────────────────────────────
    section("[4] LoRA adapter attachment")
    try:
        # TinyLlama target modules
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        pct       = 100 * trainable / total
        print(f"  Trainable   : {trainable:,} / {total:,} ({pct:.2f}%)")
        print(f"  {PASS} LoRA attached OK")
        results["lora"] = True
    except Exception as e:
        print(f"  {FAIL} LoRA FAILED: {e}")
        results["lora"] = False
        raise

    # ── [5] Training run (20 steps) ────────────────────────────────────────────
    section(f"[5] SFTTrainer — {NUM_STEPS} steps")
    try:
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

        sft_config = SFTConfig(
            output_dir=OUT_DIR,
            num_train_epochs=1,
            max_steps=NUM_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            bf16=True,
            fp16=False,
            max_length=MAX_SEQ_LEN,
            packing=True,
            eval_strategy="no",
            save_strategy="no",
            logging_steps=5,
            logging_first_step=True,
            report_to="none",
            dataset_text_field="text",
            dataloader_num_workers=0,
            seed=42,
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            args=sft_config,
        )

        print(f"  Starting {NUM_STEPS} training steps ...")
        train_result = trainer.train()
        loss = train_result.training_loss
        print(f"\n  Final loss   : {loss:.4f}")
        print(f"  {PASS if loss < 10 else FAIL} Training ran OK (loss={loss:.4f})")
        results["training"] = True
    except Exception as e:
        print(f"  {FAIL} Training FAILED: {e}")
        import traceback; traceback.print_exc()
        results["training"] = False

    # ── [6] Save adapter ───────────────────────────────────────────────────────
    section("[6] Save & reload adapter")
    try:
        Path(FINAL_DIR).mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(FINAL_DIR)
        tokenizer.save_pretrained(FINAL_DIR)
        saved = list(Path(FINAL_DIR).iterdir())
        print(f"  Saved files : {[f.name for f in saved]}")

        # Reload check
        from peft import PeftModel
        base2 = AutoModelForCausalLM.from_pretrained(
            SMOKE_MODEL,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        reloaded = PeftModel.from_pretrained(base2, FINAL_DIR)
        reloaded.eval()
        tok2 = AutoTokenizer.from_pretrained(FINAL_DIR)
        print(f"  Reloaded    : OK ({sum(p.numel() for p in reloaded.parameters())/1e9:.2f}B params)")
        print(f"  {PASS} Save/reload OK")
        results["save"] = True
    except Exception as e:
        print(f"  {FAIL} Save/reload FAILED: {e}")
        results["save"] = False

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  SMOKE TEST RESULTS")
    print("="*55)
    all_pass = True
    for step, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {step}")
        if not ok:
            all_pass = False

    print("="*55)
    if all_pass:
        print("\n  ALL CHECKS PASSED")
        print("  The QLoRA pipeline is fully working.")
        print("\n  Next: get your HuggingFace token for LLaMA 3 8B")
        print("  then run: python -m chatbot.fine_tuning.train --smoke_test")
    else:
        print("\n  SOME CHECKS FAILED — see above for details")
    print()


if __name__ == "__main__":
    main()
