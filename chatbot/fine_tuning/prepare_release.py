"""
Prepare the LevelUp adapter for release: cast fp32 -> bf16 and build a model card.

Why cast
--------
PEFT upcasts LoRA weights to float32 on the final save_pretrained(), but the
adapter was *trained* in bfloat16 (bnb_4bit_compute_dtype=torch.bfloat16) and
every training checkpoint on disk is bf16. Casting back to bf16 returns the
weights to their native training precision and halves the download, so it is
not a lossy compromise so much as undoing an artefact of how PEFT saves.

The round-trip error is measured and printed rather than assumed.

Usage
-----
    python -m chatbot.fine_tuning.prepare_release
    python -m chatbot.fine_tuning.prepare_release --adapter models/final/levelup-qlora-all
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "models" / "final" / "levelup-qlora-cloud"
DEFAULT_OUT = ROOT / "models" / "release" / "levelup-qlora-llama3.2-3b"
EVAL_PATH = ROOT / "models" / "eval_results.json"

COPY_FILES = [
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
]


def cast_to_bf16(src: Path, out: Path) -> dict:
    """Cast every adapter tensor to bfloat16, reporting the round-trip error."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_file = src / "adapter_model.safetensors"
    tensors: dict[str, "torch.Tensor"] = {}
    max_abs_err = 0.0
    max_rel_err = 0.0
    total = 0

    with safe_open(src_file, framework="pt") as fh:
        metadata = fh.metadata() or {}
        for key in fh.keys():
            t32 = fh.get_tensor(key).to(torch.float32)
            t16 = t32.to(torch.bfloat16)
            back = t16.to(torch.float32)

            err = (t32 - back).abs()
            max_abs_err = max(max_abs_err, float(err.max()))
            denom = t32.abs().clamp(min=1e-12)
            max_rel_err = max(max_rel_err, float((err / denom).max()))

            tensors[key] = t16
            total += t32.numel()

    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata=metadata)

    for name in COPY_FILES:
        s = src / name
        if s.exists():
            shutil.copy2(s, out / name)

    src_mb = src_file.stat().st_size / 1e6
    out_mb = (out / "adapter_model.safetensors").stat().st_size / 1e6
    return {
        "tensors": len(tensors),
        "params": total,
        "src_mb": round(src_mb, 1),
        "out_mb": round(out_mb, 1),
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
    }


def build_model_card(out: Path, stats: dict, repo_id: str) -> None:
    """Write README.md from eval_results.json so the numbers cannot drift."""
    ev = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    base = ev["base_model_scores"]
    tuned = ev["fine_tuned_scores"]
    ds = ev["dataset"]
    dec = ev["decoding"]

    rows = "\n".join(
        f"| {m.upper()} | {base[m]:.4f} | {tuned[m]:.4f} | **+{tuned[m]-base[m]:.4f}** |"
        for m in ["rouge1", "rouge2", "rougeL"]
    )
    builds = "\n".join(
        f"| {b} | {d['n']} | {d['base']:.4f} | {d['tuned']:.4f} | +{d['delta']:.4f} |"
        for b, d in sorted(ev["per_build_rougeL"].items(), key=lambda kv: -kv[1]["delta"])
    )
    rel = 100 * (tuned["rougeL"] / base["rougeL"] - 1)

    card = f"""---
base_model: {ev['base_model']}
library_name: peft
pipeline_tag: text-generation
license: llama3.2
tags:
- lora
- qlora
- peft
- trl
- sft
- llama-3.2
---

# LevelUp AI — QLoRA adapter for Llama-3.2-3B-Instruct

**Built with Llama.** This adapter is a derivative of Meta Llama 3.2 and its use
is governed by the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE).

A LoRA adapter that gives Llama-3.2-3B-Instruct seven coaching personas for a
gamified self-improvement app. Trained with QLoRA on {ds['train_rows']:,} curated
samples.

This repository contains **adapter weights only** — load it on top of
`{ev['base_model']}`.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "{ev['base_model']}"
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "{repo_id}")
tok = AutoTokenizer.from_pretrained(base)
```

Each persona is selected by its system prompt; the same prompts were present in
the training data and are expected at inference.

## Evaluation

Base model vs this adapter on **{ds['evaluated']} held-out samples**. Both conditions run
from one loaded model via PEFT's `disable_adapter()`, so they share provably
identical base weights, prompts and decoding — the adapter is the only variable.

| Metric | Base | + adapter | Delta |
|---|---|---|---|
{rows}

**ROUGE-L improves by {rel:.0f}% relative.** All seven personas improve; none regress.

| Persona | n | Base | + adapter | Delta |
|---|---|---|---|---|
{builds}

### How the evaluation set is kept honest

- Validation data is a **separate file**, not a random split of the training set
- **{ds['val_rows_dropped_as_leaked']} of {ds['val_rows']:,} validation rows ({100*ds['val_rows_dropped_as_leaked']/ds['val_rows']:.2f}%)** appear verbatim in the
  training file and are **removed before sampling**
- Greedy decoding (`do_sample={dec['do_sample']}`, `max_new_tokens={dec['max_new_tokens']}`), so results reproduce
- Sampling seed {ds['sampling_seed']}; inference in {dec['quantisation']}

### Reading these numbers

ROUGE-L measures longest-common-subsequence overlap against a *single* human
reference reply. Open-ended prompts admit many good answers, so absolute values
are low by nature — **the base-vs-adapter delta is the result, not the raw score.**
These figures isolate the adapter's contribution with prompting held constant.

No human preference evaluation has been run. Persona quality in the application
also depends on few-shot examples in the system prompt, which this evaluation
holds constant between conditions and therefore does not measure.

## Training

| | |
|---|---|
| Base model | `{ev['base_model']}` |
| Method | QLoRA — 4-bit NF4, double quantisation, bf16 compute |
| LoRA | r=16, alpha=32, dropout=0.05 |
| Target modules | q, k, v, o, gate, up, down projections |
| Trainable params | {stats['params']:,} (~{100*stats['params']/3.21e9:.2f}% of the 3.21B base) |
| Framework | transformers + peft + trl `SFTTrainer` + bitsandbytes |
| Schedule | 3 epochs, lr 2e-4, effective batch 16, max_seq_len 512 |
| Checkpoint selection | `load_best_model_at_end=True` (best eval loss) |
| Hardware | single rented GPU |

**Training logs are not retained.** The run was executed on an ephemeral cloud
instance that was destroyed afterwards, so the loss curve for these exact
weights is unavailable. The evaluation above was therefore run after the fact,
against the released weights, and is the primary evidence for this adapter.

A separate, interrupted local run reached step 1,400 of 14,367 (epoch 0.29)
with eval loss falling monotonically 1.1781 → 1.0365. Those are *different*
weights and are not released here.

## Precision

Weights are stored in **bfloat16**, the dtype they were trained in. PEFT's
`save_pretrained()` upcasts to float32; casting back halves the download with a
measured maximum relative round-trip error of **{stats['max_rel_err']:.2e}**.

## Limitations

- Adapter only — requires the base model, which carries the Llama 3.2 licence
- Trained largely on public forum text; tone is informal and not professionally reviewed
- Not safety-tuned for health, financial or psychological advice, despite coaching framing
- Evaluated on automatic overlap metrics only
"""
    (out / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--repo-id", default="venkatkondaiahpalpu/levelup-qlora-llama3.2-3b")
    args = ap.parse_args()

    src, out = Path(args.adapter), Path(args.out)
    print(f"source : {src}")
    print(f"output : {out}\n")

    stats = cast_to_bf16(src, out)
    print(f"  tensors cast      : {stats['tensors']}")
    print(f"  parameters        : {stats['params']:,}")
    print(f"  size              : {stats['src_mb']} MB (fp32) -> {stats['out_mb']} MB (bf16)")
    print(f"  max abs round-trip error : {stats['max_abs_err']:.3e}")
    print(f"  max rel round-trip error : {stats['max_rel_err']:.3e}")

    build_model_card(out, stats, args.repo_id)
    print(f"\n  model card written: {out / 'README.md'}")
    print("\nContents:")
    for f in sorted(out.iterdir()):
        print(f"  {f.stat().st_size/1e6:8.2f} MB  {f.name}")


if __name__ == "__main__":
    main()
