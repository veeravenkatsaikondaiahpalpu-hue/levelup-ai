"""
Held-out evaluation for the LevelUp QLoRA adapter.

What this answers
-----------------
"What did the fine-tuning actually buy you?" — by scoring the base model and the
fine-tuned model on the *same* held-out samples, with the same prompts and the
same decoding settings. A score on its own says little; the delta is the result.

Design notes
------------
- The validation set is a separate file (finetune_val_v2.jsonl), not a random
  split of the training data, so it is reusable and reproducible.
- Rows whose (user, assistant) pair also appears in the training file are
  dropped before sampling. Roughly 2% of the val set overlaps; leaving it in
  would flatter the fine-tuned model specifically.
- Base and fine-tuned responses come from ONE loaded model via PEFT's
  disable_adapter() context, which guarantees the base weights are identical
  between the two conditions and halves load time.
- Greedy decoding (do_sample=False) so the numbers are reproducible.

Expect modest absolute ROUGE-L. The references are single human replies to
open-ended prompts, and there are many good answers to any such prompt; ROUGE-L
measures longest-common-subsequence overlap with one of them. The comparison
against the base model is the meaningful part.

Usage
-----
    python -m chatbot.fine_tuning.evaluate
    python -m chatbot.fine_tuning.evaluate --n 400 --batch-size 8
    python -m chatbot.fine_tuning.evaluate --adapter models/final/levelup-qlora-all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_PATH = ROOT / "data" / "raw" / "finetune_train_v2.jsonl"
VAL_PATH = ROOT / "data" / "raw" / "finetune_val_v2.jsonl"
DEFAULT_ADAPTER = ROOT / "models" / "final" / "levelup-qlora-cloud"
DEFAULT_BASE = "unsloth/Llama-3.2-3B-Instruct"
RESULTS_PATH = ROOT / "models" / "eval_results.json"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _norm(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def _pair_key(row: dict) -> str:
    blob = _norm(row.get("user")) + "||" + _norm(row.get("assistant"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def build_eval_set(n: int, seed: int) -> tuple[list[dict], dict]:
    """Return n leak-free validation rows, plus provenance stats for the report."""
    _log(f"Loading datasets\n  train: {TRAIN_PATH}\n  val  : {VAL_PATH}")
    train_rows = _load_jsonl(TRAIN_PATH)
    val_rows = _load_jsonl(VAL_PATH)

    train_keys = {_pair_key(r) for r in train_rows}
    clean = [r for r in val_rows if _pair_key(r) not in train_keys]
    dropped = len(val_rows) - len(clean)

    # Rows need all three fields to be scoreable.
    usable = [
        r for r in clean
        if (r.get("system") or "").strip()
        and (r.get("user") or "").strip()
        and (r.get("assistant") or "").strip()
    ]

    rng = random.Random(seed)
    sample = usable if len(usable) <= n else rng.sample(usable, n)

    stats = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_rows_dropped_as_leaked": dropped,
        "val_rows_after_leak_removal": len(clean),
        "val_rows_usable": len(usable),
        "evaluated": len(sample),
        "sampling_seed": seed,
    }
    _log(
        f"  train {stats['train_rows']:,} | val {stats['val_rows']:,} "
        f"| dropped as leaked {dropped:,} ({100*dropped/max(len(val_rows),1):.2f}%) "
        f"| evaluating {len(sample):,}"
    )
    return sample, stats


def load_model(base: str, adapter: str, four_bit: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    _log(f"\nLoading base model : {base}")
    quant = None
    if four_bit and torch.cuda.is_available():
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        _log("  quantisation     : 4-bit NF4 (matches training)")

    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=quant,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    # Decoder-only batched generation requires left padding.
    tok.padding_side = "left"

    _log(f"Attaching adapter  : {adapter}")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def generate(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int,
             label: str, use_adapter: bool) -> list[str]:
    """Greedy-decode every prompt, with the adapter on or bypassed."""
    import torch
    from contextlib import nullcontext

    outputs: list[str] = []
    started = time.time()
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for bi in range(total_batches):
        chunk = prompts[bi * batch_size:(bi + 1) * batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(model.device)

        ctx = nullcontext() if use_adapter else model.disable_adapter()
        with torch.no_grad(), ctx:
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,   # unset: only meaningful for sampling
                top_p=None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        for row_in, row_out in zip(enc["input_ids"], gen):
            outputs.append(tok.decode(row_out[len(row_in):], skip_special_tokens=True).strip())

        done = len(outputs)
        rate = done / max(time.time() - started, 1e-9)
        eta = (len(prompts) - done) / max(rate, 1e-9)
        _log(f"  [{label}] {done:>4}/{len(prompts)}  "
             f"({100*done/len(prompts):5.1f}%)  {rate:4.2f} samples/s  ETA {eta/60:5.1f} min")

    return outputs


def score(preds: list[str], refs: list[str]) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        for k in agg:
            agg[k].append(s[k].fmeasure)
    return {k: round(sum(v) / max(len(v), 1), 4) for k, v in agg.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="held-out samples to evaluate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--adapter", type=str, default=str(DEFAULT_ADAPTER))
    ap.add_argument("--base", type=str, default=DEFAULT_BASE)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    from chatbot.prompt_template import detect_build, format_inference_prompt

    t0 = time.time()
    _log("=" * 68)
    _log("LevelUp AI — held-out evaluation (base vs fine-tuned)")
    _log("=" * 68)

    rows, stats = build_eval_set(args.n, args.seed)
    prompts = [format_inference_prompt(r["system"], r["user"]) for r in rows]
    refs = [r["assistant"].strip() for r in rows]

    model, tok = load_model(args.base, args.adapter, not args.no_4bit)

    _log(f"\nGenerating {len(prompts)} responses per condition "
         f"(greedy, max_new_tokens={args.max_new_tokens}, batch={args.batch_size})")
    _log("\n-- condition 1/2: BASE model (adapter bypassed) --")
    base_out = generate(model, tok, prompts, args.max_new_tokens,
                        args.batch_size, "base", use_adapter=False)
    _log("\n-- condition 2/2: FINE-TUNED model (adapter active) --")
    tuned_out = generate(model, tok, prompts, args.max_new_tokens,
                         args.batch_size, "tuned", use_adapter=True)

    _log("\nScoring with ROUGE (stemmed)...")
    base_scores = score(base_out, refs)
    tuned_scores = score(tuned_out, refs)

    # Per-persona ROUGE-L, so a single average cannot hide a weak build.
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    per_build: dict[str, dict] = {}
    for row, b, t, ref in zip(rows, base_out, tuned_out, refs):
        build = detect_build(row.get("system", "")) or "UNKNOWN"
        d = per_build.setdefault(build, {"n": 0, "base": 0.0, "tuned": 0.0})
        d["n"] += 1
        d["base"] += sc.score(ref, b)["rougeL"].fmeasure
        d["tuned"] += sc.score(ref, t)["rougeL"].fmeasure
    for b, d in per_build.items():
        d["base"] = round(d["base"] / d["n"], 4)
        d["tuned"] = round(d["tuned"] / d["n"], 4)
        d["delta"] = round(d["tuned"] - d["base"], 4)

    results = {
        "adapter": args.adapter,
        "base_model": args.base,
        "dataset": stats,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "max_input_tokens": 1024,
            "quantisation": "4-bit NF4" if not args.no_4bit else "none",
        },
        "base_model_scores": base_scores,
        "fine_tuned_scores": tuned_scores,
        "delta": {k: round(tuned_scores[k] - base_scores[k], 4) for k in base_scores},
        "per_build_rougeL": per_build,
        "runtime_minutes": round((time.time() - t0) / 60, 2),
        "caveats": [
            "ROUGE-L measures longest-common-subsequence overlap against a single "
            "human reference reply. Open-ended prompts admit many good answers, so "
            "absolute values are low by nature; the base-vs-tuned delta is the result.",
            "Leaked validation rows were removed before sampling.",
            "Greedy decoding; no sampling, no prompt tuning beyond the training format.",
        ],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    _log("\n" + "=" * 68)
    _log("RESULTS")
    _log("=" * 68)
    _log(f"{'metric':<10} {'base':>9} {'fine-tuned':>12} {'delta':>9}")
    for k in ["rouge1", "rouge2", "rougeL"]:
        d = tuned_scores[k] - base_scores[k]
        _log(f"{k:<10} {base_scores[k]:>9.4f} {tuned_scores[k]:>12.4f} {d:>+9.4f}")
    _log("\nPer-persona ROUGE-L:")
    for b, d in sorted(per_build.items(), key=lambda kv: -kv[1]["n"]):
        _log(f"  {b:<10} n={d['n']:<4} base {d['base']:.4f} -> tuned {d['tuned']:.4f}  ({d['delta']:+.4f})")
    _log(f"\nEvaluated {stats['evaluated']} held-out samples "
         f"({stats['val_rows_dropped_as_leaked']} leaked rows removed first)")
    _log(f"Runtime: {results['runtime_minutes']} min")
    _log(f"Written to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
