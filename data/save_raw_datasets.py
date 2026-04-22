"""
save_raw_datasets.py
Downloads and saves every dataset used in the LevelUp pipeline as raw CSV files
into data/raw/datasets/<name>.csv — so the original source data is visible in
the project folder.

Large datasets are capped at 10,000 rows to keep disk usage reasonable while
still showing real, substantial data. Small datasets are saved in full.

Run from project root:
    PYTHONPATH=. venv/Scripts/python.exe data/save_raw_datasets.py
"""

import os
import csv
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join("data", "raw", "datasets")
os.makedirs(OUT_DIR, exist_ok=True)


def _try_load(hf_id, split="train", streaming=False):
    try:
        from datasets import load_dataset
        print(f"  Loading {hf_id} ...")
        return load_dataset(hf_id, split=split, streaming=streaming)
    except Exception as e:
        print(f"  [SKIP] {hf_id}: {e}")
        return None


def save_csv(name, rows, fieldnames):
    """Write rows (list of dicts) to data/raw/datasets/<name>.csv"""
    path = os.path.join(OUT_DIR, f"{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows):,} rows -> {path}")
    return path


def save_jsonl(name, rows):
    path = os.path.join(OUT_DIR, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(rows):,} rows -> {path}")
    return path


# ── 1. HealthCareMagic ────────────────────────────────────────────────────────
def save_healthcare_magic(cap=5000):
    ds = _try_load("lavita/ChatDoctor-HealthCareMagic-100k", streaming=True)
    if not ds: return
    rows = []
    for row in ds:
        rows.append({
            "input":  str(row.get("input",  "") or ""),
            "output": str(row.get("output", "") or row.get("answer", "") or ""),
        })
        if len(rows) >= cap: break
    save_csv("01_healthcare_magic", rows, ["input", "output"])


# ── 2. MentalChat16K ─────────────────────────────────────────────────────────
def save_mentalchat():
    ds = _try_load("ShenLab/MentalChat16K")
    if not ds: return
    rows = [{"Context": str(r.get("Context","") or r.get("input","") or ""),
             "Response": str(r.get("Response","") or r.get("output","") or "")}
            for r in ds]
    save_csv("02_mentalchat16k", rows, ["Context", "Response"])


# ── 3. Fitness Q&A ────────────────────────────────────────────────────────────
def save_fitness_qa():
    ds = _try_load("its-myrto/fitness-question-answers")
    if not ds: return
    rows = []
    for r in ds:
        # Check actual columns
        row_dict = dict(r)
        rows.append(row_dict)
    fields = list(rows[0].keys()) if rows else ["question", "answer"]
    save_csv("03_fitness_qa", rows, fields)


# ── 4. Fitness Chat ───────────────────────────────────────────────────────────
def save_fitness_chat():
    ds = _try_load("chibbss/fitness-chat-prompt-completion-dataset")
    if not ds: return
    rows = []
    for r in ds:
        row_dict = dict(r)
        rows.append(row_dict)
    fields = list(rows[0].keys()) if rows else ["prompt", "completion"]
    save_csv("04_fitness_chat", rows, fields)


# ── 5. Mental Health Counseling ───────────────────────────────────────────────
def save_mental_counseling(cap=5000):
    ds = _try_load("Amod/mental_health_counseling_conversations")
    if not ds: return
    rows = []
    for r in ds:
        rows.append({
            "Context":  str(r.get("Context",  "") or r.get("input",  "") or ""),
            "Response": str(r.get("Response", "") or r.get("output", "") or ""),
        })
        if len(rows) >= cap: break
    save_csv("05_mental_health_counseling", rows, ["Context", "Response"])


# ── 6. NPC Dialogue ───────────────────────────────────────────────────────────
def save_npc_dialogue():
    ds = _try_load("amaydle/npc-dialogue")
    if not ds: return
    rows = []
    for r in ds:
        row_dict = dict(r)
        rows.append(row_dict)
    fields = list(rows[0].keys()) if rows else ["context", "response"]
    save_csv("06_npc_dialogue", rows, fields)


# ── 7. GYM Exercise ───────────────────────────────────────────────────────────
def save_gym_exercise():
    ds = _try_load("onurSakar/GYM-Exercise")
    if not ds: return
    rows = [{"text": str(r.get("text", "") or "")} for r in ds]
    save_csv("07_gym_exercise", rows, ["text"])


# ── 8. Fitness Q&A Large ─────────────────────────────────────────────────────
def save_fitness_qa_large(cap=10000):
    ds = _try_load("hammamwahab/fitness-qa")
    if not ds: return
    rows = []
    for r in ds:
        rows.append({
            "context":  str(r.get("context",  "") or ""),
            "question": str(r.get("question", "") or ""),
            "answer":   str(r.get("answer",   "") or ""),
        })
        if len(rows) >= cap: break
    save_csv("08_fitness_qa_large", rows, ["context", "question", "answer"])


# ── 9. Motivational Interviewing ─────────────────────────────────────────────
def save_motivational_interviewing():
    ds = _try_load("to-be/annomi-motivational-interviewing-therapy-conversations")
    if not ds: return
    rows = []
    for r in ds:
        rows.append({
            "id":            str(r.get("id", "") or ""),
            "conversations": json.dumps(r.get("conversations", []), ensure_ascii=False),
        })
    save_csv("09_motivational_interviewing", rows, ["id", "conversations"])


# ── 10. D&D CRD3 ─────────────────────────────────────────────────────────────
def save_dnd_crd3():
    print("  [INFO] microsoft/crd3 uses deprecated HF scripts — skipping raw save.")
    # Write a placeholder note file instead
    note_path = os.path.join(OUT_DIR, "10_dnd_crd3_NOTE.txt")
    with open(note_path, "w") as f:
        f.write(
            "Dataset: microsoft/crd3\n"
            "HuggingFace: https://huggingface.co/datasets/microsoft/crd3\n"
            "Size: 398,000 dialogue turns\n"
            "License: CC-BY-SA-4.0\n\n"
            "Status: This dataset uses a legacy HuggingFace loading script (crd3.py) "
            "that is no longer supported in datasets>=2.x. "
            "The dataset is publicly accessible at the URL above. "
            "Raw data can be downloaded manually from the HuggingFace dataset page.\n"
        )
    print(f"  Note saved -> {note_path}")


# ── 11-15. Build-Specific Handcrafted Q&A ────────────────────────────────────
def save_handcrafted_qa():
    """Save all 5 handcrafted build Q&A sets as a single CSV."""
    from data.preprocess import (
        INTELLIGENCE_QA, DEXTERITY_QA, CREATIVE_QA,
        ENTREPRENEUR_QA, GAMER_QA
    )
    rows = []
    for build, pairs in [
        ("intelligence",  INTELLIGENCE_QA),
        ("dexterity",     DEXTERITY_QA),
        ("creative",      CREATIVE_QA),
        ("entrepreneur",  ENTREPRENEUR_QA),
        ("gamer",         GAMER_QA),
    ]:
        for q, a in pairs:
            rows.append({"build": build, "question": q, "answer": a})
    save_csv("11_15_handcrafted_build_qa", rows, ["build", "question", "answer"])


# ── 16. Education High School Students ───────────────────────────────────────
def save_education_hf(cap=5000):
    ds = _try_load("ajibawa-2023/Education-High-School-Students")
    if not ds: return
    rows = []
    for r in ds:
        rows.append({
            "prompt":            str(r.get("prompt", "") or ""),
            "text":              str(r.get("text",   "") or "")[:500],  # trim long essays
            "text_token_length": str(r.get("text_token_length", "") or ""),
        })
        if len(rows) >= cap: break
    save_csv("16_education_high_school", rows, ["prompt", "text", "text_token_length"])


# ── 17. Socratic Conversations ────────────────────────────────────────────────
def save_socratic_conversations():
    ds = _try_load("sanjaypantdsd/socratic-method-conversations")
    if not ds: return
    rows = []
    for r in ds:
        msgs = r.get("messages") or []
        user_msg  = next((m["content"] for m in msgs if m.get("role") == "user"),      "")
        asst_msg  = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        rows.append({"user": str(user_msg), "assistant": str(asst_msg)})
    save_csv("17_socratic_conversations", rows, ["user", "assistant"])


# ── 18. Writing Prompts ───────────────────────────────────────────────────────
def save_writing_prompts(cap=5000):
    ds = _try_load("euclaise/writingprompts")
    if not ds: return
    rows = []
    for r in ds:
        rows.append({
            "prompt": str(r.get("prompt", "") or r.get("wp",       "") or ""),
            "story":  str(r.get("story",  "") or r.get("response", "") or "")[:600],
        })
        if len(rows) >= cap: break
    save_csv("18_writing_prompts", rows, ["prompt", "story"])


# ── 19. Startup Interviews (YC) ──────────────────────────────────────────────
def save_startup_interviews():
    ds = _try_load("Glavin001/startup-interviews")
    if not ds: return
    rows = []
    for r in ds:
        row_dict = dict(r)
        rows.append(row_dict)
    fields = list(rows[0].keys()) if rows else ["instruction", "output"]
    save_csv("19_startup_interviews_yc", rows, fields)


# ── 20. Sales Conversations ───────────────────────────────────────────────────
def save_sales_conversations(cap=5000):
    ds = _try_load("goendalf666/sales-conversations-instruction-base")
    if not ds: return
    rows = []
    for r in ds:
        rows.append({"conversation": str(r.get("0", "") or "")})
        if len(rows) >= cap: break
    save_csv("20_sales_conversations", rows, ["conversation"])


# ── 21. Dota2 Instruct ────────────────────────────────────────────────────────
def save_dota2_instruct():
    ds = _try_load("Aiden07/dota2_instruct_prompt")
    if not ds: return
    rows = []
    for r in ds:
        row_dict = dict(r)
        rows.append(row_dict)
    fields = list(rows[0].keys()) if rows else ["instruction", "output"]
    save_csv("21_dota2_instruct", rows, fields)


# ── 22. English Quotes (Sentiment) ───────────────────────────────────────────
def save_english_quotes():
    ds = _try_load("Abirate/english_quotes")
    if not ds: return
    rows = []
    for r in ds:
        row_dict = dict(r)
        rows.append(row_dict)
    fields = list(rows[0].keys()) if rows else ["quote", "author", "tags"]
    save_csv("22_english_quotes_sentiment", rows, fields)


# ── 23. Synthetic Anomaly Dataset ────────────────────────────────────────────
def save_anomaly_dataset():
    """Already generated — just copy/reference the existing file."""
    src  = os.path.join("data", "raw", "anomaly_logs.csv")
    dest = os.path.join(OUT_DIR, "23_synthetic_anomaly_logs.csv")
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dest)
        with open(src) as f:
            count = sum(1 for _ in f) - 1
        print(f"  Copied {count:,} rows -> {dest}")
    else:
        print(f"  [WARN] {src} not found — run generate_anomaly_data.py first")


# ── Manifest ──────────────────────────────────────────────────────────────────
def write_manifest(results):
    path = os.path.join(OUT_DIR, "00_MANIFEST.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("LevelUp AI Companion — Raw Dataset Index\n")
        f.write("=" * 50 + "\n")
        f.write("Applied AI Exam Project | SRH Stuttgart | 2026\n\n")
        f.write("All datasets saved in data/raw/datasets/\n")
        f.write("Large datasets are capped at 5,000-10,000 rows for storage.\n")
        f.write("Full datasets are available on HuggingFace as listed below.\n\n")
        f.write(f"{'File':<45} {'HuggingFace ID':<55} {'License'}\n")
        f.write("-" * 130 + "\n")
        for fname, hf_id, license_ in results:
            f.write(f"{fname:<45} {hf_id:<55} {license_}\n")
    print(f"\n  Manifest saved -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LevelUp AI — Saving Raw Datasets")
    print("=" * 60)

    tasks = [
        ("01_healthcare_magic.csv",            save_healthcare_magic,        "lavita/ChatDoctor-HealthCareMagic-100k",                     "CC-BY-NC-4.0"),
        ("02_mentalchat16k.csv",               save_mentalchat,              "ShenLab/MentalChat16K",                                      "MIT"),
        ("03_fitness_qa.csv",                  save_fitness_qa,              "its-myrto/fitness-question-answers",                         "Apache 2.0"),
        ("04_fitness_chat.csv",                save_fitness_chat,            "chibbss/fitness-chat-prompt-completion-dataset",              "Apache 2.0"),
        ("05_mental_health_counseling.csv",    save_mental_counseling,       "Amod/mental_health_counseling_conversations",                "CC-BY-4.0"),
        ("06_npc_dialogue.csv",                save_npc_dialogue,            "amaydle/npc-dialogue",                                       "MIT"),
        ("07_gym_exercise.csv",                save_gym_exercise,            "onurSakar/GYM-Exercise",                                     "Open"),
        ("08_fitness_qa_large.csv",            save_fitness_qa_large,        "hammamwahab/fitness-qa",                                     "Apache 2.0"),
        ("09_motivational_interviewing.csv",   save_motivational_interviewing,"to-be/annomi-motivational-interviewing-therapy-conversations","Research"),
        ("10_dnd_crd3_NOTE.txt",               save_dnd_crd3,                "microsoft/crd3",                                             "CC-BY-SA-4.0"),
        ("11_15_handcrafted_build_qa.csv",     save_handcrafted_qa,          "Handcrafted (in-house)",                                     "In-house"),
        ("16_education_high_school.csv",       save_education_hf,            "ajibawa-2023/Education-High-School-Students",                "Apache 2.0"),
        ("17_socratic_conversations.csv",      save_socratic_conversations,  "sanjaypantdsd/socratic-method-conversations",                "MIT"),
        ("18_writing_prompts.csv",             save_writing_prompts,         "euclaise/writingprompts",                                    "MIT"),
        ("19_startup_interviews_yc.csv",       save_startup_interviews,      "Glavin001/startup-interviews",                               "CC-BY-NC-2.0"),
        ("20_sales_conversations.csv",         save_sales_conversations,     "goendalf666/sales-conversations-instruction-base",            "Open"),
        ("21_dota2_instruct.csv",              save_dota2_instruct,          "Aiden07/dota2_instruct_prompt",                              "MIT"),
        ("22_english_quotes_sentiment.csv",    save_english_quotes,          "Abirate/english_quotes",                                     "CC-BY-4.0"),
        ("23_synthetic_anomaly_logs.csv",      save_anomaly_dataset,         "Generated (data/generate_anomaly_data.py)",                  "N/A"),
    ]

    manifest_rows = []
    for i, (fname, fn, hf_id, license_) in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {fname}")
        fn()
        manifest_rows.append((fname, hf_id, license_))

    write_manifest(manifest_rows)

    # Final file list
    print("\n" + "=" * 60)
    print("Raw dataset files saved in data/raw/datasets/")
    print("=" * 60)
    for f in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f:<50} {size_kb:>8.1f} KB")
