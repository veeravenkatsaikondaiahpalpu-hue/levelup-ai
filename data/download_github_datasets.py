"""
download_github_datasets.py
Downloads the top 3 research datasets from GitHub into data/raw/staging/github/
These are saved RAW for inspection — NOT added to the fine-tuning pipeline yet.

Datasets:
  1. CIMA      — Real tutor-student dialogue (ACL 2020, CC 2.5)
  2. MathDial  — Teacher scaffolding dialogue (EMNLP 2023, research)
  3. FIREBALL  — 25k real D&D sessions from Discord (ACL 2023, research)

Run:
    PYTHONPATH=. venv/Scripts/python.exe data/download_github_datasets.py
"""

import os
import json
import csv
import zipfile
import tarfile
import requests
import shutil

STAGING = os.path.join("data", "raw", "staging", "github")
os.makedirs(STAGING, exist_ok=True)


def download_file(url, dest_path, desc=""):
    print(f"  Downloading {desc or url} ...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    wrote = 0
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            wrote += len(chunk)
            if total:
                pct = wrote / total * 100
                print(f"\r    {pct:.1f}%  ({wrote/1024/1024:.1f} MB)", end="", flush=True)
    print()
    return dest_path


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CIMA — Tutoring Dialogues (ACL 2020)
#    https://github.com/kstats/CIMA
#    Format: JSON files with multi-turn tutor-student sessions
# ═══════════════════════════════════════════════════════════════════════════════

def download_cima():
    out_dir = os.path.join(STAGING, "01_cima_tutoring")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("1. CIMA — Tutor-Student Dialogue (ACL 2020)")
    print("=" * 60)

    # Download the zip of the full repo
    zip_url  = "https://github.com/kstats/CIMA/archive/refs/heads/master.zip"
    zip_path = os.path.join(out_dir, "cima_repo.zip")
    download_file(zip_url, zip_path, "CIMA GitHub repo")

    # Extract
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    os.remove(zip_path)

    # Find JSON data files and flatten into one JSONL for easy inspection
    extract_dir = os.path.join(out_dir, "CIMA-master")
    all_sessions = []
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            if fname.endswith(".json") and "data" in root.lower():
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_sessions.extend(data)
                    elif isinstance(data, dict):
                        all_sessions.append(data)
                except Exception:
                    pass

    # Save as JSONL for easy inspection
    out_jsonl = os.path.join(out_dir, "cima_sessions.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for sess in all_sessions:
            f.write(json.dumps(sess, ensure_ascii=False) + "\n")

    # Also save a 20-row preview CSV
    preview_path = os.path.join(out_dir, "cima_PREVIEW_20rows.csv")
    previews = []
    for sess in all_sessions[:100]:
        turns = sess.get("turns") or sess.get("utterances") or sess.get("dialogue") or []
        if isinstance(turns, list):
            for i in range(len(turns) - 1):
                u = turns[i]
                v = turns[i + 1]
                role_u = str(u.get("role", u.get("speaker", ""))).lower()
                role_v = str(v.get("role", v.get("speaker", ""))).lower()
                text_u = str(u.get("text", u.get("utterance", u.get("content", "")))).strip()
                text_v = str(v.get("text", v.get("utterance", v.get("content", "")))).strip()
                if text_u and text_v:
                    previews.append({"role_user": role_u, "user": text_u,
                                     "role_asst": role_v, "assistant": text_v})
                if len(previews) >= 20:
                    break
        if len(previews) >= 20:
            break

    if previews:
        with open(preview_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["role_user","user","role_asst","assistant"])
            writer.writeheader()
            writer.writerows(previews)

    print(f"  Sessions extracted : {len(all_sessions):,}")
    print(f"  JSONL saved        : {out_jsonl}")
    print(f"  Preview CSV        : {preview_path}")
    print(f"  Full repo          : {extract_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MathDial — Teacher Scaffolding Dialogue (EMNLP 2023)
#    https://github.com/eth-nlped/mathdial
#    Format: CSV files (train.csv, test.csv) — ready to use
# ═══════════════════════════════════════════════════════════════════════════════

def download_mathdial():
    out_dir = os.path.join(STAGING, "02_mathdial_tutoring")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("2. MathDial — Teacher-Student Maths Tutoring (EMNLP 2023)")
    print("=" * 60)

    # Download repo zip
    zip_url  = "https://github.com/eth-nlped/mathdial/archive/refs/heads/main.zip"
    zip_path = os.path.join(out_dir, "mathdial_repo.zip")
    download_file(zip_url, zip_path, "MathDial GitHub repo")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    os.remove(zip_path)

    # Find CSV files
    extract_dir = os.path.join(out_dir, "mathdial-main")
    csv_files_found = []
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            if fname.endswith(".csv") or fname.endswith(".jsonl"):
                src = os.path.join(root, fname)
                dst = os.path.join(out_dir, fname)
                shutil.copy2(src, dst)
                csv_files_found.append(dst)
                print(f"  Copied: {fname}")

    # Count rows
    for csv_path in csv_files_found:
        if csv_path.endswith(".csv"):
            try:
                with open(csv_path, encoding="utf-8") as f:
                    rows = sum(1 for _ in f) - 1
                print(f"  {os.path.basename(csv_path)}: {rows:,} rows")
            except Exception:
                pass

    # Print first row as preview
    for csv_path in csv_files_found:
        if "train" in csv_path.lower() and csv_path.endswith(".csv"):
            try:
                with open(csv_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    cols = reader.fieldnames
                    row1 = next(reader)
                print(f"\n  Columns: {cols}")
                print(f"  Sample row:")
                for k, v in row1.items():
                    print(f"    {k}: {str(v)[:100]}")
            except Exception as e:
                print(f"  Preview error: {e}")
            break

    print(f"  Full repo: {extract_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FIREBALL — Real D&D Sessions from Discord (ACL 2023)
#    https://github.com/zhudotexe/FIREBALL
#    Format: JSONL — game state + dialogue turns
#    Note: Full dataset is ~25k sessions hosted via Zenodo/HF.
#          We download the sample/dev split first for inspection.
# ═══════════════════════════════════════════════════════════════════════════════

def download_fireball():
    out_dir = os.path.join(STAGING, "03_fireball_dnd")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("3. FIREBALL — Real D&D Sessions (ACL 2023)")
    print("=" * 60)

    # Download repo for README, processing scripts, and sample data
    zip_url  = "https://github.com/zhudotexe/FIREBALL/archive/refs/heads/main.zip"
    zip_path = os.path.join(out_dir, "fireball_repo.zip")
    download_file(zip_url, zip_path, "FIREBALL GitHub repo (scripts + samples)")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    os.remove(zip_path)

    extract_dir = os.path.join(out_dir, "FIREBALL-main")

    # Look for any sample/data files in the repo
    sample_files = []
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            if fname.endswith((".jsonl", ".json", ".csv", ".txt")):
                fpath = os.path.join(root, fname)
                size_kb = os.path.getsize(fpath) / 1024
                sample_files.append((fname, fpath, size_kb))
                print(f"  Found: {fname}  ({size_kb:.1f} KB)")

    # Try downloading the dev/sample split from HuggingFace (same data, smaller split)
    # FIREBALL is also on HF: https://huggingface.co/datasets/zhudotexe/FIREBALL
    print("\n  Downloading FIREBALL sample split from HuggingFace ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("zhudotexe/FIREBALL", split="train", streaming=True)
        rows = []
        print("  Collecting first 500 sessions for inspection ...")
        for i, row in enumerate(ds):
            rows.append(row)
            if i >= 499:
                break

        # Save as JSONL
        sample_path = os.path.join(out_dir, "fireball_sample_500.jsonl")
        with open(sample_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"  Sample JSONL saved: {sample_path}  ({len(rows)} sessions)")

        # Preview — show structure of one session
        if rows:
            row0 = rows[0]
            preview_path = os.path.join(out_dir, "fireball_STRUCTURE_PREVIEW.txt")
            with open(preview_path, "w", encoding="utf-8") as f:
                f.write("FIREBALL — Session Structure Preview\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Top-level keys: {list(row0.keys())}\n\n")
                for k, v in row0.items():
                    if isinstance(v, list):
                        f.write(f"{k} (list, len={len(v)}):\n")
                        if v:
                            f.write(f"  [0]: {json.dumps(v[0], default=str)[:300]}\n")
                    else:
                        f.write(f"{k}: {str(v)[:200]}\n")
            print(f"  Structure preview: {preview_path}")
            print(f"  Session keys: {list(row0.keys())}")

    except Exception as e:
        print(f"  HF sample download failed: {e}")
        print("  The full dataset is at: https://huggingface.co/datasets/zhudotexe/FIREBALL")
        print("  Manual download: huggingface-cli download zhudotexe/FIREBALL")

    print(f"  Full repo + scripts: {extract_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Also grab CRD3 directly from GitHub (bypasses the HF deprecated script issue)
# ═══════════════════════════════════════════════════════════════════════════════

def download_crd3_github():
    out_dir = os.path.join(STAGING, "04_crd3_critical_role")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("4. CRD3 — Critical Role D&D (ACL 2020) [GitHub direct]")
    print("=" * 60)

    zip_url  = "https://github.com/RevanthRameshkumar/CRD3/archive/refs/heads/master.zip"
    zip_path = os.path.join(out_dir, "crd3_repo.zip")
    download_file(zip_url, zip_path, "CRD3 GitHub repo")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    os.remove(zip_path)

    extract_dir = os.path.join(out_dir, "CRD3-master")

    # Inventory what's in the repo
    all_files = []
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            size_kb = os.path.getsize(fpath) / 1024
            all_files.append((fname, fpath, size_kb))
            if fname.endswith((".json", ".jsonl", ".csv", ".txt")) and size_kb > 1:
                print(f"  Data file: {os.path.relpath(fpath, extract_dir)}  ({size_kb:.1f} KB)")

    # Count total rows in JSON data files
    total_turns = 0
    sample_rows = []
    for fname, fpath, size_kb in all_files:
        if fname.endswith(".json") and size_kb > 10:
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    total_turns += len(data)
                    sample_rows.extend(data[:3])
            except Exception:
                pass

    if total_turns:
        print(f"\n  Total turns/records found: {total_turns:,}")

    # Save structure preview
    if sample_rows:
        preview_path = os.path.join(out_dir, "crd3_PREVIEW_3rows.json")
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(sample_rows[:3], f, indent=2, ensure_ascii=False)
        print(f"  Preview saved: {preview_path}")
        print(f"  Sample keys: {list(sample_rows[0].keys()) if sample_rows else 'N/A'}")

    print(f"  Full repo: {extract_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LevelUp AI — Downloading GitHub Research Datasets")
    print("Saving to: data/raw/staging/github/")
    print("NOTE: These are for INSPECTION only — not yet in the pipeline.")
    print("=" * 60)

    download_cima()
    download_mathdial()
    download_fireball()
    download_crd3_github()

    print("\n" + "=" * 60)
    print("All downloads complete.")
    print("Inspect files in: data/raw/staging/github/")
    print("=" * 60)
    for d in sorted(os.listdir(STAGING)):
        dpath = os.path.join(STAGING, d)
        if os.path.isdir(dpath):
            total_kb = sum(
                os.path.getsize(os.path.join(r, f)) / 1024
                for r, _, fs in os.walk(dpath)
                for f in fs
            )
            print(f"  {d:<45} {total_kb/1024:>6.1f} MB")
