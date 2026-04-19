"""
preprocess.py - Fine-tuning data pipeline for the LevelUp AI chatbot.

Downloads and formats datasets into instruction-tuning JSONL pairs:
  {"system": <system_prompt>, "user": <message>, "assistant": <response>}

Sources:
  1. HealthCareMagic-100k  (HuggingFace: lavita/ChatDoctor-HealthCareMagic-100k)
     → filtered to wellness / fitness / non-diagnostic questions only
  2. Synthetic RPG Dialogue (local: data/raw/rpg_dialogues.jsonl)
     → custom RPG-style coaching conversations (generated separately)
  3. Built-in XP Q&A       (hardcoded: data/xp_qa_pairs.py)
     → covers every XP mechanic the chatbot must know

Output:
  data/raw/finetune_train.jsonl   — training split (~90%)
  data/raw/finetune_val.jsonl     — validation split (~10%)

Run:
  python data/preprocess.py --sources all --max_samples 5000
  python data/preprocess.py --sources xp_qa  (offline, no download needed)
"""

import json
import os
import random
import argparse
from typing import Iterator

SEED = 42
random.seed(SEED)

OUTPUT_DIR = "data/raw"

# ── Wellness / fitness keywords to filter HealthCareMagic ────────────────────
WELLNESS_KEYWORDS = {
    "exercise", "workout", "gym", "fitness", "weight", "diet", "nutrition",
    "sleep", "stress", "anxiety", "meditation", "running", "yoga", "cardio",
    "protein", "calories", "muscle", "fatigue", "energy", "motivation",
    "mental health", "depression", "burnout", "healthy", "lifestyle",
}

# ── Generic system prompt for fine-tuning (build-neutral) ────────────────────
GENERIC_SYSTEM = (
    "You are an AI companion for LevelUp, an RPG-inspired self-improvement app. "
    "You help users earn XP, track activities, maintain streaks, and stay motivated. "
    "Always stay in character as an encouraging RPG coach. "
    "Never give medical diagnoses. Keep answers concise and action-oriented."
)


# ─────────────────────────────────────────────────────────────────────────────
# Source 1: HealthCareMagic (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

def load_healthcare_magic(max_samples: int = 2000) -> Iterator[dict]:
    """
    Streams HealthCareMagic-100k from HuggingFace, filters to wellness/fitness
    questions, and yields instruction-tuning pairs.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [SKIP] datasets not installed. Run: pip install datasets")
        return

    print("  Downloading lavita/ChatDoctor-HealthCareMagic-100k ...")
    try:
        ds = load_dataset(
            "lavita/ChatDoctor-HealthCareMagic-100k",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  [SKIP] Could not load HealthCareMagic: {e}")
        return

    count = 0
    for row in ds:
        if count >= max_samples:
            break

        patient_msg = (row.get("input") or "").strip()
        doctor_resp = (row.get("output") or "").strip()

        if not patient_msg or not doctor_resp:
            continue

        # Filter: keep only wellness / fitness questions
        msg_lower = patient_msg.lower()
        if not any(kw in msg_lower for kw in WELLNESS_KEYWORDS):
            continue

        # Reframe doctor response to sound like an AI fitness coach
        # (just use as-is — the LLM will learn the wellness domain;
        #  RPG style comes from the synthetic dialogues)
        yield {
            "system":    GENERIC_SYSTEM,
            "user":      patient_msg,
            "assistant": doctor_resp,
        }
        count += 1
        if count % 500 == 0:
            print(f"    ... {count} wellness samples collected")

    print(f"  HealthCareMagic: {count} samples loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: Synthetic RPG Dialogues
# ─────────────────────────────────────────────────────────────────────────────

def load_rpg_dialogues(path: str = "data/raw/rpg_dialogues.jsonl") -> Iterator[dict]:
    """
    Loads custom RPG dialogue pairs from a local JSONL file.
    Each line: {"system": ..., "user": ..., "assistant": ...}
    File is generated separately (by GPT-4 or manually).
    Falls back to built-in seed dialogues if the file doesn't exist.
    """
    if os.path.exists(path):
        print(f"  Loading RPG dialogues from {path} ...")
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        continue
        print(f"  RPG dialogues: {count} samples loaded.")
    else:
        print(f"  [INFO] {path} not found — using built-in seed dialogues.")
        yield from _builtin_rpg_seed_dialogues()


def _builtin_rpg_seed_dialogues() -> Iterator[dict]:
    """
    Hardcoded seed RPG dialogues — covers the most common chatbot interactions.
    These teach the model its personality and RPG framing.
    Expand this list or replace with the full GPT-4 generated set.
    """
    pairs = [
        # ── Level Up ─────────────────────────────────────────────────────────
        ("I just hit level 5!",
         "LEVEL 5. You are no longer a rookie — you are a warrior in training. "
         "Every rep, every session, every log has built this. "
         "The next milestone is Level 10. Keep the streak alive."),

        # ── Streak Warning ────────────────────────────────────────────────────
        ("I haven't logged anything in 2 days. Am I going to lose my streak?",
         "Your streak shield activated automatically — one missed day is covered. "
         "But you have used one shield. You have a limited number per month, so "
         "log something today, even a 15-minute walk. Anything counts. "
         "Your streak is still intact. Let's keep it that way."),

        # ── XP Question ───────────────────────────────────────────────────────
        ("How is my XP calculated?",
         "Here is the formula: XP = Duration (min) x Intensity x Relevance x Streak multiplier. "
         "For example: 60-min intense gym session on a Strength build with a 5-day streak = "
         "60 x 2.0 x 1.0 x 1.5 = 180 XP. "
         "Primary activities for your build get 1.0x relevance. "
         "Everything else gets 0.5x. Keep that streak going for bigger multipliers."),

        # ── Build Question ────────────────────────────────────────────────────
        ("What build should I choose?",
         "Tell me what you want to improve most this year. "
         "If it is your body — Strength or Dexterity. "
         "Your mind — Intelligence. "
         "Your peace — Wellness. "
         "Your craft — Creative. "
         "Your empire — Entrepreneur. "
         "Or if gaming IS your thing — the GAMER build has its own unique mechanics. "
         "What calls to you?"),

        # ── Motivation ────────────────────────────────────────────────────────
        ("I don't feel like doing anything today.",
         "That feeling is real. But here is the truth: "
         "even a 10-minute walk earns XP and keeps your streak alive. "
         "You do not need your best session today. You just need to show up. "
         "Log one small thing. That is the move."),

        # ── Badge Unlock ──────────────────────────────────────────────────────
        ("I just earned the Scholar badge!",
         "Scholar. That is not just a badge — that is 5,000 XP of real studying. "
         "Every page, every course, every session compounded into that title. "
         "The next rank is Sage. You have 10,000 XP to go. The mind keeps growing."),

        # ── Daily Cap ────────────────────────────────────────────────────────
        ("Why did I stop earning XP?",
         "You have hit your daily XP cap. "
         "This resets at midnight. "
         "The cap exists to reward consistency over grinding — "
         "it is better to show up every day than to burn out in one session. "
         "Come back tomorrow and keep the streak alive."),

        # ── Gamer Build ───────────────────────────────────────────────────────
        ("I just hit my 7-day streak as a Gamer. What is Boss Day?",
         "BOSS DAY ACTIVATED. Every 7th streak day your daily XP cap DOUBLES. "
         "Normally 600 XP max — today you can earn 1,200. "
         "Stack it with your Combo Multiplier from back-to-back sessions and "
         "you are looking at massive gains. "
         "Today is a big day. Do not waste it."),

        # ── Build Switch ──────────────────────────────────────────────────────
        ("Can I change my build?",
         "Build switching unlocks at the Legendary badge — 150,000 XP. "
         "You have two options once you get there: "
         "Option A: keep your primary build and unlock a secondary slot. "
         "Option B: archive your current build to the Hall of Fame and start fresh "
         "with a 500 XP legacy bonus on the new build. "
         "Either way, your badges are yours forever."),

        # ── Workout Suggestion ────────────────────────────────────────────────
        ("What should I do today to maximise XP on my Strength build?",
         "For Strength, primary activities give you 1.0x relevance. "
         "Best options for max XP: gym session, running, swimming, or sports. "
         "Go intense (2.0x multiplier) for 60 minutes and you earn 120 base XP. "
         "If your streak is 5+ days you get a Power-Up bonus on top of that. "
         "Hit the gym. Make it count."),

        # ── Wellness ─────────────────────────────────────────────────────────
        ("I've been feeling really stressed lately.",
         "Stress is a signal, not a weakness. "
         "On the Wellness build, meditation and journaling are primary activities — "
         "even 20 minutes earns real XP. "
         "Your body is asking for recovery. That is progress too. "
         "What feels manageable right now — a short walk, 5 minutes of breathing, or just journaling?"),
    ]

    builds = ["strength", "intelligence", "dexterity", "wellness", "creative", "entrepreneur", "gamer"]
    for user_msg, assistant_msg in pairs:
        build = random.choice(builds)
        yield {
            "system":    GENERIC_SYSTEM + f" The user is on the {build} build.",
            "user":      user_msg,
            "assistant": assistant_msg,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Source 3: XP Mechanic Q&A (built-in, always available)
# ─────────────────────────────────────────────────────────────────────────────

XP_QA_PAIRS = [
    # Formula
    ("What is the XP formula?",
     "XP = Duration (minutes) x Intensity Multiplier x Relevance Multiplier x Streak Multiplier. "
     "Light intensity = 1.0x, Moderate = 1.5x, Intense = 2.0x. "
     "Primary activities for your build = 1.0x relevance. Non-primary = 0.5x."),

    # Streak
    ("When does the streak multiplier kick in?",
     "Day 5: 1.5x (Power-Up). Day 10: 1.75x (Charged). Day 20: 2.0x (Blazing). "
     "Day 30: 2.25x (Unstoppable). Day 60: 2.5x (Legendary Streak)."),

    # Daily cap
    ("What is the daily XP cap?",
     "Level 1-10: 600 XP per day. Level 11-20: 800 XP. Level 21+: 1000 XP. "
     "The cap resets at midnight every day."),

    # Shields
    ("How do streak shields work?",
     "You get 4 shields per month, reset on the 1st. "
     "A shield is automatically consumed when you miss exactly 1 day — "
     "no manual activation needed. "
     "If you miss 2 or more days, the streak resets to 1."),

    # Badges
    ("How many XP do I need for each badge?",
     "Tier 1: 1,000 XP. Tier 2: 5,000. Tier 3: 15,000. "
     "Tier 4: 35,000. Tier 5: 75,000. Tier 6 (Legendary): 150,000 XP."),

    # Level progression
    ("How does levelling up work?",
     "Level 1-10: 500 XP per level. Level 10-20: 1,500 XP per level. "
     "Level 20+: 3,000 XP per level. "
     "Levels get harder over time — just like a real RPG."),

    # Non-primary
    ("Do I earn XP for activities outside my build?",
     "Yes, but at 0.5x relevance. If you are on the Strength build and you meditate, "
     "you still earn XP — just half the normal rate. "
     "Everything you do moves you forward."),

    # GAMER combo
    ("What is the Combo Multiplier on the GAMER build?",
     "Each consecutive primary GAMER session in the same day adds a bonus: "
     "2nd session +0.2x, 3rd +0.3x, 4th and beyond +0.4x. "
     "Log multiple sessions in a day to stack the combo."),

    # GAMER boss day
    ("What is Boss Day on the GAMER build?",
     "Every 7th streak day is Boss Day. Your daily XP cap doubles for that day. "
     "At level 1-10 that means 1,200 XP instead of 600. Use it."),

    # GAMER overtime
    ("What is Overtime Mode on the GAMER build?",
     "After you hit the daily XP cap, primary GAMER activities can still earn "
     "from a separate 150 XP Overtime pool. It refills every midnight. "
     "So you can keep grinding even after the cap."),

    # GAMER speedrun
    ("What is the Speedrun Bonus on the GAMER build?",
     "Any intense primary activity completed in 25 minutes or less earns +15 flat XP. "
     "Great for quick warm-up drills, APM training sessions, or fast puzzle runs."),

    # Build switch
    ("How do I switch builds?",
     "You need to reach the Legendary badge first — that is 150,000 XP. "
     "Then you have two options: unlock a secondary build slot while keeping your primary, "
     "or archive your primary to the Hall of Fame and start fresh with a 500 XP legacy bonus."),
]


def load_xp_qa() -> Iterator[dict]:
    """Yields the built-in XP Q&A pairs with varied system prompts per build."""
    from xp_engine import BuildType as BT
    builds = [b.value for b in BT]
    for user_msg, assistant_msg in XP_QA_PAIRS:
        build = random.choice(builds)
        yield {
            "system":    GENERIC_SYSTEM + f" The user is on the {build} build.",
            "user":      user_msg,
            "assistant": assistant_msg,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Assembler + Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_dataset(
    sources: list[str] = None,
    max_hcm_samples: int = 2000,
    val_ratio: float = 0.10,
) -> tuple[str, str]:
    """
    Assemble the fine-tuning dataset from selected sources, shuffle,
    and write train/val JSONL splits.

    sources options: "healthcare_magic", "rpg_dialogues", "xp_qa", "all"
    """
    if sources is None or "all" in sources:
        sources = ["healthcare_magic", "rpg_dialogues", "xp_qa"]

    records: list[dict] = []

    if "xp_qa" in sources:
        print("\n[1/3] Loading XP Q&A pairs ...")
        records.extend(load_xp_qa())

    if "rpg_dialogues" in sources:
        print("\n[2/3] Loading RPG dialogues ...")
        records.extend(load_rpg_dialogues())

    if "healthcare_magic" in sources:
        print("\n[3/3] Loading HealthCareMagic ...")
        records.extend(load_healthcare_magic(max_samples=max_hcm_samples))

    print(f"\nTotal samples before split: {len(records):,}")

    random.shuffle(records)

    split_idx   = int(len(records) * (1 - val_ratio))
    train_recs  = records[:split_idx]
    val_recs    = records[split_idx:]

    train_path  = os.path.join(OUTPUT_DIR, "finetune_train.jsonl")
    val_path    = os.path.join(OUTPUT_DIR, "finetune_val.jsonl")

    write_jsonl(train_recs, train_path)
    write_jsonl(val_recs,   val_path)

    print(f"\nDataset written:")
    print(f"  Train : {train_path}  ({len(train_recs):,} samples)")
    print(f"  Val   : {val_path}  ({len(val_recs):,} samples)")

    return train_path, val_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LevelUp fine-tuning data pipeline")
    parser.add_argument(
        "--sources", nargs="+",
        default=["xp_qa", "rpg_dialogues"],
        choices=["healthcare_magic", "rpg_dialogues", "xp_qa", "all"],
        help="Which sources to include (default: xp_qa rpg_dialogues)"
    )
    parser.add_argument(
        "--max_hcm", type=int, default=2000,
        help="Max samples from HealthCareMagic (default: 2000)"
    )
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    train, val = build_dataset(sources=args.sources, max_hcm_samples=args.max_hcm)
    print("\nPreview of first training sample:")
    with open(train) as f:
        sample = json.loads(f.readline())
    print(f"  system   : {sample['system'][:80]}...")
    print(f"  user     : {sample['user'][:80]}")
    print(f"  assistant: {sample['assistant'][:80]}...")
    print("\nDone. Ready for QLoRA fine-tuning.")
