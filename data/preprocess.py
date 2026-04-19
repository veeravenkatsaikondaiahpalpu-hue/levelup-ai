"""
preprocess.py - Fine-tuning data pipeline for the LevelUp AI chatbot.

Downloads and formats datasets into instruction-tuning JSONL pairs:
  {"system": <system_prompt>, "user": <message>, "assistant": <response>}

Sources (all free, no login needed):
  1.  HealthCareMagic-100k   lavita/ChatDoctor-HealthCareMagic-100k  (112k rows)
  2.  MentalChat16K          ShenLab/MentalChat16K                   (16k rows, MIT)
  3.  Fitness Q&A            its-myrto/fitness-question-answers       (965 rows)
  4.  Fitness Chat           chibbss/fitness-chat-prompt-completion-dataset (245 rows)
  5.  Mental Counseling      Amod/mental_health_counseling_conversations (3.5k rows)
  6.  NPC Dialogue           amaydle/npc-dialogue                    (1.9k rows)
  7.  English Quotes         Abirate/english_quotes                  (2.5k rows, CC-BY-4.0)
      → used for BERT sentiment weak-labelling, not chatbot fine-tuning
  8.  RPG Dialogue           local: data/raw/rpg_dialogues.jsonl     (custom)
  9.  XP Q&A                 built-in                                (13 pairs)

Output:
  data/raw/finetune_train.jsonl     — chatbot fine-tuning train split
  data/raw/finetune_val.jsonl       — chatbot fine-tuning val split
  data/raw/sentiment_dataset.jsonl  — BERT sentiment train data

Run (offline, no download):
  python data/preprocess.py --sources xp_qa rpg_dialogues

Run (download all):
  python data/preprocess.py --sources all

Run (selective):
  python data/preprocess.py --sources healthcare_magic mentalchat fitness_qa
"""

import json
import os
import random
import argparse
import sys

SEED = 42
random.seed(SEED)

OUTPUT_DIR = "data/raw"

# ── Wellness / fitness filter keywords ───────────────────────────────────────
WELLNESS_KEYWORDS = {
    "exercise", "workout", "gym", "fitness", "weight", "diet", "nutrition",
    "sleep", "stress", "anxiety", "meditation", "running", "yoga", "cardio",
    "protein", "calories", "muscle", "fatigue", "energy", "motivation",
    "mental health", "depression", "burnout", "healthy", "lifestyle",
    "hydration", "recovery", "stretch", "flexibility", "strength", "endurance",
}

# ── Generic system prompt for fine-tuning ────────────────────────────────────
GENERIC_SYSTEM = (
    "You are an AI companion for LevelUp, an RPG-inspired self-improvement app. "
    "You help users earn XP, track activities, maintain streaks, and stay motivated. "
    "Always stay in character as an encouraging RPG coach. "
    "Never give medical diagnoses. Keep answers concise and action-oriented."
)

# ── Sentiment tag mappings (for quotes dataset) ───────────────────────────────
MOTIVATED_TAGS  = {"inspirational", "motivation", "success", "courage", "achievement",
                   "confidence", "determination", "perseverance", "strength", "goals",
                   "winning", "victory", "champion", "power", "ambition"}
STRUGGLING_TAGS = {"sadness", "struggle", "grief", "depression", "pain", "failure",
                   "loss", "fear", "doubt", "despair", "difficult", "adversity"}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _try_load(dataset_id: str, **kwargs):
    """Wrapper around load_dataset with clear error output."""
    try:
        from datasets import load_dataset
        print(f"  Downloading {dataset_id} ...")
        return load_dataset(dataset_id, **kwargs)
    except Exception as e:
        print(f"  [SKIP] {dataset_id} failed: {e}")
        return None


# 1. HealthCareMagic ──────────────────────────────────────────────────────────

def load_healthcare_magic(max_samples: int = 3000):
    ds = _try_load("lavita/ChatDoctor-HealthCareMagic-100k",
                   split="train", streaming=True)
    if ds is None:
        return
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        user_msg = (row.get("input") or "").strip()
        reply    = (row.get("output") or "").strip()
        if not user_msg or not reply:
            continue
        if not any(kw in user_msg.lower() for kw in WELLNESS_KEYWORDS):
            continue
        yield {"system": GENERIC_SYSTEM, "user": user_msg, "assistant": reply}
        count += 1
        if count % 500 == 0:
            print(f"    ... {count} samples")
    print(f"  HealthCareMagic: {count} samples loaded.")


# 2. MentalChat16K ────────────────────────────────────────────────────────────

def load_mentalchat(max_samples: int = 5000):
    ds = _try_load("ShenLab/MentalChat16K", split="train")
    if ds is None:
        return
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        instruction = (row.get("instruction") or "").strip()
        user_msg    = (row.get("input") or "").strip()
        reply       = (row.get("output") or "").strip()
        if not reply:
            continue
        # Merge instruction + input as the user message
        full_user = f"{instruction} {user_msg}".strip() if user_msg else instruction
        if not full_user:
            continue
        yield {"system": GENERIC_SYSTEM, "user": full_user, "assistant": reply}
        count += 1
    print(f"  MentalChat16K: {count} samples loaded.")


# 3. Fitness Q&A ──────────────────────────────────────────────────────────────

def load_fitness_qa():
    ds = _try_load("its-myrto/fitness-question-answers", split="train")
    if ds is None:
        return
    count = 0
    for row in ds:
        q = (row.get("Question") or "").strip()
        a = (row.get("Answer") or "").strip()
        if q and a:
            yield {"system": GENERIC_SYSTEM, "user": q, "assistant": a}
            count += 1
    print(f"  Fitness Q&A: {count} samples loaded.")


# 4. Fitness Chat (instruction-format) ────────────────────────────────────────

def load_fitness_chat():
    ds = _try_load("chibbss/fitness-chat-prompt-completion-dataset", split="train")
    if ds is None:
        return
    count = 0
    for row in ds:
        instr = (row.get("instruction") or "").strip()
        out   = (row.get("output") or "").strip()
        if instr and out:
            yield {"system": GENERIC_SYSTEM, "user": instr, "assistant": out}
            count += 1
    print(f"  Fitness Chat: {count} samples loaded.")


# 5. Mental Health Counseling Conversations ───────────────────────────────────

def load_mental_counseling(max_samples: int = 2000):
    ds = _try_load("Amod/mental_health_counseling_conversations", split="train")
    if ds is None:
        return
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        ctx   = (row.get("Context") or "").strip()
        resp  = (row.get("Response") or "").strip()
        if ctx and resp:
            yield {"system": GENERIC_SYSTEM, "user": ctx, "assistant": resp}
            count += 1
    print(f"  Mental Counseling: {count} samples loaded.")


# 6. NPC Dialogue (RPG style injection) ───────────────────────────────────────

def load_npc_dialogue():
    ds = _try_load("amaydle/npc-dialogue", split="train")
    if ds is None:
        return
    count = 0
    for row in ds:
        query  = (row.get("Query") or "").strip()
        resp   = (row.get("Response") or "").strip()
        bio    = (row.get("Biography") or "").strip()
        if not query or not resp:
            continue
        # Use NPC biography as system context
        system = (
            f"{GENERIC_SYSTEM} "
            f"{'Speak in the style of this character: ' + bio if bio else ''}"
        ).strip()
        yield {"system": system, "user": query, "assistant": resp}
        count += 1
    print(f"  NPC Dialogue: {count} samples loaded.")


# 7. English Quotes → BERT Sentiment dataset ──────────────────────────────────

def load_sentiment_quotes(output_path: str = "data/raw/sentiment_dataset.jsonl"):
    """
    Loads quotes and weak-labels them for BERT sentiment training.
    Label map:
      motivated  — tags overlap with MOTIVATED_TAGS
      struggling — tags overlap with STRUGGLING_TAGS
      neutral    — everything else
    Writes directly to output_path (not returned as instruction pairs).
    """
    ds = _try_load("Abirate/english_quotes", split="train")
    if ds is None:
        return 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = {"motivated": 0, "neutral": 0, "struggling": 0}

    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            quote = (row.get("quote") or "").strip().strip('"').strip()
            tags  = set(t.lower().replace("-", " ") for t in (row.get("tags") or []))
            if not quote:
                continue

            if tags & MOTIVATED_TAGS:
                label = "motivated"
            elif tags & STRUGGLING_TAGS:
                label = "struggling"
            else:
                label = "neutral"

            f.write(json.dumps({"text": quote, "label": label}, ensure_ascii=False) + "\n")
            count[label] += 1

    total = sum(count.values())
    print(f"  English Quotes -> sentiment: {total} samples")
    print(f"    motivated={count['motivated']}, neutral={count['neutral']}, struggling={count['struggling']}")
    return total


# 8. Local RPG Dialogues ───────────────────────────────────────────────────────

def load_rpg_dialogues(path: str = "data/raw/rpg_dialogues.jsonl"):
    if os.path.exists(path):
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


# 9. Built-in XP Q&A ──────────────────────────────────────────────────────────

XP_QA_PAIRS = [
    ("What is the XP formula?",
     "XP = Duration (minutes) x Intensity Multiplier x Relevance Multiplier x Streak Multiplier. "
     "Light = 1.0x, Moderate = 1.5x, Intense = 2.0x. "
     "Primary activities = 1.0x relevance. Non-primary = 0.5x."),
    ("When does the streak multiplier kick in?",
     "Day 5: 1.5x (Power-Up). Day 10: 1.75x (Charged). Day 20: 2.0x (Blazing). "
     "Day 30: 2.25x (Unstoppable). Day 60: 2.5x (Legendary Streak)."),
    ("What is the daily XP cap?",
     "Level 1-10: 600 XP per day. Level 11-20: 800 XP. Level 21+: 1,000 XP. Resets at midnight."),
    ("How do streak shields work?",
     "You get 4 shields per month, auto-reset on the 1st. "
     "A shield is consumed when you miss exactly 1 day — no manual activation needed. "
     "Missing 2+ days resets the streak to 1."),
    ("How many XP do I need for each badge?",
     "Tier 1: 1,000 XP. Tier 2: 5,000. Tier 3: 15,000. "
     "Tier 4: 35,000. Tier 5: 75,000. Tier 6 (Legendary): 150,000 XP."),
    ("How does levelling up work?",
     "Level 1-10: 500 XP per level. Level 10-20: 1,500 XP per level. "
     "Level 20+: 3,000 XP per level."),
    ("Do I earn XP for activities outside my build?",
     "Yes, at 0.5x relevance (0.4x for the GAMER build side quests). "
     "Everything you do moves you forward."),
    ("What is the Combo Multiplier on the GAMER build?",
     "Each consecutive primary session in the same day adds a bonus: "
     "2nd session +0.2x, 3rd +0.3x, 4th+ +0.4x. Primary activities only."),
    ("What is Boss Day on the GAMER build?",
     "Every 7th streak day is Boss Day. Your daily XP cap doubles. "
     "At level 1-10 that means 1,200 XP instead of 600."),
    ("What is Overtime Mode on the GAMER build?",
     "After hitting the daily cap, primary GAMER activities draw from a 150 XP Overtime pool. "
     "It refills every midnight."),
    ("What is the Speedrun Bonus?",
     "Any intense primary activity in 25 minutes or less earns +15 flat XP."),
    ("What is the Side Quest Penalty on the GAMER build?",
     "Non-GAMER activities score at 0.4x relevance instead of the standard 0.5x."),
    ("How do I switch builds?",
     "Reach the Legendary badge (150,000 XP) first. "
     "Then choose: Option A — unlock a secondary build slot. "
     "Option B — archive primary to Hall of Fame and start fresh with 500 XP legacy bonus."),
]


def load_xp_qa():
    from xp_engine import BuildType as BT
    builds = [b.value for b in BT]
    for user_msg, assistant_msg in XP_QA_PAIRS:
        build = random.choice(builds)
        yield {
            "system":    GENERIC_SYSTEM + f" The user is on the {build} build.",
            "user":      user_msg,
            "assistant": assistant_msg,
        }


def _builtin_rpg_seed_dialogues():
    pairs = [
        ("I just hit level 5!",
         "LEVEL 5. You are no longer a rookie — you are a warrior in training. "
         "Every rep, every session, every log has built this. Keep the streak alive."),
        ("I haven't logged anything in 2 days. Am I going to lose my streak?",
         "Your streak shield activated automatically — one missed day is covered. "
         "You have used one shield. Log something today, even a 15-minute walk. "
         "Your streak is still intact. Let's keep it that way."),
        ("How is my XP calculated?",
         "XP = Duration x Intensity x Relevance x Streak multiplier. "
         "Example: 60-min intense gym on a Strength build with 5-day streak = "
         "60 x 2.0 x 1.0 x 1.5 = 180 XP."),
        ("What build should I choose?",
         "Tell me your main goal. Body — Strength or Dexterity. "
         "Mind — Intelligence. Peace — Wellness. Craft — Creative. "
         "Empire — Entrepreneur. Gaming is your life — GAMER build. What calls to you?"),
        ("I don't feel like doing anything today.",
         "Even a 10-minute walk earns XP and keeps your streak alive. "
         "You do not need your best session. You just need to show up."),
        ("I just earned the Scholar badge!",
         "Scholar. That is 5,000 XP of real studying. "
         "Every page, every course compounded into that title. "
         "Next rank is Sage — 10,000 XP to go."),
        ("Why did I stop earning XP?",
         "You have hit your daily XP cap. Resets at midnight. "
         "Come back tomorrow and keep the streak alive."),
        ("I just hit my 7-day streak as a Gamer. What is Boss Day?",
         "BOSS DAY ACTIVATED. Your daily XP cap DOUBLES — 600 becomes 1,200. "
         "Stack with your Combo Multiplier and you are looking at massive gains. Today is big."),
        ("Can I change my build?",
         "Build switching unlocks at Legendary badge — 150,000 XP. "
         "Option A: unlock secondary slot. Option B: archive to Hall of Fame, "
         "start fresh with 500 XP legacy bonus. Your badges are yours forever."),
        ("What should I do today to maximise XP on my Strength build?",
         "Primary activities: gym session, running, swimming, sports. "
         "60 min intense = 120 base XP. With a 5-day streak: 180 XP. Hit the gym."),
        ("I've been feeling really stressed lately.",
         "Stress is a signal. On the Wellness build, meditation and journaling are primary — "
         "20 minutes earns real XP. What feels manageable right now?"),
    ]
    from xp_engine import BuildType as BT
    builds = [b.value for b in BT]
    for user_msg, assistant_msg in pairs:
        build = random.choice(builds)
        yield {
            "system":    GENERIC_SYSTEM + f" The user is on the {build} build.",
            "user":      user_msg,
            "assistant": assistant_msg,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Assembler
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_MAP = {
    "healthcare_magic":  load_healthcare_magic,
    "mentalchat":        load_mentalchat,
    "fitness_qa":        load_fitness_qa,
    "fitness_chat":      load_fitness_chat,
    "mental_counseling": load_mental_counseling,
    "npc_dialogue":      load_npc_dialogue,
    "rpg_dialogues":     load_rpg_dialogues,
    "xp_qa":             load_xp_qa,
}

ALL_SOURCES = list(SOURCE_MAP.keys())


def write_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_dataset(
    sources: list[str] = None,
    max_hcm_samples: int = 3000,
    val_ratio: float = 0.10,
    include_sentiment: bool = True,
) -> tuple[str, str]:
    if sources is None or "all" in sources:
        sources = ALL_SOURCES

    records: list[dict] = []

    for i, source in enumerate(sources, 1):
        if source not in SOURCE_MAP:
            print(f"  [WARN] Unknown source '{source}' — skipping.")
            continue
        print(f"\n[{i}/{len(sources)}] {source}")
        records.extend(SOURCE_MAP[source]())

    if include_sentiment and ("all" in (sources or []) or "sentiment_quotes" in (sources or [])):
        print(f"\n[+] Building sentiment dataset from English Quotes ...")
        load_sentiment_quotes()

    print(f"\nTotal samples collected: {len(records):,}")
    if not records:
        print("No records — check your sources or internet connection.")
        return "", ""

    random.shuffle(records)
    split_idx   = int(len(records) * (1 - val_ratio))
    train_recs  = records[:split_idx]
    val_recs    = records[split_idx:]

    train_path = os.path.join(OUTPUT_DIR, "finetune_train.jsonl")
    val_path   = os.path.join(OUTPUT_DIR, "finetune_val.jsonl")

    write_jsonl(train_recs, train_path)
    write_jsonl(val_recs,   val_path)

    print(f"\nDataset written:")
    print(f"  Train : {train_path}  ({len(train_recs):,} samples)")
    print(f"  Val   : {val_path}  ({len(val_recs):,} samples)")
    return train_path, val_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(description="LevelUp fine-tuning data pipeline")
    parser.add_argument(
        "--sources", nargs="+", default=["xp_qa", "rpg_dialogues"],
        help=f"Sources to include. Options: {ALL_SOURCES + ['all']}",
    )
    parser.add_argument("--max_hcm", type=int, default=3000,
                        help="Max samples from HealthCareMagic")
    parser.add_argument("--no_sentiment", action="store_true",
                        help="Skip building sentiment dataset from quotes")
    args = parser.parse_args()

    print("=" * 60)
    print("LevelUp AI — Dataset Pipeline")
    print("=" * 60)
    print(f"Sources: {args.sources}")

    train, val = build_dataset(
        sources=args.sources,
        max_hcm_samples=args.max_hcm,
        include_sentiment=not args.no_sentiment,
    )

    if train:
        print("\nPreview (first training sample):")
        with open(train) as f:
            s = json.loads(f.readline())
        print(f"  system   : {s['system'][:90]}...")
        print(f"  user     : {s['user'][:80]}")
        print(f"  assistant: {s['assistant'][:80]}...")
        print("\nDone. Ready for QLoRA fine-tuning.")
