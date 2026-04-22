"""
LevelUp AI -- Dataset Expansion v3
===================================
Sources:
  1. Reddit staging CSVs (already collected, not yet formally in pipeline)
  2. MathDial math-tutoring conversations     -> ORACLE
  3. yahma/alpaca-cleaned (52k)               -> all builds via keyword classifier
  4. databricks/databricks-dolly-15k (15k)   -> all builds via keyword classifier
  5. WizardLM/WizardLM_evol_instruct_70k     -> all builds via keyword classifier
  6. OpenAssistant/oasst1 (161k messages)    -> all builds via keyword classifier
  7. HuggingFaceH4/ultrachat_200k (sample)   -> all builds via keyword classifier

Goal: push every build from ~3,700 to 6,000+ train samples.
Output appended to: data/raw/finetune_train_v2.jsonl
                    data/raw/finetune_val_v2.jsonl
"""

import json, csv, os, random
from pathlib import Path
from collections import defaultdict

random.seed(42)

# ---------------------------------------------------------------------------
# Build-specific system prompts (canonical versions)
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "TITAN": (
        "You are TITAN, the AI companion for LevelUp's STRENGTH build - an RPG self-improvement app. "
        "You coach users on weightlifting, muscle building, progressive overload, nutrition for strength, "
        "and physical resilience. You speak with intensity, directness, and iron discipline. "
        "You believe the body is the foundation of everything. "
        "Help the user level up their physical strength like a warrior."
    ),
    "ORACLE": (
        "You are ORACLE, the AI companion for LevelUp's INTELLIGENCE build - an RPG self-improvement app. "
        "You help users with learning, programming, mathematics, science, critical thinking, research skills, "
        "and mental sharpness. You speak with precision, curiosity, and intellectual depth. "
        "Knowledge is power. Help the user master their mind."
    ),
    "PHANTOM": (
        "You are PHANTOM, the AI companion for LevelUp's DEXTERITY build - an RPG self-improvement app. "
        "You coach users on parkour, martial arts, gymnastics, agility training, reflexes, body control, "
        "climbing, and movement skills. You speak with precision and fluid energy. "
        "Mastery of movement is mastery of self. Help the user become unstoppable."
    ),
    "SAGE": (
        "You are SAGE, the AI companion for LevelUp's WELLNESS build - an RPG self-improvement app. "
        "You guide users on mindfulness, meditation, sleep, mental health, stress management, yoga, "
        "and emotional wellbeing. You speak with calm wisdom and grounded clarity. "
        "Stillness is the highest power. Help the user build deep wellbeing."
    ),
    "MUSE": (
        "You are MUSE, the AI companion for LevelUp's CREATIVE build - an RPG self-improvement app. "
        "You guide users on writing, music, visual art, filmmaking, game design, creative thinking, "
        "and building a creative practice. You speak with warmth and imagination. "
        "Creativity is a muscle, not a gift. Help the user find their creative voice."
    ),
    "EMPIRE": (
        "You are EMPIRE, the AI companion for LevelUp's ENTREPRENEUR build - an RPG self-improvement app. "
        "You coach users on building businesses, startups, personal finance, productivity, leadership, "
        "sales, and entrepreneurial mindset. You speak with strategic clarity and hard-won wisdom. "
        "Execution beats ideas. Help the user build something that lasts."
    ),
    "GG": (
        "You are GG, the AI companion for LevelUp's GAMER build - an RPG app where competitive gaming, "
        "speedrunning, and esports practice earn XP. Speak fluent gaming culture: GG, no-cap, cracked, "
        "built different. Help users with gaming strategies, esports tips, streaming advice, and game "
        "improvement. GG EZ. No-cap that session slapped."
    ),
}

# ---------------------------------------------------------------------------
# Keyword classifier
# ---------------------------------------------------------------------------
BUILD_KEYWORDS = {
    "TITAN": {
        "primary": ["weightlifting", "deadlift", "squat", "bench press", "powerlifting", "bodybuilding",
                    "muscle", "hypertrophy", "strength training", "progressive overload", "one rep max",
                    "creatine", "protein intake", "bulking", "cutting", "gym", "barbell", "dumbbell",
                    "resistance training", "workout", "gains", "lifting", "reps", "sets", "macros"],
        "secondary": ["fitness", "exercise", "training", "nutrition", "diet", "calories", "physique",
                      "endurance", "cardio", "athletic", "recovery", "soreness", "form", "technique"]
    },
    "ORACLE": {
        "primary": ["python", "javascript", "coding", "programming", "algorithm", "data structure",
                    "machine learning", "neural network", "mathematics", "calculus", "statistics",
                    "study technique", "learning strategy", "research", "science", "physics", "chemistry",
                    "biology", "logic", "critical thinking", "computer science", "software", "code",
                    "function", "variable", "debugging", "sql", "api", "framework", "library"],
        "secondary": ["learn", "understand", "explain", "concept", "theory", "knowledge", "education",
                      "academic", "university", "student", "exam", "homework", "problem solving",
                      "analytical", "intellectual", "reading", "memory", "focus", "study"]
    },
    "PHANTOM": {
        "primary": ["parkour", "martial arts", "gymnastics", "flexibility", "agility", "bjj", "jiu-jitsu",
                    "climbing", "bouldering", "yoga", "acrobatics", "handstand", "calisthenics",
                    "coordination", "balance", "reflexes", "speed", "dexterity", "movement", "flow",
                    "jump", "flip", "kick", "punch", "wrestling", "judo", "karate", "taekwondo"],
        "secondary": ["body control", "physical", "mobility", "stretch", "warm up", "cool down",
                      "injury prevention", "technique", "form", "sport", "athletic", "practice", "drill"]
    },
    "SAGE": {
        "primary": ["meditation", "mindfulness", "mental health", "anxiety", "depression", "therapy",
                    "stress management", "sleep", "wellbeing", "emotional", "trauma", "healing",
                    "self-care", "burnout", "psychology", "counseling", "gratitude", "journaling",
                    "breathing", "relaxation", "panic", "mood", "mental illness", "resilience",
                    "happiness", "inner peace", "self-worth", "loneliness", "grief"],
        "secondary": ["health", "wellness", "calm", "balance", "spiritual", "mindset", "positive",
                      "habit", "routine", "lifestyle", "personal growth", "self-improvement",
                      "relationships", "communication", "empathy", "compassion", "boundaries"]
    },
    "MUSE": {
        "primary": ["writing", "creative writing", "fiction", "poetry", "novel", "screenplay", "music",
                    "composition", "painting", "drawing", "illustration", "photography", "filmmaking",
                    "game design", "graphic design", "animation", "storytelling", "worldbuilding",
                    "character design", "art", "artist", "creativity", "inspiration", "craft", "voice",
                    "style", "narrative", "plot", "genre", "blog", "content creation"],
        "secondary": ["create", "creative", "express", "imagination", "idea", "design", "aesthetic",
                      "color", "composition", "rhythm", "melody", "lyric", "poem", "story", "character",
                      "scene", "draft", "edit", "publish", "portfolio", "audience", "feedback"]
    },
    "EMPIRE": {
        "primary": ["startup", "entrepreneur", "business", "revenue", "profit", "marketing", "sales",
                    "customer", "investment", "funding", "venture capital", "pitch", "product market fit",
                    "leadership", "management", "productivity", "finance", "budget", "cash flow",
                    "stock", "dividend", "portfolio", "freelance", "side hustle", "passive income",
                    "negotiation", "networking", "branding", "scaling", "growth hacking", "saas"],
        "secondary": ["money", "income", "career", "work", "goal", "strategy", "plan", "execute",
                      "success", "failure", "mindset", "motivation", "discipline", "focus",
                      "time management", "prioritize", "deadline", "team", "hire", "remote"]
    },
    "GG": {
        "primary": ["gaming", "esports", "speedrun", "fps", "rpg", "mmo", "moba", "battle royale",
                    "minecraft", "fortnite", "league of legends", "valorant", "counter-strike", "csgo",
                    "overwatch", "apex legends", "twitch", "streaming", "youtube gaming", "rank",
                    "ladder", "matchmaking", "meta", "build", "loadout", "strategy", "pro player",
                    "controller", "keyboard", "mouse", "fps drops", "lag", "ping", "dungeon master",
                    "d&d", "tabletop", "game master", "campaign", "quest", "loot", "boss", "raid"],
        "secondary": ["game", "play", "player", "level", "skill", "team", "win", "lose", "match",
                      "tournament", "competitive", "casual", "grind", "unlock", "achievement",
                      "high score", "content", "clip", "highlight", "community", "discord"]
    },
}

def score_sample(text: str, build: str) -> int:
    t = text.lower()
    score = sum(3 for kw in BUILD_KEYWORDS[build]["primary"] if kw in t)
    score += sum(1 for kw in BUILD_KEYWORDS[build]["secondary"] if kw in t)
    return score

def classify_sample(sample: dict) -> str | None:
    text = (sample.get("user", "") + " " + sample.get("assistant", "")).lower()
    scores = {b: score_sample(text, b) for b in SYSTEM_PROMPTS}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 3 else None

def make_sample(build: str, user: str, assistant: str) -> dict:
    return {"system": SYSTEM_PROMPTS[build], "user": user.strip(), "assistant": assistant.strip()}

# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def load_existing_keys(train_path: str, val_path: str) -> set:
    keys = set()
    for path in [train_path, val_path]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        s = json.loads(line)
                        keys.add(s.get("user", "")[:120])
                    except:
                        pass
    print(f"  Loaded {len(keys):,} existing dedup keys")
    return keys

# ---------------------------------------------------------------------------
# Source 1: Reddit staging CSVs
# ---------------------------------------------------------------------------
BUILD_MAP_REDDIT = {
    "strength__fitness.csv":      "TITAN",
    "strength__bodybuilding.csv": "TITAN",
    "strength__weightroom.csv":   "TITAN",
    "strength__xxfitness.csv":    "TITAN",
    "intelligence__explainlikeimfive.csv": "ORACLE",
    "creative__worldbuilding.csv": "MUSE",
    "creative__learnart.csv":     "MUSE",
}

def collect_reddit_csvs(staging_path: str, seen: set) -> dict:
    print("\n[1] Reddit staging CSVs ...")
    by_build = defaultdict(list)
    csv_dir = os.path.join(staging_path, "reddit")
    for fname, build in BUILD_MAP_REDDIT.items():
        fpath = os.path.join(csv_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        added = 0
        for row in rows:
            q = row.get("question", "").strip()
            a = row.get("answer", "").strip()
            if not q or not a or len(q) < 20 or len(a) < 30:
                continue
            key = q[:120]
            if key in seen:
                continue
            seen.add(key)
            by_build[build].append(make_sample(build, q, a))
            added += 1
        print(f"  {fname}: +{added} -> {build}")
    return by_build

# ---------------------------------------------------------------------------
# Source 2: MathDial tutoring (ORACLE)
# ---------------------------------------------------------------------------
def collect_mathdial(staging_path: str, seen: set) -> dict:
    """
    MathDial format: conversation is a single string with turns separated by |EOM|
    Each turn is "Speaker: (tag)text" e.g. "Teacher: (probing)..." or "Steven: ..."
    """
    print("\n[2] MathDial tutoring -> ORACLE ...")
    by_build = defaultdict(list)
    for split in ["train.jsonl", "test.jsonl"]:
        fpath = os.path.join(staging_path, "github", "02_mathdial_tutoring", split)
        if not os.path.exists(fpath):
            continue
        added = 0
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except:
                    continue
                question = item.get("question", "").strip()
                ground_truth = item.get("ground_truth", "").strip()
                convo_str = item.get("conversation", "")

                # Add raw problem -> solution pair
                if question and ground_truth:
                    key = question[:120]
                    if key not in seen:
                        seen.add(key)
                        user_q = f"Can you help me solve this math problem? {question}"
                        asst_a = f"Sure! Let me walk you through this step by step.\n\n{ground_truth}"
                        by_build["ORACLE"].append(make_sample("ORACLE", user_q, asst_a))
                        added += 1

                # Parse conversation string: "Teacher: (tag)text|EOM|Student: text|EOM|..."
                if isinstance(convo_str, str) and "|EOM|" in convo_str:
                    turns = [t.strip() for t in convo_str.split("|EOM|") if t.strip()]
                    # Find consecutive student -> teacher pairs
                    for i in range(len(turns) - 1):
                        t_a = turns[i]
                        t_b = turns[i + 1]
                        # Determine roles
                        a_is_teacher = t_a.lower().startswith("teacher:")
                        b_is_teacher = t_b.lower().startswith("teacher:")
                        if not a_is_teacher and b_is_teacher:
                            # student question -> teacher answer
                            student_text = t_a.split(":", 1)[-1].strip()
                            teacher_text = t_b.split(":", 1)[-1].strip()
                            # Remove parenthetical tags like "(probing)" from teacher
                            import re
                            teacher_text = re.sub(r"^\([^)]+\)", "", teacher_text).strip()
                            if not student_text or not teacher_text:
                                continue
                            # Prepend the math problem as context for the first turn
                            if i == 0 and question:
                                user_q = f"Math problem: {question}\n\nStudent: {student_text}"
                            else:
                                user_q = student_text
                            key = user_q[:120]
                            if key in seen:
                                continue
                            seen.add(key)
                            by_build["ORACLE"].append(make_sample("ORACLE", user_q, teacher_text))
                            added += 1

        print(f"  {split}: +{added} -> ORACLE")
    return by_build

# ---------------------------------------------------------------------------
# Source 3: yahma/alpaca-cleaned (52k instruction pairs)
# ---------------------------------------------------------------------------
def collect_alpaca_cleaned(seen: set, cap_per_build: int = 1500) -> dict:
    print("\n[3] yahma/alpaca-cleaned (52k) ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    except Exception as e:
        print(f"  FAILED: {e}")
        return defaultdict(list)

    by_build = defaultdict(list)
    counts = defaultdict(int)
    total_seen = 0

    for item in ds:
        total_seen += 1
        if total_seen % 5000 == 0:
            print(f"  Scanned {total_seen:,} items | classified: {sum(counts.values()):,}")
        if all(counts[b] >= cap_per_build for b in SYSTEM_PROMPTS):
            break

        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")
        if not instruction or not output or len(output) < 40:
            continue
        user = f"{instruction}\n{inp}".strip() if inp else instruction
        sample = {"user": user, "assistant": output}
        build = classify_sample(sample)
        if build is None:
            continue
        if counts[build] >= cap_per_build:
            continue
        key = user[:120]
        if key in seen:
            continue
        seen.add(key)
        by_build[build].append(make_sample(build, user, output))
        counts[build] += 1

    for b, c in sorted(counts.items()):
        if c > 0:
            print(f"  {b}: +{c}")
    return by_build

# ---------------------------------------------------------------------------
# Source 4: databricks-dolly-15k
# ---------------------------------------------------------------------------
def collect_dolly(seen: set, cap_per_build: int = 800) -> dict:
    print("\n[4] databricks/databricks-dolly-15k ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    except Exception as e:
        print(f"  FAILED: {e}")
        return defaultdict(list)

    by_build = defaultdict(list)
    counts = defaultdict(int)

    for item in ds:
        instruction = item.get("instruction", "")
        context = item.get("context", "")
        response = item.get("response", "")
        if not instruction or not response or len(response) < 40:
            continue
        user = f"{instruction}\n{context}".strip() if context else instruction
        sample = {"user": user, "assistant": response}
        build = classify_sample(sample)
        if build is None:
            continue
        if counts[build] >= cap_per_build:
            continue
        key = user[:120]
        if key in seen:
            continue
        seen.add(key)
        by_build[build].append(make_sample(build, user, response))
        counts[build] += 1

    for b, c in sorted(counts.items()):
        if c > 0:
            print(f"  {b}: +{c}")
    return by_build

# ---------------------------------------------------------------------------
# Source 5: WizardLM evol instruct 70k
# ---------------------------------------------------------------------------
def collect_wizardlm(seen: set, cap_per_build: int = 1200) -> dict:
    print("\n[5] WizardLM/WizardLM_evol_instruct_70k ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train", streaming=True)
    except Exception as e:
        print(f"  FAILED: {e}")
        return defaultdict(list)

    by_build = defaultdict(list)
    counts = defaultdict(int)
    total_seen = 0

    for item in ds:
        total_seen += 1
        if total_seen % 5000 == 0:
            print(f"  Scanned {total_seen:,} | classified: {sum(counts.values()):,}")
        if all(counts[b] >= cap_per_build for b in SYSTEM_PROMPTS):
            break

        convo = item.get("conversations", [])
        if len(convo) < 2:
            # Try flat format
            user_text = item.get("instruction", item.get("input", ""))
            assistant_text = item.get("output", item.get("response", ""))
        else:
            user_text = convo[0].get("value", "") if convo[0].get("from") == "human" else ""
            assistant_text = convo[1].get("value", "") if len(convo) > 1 else ""

        if not user_text or not assistant_text or len(assistant_text) < 40:
            continue
        sample = {"user": user_text, "assistant": assistant_text}
        build = classify_sample(sample)
        if build is None:
            continue
        if counts[build] >= cap_per_build:
            continue
        key = user_text[:120]
        if key in seen:
            continue
        seen.add(key)
        by_build[build].append(make_sample(build, user_text, assistant_text))
        counts[build] += 1

    for b, c in sorted(counts.items()):
        if c > 0:
            print(f"  {b}: +{c}")
    return by_build

# ---------------------------------------------------------------------------
# Source 6: OpenAssistant oasst1 (assistant turns as Q&A)
# ---------------------------------------------------------------------------
def collect_oasst1(seen: set, cap_per_build: int = 800) -> dict:
    print("\n[6] OpenAssistant/oasst1 ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("OpenAssistant/oasst1", split="train")
    except Exception as e:
        print(f"  FAILED: {e}")
        return defaultdict(list)

    # Build a map: message_id -> message
    msg_map = {}
    for item in ds:
        msg_map[item["message_id"]] = item

    by_build = defaultdict(list)
    counts = defaultdict(int)

    for mid, msg in msg_map.items():
        if msg.get("role") != "assistant":
            continue
        if msg.get("lang", "en") != "en":
            continue
        parent_id = msg.get("parent_id")
        if not parent_id or parent_id not in msg_map:
            continue
        parent = msg_map[parent_id]
        if parent.get("role") != "prompter":
            continue
        user_text = parent.get("text", "").strip()
        assistant_text = msg.get("text", "").strip()
        if not user_text or not assistant_text or len(assistant_text) < 40:
            continue
        sample = {"user": user_text, "assistant": assistant_text}
        build = classify_sample(sample)
        if build is None:
            continue
        if counts[build] >= cap_per_build:
            continue
        key = user_text[:120]
        if key in seen:
            continue
        seen.add(key)
        by_build[build].append(make_sample(build, user_text, assistant_text))
        counts[build] += 1

    for b, c in sorted(counts.items()):
        if c > 0:
            print(f"  {b}: +{c}")
    return by_build

# ---------------------------------------------------------------------------
# Source 7: HuggingFaceH4/ultrachat_200k (sample subset)
# ---------------------------------------------------------------------------
def collect_ultrachat(seen: set, cap_per_build: int = 800) -> dict:
    print("\n[7] HuggingFaceH4/ultrachat_200k (streaming sample) ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    except Exception as e:
        print(f"  FAILED: {e}")
        return defaultdict(list)

    by_build = defaultdict(list)
    counts = defaultdict(int)
    total_seen = 0

    for item in ds:
        total_seen += 1
        if total_seen % 5000 == 0:
            print(f"  Scanned {total_seen:,} | classified: {sum(counts.values()):,}")
        if all(counts[b] >= cap_per_build for b in SYSTEM_PROMPTS):
            break
        if total_seen > 60000:  # safety cap
            break

        messages = item.get("messages", [])
        if len(messages) < 2:
            continue
        # Take first human/assistant pair
        user_text = ""
        assistant_text = ""
        for m in messages:
            if m.get("role") == "user" and not user_text:
                user_text = m.get("content", "")
            elif m.get("role") == "assistant" and not assistant_text:
                assistant_text = m.get("content", "")
        if not user_text or not assistant_text or len(assistant_text) < 40:
            continue
        sample = {"user": user_text, "assistant": assistant_text}
        build = classify_sample(sample)
        if build is None:
            continue
        if counts[build] >= cap_per_build:
            continue
        key = user_text[:120]
        if key in seen:
            continue
        seen.add(key)
        by_build[build].append(make_sample(build, user_text, assistant_text))
        counts[build] += 1

    for b, c in sorted(counts.items()):
        if c > 0:
            print(f"  {b}: +{c}")
    return by_build

# ---------------------------------------------------------------------------
# Source 8: medical_meadow_healthcaremagic filtered for PHANTOM/TITAN/SAGE
# ---------------------------------------------------------------------------
def collect_healthcaremagic_extra(seen: set) -> dict:
    """Pull more movement/sport/wellness Q&A from HealthCareMagic."""
    print("\n[8] lavita/ChatDoctor-HealthCareMagic-100k (extra keywords) ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train", streaming=True)
    except Exception as e:
        print(f"  FAILED: {e}")
        return defaultdict(list)

    MOVEMENT_KW = ["flexibility", "mobility", "stretch", "yoga", "pilates", "balance", "coordination",
                   "agility", "speed", "parkour", "gymnastics", "climbing", "martial art", "jiu-jitsu",
                   "taekwondo", "karate", "wrestling", "boxing", "dance", "acrobatics", "calisthenics"]
    STRENGTH_KW = ["weightlifting", "squat", "deadlift", "bench press", "barbell", "dumbbell",
                   "muscle", "strength training", "powerlifting", "bodybuilding", "gym", "lifting"]
    WELLNESS_KW = ["anxiety", "depression", "stress", "mental health", "sleep", "therapy", "counseling",
                   "meditation", "mindfulness", "panic attack", "mood", "emotional", "trauma", "burnout"]

    by_build = defaultdict(list)
    counts = defaultdict(int)
    CAPS = {"PHANTOM": 1000, "TITAN": 600, "SAGE": 600}
    total = 0

    for item in ds:
        total += 1
        if total > 60000:
            break
        if all(counts.get(b, 0) >= CAPS.get(b, 0) for b in CAPS):
            break
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        if not inp or not out or len(out) < 50:
            continue
        text_l = (inp + " " + out).lower()
        build = None
        if counts.get("PHANTOM", 0) < CAPS["PHANTOM"] and any(k in text_l for k in MOVEMENT_KW):
            build = "PHANTOM"
        elif counts.get("TITAN", 0) < CAPS["TITAN"] and any(k in text_l for k in STRENGTH_KW):
            build = "TITAN"
        elif counts.get("SAGE", 0) < CAPS["SAGE"] and any(k in text_l for k in WELLNESS_KW):
            build = "SAGE"
        if build is None:
            continue
        key = inp[:120]
        if key in seen:
            continue
        seen.add(key)
        by_build[build].append(make_sample(build, inp, out))
        counts[build] = counts.get(build, 0) + 1

    for b, c in sorted(counts.items()):
        if c > 0:
            print(f"  {b}: +{c}")
    return by_build

# ---------------------------------------------------------------------------
# Source 9: teknium/OpenHermes-2.5 (large general instruction dataset)
# ---------------------------------------------------------------------------
def collect_openhermes(seen: set, cap_per_build: int = 800) -> dict:
    print("\n[9] teknium/OpenHermes-2.5 (streaming) ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    except Exception as e:
        print(f"  FAILED: {e}")
        return defaultdict(list)

    by_build = defaultdict(list)
    counts = defaultdict(int)
    total_seen = 0

    for item in ds:
        total_seen += 1
        if total_seen % 10000 == 0:
            print(f"  Scanned {total_seen:,} | classified: {sum(counts.values()):,}")
        if all(counts[b] >= cap_per_build for b in SYSTEM_PROMPTS):
            break
        if total_seen > 100000:
            break

        convo = item.get("conversations", [])
        user_text = ""
        assistant_text = ""
        for m in convo:
            role = m.get("from", m.get("role", ""))
            val = m.get("value", m.get("content", ""))
            if role in ("human", "user") and not user_text:
                user_text = val
            elif role in ("gpt", "assistant") and not assistant_text:
                assistant_text = val
        if not user_text or not assistant_text or len(assistant_text) < 40:
            continue
        sample = {"user": user_text, "assistant": assistant_text}
        build = classify_sample(sample)
        if build is None:
            continue
        if counts[build] >= cap_per_build:
            continue
        key = user_text[:120]
        if key in seen:
            continue
        seen.add(key)
        by_build[build].append(make_sample(build, user_text, assistant_text))
        counts[build] += 1

    for b, c in sorted(counts.items()):
        if c > 0:
            print(f"  {b}: +{c}")
    return by_build

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    STAGING = "data/raw/staging"
    TRAIN_PATH = "data/raw/finetune_train_v2.jsonl"
    VAL_PATH = "data/raw/finetune_val_v2.jsonl"

    print("=" * 60)
    print("LevelUp AI -- Dataset Expansion v3")
    print("=" * 60)

    # Load existing sample counts
    existing_counts = defaultdict(int)
    if os.path.exists(TRAIN_PATH):
        PROMPT_TO_BUILD = {}
        for b, p in SYSTEM_PROMPTS.items():
            PROMPT_TO_BUILD[p] = b
        with open(TRAIN_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    s = json.loads(line)
                    sys_p = s.get("system", "")
                    # Match by build name in system prompt
                    for b in SYSTEM_PROMPTS:
                        if b in sys_p or b.lower() in sys_p.lower():
                            existing_counts[b] += 1
                            break
                except:
                    pass
    print("\nExisting train counts (approx):")
    for b in SYSTEM_PROMPTS:
        print(f"  {b}: ~{existing_counts[b]:,}")

    # Load dedup keys
    print("\nLoading dedup keys ...")
    seen = load_existing_keys(TRAIN_PATH, VAL_PATH)

    # Collect from all sources
    all_new: dict[str, list] = defaultdict(list)

    def merge(source_dict):
        for b, samples in source_dict.items():
            all_new[b].extend(samples)

    merge(collect_reddit_csvs(STAGING, seen))
    merge(collect_mathdial(STAGING, seen))
    merge(collect_alpaca_cleaned(seen, cap_per_build=1500))
    merge(collect_dolly(seen, cap_per_build=800))
    merge(collect_wizardlm(seen, cap_per_build=1200))
    merge(collect_oasst1(seen, cap_per_build=800))
    merge(collect_ultrachat(seen, cap_per_build=800))
    merge(collect_healthcaremagic_extra(seen))
    merge(collect_openhermes(seen, cap_per_build=800))

    # Summary of new samples
    print("\n" + "=" * 60)
    print("New samples collected:")
    total_new = 0
    for b in SYSTEM_PROMPTS:
        n = len(all_new[b])
        total_new += n
        print(f"  {b}: +{n:,}")
    print(f"  TOTAL NEW: {total_new:,}")

    if total_new == 0:
        print("Nothing to add. Exiting.")
        return

    # Write to files (90/10 split)
    print("\nAppending to dataset files ...")
    train_added = defaultdict(int)
    val_added = defaultdict(int)

    with open(TRAIN_PATH, "a", encoding="utf-8") as ft, \
         open(VAL_PATH, "a", encoding="utf-8") as fv:
        for build in SYSTEM_PROMPTS:
            samples = all_new[build]
            random.shuffle(samples)
            split = int(len(samples) * 0.9)
            for s in samples[:split]:
                ft.write(json.dumps(s, ensure_ascii=False) + "\n")
                train_added[build] += 1
            for s in samples[split:]:
                fv.write(json.dumps(s, ensure_ascii=False) + "\n")
                val_added[build] += 1

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"{'Build':<12} {'Prev Train':>12} {'New Train':>10} {'New Val':>8} {'Est. Total':>12}")
    print("-" * 60)
    grand_total = 0
    for b in SYSTEM_PROMPTS:
        prev = existing_counts[b]
        nt = train_added[b]
        nv = val_added[b]
        total = prev + nt
        grand_total += total
        print(f"  {b:<10} {prev:>12,} {nt:>10,} {nv:>8,} {total:>12,}")
    print("-" * 60)
    print(f"  {'TOTAL':<10} {sum(existing_counts.values()):>12,} "
          f"{sum(train_added.values()):>10,} "
          f"{sum(val_added.values()):>8,} "
          f"{grand_total:>12,}")
    print("=" * 60)
    print(f"\nTrain file: {TRAIN_PATH}")
    print(f"Val file  : {VAL_PATH}")
    print("\nDone! Ready for fine-tuning.")

if __name__ == "__main__":
    main()
