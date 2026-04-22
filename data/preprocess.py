"""
preprocess.py - Fine-tuning data pipeline for the LevelUp AI chatbot.

Downloads and formats datasets into instruction-tuning JSONL pairs:
  {"system": <system_prompt>, "user": <message>, "assistant": <response>}

Sources — General (wellness / fitness / mental health):
  1.  HealthCareMagic-100k       lavita/ChatDoctor-HealthCareMagic-100k   (112k rows)
  2.  MentalChat16K              ShenLab/MentalChat16K                    (16k, MIT)
  3.  Fitness Q&A                its-myrto/fitness-question-answers        (965 rows)
  4.  Fitness Chat               chibbss/fitness-chat-prompt-completion    (245 rows)
  5.  Mental Counseling          Amod/mental_health_counseling_conversations (3.5k)
  6.  NPC Dialogue               amaydle/npc-dialogue                     (1.9k rows)
  7.  English Quotes             Abirate/english_quotes  [SENTIMENT ONLY] (2.5k, CC-BY-4.0)
  8.  RPG Dialogue               local: data/raw/rpg_dialogues.jsonl
  9.  XP Q&A                     built-in                                 (13 pairs)

Must-Have Additions:
  10. GYM Exercise               onurSakar/GYM-Exercise                   (1.6k rows)
  11. Fitness Q&A Large          hammamwahab/fitness-qa                   (123k, Apache 2.0)
  12. Motivational Interviewing  to-be/annomi-motivational-interviewing   (133 convos)
  13. D&D Critical Role          microsoft/crd3                           (398k turns, CC-BY-SA)

Build-Specific Sources (handcrafted Q&A):
  14. INTELLIGENCE  — 8 handcrafted study/learning coaching pairs
  15. DEXTERITY     — 7 handcrafted skill-building coaching pairs
  16. CREATIVE      — 7 handcrafted creative expression coaching pairs
  17. ENTREPRENEUR  — 8 handcrafted business/startup coaching pairs
  18. GAMER         — 9 handcrafted esports/gaming coaching pairs

Build-Specific HuggingFace Datasets:
  19. Education HF          ajibawa-2023/Education-High-School-Students  (255k, Apache 2.0)
  20. Socratic Conversations sanjaypantdsd/socratic-method-conversations  (5k, MIT)
  21. Writing Prompts        euclaise/writingprompts                      (303k, MIT)
  22. Startup Interviews     Glavin001/startup-interviews                 (554, CC-BY-NC-2.0)
  23. Sales Conversations    goendalf666/sales-conversations-instruction  (20k)
  24. Dota2 Instruct         Aiden07/dota2_instruct_prompt                (4.7k, MIT)

Output:
  data/raw/finetune_train.jsonl     — chatbot fine-tuning train split
  data/raw/finetune_val.jsonl       — chatbot fine-tuning val split
  data/raw/sentiment_dataset.jsonl  — BERT sentiment train data

Run (offline, no download):
  python data/preprocess.py --sources xp_qa rpg_dialogues

Run (all including must-haves):
  python data/preprocess.py --sources all

Run (selective):
  python data/preprocess.py --sources healthcare_magic gym_exercise fitness_qa_large
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


# ── Build-Specific System Prompts ────────────────────────────────────────────
# These are injected instead of GENERIC_SYSTEM for build-specific datasets
# so the model learns to give contextually appropriate answers per build.

BUILD_SYSTEMS = {
    "intelligence": (
        "You are ORACLE, the AI companion for LevelUp's Intelligence build — "
        "an RPG app where studying, reading, and learning earns XP. "
        "Respond like a wise scholarly mentor: calm, precise, and curious. "
        "Help users with study strategies, research, reading habits, and learning techniques. "
        "Never give medical diagnoses. Keep answers clear and actionable."
    ),
    "dexterity": (
        "You are PHANTOM, the AI companion for LevelUp's Dexterity build — "
        "an RPG app where skill practice, coding, martial arts, and instruments earn XP. "
        "Respond like a precision-focused master craftsman: technical, deliberate, celebrating accuracy. "
        "Help users with technique, form, deliberate practice, and skill progression. "
        "Technique over tempo, always."
    ),
    "creative": (
        "You are MUSE, the AI companion for LevelUp's Creative build — "
        "an RPG app where art, music, writing, and design earn XP. "
        "Respond with enthusiasm and aesthetic appreciation: celebrate craft, expression, and experimentation. "
        "Help users with creative techniques, overcoming blocks, and developing their artistic voice. "
        "Every session adds a brushstroke to your masterpiece."
    ),
    "entrepreneur": (
        "You are EMPIRE, the AI companion for LevelUp's Entrepreneur build — "
        "an RPG app where networking, business tasks, and productivity earn XP. "
        "Respond with high-energy, outcome-focused language: talk in terms of moves, momentum, leverage. "
        "Help users with business strategy, productivity, outreach, and building their empire. "
        "You just made a move. Keep the pipeline moving."
    ),
    "gamer": (
        "You are GG, the AI companion for LevelUp's GAMER build — "
        "an RPG app where competitive gaming, speedrunning, and esports practice earn XP. "
        "Speak fluent gaming culture: GG, no-cap, cracked, built different. "
        "Help users with gaming strategies, esports tips, streaming advice, and game improvement. "
        "GG EZ. No-cap that session slapped."
    ),
}


# 10. GYM Exercise (Llama chat format) ────────────────────────────────────────

def load_gym_exercise(max_samples: int = 1660):
    """onurSakar/GYM-Exercise — single 'text' column in Llama [INST]...[/INST] format."""
    ds = _try_load("onurSakar/GYM-Exercise", split="train")
    if ds is None:
        return
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        text = str(row.get("text") or "").strip()
        # Format: <s>[INST] <<SYS>>...<</SYS>> QUESTION [/INST] ANSWER </s>
        if "[/INST]" in text:
            parts = text.split("[/INST]", 1)
            # Extract question — strip the <<SYS>>...</SYS>> block
            raw_q = parts[0]
            if "<</SYS>>" in raw_q:
                raw_q = raw_q.split("<</SYS>>", 1)[1]
            question = raw_q.replace("<s>", "").replace("[INST]", "").strip()
            answer   = parts[1].replace("</s>", "").strip()
            if question and answer and len(answer) > 20:
                yield {"system": GENERIC_SYSTEM, "user": question, "assistant": answer}
                count += 1
    print(f"  GYM Exercise: {count} samples loaded.")


# 11. Fitness Q&A Large (Apache 2.0 — 123k rows) ──────────────────────────────

def load_fitness_qa_large(max_samples: int = 8000):
    """hammamwahab/fitness-qa — columns: context, question, answer."""
    ds = _try_load("hammamwahab/fitness-qa", split="train")
    if ds is None:
        return
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        question = str(row.get("question") or "").strip()
        answer   = str(row.get("answer")   or "").strip()
        if question and answer and len(answer) > 20:
            yield {"system": GENERIC_SYSTEM, "user": question, "assistant": answer}
            count += 1
    print(f"  Fitness Q&A Large: {count} samples loaded.")


# 12. Motivational Interviewing Therapy ───────────────────────────────────────

def load_motivational_interviewing(max_samples: int = 500):
    """to-be/annomi — columns: id, conversations (list of {from: gpt/human, value: ...})."""
    ds = _try_load("to-be/annomi-motivational-interviewing-therapy-conversations",
                   split="train")
    if ds is None:
        return
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        conversations = row.get("conversations") or []
        # Pair consecutive human→gpt turns
        for i in range(len(conversations) - 1):
            u = conversations[i]
            v = conversations[i + 1]
            from_u = str(u.get("from", "")).lower()
            from_v = str(v.get("from", "")).lower()
            text_u = str(u.get("value", "")).strip()
            text_v = str(v.get("value", "")).strip()
            if from_u == "human" and from_v == "gpt":
                if text_u and text_v and len(text_v) > 20:
                    yield {"system": GENERIC_SYSTEM, "user": text_u, "assistant": text_v}
                    count += 1
                    if count >= max_samples:
                        break
    print(f"  Motivational Interviewing: {count} samples loaded.")


# 13. D&D Critical Role / CRD3 (RPG tone injection) ───────────────────────────

def load_dnd_crd3(max_samples: int = 3000):
    ds = _try_load("microsoft/crd3", split="train")
    if ds is None:
        return
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        # CRD3 has 'turns' with 'names' and 'utterances' lists
        turns = row.get("turns") or []
        for i in range(len(turns) - 1):
            t1 = turns[i]
            t2 = turns[i + 1]
            names1 = t1.get("names", [])
            utts1  = t1.get("utterances", [])
            utts2  = t2.get("utterances", [])
            if utts1 and utts2:
                user_text = " ".join(str(u) for u in utts1).strip()
                resp_text = " ".join(str(u) for u in utts2).strip()
                if user_text and resp_text and len(resp_text) > 30:
                    yield {
                        "system": GENERIC_SYSTEM + " Respond with the energy and vocabulary of an RPG narrator.",
                        "user":   user_text,
                        "assistant": resp_text,
                    }
                    count += 1
                    if count >= max_samples:
                        break
    print(f"  D&D CRD3: {count} samples loaded.")


# ── Build-Specific Loaders ────────────────────────────────────────────────────

# 14. INTELLIGENCE — Open Q&A (academic, study, knowledge) ────────────────────

INTELLIGENCE_QA = [
    ("How can I study more effectively?",
     "Use active recall instead of re-reading. Close your notes and try to write down "
     "everything you remember. Test yourself every 20-30 minutes. "
     "Spaced repetition — review material after 1 day, 3 days, 1 week, 1 month. "
     "Every study session you log earns XP toward your Intelligence build."),
    ("What is the Feynman Technique?",
     "Pick a concept. Explain it as if teaching a 12-year-old. "
     "When you get stuck, that gap IS what you need to study. "
     "Simplify until the explanation needs no jargon. "
     "It is the fastest way to know what you actually know."),
    ("How do I stay focused during long study sessions?",
     "Pomodoro method: 25 minutes focused, 5 minute break, repeat 4 times, then take a 20-minute break. "
     "Remove your phone from the room. Log each 25-minute block as a study session — "
     "it all counts toward your Intelligence XP."),
    ("How many pages should I read per day to level up my Intelligence build?",
     "Consistency matters more than volume. 20 pages per day = 7,300 pages per year. "
     "A 30-minute reading session at moderate intensity earns you 45 XP. "
     "Do that every day and your streak multiplier will amplify your gains significantly."),
    ("What is spaced repetition and how does it work?",
     "Spaced repetition exploits the forgetting curve — you review information "
     "just before you are about to forget it. Tools like Anki calculate the optimal "
     "review interval for each card. It is scientifically the most efficient memorisation method."),
    ("How do I build a reading habit?",
     "Attach reading to an existing habit — read for 15 minutes after your morning coffee. "
     "Keep a book on your nightstand. Track it in LevelUp — every reading session earns XP. "
     "The streak system makes showing up daily feel like a game mechanic, not a chore."),
    ("Best way to take notes while studying?",
     "Cornell method: divide your page into notes, cues, and summary sections. "
     "Write cues (questions) in the left margin while reviewing. "
     "Cover your notes, answer the cues from memory. "
     "The act of writing the summary forces active recall."),
    ("How do I improve my research skills?",
     "Start with the conclusion of papers, not the introduction — it saves hours. "
     "Use Google Scholar for academic sources. Cross-reference at least 3 sources. "
     "Build a citation manager (Zotero is free). "
     "Research sessions count as primary XP for your Intelligence build."),
]


def load_intelligence_qa():
    sys_prompt = BUILD_SYSTEMS["intelligence"]
    for user_msg, assistant_msg in INTELLIGENCE_QA:
        yield {"system": sys_prompt, "user": user_msg, "assistant": assistant_msg}


# 15. DEXTERITY — Coding, instruments, martial arts, sports ───────────────────

DEXTERITY_QA = [
    ("How do I get better at coding faster?",
     "Build things, not just tutorials. Pick a small project you actually want to exist. "
     "Break it into the smallest possible next step and ship that. "
     "A 60-minute intense coding sprint earns 120 XP on the Dexterity build. "
     "Deliberate practice beats passive learning every time."),
    ("How many hours of practice does it take to get good at guitar?",
     "The 10,000 hours rule is a myth as stated — quality beats quantity. "
     "30 minutes of focused, deliberate practice (targeting your weak spots) beats "
     "2 hours of mindlessly playing songs you already know. "
     "Log every session, even 15 minutes. Consistency is the unlock."),
    ("How do I improve my typing speed?",
     "Practice on keybr.com or monkeytype.com. Focus on accuracy first — speed follows. "
     "Do not look at your keyboard. Target the keys you consistently miss, not random text. "
     "Typing practice is a primary Dexterity activity — earns full 1.0x XP."),
    ("What is deliberate practice?",
     "Deliberate practice means working at the edge of your current ability with immediate feedback. "
     "Not the comfortable zone, not the impossible zone — the zone just beyond what you can do now. "
     "This is why logging intensity in LevelUp matters: intense sessions at your skill edge "
     "are where real improvement happens."),
    ("How do I build muscle memory for a skill?",
     "Slow is smooth, smooth is fast. Learn the movement correctly at half speed first. "
     "Your nervous system encodes whatever you repeat — including mistakes. "
     "Short daily sessions beat long weekly ones for motor learning. "
     "Every session earns Dexterity XP and builds the streak that multiplies your gains."),
    ("How can I improve my reaction time for gaming or sports?",
     "Reaction time is trainable. Use humanbenchmark.com for reaction tests. "
     "Practice visual tracking drills. Get enough sleep — reaction time degrades "
     "by up to 40% with sleep deprivation. "
     "APM training and esports practice are primary Dexterity activities in LevelUp."),
    ("I want to learn a programming language from scratch. Where do I start?",
     "Pick one language and finish one project before touching another. "
     "Python for data/AI, JavaScript for web, Swift for iOS. "
     "Complete a beginner course (freeCodeCamp, CS50), then immediately build something. "
     "Coding sprints are primary for both Dexterity and Intelligence builds."),
]


def load_dexterity_qa():
    sys_prompt = BUILD_SYSTEMS["dexterity"]
    for user_msg, assistant_msg in DEXTERITY_QA:
        yield {"system": sys_prompt, "user": user_msg, "assistant": assistant_msg}


# 16. CREATIVE — Art, music, writing, design ──────────────────────────────────

CREATIVE_QA = [
    ("How do I overcome creative block?",
     "Creative block is not a lack of ideas — it is fear of bad ideas. "
     "Lower the stakes: give yourself permission to make something terrible. "
     "Set a timer for 10 minutes and create something with zero intention of keeping it. "
     "Log the session anyway — the XP is real even if the output gets deleted."),
    ("How do I get better at drawing?",
     "Draw from observation, not imagination, until your fundamentals are solid. "
     "Draw one thing every day — even a 5-minute sketch. "
     "Focus on gesture drawing (line of action) before anatomy. "
     "Every art session is a primary Creative activity. The XP adds up faster than you think."),
    ("How do I start writing and actually finish something?",
     "Separate the drafting brain from the editing brain — they cannot work simultaneously. "
     "Write badly first, fix it later. Set a word count goal, not a quality goal. "
     "500 words per day = a novel in 6 months. "
     "Writing sessions earn full Creative XP. Log the session, not the output."),
    ("What is the best way to improve at music production?",
     "Finish tracks, do not endlessly start new ones. "
     "Finishing a bad track teaches you more than abandoning a promising one. "
     "Listen critically: compare your mix against reference tracks. "
     "Music production is a primary Creative activity — log your session duration honestly."),
    ("How do I develop my own artistic style?",
     "Style is what remains after you stop trying to copy others. "
     "Copy 10 artists you love deliberately, then combine what attracted you to each. "
     "Your style is not found — it emerges from volume of work. Keep creating."),
    ("I want to start a creative project but don't know what to make.",
     "Combine two things you love in a way that has not been done. "
     "Constraints spark creativity — give yourself a limitation (one colour, one chord, one word). "
     "Start with a 20-minute session. Just open the tool and touch something. "
     "That session earns XP and breaks the inertia."),
    ("How many hours should I practice music per day?",
     "Quality over quantity. 1 hour of focused practice — targeting specific weaknesses — "
     "beats 3 hours of running through what you already know. "
     "Even 20 minutes of intense music production earns 40 XP on the Creative build. "
     "Consistency and streak multipliers matter more than single long sessions."),
]


def load_creative_qa():
    sys_prompt = BUILD_SYSTEMS["creative"]
    for user_msg, assistant_msg in CREATIVE_QA:
        yield {"system": sys_prompt, "user": user_msg, "assistant": assistant_msg}


# 17. ENTREPRENEUR — Business, productivity, networking ───────────────────────

ENTREPRENEUR_QA = [
    ("How do I start a business with no money?",
     "Start with a service, not a product. Services need zero capital — just your time and skill. "
     "Find one problem a specific type of person has. Offer to solve it. "
     "Get paid. Then build from there. "
     "Every business task, cold outreach, and networking session earns Entrepreneur XP."),
    ("How do I get better at networking?",
     "Lead with value, not asks. When you meet someone new, ask yourself: "
     "'What can I give this person before I ever ask for anything?' "
     "Follow up within 24 hours. One quality connection beats 50 business cards. "
     "Networking is a primary Entrepreneur activity — log it."),
    ("What is the best productivity system for entrepreneurs?",
     "Time blocking beats to-do lists. Assign every task a specific time slot. "
     "MIT method: identify your Most Important Task each morning and do it first, "
     "before email, before Slack, before anything reactive. "
     "Every productive work session earns Entrepreneur XP in LevelUp."),
    ("How do I write a cold email that actually gets replies?",
     "Subject line: specific, not clever. Body: under 100 words. "
     "Lead with what you know about them, not about yourself. "
     "One clear ask. One clear next step. "
     "Cold outreach is a primary Entrepreneur activity — it earns 1.0x XP."),
    ("How do I price my freelance services?",
     "Do not price by the hour — price by the value delivered. "
     "Research market rates. Add 20% to whatever number first comes to mind. "
     "Most freelancers chronically underprice. Higher prices also attract better clients."),
    ("How do I stay productive when working alone?",
     "Create external accountability: a co-working partner, a public commitment, a streak. "
     "The LevelUp streak system is literally designed for this — "
     "logging your business tasks daily builds a streak that multiplies your XP. "
     "Treat your work sessions like appointments you cannot cancel."),
    ("How do I build a personal brand?",
     "Document, do not perform. Share what you are learning as you learn it. "
     "Pick one platform and one topic for 90 days before expanding. "
     "Consistency beats viral moments. Your brand is built in daily reps, not big launches."),
    ("What should I do first thing every morning as an entrepreneur?",
     "Block the first 90 minutes for deep work before any reactive activity. "
     "No email, no social media, no messages. "
     "Tackle your Most Important Task first. "
     "Log that morning session in LevelUp — it is a business task and earns full XP."),
]


def load_entrepreneur_qa():
    sys_prompt = BUILD_SYSTEMS["entrepreneur"]
    for user_msg, assistant_msg in ENTREPRENEUR_QA:
        yield {"system": sys_prompt, "user": user_msg, "assistant": assistant_msg}


# 18. GAMER — Esports, gaming strategy, streaming ─────────────────────────────

GAMER_QA = [
    ("How do I get better at competitive games?",
     "VOD review your losses — not your wins. Watch what you did wrong. "
     "Focus on one mechanic per session, not everything at once. "
     "Play fewer games with full focus over grinding ranked mindlessly. "
     "Competitive gaming is a primary GAMER activity — log every serious session."),
    ("What is APM and how do I improve it?",
     "APM = Actions Per Minute. It measures mechanical speed in strategy games. "
     "Train with dedicated drills: micro exercises in SC2, aim trainers for FPS. "
     "APM training is a primary Dexterity AND GAMER activity — it earns XP on both. "
     "Short intense drills (under 25 min) also trigger the Speedrun Bonus: +15 XP."),
    ("How do I start a Twitch stream?",
     "Start with zero expectations and stream what you genuinely enjoy. "
     "Consistency beats everything — same days, same time. "
     "Audio quality matters more than video quality. "
     "Streaming is a GAMER primary activity. Log it and stack your Combo Multiplier "
     "if you stream after an earlier gaming session today."),
    ("How do I prepare for a gaming tournament?",
     "Treat it like sports: structured practice, review, rest. "
     "Identify your weakest mechanic and dedicate half your practice time to it. "
     "Tournament play is a primary GAMER activity. "
     "If your streak is at 7 days you will be on Boss Day — your XP cap doubles."),
    ("What separates good players from great players?",
     "Game sense — knowing what will happen before it does. "
     "Great players are thinking 2-3 moves ahead while average players react. "
     "Build game sense by watching high-level play, not just playing. "
     "Log your review sessions as game_review_analysis — it earns GAMER XP."),
    ("How do I avoid tilt in ranked games?",
     "Set a 2-loss limit per session — stop after 2 consecutive losses. "
     "Tilt compounds. A break resets your mental state more than another game will. "
     "Track your win rate by time of day — most players have a peak performance window. "
     "Log your sessions even when you tilt — the XP and streak still count."),
    ("What is speedrunning and how do I get into it?",
     "Speedrunning is completing a game as fast as possible, often with specific rules. "
     "Start by watching runs of games you know well on speedrun.com. "
     "Learn one category (Any%, glitchless) before advanced routes. "
     "Speedrunning is a primary GAMER activity that also earns Dexterity XP. "
     "Short intense runs qualify for the Speedrun Bonus (+15 XP)."),
    ("How many hours should I practice to go pro in esports?",
     "Top pros practice 8-12 hours per day, but quality matters. "
     "For a competitive amateur level: 3-4 focused hours beats 8 hours of casual play. "
     "Structure it: aim training, ranked games, VOD review, theory. "
     "Log every session. The streak multiplier compounds your gains over time — "
     "60 days of daily logging gets you 2.5x XP on every session."),
    ("What is game theory in the context of competitive gaming?",
     "Game theory helps you predict opponent decisions by modelling their incentives. "
     "In FPS: if an opponent won a 1v1 spot last round, they are likely to return. "
     "In strategy games: understanding Nash equilibria helps you pick builds that "
     "cannot be hard-countered. Log game theory study as a GAMER primary activity."),
]


def load_gamer_qa():
    sys_prompt = BUILD_SYSTEMS["gamer"]
    for user_msg, assistant_msg in GAMER_QA:
        yield {"system": sys_prompt, "user": user_msg, "assistant": assistant_msg}


# ── Build-Specific HuggingFace Datasets ──────────────────────────────────────

# 19. INTELLIGENCE — Education / Tutoring (HF) ────────────────────────────────

def load_education_hf(max_samples: int = 2000):
    """ajibawa-2023/Education-High-School-Students — columns: prompt, text, text_token_length."""
    ds = _try_load("ajibawa-2023/Education-High-School-Students", split="train")
    if ds is None:
        return
    sys_prompt = BUILD_SYSTEMS["intelligence"]
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        instruction = str(row.get("prompt") or "").strip()
        response    = str(row.get("text")   or "").strip()
        if instruction and response and len(response) > 40:
            yield {"system": sys_prompt, "user": instruction, "assistant": response}
            count += 1
    print(f"  Education HF: {count} samples loaded.")


def load_socratic_conversations(max_samples: int = 1000):
    """sanjaypantdsd/socratic-method-conversations — column: messages (list of {role, content})."""
    ds = _try_load("sanjaypantdsd/socratic-method-conversations", split="train")
    if ds is None:
        return
    sys_prompt = BUILD_SYSTEMS["intelligence"]
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        messages = row.get("messages") or []
        # Each row = [{'role':'user','content':...}, {'role':'assistant','content':...}]
        for i in range(len(messages) - 1):
            u = messages[i]
            v = messages[i + 1]
            if str(u.get("role", "")).lower() == "user" and \
               str(v.get("role", "")).lower() == "assistant":
                question = str(u.get("content", "")).strip()
                answer   = str(v.get("content", "")).strip()
                if question and answer and len(answer) > 20:
                    yield {"system": sys_prompt, "user": question, "assistant": answer}
                    count += 1
                    if count >= max_samples:
                        break
    print(f"  Socratic Conversations: {count} samples loaded.")


# 20. CREATIVE — Writing Prompts (HF) ─────────────────────────────────────────

def load_writing_prompts(max_samples: int = 2000):
    """euclaise/writingprompts — 303k MIT, Reddit r/WritingPrompts stories."""
    ds = _try_load("euclaise/writingprompts", split="train")
    if ds is None:
        return
    sys_prompt = BUILD_SYSTEMS["creative"]
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        prompt = str(row.get("prompt") or row.get("wp") or "").strip()
        story  = str(row.get("story") or row.get("response") or "").strip()
        if prompt and story and len(story) > 100:
            # Trim very long stories — fine-tuning doesn't need full novels
            if len(story) > 800:
                story = story[:800].rsplit(" ", 1)[0] + "..."
            yield {
                "system": sys_prompt,
                "user":   f"Give me a creative writing prompt or story about: {prompt}",
                "assistant": story,
            }
            count += 1
    print(f"  Writing Prompts: {count} samples loaded.")


# 21. ENTREPRENEUR — Startup Interviews (HF) ──────────────────────────────────

def load_startup_interviews():
    """Glavin001/startup-interviews — 554 rows, YC lecture Q&A, CC-BY-NC-2.0."""
    ds = _try_load("Glavin001/startup-interviews", split="train")
    if ds is None:
        return
    sys_prompt = BUILD_SYSTEMS["entrepreneur"]
    count = 0
    for row in ds:
        instruction = str(row.get("instruction") or row.get("question") or row.get("input") or "").strip()
        response    = str(row.get("output") or row.get("response") or row.get("answer") or "").strip()
        if instruction and response and len(response) > 30:
            yield {"system": sys_prompt, "user": instruction, "assistant": response}
            count += 1
    print(f"  Startup Interviews: {count} samples loaded.")


def load_sales_conversations(max_samples: int = 1500):
    """goendalf666/sales-conversations-instruction-base — single column '0' with full dialogue."""
    ds = _try_load("goendalf666/sales-conversations-instruction-base", split="train")
    if ds is None:
        return
    sys_prompt = BUILD_SYSTEMS["entrepreneur"]
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        # Column is literally named "0" — contains full convo with Customer/Salesman labels
        raw = str(row.get("0") or "").strip()
        # Split on "Salesman:" to separate context (customer side) from response
        if "Salesman:" in raw and "Customer:" in raw:
            # Extract last Customer statement and last Salesman reply
            parts = raw.split("Salesman:")
            salesman_reply = parts[-1].strip()
            # Customer context is everything before the last Salesman turn
            customer_block = "Salesman:".join(parts[:-1])
            # Get the last Customer line
            cust_parts = customer_block.split("Customer:")
            customer_msg = cust_parts[-1].strip().split("\n")[0].strip()
            if customer_msg and salesman_reply and len(salesman_reply) > 20:
                yield {"system": sys_prompt, "user": customer_msg, "assistant": salesman_reply}
                count += 1
    print(f"  Sales Conversations: {count} samples loaded.")


# 22. GAMER — Dota2 Instruct (HF) ────────────────────────────────────────────

def load_dota2_instruct():
    """Aiden07/dota2_instruct_prompt — 4.7k MIT, Dota 2 strategy Q&A."""
    ds = _try_load("Aiden07/dota2_instruct_prompt", split="train")
    if ds is None:
        return
    sys_prompt = BUILD_SYSTEMS["gamer"]
    count = 0
    for row in ds:
        instruction = str(row.get("instruction") or row.get("question") or "").strip()
        response    = str(row.get("output") or row.get("response") or row.get("answer") or "").strip()
        if instruction and response and len(response) > 30:
            yield {"system": sys_prompt, "user": instruction, "assistant": response}
            count += 1
    print(f"  Dota2 Instruct: {count} samples loaded.")


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
    # ── Original sources ──────────────────────────────────────────────────────
    "healthcare_magic":          load_healthcare_magic,
    "mentalchat":                load_mentalchat,
    "fitness_qa":                load_fitness_qa,
    "fitness_chat":              load_fitness_chat,
    "mental_counseling":         load_mental_counseling,
    "npc_dialogue":              load_npc_dialogue,
    "rpg_dialogues":             load_rpg_dialogues,
    "xp_qa":                     load_xp_qa,
    # ── Must-have additions ───────────────────────────────────────────────────
    "gym_exercise":              load_gym_exercise,
    "fitness_qa_large":          load_fitness_qa_large,
    "motivational_interviewing": load_motivational_interviewing,
    "dnd_crd3":                  load_dnd_crd3,
    # ── Build-specific handcrafted Q&A ────────────────────────────────────────
    "intelligence_qa":           load_intelligence_qa,
    "dexterity_qa":              load_dexterity_qa,
    "creative_qa":               load_creative_qa,
    "entrepreneur_qa":           load_entrepreneur_qa,
    "gamer_qa":                  load_gamer_qa,
    # ── Build-specific HuggingFace datasets ───────────────────────────────────
    "education_hf":              load_education_hf,
    "socratic_conversations":    load_socratic_conversations,
    "writing_prompts":           load_writing_prompts,
    "startup_interviews":        load_startup_interviews,
    "sales_conversations":       load_sales_conversations,
    "dota2_instruct":            load_dota2_instruct,
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
        if source == "healthcare_magic":
            records.extend(SOURCE_MAP[source](max_samples=max_hcm_samples))
        else:
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
