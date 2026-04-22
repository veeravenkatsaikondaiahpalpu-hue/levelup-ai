# ⚔️ LevelUp — Gamify Your Entire Life

> *Every grind earns XP. Every build earns a legend.*

**LevelUp is a real-life RPG for everything you do.**
Hit the gym? XP. Ship a side project? XP. Finish a chapter, paint something, meditate, rank up in-game, learn to code? **XP.**

Most self-improvement apps lock you into one lane — fitness, productivity, or mindfulness. LevelUp covers all of it. Pick your **Build**, earn XP for the things you already do, and get coached by an AI that speaks your language — whether that's warrior intensity, scholarly precision, startup energy, or pure gaming hype.

This isn't a fitness app. This isn't a productivity app. This is **your life, gamified.**

---

## 🎮 The Builds

| Build | AI Coach | Focus |
|-------|----------|-------|
| ⚔️ **TITAN** | Warrior coach | Strength, lifting, nutrition |
| 🧠 **ORACLE** | Scholarly mentor | Programming, learning, science |
| 👁️ **PHANTOM** | Precision coach | Martial arts, parkour, agility |
| 🌿 **SAGE** | Mindful guide | Wellness, meditation, mental health |
| 🎨 **MUSE** | Creative director | Art, music, writing, design |
| 💼 **EMPIRE** | Startup co-founder | Business, finance, productivity |
| 🎮 **GG** | Esports hype coach | Gaming, streaming, competitive |

---

## 🏗️ Current Architecture

```
levelup-ai/
├── xp_engine/          XP calculations, streaks, badges, build mechanics
├── chatbot/
│   ├── fine_tuning/    QLoRA training pipeline (LLaMA 3.2 3B)
│   ├── inference.py    Model loading + multi-turn chat
│   ├── prompt_template.py  LLaMA 3 chat format utilities
│   └── system_prompt.py    Dynamic runtime prompts with user stats
├── anomaly_detection/  ML models for activity anomaly detection
├── api/                FastAPI backend (chat, XP, profile routes)
├── voice/              Whisper STT + ElevenLabs TTS
├── data/               Dataset collection + fine-tuning pipeline
└── models/             Trained LoRA adapters
```

---

## 🤖 AI Chatbot — Fine-Tuning Pipeline

The chatbot is a **QLoRA fine-tuned LLaMA 3.2 3B** model with build-specific personas trained on **76,616 samples** across 7 builds.

### Dataset Breakdown

| Build | Train Samples | Sources |
|-------|:------------:|---------|
| ORACLE | 23,268 | MathDial tutoring, Python/CS datasets, Reddit r/explainlikeimfive |
| GG | 9,290 | Gaming forums, D&D sessions, esports Q&A |
| PHANTOM | 9,419 | HealthCareMagic (movement), martial arts forums |
| MUSE | 9,079 | Writing prompts, creative HF datasets, r/worldbuilding |
| EMPIRE | 8,695 | Finance-Alpaca, business Q&A, productivity datasets |
| SAGE | 8,597 | Psychology-10k, mental health counseling, mindfulness |
| TITAN | 8,268 | Reddit fitness CSVs, HealthCareMagic, strength training |
| **TOTAL** | **76,616** | **9 sources across HuggingFace + Reddit + GitHub** |

### Training Config
- **Model**: `unsloth/Llama-3.2-3B-Instruct` (QLoRA 4-bit NF4)
- **LoRA**: r=16, α=32, targeting all attention + MLP projections
- **Trainer**: TRL SFTTrainer with sequence packing
- **Hardware**: NVIDIA RTX 4060 Laptop GPU (8GB VRAM)

### Run Training
```bash
# Overnight full training (~8-12 hours on RTX 4060)
python -m chatbot.fine_tuning.train

# Single build
python -m chatbot.fine_tuning.train --build ORACLE

# Quick pipeline test
python -m chatbot.fine_tuning.train --smoke_test
```

---

## 🚀 Running the API

```bash
# Start the FastAPI backend
uvicorn api.main:app --port 8000

# Chat endpoint
POST http://localhost:8000/api/chat
{
  "build": "TITAN",
  "message": "How do I break through a deadlift plateau?",
  "history": []
}
```

---

## ⚙️ Setup

```bash
# 1. Clone and create venv
git clone <repo>
cd levelup-ai
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set HuggingFace token (for model downloads)
echo "HF_TOKEN=your_token_here" > .env

# 4. Run smoke test
python -m chatbot.fine_tuning.train --smoke_test
```

---

## ✅ What's Built (v1)

- [x] **XP Engine** — full calculation engine, streaks, shields, badges, 52 tests passing
- [x] **7 Build Mechanics** — GAMER build with Boss Day, Overtime Mode, Combo Multiplier
- [x] **Dataset Pipeline** — 76,616 training samples across 9 sources, fully deduplicated
- [x] **QLoRA Training Script** — SFTTrainer + 4-bit quantisation, smoke-tested on GPU
- [x] **FastAPI Backend** — `/api/chat` endpoint with dynamic system prompts
- [x] **Inference Engine** — multi-turn chat, streaming, build persona switching
- [x] **Anomaly Detection** — Logistic Regression + Decision Tree on activity logs

---

## 🔭 Roadmap

### Phase 2 — Social Layer

The next major evolution of LevelUp is turning it from a solo app into a **community platform** where your real-life grind connects you with others on the same path.

#### 🏆 Global & Build Leaderboards
- Weekly and all-time XP leaderboards per build (TITAN vs TITAN, GG vs GG)
- Regional leaderboards — see who's grinding hardest in your city
- Streak Hall of Fame — longest active streaks per build
- Badge showcase on public profiles

#### 📍 Nearby Builders — Location-Based Discovery
- Find people with your build in your area
- See what activities they're logging (public feed opt-in)
- Gym partners for TITAN builds, study groups for ORACLE, jam sessions for MUSE
- Privacy-first: only shows city/district, never exact location

#### 🤝 Connection Suggestions
- AI-powered connection recommendations based on:
  - Same primary build
  - Complementary builds (e.g. EMPIRE + ORACLE = business + tech)
  - Similar XP level (avoid matching beginners with veterans)
  - Overlapping activity times — "You both grind at 6am"
- Mutual build unlock progress (users who unlocked the same secondary build)

#### 📣 Build Feed — Social Posts
- Post thoughts, session highlights, and milestones to your build's public feed
- "Just hit Diamond on TITAN after 6 months. Here's what changed my training 🧵"
- React with build-specific emojis (⚔️ for TITAN, 🧠 for ORACLE)
- Comments stay in-character: TITAN users see warrior energy, SAGE users see calm wisdom
- Share XP milestones, badge unlocks, and streak records directly to feed

#### 🔥 Challenges & Guilds
- Weekly build challenges: "ORACLE Challenge: 10 hours of deep study this week"
- Guilds: form groups of up to 10 builders, pool XP for group rewards
- Guild leaderboards and shared milestone celebrations
- Cross-build guilds (e.g. "Grind & Flow" = TITAN + SAGE members)

#### 🎯 Accountability Partners
- Pair with one accountability partner from your build
- Daily check-in notifications: "Your ORACLE partner logged 2 hours of coding today"
- Shared streak — if either partner breaks streak, both get notified
- Weekly summary comparing your progress side-by-side

### Phase 3 — Intelligence Layer

- **Smart Scheduling**: AI suggests optimal training times based on your logged patterns
- **Build DNA**: visual breakdown of your activity mix — are you a pure TITAN or TITAN/ORACLE hybrid?
- **Anomaly Coaching**: when the anomaly detection flags unusual drops, SAGE/TITAN step in with personalised recovery plans
- **Voice Mode**: full Whisper STT + ElevenLabs TTS pipeline for hands-free coaching mid-workout

### Phase 4 — Ecosystem

- **Marketplace**: creators sell custom build challenges, workout plans, and study roadmaps
- **Brand Partnerships**: gyms, learning platforms, and productivity tools unlock bonus XP
- **API for Wearables**: auto-log activities from Apple Watch, Fitbit, Garmin
- **LevelUp for Teams**: enterprise version for companies to gamify team productivity

---

## 👥 Team


A team of 3 building the platform where **every aspect of your life** earns XP — strength, intelligence, creativity, wellness, hustle, dexterity, and gaming.

| Role | Focus |
|------|-------|
| AI & Backend | LLM fine-tuning, XP engine, FastAPI |
| ML Models | Anomaly detection, sentiment classifier |
| Mobile | React Native app, UI/UX |

---

## 📄 License

MIT — build your legend, share the code.
