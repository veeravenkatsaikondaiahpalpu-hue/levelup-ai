# ⚔️ LevelUp AI — Gamify Your Entire Life

> *Every grind earns XP. Every build earns a legend.*

**LevelUp is a real-life RPG for everything you do.**
Hit the gym? XP. Ship a side project? XP. Finish a chapter, paint something, meditate, rank up in-game, learn to code? **XP.**

Most self-improvement apps lock you into one lane. LevelUp covers all of it. Pick your **Build**, earn XP for the things you already do, and get coached by an AI that speaks your language — whether that's warrior intensity, scholarly precision, startup energy, or pure gaming hype.

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

## 🏗️ Architecture

```
levelup-ai/
├── frontend/
│   └── app.html            Single-page RPG app (HTML/CSS/JS, no framework)
├── api/
│   ├── main.py             FastAPI entry point — chat, TTS, STT, anomaly, sentiment
│   └── routes/
│       ├── chat.py         /api/chat — fine-tuned LLM chatbot
│       ├── activity.py     /api/activity/check — anomaly detection
│       ├── sentiment.py    /api/sentiment/analyze — sentiment classifier
│       ├── profile.py      /api/profile — user profile helpers
│       └── vision.py       /api/vision — Qwen2.5-VL visual analysis
├── chatbot/
│   ├── fine_tuning/        QLoRA training pipeline (LLaMA 3.2 3B)
│   ├── inference.py        Model loading + multi-turn chat
│   ├── sentiment.py        Sentiment analysis model
│   └── system_prompt.py    Dynamic runtime prompts per build
├── anomaly_detection/      Logistic Regression + Decision Tree models
├── voice/
│   ├── tts.py              ElevenLabs TTS — per-build voice profiles
│   └── stt.py              OpenAI Whisper STT — local transcription
├── data/                   Dataset collection + fine-tuning pipeline
├── models/                 Trained LoRA adapters
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/veerapal/levelup-ai.git
cd levelup-ai

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Install PyTorch with CUDA 12.8 FIRST
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
LEVELUP_LOAD_MODEL=true
ELEVENLABS_API_KEY=your_key_here     # optional — browser TTS used as fallback
WHISPER_MODEL=base                   # optional — base / small / medium
```

- **HF_TOKEN** — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (needed to download LLaMA 3.2)
- **ELEVENLABS_API_KEY** — [elevenlabs.io](https://elevenlabs.io) (optional, for per-build AI voices)

### 3. Run the Project

Open **two terminals**:

**Terminal 1 — Backend API (port 8000)**
```bash
uvicorn api.main:app --port 8000 --reload
```
Wait for `[LevelUp API] Model ready.` — then the chatbot is live.

**Terminal 2 — Frontend (port 8080)**
```bash
cd frontend
python -m http.server 8080
```

Open → **http://localhost:8080/app.html**

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | Fine-tuned LLM chatbot with build persona |
| `/api/activity/check` | POST | Anomaly detection on activity patterns |
| `/api/sentiment/analyze` | POST | Real-time sentiment from chat input |
| `/api/tts` | POST | ElevenLabs TTS — build-specific voice |
| `/api/stt` | POST | Whisper STT — audio file to text |
| `/api/voices` | GET | List all voice profiles per build |
| `/health` | GET | Server health + model load status |
| `/docs` | GET | Interactive Swagger API docs |

---

## 🤖 AI Chatbot — Fine-Tuning Pipeline

**QLoRA fine-tuned LLaMA 3.2 3B** with build-specific personas trained on **76,616 samples** across 7 builds.

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

```bash
# Full training (~8–12 hours on RTX 4060)
python -m chatbot.fine_tuning.train

# Single build
python -m chatbot.fine_tuning.train --build TITAN

# Quick smoke test (5 min)
python -m chatbot.fine_tuning.train --smoke_test
```

---

## 🎤 Voice Features

### STT — Speech to Text (Mic Input)
- Click the 🎤 mic button in the AI Coach chat
- Uses **browser Web Speech API** by default (zero setup, real-time)
- Falls back to **Whisper** (`/api/stt`) if browser STT is unavailable
- Transcribed speech auto-fills and sends the message

### TTS — Text to Speech (AI Voice Output)
- Click the 🔊 speaker icon in the chat header to toggle voice responses
- Uses **ElevenLabs** (`/api/tts`) when `ELEVENLABS_API_KEY` is set
- Each build has a unique voice profile:

| Build | Voice Style |
|-------|------------|
| TITAN | Deep, intense, punchy — low stability, high drama |
| ORACLE | Clear, measured, analytical — high stability, minimal style |
| PHANTOM | Fluid, energetic, dynamic — medium expressiveness |
| SAGE | Ultra-calm, slow, grounded — highest stability, near-zero style |
| MUSE | Warm, expressive, creative — lowest stability, most varied |
| EMPIRE | Authoritative, commanding — high stability, professional |
| GG | Hype, chaotic, gamer energy — max style, maximum variation |

- Falls back to **browser speechSynthesis** automatically if no ElevenLabs key is set

---

## 🛡️ Anomaly Detection

Two ML models detect suspicious activity patterns to prevent XP farming:

| Model | Detects |
|---|---|
| Logistic Regression | XP grinding, impossible streaks |
| Decision Tree | Intensity spoofing, session abuse |

**8 engineered features:** `activities_per_day`, `daily_xp_total`, `avg_session_duration`, `max_session_duration`, `xp_per_minute`, `intensity_switch_rate`, `streak_gap_days`, `sessions_at_cap_ratio`

---

## 🏆 XP & Badge System

| Badge | Levels | XP to Complete Tier |
|-------|--------|---------------------|
| 🥉 Bronze | 0–9 | ~50,000 XP |
| 🥈 Silver | 10–19 | ~55,000 XP |
| 🥇 Gold | 20–29 | ~60,000 XP |
| 💎 Diamond | 30–39 | ~65,000 XP |
| 👑 Legend | 40–49 | ~70,000 XP |
| 🏆 Build Complete | Lv 50 | ~360,000 XP total (~2 years avg) |

**Daily XP Cap:** 600 XP/day base. Bonus on exact streak milestone days only:
- Day 7 → 800 XP cap | Day 14 → 1,000 XP cap | Day 21 → 1,200 XP cap

---

## ✅ What's Built

- [x] **RPG Frontend** — Full SPA: splash, login, register, build selection, dashboard, quests, AI coach, journal
- [x] **XP Engine** — Daily cap, streak bonuses, level-up, badge tiers, Build Complete modal
- [x] **7 Build Personas** — Unique AI coach voice, quests, and XP rules per build
- [x] **Fine-tuned Chatbot** — QLoRA LLaMA 3.2 3B, 76k samples, 7 build personas
- [x] **Anomaly Detection** — Logistic Regression + Decision Tree on 8 engineered features
- [x] **Sentiment Analysis** — Real-time input mood detection with coach nudges
- [x] **Voice Input (STT)** — Browser Web Speech API + Whisper backend fallback
- [x] **Voice Output (TTS)** — ElevenLabs per-build voices + browser speechSynthesis fallback
- [x] **FastAPI Backend** — Chat, anomaly, sentiment, TTS, STT, profile routes

---

## 🔭 Roadmap

### Phase 2 — Social Layer
- Global & build leaderboards (weekly + all-time XP)
- Nearby Builders — location-based discovery (city-level, privacy-first)
- Build Feed — social posts, milestones, badge showcases
- Guilds — groups of up to 10 builders, pooled XP, shared challenges
- Accountability Partners — shared streaks, daily check-ins

### Phase 3 — Intelligence Layer
- **Smart Scheduling** — AI suggests optimal training times from logged patterns
- **Build DNA** — visual breakdown of your activity mix across builds
- **Anomaly Coaching** — personalised recovery plans when anomalies are detected
- **Wearable Sync** — auto-log from Apple Watch, Fitbit, Garmin

### Phase 4 — Ecosystem
- Marketplace — creators sell custom challenges, workout plans, study roadmaps
- Brand Partnerships — gyms, learning platforms unlock bonus XP
- LevelUp for Teams — enterprise version for company productivity gamification

---

## 👥 Team

A team of 3 — Applied AI, SRH Stuttgart

| Role | Focus |
|------|-------|
| AI & Backend | LLM fine-tuning, FastAPI, voice pipeline |
| ML Models | Anomaly detection, sentiment classifier |
| Frontend & UX | RPG SPA design, XP engine, game mechanics |

---

## 📄 License

MIT — build your legend, share the code.
