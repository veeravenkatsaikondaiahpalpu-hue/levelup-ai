---
title: LevelUp AI API
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
short_description: FastAPI backend for LevelUp AI — fine-tuned LLaMA 3.2 chatbot
tags:
  - fastapi
  - llama
  - qlora
  - text-generation
---

# LevelUp AI — Backend API

FastAPI backend powering the LevelUp AI RPG self-improvement app.

## Endpoints

| Route | Method | Description |
|---|---|---|
| `/api/chat` | POST | Fine-tuned LLaMA 3.2 3B chatbot |
| `/api/activity/check` | POST | Anomaly detection (XP fraud) |
| `/api/sentiment/analyze` | POST | Real-time sentiment analysis |
| `/api/tts` | POST | ElevenLabs TTS per build |
| `/health` | GET | Health check |

## Environment Variables

Set these in the Space **Settings → Variables and Secrets**:

- `HF_TOKEN` — HuggingFace token (to download LLaMA 3.2)
- `ELEVENLABS_API_KEY` — optional, enables per-build AI voices
- `LEVELUP_ADAPTER` — HF repo ID of your LoRA adapter (e.g. `your-username/levelup-qlora`)
