"""
chat_test.py — Quick interactive test for the LevelUp API
Run: python chat_test.py
"""
import requests
import json

API = "http://localhost:8001/api/chat"
BUILDS = ["TITAN", "ORACLE", "PHANTOM", "SAGE", "MUSE", "EMPIRE", "GG"]

def chat(build, message, history):
    payload = {"build": build, "message": message, "history": history}
    r = requests.post(API, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def main():
    print("\n" + "="*60)
    print("  🎮 LevelUp AI — Interactive Chat Test")
    print("="*60)
    print(f"  Builds: {' | '.join(BUILDS)}")
    print("  Type 'switch' to change build, 'quit' to exit")
    print("="*60 + "\n")

    build = input("Pick your build (default TITAN): ").strip().upper() or "TITAN"
    if build not in BUILDS:
        build = "TITAN"

    history = []
    print(f"\n✅ Chatting with {build} build. Go!\n")

    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGG! Later 👋")
            break

        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "q"):
            print("GG! Later 👋")
            break
        if msg.lower() == "switch":
            build = input("New build: ").strip().upper()
            if build not in BUILDS:
                print(f"Unknown build. Choose from: {BUILDS}")
            else:
                history = []
                print(f"Switched to {build} (history cleared)\n")
            continue

        try:
            print(f"\n{build}: ", end="", flush=True)
            result = chat(build, msg, history)
            print(result["reply"])
            print(f"  [{result['latency_ms']}ms]\n")

            history.append({"role": "user",      "content": msg})
            history.append({"role": "assistant",  "content": result["reply"]})
            if len(history) > 12:
                history = history[-12:]
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
