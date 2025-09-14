import os
import time
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv()  # Load .env if needed

# Read Discord webhook
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Your bot start command
BOT_COMMAND = ["python", "runner.py"]


def is_connected():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

def send_discord_log(message):
    if not DISCORD_WEBHOOK_URL:
        print("[!] No webhook set")
        return
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
        if response.status_code not in [200, 204]:
            print("[!] Discord error:", response.status_code, response.text)
    except Exception as e:
        print("[!] Discord send failed:", str(e))

def run_bot():
    while True:
        if is_connected():
            send_discord_log("✅ Bot starting from watchdog")
            print("[Watchdog] Internet available. Starting bot...")

            # Run the bot
            process = subprocess.Popen(BOT_COMMAND)

            # Wait for bot to exit or crash
            process.wait()

            send_discord_log("⚠️ Bot stopped or crashed. Restarting in 30 seconds.")
            print("[Watchdog] Bot exited. Waiting to restart...")

            time.sleep(30)  # Wait before restarting
        else:
            print("[Watchdog] No internet. Waiting...")
            time.sleep(10)

if __name__ == "__main__":
    send_discord_log("🔁 Watchdog started. Waiting for bot conditions...")
    run_bot()
