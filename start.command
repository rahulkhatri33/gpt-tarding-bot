

cd /Users/yogi/Documents/YOGI_bot || {
  echo "❌ Failed to navigate to bot directory."
  exit 1
}

echo "🚀 Starting YOGI's trading bot..."
python3 -m live.runner

read -p "✅ Bot stopped. Press any key to exit..."
