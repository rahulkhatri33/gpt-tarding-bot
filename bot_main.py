# bot_main.py
import time
import pandas as pd
from typing import Dict, Optional
from core.logger import global_logger as logger
from core.config import get_scalper_config, get_scalper_usd_allocation
from binance_utils import BinanceUtils
from scalper_strategy import generate_binance_signal
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class BinanceBot:
    def __init__(self):
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError("API key or secret not found in .env")
        self.utils = BinanceUtils(api_key, api_secret)
        self.utils.load_symbol_info()
        self.config = get_scalper_config()
        self.positions = {}  # Store open positions

    def sync_positions(self):
        """Sync open positions with Binance for accurate state."""
        try:
            self.positions.clear()  # Clear stale positions
            positions = self.utils.client.get_position_risk()  # Fetch from Binance
            for pos in positions:
                if float(pos['positionAmt']) != 0:  # Non-zero positions only
                    symbol = pos['symbol']
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'direction': 'long' if float(pos['positionAmt']) > 0 else 'short',
                        'entry_price': float(pos['entryPrice']),
                        'size': abs(float(pos['positionAmt'])),
                        'stop_loss': float(pos['stopLoss']) if pos.get('stopLoss') else None,
                        'take_profit': float(pos['takeProfit']) if pos.get('takeProfit') else None
                    }
            logger.log_info(f"Synced {len(self.positions)} positions: {self.positions}")
        except Exception as e:
            logger.log_error(f"Failed to sync positions: {str(e)[:200]}")

    def run(self):
        """Main bot loop, reading base_pairs from config.json."""
        symbols = self.config['base_pairs']  # Read from config.json
        while True:
            self.sync_positions()  # Sync positions at start of cycle
            for symbol in symbols:
                try:
                    self.process_symbol(symbol)
                except Exception as e:
                    logger.log_error(f"Error processing {symbol}: {str(e)[:200]}")
            
            # Wait for next 5m candle
            time.sleep(300 - (time.time() % 300) + 2)  # +2 sec buffer

    def process_symbol(self, symbol: str):
        """Full processing pipeline for one symbol."""
        # Check max concurrent trades
        open_positions = len(self.positions)
        if open_positions >= self.config['max_concurrent_trades']['scalper']:
            logger.log_error(f"Max positions ({self.config['max_concurrent_trades']['scalper']}) reached. Skipping {symbol}")
            return

        # Check if position already exists
        if symbol in self.positions:
            logger.log_info(f"{symbol} â�³ Position already open: {self.positions[symbol]['direction']}")
            return

        # 1. Fetch data
        df = self.utils.fetch_klines(symbol, Client.KLINE_INTERVAL_5MINUTE, 300)
        if df.empty:
            return
            
        # 2. Generate signal
        signal = generate_binance_signal(symbol, df)
        if not signal:
            if self.config.get('verbose_no_signal', False):
                logger.log_info(f"{symbol} ðŸ“¡ No signal generated")
            return
            
        # 3. Prepare order
        order = self.prepare_order(symbol, signal)
        if not order:
            return
            
        # 4. Execute (live/dry run)
        if self.config.get('dry_run', True):
            logger.log_info(f"DRY RUN: Would execute {order}")
        else:
            self.execute_order(order)

    def prepare_order(self, symbol: str, signal: Dict) -> Optional[Dict]:
        """Create Binance-compatible order."""
        usd_amount = get_scalper_usd_allocation(symbol)
        qty = self.utils.calculate_quantity(symbol, usd_amount, signal['entry'])
        
        if not self.utils.validate_order(symbol, signal['entry'], qty):
            return None
            
        return {
            'symbol': symbol,
            'side': signal['side'],
            'type': 'LIMIT',
            'quantity': qty,
            'price': signal['entry'],
            'stopPrice': signal['sl'],
            'stopLimitPrice': signal['sl'] * 0.998,
            'timeInForce': 'GTC'
        }

    def execute_order(self, order: Dict):
        """Send order to Binance and update positions."""
        try:
            response = self.utils.client.create_order(**order)
            logger.log_trade(
                f"Executed {order['side']} {order['symbol']} "
                f"@{order['price']} SL:{order['stopPrice']}",
                source="SCALPER"
            )
            # Update positions after execution
            self.positions[order['symbol']] = {
                'symbol': order['symbol'],
                'direction': 'long' if order['side'] == 'BUY' else 'short',
                'entry_price': order['price'],
                'size': order['quantity'],
                'stop_loss': order['stopPrice'],
                'take_profit': None  # TP not set in order, fetch later if needed
            }
            return response
        except Exception as e:
            logger.log_error(f"Order failed: {str(e)[:200]}")
            return None

if __name__ == "__main__":
    bot = BinanceBot()
    bot.run()