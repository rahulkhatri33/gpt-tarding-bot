import json
import os
from core.logger import global_logger as logger
from core.position_manager import position_manager
from binance_utils import BinanceClient

def main():
    logger.log_debug("Getting all positions")
    positions = position_manager.get_all_positions()
    logger.log_info(f"🔧 Starting open positions recovery process... {len(positions)} open positions found.")
    
    if not positions:
        logger.log_info("✅ No open positions to recover.")
        return

    binance_utils = BinanceClient()  # Fixed: Removed api_key and api_secret arguments

    for key, pos in positions.items():
        try:
            # Extract symbol and direction from key (e.g., "BTCUSDT_long")
            symbol, direction = key.split("_")
            logger.log_debug(f"Recovering position: Symbol={symbol}, Direction={direction}, Data={pos}")
            
            # Validate position data
            if not all(k in pos for k in ["entry", "qty", "sl", "tp"]):
                logger.log_error(f"Invalid position data for {key}: {pos}")
                continue

            # Check if position still exists on Binance
            binance_pos = binance_utils.get_futures_position(symbol)
            if not binance_pos or float(binance_pos["positionAmt"]) == 0:
                logger.log_info(f"Position {key} not found on Binance. Removing.")
                position_manager.remove_position(symbol, direction)
                continue

            # Update position with latest data (e.g., mark-to-market)
            updated_data = {
                "entry": float(pos["entry"]),
                "qty": float(pos["qty"]),
                "sl": float(pos["sl"]),
                "tp": float(pos["tp"]),
                "last_updated": binance_utils.get_server_time()
            }
            position_manager.update_position(symbol, direction, updated_data)
            logger.log_info(f"✅ Recovered position: {key}")

        except Exception as e:
            logger.log_error(f"❌ Failed to recover position {key}: {str(e)}")

if __name__ == "__main__":
    main()