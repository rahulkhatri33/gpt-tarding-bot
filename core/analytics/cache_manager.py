# core/analytics/cache_manager.py

class RollingCache:
    def __init__(self):
        self.cache = {}

    def get(self, symbol):
        return self.cache.get(symbol)

    def set(self, symbol, df):
        self.cache[symbol] = df

    def clear(self):
        self.cache.clear()

# Global instance for shared cache
rolling_cache = RollingCache()
