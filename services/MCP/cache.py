# File: cache.py
# Contains CacheManager class for handling caching logic

from typing import Dict, Optional
import time
from .enums import AgentConfig, AgentType  # Assuming enums.py is in the same package

class CacheManager:
    def __init__(self, enable_cache: bool = True, agent_configs: Dict[AgentType, AgentConfig] = None):
        self.enable_cache = enable_cache
        self.agent_configs = agent_configs or {}
        self._cache: Dict[str, str] = {} if enable_cache else None
        self._cache_timestamps: Dict[str, float] = {} if enable_cache else None

    def _get_cache_key(self, agent_type: str, query: str) -> str:
        return f"{agent_type}:{hash(query.lower().strip())}"

    def get_cached(self, agent_type: str, query: str) -> Optional[str]:
        if not self.enable_cache or not self._cache:
            return None
        config = self.agent_configs.get(AgentType(agent_type))
        if not config or config.cache_ttl == 0:
            return None
        key = self._get_cache_key(agent_type, query)
        if (
            key in self._cache
            and time.time() - self._cache_timestamps[key] < config.cache_ttl
        ):
            return self._cache[key]
        self._cache.pop(key, None)
        self._cache_timestamps.pop(key, None)
        return None

    def set_cached(self, agent_type: str, query: str, result: str):
        if not self.enable_cache or not self._cache:
            return
        config = self.agent_configs.get(AgentType(agent_type))
        if not config or config.cache_ttl == 0:
            return
        key = self._get_cache_key(agent_type, query)
        self._cache[key] = result
        self._cache_timestamps[key] = time.time()

    def clear_cache(self):
        if self._cache:
            self._cache.clear()
            self._cache_timestamps.clear()

    def get_cache_stats(self) -> Dict:
        return {
            "enabled": self.enable_cache,
            "size": len(self._cache) if self._cache else 0,
            "entries": list(self._cache.keys()) if self._cache else [],
        }