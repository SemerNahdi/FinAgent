# File: cache.py
# Contains CacheManager class for handling caching logic

from typing import Dict, Optional, Any
import time
import json
from .enums import AgentConfig, AgentType  # Assuming enums.py is in the same package

class CacheManager:
    def __init__(self, enable_cache: bool = True, agent_configs: Dict[AgentType, AgentConfig] = None):
        self.enable_cache = enable_cache
        self.agent_configs = agent_configs or {}
        self._cache: Dict[str, str] = {} if enable_cache else None
        self._cache_timestamps: Dict[str, float] = {} if enable_cache else None

    def _get_cache_key(self, agent_type: str, query: str) -> str:
        return f"{agent_type}:{hash(query.lower().strip())}"

    def get_cached(self, agent_type: str, query: str) -> Optional[Any]:
        if not self.enable_cache or not self._cache:
            return None
        # Handle "final_response" as a special case (not a valid AgentType)
        if agent_type == "final_response":
            # Use a default TTL of 300 seconds for final responses
            cache_ttl = 300
        else:
            try:
                config = self.agent_configs.get(AgentType(agent_type))
                if not config or config.cache_ttl == 0:
                    return None
                cache_ttl = config.cache_ttl
            except (ValueError, KeyError):
                # Invalid agent_type, don't cache
                return None
        
        key = self._get_cache_key(agent_type, query)
        if (
            key in self._cache
            and time.time() - self._cache_timestamps[key] < cache_ttl
        ):
            cached_value = self._cache[key]
            # Try to deserialize JSON if it's a dict-like string
            try:
                return json.loads(cached_value)
            except (json.JSONDecodeError, TypeError):
                # If it's not JSON, return as-is (for backward compatibility)
                return cached_value
        self._cache.pop(key, None)
        self._cache_timestamps.pop(key, None)
        return None

    def set_cached(self, agent_type: str, query: str, result: Any):
        if not self.enable_cache or not self._cache:
            return
        # Handle "final_response" as a special case (not a valid AgentType)
        if agent_type == "final_response":
            # Allow caching final responses
            pass
        else:
            try:
                config = self.agent_configs.get(AgentType(agent_type))
                if not config or config.cache_ttl == 0:
                    return
            except (ValueError, KeyError):
                # Invalid agent_type, don't cache
                return
        
        key = self._get_cache_key(agent_type, query)
        # Serialize dict/list objects to JSON, keep strings as-is
        if isinstance(result, (dict, list)):
            self._cache[key] = json.dumps(result)
        else:
            self._cache[key] = str(result)
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