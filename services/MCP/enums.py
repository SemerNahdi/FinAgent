# File: enums.py
from enum import Enum
from dataclasses import dataclass
from typing import Set
import re


class AgentType(Enum):
    STOCK = "stock"
    RAG = "rag"
    PORTFOLIO = "portfolio"
    EMAIL = "email"
    WEBSEARCH = "websearch"


@dataclass
class AgentConfig:
    type: AgentType
    pattern: re.Pattern
    priority: int
    dependencies: Set[AgentType]
    required_keywords: Set[str]
    cache_ttl: int = 0
