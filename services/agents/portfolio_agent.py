# services/agents/portfolio_agent.py

import re
from typing import Dict, List, Optional
import spacy
from spacy.matcher import Matcher
from datetime import datetime, timedelta
import logging

from services.tools.portfolio_tool import PortfolioTool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ParamError(ValueError):
    """Raised when required parameters are missing or malformed."""


class PortfolioAgent:
    """
    Refactored PortfolioAgent:
      - Removed redundant wrappers for missing tool methods (has_stock, quantity, purchase_info).
      - Simplified patterns and centralized them in a dict for scalability (easier to add/extend).
      - Optimized param extraction: Combined date logic, used list comprehensions.
      - Improved intent resolution: Simplified conflict checks, used sets efficiently.
      - Aligned method calls with refactored PortfolioTool (e.g., fetch_latest_prices).
      - Made debug logging more concise.
      - Consistent run() return: List[Dict] with safe normalization.
      - Added caching for parse_query if needed, but kept lightweight for one-off queries.
      - Scalability: Intent priorities/combinable as class vars; patterns dict allows easy extension.
      - Removed unused heuristics (e.g., ticker extraction if not used in wrappers).
    """

    INTENT_PRIORITY = {
        "FULL_RECORDS": 100,
        "FILTER_BY_DATE": 90,
        "SECTOR_ALLOCATION": 80,
        "TOP_HOLDINGS": 80,
        "PRICE_CHANGE_SINCE": 75,
        "PRICE_CHANGES": 70,
        "HISTORICAL_PRICES": 70,
        "PURCHASE_TIMELINE": 60,
        "LIVE_PRICES": 50,
        "ANALYZE": 10,
    }

    COMBINABLE = {
        ("FULL_RECORDS", "HISTORICAL_PRICES"),
        ("TOP_HOLDINGS", "LIVE_PRICES"),
        ("SECTOR_ALLOCATION", "LIVE_PRICES"),
        ("FILTER_BY_DATE", "TOP_HOLDINGS"),
    }

    PATTERNS = {
        "SECTOR_ALLOCATION": [
            [
                {"LOWER": "sector"},
                {
                    "LOWER": {
                        "IN": [
                            "allocation",
                            "allocation:",
                            "breakdown",
                            "distribution",
                            "weights",
                        ]
                    }
                },
            ],
            [
                {"LOWER": {"IN": ["sectors", "sectoral"]}},
                {
                    "OP": "?",
                    "LOWER": {"IN": ["allocation", "breakdown", "distribution"]},
                },
            ],
            [{"LOWER": "breakdown"}, {"LOWER": "by"}, {"LOWER": "sector"}],
        ],
        "TOP_HOLDINGS": [
            [
                {"LOWER": "top"},
                {"LIKE_NUM": True, "OP": "?"},
                {"LOWER": {"IN": ["holdings", "stocks", "positions", "holdings:"]}},
            ],
            [
                {"LOWER": {"IN": ["largest", "biggest"]}},
                {"LOWER": {"IN": ["holdings", "positions"]}},
            ],
            [{"LOWER": "top"}, {"LOWER": "holdings"}],
        ],
        "PRICE_CHANGE_SINCE": [
            [
                {"LOWER": {"IN": ["price", "value", "portfolio"]}},
                {"LOWER": {"IN": ["change", "changes", "performance"]}},
                {"LOWER": "since"},
            ],
            [{"LOWER": "performance"}, {"LOWER": "since"}],
            [{"LOWER": "changes"}, {"LOWER": "since"}],
        ],
        "PURCHASE_TIMELINE": [
            [
                {"LOWER": "purchase"},
                {"LOWER": {"IN": ["timeline", "history", "history:"]}},
            ],
            [{"LOWER": "buy"}, {"LOWER": {"IN": ["history", "timeline"]}}],
            [{"LOWER": "purchases"}, {"LOWER": "over"}, {"LOWER": "time"}],
        ],
        "FILTER_BY_DATE": [
            [
                {"LOWER": "filter"},
                {"LOWER": {"IN": ["by", "between", "range"]}},
                {"LOWER": "date", "OP": "?"},
            ],
            [{"LOWER": "from"}, {"IS_ALPHA": False, "OP": "?"}, {"LOWER": "to"}],
            [
                {"LOWER": "between"},
                {"OP": "?"},
                {"LOWER": {"IN": ["dates", "date"]}, "OP": "?"},
            ],
        ],
        "FULL_RECORDS": [
            [{"LOWER": "full"}, {"LOWER": {"IN": ["portfolio", "records", "summary"]}}],
            [{"LOWER": "all"}, {"LOWER": {"IN": ["records", "holdings"]}}],
            [{"LOWER": "portfolio"}, {"LOWER": "summary"}],
        ],
        "LIVE_PRICES": [
            [
                {"LOWER": {"IN": ["current", "live", "latest"]}},
                {"LOWER": {"IN": ["prices", "price"]}},
            ],
            [{"LOWER": "quote"}, {"LOWER": {"IN": ["prices", "price"]}}],
        ],
        "HISTORICAL_PRICES": [
            [
                {"LOWER": "historical"},
                {"LOWER": {"IN": ["prices", "price"]}},
                {"LOWER": "on", "OP": "?"},
            ],
            [{"LOWER": "prices"}, {"LOWER": "on"}, {"IS_PUNCT": False, "OP": "?"}],
        ],
        "PRICE_CHANGES": [
            [
                {"LOWER": "price"},
                {"LOWER": "changes"},
                {"LOWER": {"IN": ["from", "since"]}, "OP": "?"},
            ],
            [{"LOWER": "changes"}, {"LOWER": "since"}],
        ],
    }

    INTENT_TO_METHOD = {
        "SECTOR_ALLOCATION": "sector_allocation",
        "TOP_HOLDINGS": "top_holdings",
        "PRICE_CHANGE_SINCE": "price_change_since",
        "PURCHASE_TIMELINE": "purchase_timeline",
        "FILTER_BY_DATE": "filter_by_date",
        "FULL_RECORDS": "full_records",
        "LIVE_PRICES": "live_prices",
        "HISTORICAL_PRICES": "historical_prices",
        "PRICE_CHANGES": "price_changes",
        "ANALYZE": "analyze",
    }

    def __init__(self, portfolio_csv: str, metadata_csv: str, debug: bool = True):
        self.tool = PortfolioTool(portfolio_csv, metadata_csv)
        self.nlp = spacy.blank("en")
        self.matcher = Matcher(self.nlp.vocab)
        self.debug = debug
        self._define_patterns()

    def _log(self, *msg):
        if self.debug:
            print("[AGENT]", *msg)

    def _define_patterns(self):
        for intent, patterns in self.PATTERNS.items():
            self.matcher.add(intent, patterns)

    def parse_query(self, query: str) -> List[Dict[str, Optional[Dict]]]:
        if not query or not query.strip():
            self._log("No query provided, returning ANALYZE fallback.")
            return [
                {
                    "intent": "ANALYZE",
                    "method": "analyze",
                    "params": None,
                    "score": self.INTENT_PRIORITY.get("ANALYZE", 10),
                    "matches": [],
                }
            ]

        original_query = query.strip()
        doc = self.nlp(original_query.lower())
        matches = self.matcher(doc)

        detected = []
        seen = set()
        if matches and self.debug:
            self._log("Raw matcher hits:", matches)

        for match_id, start, end in matches:
            intent = self.nlp.vocab.strings[match_id]
            if intent in seen:
                continue
            seen.add(intent)

            span = doc[start:end]
            base_score = self.INTENT_PRIORITY.get(intent, 50)
            if len(span) == 1 and len(doc) > 4:
                base_score *= 0.6
            if start <= 2:
                base_score *= 1.05

            params = self._extract_params(intent, doc, start, end, original_query)
            if self.debug:
                self._log(
                    f"Matched intent '{intent}' -> span: '{span.text}' (start={start}, end={end})"
                )
            detected.append(
                {
                    "intent": intent,
                    "method": self.INTENT_TO_METHOD.get(intent, "analyze"),
                    "params": params,
                    "score": base_score,
                    "matches": [(intent, span.text, start, end)],
                }
            )

        if not detected:
            self._log("No intents detected; returning ANALYZE fallback.")
            return [
                {
                    "intent": "ANALYZE",
                    "method": "analyze",
                    "params": None,
                    "score": self.INTENT_PRIORITY.get("ANALYZE", 10),
                    "matches": [],
                }
            ]

        detected.sort(key=lambda x: x["score"], reverse=True)
        if self.debug:
            self._log(
                "Detected intents (sorted):",
                [(d["intent"], d["score"]) for d in detected],
            )

        resolved = self._resolve_and_combine(detected)
        if self.debug:
            self._log(
                "Resolved intents:",
                [(r["intent"], r["method"], r.get("params")) for r in resolved],
            )

        return resolved

    def _resolve_and_combine(self, detected: List[Dict]) -> List[Dict]:
        resolved = []
        included = set()

        for item in detected:
            intent = item["intent"]
            if intent in included:
                continue

            if any(
                self.INTENT_PRIORITY.get(inc, 0) - self.INTENT_PRIORITY.get(intent, 0)
                > 10
                and (inc, intent) not in self.COMBINABLE
                and (intent, inc) not in self.COMBINABLE
                for inc in included
            ):
                self._log(f"Skipping {intent} due to conflict.")
                continue

            combined = False
            for other in detected:
                if other is item:
                    continue
                pair = (intent, other["intent"])
                rev_pair = (other["intent"], intent)
                if pair in self.COMBINABLE or rev_pair in self.COMBINABLE:
                    if intent not in included:
                        resolved.append(item)
                        included.add(intent)
                    if other["intent"] not in included:
                        resolved.append(other)
                        included.add(other["intent"])
                    combined = True
                    break
            if combined:
                continue

            resolved.append(item)
            included.add(intent)

        # Deduplicate
        out = []
        seen = set()
        for r in resolved:
            key = (r["intent"], frozenset((r["params"] or {}).items()))
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out

    def _extract_params(
        self, intent: str, doc, start: int, end: int, original_query: str
    ) -> Optional[Dict]:
        params = {}
        iso_dates = re.findall(r"\d{4}-\d{2}-\d{2}", original_query)
        if iso_dates:
            if len(iso_dates) >= 2:
                params["start_date"], params["end_date"] = iso_dates[:2]
            else:
                params["date"] = iso_dates[0]

        if intent == "TOP_HOLDINGS":
            n = next(
                (int(token.text) for token in doc[start:end] if token.like_num), None
            )
            if n is None:
                n = next((int(token.text) for token in doc if token.like_num), None)
            params["n"] = n or 5

        date_intents = {"PRICE_CHANGE_SINCE", "HISTORICAL_PRICES", "PRICE_CHANGES"}
        if intent in date_intents and "date" not in params:
            params["date"] = self._parse_natural_date(original_query)

        if intent == "FILTER_BY_DATE" and (
            "start_date" not in params or "end_date" not in params
        ):
            m = re.search(r"last\s+(week|month|year)", original_query, flags=re.I)
            if m:
                unit = m.group(1).lower()
                end_d = datetime.utcnow().date()
                days = {"week": 7, "month": 30, "year": 365}.get(unit, 0)
                start_d = end_d - timedelta(days=days)
                params["start_date"] = start_d.isoformat()
                params["end_date"] = end_d.isoformat()

        # Ticker extraction (if needed in future; currently unused)
        # tickers = re.findall(r"\b[A-Z]{2,5}\b", original_query)
        # if tickers:
        #     params["ticker"] = tickers[0]

        if self.debug:
            self._log(f"Extracted params for {intent} ->", params or None)
        return params if params else None

    def _parse_natural_date(self, text: str) -> Optional[str]:
        t = text.lower()
        today = datetime.utcnow().date()

        if "today" in t:
            return today.isoformat()
        if "yesterday" in t:
            return (today - timedelta(days=1)).isoformat()

        m = re.search(r"last\s+(week|month|year)", t)
        if m:
            unit = m.group(1)
            days = {"week": 7, "month": 30, "year": 365}.get(unit)
            if days:
                return (today - timedelta(days=days)).isoformat()

        m2 = re.search(r"since\s+(\d{4})\b", t)
        if m2:
            year = int(m2.group(1))
            return datetime(year, 1, 1).date().isoformat()

        return None

    def run(self, query: str = None) -> list:
        if not query or not query.strip():
            self._log("Empty query; running analyze fallback.")
            try:
                res = self.analyze()
                return [{"method": "analyze", "result": res, "meta": {}}]
            except Exception as e:
                return [{"method": "analyze", "result": f"Error: {str(e)}", "meta": {}}]

        actions = self.parse_query(query)
        results = []

        for action in actions:
            method_name = action["method"]
            params = action.get("params") or {}
            intent = action["intent"]
            self._log(f"Running: {method_name} (intent: {intent}, params: {params})")

            method = getattr(self, method_name, None)
            if not method:
                msg = f"Method not found for {intent}"
                self._log(msg)
                results.append(
                    {"method": method_name, "result": msg, "meta": {"intent": intent}}
                )
                continue

            try:
                result = method(**params) if params else method()
                results.append(
                    {
                        "method": method_name,
                        "result": result or {},
                        "meta": {"intent": intent, "params": params},
                    }
                )
            except Exception as e:
                self._log(f"Error in {method_name}: {e}")
                results.append(
                    {
                        "method": method_name,
                        "result": f"Error: {str(e)}",
                        "meta": {"intent": intent, "params": params},
                    }
                )

        return results

    # Wrapper methods (aligned with refactored PortfolioTool)
    def price_change_since(self, date: str = None):
        if not date:
            raise ParamError("Requires 'date' (ISO or natural).")
        return self._safe_tool_call(self.tool.analyze, include_changes=date)

    def purchase_timeline(self):
        return self._safe_tool_call(self.tool.get_purchase_timeline)

    def top_holdings(self, n: int = 5):
        n = int(n)
        if n <= 0:
            raise ParamError("n must be positive integer.")
        return self._safe_tool_call(self.tool.top_holdings, n)

    def sector_allocation(self):
        return self._safe_tool_call(self.tool.get_sector_allocation)

    def filter_by_date(self, start_date: str = None, end_date: str = None):
        if not (start_date and end_date):
            raise ParamError("Requires 'start_date' and 'end_date' (YYYY-MM-DD).")
        if not all(re.match(r"^\d{4}-\d{2}-\d{2}$", d) for d in (start_date, end_date)):
            raise ParamError("Invalid date format.")
        return self._safe_tool_call(
            self.tool.filter_by_purchase_date, start_date, end_date
        )

    def full_records(self):
        return self._safe_tool_call(self.tool.get_portfolio_summary)

    def live_prices(self):
        res = self._safe_tool_call(self.tool.fetch_latest_prices)
        return res[0] if isinstance(res, tuple) and res else res

    def historical_prices(self, date: str = None):
        if not date:
            raise ParamError("Requires 'date'.")
        res = self._safe_tool_call(self.tool.fetch_historical_prices, date)
        return res[0] if isinstance(res, tuple) and res else res

    def price_changes(self, date: str = None):
        if not date:
            raise ParamError("Requires 'date'.")
        return self._safe_tool_call(self.tool.get_price_changes, date)

    def analyze(self, include_changes: str = None):
        return self._safe_tool_call(self.tool.analyze, include_changes=include_changes)

    def _safe_tool_call(self, func, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 1:
                return result[0]
            return result
        except Exception as e:
            logger.exception(
                f"Tool call failed: {func.__name__} args={args} kwargs={kwargs}"
            )
            raise
