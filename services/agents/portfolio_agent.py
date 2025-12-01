# services/agents/portfolio_agent.py

import re
from typing import Dict, List, Optional, Union, Tuple
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
    Improved PortfolioAgent with debug printing:
      - Correct spaCy numeric usage (LIKE_NUM)
      - Stronger patterns and heuristics to reduce false positives
      - Intent scoring and conflict resolution
      - Robust date parsing (ISO + simple natural phrases)
      - Centralized param extraction and validation
      - Consistent run() return: List[Dict]
      - Safe normalization of PortfolioTool return types
      - Console debug printing of matches/intents/params/actions when self.debug is True
    """

    # INTENT PRIORITIES (higher => handled earlier/more "important")
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

    # Intents that are allowed to combine (rather than conflict)
    COMBINABLE = {
        ("FULL_RECORDS", "HISTORICAL_PRICES"),
        ("TOP_HOLDINGS", "LIVE_PRICES"),
        ("SECTOR_ALLOCATION", "LIVE_PRICES"),
        ("FILTER_BY_DATE", "TOP_HOLDINGS"),
    }

    def __init__(self, portfolio_csv: str, metadata_csv: str, debug: bool = True):
        """
        :param portfolio_csv: path to portfolio CSV for PortfolioTool
        :param metadata_csv: path to metadata CSV for PortfolioTool
        :param debug: if True, prints debug traces to console
        """
        self.tool = PortfolioTool(portfolio_csv, metadata_csv)

        # spaCy blank model (fast)
        self.nlp = spacy.blank("en")
        self.matcher = Matcher(self.nlp.vocab)

        # debug printing toggle
        self.debug = debug

        # Define patterns
        self._define_patterns()

    # -------------------------
    # Debug helper
    # -------------------------
    def _log(self, *msg):
        """Console-only debug helper (prints only when self.debug is True)."""
        if self.debug:
            print("[AGENT]", *msg)

    # -------------------------
    # Patterns
    # -------------------------
    def _define_patterns(self):
        """Define spaCy rule-based patterns. Use LIKE_NUM for numbers."""
        # Sector allocation: require 'sector' near allocation words
        sector_patterns = [
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
        ]
        self.matcher.add("SECTOR_ALLOCATION", sector_patterns)

        # Top holdings: allow optional numeric token
        top_holdings_patterns = [
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
        ]
        self.matcher.add("TOP_HOLDINGS", top_holdings_patterns)

        # Price change since / performance since
        price_change_patterns = [
            [
                {"LOWER": {"IN": ["price", "value", "portfolio"]}},
                {"LOWER": {"IN": ["change", "changes", "performance"]}},
                {"LOWER": "since"},
            ],
            [{"LOWER": "performance"}, {"LOWER": "since"}],
            [{"LOWER": "changes"}, {"LOWER": "since"}],
        ]
        self.matcher.add("PRICE_CHANGE_SINCE", price_change_patterns)

        # Purchase timeline / buy history
        timeline_patterns = [
            [
                {"LOWER": "purchase"},
                {"LOWER": {"IN": ["timeline", "history", "history:"]}},
            ],
            [{"LOWER": "buy"}, {"LOWER": {"IN": ["history", "timeline"]}}],
            [{"LOWER": "purchases"}, {"LOWER": "over"}, {"LOWER": "time"}],
        ]
        self.matcher.add("PURCHASE_TIMELINE", timeline_patterns)

        # Filter by date
        filter_date_patterns = [
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
        ]
        self.matcher.add("FILTER_BY_DATE", filter_date_patterns)

        # Full records / all records
        full_records_patterns = [
            [{"LOWER": "full"}, {"LOWER": {"IN": ["portfolio", "records", "summary"]}}],
            [{"LOWER": "all"}, {"LOWER": {"IN": ["records", "holdings"]}}],
            [{"LOWER": "portfolio"}, {"LOWER": "summary"}],
        ]
        self.matcher.add("FULL_RECORDS", full_records_patterns)

        # Live prices
        live_prices_patterns = [
            [
                {"LOWER": {"IN": ["current", "live", "latest"]}},
                {"LOWER": {"IN": ["prices", "price"]}},
            ],
            [{"LOWER": "quote"}, {"LOWER": {"IN": ["prices", "price"]}}],
        ]
        self.matcher.add("LIVE_PRICES", live_prices_patterns)

        # Historical prices
        historical_prices_patterns = [
            [
                {"LOWER": "historical"},
                {"LOWER": {"IN": ["prices", "price"]}},
                {"LOWER": "on", "OP": "?"},
            ],
            [{"LOWER": "prices"}, {"LOWER": "on"}, {"IS_PUNCT": False, "OP": "?"}],
        ]
        self.matcher.add("HISTORICAL_PRICES", historical_prices_patterns)

        # Price changes (more general)
        price_changes_patterns = [
            [
                {"LOWER": "price"},
                {"LOWER": "changes"},
                {"LOWER": {"IN": ["from", "since"]}, "OP": "?"},
            ],
            [{"LOWER": "changes"}, {"LOWER": "since"}],
        ]
        self.matcher.add("PRICE_CHANGES", price_changes_patterns)

    # -------------------------
    # Query Parsing + Scoring + Conflict Resolution
    # -------------------------
    def parse_query(self, query: str) -> List[Dict[str, Optional[Dict]]]:
        """
        Analyze query using spaCy Matcher and return actions list:
        [{'intent','method','params','score','matches':[...]}, ...]
        """
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
        query_lower = original_query.lower()
        doc = self.nlp(query_lower)
        matches = self.matcher(doc)

        detected = []
        seen = set()
        # Print raw matches for debugging
        if matches:
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
            # Log which pattern produced which span
            self._log(
                f"Matched intent '{intent}' -> span: '{span.text}' (start={start}, end={end})"
            )
            detected.append(
                {
                    "intent": intent,
                    "method": self._intent_to_method(intent),
                    "params": params,
                    "score": base_score,
                    "matches": [(intent, span.text, start, end)],
                }
            )

        if not detected:
            self._log("No intents detected by matcher; returning ANALYZE fallback.")
            return [
                {
                    "intent": "ANALYZE",
                    "method": "analyze",
                    "params": None,
                    "score": self.INTENT_PRIORITY.get("ANALYZE", 10),
                    "matches": [],
                }
            ]

        # Sort by score desc
        detected.sort(key=lambda x: x["score"], reverse=True)

        self._log(
            "Detected intents (sorted):", [(d["intent"], d["score"]) for d in detected]
        )

        resolved = self._resolve_and_combine(detected)

        # Log resolved intents
        self._log(
            "Resolved intents to execute:",
            [(r["intent"], r["method"], r.get("params")) for r in resolved],
        )

        return resolved

    def _intent_to_method(self, intent: str) -> str:
        mapping = {
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
        return mapping.get(intent, "analyze")

    def _resolve_and_combine(self, detected: List[Dict]) -> List[Dict]:
        """
        Resolve detection list into final actionable intents.
        Combines non-conflicting intents defined in COMBINABLE, prefers higher priority.
        """
        if not detected:
            return []

        resolved = []
        included_intents = set()

        for item in detected:
            intent = item["intent"]
            if intent in included_intents:
                continue

            # skip if a higher-priority included intent strongly outranks this one
            conflict = False
            for inc in included_intents:
                if (inc, intent) not in self.COMBINABLE and (
                    intent,
                    inc,
                ) not in self.COMBINABLE:
                    if (
                        self.INTENT_PRIORITY.get(inc, 0)
                        - self.INTENT_PRIORITY.get(intent, 0)
                        > 10
                    ):
                        conflict = True
                        break
            if conflict:
                self._log(f"Skipping intent {intent} due to higher-priority conflict.")
                continue

            # check for combinable partner
            combined = False
            for other in detected:
                if other is item:
                    continue
                pair = (intent, other["intent"])
                if (
                    pair in self.COMBINABLE
                    or (other["intent"], intent) in self.COMBINABLE
                ):
                    # include both if not already
                    if intent not in included_intents:
                        resolved.append(item)
                        included_intents.add(intent)
                    if other["intent"] not in included_intents:
                        resolved.append(other)
                        included_intents.add(other["intent"])
                    combined = True
                    break
            if combined:
                continue

            # include normally
            resolved.append(item)
            included_intents.add(intent)

        # Deduplicate while preserving order and normalize
        out = []
        seen = set()
        for r in resolved:
            key = (
                r["intent"],
                tuple(sorted((r["params"] or {}).items())) if r["params"] else None,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(r)

        return out

    # -------------------------
    # Parameter extraction (improved)
    # -------------------------
    def _extract_params(
        self, intent: str, doc, start: int, end: int, original_query: str
    ) -> Optional[Dict]:
        """
        Extract parameters:
          - ISO dates (YYYY-MM-DD)
          - natural dates (today, yesterday, last week/month/year, since <year>)
          - numeric n for top holdings
          - tickers (heuristic)
        Returns params dict or None.
        """
        params = {}
        # ISO dates
        iso_dates = re.findall(r"\d{4}-\d{2}-\d{2}", original_query)
        if iso_dates:
            if len(iso_dates) >= 2:
                params["start_date"] = iso_dates[0]
                params["end_date"] = iso_dates[1]
            else:
                params["date"] = iso_dates[0]

        # TOP_HOLDINGS numeric extraction
        if intent == "TOP_HOLDINGS":
            n = None
            for token in doc[start:end]:
                if token.like_num:
                    try:
                        n = int(token.text)
                        break
                    except Exception:
                        continue
            if n is None:
                for token in doc:
                    if token.like_num:
                        try:
                            n = int(token.text)
                            break
                        except Exception:
                            continue
            params["n"] = n or 5

        # PRICE_CHANGE_SINCE / HISTORICAL_PRICES / PRICE_CHANGES natural dates
        if intent in {"PRICE_CHANGE_SINCE", "HISTORICAL_PRICES", "PRICE_CHANGES"}:
            if "date" not in params:
                natural_date = self._parse_natural_date(original_query)
                if natural_date:
                    params["date"] = natural_date

        # FILTER_BY_DATE: ensure start/end
        if intent == "FILTER_BY_DATE":
            if "start_date" in params and "end_date" in params:
                pass
            else:
                m = re.search(r"last\s+(week|month|year)", original_query, flags=re.I)
                if m:
                    delta_word = m.group(1).lower()
                    end_d = datetime.utcnow().date()
                    if delta_word == "week":
                        start_d = end_d - timedelta(days=7)
                    elif delta_word == "month":
                        start_d = end_d - timedelta(days=30)
                    else:
                        start_d = end_d - timedelta(days=365)
                    params["start_date"] = start_d.isoformat()
                    params["end_date"] = end_d.isoformat()

        # Heuristic ticker extraction (uppercase tokens length 2-5)
        tickers = re.findall(r"\b[A-Z]{2,5}\b", original_query)
        if tickers:
            params["ticker"] = tickers[0]

        # Debug print extracted params
        self._log(
            f"Extracted params for intent {intent} ->", params if params else None
        )

        return params if params else None

    def _parse_natural_date(self, text: str) -> Optional[str]:
        """
        Support tokens:
          - today
          - yesterday
          - last week/month/year
          - since <year>
        Returns ISO date string or None.
        """
        t = text.lower()
        today = datetime.utcnow().date()

        if "today" in t:
            return today.isoformat()
        if "yesterday" in t:
            return (today - timedelta(days=1)).isoformat()

        m = re.search(r"last\s+(week|month|year)", t)
        if m:
            unit = m.group(1)
            if unit == "week":
                return (today - timedelta(days=7)).isoformat()
            if unit == "month":
                return (today - timedelta(days=30)).isoformat()
            if unit == "year":
                return (today - timedelta(days=365)).isoformat()

        m2 = re.search(r"since\s+(\d{4})\b", t)
        if m2:
            try:
                year = int(m2.group(1))
                return datetime(year, 1, 1).date().isoformat()
            except Exception:
                return None

        return None

    # -------------------------
    # Primary interface (Adaptive)
    # -------------------------
    def run(self, query: str = None) -> List[Dict]:
        """
        Main entry: Parse and execute. Always returns a list of dicts:
        [{'method': <name>, 'result': <data or error str>, 'meta': {...}}, ...]
        """
        if not query or not query.strip():
            self._log("run() called with empty query; running analyze() fallback.")
            try:
                res = self.analyze()
                return [{"method": "analyze", "result": res, "meta": {}}]
            except Exception as e:
                return [{"method": "analyze", "result": f"Error: {str(e)}", "meta": {}}]

        actions = self.parse_query(query)
        results: List[Dict] = []

        for action in actions:
            method_name = action.get("method")
            params = action.get("params") or {}
            intent = action.get("intent")

            # Print which action will run
            self._log(
                "Running action:", method_name, "intent:", intent, "params:", params
            )

            method = getattr(self, method_name, None)
            if not method:
                msg = f"Method not found for intent {intent}"
                self._log(msg)
                results.append(
                    {"method": method_name, "result": msg, "meta": {"intent": intent}}
                )
                continue

            try:
                result = method(**params) if params else method()
                self._log("Action result for", method_name, "->", type(result).__name__)
                results.append(
                    {
                        "method": method_name,
                        "result": result,
                        "meta": {"intent": intent, "params": params},
                    }
                )
            except ParamError as pe:
                self._log(f"Parameter error in {method_name}: {pe}")
                results.append(
                    {
                        "method": method_name,
                        "result": f"Parameter error: {str(pe)}",
                        "meta": {"intent": intent, "params": params},
                    }
                )
            except TypeError as te:
                self._log(f"Bad parameters for {method_name}: {te}")
                results.append(
                    {
                        "method": method_name,
                        "result": f"Bad parameters: {str(te)}",
                        "meta": {"intent": intent, "params": params},
                    }
                )
            except Exception as e:
                logger.exception("Error executing method %s", method_name)
                self._log(f"Exception running {method_name}: {e}")
                results.append(
                    {
                        "method": method_name,
                        "result": f"Error: {str(e)}",
                        "meta": {"intent": intent, "params": params},
                    }
                )

        return results

    # -------------------------
    # Wrapper methods with validation and safe returns
    # -------------------------
    def price_change_since(self, date: str = None):
        if not date:
            raise ParamError(
                "price_change_since requires 'date' parameter (ISO or natural)."
            )
        try:
            return self._safe_tool_call(self.tool.analyze, include_changes=date)
        except TypeError:
            return self._safe_tool_call(self.tool.analyze, date)

    # Simple data helpers
    def has_stock(self, ticker: str):
        if not ticker:
            raise ParamError("ticker required for has_stock")
        return self._safe_tool_call(self.tool.has_stock, ticker)

    def quantity(self, ticker: str):
        if not ticker:
            raise ParamError("ticker required for quantity")
        return self._safe_tool_call(self.tool.get_quantity, ticker)

    def purchase_info(self, ticker: str):
        if not ticker:
            raise ParamError("ticker required for purchase_info")
        return self._safe_tool_call(self.tool.get_purchase_info, ticker)

    def purchase_timeline(self):
        return self._safe_tool_call(self.tool.get_purchase_timeline)

    # Portfolio views
    def top_holdings(self, n: int = 5):
        try:
            n_int = int(n)
            if n_int <= 0:
                raise ParamError("n must be positive")
        except Exception:
            raise ParamError("n must be an integer")
        return self._safe_tool_call(self.tool.top_holdings, n_int)

    def sector_allocation(self):
        return self._safe_tool_call(self.tool.get_sector_allocation)

    def filter_by_date(self, start_date: str = None, end_date: str = None):
        if not start_date or not end_date:
            raise ParamError(
                "filter_by_date requires 'start_date' and 'end_date' (ISO YYYY-MM-DD)."
            )
        for d in (start_date, end_date):
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                raise ParamError(f"Date '{d}' is not in YYYY-MM-DD format.")
        return self._safe_tool_call(
            self.tool.filter_by_purchase_date, start_date, end_date
        )

    def full_records(self):
        return self._safe_tool_call(self.tool.get_portfolio_summary)

    # Price-level analytics
    def live_prices(self):
        res = self._safe_tool_call(self.tool.fetch_prices)
        if isinstance(res, (list, tuple)) and res:
            return res[0]
        return res

    def historical_prices(self, date: str = None):
        if not date:
            raise ParamError("historical_prices requires 'date' parameter.")
        res = self._safe_tool_call(self.tool.fetch_historical_prices, date)
        if isinstance(res, (list, tuple)) and res:
            return res[0]
        return res

    def price_changes(self, date: str = None):
        if not date:
            raise ParamError("price_changes requires 'date' parameter.")
        return self._safe_tool_call(self.tool.get_price_changes, date)

    # Main analysis
    def analyze(self, include_changes: str = None):
        return self._safe_tool_call(self.tool.analyze, include_changes=include_changes)

    # -------------------------
    # Utility: safe tool caller
    # -------------------------
    def _safe_tool_call(self, func, *args, **kwargs):
        """
        Call a PortfolioTool function defensively:
          - Catch unexpected return types
          - Log exceptions
        """
        try:
            result = func(*args, **kwargs)
            # Normalize common container return types
            if isinstance(result, (list, tuple)):
                if len(result) == 1:
                    return result[0]
                return result
            return result
        except Exception as e:
            logger.exception(
                "PortfolioTool call failed for %s with args=%s kwargs=%s",
                getattr(func, "__name__", str(func)),
                args,
                kwargs,
            )
            raise
