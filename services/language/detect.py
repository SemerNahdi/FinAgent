import os
from dataclasses import dataclass
from typing import Optional

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42


# =========================
# Result model
# =========================


@dataclass
class LanguageDetectionResult:
    language: str
    confidence: float
    dialect: Optional[str] = None
    detected_by: str = "local"

    @property
    def is_dialect(self) -> bool:
        return self.dialect is not None


# =========================
# Local detector
# =========================


class LocalLanguageDetector:
    """
    Rule-based + langdetect detector.
    Fast, offline, deterministic.
    """

    AR_DIALECTS = {
        "tunisian": ["شنوة", "برشا", "يزي", "باش", "نلقى"],
        "moroccan": ["واش", "بزاف", "شنو", "ديال"],
        "algerian": ["وين", "راني", "نحوس", "شحال"],
    }

    FR_VERLAN = ["chelou", "ouf", "wesh", "frero"]

    def detect(self, text: str) -> LanguageDetectionResult:
        text_lower = text.lower()

        # Arabic dialect detection
        for dialect, markers in self.AR_DIALECTS.items():
            if any(m in text_lower for m in markers):
                return LanguageDetectionResult(
                    language="arabic",
                    dialect=dialect,
                    confidence=0.75,
                )

        # French slang / verlan
        if any(m in text_lower for m in self.FR_VERLAN):
            return LanguageDetectionResult(
                language="french",
                dialect="verlan",
                confidence=0.7,
            )

        # Standard language detection
        try:
            lang = detect(text)
        except Exception:
            return LanguageDetectionResult(
                language="unknown",
                confidence=0.3,
            )

        mapping = {
            "en": "english",
            "fr": "french",
            "ar": "arabic",
        }

        language = mapping.get(lang, "unknown")
        confidence = 0.8 if language != "unknown" else 0.4

        return LanguageDetectionResult(
            language=language,
            confidence=confidence,
        )


# =========================
# Groq detector (lazy)
# =========================


class GroqLanguageDetector:
    """
    LLM-based detector.
    Only instantiated if explicitly enabled and needed.
    """

    def __init__(self):
        from services.tools.groq_wrapper import GroqLLM

        self.llm = GroqLLM()

    async def detect(self, text: str) -> LanguageDetectionResult:
        prompt = (
            "Detect the language and dialect of the following text. "
            "Respond in JSON with keys: language, dialect, confidence.\n\n"
            f"Text: {text}"
        )

        response = await self.llm.complete(prompt)

        # Minimal defensive parsing
        language = response.get("language", "unknown")
        dialect = response.get("dialect")
        confidence = float(response.get("confidence", 0.6))

        return LanguageDetectionResult(
            language=language,
            dialect=dialect,
            confidence=confidence,
            detected_by="groq",
        )


# =========================
# Detection service
# =========================


class LanguageDetectionService:
    """
    Orchestrates detection with fallback.
    """

    def __init__(self, enable_groq: bool = True):
        self.local_detector = LocalLanguageDetector()
        self.enable_groq = enable_groq
        self.groq_detector = None  # lazy init

    async def detect(self, text: str) -> LanguageDetectionResult:
        local_result = self.local_detector.detect(text)

        # High confidence → stop
        if local_result.confidence >= 0.7:
            local_result.detected_by = "local"
            return local_result

        # Low confidence → optional Groq fallback
        if self.enable_groq:
            if self.groq_detector is None:
                self.groq_detector = GroqLanguageDetector()

            groq_result = await self.groq_detector.detect(text)
            return groq_result

        # Fallback disabled
        local_result.detected_by = "local"
        return local_result

    def get_response_language(self, result: LanguageDetectionResult) -> str:
        return result.language


# =========================
# Factory
# =========================


import os
from dataclasses import dataclass
from typing import Optional

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42


# =========================
# Result model
# =========================


@dataclass
class LanguageDetectionResult:
    language: str
    confidence: float
    dialect: Optional[str] = None
    detected_by: str = "local"

    @property
    def is_dialect(self) -> bool:
        return self.dialect is not None


# =========================
# Local detector
# =========================


class LocalLanguageDetector:
    """
    Rule-based + langdetect detector.
    Fast, offline, deterministic.
    """

    AR_DIALECTS = {
        "tunisian": ["شنوة", "برشا", "يزي", "باش", "نلقى"],
        "moroccan": ["واش", "بزاف", "شنو", "ديال"],
        "algerian": ["وين", "راني", "نحوس", "شحال"],
    }

    FR_VERLAN = ["chelou", "ouf", "wesh", "frero"]

    def detect(self, text: str) -> LanguageDetectionResult:
        text_lower = text.lower()

        # Arabic dialect detection
        for dialect, markers in self.AR_DIALECTS.items():
            if any(m in text_lower for m in markers):
                return LanguageDetectionResult(
                    language="arabic",
                    dialect=dialect,
                    confidence=0.75,
                )

        # French slang / verlan
        if any(m in text_lower for m in self.FR_VERLAN):
            return LanguageDetectionResult(
                language="french",
                dialect="verlan",
                confidence=0.7,
            )

        # Standard language detection
        try:
            lang = detect(text)
        except Exception:
            return LanguageDetectionResult(
                language="unknown",
                confidence=0.3,
            )

        mapping = {
            "en": "english",
            "fr": "french",
            "ar": "arabic",
        }

        language = mapping.get(lang, "unknown")
        confidence = 0.8 if language != "unknown" else 0.4

        return LanguageDetectionResult(
            language=language,
            confidence=confidence,
        )


# =========================
# Groq detector (lazy)
# =========================


class GroqLanguageDetector:
    """
    LLM-based detector.
    Only instantiated if explicitly enabled and needed.
    """

    def __init__(self):
        from services.tools.groq_wrapper import GroqLLM

        self.llm = GroqLLM()

    async def detect(self, text: str) -> LanguageDetectionResult:
        prompt = (
            "Detect the language and dialect of the following text. "
            "Respond in JSON with keys: language, dialect, confidence.\n\n"
            f"Text: {text}"
        )

        response = await self.llm.complete(prompt)

        # Minimal defensive parsing
        language = response.get("language", "unknown")
        dialect = response.get("dialect")
        confidence = float(response.get("confidence", 0.6))

        return LanguageDetectionResult(
            language=language,
            dialect=dialect,
            confidence=confidence,
            detected_by="groq",
        )


# =========================
# Detection service
# =========================


class LanguageDetectionService:
    """
    Orchestrates detection with fallback.
    """

    def __init__(self, enable_groq: bool = True):
        self.local_detector = LocalLanguageDetector()
        self.enable_groq = enable_groq
        self.groq_detector = None  # lazy init

    async def detect(self, text: str) -> LanguageDetectionResult:
        local_result = self.local_detector.detect(text)

        # High confidence → stop
        if local_result.confidence >= 0.7:
            local_result.detected_by = "local"
            return local_result

        # Low confidence → optional Groq fallback
        if self.enable_groq:
            if self.groq_detector is None:
                self.groq_detector = GroqLanguageDetector()

            groq_result = await self.groq_detector.detect(text)
            return groq_result

        # Fallback disabled
        local_result.detected_by = "local"
        return local_result

    def get_response_language(self, result: LanguageDetectionResult) -> str:
        return result.language


# =========================
# Factory
# =========================
def get_language_service() -> LanguageDetectionService:
    return LanguageDetectionService(enable_groq=True)
