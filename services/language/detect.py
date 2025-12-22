import os
from dataclasses import dataclass
from typing import Optional
from langdetect import detect, DetectorFactory
import json

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
            return LanguageDetectionResult(language="unknown", confidence=0.3)

        mapping = {"en": "english", "fr": "french", "ar": "arabic"}
        language = mapping.get(lang, "unknown")
        confidence = 0.8 if language != "unknown" else 0.4

        return LanguageDetectionResult(language=language, confidence=confidence)


# =========================
# Groq detector (lazy)
# =========================


class GroqLanguageDetector:
    """
    LLM-based detector using GroqLLM.
    Lazily instantiated and handles async safely.
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

        try:
            # Use the correct async call
            result = await self.llm.call_async(prompt, max_tokens=150)

            # Normalize response
            if isinstance(result, dict):
                response = result
            elif isinstance(result, str):
                try:
                    response = json.loads(result)
                except json.JSONDecodeError:
                    response = {
                        "language": "unknown",
                        "dialect": None,
                        "confidence": 0.6,
                    }
            else:
                response = {"language": "unknown", "dialect": None, "confidence": 0.6}

            language = response.get("language", "unknown")
            dialect = response.get("dialect")
            confidence = float(response.get("confidence", 0.6))

            return LanguageDetectionResult(
                language=language,
                dialect=dialect,
                confidence=confidence,
                detected_by="groq",
            )

        except Exception:
            return LanguageDetectionResult(
                language="unknown",
                dialect=None,
                confidence=0.5,
                detected_by="groq-fallback",
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
        self._llm = None  # lazy init for translation

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

    async def translate_to_english(self, text: str, source_language: str) -> str:
        """
        Translate text to English if it's not already in English.
        Returns the original text if already in English or translation fails.
        """
        # Normalize language name
        source_lang_lower = source_language.lower()
        
        # Check if already English
        if source_lang_lower in ["english", "en"]:
            return text
        
        # Initialize LLM if needed
        if self._llm is None:
            from services.tools.groq_wrapper import GroqLLM
            self._llm = GroqLLM()
        
        # Translate using Groq LLM
        translation_prompt = (
            f"Translate the following text from {source_language} to English. "
            f"Provide only the translation, no explanations or additional text.\n\n"
            f"Text: {text}"
        )
        
        try:
            translated = await self._llm.call_async(
                prompt=translation_prompt,
                model="llama-3.3-70b-versatile",
                max_tokens=512
            )
            
            # Extract translation from response
            if isinstance(translated, dict):
                translated_text = translated.get("content") or translated.get("text") or str(translated)
            elif isinstance(translated, str):
                translated_text = translated.strip()
            else:
                translated_text = str(translated).strip()
            
            # Return translation if valid, otherwise return original
            if translated_text and len(translated_text) > 0:
                return translated_text
            else:
                return text
                
        except Exception as e:
            # On error, return original text
            print(f"Translation error: {e}")
            return text


# =========================
# Factory
# =========================


def get_language_service() -> LanguageDetectionService:
    return LanguageDetectionService(enable_groq=True)
