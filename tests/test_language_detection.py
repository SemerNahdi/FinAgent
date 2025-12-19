# tests/test_language_detection.py

import asyncio
import pytest
from services.language.detect import (
    LocalLanguageDetector,
    GroqLanguageDetector,
    LanguageDetectionService,
    get_language_service,
)


class TestLocalLanguageDetector:
    """Test local language detection without external API."""

    def setup_method(self):
        self.detector = LocalLanguageDetector()

    def test_english_detection(self):
        text = "What is the best stock to buy today?"
        result = self.detector.detect(text)
        assert result.language == "english"
        assert result.dialect is None
        assert result.confidence > 0.7
        assert not result.is_dialect

    def test_french_detection(self):
        text = "Quel est le meilleur action à acheter aujourd'hui?"
        result = self.detector.detect(text)
        assert result.language == "french"
        assert result.dialect is None
        assert result.confidence > 0.7
        assert not result.is_dialect

    def test_french_verlan_detection(self):
        text = "C'est chelou ce truc, tu peux m'expliquer?"
        result = self.detector.detect(text)
        assert result.language == "french"
        assert result.dialect == "verlan"
        assert result.confidence > 0.6
        assert result.is_dialect

    def test_tunisian_arabic_detection(self):
        text = "شنوة أحسن stock باش نشري اليوم؟"
        result = self.detector.detect(text)
        assert result.language == "arabic"
        assert result.dialect == "tunisian"
        assert result.confidence > 0.6
        assert result.is_dialect

    def test_moroccan_arabic_detection(self):
        text = "واش هذا مزيان؟ شنو هو السعر؟"
        result = self.detector.detect(text)
        assert result.language == "arabic"
        assert result.dialect == "moroccan"
        assert result.confidence > 0.6
        assert result.is_dialect

    def test_algerian_arabic_detection(self):
        text = "وين راني نلقى هذا؟ شحال السعر؟"
        result = self.detector.detect(text)
        assert result.language == "arabic"
        assert result.dialect == "algerian"
        assert result.confidence > 0.6
        assert result.is_dialect

    def test_unknown_language_low_confidence(self):
        text = "xyz abc def"
        result = self.detector.detect(text)
        assert result.confidence < 0.5


@pytest.mark.asyncio
class TestLanguageDetectionService:
    """Test full language detection service with fallback."""

    async def test_high_confidence_local_only(self):
        service = LanguageDetectionService()
        text = "What is the weather today?"
        result = await service.detect(text)

        assert result.language == "english"
        assert result.detected_by == "local"
        assert result.confidence >= 0.7

    async def test_response_language_mapping(self):
        service = LanguageDetectionService()

        # English -> English
        text = "Hello world"
        result = await service.detect(text)
        response_lang = service.get_response_language(result)
        assert response_lang == "english"

        # French -> French
        text = "Bonjour le monde"
        result = await service.detect(text)
        response_lang = service.get_response_language(result)
        assert response_lang == "french"

    async def test_dialect_response_instruction(self):
        service = LanguageDetectionService()

        # Tunisian Arabic
        text = "شنوة الوقت؟"
        result = await service.detect(text)
        instruction = service.format_response_instruction(result)

        assert "tunisian" in instruction.lower() or "french" in instruction.lower()
        assert result.is_dialect


# Manual test examples (run with: python -m pytest tests/test_language_detection.py -v -s)
async def manual_test_examples():
    """
    Manual testing with various inputs.
    Run this to see detection in action.
    """
    service = get_language_service()

    test_cases = [
        # English
        "What's the current stock price?",
        "Can you help me with my portfolio?",
        # French
        "Quel est le prix de l'action?",
        "Peux-tu m'aider avec mon portefeuille?",
        # French Verlan/Slang
        "C'est chelou ce truc",
        "T'es ouf ou quoi?",
        # Tunisian Arabic
        "شنوة أحسن stock؟",
        "برشا يزي هذا",
        # Moroccan Arabic
        "واش هذا مزيان؟",
        "بزاف ديال الفلوس",
        # Algerian Arabic
        "راني نحوس على stock",
        "شحال السعر؟",
    ]

    print("\n" + "=" * 80)
    print("LANGUAGE DETECTION TEST RESULTS")
    print("=" * 80 + "\n")

    for text in test_cases:
        result = await service.detect(text)
        response_lang = service.get_response_language(result)

        print(f"Input: {text}")
        print(f"  Language: {result.language}")
        print(f"  Dialect: {result.dialect or 'None'}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Is Dialect: {result.is_dialect}")
        print(f"  Detected By: {result.detected_by}")
        print(f"  Response Language: {response_lang}")
        print(f"  Instruction: {service.format_response_instruction(result)[:80]}...")
        print()


if __name__ == "__main__":
    # Run manual tests
    asyncio.run(manual_test_examples())
