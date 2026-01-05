"""Translation module using deep-translator library.

This module provides text translation functionality with:
- Multiple translation engine support
- Automatic language detection
- Batch translation optimization
- Error handling and retry logic
"""

import logging
import time
from typing import List, Optional

from deep_translator import GoogleTranslator
from deep_translator.exceptions import (
    LanguageNotSupportedException,
    TranslationNotFound,
    NotValidPayload
)

logger = logging.getLogger(__name__)


class TextTranslator:
    """Handles text translation with robust error handling."""

    def __init__(
        self,
        source_lang: str = 'auto',
        target_lang: str = 'en',
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize the translator.

        Args:
            source_lang: Source language code (use 'auto' for auto-detection).
            target_lang: Target language code.
            max_retries: Maximum number of retry attempts.
            retry_delay: Delay between retries in seconds.

        Raises:
            LanguageNotSupportedException: If language codes are invalid.
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        try:
            # Initialize translator
            self.translator = GoogleTranslator(
                source=source_lang,
                target=target_lang
            )
            logger.info(
                f"Translator initialized: {source_lang} -> {target_lang}"
            )

            # Validate languages
            self._validate_languages()

        except LanguageNotSupportedException as e:
            logger.error(f"Invalid language configuration: {e}")
            raise

    def _validate_languages(self) -> None:
        """Validate that source and target languages are supported.

        Raises:
            LanguageNotSupportedException: If languages are not supported.
        """
        supported_langs = GoogleTranslator().get_supported_languages(
            as_dict=True
        )

        if (self.target_lang not in supported_langs.values() and
                self.target_lang not in supported_langs.keys()):
            raise LanguageNotSupportedException(
                f"Target language '{self.target_lang}' not supported"
            )

        logger.info("Language validation successful")

    def translate(self, text: str) -> str:
        """Translate a single text string.

        Args:
            text: Text to translate.

        Returns:
            Translated text, or original text if translation fails.
        """
        if not text or not text.strip():
            return text

        # Clean text
        text = text.strip()

        for attempt in range(self.max_retries):
            try:
                translated = self.translator.translate(text)

                if translated:
                    logger.debug(f"Translated: '{text[:30]}...' -> '{translated[:30]}...'")
                    return translated
                else:
                    logger.warning(f"Empty translation result for: {text[:50]}")
                    return text

            except NotValidPayload as e:
                logger.warning(f"Invalid payload: {e}")
                return text

            except TranslationNotFound as e:
                logger.warning(f"Translation not found: {e}")
                return text

            except Exception as e:
                logger.error(
                    f"Translation attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"Max retries reached, returning original text"
                    )
                    return text

        return text

    def translate_batch(
        self,
        texts: List[str],
        batch_delay: float = 0.5
    ) -> List[str]:
        """Translate multiple text strings.

        Args:
            texts: List of texts to translate.
            batch_delay: Delay between translations to avoid rate limiting.

        Returns:
            List of translated texts.
        """
        if not texts:
            return []

        logger.info(f"Translating batch of {len(texts)} texts")
        translated_texts = []

        for i, text in enumerate(texts):
            translated = self.translate(text)
            translated_texts.append(translated)

            # Add delay to avoid rate limiting (except for last item)
            if i < len(texts) - 1:
                time.sleep(batch_delay)

        logger.info(f"Batch translation completed: {len(translated_texts)} texts")
        return translated_texts

    def detect_language(self, text: str) -> Optional[str]:
        """Detect the language of the input text.

        Args:
            text: Text to analyze.

        Returns:
            ISO language code, or None if detection fails.
        """
        try:
            # Use a temporary translator for detection
            detector = GoogleTranslator(source='auto', target='en')
            # Translate to trigger detection
            detector.translate(text)
            detected_lang = detector.source

            logger.info(f"Detected language: {detected_lang}")
            return detected_lang

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return None

    def get_supported_languages(self) -> dict:
        """Get dictionary of supported languages.

        Returns:
            Dictionary mapping language codes to names.
        """
        try:
            langs = GoogleTranslator().get_supported_languages(as_dict=True)
            logger.info(f"Retrieved {len(langs)} supported languages")
            return langs
        except Exception as e:
            logger.error(f"Failed to retrieve supported languages: {e}")
            return {}

    def change_target_language(self, new_target: str) -> bool:
        """Change the target language for translation.

        Args:
            new_target: New target language code.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.translator = GoogleTranslator(
                source=self.source_lang,
                target=new_target
            )
            self.target_lang = new_target
            logger.info(f"Target language changed to: {new_target}")
            return True

        except LanguageNotSupportedException as e:
            logger.error(f"Invalid target language: {e}")
            return False

    def split_long_text(
        self,
        text: str,
        max_length: int = 5000
    ) -> List[str]:
        """Split long text into chunks for translation.

        Google Translate has a character limit, so we split long texts.

        Args:
            text: Text to split.
            max_length: Maximum characters per chunk.

        Returns:
            List of text chunks.
        """
        if len(text) <= max_length:
            return [text]

        chunks = []
        sentences = text.split('. ')
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= max_length:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '

        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def translate_long_text(self, text: str) -> str:
        """Translate text that may exceed API limits.

        Args:
            text: Text to translate (any length).

        Returns:
            Translated text.
        """
        chunks = self.split_long_text(text)

        if len(chunks) == 1:
            return self.translate(text)

        logger.info(f"Translating long text in {len(chunks)} chunks")
        translated_chunks = self.translate_batch(chunks)

        return ' '.join(translated_chunks)
