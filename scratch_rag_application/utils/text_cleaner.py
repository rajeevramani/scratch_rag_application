import re
import logging
from typing import Optional


class TextCleaner:
    """Clean text by removing excessive whitespace and normalizing line breaks."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def clean(self, text: str) -> Optional[str]:
        """
        Clean the input text by removing excessive whitespace and normalizing line breaks.

        Args:
            text (str): Input text to clean

        Returns:
            Optional[str]: Cleaned text, or None if cleaning fails
        """
        if not text:
            self.logger.warning("Received empty text for cleaning")
            return text

        try:
            original_length = len(text)

            self.logger.info(f"Length before cleaning: {original_length}")

            # Perform cleaning steps
            cleaned = text

            # First collapse all whitespace including newlines to single spaces
            cleaned = re.sub(r'\s+', ' ', cleaned)
            self.logger.info(f"After collapsing whitespace: {len(cleaned)}")

            # Restore meaningful line breaks
            # Add newline before headers
            cleaned = re.sub(r'\s(#+ )', r'\n\1', cleaned)
            self.logger.info(f"After restoring headers: {len(cleaned)}")

            # Add newline after periods that end sentences
            cleaned = re.sub(r'\. ([A-Z])', r'.\n\1', cleaned)
            self.logger.info(
                f"After restoring sentence breaks: {len(cleaned)}")

            # Ensure only single newlines
            cleaned = re.sub(r'\n+', '\n', cleaned)
            self.logger.info(f"After normalizing newlines: {len(cleaned)}")

            # Trim leading/trailing whitespace
            cleaned = cleaned.strip()

            final_length = len(cleaned)
            chars_removed = original_length - final_length

            self.logger.info(f"Length after cleaning: {final_length}")

            if chars_removed > 0:
                self.logger.debug(
                    f"Cleaned text: removed {chars_removed} characters "
                    f"({original_length} → {final_length})"
                )

            return cleaned

        except Exception as e:
            self.logger.error(f"Error cleaning text: {str(e)}")
            return text  # Return original text if cleaning fails
