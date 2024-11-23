# content_parser/qa_parser.py
import re
import uuid
from typing import List, Pattern
from .base_parser import BaseContentParser, ParsedContent


class QAContentParser(BaseContentParser):
    """Parser for identifying and extracting Q&A content."""

    def __init__(self, params: dict):
        super().__init__(params)
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[Pattern]:
        """Compile regex patterns from configuration."""
        try:
            qa_config = self.params.get("content_parser", {}).get("qa", {})
            patterns = qa_config.get("patterns", [])

            compiled_patterns = []
            for pattern in patterns:
                if pattern["type"] == "explicit":
                    # Pattern for explicit Q&A format
                    pattern_str = f"(?:{pattern['question_pattern']})\\s*(.+?)\\s*(?:{
                        pattern['answer_pattern']})\\s*(.+?)(?=(?:{pattern['question_pattern']})|$)"
                    compiled_patterns.append(re.compile(
                        pattern_str, re.DOTALL | re.IGNORECASE))
                elif pattern["type"] == "header":
                    # Pattern for header-based Q&A format
                    pattern_str = f"({
                        pattern['header_pattern']})\\s*(.+?)(?=(?:{pattern['header_pattern']})|$)"
                    compiled_patterns.append(
                        re.compile(pattern_str, re.DOTALL))

            return compiled_patterns

        except Exception as e:
            self.logger.error(f"Error compiling patterns: {str(e)}")
            return []

    def parse(self, content: str) -> List[ParsedContent]:
        """
        Parse content to extract Q&A sections.

        Args:
            content: Content to parse

        Returns:
            List of ParsedContent objects containing Q&A sections
        """
        if not self._validate_content(content):
            return []

        try:
            parsed_sections = []

            for pattern in self.patterns:
                matches = pattern.finditer(content)
                for match in matches:
                    question = match.group(1).strip()
                    answer = match.group(2).strip()

                    if question and answer:
                        parsed_sections.append(
                            ParsedContent(
                                content=f"Q: {question}\nA: {answer}",
                                content_type="qa",
                                section_id=str(uuid.uuid4())
                            )
                        )

            return parsed_sections

        except Exception as e:
            self.logger.error(f"Error parsing content: {str(e)}")
            return []
