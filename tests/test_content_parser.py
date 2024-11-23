# tests/test_content_parser.py
import pytest
from scratch_rag_application.content_parser.qa_parser import QAContentParser
from scratch_rag_application.content_parser.base_parser import ParsedContent


@pytest.fixture
def qa_config():
    """Fixture for QA parser configuration."""
    return {
        "content_parser": {
            "qa": {
                "patterns": [
                    {
                        "type": "explicit",
                        "question_pattern": "Q:|Question:",
                        "answer_pattern": "A:|Answer:"
                    },
                    {
                        "type": "header",
                        "header_pattern": "##\\s+.+\\?"
                    }
                ]
            }
        }
    }


@pytest.fixture
def sample_qa_content():
    """Fixture for sample Q&A content."""
    return """
    Q: How do the control plane and data plane communicate?
    A: The control plane and data plane communicate through secure channels. When a data plane node receives new configuration, it immediately loads that config into memory.

    Question: What happens if communication is interrupted?
    Answer: If communication is interrupted, data plane nodes can still continue proxying traffic to clients.
    
    ## How is configuration stored?
    The configuration is stored in an unencrypted cache file.
    """


class TestQAContentParser:
    def test_parse_explicit_qa(self, qa_config, sample_qa_content):
        """Test parsing of explicit Q&A format."""
        parser = QAContentParser(qa_config)
        results = parser.parse(sample_qa_content)

        assert len(results) == 3
        assert all(isinstance(r, ParsedContent) for r in results)
        assert all(r.content_type == "qa" for r in results)
        assert "control plane and data plane" in results[0].content

    def test_parse_empty_content(self, qa_config):
        """Test parsing empty content."""
        parser = QAContentParser(qa_config)
        results = parser.parse("")
        assert len(results) == 0

    def test_parse_invalid_content(self, qa_config):
        """Test parsing invalid content."""
        parser = QAContentParser(qa_config)
        results = parser.parse(None)
        assert len(results) == 0

    def test_malformed_qa_content(self, qa_config):
        """Test parsing malformed Q&A content."""
        malformed_content = """
        Q: Question without answer
        
        Answer: Answer without question
        """
        parser = QAContentParser(qa_config)
        results = parser.parse(malformed_content)
        assert len(results) == 0
