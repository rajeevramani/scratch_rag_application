# tests/test_url_loader.py
import pytest
import aiohttp
import asyncio
from unittest.mock import Mock, patch
from scratch_rag_application.loader.url_loader import URLLoader
from langchain_core.documents import Document


@pytest.fixture
def sample_html():
    """Fixture for sample HTML content."""
    return """
    <div class="page-content">
        <h2>FAQ</h2>
        <p>Q: How do the control plane and data plane communicate?</p>
        <p>A: Through secure channels with immediate config loading.</p>
        
        <h2>Code Examples</h2>
        <pre><code>
        config = load_config()
        </code></pre>
    </div>
    """


@pytest.fixture
def mock_response(sample_html):
    """Fixture for mocked aiohttp response."""
    mock = Mock()
    mock.text = asyncio.coroutine(lambda: sample_html)
    mock.raise_for_status = Mock()
    return mock


@pytest.fixture
def mock_session(mock_response):
    """Fixture for mocked aiohttp ClientSession."""
    mock = Mock()
    mock.__aenter__ = asyncio.coroutine(lambda *_: mock)
    mock.__aexit__ = asyncio.coroutine(lambda *_: None)
    mock.get = asyncio.coroutine(lambda *_, **__: mock_response)
    return mock


class TestURLLoader:
    @pytest.mark.asyncio
    async def test_fetch_url(self, mock_session):
        """Test fetching and parsing a single URL."""
        loader = URLLoader()
        url = "https://docs.example.com/pagehttps://docs.konghq.com/konnect/network-resiliency/"

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value = mock_session
            documents = await loader._fetch_url(mock_session, url)

        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)
        assert any(doc.metadata['content_type'] == 'qa' for doc in documents)

    @pytest.mark.asyncio
    async def test_load_urls(self):
        """Test loading multiple URLs concurrently."""
        loader = URLLoader()

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value = mock_session
            documents = await loader.load_urls()

        assert len(documents) > 0

    def test_error_handling(self, mock_session):
        """Test error handling for failed requests."""
        loader = URLLoader()
        mock_session.get.side_effect = aiohttp.ClientError()

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value = mock_session
            documents = loader.load()

        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_content_parsing(self, mock_session):
        """Test content type identification and parsing."""
        loader = URLLoader()
        url = "https://docs.example.com/page"

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value = mock_session
            documents = await loader._fetch_url(mock_session, url)

        # Verify content types
        content_types = [doc.metadata['content_type'] for doc in documents]
        assert 'qa' in content_types
        assert 'general' in content_types
