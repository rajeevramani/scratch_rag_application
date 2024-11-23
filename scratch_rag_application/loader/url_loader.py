# loader/url_loader.py
from typing import List
import aiohttp
import asyncio
import logging
from bs4 import BeautifulSoup, SoupStrainer
from langchain_core.documents import Document
from langchain_community.document_transformers import MarkdownifyTransformer
from scratch_rag_application.config.config_handler import ConfigHandler
from scratch_rag_application.utils.text_cleaner import TextCleaner
from scratch_rag_application.content_parser.parser_factory import ContentParserFactory
import re


class URLLoader:
    """Asynchronous loader for processing URLs and converting content to markdown."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_handler = ConfigHandler("config.yaml")
        self.transformer = MarkdownifyTransformer()
        self.text_cleaner = TextCleaner()
        self.parser_factory = ContentParserFactory(self.config_handler)

        # Initialize parsers
        self.parsers = {
            parser_type: self.parser_factory.create_parser(parser_type)
            # Add "table", "code" as we implement them
            for parser_type in ["qa"]
        }

        # Get configuration
        self.urls = self.config_handler.get(
            "pipeline.sources.website.urls", [])
        self.content_class = self.config_handler.get(
            "pipeline.sources.website.content_class", "page-content")

    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> List[Document]:
        """
        Fetch and process a single URL.

        Args:
            session: aiohttp client session
            url: URL to fetch

        Returns:
            List of Document objects containing parsed content sections
        """
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                html = await response.text()

                # Parse only the content we need
                soup = BeautifulSoup(
                    html,
                    'html.parser',
                    parse_only=SoupStrainer(class_=self.content_class)
                )

                # Convert to markdown and clean
                markdown_text = str(soup)
                transformed_docs = self.transformer.transform_documents(
                    [Document(page_content=markdown_text,
                              metadata={'source': url})]
                )

                if not transformed_docs:
                    return []

                # Clean the transformed text
                cleaned_text = self.text_cleaner.clean(
                    transformed_docs[0].page_content)

                # Parse content using all configured parsers
                documents = []
                remaining_content = cleaned_text

                for parser_type, parser in self.parsers.items():
                    parsed_sections = parser.parse(remaining_content)

                    # Create documents for parsed sections
                    for section in parsed_sections:
                        documents.append(
                            Document(
                                page_content=section.content,
                                metadata={
                                    'source': url,
                                    'content_type': section.content_type,
                                    'section_id': section.section_id
                                }
                            )
                        )

                        # Remove parsed content from remaining text
                        remaining_content = remaining_content.replace(
                            section.content, '')

                # Create document for any remaining content
                if remaining_content.strip():
                    documents.append(
                        Document(
                            page_content=remaining_content.strip(),
                            metadata={
                                'source': url,
                                'content_type': 'general',
                                'section_id': 'default'
                            }
                        )
                    )

                return documents

        except Exception as e:
            self.logger.error(f"Error processing {url}: {str(e)}")
            return []

    async def load_urls(self) -> List[Document]:
        """Load and process URLs concurrently."""
        if not self.urls:
            self.logger.warning("No URLs configured")
            return []

        self.logger.info(f"Processing {len(self.urls)} URLs")

        async with aiohttp.ClientSession() as session:
            # Create tasks for all URLs
            tasks = [self._fetch_url(session, url) for url in self.urls]
            all_documents = await asyncio.gather(*tasks, return_exceptions=False)

            # Flatten the list of document lists
            documents = [doc for docs in all_documents for doc in docs if doc]

            self.logger.info(
                f"Successfully processed {len(documents)} document sections")
            return documents

    def load(self) -> List[Document]:
        """Synchronous wrapper for async load_urls method."""
        return asyncio.run(self.load_urls())
