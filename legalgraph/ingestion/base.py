
import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import aiohttp
import requests
from bs4 import BeautifulSoup

from ..core.config import settings
from ..core.exceptions import ScrapingError, ParseError, RateLimitError
from ..core.logging import get_logger
from .models import Document, IngestionResult


class BaseSource(ABC):
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"legalgraph.ingestion.{name}")
        self.session: Optional[requests.Session] = None
        self.async_session: Optional[aiohttp.ClientSession] = None
        
    def __enter__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': settings.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    async def __aenter__(self):
        self.async_session = aiohttp.ClientSession(
            headers={'User-Agent': settings.user_agent},
            timeout=aiohttp.ClientTimeout(total=settings.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.async_session:
            await self.async_session.close()
    
    def _get_with_retry(self, url: str, **kwargs) -> requests.Response:
        # Don't retry certain error codes that indicate access is blocked
        non_retryable_codes = [403, 404, 401, 410]
        
        for attempt in range(settings.max_retries):
            try:
                response = self.session.get(url, timeout=settings.timeout, **kwargs)
                
                if response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', 60))
                    self.logger.info(f"Rate limited by {url}, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                if response.status_code in non_retryable_codes:
                    # Don't retry these - likely access denied or not found
                    self.logger.warning(f"Access denied or not found for {url} (HTTP {response.status_code})")
                    raise ScrapingError(f"Access denied to {url} (HTTP {response.status_code})", 
                                      url=url, status_code=response.status_code)
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if hasattr(e, 'response') and e.response and e.response.status_code in non_retryable_codes:
                    # Don't retry access denied errors
                    self.logger.warning(f"Access denied to {url}: {e}")
                    raise ScrapingError(f"Access denied to {url}", url=url) from e
                
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{settings.max_retries}): {e}",
                    extra={"url": url, "attempt": attempt + 1}
                )
                
                if attempt == settings.max_retries - 1:
                    raise ScrapingError(f"Failed to fetch {url} after {settings.max_retries} attempts", 
                                      url=url) from e
                
                # Much longer delay between retries
                wait_time = settings.retry_delay * (attempt + 1)
                self.logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        raise ScrapingError(f"Max retries exceeded for {url}", url=url)
    
    async def _get_async_with_retry(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        for attempt in range(settings.max_retries):
            try:
                async with self.async_session.get(url, **kwargs) as response:
                    if response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        raise RateLimitError(f"Rate limited by {url}", retry_after=retry_after)
                    
                    response.raise_for_status()
                    return response
                    
            except aiohttp.ClientError as e:
                self.logger.warning(
                    f"Async request failed (attempt {attempt + 1}/{settings.max_retries}): {e}",
                    extra={"url": url, "attempt": attempt + 1}
                )
                
                if attempt == settings.max_retries - 1:
                    raise ScrapingError(f"Failed to fetch {url} after {settings.max_retries} attempts", 
                                      url=url) from e
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        raise ScrapingError(f"Max retries exceeded for {url}", url=url)
    
    def _rate_limit(self):
        time.sleep(settings.request_delay)
    
    async def _rate_limit_async(self):
        await asyncio.sleep(settings.request_delay)
    
    @abstractmethod
    def fetch_recent(self, days: int = 30) -> List[Document]:
        pass
    
    @abstractmethod
    async def fetch_recent_async(self, days: int = 30) -> List[Document]:
        pass
    
    def validate_document(self, doc: Document) -> bool:
        if not doc.title or not doc.content:
            return False
        
        if len(doc.content) > settings.max_document_size:
            return False
        
        return True


class WebScrapingSource(BaseSource):
    
    def __init__(self, name: str, base_url: str):
        super().__init__(name)
        self.base_url = base_url
    
    def _parse_html(self, html_content: str, url: str = "") -> BeautifulSoup:
        try:
            return BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            raise ParseError(f"Failed to parse HTML from {url}", 
                           document_type="html", source=self.name) from e
    
    def _extract_text(self, soup: BeautifulSoup, selector: Dict[str, Any]) -> Optional[str]:
        try:
            element = soup.find(**selector)
            return element.get_text(strip=True) if element else None
        except Exception as e:
            self.logger.warning(f"Failed to extract text with selector {selector}: {e}")
            return None
    
    def _extract_all_text(self, soup: BeautifulSoup, selector: Dict[str, Any]) -> List[str]:
        try:
            elements = soup.find_all(**selector)
            return [elem.get_text(strip=True) for elem in elements if elem.get_text(strip=True)]
        except Exception as e:
            self.logger.warning(f"Failed to extract all text with selector {selector}: {e}")
            return []
    
    def _extract_links(self, soup: BeautifulSoup, selector: Dict[str, Any]) -> List[str]:
        try:
            elements = soup.find_all('a', **selector)
            links = []
            for elem in elements:
                href = elem.get('href')
                if href:
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        href = self.base_url + href
                    elif not href.startswith('http'):
                        href = f"{self.base_url}/{href}"
                    links.append(href)
            return links
        except Exception as e:
            self.logger.warning(f"Failed to extract links with selector {selector}: {e}")
            return []


class RSSSource(BaseSource):
    
    def __init__(self, name: str, rss_url: str):
        super().__init__(name)
        self.rss_url = rss_url
    
    def _parse_rss_feed(self, xml_content: str) -> BeautifulSoup:
        try:
            return BeautifulSoup(xml_content, 'xml')
        except Exception as e:
            raise ParseError(f"Failed to parse RSS feed from {self.rss_url}", 
                           document_type="xml", source=self.name) from e
    
    def _extract_rss_items(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        items = []
        try:
            for item in soup.find_all('item'):
                item_data = {
                    'title': item.find('title').get_text(strip=True) if item.find('title') else "",
                    'link': item.find('link').get_text(strip=True) if item.find('link') else "",
                    'description': item.find('description').get_text(strip=True) if item.find('description') else "",
                    'pub_date': item.find('pubDate').get_text(strip=True) if item.find('pubDate') else "",
                    'guid': item.find('guid').get_text(strip=True) if item.find('guid') else ""
                }
                items.append(item_data)
        except Exception as e:
            self.logger.warning(f"Failed to extract RSS items: {e}")
        
        return items 