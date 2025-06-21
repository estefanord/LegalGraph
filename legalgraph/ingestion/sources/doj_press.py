
import re
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from dateutil import parser as date_parser
from bs4 import BeautifulSoup

from ..base import WebScrapingSource
from ..models import Document, DOJPressDocument, DocumentType
from ...core.config import settings
from ...core.exceptions import ParseError


class DOJPressSource(WebScrapingSource):
    
    def __init__(self):
        super().__init__("doj_press", settings.doj_press_base_url)
        
        # DOJ page selectors (from original definitions.py)
        self.selectors = {
            'main_page_last': {'title': 'Go to last page'},
            'main_page_release': {'class': 'views-field views-field-title'},
            'page_text': {'class': 'field field--name-field-pr-body field--type-text-long field--label-hidden'},
            'page_title': {'id': 'node-title'},
            'page_date': {'class': 'date-display-single'},
            'page_topic_list': {'class': 'field field--name-field-pr-topic field--type-taxonomy-term-reference field--label-above'},
            'page_topic': {'class': 'field__item'},
            'page_component_list': {'class': 'field field--name-field-pr-component field--type-taxonomy-term-reference field--label-above'},
            'page_id_container': {'class': 'field field--name-field-pr-number field--type-text field--label-above'},
            'page_id': {'class': 'field__item'}
        }
        
        # Regex patterns for content analysis
        self.enforcement_patterns = {
            'criminal': re.compile(r'criminal|indictment|guilty|plea|sentenced|prison', re.IGNORECASE),
            'civil': re.compile(r'civil|lawsuit|settlement|consent decree', re.IGNORECASE),
            'administrative': re.compile(r'administrative|regulatory|compliance', re.IGNORECASE)
        }
        
        self.penalty_pattern = re.compile(r'\$([0-9,]+(?:\.[0-9]{2})?)\s*(?:million|thousand|billion)?', re.IGNORECASE)
        
        self.target_patterns = [
            re.compile(r'(?:individual|person|defendant)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
            re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc\.|LLC|Corp\.|Company|Corporation)', re.IGNORECASE),
            re.compile(r'charged\s+([^,\.]+)', re.IGNORECASE)
        ]
    
    def fetch_recent(self, days: int = 30, limit: Optional[int] = None) -> List[Document]:
        documents = []
        
        with self:
            try:
                # Get page links first
                links = self._get_page_links(max_pages=5)  # Limit for recent documents
                self.logger.info(f"Found {len(links)} press release links")
                
                # Filter by date and process
                cutoff_date = datetime.now() - timedelta(days=days)
                
                for i, link in enumerate(links):
                    try:
                        # Check limit
                        if limit and len(documents) >= limit:
                            self.logger.info(f"Reached limit of {limit} documents")
                            break
                        
                        # Skip speeches as in original scraper
                        if '/speech/' in link:
                            self.logger.debug(f"Skipping speech: {link}")
                            continue
                        
                        document = self._process_press_release(link)
                        if document and self.validate_document(document):
                            # Check date filter
                            if document.publish_date and datetime.combine(document.publish_date, datetime.min.time()) >= cutoff_date:
                                documents.append(document)
                            elif not document.publish_date:  # Include if date unknown
                                documents.append(document)
                        
                        self._rate_limit()
                        
                        # Progress logging
                        if i % 10 == 0:
                            self.logger.info(f"Processed {i}/{len(links)} press releases")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process press release {link}: {e}")
                        continue
                
            except Exception as e:
                self.logger.error(f"Failed to fetch DOJ press releases: {e}")
                raise
        
        self.logger.info(f"Successfully processed {len(documents)} DOJ press releases")
        return documents
    
    async def fetch_recent_async(self, days: int = 30) -> List[Document]:
        return self.fetch_recent(days)
    
    def fetch_by_component(self, component: str, days: int = 30) -> List[Document]:
        documents = self.fetch_recent(days)
        
        # Filter by component
        filtered_docs = []
        for doc in documents:
            if isinstance(doc, DOJPressDocument) and doc.component:
                if component.lower() in doc.component.lower():
                    filtered_docs.append(doc)
        
        return filtered_docs
    
    def _get_page_links(self, start_page: int = 0, max_pages: Optional[int] = None) -> List[str]:
        
        # Get first page to determine total pages
        first_page_url = self._format_page_url(start_page)
        response = self._get_with_retry(first_page_url)
        soup = self._parse_html(response.text, first_page_url)
        
        # Updated: Use modern DOJ selectors
        # The current DOJ site uses article elements instead of the old views-field classes
        articles = soup.find_all('article')
        self.logger.info(f"Found {len(articles)} article elements on page")
        
        links = []
        for article in articles:
            link_elem = article.find('a', href=True)
            if link_elem:
                href = link_elem['href']
                # Filter for press releases (contains /opa/pr/)
                if '/opa/pr/' in href:
                    if href.startswith('/'):
                        href = self.base_url + href
                    links.append(href)
                    self.logger.debug(f"Found press release: {href}")
        
        self.logger.info(f"Found {len(links)} press release links on page {start_page}")
        
        # For now, just process the first page since pagination structure has changed
        # We can add pagination later if needed
        if max_pages and max_pages > 1:
            self.logger.info("Pagination not yet implemented for new DOJ structure - processing first page only")
        
        return links
    
    def _format_page_url(self, page_num: int) -> str:
        if page_num == 0:
            return f"{self.base_url}/news"
        else:
            # DOJ pagination structure may have changed - using basic approach
            return f"{self.base_url}/news?page={page_num}"
    
    def _process_press_release(self, url: str) -> Optional[DOJPressDocument]:
        try:
            response = self._get_with_retry(url)
            soup = self._parse_html(response.text, url)
            
            # Extract title - use working selector
            title = ""
            title_elem = soup.find('h1')  # Simple h1 works
            if title_elem:
                title = title_elem.get_text(strip=True)
            
            # Extract date - use working selectors
            pub_date = None
            
            # Try time element first (has datetime attribute)
            time_elem = soup.find('time')
            if time_elem and time_elem.get('datetime'):
                pub_date = self._parse_date(time_elem['datetime'])
            
            # Try meta tag as fallback
            if not pub_date:
                meta_elem = soup.find('meta', {'property': 'article:published_time'})
                if meta_elem and meta_elem.get('content'):
                    pub_date = self._parse_date(meta_elem['content'])
            
            # Extract main content
            content = self._extract_content(soup)
            
            # Extract topics
            topics = self._extract_topics(soup)
            
            # Extract components (agencies/departments)
            components = self._extract_components(soup)
            component = components[0] if components else None
            
            # Extract press release ID
            source_id = self._extract_press_release_id(soup)
            
            # Analyze content for enforcement type and other metadata
            enforcement_type = self._classify_enforcement_type(content)
            financial_penalty = self._extract_financial_penalties(content)
            targets = self._extract_targets(content)
            
            # Create document
            document = DOJPressDocument(
                id="",  # Will be auto-generated
                source_url=url,
                document_type=DocumentType.DOJ_PRESS_RELEASE,
                title=title,
                content=content,
                publish_date=pub_date,
                source_id=source_id,
                agencies=['DOJ'] + components,
                topics=topics,
                
                # DOJ-specific fields
                component=component,
                enforcement_type=enforcement_type,
                financial_penalty=financial_penalty,
                targets=targets
            )
            
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to process DOJ press release {url}: {e}")
            return None
    
    def _extract_topics(self, soup: BeautifulSoup) -> List[str]:
        topics = []
        
        topic_list = soup.find('div', self.selectors['page_topic_list'])
        if topic_list:
            topic_items = topic_list.find_all('div', self.selectors['page_topic'])
            topics = [item.get_text(strip=True) for item in topic_items]
        
        return topics
    
    def _extract_components(self, soup: BeautifulSoup) -> List[str]:
        components = []
        
        component_list = soup.find('div', self.selectors['page_component_list'])
        if component_list:
            component_links = component_list.find_all('a')
            components = [link.get_text(strip=True) for link in component_links]
        
        return components
    
    def _extract_press_release_id(self, soup: BeautifulSoup) -> Optional[str]:
        id_container = soup.find('div', self.selectors['page_id_container'])
        if id_container:
            id_elem = id_container.find('div', self.selectors['page_id'])
            if id_elem:
                return id_elem.get_text(strip=True)
        return None
    
    def _classify_enforcement_type(self, content: str) -> Optional[str]:
        content_lower = content.lower()
        
        for enforcement_type, pattern in self.enforcement_patterns.items():
            if pattern.search(content_lower):
                return enforcement_type
        
        return None
    
    def _extract_financial_penalties(self, content: str) -> Optional[float]:
        penalties = []
        
        matches = self.penalty_pattern.findall(content)
        for match in matches:
            try:
                amount_str = match.replace(',', '')
                amount = float(amount_str)
                
                # Check context for scale
                context = content[content.find(match):content.find(match) + 50].lower()
                if 'billion' in context:
                    amount *= 1_000_000_000
                elif 'million' in context:
                    amount *= 1_000_000
                elif 'thousand' in context:
                    amount *= 1_000
                
                penalties.append(amount)
            except ValueError:
                continue
        
        return max(penalties) if penalties else None
    
    def _extract_targets(self, content: str) -> List[str]:
        targets = set()
        
        for pattern in self.target_patterns:
            matches = pattern.findall(content)
            for match in matches:
                target = match.strip().rstrip(',.')
                if len(target) > 2 and len(target) < 100:
                    targets.add(target)
        
        return list(targets)
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        try:
            parsed_date = date_parser.parse(date_str)
            return parsed_date.date()
        except Exception as e:
            self.logger.warning(f"Failed to parse date '{date_str}': {e}")
            return None
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        # Updated selectors for current DOJ website structure
        content_selectors = [
            # Try article within main first (most likely)
            lambda s: s.find('main', {}).find('article') if s.find('main') else None,
            # Try main content directly
            lambda s: s.find('main'),
            # Fallback to substantial paragraphs
            lambda s: self._extract_paragraph_content(s),
            # Legacy selectors as final fallback
            lambda s: s.find('div', {'class': 'field field--name-field-pr-body field--type-text-long field--label-hidden'}),
            lambda s: s.find('div', {'class': 'field--name-body'}),
            lambda s: s.find('div', {'class': 'content'}),
        ]
        
        for selector_func in content_selectors:
            try:
                element = selector_func(soup)
                if element:
                    content = element.get_text(strip=True)
                    # Ensure we have substantial content
                    if len(content) > 200:
                        return content
            except Exception as e:
                self.logger.debug(f"Content selector failed: {e}")
                continue
        
        return ""
    
    def _extract_paragraph_content(self, soup: BeautifulSoup) -> str:
        paragraphs = soup.find_all('p')
        substantial_paragraphs = [
            p.get_text(strip=True) 
            for p in paragraphs 
            if len(p.get_text(strip=True)) > 50
        ]
        
        if substantial_paragraphs:
            # Skip the first few paragraphs which are usually navigation/header
            content_paragraphs = substantial_paragraphs[3:] if len(substantial_paragraphs) > 5 else substantial_paragraphs
            return '\n\n'.join(content_paragraphs)
        
        return "" 