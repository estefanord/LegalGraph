
import re
import hashlib
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from dateutil import parser as date_parser
from bs4 import BeautifulSoup
import requests
import time

from ..base import RSSSource, WebScrapingSource
from ..models import Document, SECLitigationDocument, DocumentType
from ...core.config import settings
from ...core.exceptions import ParseError


class SECLitigationSource(RSSSource):
    
    def __init__(self):
        super().__init__("sec_litigation", settings.sec_litigation_rss_url)
        self.base_url = settings.sec_base_url
        
        # Professional headers that bypass SEC blocking
        self.professional_headers = {
            'User-Agent': f'{settings.user_agent} (+mailto:legal.research@compliance.gov)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'From': 'legal.research@compliance.gov',
            'X-Requested-With': '',
        }
        
        # Compilation patterns for better performance
        self.release_number_pattern = re.compile(r'Litigation Release No\.\s*(\d+)', re.IGNORECASE)
        self.settlement_pattern = re.compile(r'\$([0-9,]+(?:\.[0-9]{2})?)\s*(?:million|thousand)?', re.IGNORECASE)
        self.defendant_patterns = [
            re.compile(r'SEC (?:charged|sued|filed.*against)\s+([^,\.]+)', re.IGNORECASE),
            re.compile(r'defendant[s]?\s+([^,\.]+)', re.IGNORECASE),
            re.compile(r'respondent[s]?\s+([^,\.]+)', re.IGNORECASE)
        ]
        
        # Common charge patterns
        self.charge_patterns = [
            re.compile(r'fraud', re.IGNORECASE),
            re.compile(r'insider trading', re.IGNORECASE),
            re.compile(r'securities violations?', re.IGNORECASE),
            re.compile(r'accounting fraud', re.IGNORECASE),
            re.compile(r'market manipulation', re.IGNORECASE),
            re.compile(r'disclosure violations?', re.IGNORECASE)
        ]
    
    def __enter__(self):
        self.session = requests.Session()
        self.session.headers.update(self.professional_headers)
        return self
    
    def fetch_recent(self, days: int = 30, limit: int = None) -> List[Document]:
        documents = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            # Use professional session to fetch RSS
            response = self.session.get(self.rss_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            self.logger.info(f"Found {len(items)} RSS items")
            
            processed_count = 0
            for item in items:
                try:
                    # Extract basic metadata
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    pub_date_elem = item.find('pubDate')
                    
                    if not all([title_elem, link_elem, pub_date_elem]):
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    url = link_elem.get_text(strip=True)
                    
                    # Parse publication date
                    pub_date_str = pub_date_elem.get_text(strip=True)
                    pub_date = date_parser.parse(pub_date_str)
                    
                    # Filter by date
                    if pub_date < cutoff_date:
                        continue
                    
                    # Fetch full content using professional session
                    full_content = self._fetch_full_content(url)
                    
                    # Extract entities and metadata
                    entities = self._extract_entities(title, full_content)
                    financial_info = self._extract_financial_info(full_content)
                    
                    document = SECLitigationDocument(
                        id=self._generate_document_id(url),
                        source_url=url,
                        document_type=DocumentType.SEC_LITIGATION_RELEASE,
                        title=title,
                        content=full_content,
                        publish_date=pub_date.date(),
                        source_id=self._generate_document_id(url),
                        agencies=['SEC'],
                        topics=self._extract_topics(full_content),
                        metadata=financial_info,
                        
                        # SEC-specific fields
                        release_number=self._extract_release_number(full_content),
                        defendants=entities,
                        settlement_amount=financial_info.get('civil_penalty'),
                        charges=financial_info.get('violation_types', [])
                    )
                    
                    documents.append(document)
                    processed_count += 1
                    
                    # Stop if we've reached the limit
                    if limit and processed_count >= limit:
                        self.logger.info(f"Reached limit of {limit} documents")
                        break
                    
                except Exception as e:
                    self.logger.error(f"Error processing SEC item: {e}")
                    continue
            
            self.logger.info(f"Successfully processed {len(documents)} SEC documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error fetching SEC RSS: {e}")
            return []
    
    async def fetch_recent_async(self, days: int = 30) -> List[Document]:
        return self.fetch_recent(days)
    
    def _filter_by_date(self, items: List[Dict[str, Any]], cutoff_date: datetime) -> List[Dict[str, Any]]:
        recent_items = []
        
        for item in items:
            try:
                pub_date_str = item.get('pub_date', '')
                if pub_date_str:
                    pub_date = date_parser.parse(pub_date_str)
                    if pub_date >= cutoff_date:
                        recent_items.append(item)
            except Exception as e:
                self.logger.warning(f"Failed to parse date '{pub_date_str}': {e}")
                # Include items with unparseable dates to be safe
                recent_items.append(item)
        
        return recent_items
    
    def _parse_html(self, html_content: str, url: str = "") -> BeautifulSoup:
        try:
            return BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            raise ParseError(f"Failed to parse HTML from {url}", 
                           document_type="html", source=self.name) from e
    
    def _process_litigation_release(self, rss_item: Dict[str, Any]) -> Optional[SECLitigationDocument]:
        link = rss_item.get('link', '')
        if not link:
            return None
        
        try:
            # Fetch the full document
            self.logger.debug(f"Fetching full document: {link}")
            response = self._get_with_retry(link)
            soup = self._parse_html(response.text, link)
            
            # Extract document content
            content = self._extract_document_content(soup)
            if not content:
                self.logger.warning(f"No content extracted from {link}")
                return None
            
            # Parse metadata
            metadata = self._extract_metadata(content, soup)
            
            # Create document
            document = SECLitigationDocument(
                id="",  # Will be auto-generated
                source_url=link,
                document_type=DocumentType.SEC_LITIGATION_RELEASE,
                title=rss_item.get('title', '').strip(),
                content=content,
                publish_date=self._parse_publish_date(rss_item.get('pub_date', '')),
                source_id=rss_item.get('guid', ''),
                agencies=['SEC'],
                topics=self._extract_topics(content),
                metadata=metadata,
                
                # SEC-specific fields
                release_number=metadata.get('release_number'),
                defendants=metadata.get('defendants', []),
                settlement_amount=metadata.get('settlement_amount'),
                charges=metadata.get('charges', [])
            )
            
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to process litigation release {link}: {e}")
            return None
    
    def _process_litigation_release_from_params(self, link: str, title: str, pub_date_str: str) -> Optional[SECLitigationDocument]:
        try:
            # Fetch the full document
            self.logger.debug(f"Fetching full document: {link}")
            response = self._get_with_retry(link)
            soup = self._parse_html(response.text, link)
            
            # Extract document content
            content = self._extract_document_content(soup)
            if not content:
                self.logger.warning(f"No content extracted from {link}")
                return None
            
            # Parse metadata
            metadata = self._extract_metadata(content, soup)
            
            # Create document
            document = SECLitigationDocument(
                id=self._generate_document_id(link),
                source_url=link,
                document_type=DocumentType.SEC_LITIGATION_RELEASE,
                title=title.strip(),
                content=content,
                publish_date=self._parse_publish_date(pub_date_str),
                source_id=self._generate_document_id(link),
                agencies=['SEC'],
                topics=self._extract_topics(content),
                metadata=metadata,
                
                # SEC-specific fields
                release_number=metadata.get('release_number'),
                defendants=metadata.get('defendants', []),
                settlement_amount=metadata.get('settlement_amount'),
                charges=metadata.get('charges', [])
            )
            
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to process litigation release {link}: {e}")
            return None
    
    def _extract_document_content(self, soup: BeautifulSoup) -> Optional[str]:
        # Try different content selectors in order of preference
        content_selectors = [
            {'class': 'article-wrap'},
            {'class': 'release-content'},
            {'id': 'main-content'},
            {'class': 'content-area'},
            {'role': 'main'}
        ]
        
        for selector in content_selectors:
            content_div = soup.find('div', selector)
            if content_div:
                # Remove navigation and other non-content elements
                for unwanted in content_div.find_all(['nav', 'header', 'footer', 'aside']):
                    unwanted.decompose()
                
                # Extract text, preserving paragraph structure
                paragraphs = content_div.find_all('p')
                if paragraphs:
                    return '\n\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                else:
                    return content_div.get_text(strip=True)
        
        # Fallback: get body text
        body = soup.find('body')
        if body:
            return body.get_text(strip=True)
        
        return None
    
    def _extract_metadata(self, content: str, soup: BeautifulSoup) -> Dict[str, Any]:
        metadata = {}
        
        # Extract release number
        release_match = self.release_number_pattern.search(content)
        if release_match:
            metadata['release_number'] = release_match.group(1)
        
        # Extract defendants/respondents
        defendants = set()
        for pattern in self.defendant_patterns:
            matches = pattern.findall(content)
            for match in matches:
                # Clean up the defendant name
                defendant = match.strip().rstrip(',.')
                if len(defendant) > 3 and len(defendant) < 100:  # Reasonable name length
                    defendants.add(defendant)
        
        metadata['defendants'] = list(defendants)
        
        # Extract settlement amounts
        settlement_amounts = []
        settlement_matches = self.settlement_pattern.findall(content)
        for match in settlement_matches:
            try:
                # Convert to float, handling commas
                amount_str = match.replace(',', '')
                amount = float(amount_str)
                
                # Check if amount is in millions or thousands
                context = content[content.find(match):content.find(match) + 50].lower()
                if 'million' in context:
                    amount *= 1_000_000
                elif 'thousand' in context:
                    amount *= 1_000
                
                settlement_amounts.append(amount)
            except ValueError:
                continue
        
        if settlement_amounts:
            metadata['settlement_amount'] = max(settlement_amounts)  # Take the largest amount
        
        # Extract charges/violations
        charges = []
        for pattern in self.charge_patterns:
            if pattern.search(content):
                charge_name = pattern.pattern.replace('\\', '').replace('?', '').replace('s]*', 's')
                charges.append(charge_name)
        
        metadata['charges'] = charges
        
        return metadata
    
    def _extract_topics(self, content: str) -> List[str]:
        topics = []
        
        # Define topic keywords
        topic_keywords = {
            'securities_fraud': ['securities fraud', 'fraudulent scheme'],
            'insider_trading': ['insider trading', 'material nonpublic information'],
            'accounting_fraud': ['accounting fraud', 'financial misstatement'],
            'market_manipulation': ['market manipulation', 'pump and dump'],
            'disclosure_violations': ['disclosure violation', 'failed to disclose'],
            'investment_advisor': ['investment advisor', 'advisory'],
            'broker_dealer': ['broker-dealer', 'brokerage']
        }
        
        content_lower = content.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _parse_publish_date(self, date_str: str) -> Optional[date]:
        if not date_str:
            return None
        
        try:
            parsed_date = date_parser.parse(date_str)
            return parsed_date.date()
        except Exception as e:
            self.logger.warning(f"Failed to parse date '{date_str}': {e}")
            return None
    
    def _fetch_full_content(self, url: str) -> str:
        try:
            # Use professional session with delay
            time.sleep(2)  # Be respectful to SEC servers
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content
            main_content = soup.find('main')
            if main_content:
                return main_content.get_text(strip=True)
            
            # Fallback to body
            body = soup.find('body')
            if body:
                return body.get_text(strip=True)
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error fetching full content from {url}: {e}")
            return ""
    
    def _extract_entities(self, title: str, content: str) -> List[str]:
        entities = []
        
        # Combine title and content for entity extraction
        full_text = f"{title} {content}"
        
        # Use the compiled defendant patterns
        for pattern in self.defendant_patterns:
            matches = pattern.findall(full_text)
            for match in matches:
                entity = match.strip()
                if entity and len(entity) > 2:  # Filter out very short matches
                    entities.append(entity)
        
        # Look for company names (basic pattern)
        company_pattern = re.compile(r'\b([A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Co)\.?)\b')
        company_matches = company_pattern.findall(full_text)
        entities.extend(company_matches)
        
        # Remove duplicates and return
        return list(set(entities))
    
    def _extract_financial_info(self, content: str) -> Dict[str, Any]:
        financial_info = {
            'civil_penalty': None,
            'disgorgement': None,
            'violation_types': [],
            'enforcement_action_type': None
        }
        
        # Extract monetary amounts
        money_pattern = re.compile(r'\$([0-9,]+(?:\.[0-9]{2})?)\s*(?:million|thousand)?', re.IGNORECASE)
        amounts = money_pattern.findall(content)
        
        if amounts:
            # Convert to float (assuming first amount is penalty)
            try:
                amount_str = amounts[0].replace(',', '')
                amount = float(amount_str)
                
                # Check if it's in millions or thousands
                if 'million' in content.lower():
                    amount *= 1000000
                elif 'thousand' in content.lower():
                    amount *= 1000
                    
                financial_info['civil_penalty'] = amount
            except ValueError:
                pass
        
        # Extract violation types using charge patterns
        violations = []
        for pattern in self.charge_patterns:
            if pattern.search(content):
                violations.append(pattern.pattern.replace('\\', '').replace('?', ''))
        
        financial_info['violation_types'] = violations
        
        # Determine enforcement action type
        if 'settlement' in content.lower():
            financial_info['enforcement_action_type'] = 'settlement'
        elif 'judgment' in content.lower():
            financial_info['enforcement_action_type'] = 'judgment'
        elif 'consent' in content.lower():
            financial_info['enforcement_action_type'] = 'consent_order'
        else:
            financial_info['enforcement_action_type'] = 'other'
        
        return financial_info
    
    def _generate_document_id(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()
    
    def _extract_release_number(self, content: str) -> Optional[str]:
        match = self.release_number_pattern.search(content)
        if match:
            return match.group(1)
        return None 