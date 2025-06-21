
import re
import json
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from dateutil import parser as date_parser
import hashlib
from bs4 import BeautifulSoup
import requests

from ..base import WebScrapingSource
from ..models import Document, CFPBEnforcementDocument, DocumentType
from ...core.config import settings
from ...core.exceptions import ParseError, ScrapingError


class CFPBEnforcementSource(WebScrapingSource):
    
    def __init__(self):
        super().__init__("cfpb_enforcement", settings.cfpb_enforcement_url)
        self.enforcement_url = settings.cfpb_enforcement_url
        self.base_url = "https://www.consumerfinance.gov"  # Separate base URL without path
        
        # Regex patterns for parsing enforcement actions
        self.penalty_patterns = {
            'civil_penalty': re.compile(r'civil money penalty.*?\$([0-9,]+(?:\.[0-9]{2})?)', re.IGNORECASE),
            'consumer_redress': re.compile(r'consumer (?:redress|relief|restitution).*?\$([0-9,]+(?:\.[0-9]{2})?)', re.IGNORECASE),
            'total_penalty': re.compile(r'total.*?\$([0-9,]+(?:\.[0-9]{2})?)', re.IGNORECASE)
        }
        
        self.action_type_patterns = {
            'consent_order': re.compile(r'consent order|administrative order', re.IGNORECASE),
            'federal_court_action': re.compile(r'federal court|district court|lawsuit', re.IGNORECASE),
            'administrative_action': re.compile(r'administrative action|cease and desist', re.IGNORECASE)
        }
        
        self.violation_patterns = {
            'unfair_practices': re.compile(r'unfair.*practices?', re.IGNORECASE),
            'deceptive_practices': re.compile(r'deceptive.*practices?', re.IGNORECASE),
            'abusive_practices': re.compile(r'abusive.*practices?', re.IGNORECASE),
            'discrimination': re.compile(r'discrimination|fair lending', re.IGNORECASE),
            'privacy_violations': re.compile(r'privacy|data protection|gramm.leach.bliley', re.IGNORECASE),
            'debt_collection': re.compile(r'debt collection|fdcpa', re.IGNORECASE),
            'mortgage_violations': re.compile(r'mortgage|tila|respa|qualified mortgage', re.IGNORECASE),
            'credit_reporting': re.compile(r'credit report|fcra|background check', re.IGNORECASE)
        }
        
        # Industry classification patterns
        self.industry_patterns = {
            'banking': re.compile(r'bank|credit union|financial institution', re.IGNORECASE),
            'credit_cards': re.compile(r'credit card|payment card', re.IGNORECASE),
            'debt_collection': re.compile(r'debt collector|collection agency', re.IGNORECASE),
            'mortgage': re.compile(r'mortgage|home loan|foreclosure', re.IGNORECASE),
            'payday_lending': re.compile(r'payday loan|short.term lending', re.IGNORECASE),
            'student_loans': re.compile(r'student loan|education financing', re.IGNORECASE),
            'credit_reporting': re.compile(r'credit reporting|background screening', re.IGNORECASE),
            'auto_lending': re.compile(r'auto loan|vehicle financing', re.IGNORECASE)
        }
    
    def fetch_recent(self, days: int = 30) -> List[Document]:
        documents = []
        
        try:
            with self:
                self.logger.info(f"Fetching CFPB enforcement actions from {self.enforcement_url}")
                
                # Get the main enforcement actions page
                response = self._get_with_retry(self.enforcement_url)
                soup = self._parse_html(response.text, self.enforcement_url)
                
                # Extract enforcement action articles
                articles = soup.find_all('article', class_='o-post-preview')
                self.logger.info(f"Found {len(articles)} enforcement action articles")
                
                cutoff_date = date.today() - timedelta(days=days)
                
                for article in articles:
                    try:
                        doc = self._extract_enforcement_action(article)
                        if doc and doc.publish_date and doc.publish_date >= cutoff_date:
                            documents.append(doc)
                            self.logger.debug(f"Extracted enforcement action: {doc.title}")
                        elif doc:
                            self.logger.debug(f"Skipping old action: {doc.title} ({doc.publish_date})")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract enforcement action: {e}")
                        continue
                    
                    # Rate limiting
                    self._rate_limit()
                    
                    # Respect processing limits
                    if len(documents) >= settings.max_documents_per_source:
                        break
                
                self.logger.info(f"Successfully extracted {len(documents)} recent CFPB enforcement actions")
                return documents
                
        except Exception as e:
            self.logger.error(f"Failed to fetch CFPB enforcement actions: {e}")
            raise ScrapingError(f"CFPB enforcement actions fetch failed: {e}") from e
    
    async def fetch_recent_async(self, days: int = 30) -> List[Document]:
        return self.fetch_recent(days)
    
    def _scrape_enforcement_actions_page(self) -> List[Dict[str, Any]]:
        try:
            response = self._get_with_retry(self.enforcement_url)
            soup = self._parse_html(response.text, self.enforcement_url)
            
            actions = []
            
            # Look for enforcement action entries
            # CFPB typically displays actions in a table or list format
            action_selectors = [
                {'class': 'enforcement-action'},
                {'class': 'action-item'},
                {'class': 'content-l_col-1-2'}  # Common CFPB layout class
            ]
            
            action_elements = []
            for selector in action_selectors:
                elements = soup.find_all('div', selector)
                if elements:
                    action_elements = elements
                    break
            
            # If no specific action containers found, look for article/section tags
            if not action_elements:
                action_elements = soup.find_all(['article', 'section'])
            
            for element in action_elements:
                try:
                    action_data = self._extract_action_from_element(element)
                    if action_data:
                        actions.append(action_data)
                except Exception as e:
                    self.logger.debug(f"Failed to extract action from element: {e}")
                    continue
            
            # If still no actions found, try looking for links to individual action pages
            if not actions:
                self.logger.info("No actions found in main content, looking for action links")
                actions = self._find_action_links(soup)
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Failed to scrape CFPB enforcement actions page: {e}")
            return []
    
    def _extract_action_from_element(self, element) -> Optional[Dict[str, Any]]:
        action_data = {}
        
        # Try to find title/heading
        title_selectors = ['h1', 'h2', 'h3', 'h4']
        title = None
        for selector in title_selectors:
            title_elem = element.find(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                break
        
        if not title:
            # Try to find title in links
            link_elem = element.find('a')
            if link_elem:
                title = link_elem.get_text(strip=True)
        
        if not title:
            return None
        
        action_data['title'] = title
        
        # Extract link to full action
        link_elem = element.find('a')
        if link_elem and link_elem.get('href'):
            href = link_elem['href']
            if href.startswith('/'):
                href = self.base_url + href
            action_data['url'] = href
        else:
            action_data['url'] = self.enforcement_url  # Fallback
        
        # Extract date if visible
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\w+ \d{1,2}, \d{4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        
        element_text = element.get_text()
        for pattern in date_patterns:
            match = re.search(pattern, element_text)
            if match:
                action_data['date_str'] = match.group(1)
                break
        
        # Extract summary text
        paragraphs = element.find_all('p')
        if paragraphs:
            summary_text = ' '.join(p.get_text(strip=True) for p in paragraphs[:2])  # First 2 paragraphs
            action_data['summary'] = summary_text[:500]  # Limit length
        else:
            action_data['summary'] = element.get_text(strip=True)[:500]
        
        return action_data
    
    def _find_action_links(self, soup) -> List[Dict[str, Any]]:
        actions = []
        
        # Look for links that might be enforcement actions
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link['href']
            link_text = link.get_text(strip=True)
            
            # Filter for enforcement-related links
            if any(keyword in href.lower() for keyword in ['enforcement', 'action', 'consent-order']):
                if len(link_text) > 10:  # Meaningful link text
                    if href.startswith('/'):
                        href = self.base_url + href
                    
                    actions.append({
                        'title': link_text,
                        'url': href,
                        'summary': link_text
                    })
        
        return actions[:20]  # Limit to prevent overwhelming
    
    def _filter_by_date(self, actions: List[Dict[str, Any]], cutoff_date: datetime) -> List[Dict[str, Any]]:
        recent_actions = []
        
        for action in actions:
            date_str = action.get('date_str', '')
            if date_str:
                try:
                    action_date = date_parser.parse(date_str)
                    if action_date >= cutoff_date:
                        recent_actions.append(action)
                except Exception as e:
                    self.logger.debug(f"Failed to parse date '{date_str}': {e}")
                    # Include actions with unparseable dates
                    recent_actions.append(action)
            else:
                # Include actions without explicit dates
                recent_actions.append(action)
        
        return recent_actions
    
    def _process_enforcement_action(self, action_data: Dict[str, Any]) -> Optional[CFPBEnforcementDocument]:
        try:
            url = action_data.get('url', '')
            if not url:
                return None
            
            # If we only have summary data, use it; otherwise fetch full document
            if url == self.enforcement_url:
                # Summary only
                content = action_data.get('summary', '')
                soup = None
            else:
                # Fetch full document
                response = self._get_with_retry(url)
                soup = self._parse_html(response.text, url)
                content = self._extract_full_content(soup)
            
            if not content:
                self.logger.warning(f"No content extracted from {url}")
                return None
            
            # Parse metadata
            metadata = self._parse_enforcement_metadata(content)
            
            # Determine publish date
            pub_date = None
            if action_data.get('date_str'):
                pub_date = self._parse_date(action_data['date_str'])
            
            # Create document
            document = CFPBEnforcementDocument(
                id="",  # Will be auto-generated
                source_url=url,
                document_type=DocumentType.CFPB_ENFORCEMENT_ACTION,
                title=action_data.get('title', ''),
                content=content,
                publish_date=pub_date,
                agencies=['CFPB'],
                topics=self._extract_topics(content),
                metadata=metadata,
                
                # CFPB-specific fields
                action_type=metadata.get('action_type'),
                consumer_redress=metadata.get('consumer_redress'),
                civil_penalty=metadata.get('civil_penalty'),
                industry=metadata.get('industry'),
                violation_types=metadata.get('violation_types', [])
            )
            
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to process CFPB enforcement action: {e}")
            return None
    
    def _process_enforcement_action_from_url(self, url: str) -> Optional[CFPBEnforcementDocument]:
        try:
            # Fetch full document
            response = self._get_with_retry(url)
            soup = self._parse_html(response.text, url)
            content = self._extract_full_content(soup)
            
            if not content:
                self.logger.warning(f"No content extracted from {url}")
                return None
            
            # Parse metadata
            metadata = self._parse_enforcement_metadata(content)
            
            # Extract title from page
            title = ""
            title_elem = soup.find(['h1', 'h2', 'title'])
            if title_elem:
                title = title_elem.get_text(strip=True)
            
            # Try to extract date from content
            import re
            date_patterns = [
                r'Date filed:\s*([A-Z]{3} \d{1,2}, \d{4})',
                r'On ([A-Za-z]+ \d{1,2}, \d{4})',
                r'(\d{1,2}/\d{1,2}/\d{4})'
            ]
            
            publish_date = None
            for pattern in date_patterns:
                match = re.search(pattern, content)
                if match:
                    try:
                        date_str = match.group(1)
                        if '/' in date_str:
                            publish_date = datetime.strptime(date_str, '%m/%d/%Y').date()
                        else:
                            publish_date = date_parser.parse(date_str).date()
                        break
                    except Exception as e:
                        continue
            
            # If no date found, use today
            if not publish_date:
                publish_date = date.today()
            
            # Generate document ID
            doc_id = f"cfpb_{hashlib.sha256(url.encode()).hexdigest()[:12]}"
            
            # Create document
            document = CFPBEnforcementDocument(
                id=doc_id,
                source_url=url,
                document_type=DocumentType.CFPB_ENFORCEMENT_ACTION,
                title=title,
                content=content,
                publish_date=publish_date,
                agencies=['CFPB'],
                topics=self._extract_topics(content),
                metadata=metadata,
                
                # CFPB-specific fields
                action_type=metadata.get('action_type'),
                consumer_redress=metadata.get('consumer_redress'),
                civil_penalty=metadata.get('civil_penalty'),
                industry=metadata.get('industry'),
                violation_types=metadata.get('violation_types', [])
            )
            
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to process CFPB enforcement action from URL {url}: {e}")
            return None
    
    def _extract_full_content(self, soup) -> Optional[str]:
        # Common CFPB content selectors
        content_selectors = [
            {'class': 'content-l'},
            {'class': 'content-main'},
            {'role': 'main'},
            {'class': 'page-content'},
            {'id': 'main'}
        ]
        
        for selector in content_selectors:
            content_div = soup.find('div', selector)
            if content_div:
                # Remove navigation and sidebar elements
                for unwanted in content_div.find_all(['nav', 'aside', 'footer', 'header']):
                    unwanted.decompose()
                
                # Extract paragraphs
                paragraphs = content_div.find_all('p')
                if paragraphs:
                    return '\n\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                else:
                    return content_div.get_text(strip=True)
        
        # Fallback to body
        body = soup.find('body')
        if body:
            return body.get_text(strip=True)
        
        return None
    
    def _parse_enforcement_metadata(self, content: str) -> Dict[str, Any]:
        metadata = {}
        
        # Extract penalty amounts
        for penalty_type, pattern in self.penalty_patterns.items():
            matches = pattern.findall(content)
            if matches:
                try:
                    # Take the largest amount found
                    amounts = [float(match.replace(',', '')) for match in matches]
                    metadata[penalty_type] = max(amounts)
                except ValueError:
                    continue
        
        # Determine action type
        for action_type, pattern in self.action_type_patterns.items():
            if pattern.search(content):
                metadata['action_type'] = action_type
                break
        
        # Extract violation types
        violations = []
        for violation_type, pattern in self.violation_patterns.items():
            if pattern.search(content):
                violations.append(violation_type)
        metadata['violation_types'] = violations
        
        # Determine industry
        for industry, pattern in self.industry_patterns.items():
            if pattern.search(content):
                metadata['industry'] = industry
                break
        
        return metadata
    
    def _extract_topics(self, content: str) -> List[str]:
        topics = []
        
        # Map violation types to topics
        content_lower = content.lower()
        
        topic_mapping = {
            'consumer_protection': ['consumer', 'protection', 'unfair', 'deceptive'],
            'financial_services': ['financial', 'banking', 'credit'],
            'mortgage_lending': ['mortgage', 'home loan', 'foreclosure'],
            'debt_collection': ['debt collection', 'collector'],
            'credit_reporting': ['credit report', 'background check'],
            'fair_lending': ['discrimination', 'fair lending'],
            'privacy': ['privacy', 'data protection']
        }
        
        for topic, keywords in topic_mapping.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        try:
            parsed_date = date_parser.parse(date_str)
            return parsed_date.date()
        except Exception as e:
            self.logger.warning(f"Failed to parse date '{date_str}': {e}")
            return None
    
    def _extract_enforcement_action(self, article) -> Optional[CFPBEnforcementDocument]:
        try:
            # Extract title from heading
            title_elem = article.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if not title_elem:
                return None
            
            title = title_elem.get_text(strip=True)
            
            # Extract link
            link_elem = article.find('a', href=True)
            if not link_elem:
                return None
            
            href = link_elem.get('href')
            if href.startswith('/'):
                source_url = 'https://www.consumerfinance.gov' + href
            else:
                source_url = href
            
            # Extract content and metadata
            content = article.get_text(strip=True)
            
            # Extract date from content
            import re
            date_patterns = [
                r'Date filed:\s*([A-Z]{3} \d{1,2}, \d{4})',
                r'On ([A-Za-z]+ \d{1,2}, \d{4})',
                r'(\d{1,2}/\d{1,2}/\d{4})'
            ]
            
            publish_date = None
            for pattern in date_patterns:
                match = re.search(pattern, content)
                if match:
                    try:
                        date_str = match.group(1)
                        # Handle different date formats
                        if '/' in date_str:
                            publish_date = datetime.strptime(date_str, '%m/%d/%Y').date()
                        else:
                            publish_date = date_parser.parse(date_str).date()
                        break
                    except Exception as e:
                        self.logger.debug(f"Failed to parse date '{date_str}': {e}")
                        continue
            
            # If we can't find a date, use today (better than failing)
            if not publish_date:
                publish_date = date.today()
                self.logger.warning(f"No date found for {title}, using today's date")
            
            # Extract financial penalties
            penalty_amount = self._extract_penalty_amount(content)
            
            # Extract target entity (company name from title)
            entity_name = title.strip()
            
            # Extract violation types from content
            violations = self._extract_violations(content)
            
            # Extract industry classification
            industry = self._classify_industry(content)
            
            # Generate document ID
            doc_id = f"cfpb_{hashlib.sha256(source_url.encode()).hexdigest()[:12]}"
            
            return CFPBEnforcementDocument(
                id=doc_id,
                title=title,
                content=content,
                source_url=source_url,
                document_type=DocumentType.CFPB_ENFORCEMENT_ACTION,
                publish_date=publish_date,
                # Use correct field names from the model
                civil_penalty=penalty_amount,  # Changed from penalty_amount
                violation_types=violations,    # Changed from violations
                industry=industry,
                action_type=self._classify_action_type(content)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract enforcement action from article: {e}")
            return None
    
    def _extract_penalty_amount(self, content: str) -> Optional[float]:
        penalty_amount = None
        
        for penalty_type, pattern in self.penalty_patterns.items():
            matches = pattern.findall(content)
            if matches:
                try:
                    # Take the largest amount found
                    amounts = [float(match.replace(',', '')) for match in matches]
                    penalty_amount = max(amounts)
                    break
                except ValueError:
                    continue
        
        return penalty_amount
    
    def _extract_violations(self, content: str) -> List[str]:
        violations = []
        
        for violation_type, pattern in self.violation_patterns.items():
            if pattern.search(content):
                violations.append(violation_type)
        
        return violations
    
    def _classify_industry(self, content: str) -> Optional[str]:
        industry = None
        
        for industry_type, pattern in self.industry_patterns.items():
            if pattern.search(content):
                industry = industry_type
                break
        
        return industry
    
    def _classify_action_type(self, content: str) -> Optional[str]:
        action_type = None
        
        for action_type_pattern, pattern in self.action_type_patterns.items():
            if pattern.search(content):
                action_type = action_type_pattern
                break
        
        return action_type 