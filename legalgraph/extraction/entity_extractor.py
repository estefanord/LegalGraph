import re
import logging
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from .models import Entity, EntityType
from ..ingestion.models import Document
from ..core.config import settings
from ..core.logging import get_logger


class HybridEntityExtractor:
    
    def __init__(self):
        self.logger = get_logger("entity_extractor")
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            try:
                self.spacy_nlp = spacy.load("en_core_web_sm")
                self.logger.info("spaCy model loaded successfully")
            except OSError:
                self.spacy_nlp = None
                self.logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Initialize OpenAI client (modern v1.x way)
        if settings.openai_api_key:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
            self.logger.info("OpenAI API enabled for entity extraction")
        else:
            self.openai_client = None
            self.logger.warning("OpenAI API not configured")
        
        # Enhanced regex patterns
        self.regex_patterns = {
            'companies': [
                re.compile(r'\b[A-Z][a-zA-Z\s&,.-]{2,50}(?:Inc\.?|LLC|Corp\.?|Corporation|Company|Co\.?|Ltd\.?|Limited|LP|LLP|Group|Bank|Financial|Securities|Investment|Capital|Trust|Holdings)\b', re.IGNORECASE),
                re.compile(r'\b(?:Bank|Credit Union|Savings|Trust)\s+(?:of\s+)?[A-Z][a-zA-Z\s&.-]{2,30}\b', re.IGNORECASE),
            ],
            'people': [
                re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'),
                re.compile(r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'),
            ],
            'government_agencies': [
                re.compile(r'\b(?:Securities and Exchange Commission|SEC)\b', re.IGNORECASE),
                re.compile(r'\b(?:Department of Justice|DOJ)\b', re.IGNORECASE),
                re.compile(r'\b(?:Consumer Financial Protection Bureau|CFPB)\b', re.IGNORECASE),
                re.compile(r'\b(?:Federal Trade Commission|FTC|Federal Reserve|Fed|FDIC|OCC|CFTC|FINRA)\b', re.IGNORECASE),
            ],
            'financial_amounts': [
                re.compile(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?', re.IGNORECASE),
                re.compile(r'\b(?:\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:million|billion|trillion)\s+dollars?)\b', re.IGNORECASE),
            ],
            'legal_violations': [
                re.compile(r'\b(?:securities fraud|insider trading|market manipulation|disclosure violations|wire fraud|mail fraud|bank fraud)\b', re.IGNORECASE),
                re.compile(r'\b(?:money laundering|anti-money laundering|AML|BSA violations|FCPA|racketeering|bribery|embezzlement)\b', re.IGNORECASE),
            ],
            'legal_statutes': [
                re.compile(r'\b(?:Section\s+\d+(?:\([a-z]\))?(?:\(\d+\))?|Rule\s+\d+[a-z]?-\d+)\b', re.IGNORECASE),
                re.compile(r'\b(?:Securities Act|Exchange Act|Investment Company Act|Investment Advisers Act|Sarbanes-Oxley Act|Dodd-Frank Act)\b', re.IGNORECASE),
            ],
            'law_firms': [
                re.compile(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:&\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+)*(?:LLP|PLLC|P\.A\.)\b'),
                re.compile(r'\b(?:Law Offices? of|The Law Firm of)\s+[A-Z][a-zA-Z\s&.-]+\b', re.IGNORECASE),
            ],
            'courts': [
                re.compile(r'\b(?:U\.S\.|United States)\s+District Court\s+for the\s+[A-Z][a-zA-Z\s]+District\b', re.IGNORECASE),
                re.compile(r'\b(?:U\.S\.|United States)\s+Court of Appeals\s+for the\s+[A-Z][a-zA-Z\s]+Circuit\b', re.IGNORECASE),
            ]
        }
        
        # Enhanced financial institution keywords
        self.financial_keywords = {
            'investment_bank': ['investment bank', 'bulge bracket', 'securities underwriting', 'capital markets'],
            'commercial_bank': ['commercial bank', 'retail banking', 'deposits', 'lending'],
            'asset_manager': ['asset management', 'fund management', 'portfolio management', 'wealth management'],
            'hedge_fund': ['hedge fund', 'alternative investment', 'private fund'],
            'private_equity': ['private equity', 'buyout', 'venture capital'],
            'insurance': ['insurance', 'life insurance', 'property casualty'],
            'fintech': ['fintech', 'financial technology', 'digital banking', 'cryptocurrency', 'blockchain']
        }
    
    def extract_entities(self, document: Document) -> List[Entity]:
        entities = []
        
        # Combine title and content for extraction
        text = f"{document.title}\n\n{document.content}"
        
        # 1. Regex-based extraction
        regex_entities = self._extract_with_regex(text)
        entities.extend(regex_entities)
        self.logger.info(f"Regex extraction: {len(regex_entities)} entities")
        
        # 2. spaCy-based extraction
        if self.spacy_nlp:
            spacy_entities = self._extract_with_spacy(text)
            entities.extend(spacy_entities)
            self.logger.info(f"spaCy extraction: {len(spacy_entities)} entities")
        
        # 3. OpenAI-based extraction for complex entities
        if self.openai_client and len(text) > 500:
            try:
                if len(text) > 15000:
                    if len(text) > 3000:
                        text = text[:2000] + text[-1000:]
                
                openai_entities = self._extract_with_openai(text, document)
                entities.extend(openai_entities)
                self.logger.info(f"OpenAI extraction: {len(openai_entities)} entities")
            except Exception as e:
                self.logger.error(f"OpenAI extraction failed: {e}")
        
        # 4. Deduplicate and enhance entities
        processed_entities = self._post_process_entities(entities)
        processed_entities = self._enhance_entities(processed_entities, text)
        
        self.logger.info(f"Total entities after processing: {len(processed_entities)}")
        return processed_entities
    
    def _extract_with_regex(self, text: str) -> List[Entity]:
        entities = []
        
        for entity_type, patterns in self.regex_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity_text = match.group().strip()
                    
                    # Skip very short matches or common false positives
                    if len(entity_text) < 2 or entity_text.lower() in ['the', 'and', 'or', 'of', 'in', 'to', 'for', 'a', 'an']:
                        continue
                    
                    entity = Entity(
                        id=f"regex_{len(entities)}",
                        name=entity_text,
                        type=entity_type,
                        confidence=0.70,
                        source="regex",
                        context=text[max(0, match.start()-50):match.end()+50],
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> List[Entity]:
        entities = []
        
        try:
            doc = self.spacy_nlp(text)
            
            # Entity type mapping from spaCy to our types
            type_mapping = {
                'PERSON': 'people',
                'ORG': 'companies',
                'MONEY': 'financial_amounts',
                'GPE': 'locations',
                'LOC': 'locations',
                'DATE': 'dates',
                'TIME': 'dates',
                'LAW': 'legal_statutes',
                'NORP': 'organizations'
            }
            
            for ent in doc.ents:
                entity_type = type_mapping.get(ent.label_, 'other')
                
                # Skip if not a relevant entity type
                if entity_type == 'other':
                    continue
                
                entity = Entity(
                    id=f"spacy_{len(entities)}",
                    name=ent.text.strip(),
                    type=entity_type,
                    confidence=0.80,
                    source="spacy",
                    context=text[max(0, ent.start_char-50):ent.end_char+50],
                    start_pos=ent.start_char,
                    end_pos=ent.end_char
                )
                entities.append(entity)
        
        except Exception as e:
            self.logger.error(f"spaCy extraction error: {e}")
        
        return entities
    
    def _extract_with_openai(self, text: str, document: Document) -> List[Entity]:
        entities = []
        
        prompt = f"""
        Extract legal entities from this text. Focus on:
        - Companies and financial institutions
        - Government agencies and regulators
        - People (executives, officials)
        - Financial amounts (settlements, fines, penalties)
        - Legal violations and charges
        - Law firms and courts
        
        Text: {text}
        
        Return entities in this format:
        ENTITY: [name] | TYPE: [companies/people/government_agencies/financial_amounts/legal_violations/law_firms/courts] | CONFIDENCE: [0.0-1.0]
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            
            entities = self._parse_openai_response(response.choices[0].message.content)
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
        
        return entities
