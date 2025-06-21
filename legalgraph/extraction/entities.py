import re
import time
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import logging

try:
    import spacy
    from spacy.matcher import Matcher
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from .models import Entity, EntityType, EntityExtractionResult
from ..core.config import settings
from ..core.logging import get_logger
from ..core.exceptions import EntityExtractionError
from ..ingestion.models import Document

logger = logging.getLogger(__name__)


class LegalEntityExtractor:
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self.matcher = None
        self.patterns = {}
        self.company_suffixes = self._build_company_suffixes()
        self.financial_patterns = self._build_financial_patterns()
        self.entity_type_mapping = self._build_entity_type_mapping()
        
        # Initialize spaCy
        self._initialize_spacy()
    
    def _initialize_spacy(self):
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
            
            # Add custom patterns
            self._add_custom_patterns()
            
        except OSError as e:
            logger.error(f"Failed to load spaCy model {self.model_name}: {e}")
            logger.info("Please install the model with: python -m spacy download en_core_web_sm")
            raise
    
    def extract_entities(self, document: 'Document') -> List[Entity]:
        all_entities = []
        
        if self.use_spacy and self.nlp:
            spacy_entities = self._extract_with_spacy(document)
            all_entities.extend(spacy_entities)
        
        regex_entities = self._extract_with_regex(document)
        all_entities.extend(regex_entities)
        
        merged_entities = self._merge_entities(all_entities)
        return merged_entities
    
    def _extract_with_spacy(self, document: 'Document') -> List[Entity]:
        if not self.nlp:
            return []
        
        text = f"{document.title} {document.content}"
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            if entity_type:
                entity = Entity(
                    text=ent.text,
                    entity_type=entity_type,
                    confidence=0.8,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    extraction_method='spacy'
                )
                entities.append(entity)
        
        return entities
    
    def _extract_with_regex(self, document: 'Document') -> List[Entity]:
        text = f"{document.title} {document.content}"
        entities = []
        
        for pattern_name, pattern_info in self.patterns.items():
            pattern = pattern_info['pattern']
            entity_type = pattern_info['type']
            confidence = pattern_info.get('confidence', 0.7)
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = Entity(
                    text=match.group(),
                    entity_type=entity_type,
                    confidence=confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    extraction_method='regex'
                )
                entities.append(entity)
        
        return entities
    
    def _build_regex_patterns(self) -> Dict[str, List[Tuple[str, re.Pattern]]]:
        patterns = {
            'companies': [
                ('company_with_suffix', re.compile(r'\b[A-Z][a-zA-Z\s&,.-]{2,50}(?:' + '|'.join(self.company_suffixes) + r')\b', re.IGNORECASE)),
                ('bank_names', re.compile(r'\b(?:Bank|Credit Union|Savings|Trust)\s+(?:of\s+)?[A-Z][a-zA-Z\s&.-]{2,30}\b', re.IGNORECASE)),
                ('financial_institutions', re.compile(r'\b[A-Z][a-zA-Z\s&.-]{2,30}\s+(?:Bank|Credit Union|Financial|Securities|Investment|Capital)\b', re.IGNORECASE)),
            ],
            'people': [
                ('full_names', re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b')),
                ('titled_names', re.compile(r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b')),
            ],
            'government_agencies': [
                ('sec_references', re.compile(r'\b(?:Securities and Exchange Commission|SEC)\b', re.IGNORECASE)),
                ('doj_references', re.compile(r'\b(?:Department of Justice|DOJ)\b', re.IGNORECASE)),
                ('cfpb_references', re.compile(r'\b(?:Consumer Financial Protection Bureau|CFPB)\b', re.IGNORECASE)),
                ('federal_agencies', re.compile(r'\b(?:Federal Trade Commission|FTC|Federal Reserve|Fed|FDIC|OCC|CFTC)\b', re.IGNORECASE)),
            ],
            'financial_amounts': self.financial_patterns,
            'legal_violations': [
                ('securities_violations', re.compile(r'\b(?:securities fraud|insider trading|market manipulation|disclosure violations)\b', re.IGNORECASE)),
                ('banking_violations', re.compile(r'\b(?:money laundering|anti-money laundering|AML|BSA violations|FCPA)\b', re.IGNORECASE)),
            ]
        }
        
        return patterns
    
    def _build_company_suffixes(self) -> Set[str]:
        return {
            'Inc', 'Inc.', 'LLC', 'Corp', 'Corp.', 'Ltd', 'Ltd.', 'LP', 'LLP',
            'Company', 'Co', 'Co.', 'Corporation', 'Limited', 'Partners', 'Group'
        }
    
    def _build_financial_patterns(self) -> List[Tuple[str, re.Pattern]]:
        return [
            ('dollar_amounts', re.compile(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?', re.IGNORECASE)),
            ('written_amounts', re.compile(r'\b(?:million|billion|trillion)\s+dollars?\b', re.IGNORECASE)),
        ]
    
    def _build_entity_type_mapping(self) -> Dict[str, str]:
        return {
            'PERSON': 'people',
            'ORG': 'companies',
            'MONEY': 'financial_amounts',
            'GPE': 'locations',
            'DATE': 'dates',
            'LAW': 'legal_terms',
            'NORP': 'organizations',
        }
    
    def _add_custom_patterns(self):
        if not self.matcher:
            return
        
        # Add patterns for legal entities
        bank_pattern = [{"LOWER": {"IN": ["bank", "credit", "financial"]}}, {"IS_TITLE": True}]
        self.matcher.add("BANK", [bank_pattern])
        
        # Add patterns for regulatory agencies
        sec_pattern = [{"LOWER": "securities"}, {"LOWER": "and"}, {"LOWER": "exchange"}, {"LOWER": "commission"}]
        self.matcher.add("SEC", [sec_pattern])
    
    def _map_spacy_entity_type(self, spacy_label: str) -> Optional[str]:
        return self.entity_type_mapping.get(spacy_label)
    
    def _map_pattern_to_entity_type(self, pattern_name: str) -> Optional[str]:
        mapping = {
            'BANK': 'companies',
            'SEC': 'government_agencies',
        }
        return mapping.get(pattern_name)
    
    def _calculate_spacy_confidence(self, ent, doc) -> float:
        # Base confidence from spaCy
        base_confidence = 0.8
        
        # Adjust based on entity length
        if len(ent.text) < 3:
            base_confidence -= 0.2
        elif len(ent.text) > 50:
            base_confidence -= 0.1
        
        # Adjust based on capitalization
        if ent.text.isupper() or ent.text.istitle():
            base_confidence += 0.1
        
        return min(max(base_confidence, 0.1), 1.0)
    
    def _normalize_entity_name(self, name: str) -> str:
        # Remove extra whitespace
        normalized = ' '.join(name.split())
        
        # Remove common prefixes/suffixes that don't affect identity
        prefixes = ['The ', 'A ', 'An ']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        return normalized.strip()
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _is_duplicate_entity(self, entity: Entity, existing_entities: List[Entity]) -> bool:
        for existing in existing_entities:
            # Check for text overlap
            if (entity.start_pos <= existing.end_pos and 
                entity.end_pos >= existing.start_pos):
                
                # Calculate overlap ratio
                overlap_start = max(entity.start_pos, existing.start_pos)
                overlap_end = min(entity.end_pos, existing.end_pos)
                overlap_length = overlap_end - overlap_start
                
                entity_length = entity.end_pos - entity.start_pos
                existing_length = existing.end_pos - existing.start_pos
                
                overlap_ratio = overlap_length / min(entity_length, existing_length)
                
                # Consider duplicate if significant overlap
                if overlap_ratio > 0.8:
                    return True
        
        return False
    
    def _post_process_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        # Remove duplicates and merge similar entities
        unique_entities = []
        
        for entity in entities:
            if not self._is_duplicate_entity(entity, unique_entities):
                unique_entities.append(entity)
        
        # Group entities by normalized name for merging
        entity_groups = defaultdict(list)
        for entity in unique_entities:
            normalized_name = self._normalize_entity_name(entity.name)
            entity_groups[normalized_name].append(entity)
        
        # Merge entities with same normalized name
        merged_entities = []
        for normalized_name, group in entity_groups.items():
            if len(group) == 1:
                merged_entities.append(group[0])
            else:
                merged_entity = self._merge_entities(group)
                merged_entities.append(merged_entity)
        
        return merged_entities
    
    def _merge_entities(self, entities: List[Entity]) -> Entity:
        # Take the entity with highest confidence as base
        best_entity = max(entities, key=lambda e: e.confidence)
        
        # Combine sources
        sources = set(e.source for e in entities)
        combined_source = ", ".join(sorted(sources))
        
        # Use average confidence
        avg_confidence = sum(e.confidence for e in entities) / len(entities)
        
        merged = Entity(
            id=best_entity.id,
            name=best_entity.name,
            type=best_entity.type,
            confidence=avg_confidence,
            source=combined_source,
            context=best_entity.context,
            start_pos=best_entity.start_pos,
            end_pos=best_entity.end_pos
        )
        
        return merged
    
    def _parse_financial_amount(self, text: str) -> Optional[float]:
        # Remove commas and dollar signs
        clean_text = text.replace(',', '').replace('$', '').strip()
        
        # Handle millions, billions, etc.
        multiplier = 1
        if 'million' in clean_text.lower():
            multiplier = 1_000_000
            clean_text = clean_text.lower().replace('million', '').strip()
        elif 'billion' in clean_text.lower():
            multiplier = 1_000_000_000
            clean_text = clean_text.lower().replace('billion', '').strip()
        elif 'trillion' in clean_text.lower():
            multiplier = 1_000_000_000_000
            clean_text = clean_text.lower().replace('trillion', '').strip()
        
        try:
            # Extract numeric part
            import re
            number_match = re.search(r'[\d.]+', clean_text)
            if number_match:
                number = float(number_match.group())
                return number * multiplier
        except ValueError:
            pass
        
        return None 