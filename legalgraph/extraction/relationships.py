import re
import time
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from datetime import date

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .models import Entity, Relationship, RelationType, EntityType, RelationshipExtractionResult
from ..core.config import settings
from ..core.logging import get_logger
from ..core.exceptions import RelationshipExtractionError
from ..ingestion.models import Document


class RelationshipExtractor:
    
    def __init__(self):
        self.logger = get_logger("legalgraph.extraction.relationships")
        self.confidence_threshold = settings.relationship_extraction_confidence_threshold
        
        # Relationship patterns for rule-based extraction
        self.relationship_patterns = self._build_relationship_patterns()
        self.financial_relationship_patterns = self._build_financial_patterns()
        self.temporal_patterns = self._build_temporal_patterns()
        
        # OpenAI client for LLM-based extraction
        self.openai_client = None
        if OPENAI_AVAILABLE and settings.openai_api_key:
            try:
                openai.api_key = settings.openai_api_key
                self.openai_client = openai
                self.logger.info("OpenAI client initialized for relationship extraction")
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenAI client: {e}")
    
    def extract_relationships(self, document: 'Document', entities: List[Entity] = None) -> List[Relationship]:
        if entities is None:
            entities = []
        
        entity_lookup = {entity.text.lower(): entity for entity in entities}
        
        relationships = self._extract_rule_based(document, entities)
        
        if self.openai_client:
            try:
                llm_relationships = self._extract_with_openai(document, entities)
                relationships.extend(llm_relationships)
            except Exception as e:
                print(f"OpenAI extraction failed: {e}")
        
        return self._post_process_relationships(relationships)
    
    def _extract_rule_based(self, document: 'Document', entities: List[Entity]) -> List[Relationship]:
        relationships = []
        text = f"{document.title} {document.content}"
        
        relationships.extend(self._extract_enforcement_relationships(text, entities))
        relationships.extend(self._extract_financial_relationships(text, entities))
        relationships.extend(self._extract_temporal_relationships(text, entities))
        relationships.extend(self._extract_proximity_relationships(text, entities))
        
        return relationships
    
    def _build_entity_lookup(self, entities: List[Entity]) -> Dict[str, Entity]:
        lookup = {}
        for entity in entities:
            # Add by text (case insensitive)
            lookup[entity.text.lower()] = entity
            lookup[entity.canonical_name.lower()] = entity
            
            # Add by aliases
            for alias in entity.aliases:
                lookup[alias.lower()] = entity
        
        return lookup
    
    def _extract_enforcement_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        relationships = []
        
        # Look for enforcement patterns
        for pattern_name, pattern_info in self.relationship_patterns['enforcement'].items():
            pattern = pattern_info['pattern']
            relationship_type = pattern_info['type']
            
            for match in pattern.finditer(text):
                match_start, match_end = match.span()
                match_text = match.group(0)
                
                # Find entities around this match
                nearby_entities = self._find_entities_near_match(
                    entities, match_start, match_end, window=200
                )
                
                # Create relationships between relevant entities
                for i, source_entity in enumerate(nearby_entities):
                    for target_entity in nearby_entities[i+1:]:
                        
                        # Determine relationship direction based on entity types
                        if self._should_create_enforcement_relationship(
                            source_entity, target_entity, pattern_name
                        ):
                            relationship = Relationship(
                                source_entity_id=source_entity.id,
                                target_entity_id=target_entity.id,
                                relationship_type=relationship_type,
                                confidence=0.8,
                                extraction_method=f"pattern_{pattern_name}",
                                source_document_id=entities[0].source_document_id if entities else "",
                                description=f"{source_entity.canonical_name} {pattern_name} {target_entity.canonical_name}",
                                evidence_text=match_text
                            )
                            
                            relationships.append(relationship)
        
        return relationships
    
    def _extract_financial_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        relationships = []
        
        # Find money entities
        money_entities = [e for e in entities if e.entity_type == EntityType.MONEY]
        other_entities = [e for e in entities if e.entity_type != EntityType.MONEY]
        
        # Look for settlement/payment patterns
        for pattern_name, pattern_info in self.financial_relationship_patterns.items():
            pattern = pattern_info['pattern']
            relationship_type = pattern_info['type']
            
            for match in pattern.finditer(text):
                match_start, match_end = match.span()
                match_text = match.group(0)
                
                # Find money entities near the match
                nearby_money = self._find_entities_near_match(
                    money_entities, match_start, match_end, window=100
                )
                
                # Find other entities near the match
                nearby_entities = self._find_entities_near_match(
                    other_entities, match_start, match_end, window=150
                )
                
                # Create relationships between entities and money
                for money_entity in nearby_money:
                    for other_entity in nearby_entities:
                        if other_entity.entity_type in [EntityType.COMPANY, EntityType.PERSON]:
                            
                            relationship = Relationship(
                                source_entity_id=other_entity.id,
                                target_entity_id=money_entity.id,
                                relationship_type=relationship_type,
                                confidence=0.85,
                                extraction_method=f"financial_pattern_{pattern_name}",
                                source_document_id=entities[0].source_document_id if entities else "",
                                description=f"{other_entity.canonical_name} {pattern_name} {money_entity.canonical_name}",
                                evidence_text=match_text
                            )
                            
                            # Try to extract amount from money entity
                            if 'parsed_amount' in money_entity.attributes:
                                relationship.set_financial_info(money_entity.attributes['parsed_amount'])
                            
                            relationships.append(relationship)
        
        return relationships
    
    def _extract_temporal_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        relationships = []
        
        date_entities = [e for e in entities if e.entity_type == EntityType.DATE]
        other_entities = [e for e in entities if e.entity_type != EntityType.DATE]
        
        # Connect events with dates
        for date_entity in date_entities:
            nearby_entities = self._find_entities_near_match(
                other_entities, date_entity.start_char, date_entity.end_char, window=100
            )
            
            for entity in nearby_entities:
                relationship = Relationship(
                    source_entity_id=entity.id,
                    target_entity_id=date_entity.id,
                    relationship_type=RelationType.OCCURRED_ON,
                    confidence=0.7,
                    extraction_method="temporal_proximity",
                    source_document_id=entities[0].source_document_id if entities else "",
                    description=f"{entity.canonical_name} occurred on {date_entity.canonical_name}",
                    evidence_text=text[date_entity.start_char-50:date_entity.end_char+50]
                )
                
                relationships.append(relationship)
        
        return relationships
    
    def _extract_proximity_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        relationships = []
        
        # Group entities by type for more targeted relationship extraction
        company_entities = [e for e in entities if e.entity_type == EntityType.COMPANY]
        person_entities = [e for e in entities if e.entity_type == EntityType.PERSON]
        agency_entities = [e for e in entities if e.entity_type == EntityType.GOVERNMENT_AGENCY]
        
        # Create relationships between companies and agencies (regulation)
        for company in company_entities:
            for agency in agency_entities:
                distance = abs(company.start_char - agency.start_char)
                if distance < 500:  # Within reasonable proximity
                    
                    relationship = Relationship(
                        source_entity_id=company.id,
                        target_entity_id=agency.id,
                        relationship_type=RelationType.REGULATED_BY,
                        confidence=0.6,
                        extraction_method="proximity_regulation",
                        source_document_id=company.source_document_id,
                        description=f"{company.canonical_name} regulated by {agency.canonical_name}",
                        strength=max(0.1, 1.0 - (distance / 500.0))  # Closer = stronger
                    )
                    
                    relationships.append(relationship)
        
        # Create relationships between people and companies (employment/involvement)
        for person in person_entities:
            for company in company_entities:
                distance = abs(person.start_char - company.start_char)
                if distance < 300:  # Closer proximity for person-company relationships
                    
                    relationship = Relationship(
                        source_entity_id=person.id,
                        target_entity_id=company.id,
                        relationship_type=RelationType.EMPLOYED_BY,
                        confidence=0.5,
                        extraction_method="proximity_employment",
                        source_document_id=person.source_document_id,
                        description=f"{person.canonical_name} associated with {company.canonical_name}",
                        strength=max(0.1, 1.0 - (distance / 300.0))
                    )
                    
                    relationships.append(relationship)
        
        return relationships
    
    def _extract_with_llm(self, text: str, entities: List[Entity], 
                         entity_lookup: Dict[str, Entity]) -> List[Relationship]:
        if not self.openai_client:
            return []
        
        relationships = []
        
        try:
            # Prepare entity list for the prompt
            entity_list = []
            for i, entity in enumerate(entities[:20]):  # Limit to avoid token limits
                entity_list.append(f"Entity {i}: {entity.canonical_name} ({entity.entity_type.value})")
            
            entity_text = "\n".join(entity_list)
            
            # Create prompt for relationship extraction
            prompt = f"""
            You are a legal document analyst. Given the following legal document text and entities, identify relationships between the entities.

            Entities:
            {entity_text}

            Document text (first 2000 characters):
            {text[:2000]}

            Please identify relationships between the entities. For each relationship, provide:
            1. Source entity number
            2. Target entity number  
            3. Relationship type (enforcement_action, settlement, violation_of, charged_with, etc.)
            4. Confidence (0.0-1.0)
            5. Brief description

            Format your response as JSON array of relationships.
            """
            
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal document analyst specializing in relationship extraction."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            llm_relationships = self._parse_llm_response(response_text)
            
            # Convert to Relationship objects
            for rel_data in llm_relationships:
                if self._validate_llm_relationship(rel_data, entities):
                    relationship = self._create_relationship_from_llm(rel_data, entities)
                    relationships.append(relationship)
                    
        except Exception as e:
            self.logger.error(f"LLM relationship extraction failed: {e}")
        
        return relationships
    
    def _parse_llm_response(self, response_text: str) -> List[Dict]:
        # Implementation of _parse_llm_response method
        pass
    
    def _validate_llm_relationship(self, rel_data: Dict, entities: List[Entity]) -> bool:
        # Implementation of _validate_llm_relationship method
        pass
    
    def _create_relationship_from_llm(self, rel_data: Dict, entities: List[Entity]) -> Relationship:
        # Implementation of _create_relationship_from_llm method
        pass
