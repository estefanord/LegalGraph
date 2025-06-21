
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import uuid


class EntityType(Enum):
    PERSON = "person"
    COMPANY = "company"
    LAW_FIRM = "law_firm"
    GOVERNMENT_AGENCY = "government_agency"
    COURT = "court"
    JUDGE = "judge"
    LAWYER = "lawyer"
    FINANCIAL_INSTITUTION = "financial_institution"
    INVESTMENT_FIRM = "investment_firm"
    REGULATORY_BODY = "regulatory_body"
    LOCATION = "location"
    LEGAL_CASE = "legal_case"
    STATUTE = "statute"
    REGULATION = "regulation"
    VIOLATION_TYPE = "violation_type"
    INDUSTRY = "industry"
    PRODUCT = "product"
    MONEY = "money"
    DATE = "date"
    UNKNOWN = "unknown"


class RelationType(Enum):
    ENFORCEMENT_ACTION = "enforcement_action"
    SETTLEMENT = "settlement"
    VIOLATION_OF = "violation_of"
    CHARGED_WITH = "charged_with"
    REPRESENTED_BY = "represented_by"
    EMPLOYED_BY = "employed_by"
    SUBSIDIARY_OF = "subsidiary_of"
    PARTNER_OF = "partner_of"
    COMPETITOR_OF = "competitor_of"
    REGULATED_BY = "regulated_by"
    JURISDICTION_OF = "jurisdiction_of"
    PROSECUTED_BY = "prosecuted_by"
    SETTLED_WITH = "settled_with"
    PAID_TO = "paid_to"
    INVOLVED_IN = "involved_in"
    OPERATES_IN = "operates_in"
    OCCURRED_ON = "occurred_on"
    RESULTED_IN = "resulted_in"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    
    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""  # Original text mention
    canonical_name: str = ""  # Normalized name
    entity_type: EntityType = EntityType.UNKNOWN
    
    # Confidence and source tracking
    confidence: float = 0.0
    extraction_method: str = ""  # "spacy_ner", "regex", "llm", etc.
    source_document_id: str = ""
    mention_count: int = 1
    
    # Position information
    start_char: int = 0
    end_char: int = 0
    context_window: str = ""  # Surrounding text
    
    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.canonical_name:
            self.canonical_name = self.text.strip()
        
        # Ensure start/end positions are consistent
        if self.end_char <= self.start_char and self.text:
            self.end_char = self.start_char + len(self.text)
    
    def add_alias(self, alias: str):
        if alias and alias not in self.aliases and alias != self.canonical_name:
            self.aliases.append(alias)
            self.updated_at = datetime.utcnow()
    
    def increment_mention_count(self):
        self.mention_count += 1
        self.updated_at = datetime.utcnow()
    
    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type.value,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "source_document_id": self.source_document_id,
            "mention_count": self.mention_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "context_window": self.context_window,
            "attributes": self.attributes,
            "aliases": self.aliases,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class Relationship:
    
    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str = ""
    target_entity_id: str = ""
    relationship_type: RelationType = RelationType.UNKNOWN
    
    # Confidence and extraction
    confidence: float = 0.0
    extraction_method: str = ""
    source_document_id: str = ""
    
    # Relationship details
    description: str = ""  # Human-readable description
    evidence_text: str = ""  # Supporting text from document
    
    # Temporal information
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_ongoing: bool = True
    
    # Financial information
    amount: Optional[float] = None
    currency: str = "USD"
    
    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0  # Relationship strength (0-1)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def set_temporal_info(self, start_date: Optional[date] = None, 
                         end_date: Optional[date] = None, 
                         is_ongoing: bool = True):
        self.start_date = start_date
        self.end_date = end_date
        self.is_ongoing = is_ongoing
        self.updated_at = datetime.utcnow()
    
    def set_financial_info(self, amount: float, currency: str = "USD"):
        self.amount = amount
        self.currency = currency
        self.updated_at = datetime.utcnow()
    
    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relationship_type": self.relationship_type.value,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "source_document_id": self.source_document_id,
            "description": self.description,
            "evidence_text": self.evidence_text,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "is_ongoing": self.is_ongoing,
            "amount": self.amount,
            "currency": self.currency,
            "attributes": self.attributes,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class EntityExtractionResult:
    
    document_id: str
    entities: List[Entity] = field(default_factory=list)
    extraction_time: float = 0.0
    method: str = ""
    confidence_threshold: float = 0.0
    
    # Statistics
    total_mentions: int = 0
    unique_entities: int = 0
    high_confidence_entities: int = 0
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_entity(self, entity: Entity):
        self.entities.append(entity)
        self.total_mentions += entity.mention_count
        
        # Update statistics
        self.unique_entities = len(set(e.canonical_name for e in self.entities))
        self.high_confidence_entities = sum(1 for e in self.entities 
                                          if e.confidence >= self.confidence_threshold)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        return [e for e in self.entities if e.entity_type == entity_type]
    
    def get_high_confidence_entities(self) -> List[Entity]:
        return [e for e in self.entities if e.confidence >= self.confidence_threshold]


@dataclass
class RelationshipExtractionResult:
    
    document_id: str
    relationships: List[Relationship] = field(default_factory=list)
    extraction_time: float = 0.0
    method: str = ""
    confidence_threshold: float = 0.0
    
    # Statistics
    total_relationships: int = 0
    unique_relationship_types: int = 0
    high_confidence_relationships: int = 0
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_relationship(self, relationship: Relationship):
        self.relationships.append(relationship)
        self.total_relationships = len(self.relationships)
        
        # Update statistics
        relationship_types = set(r.relationship_type for r in self.relationships)
        self.unique_relationship_types = len(relationship_types)
        self.high_confidence_relationships = sum(1 for r in self.relationships 
                                               if r.confidence >= self.confidence_threshold)
    
    def get_relationships_by_type(self, relationship_type: RelationType) -> List[Relationship]:
        return [r for r in self.relationships if r.relationship_type == relationship_type]
    
    def get_high_confidence_relationships(self) -> List[Relationship]:
        return [r for r in self.relationships if r.confidence >= self.confidence_threshold] 