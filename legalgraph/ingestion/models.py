
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import hashlib
import json


class DocumentType(Enum):
    SEC_LITIGATION_RELEASE = "sec_litigation_release"
    DOJ_PRESS_RELEASE = "doj_press_release"
    CFPB_ENFORCEMENT_ACTION = "cfpb_enforcement_action"
    COURT_OPINION = "court_opinion"
    REGULATORY_FILING = "regulatory_filing"


class DocumentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Document:
    
    # Core identifiers
    id: str
    source_url: str
    document_type: DocumentType
    
    # Content
    title: str
    content: str
    
    # Metadata
    publish_date: Optional[date] = None
    source_id: Optional[str] = None  # Original ID from source system
    agencies: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    
    # Processing metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: DocumentStatus = DocumentStatus.PENDING
    
    # Additional structured data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from URL and content hash
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            url_hash = hashlib.md5(self.source_url.encode()).hexdigest()[:8]
            self.id = f"{self.document_type.value}_{url_hash}_{content_hash}"
    
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_url": self.source_url,
            "document_type": self.document_type.value,
            "title": self.title,
            "content": self.content,
            "publish_date": self.publish_date.isoformat() if self.publish_date else None,
            "source_id": self.source_id,
            "agencies": self.agencies,
            "topics": self.topics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "metadata": self.metadata,
            "content_hash": self.content_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        return cls(
            id=data["id"],
            source_url=data["source_url"],
            document_type=DocumentType(data["document_type"]),
            title=data["title"],
            content=data["content"],
            publish_date=datetime.fromisoformat(data["publish_date"]).date() if data.get("publish_date") else None,
            source_id=data.get("source_id"),
            agencies=data.get("agencies", []),
            topics=data.get("topics", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            status=DocumentStatus(data["status"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class IngestionResult:
    
    source: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Counters
    documents_found: int = 0
    documents_processed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    
    # Results
    documents: List[Document] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def success_rate(self) -> float:
        if self.documents_found == 0:
            return 0.0
        return (self.documents_processed / self.documents_found) * 100
    
    def add_document(self, document: Document):
        self.documents.append(document)
        self.documents_processed += 1
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.documents_failed += 1
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    def skip_document(self, reason: str = ""):
        self.documents_skipped += 1
        if reason:
            self.errors.append(f"Skipped: {reason}")
    
    def complete(self):
        self.end_time = datetime.utcnow()


@dataclass
class ValidationResult:
    
    is_valid: bool
    document_id: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)


@dataclass
class SECLitigationDocument(Document):
    
    release_number: Optional[str] = None
    defendants: List[str] = field(default_factory=list)
    settlement_amount: Optional[float] = None
    charges: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        self.document_type = DocumentType.SEC_LITIGATION_RELEASE


@dataclass
class DOJPressDocument(Document):
    
    component: Optional[str] = None
    enforcement_type: Optional[str] = None  # 'criminal', 'civil', etc.
    financial_penalty: Optional[float] = None
    targets: List[str] = field(default_factory=list)  # Companies/individuals
    jurisdiction: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.document_type = DocumentType.DOJ_PRESS_RELEASE


@dataclass
class CFPBEnforcementDocument(Document):
    
    action_type: Optional[str] = None  # 'consent_order', 'federal_court_action'
    consumer_redress: Optional[float] = None
    civil_penalty: Optional[float] = None
    industry: Optional[str] = None
    violation_types: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        self.document_type = DocumentType.CFPB_ENFORCEMENT_ACTION 