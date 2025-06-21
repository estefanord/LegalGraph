
from typing import Optional, Dict, Any


class LegalGraphError(Exception):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DataIngestionError(LegalGraphError):
    pass


class ScrapingError(DataIngestionError):
    
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(message, {"url": url, "status_code": status_code})
        self.url = url
        self.status_code = status_code


class ParseError(DataIngestionError):
    
    def __init__(self, message: str, document_type: Optional[str] = None, source: Optional[str] = None):
        super().__init__(message, {"document_type": document_type, "source": source})
        self.document_type = document_type
        self.source = source


class EntityExtractionError(LegalGraphError):
    pass


class RelationshipExtractionError(LegalGraphError):
    pass


class GraphError(LegalGraphError):
    pass


class GraphConstructionError(GraphError):
    pass


class GraphAnalysisError(GraphError):
    pass


class GraphQueryError(GraphError):
    pass


class StorageError(LegalGraphError):
    pass


class ConfigurationError(LegalGraphError):
    pass


class APIError(LegalGraphError):
    
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, {"status_code": status_code})
        self.status_code = status_code


class ValidationError(LegalGraphError):
    pass


class RateLimitError(LegalGraphError):
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message, {"retry_after": retry_after})
        self.retry_after = retry_after


class RiskScoringError(LegalGraphError):
    pass


class PipelineError(LegalGraphError):
    pass 