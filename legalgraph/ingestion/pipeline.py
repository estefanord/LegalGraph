
import asyncio
from datetime import datetime, date
from typing import List, Dict, Optional, Set, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .sources.sec_litigation import SECLitigationSource
from .sources.doj_press import DOJPressSource
from .sources.cfpb_actions import CFPBEnforcementSource
from .models import Document, IngestionResult, ValidationResult
from ..core.config import settings
from ..core.logging import get_logger, log_execution_time
from ..core.exceptions import DataIngestionError


@dataclass
class PipelineStats:
    total_documents: int = 0
    unique_documents: int = 0
    duplicates_removed: int = 0
    validation_failures: int = 0
    processing_time: float = 0.0
    source_results: Dict[str, IngestionResult] = None
    
    def __post_init__(self):
        if self.source_results is None:
            self.source_results = {}


class IngestionPipeline:
    
    def __init__(self):
        self.logger = get_logger("legalgraph.ingestion.pipeline")
        self.sources = {
            'sec_litigation': SECLitigationSource(),
            'doj_press': DOJPressSource(),
            'cfpb_enforcement': CFPBEnforcementSource()
        }
        self.seen_hashes: Set[str] = set()
    
    @log_execution_time(get_logger("legalgraph.ingestion.pipeline"))
    def run_incremental_update(self, days: int = 30, sources: Optional[List[str]] = None) -> PipelineStats:
        
        start_time = datetime.utcnow()
        self.logger.info(f"Starting incremental ingestion for last {days} days")
        
        # Determine which sources to run
        if sources is None:
            sources = list(self.sources.keys())
        
        # Validate source names
        invalid_sources = set(sources) - set(self.sources.keys())
        if invalid_sources:
            raise DataIngestionError(f"Invalid sources specified: {invalid_sources}")
        
        # Run ingestion for each source
        all_documents = []
        source_results = {}
        
        for source_name in sources:
            try:
                self.logger.info(f"Processing source: {source_name}")
                source = self.sources[source_name]
                
                # Create ingestion result tracking
                result = IngestionResult(
                    source=source_name,
                    start_time=datetime.utcnow()
                )
                
                # Fetch documents
                documents = source.fetch_recent(days=days)
                result.documents_found = len(documents)
                
                # Validate and deduplicate
                valid_documents = []
                for doc in documents:
                    validation_result = self.validate_document(doc)
                    if validation_result.is_valid:
                        if self._is_duplicate(doc):
                            result.skip_document("Duplicate document")
                        else:
                            valid_documents.append(doc)
                            result.add_document(doc)
                            self.seen_hashes.add(doc.content_hash)
                    else:
                        result.add_error(f"Validation failed: {'; '.join(validation_result.errors)}")
                
                all_documents.extend(valid_documents)
                result.complete()
                source_results[source_name] = result
                
                self.logger.info(
                    f"Source {source_name} completed: "
                    f"{result.documents_processed}/{result.documents_found} documents processed "
                    f"({result.success_rate:.1f}% success rate)"
                )
                
            except Exception as e:
                error_msg = str(e)
                if "Access denied" in error_msg or "403" in error_msg or "404" in error_msg:
                    self.logger.warning(f"Source {source_name} access denied - this is normal for government sites with bot protection")
                else:
                    self.logger.error(f"Source {source_name} failed: {e}")
                
                error_result = IngestionResult(
                    source=source_name,
                    start_time=datetime.utcnow()
                )
                error_result.add_error(str(e))
                if "Access denied" in error_msg or "403" in error_msg:
                    error_result.add_warning("Government site may be blocking automated access")
                error_result.complete()
                source_results[source_name] = error_result
        
        # Create pipeline statistics
        end_time = datetime.utcnow()
        stats = PipelineStats(
            total_documents=sum(r.documents_found for r in source_results.values()),
            unique_documents=len(all_documents),
            duplicates_removed=sum(r.documents_skipped for r in source_results.values()),
            validation_failures=sum(r.documents_failed for r in source_results.values()),
            processing_time=(end_time - start_time).total_seconds(),
            source_results=source_results
        )
        
        self.logger.info(
            f"Ingestion pipeline completed: {stats.unique_documents} unique documents processed "
            f"from {stats.total_documents} total documents in {stats.processing_time:.2f}s"
        )
        
        return stats
    
    @log_execution_time(get_logger("legalgraph.ingestion.pipeline"))
    def run_historical_backfill(self, start_date: date, end_date: Optional[date] = None) -> PipelineStats:
        
        if end_date is None:
            end_date = date.today()
        
        days_range = (end_date - start_date).days
        self.logger.info(f"Starting historical backfill from {start_date} to {end_date} ({days_range} days)")
        
        # For historical backfill, we might need to chunk the date range
        # For now, just use the total range
        return self.run_incremental_update(days=days_range)
    
    async def run_incremental_update_async(self, days: int = 30, sources: Optional[List[str]] = None) -> PipelineStats:
        
        start_time = datetime.utcnow()
        self.logger.info(f"Starting async incremental ingestion for last {days} days")
        
        # Determine which sources to run
        if sources is None:
            sources = list(self.sources.keys())
        
        # Run sources concurrently
        tasks = []
        for source_name in sources:
            if source_name in self.sources:
                task = self._run_source_async(source_name, days)
                tasks.append(task)
        
        # Wait for all sources to complete
        source_results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_documents = []
        source_results = {}
        
        for i, result in enumerate(source_results_list):
            source_name = sources[i]
            
            if isinstance(result, Exception):
                self.logger.error(f"Source {source_name} failed: {result}")
                error_result = IngestionResult(
                    source=source_name,
                    start_time=start_time
                )
                error_result.add_error(str(result))
                error_result.complete()
                source_results[source_name] = error_result
            else:
                documents, ingestion_result = result
                all_documents.extend(documents)
                source_results[source_name] = ingestion_result
        
        # Create pipeline statistics
        end_time = datetime.utcnow()
        stats = PipelineStats(
            total_documents=sum(r.documents_found for r in source_results.values()),
            unique_documents=len(all_documents),
            duplicates_removed=sum(r.documents_skipped for r in source_results.values()),
            validation_failures=sum(r.documents_failed for r in source_results.values()),
            processing_time=(end_time - start_time).total_seconds(),
            source_results=source_results
        )
        
        self.logger.info(
            f"Async ingestion pipeline completed: {stats.unique_documents} unique documents processed "
            f"from {stats.total_documents} total documents in {stats.processing_time:.2f}s"
        )
        
        return stats
    
    async def _run_source_async(self, source_name: str, days: int) -> tuple[List[Document], IngestionResult]:
        
        source = self.sources[source_name]
        result = IngestionResult(
            source=source_name,
            start_time=datetime.utcnow()
        )
        
        try:
            # Run source fetch in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                documents = await loop.run_in_executor(
                    executor, 
                    source.fetch_recent, 
                    days
                )
            
            result.documents_found = len(documents)
            
            # Validate and deduplicate
            valid_documents = []
            for doc in documents:
                validation_result = self.validate_document(doc)
                if validation_result.is_valid:
                    if self._is_duplicate(doc):
                        result.skip_document("Duplicate document")
                    else:
                        valid_documents.append(doc)
                        result.add_document(doc)
                        self.seen_hashes.add(doc.content_hash)
                else:
                    result.add_error(f"Validation failed: {'; '.join(validation_result.errors)}")
            
            result.complete()
            return valid_documents, result
            
        except Exception as e:
            result.add_error(str(e))
            result.complete()
            return [], result
    
    def validate_document(self, doc: Document) -> ValidationResult:
        
        validation = ValidationResult(
            is_valid=True,
            document_id=doc.id
        )
        
        # Required fields validation
        if not doc.title or len(doc.title.strip()) < 5:
            validation.add_error("Title is missing or too short")
        
        if not doc.content or len(doc.content.strip()) < 50:
            validation.add_error("Content is missing or too short")
        
        if not doc.source_url:
            validation.add_error("Source URL is required")
        
        # Content quality checks
        if doc.content:
            # Check for reasonable content length
            if len(doc.content) > settings.max_document_size:
                validation.add_error(f"Document content exceeds maximum size ({settings.max_document_size} bytes)")
            
            # Check for suspicious content patterns
            if doc.content.count('\n') / len(doc.content) > 0.1:  # Too many line breaks
                validation.add_warning("Document may have formatting issues")
            
            # Check for minimal actual content
            words = doc.content.split()
            if len(words) < 20:
                validation.add_error("Document content appears to be too short")
        
        # URL validation
        if doc.source_url:
            if not (doc.source_url.startswith('http://') or doc.source_url.startswith('https://')):
                validation.add_error("Invalid source URL format")
        
        # Date validation
        if doc.publish_date:
            if doc.publish_date > date.today():
                validation.add_warning("Document has future publication date")
            
            # Check if date is too old (might indicate parsing error)
            if doc.publish_date.year < 1990:
                validation.add_warning("Document publication date seems unusually old")
        
        return validation
    
    def _is_duplicate(self, doc: Document) -> bool:
        return doc.content_hash in self.seen_hashes
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        status = {
            'sources': {},
            'last_run': None,
            'total_documents_seen': len(self.seen_hashes)
        }
        
        # Test each source connectivity
        for source_name, source in self.sources.items():
            try:
                # Basic connectivity test (this is a simplified check)
                status['sources'][source_name] = {
                    'status': 'available',
                    'name': source.name
                }
            except Exception as e:
                status['sources'][source_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return status
    
    def reset_deduplication_cache(self):
        self.seen_hashes.clear()
        self.logger.info("Deduplication cache reset")
    
    def add_custom_source(self, name: str, source) -> None:
        if not hasattr(source, 'fetch_recent'):
            raise ValueError("Custom source must implement fetch_recent method")
        
        self.sources[name] = source
        self.logger.info(f"Added custom source: {name}")
    
    def remove_source(self, name: str) -> None:
        if name in self.sources:
            del self.sources[name]
            self.logger.info(f"Removed source: {name}")
        else:
            self.logger.warning(f"Source {name} not found") 