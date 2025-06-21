
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from ..ingestion.pipeline import IngestionPipeline
from ..ingestion.models import Document
from ..extraction.entities import LegalEntityExtractor
from ..extraction.relationships import RelationshipExtractor
from ..extraction.models import Entity, Relationship
from ..graph.constructor import GraphConstructor
from ..graph.analyzer import NetworkAnalyzer
from ..graph.scoring import RiskScorer, RiskExplanation
from ..graph.models import LegalGraph, GraphType
from ..core.config import settings
from ..core.logging import get_logger
from ..core.exceptions import PipelineError


@dataclass
class PipelineResult:
    
    # Input statistics
    total_documents: int = 0
    processed_documents: int = 0
    
    # Entity extraction results
    total_entities: int = 0
    filtered_entities: int = 0
    entity_types: Dict[str, int] = field(default_factory=dict)
    
    # Relationship extraction results
    total_relationships: int = 0
    filtered_relationships: int = 0
    relationship_types: Dict[str, int] = field(default_factory=dict)
    
    # Graph construction results
    graph_nodes: int = 0
    graph_edges: int = 0
    graph_components: int = 0
    
    # Network analysis results
    centrality_analysis: Dict[str, Any] = field(default_factory=dict)
    community_detection: Dict[str, Any] = field(default_factory=dict)
    anomaly_detection: Dict[str, Any] = field(default_factory=dict)
    temporal_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Risk scoring results
    risk_distribution: Dict[str, int] = field(default_factory=dict)
    high_risk_entities: List[RiskExplanation] = field(default_factory=list)
    
    # Performance metrics
    total_processing_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    
    # The final graph
    graph: Optional[LegalGraph] = None
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_statistics": {
                "total_documents": self.total_documents,
                "processed_documents": self.processed_documents
            },
            "entity_extraction": {
                "total_entities": self.total_entities,
                "filtered_entities": self.filtered_entities,
                "entity_types": self.entity_types
            },
            "relationship_extraction": {
                "total_relationships": self.total_relationships,
                "filtered_relationships": self.filtered_relationships,
                "relationship_types": self.relationship_types
            },
            "graph_construction": {
                "graph_nodes": self.graph_nodes,
                "graph_edges": self.graph_edges,
                "graph_components": self.graph_components
            },
            "network_analysis": {
                "centrality_analysis": self.centrality_analysis,
                "community_detection": self.community_detection,
                "anomaly_detection": self.anomaly_detection,
                "temporal_analysis": self.temporal_analysis
            },
            "risk_scoring": {
                "risk_distribution": self.risk_distribution,
                "high_risk_entities": [entity.to_dict() for entity in self.high_risk_entities]
            },
            "performance": {
                "total_processing_time": self.total_processing_time,
                "stage_times": self.stage_times
            },
            "graph_statistics": self.graph.get_statistics().to_dict() if self.graph else {},
            "errors": self.errors,
            "warnings": self.warnings
        }


class LegalGraphPipeline:
    
    def __init__(self, graph_type: GraphType = GraphType.DIRECTED):
        self.logger = get_logger("legalgraph.pipeline.integration")
        self.graph_type = graph_type
        
        # Initialize pipeline components
        self.ingestion_pipeline = IngestionPipeline()
        self.entity_extractor = LegalEntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.graph_constructor = GraphConstructor(graph_type)
        self.network_analyzer = NetworkAnalyzer()
        self.risk_scorer = RiskScorer()
        
        # Pipeline configuration
        self.enable_network_analysis = True
        self.enable_risk_scoring = True
        self.max_documents_per_run = 1000
        self.batch_size = 50
    
    async def run_full_pipeline(self, 
                               days_back: int = 30,
                               sources: Optional[List[str]] = None) -> PipelineResult:
        start_time = time.time()
        
        self.logger.info(f"Starting full LegalGraph pipeline (days_back={days_back})")
        
        try:
            result = PipelineResult()
            
            # Stage 1: Document Ingestion
            stage_start = time.time()
            documents = await self._run_ingestion_stage(days_back, sources, result)
            result.stage_times['ingestion'] = time.time() - stage_start
            
            if not documents:
                self.logger.warning("No documents retrieved from ingestion")
                result.total_processing_time = time.time() - start_time
                return result
            
            # Stage 2: Entity Extraction
            stage_start = time.time()
            entities = await self._run_entity_extraction_stage(documents, result)
            result.stage_times['entity_extraction'] = time.time() - stage_start
            
            # Stage 3: Relationship Extraction
            stage_start = time.time()
            relationships = await self._run_relationship_extraction_stage(documents, entities, result)
            result.stage_times['relationship_extraction'] = time.time() - stage_start
            
            # Stage 4: Graph Construction
            stage_start = time.time()
            graph = await self._run_graph_construction_stage(entities, relationships, result)
            result.stage_times['graph_construction'] = time.time() - stage_start
            
            if not graph or len(graph) == 0:
                self.logger.warning("Empty graph constructed")
                result.total_processing_time = time.time() - start_time
                return result
            
            result.graph = graph
            
            # Stage 5: Network Analysis (optional)
            if self.enable_network_analysis:
                stage_start = time.time()
                await self._run_network_analysis_stage(graph, result)
                result.stage_times['network_analysis'] = time.time() - stage_start
            
            # Stage 6: Risk Scoring (optional)
            if self.enable_risk_scoring:
                stage_start = time.time()
                await self._run_risk_scoring_stage(graph, result)
                result.stage_times['risk_scoring'] = time.time() - stage_start
            
            result.total_processing_time = time.time() - start_time
            
            self.logger.info(
                f"Pipeline completed successfully in {result.total_processing_time:.2f}s: "
                f"{result.graph_nodes} nodes, {result.graph_edges} edges, "
                f"{len(result.high_risk_entities)} high-risk entities"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            result.errors.append(f"Pipeline failed: {str(e)}")
            result.total_processing_time = time.time() - start_time
            raise PipelineError(f"Pipeline execution failed: {e}") from e
    
    async def run_incremental_update(self, 
                                   existing_graph: LegalGraph,
                                   days_back: int = 7) -> PipelineResult:
        start_time = time.time()
        
        self.logger.info(f"Starting incremental update (days_back={days_back})")
        
        try:
            result = PipelineResult()
            
            # Get new documents
            stage_start = time.time()
            new_documents = await self._run_ingestion_stage(days_back, None, result)
            result.stage_times['ingestion'] = time.time() - stage_start
            
            if not new_documents:
                self.logger.info("No new documents found for incremental update")
                result.graph = existing_graph
                result.total_processing_time = time.time() - start_time
                return result
            
            # Process new documents
            stage_start = time.time()
            new_entities = await self._run_entity_extraction_stage(new_documents, result)
            new_relationships = await self._run_relationship_extraction_stage(
                new_documents, new_entities, result
            )
            result.stage_times['extraction'] = time.time() - stage_start
            
            # Create incremental graph
            stage_start = time.time()
            incremental_graph = await self._run_graph_construction_stage(
                new_entities, new_relationships, result
            )
            result.stage_times['graph_construction'] = time.time() - stage_start
            
            # Merge with existing graph
            if incremental_graph and len(incremental_graph) > 0:
                merged_graph = self.graph_constructor.merge_graphs([existing_graph, incremental_graph])
                result.graph = merged_graph
                
                # Update analysis
                if self.enable_network_analysis:
                    stage_start = time.time()
                    await self._run_network_analysis_stage(merged_graph, result)
                    result.stage_times['network_analysis'] = time.time() - stage_start
                
                if self.enable_risk_scoring:
                    stage_start = time.time()
                    await self._run_risk_scoring_stage(merged_graph, result)
                    result.stage_times['risk_scoring'] = time.time() - stage_start
            else:
                result.graph = existing_graph
            
            result.total_processing_time = time.time() - start_time
            
            self.logger.info(
                f"Incremental update completed in {result.total_processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Incremental update failed: {e}")
            result.errors.append(f"Incremental update failed: {str(e)}")
            result.total_processing_time = time.time() - start_time
            raise PipelineError(f"Incremental update failed: {e}") from e
    
    async def _run_ingestion_stage(self, 
                                  days_back: int,
                                  sources: Optional[List[str]],
                                  result: PipelineResult) -> List[Document]:
        self.logger.info(f"Running ingestion stage (days_back={days_back})")
        
        try:
            # Run ingestion pipeline
            ingestion_result = await asyncio.to_thread(
                self.ingestion_pipeline.run_incremental_update,
                days=days_back,
                sources=sources
            )
            
            # Update result statistics
            result.total_documents = ingestion_result.total_documents
            result.processed_documents = ingestion_result.unique_documents
            
            # Add any errors/warnings from source results
            for source_result in ingestion_result.source_results.values():
                result.errors.extend(source_result.errors)
                result.warnings.extend(source_result.warnings)
            
            self.logger.info(
                f"Ingestion completed: {ingestion_result.unique_documents} documents processed"
            )
            
            # Extract documents from the pipeline result
            documents = []
            for source_result in ingestion_result.source_results.values():
                documents.extend(source_result.documents)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Ingestion stage failed: {e}")
            result.errors.append(f"Ingestion failed: {str(e)}")
            return []
    
    async def _run_entity_extraction_stage(self, 
                                         documents: List[Document],
                                         result: PipelineResult) -> List[Entity]:
        self.logger.info(f"Running entity extraction on {len(documents)} documents")
        
        all_entities = []
        entity_type_counts = {}
        
        try:
            # Process documents in batches
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                
                # Extract entities from batch
                batch_entities = await asyncio.to_thread(
                    self._extract_entities_batch, batch
                )
                
                all_entities.extend(batch_entities)
                
                # Update entity type counts
                for entity in batch_entities:
                    entity_type = entity.entity_type.value
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
                
                self.logger.debug(f"Processed batch {i//self.batch_size + 1}/{(len(documents) + self.batch_size - 1)//self.batch_size}")
            
            # Update result statistics
            result.total_entities = len(all_entities)
            result.entity_types = entity_type_counts
            
            # Filter entities (this will be done in graph construction, but we track stats here)
            filtered_entities = [e for e in all_entities if e.confidence >= 0.5]
            result.filtered_entities = len(filtered_entities)
            
            self.logger.info(
                f"Entity extraction completed: {len(all_entities)} entities extracted, "
                f"{len(filtered_entities)} passed confidence threshold"
            )
            
            return all_entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction stage failed: {e}")
            result.errors.append(f"Entity extraction failed: {str(e)}")
            return []
    
    async def _run_relationship_extraction_stage(self, 
                                                documents: List[Document],
                                                entities: List[Entity],
                                                result: PipelineResult) -> List[Relationship]:
        self.logger.info(f"Running relationship extraction on {len(documents)} documents")
        
        all_relationships = []
        relationship_type_counts = {}
        
        # Group entities by document
        entities_by_doc = {}
        for entity in entities:
            doc_id = entity.source_document_id
            if doc_id not in entities_by_doc:
                entities_by_doc[doc_id] = []
            entities_by_doc[doc_id].append(entity)
        
        try:
            # Process documents in batches
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                
                # Extract relationships from batch
                batch_relationships = await asyncio.to_thread(
                    self._extract_relationships_batch, batch, entities_by_doc
                )
                
                all_relationships.extend(batch_relationships)
                
                # Update relationship type counts
                for relationship in batch_relationships:
                    rel_type = relationship.relationship_type.value
                    relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1
                
                self.logger.debug(f"Processed batch {i//self.batch_size + 1}/{(len(documents) + self.batch_size - 1)//self.batch_size}")
            
            # Update result statistics
            result.total_relationships = len(all_relationships)
            result.relationship_types = relationship_type_counts
            
            # Filter relationships (this will be done in graph construction, but we track stats here)
            filtered_relationships = [r for r in all_relationships if r.confidence >= 0.4]
            result.filtered_relationships = len(filtered_relationships)
            
            self.logger.info(
                f"Relationship extraction completed: {len(all_relationships)} relationships extracted, "
                f"{len(filtered_relationships)} passed confidence threshold"
            )
            
            return all_relationships
            
        except Exception as e:
            self.logger.error(f"Relationship extraction stage failed: {e}")
            result.errors.append(f"Relationship extraction failed: {str(e)}")
            return []
    
    async def _run_graph_construction_stage(self, 
                                          entities: List[Entity],
                                          relationships: List[Relationship],
                                          result: PipelineResult) -> Optional[LegalGraph]:
        self.logger.info(f"Running graph construction with {len(entities)} entities and {len(relationships)} relationships")
        
        try:
            # Construct graph
            graph = await asyncio.to_thread(
                self.graph_constructor.construct_graph, entities, relationships
            )
            
            # Update result statistics
            result.graph_nodes = len(graph)
            result.graph_edges = len(graph.nx_graph.edges()) if graph else 0
            result.graph_components = len(graph.get_connected_components()) if graph else 0
            
            self.logger.info(
                f"Graph construction completed: {result.graph_nodes} nodes, "
                f"{result.graph_edges} edges, {result.graph_components} components"
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Graph construction stage failed: {e}")
            result.errors.append(f"Graph construction failed: {str(e)}")
            return None
    
    async def _run_network_analysis_stage(self, 
                                        graph: LegalGraph,
                                        result: PipelineResult):
        self.logger.info("Running network analysis stage")
        
        try:
            # Centrality analysis
            centrality_results = await asyncio.to_thread(
                self.network_analyzer.analyze_centrality, graph
            )
            result.centrality_analysis = {
                'metrics_calculated': list(centrality_results.keys()),
                'top_nodes': self._get_top_centrality_nodes(centrality_results, 5)
            }
            
            # Community detection
            community_results = await asyncio.to_thread(
                self.network_analyzer.detect_communities, graph
            )
            result.community_detection = community_results
            
            # Anomaly detection
            anomaly_results = await asyncio.to_thread(
                self.network_analyzer.detect_anomalies, graph
            )
            result.anomaly_detection = {
                'total_anomalies': sum(len(v) for v in anomaly_results.values() if isinstance(v, list)),
                'anomaly_types': {k: len(v) for k, v in anomaly_results.items() if isinstance(v, list)}
            }
            
            # Temporal analysis
            temporal_results = await asyncio.to_thread(
                self.network_analyzer.analyze_temporal_patterns, graph
            )
            result.temporal_analysis = {
                'temporal_edges': len(temporal_results.get('edge_timeline', [])),
                'activity_periods': len(temporal_results.get('activity_periods', [])),
                'trend_analysis': temporal_results.get('trend_analysis', {})
            }
            
            self.logger.info("Network analysis completed")
            
        except Exception as e:
            self.logger.error(f"Network analysis stage failed: {e}")
            result.errors.append(f"Network analysis failed: {str(e)}")
    
    async def _run_risk_scoring_stage(self, 
                                    graph: LegalGraph,
                                    result: PipelineResult):
        self.logger.info("Running risk scoring stage")
        
        try:
            # Calculate risk scores
            risk_scores = await asyncio.to_thread(
                self.risk_scorer.calculate_risk_scores, graph
            )
            
            # Update result statistics
            risk_levels = [rs.risk_level for rs in risk_scores.values()]
            result.risk_distribution = {
                'LOW': risk_levels.count('LOW'),
                'MEDIUM': risk_levels.count('MEDIUM'),
                'HIGH': risk_levels.count('HIGH'),
                'CRITICAL': risk_levels.count('CRITICAL')
            }
            
            # Get high-risk entities
            result.high_risk_entities = self.risk_scorer.get_high_risk_entities(
                risk_scores, min_risk_level='HIGH'
            )
            
            self.logger.info(
                f"Risk scoring completed: {len(result.high_risk_entities)} high-risk entities identified"
            )
            
        except Exception as e:
            self.logger.error(f"Risk scoring stage failed: {e}")
            result.errors.append(f"Risk scoring failed: {str(e)}")
    
    def _extract_entities_batch(self, documents: List[Document]) -> List[Entity]:
        entities = []
        
        for document in documents:
            try:
                extraction_result = self.entity_extractor.extract_entities(document)
                entities.extend(extraction_result.entities)
            except Exception as e:
                self.logger.warning(f"Entity extraction failed for document {document.id}: {e}")
        
        return entities
    
    def _extract_relationships_batch(self, 
                                   documents: List[Document],
                                   entities_by_doc: Dict[str, List[Entity]]) -> List[Relationship]:
        relationships = []
        
        for document in documents:
            try:
                doc_entities = entities_by_doc.get(document.id, [])
                if doc_entities:
                    extraction_result = self.relationship_extractor.extract_relationships(
                        document, doc_entities
                    )
                    relationships.extend(extraction_result.relationships)
            except Exception as e:
                self.logger.warning(f"Relationship extraction failed for document {document.id}: {e}")
        
        return relationships
    
    def _get_top_centrality_nodes(self, centrality_results: Dict[str, Dict[str, float]], 
                                 top_n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        top_nodes = {}
        
        for metric, scores in centrality_results.items():
            if scores:
                sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_nodes[metric] = [
                    {'node_id': node_id, 'score': score}
                    for node_id, score in sorted_nodes[:top_n]
                ]
        
        return top_nodes
    
    def configure_pipeline(self, 
                          enable_network_analysis: bool = True,
                          enable_risk_scoring: bool = True,
                          max_documents_per_run: int = 1000,
                          batch_size: int = 50):
        self.enable_network_analysis = enable_network_analysis
        self.enable_risk_scoring = enable_risk_scoring
        self.max_documents_per_run = max_documents_per_run
        self.batch_size = batch_size
        
        self.logger.info(f"Pipeline configured: network_analysis={enable_network_analysis}, "
                        f"risk_scoring={enable_risk_scoring}, max_docs={max_documents_per_run}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        return {
            'graph_type': self.graph_type.value,
            'enable_network_analysis': self.enable_network_analysis,
            'enable_risk_scoring': self.enable_risk_scoring,
            'max_documents_per_run': self.max_documents_per_run,
            'batch_size': self.batch_size,
            'components': {
                'ingestion_pipeline': type(self.ingestion_pipeline).__name__,
                'entity_extractor': type(self.entity_extractor).__name__,
                'relationship_extractor': type(self.relationship_extractor).__name__,
                'graph_constructor': type(self.graph_constructor).__name__,
                'network_analyzer': type(self.network_analyzer).__name__,
                'risk_scorer': type(self.risk_scorer).__name__
            }
        } 