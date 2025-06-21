
import time
from datetime import datetime
from typing import List, Dict, Optional, Set
from collections import defaultdict

from .models import LegalGraph, GraphNode, GraphEdge, GraphType
from ..extraction.models import Entity, Relationship, EntityType, RelationType
from ..core.config import settings
from ..core.logging import get_logger
from ..core.exceptions import GraphConstructionError


class GraphConstructor:
    
    def __init__(self, graph_type: GraphType = GraphType.DIRECTED):
        self.logger = get_logger("legalgraph.graph.constructor")
        self.graph_type = graph_type
        
        # Entity filtering criteria
        self.min_confidence = 0.5
        self.include_entity_types = {
            EntityType.PERSON,
            EntityType.COMPANY,
            EntityType.GOVERNMENT_AGENCY,
            EntityType.LAW_FIRM,
            EntityType.COURT,
            EntityType.FINANCIAL_INSTITUTION,
            EntityType.INVESTMENT_FIRM
        }
        
        # Relationship filtering criteria
        self.min_relationship_confidence = 0.4
        self.include_relationship_types = {
            RelationType.ENFORCEMENT_ACTION,
            RelationType.SETTLEMENT,
            RelationType.VIOLATION_OF,
            RelationType.CHARGED_WITH,
            RelationType.REPRESENTED_BY,
            RelationType.EMPLOYED_BY,
            RelationType.SUBSIDIARY_OF,
            RelationType.REGULATED_BY,
            RelationType.PROSECUTED_BY
        }
    
    def construct_graph(self, entities: List[Entity], 
                       relationships: List[Relationship]) -> LegalGraph:
        start_time = time.time()
        
        self.logger.info(f"Constructing graph from {len(entities)} entities and {len(relationships)} relationships")
        
        try:
            # Create new graph
            graph = LegalGraph(self.graph_type)
            
            # Filter and add entities as nodes
            filtered_entities = self._filter_entities(entities)
            entity_lookup = {}
            
            for entity in filtered_entities:
                node = self._create_graph_node(entity)
                graph.add_node(node)
                entity_lookup[entity.id] = entity
            
            self.logger.info(f"Added {len(filtered_entities)} nodes to graph")
            
            # Filter and add relationships as edges
            filtered_relationships = self._filter_relationships(relationships, entity_lookup)
            
            for relationship in filtered_relationships:
                # Only add edge if both entities are in the graph
                if (relationship.source_entity_id in entity_lookup and 
                    relationship.target_entity_id in entity_lookup):
                    
                    edge = self._create_graph_edge(relationship)
                    graph.add_edge(edge)
            
            self.logger.info(f"Added {len(filtered_relationships)} edges to graph")
            
            # Post-process graph
            self._post_process_graph(graph)
            
            construction_time = time.time() - start_time
            
            # Add metadata
            graph.metadata.update({
                'construction_time': construction_time,
                'source_entities': len(entities),
                'source_relationships': len(relationships),
                'filtered_entities': len(filtered_entities),
                'filtered_relationships': len(filtered_relationships),
                'entity_types_included': [et.value for et in self.include_entity_types],
                'relationship_types_included': [rt.value for rt in self.include_relationship_types]
            })
            
            self.logger.info(
                f"Graph construction completed in {construction_time:.2f}s: "
                f"{len(graph)} nodes, {len(graph.nx_graph.edges())} edges"
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Graph construction failed: {e}")
            raise GraphConstructionError(f"Failed to construct graph: {e}") from e
    
    def merge_graphs(self, graphs: List[LegalGraph]) -> LegalGraph:
        if not graphs:
            return LegalGraph(self.graph_type)
        
        if len(graphs) == 1:
            return graphs[0]
        
        self.logger.info(f"Merging {len(graphs)} graphs")
        
        # Create new merged graph
        merged_graph = LegalGraph(self.graph_type)
        
        # Track entities to avoid duplicates
        seen_entities = set()
        entity_mapping = {}  # old_id -> new_id
        
        # Merge nodes
        for graph in graphs:
            for node_id, node_data in graph.nx_graph.nodes(data=True):
                canonical_name = node_data.get('canonical_name', '')
                entity_type = node_data.get('entity_type', '')
                
                # Create unique key for entity
                entity_key = (canonical_name.lower(), entity_type)
                
                if entity_key in seen_entities:
                    # Find the existing node with this canonical name and type
                    for existing_node_id, existing_data in merged_graph.nx_graph.nodes(data=True):
                        if (existing_data.get('canonical_name', '').lower() == canonical_name.lower() and 
                            existing_data.get('entity_type', '') == entity_type):
                            entity_mapping[node_id] = existing_node_id
                            # Merge node data (update mention counts, etc.)
                            self._merge_node_data(merged_graph.nx_graph.nodes[existing_node_id], node_data)
                            break
                else:
                    # Add new node
                    merged_graph.nx_graph.add_node(node_id, **node_data)
                    entity_mapping[node_id] = node_id
                    seen_entities.add(entity_key)
        
        # Merge edges
        for graph in graphs:
            for source, target, edge_data in graph.nx_graph.edges(data=True):
                # Map to merged node IDs
                merged_source = entity_mapping.get(source, source)
                merged_target = entity_mapping.get(target, target)
                
                # Skip self-loops that might have been created by merging
                if merged_source == merged_target:
                    continue
                
                # Add or merge edge
                if merged_graph.nx_graph.has_edge(merged_source, merged_target):
                    # Merge edge data
                    existing_edge = merged_graph.nx_graph[merged_source][merged_target]
                    self._merge_edge_data(existing_edge, edge_data)
                else:
                    merged_graph.nx_graph.add_edge(merged_source, merged_target, **edge_data)
        
        # Update metadata
        merged_graph.metadata['merged_from'] = len(graphs)
        merged_graph.metadata['merge_timestamp'] = datetime.utcnow().isoformat()
        
        self.logger.info(
            f"Graph merge completed: {len(merged_graph)} nodes, "
            f"{len(merged_graph.nx_graph.edges())} edges"
        )
        
        return merged_graph
    
    def _filter_entities(self, entities: List[Entity]) -> List[Entity]:
        filtered = []
        
        for entity in entities:
            # Check confidence threshold
            if entity.confidence < self.min_confidence:
                continue
            
            # Check entity type
            if entity.entity_type not in self.include_entity_types:
                continue
            
            # Additional quality checks
            if not entity.canonical_name or len(entity.canonical_name.strip()) < 2:
                continue
            
            filtered.append(entity)
        
        self.logger.debug(f"Filtered entities: {len(entities)} -> {len(filtered)}")
        return filtered
    
    def _filter_relationships(self, relationships: List[Relationship], 
                            entity_lookup: Dict[str, Entity]) -> List[Relationship]:
        filtered = []
        
        for relationship in relationships:
            # Check confidence threshold
            if relationship.confidence < self.min_relationship_confidence:
                continue
            
            # Check relationship type
            if relationship.relationship_type not in self.include_relationship_types:
                continue
            
            # Check that both entities exist in our filtered set
            if (relationship.source_entity_id not in entity_lookup or 
                relationship.target_entity_id not in entity_lookup):
                continue
            
            # Skip self-relationships
            if relationship.source_entity_id == relationship.target_entity_id:
                continue
            
            filtered.append(relationship)
        
        self.logger.debug(f"Filtered relationships: {len(relationships)} -> {len(filtered)}")
        return filtered
    
    def _create_graph_node(self, entity: Entity) -> GraphNode:
        return GraphNode(
            entity_id=entity.id,
            entity_type=entity.entity_type.value,
            canonical_name=entity.canonical_name,
            first_seen=entity.created_at,
            last_seen=entity.updated_at,
            attributes={
                'original_text': entity.text,
                'confidence': entity.confidence,
                'extraction_method': entity.extraction_method,
                'mention_count': entity.mention_count,
                'aliases': entity.aliases,
                'source_attributes': entity.attributes
            }
        )
    
    def _create_graph_edge(self, relationship: Relationship) -> GraphEdge:
        # Calculate edge weight based on confidence and evidence
        weight = relationship.confidence
        if relationship.strength:
            weight *= relationship.strength
        
        return GraphEdge(
            source_id=relationship.source_entity_id,
            target_id=relationship.target_entity_id,
            relationship_type=relationship.relationship_type.value,
            weight=weight,
            confidence=relationship.confidence,
            strength=relationship.strength,
            amount=relationship.amount,
            currency=relationship.currency,
            start_date=relationship.start_date,
            end_date=relationship.end_date,
            is_ongoing=relationship.is_ongoing,
            evidence_count=1,
            source_documents=[relationship.source_document_id] if relationship.source_document_id else [],
            extraction_methods={relationship.extraction_method} if relationship.extraction_method else set(),
            attributes={
                'description': relationship.description,
                'evidence_text': relationship.evidence_text,
                'source_attributes': relationship.attributes
            }
        )
    
    def _post_process_graph(self, graph: LegalGraph):
        # Calculate node degrees
        for node_id in graph.nx_graph.nodes():
            node_data = graph.nx_graph.nodes[node_id]
            
            if graph.nx_graph.is_directed():
                node_data['in_degree'] = graph.nx_graph.in_degree(node_id)
                node_data['out_degree'] = graph.nx_graph.out_degree(node_id)
                node_data['degree'] = graph.nx_graph.degree(node_id)
            else:
                degree = graph.nx_graph.degree(node_id)
                node_data['degree'] = degree
                node_data['in_degree'] = degree
                node_data['out_degree'] = degree
        
        # Calculate basic centrality measures for small graphs
        if len(graph) <= 500:
            try:
                self._calculate_centralities(graph)
            except Exception as e:
                self.logger.warning(f"Failed to calculate centralities: {e}")
        
        # Group nodes by entity type for analysis
        self._analyze_entity_clusters(graph)
    
    def _calculate_centralities(self, graph: LegalGraph):
        import networkx as nx
        
        # Degree centrality
        degree_centrality = nx.degree_centrality(graph.nx_graph)
        
        # Betweenness centrality (for smaller graphs)
        betweenness_centrality = {}
        if len(graph) <= 200:
            betweenness_centrality = nx.betweenness_centrality(graph.nx_graph)
        
        # Closeness centrality (for undirected graphs)
        closeness_centrality = {}
        if not graph.nx_graph.is_directed() and len(graph) <= 200:
            try:
                closeness_centrality = nx.closeness_centrality(graph.nx_graph)
            except:
                pass  # May fail for disconnected graphs
        
        # Update node data with centrality scores
        for node_id in graph.nx_graph.nodes():
            centrality_scores = {}
            
            if node_id in degree_centrality:
                centrality_scores['degree'] = degree_centrality[node_id]
            
            if node_id in betweenness_centrality:
                centrality_scores['betweenness'] = betweenness_centrality[node_id]
            
            if node_id in closeness_centrality:
                centrality_scores['closeness'] = closeness_centrality[node_id]
            
            graph.nx_graph.nodes[node_id]['centrality_scores'] = centrality_scores
        
        self.logger.debug("Calculated centrality measures for graph nodes")
    
    def _analyze_entity_clusters(self, graph: LegalGraph):
        entity_clusters = defaultdict(list)
        
        for node_id, node_data in graph.nx_graph.nodes(data=True):
            entity_type = node_data.get('entity_type', 'unknown')
            entity_clusters[entity_type].append(node_id)
        
        # Store cluster information in graph metadata
        graph.metadata['entity_clusters'] = {
            entity_type: len(nodes) for entity_type, nodes in entity_clusters.items()
        }
        
        # Find inter-cluster connections
        inter_cluster_edges = 0
        intra_cluster_edges = 0
        
        for source, target, edge_data in graph.nx_graph.edges(data=True):
            source_type = graph.nx_graph.nodes[source].get('entity_type', 'unknown')
            target_type = graph.nx_graph.nodes[target].get('entity_type', 'unknown')
            
            if source_type == target_type:
                intra_cluster_edges += 1
            else:
                inter_cluster_edges += 1
        
        graph.metadata['cluster_analysis'] = {
            'inter_cluster_edges': inter_cluster_edges,
            'intra_cluster_edges': intra_cluster_edges,
            'clustering_ratio': inter_cluster_edges / (inter_cluster_edges + intra_cluster_edges) if (inter_cluster_edges + intra_cluster_edges) > 0 else 0
        }
    
    def _merge_node_data(self, existing_data: Dict, new_data: Dict):
        # Merge mention counts
        existing_mention_count = existing_data.get('attributes', {}).get('mention_count', 1)
        new_mention_count = new_data.get('attributes', {}).get('mention_count', 1)
        
        if 'attributes' not in existing_data:
            existing_data['attributes'] = {}
        
        existing_data['attributes']['mention_count'] = existing_mention_count + new_mention_count
        
        # Merge aliases
        existing_aliases = set(existing_data.get('attributes', {}).get('aliases', []))
        new_aliases = set(new_data.get('attributes', {}).get('aliases', []))
        existing_data['attributes']['aliases'] = list(existing_aliases | new_aliases)
        
        # Update confidence to higher value
        existing_conf = existing_data.get('attributes', {}).get('confidence', 0.0)
        new_conf = new_data.get('attributes', {}).get('confidence', 0.0)
        existing_data['attributes']['confidence'] = max(existing_conf, new_conf)
        
        # Update timestamps
        existing_first = existing_data.get('first_seen')
        new_first = new_data.get('first_seen')
        if new_first and (not existing_first or new_first < existing_first):
            existing_data['first_seen'] = new_first
        
        existing_last = existing_data.get('last_seen')
        new_last = new_data.get('last_seen')
        if new_last and (not existing_last or new_last > existing_last):
            existing_data['last_seen'] = new_last
    
    def _merge_edge_data(self, existing_data: Dict, new_data: Dict):
        # Increment evidence count
        existing_data['evidence_count'] = existing_data.get('evidence_count', 1) + new_data.get('evidence_count', 1)
        
        # Merge source documents
        existing_docs = set(existing_data.get('source_documents', []))
        new_docs = set(new_data.get('source_documents', []))
        existing_data['source_documents'] = list(existing_docs | new_docs)
        
        # Merge extraction methods
        existing_methods = set(existing_data.get('extraction_methods', []))
        new_methods = set(new_data.get('extraction_methods', []))
        existing_data['extraction_methods'] = list(existing_methods | new_methods)
        
        # Update confidence and weight
        existing_data['confidence'] = max(existing_data.get('confidence', 0.0), new_data.get('confidence', 0.0))
        existing_data['weight'] = max(existing_data.get('weight', 1.0), new_data.get('weight', 1.0))
        
        # Merge financial information (take higher amount if different)
        if new_data.get('amount') and new_data['amount'] > existing_data.get('amount', 0):
            existing_data['amount'] = new_data['amount']
            existing_data['currency'] = new_data.get('currency', 'USD')
    
    def set_filtering_criteria(self, min_entity_confidence: Optional[float] = None,
                             min_relationship_confidence: Optional[float] = None,
                             include_entity_types: Optional[Set[EntityType]] = None,
                             include_relationship_types: Optional[Set[RelationType]] = None):
        if min_entity_confidence is not None:
            self.min_confidence = min_entity_confidence
        
        if min_relationship_confidence is not None:
            self.min_relationship_confidence = min_relationship_confidence
        
        if include_entity_types is not None:
            self.include_entity_types = include_entity_types
        
        if include_relationship_types is not None:
            self.include_relationship_types = include_relationship_types
        
        self.logger.info("Updated graph construction filtering criteria") 