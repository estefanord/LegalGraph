
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from enum import Enum
import networkx as nx

from ..extraction.models import Entity, Relationship


class GraphType(Enum):
    UNDIRECTED = "undirected"
    DIRECTED = "directed"
    MULTI_DIRECTED = "multi_directed"


@dataclass
class GraphNode:
    
    entity_id: str
    entity_type: str
    canonical_name: str
    
    # Node attributes
    centrality_scores: Dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.0
    risk_factors: Dict[str, float] = field(default_factory=dict)
    
    # Connection information
    degree: int = 0
    in_degree: int = 0
    out_degree: int = 0
    
    # Temporal information
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "canonical_name": self.canonical_name,
            "centrality_scores": self.centrality_scores,
            "risk_score": self.risk_score,
            "risk_factors": self.risk_factors,
            "degree": self.degree,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "attributes": self.attributes
        }


@dataclass
class GraphEdge:
    
    source_id: str
    target_id: str
    relationship_type: str
    
    # Edge weights and scores
    weight: float = 1.0
    confidence: float = 0.0
    strength: float = 1.0
    
    # Financial information
    amount: Optional[float] = None
    currency: str = "USD"
    
    # Temporal information
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_ongoing: bool = True
    
    # Evidence and provenance
    evidence_count: int = 1
    source_documents: List[str] = field(default_factory=list)
    extraction_methods: Set[str] = field(default_factory=set)
    
    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "weight": self.weight,
            "confidence": self.confidence,
            "strength": self.strength,
            "amount": self.amount,
            "currency": self.currency,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "is_ongoing": self.is_ongoing,
            "evidence_count": self.evidence_count,
            "source_documents": self.source_documents,
            "extraction_methods": list(self.extraction_methods),
            "attributes": self.attributes
        }


@dataclass
class GraphStats:
    
    # Basic counts
    total_nodes: int = 0
    total_edges: int = 0
    total_components: int = 0
    
    # Node type distribution
    node_type_counts: Dict[str, int] = field(default_factory=dict)
    
    # Edge type distribution
    edge_type_counts: Dict[str, int] = field(default_factory=dict)
    
    # Network properties
    density: float = 0.0
    average_degree: float = 0.0
    clustering_coefficient: float = 0.0
    
    # Centrality statistics
    max_degree_centrality: float = 0.0
    max_betweenness_centrality: float = 0.0
    max_closeness_centrality: float = 0.0
    max_eigenvector_centrality: float = 0.0
    
    # Risk statistics
    average_risk_score: float = 0.0
    high_risk_nodes: int = 0
    max_risk_score: float = 0.0
    
    # Temporal information
    earliest_relationship: Optional[datetime] = None
    latest_relationship: Optional[datetime] = None
    
    # Financial information
    total_financial_relationships: int = 0
    total_settlement_amount: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "total_components": self.total_components,
            "node_type_counts": self.node_type_counts,
            "edge_type_counts": self.edge_type_counts,
            "density": self.density,
            "average_degree": self.average_degree,
            "clustering_coefficient": self.clustering_coefficient,
            "max_degree_centrality": self.max_degree_centrality,
            "max_betweenness_centrality": self.max_betweenness_centrality,
            "max_closeness_centrality": self.max_closeness_centrality,
            "max_eigenvector_centrality": self.max_eigenvector_centrality,
            "average_risk_score": self.average_risk_score,
            "high_risk_nodes": self.high_risk_nodes,
            "max_risk_score": self.max_risk_score,
            "earliest_relationship": self.earliest_relationship.isoformat() if self.earliest_relationship else None,
            "latest_relationship": self.latest_relationship.isoformat() if self.latest_relationship else None,
            "total_financial_relationships": self.total_financial_relationships,
            "total_settlement_amount": self.total_settlement_amount
        }


class LegalGraph:
    
    def __init__(self, graph_type: GraphType = GraphType.DIRECTED):
        self.graph_type = graph_type
        
        # Create appropriate NetworkX graph
        if graph_type == GraphType.UNDIRECTED:
            self.nx_graph = nx.Graph()
        elif graph_type == GraphType.DIRECTED:
            self.nx_graph = nx.DiGraph()
        else:  # MULTI_DIRECTED
            self.nx_graph = nx.MultiDiGraph()
        
        # Metadata
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.metadata = {}
        
        # Caches
        self._centrality_cache = {}
        self._stats_cache = None
        self._last_cache_update = None
    
    def add_node(self, node: GraphNode) -> None:
        self.nx_graph.add_node(
            node.entity_id,
            **node.to_dict()
        )
        self._invalidate_cache()
    
    def add_edge(self, edge: GraphEdge) -> None:
        edge_data = edge.to_dict()
        
        if isinstance(self.nx_graph, nx.MultiDiGraph):
            # For multigraph, we can have multiple edges between nodes
            self.nx_graph.add_edge(
                edge.source_id,
                edge.target_id,
                key=f"{edge.relationship_type}_{len(self.nx_graph.edges(edge.source_id, edge.target_id))}",
                **edge_data
            )
        else:
            # For simple graphs, merge edge data if edge already exists
            if self.nx_graph.has_edge(edge.source_id, edge.target_id):
                existing_data = self.nx_graph[edge.source_id][edge.target_id]
                
                # Merge evidence count and documents
                existing_data['evidence_count'] = existing_data.get('evidence_count', 0) + edge.evidence_count
                existing_docs = set(existing_data.get('source_documents', []))
                existing_docs.update(edge.source_documents)
                existing_data['source_documents'] = list(existing_docs)
                
                # Update weight based on evidence
                existing_data['weight'] = max(existing_data.get('weight', 1.0), edge.weight)
                
                # Merge extraction methods
                existing_methods = set(existing_data.get('extraction_methods', []))
                existing_methods.update(edge.extraction_methods)
                existing_data['extraction_methods'] = list(existing_methods)
            else:
                self.nx_graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    **edge_data
                )
        
        self._invalidate_cache()
    
    def get_node(self, entity_id: str) -> Optional[GraphNode]:
        if entity_id not in self.nx_graph.nodes:
            return None
        
        node_data = self.nx_graph.nodes[entity_id]
        return GraphNode(
            entity_id=entity_id,
            entity_type=node_data.get('entity_type', ''),
            canonical_name=node_data.get('canonical_name', ''),
            centrality_scores=node_data.get('centrality_scores', {}),
            risk_score=node_data.get('risk_score', 0.0),
            risk_factors=node_data.get('risk_factors', {}),
            degree=node_data.get('degree', 0),
            in_degree=node_data.get('in_degree', 0),
            out_degree=node_data.get('out_degree', 0),
            attributes=node_data.get('attributes', {})
        )
    
    def get_neighbors(self, entity_id: str, radius: int = 1) -> List[str]:
        if entity_id not in self.nx_graph.nodes:
            return []
        
        if radius == 1:
            return list(self.nx_graph.neighbors(entity_id))
        
        # For multiple hops, use BFS
        neighbors = set()
        current_level = {entity_id}
        
        for _ in range(radius):
            next_level = set()
            for node in current_level:
                for neighbor in self.nx_graph.neighbors(node):
                    if neighbor != entity_id:  # Exclude the original node
                        neighbors.add(neighbor)
                        next_level.add(neighbor)
            current_level = next_level
        
        return list(neighbors)
    
    def get_subgraph(self, entity_ids: List[str]) -> 'LegalGraph':
        subgraph_nx = self.nx_graph.subgraph(entity_ids).copy()
        
        # Create new LegalGraph with same type
        subgraph = LegalGraph(self.graph_type)
        subgraph.nx_graph = subgraph_nx
        
        return subgraph
    
    def get_connected_components(self) -> List[List[str]]:
        if self.nx_graph.is_directed():
            components = nx.weakly_connected_components(self.nx_graph)
        else:
            components = nx.connected_components(self.nx_graph)
        
        return [list(component) for component in components]
    
    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        try:
            return nx.shortest_path(self.nx_graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_statistics(self) -> GraphStats:
        if self._stats_cache and self._cache_is_valid():
            return self._stats_cache
        
        stats = GraphStats()
        
        # Basic counts
        stats.total_nodes = self.nx_graph.number_of_nodes()
        stats.total_edges = self.nx_graph.number_of_edges()
        
        if stats.total_nodes == 0:
            self._stats_cache = stats
            return stats
        
        # Node type distribution
        node_types = {}
        risk_scores = []
        
        for node_id, node_data in self.nx_graph.nodes(data=True):
            entity_type = node_data.get('entity_type', 'unknown')
            node_types[entity_type] = node_types.get(entity_type, 0) + 1
            risk_scores.append(node_data.get('risk_score', 0.0))
        
        stats.node_type_counts = node_types
        
        # Edge type distribution
        edge_types = {}
        financial_relationships = 0
        settlement_amounts = []
        
        for source, target, edge_data in self.nx_graph.edges(data=True):
            relationship_type = edge_data.get('relationship_type', 'unknown')
            edge_types[relationship_type] = edge_types.get(relationship_type, 0) + 1
            
            if edge_data.get('amount'):
                financial_relationships += 1
                settlement_amounts.append(edge_data['amount'])
        
        stats.edge_type_counts = edge_types
        stats.total_financial_relationships = financial_relationships
        stats.total_settlement_amount = sum(settlement_amounts) if settlement_amounts else 0.0
        
        # Network properties
        if stats.total_nodes > 1:
            stats.density = nx.density(self.nx_graph)
            stats.average_degree = sum(dict(self.nx_graph.degree()).values()) / stats.total_nodes
            
            # Clustering coefficient (for undirected graphs)
            if not self.nx_graph.is_directed():
                stats.clustering_coefficient = nx.average_clustering(self.nx_graph)
        
        # Connected components
        stats.total_components = len(self.get_connected_components())
        
        # Centrality statistics (if graph is not too large)
        if stats.total_nodes <= 1000:  # Avoid expensive calculations on very large graphs
            try:
                degree_centrality = nx.degree_centrality(self.nx_graph)
                stats.max_degree_centrality = max(degree_centrality.values()) if degree_centrality else 0.0
                
                if stats.total_nodes <= 500:  # More expensive centrality measures
                    betweenness_centrality = nx.betweenness_centrality(self.nx_graph)
                    stats.max_betweenness_centrality = max(betweenness_centrality.values()) if betweenness_centrality else 0.0
                    
                    if not self.nx_graph.is_directed():
                        closeness_centrality = nx.closeness_centrality(self.nx_graph)
                        stats.max_closeness_centrality = max(closeness_centrality.values()) if closeness_centrality else 0.0
                        
                        eigenvector_centrality = nx.eigenvector_centrality(self.nx_graph, max_iter=1000)
                        stats.max_eigenvector_centrality = max(eigenvector_centrality.values()) if eigenvector_centrality else 0.0
            except:
                pass  # Skip centrality calculations if they fail
        
        # Risk statistics
        if risk_scores:
            stats.average_risk_score = sum(risk_scores) / len(risk_scores)
            stats.max_risk_score = max(risk_scores)
            stats.high_risk_nodes = sum(1 for score in risk_scores if score > 0.7)
        
        # Cache the results
        self._stats_cache = stats
        self._last_cache_update = datetime.utcnow()
        
        return stats
    
    def _invalidate_cache(self):
        self._centrality_cache.clear()
        self._stats_cache = None
        self._last_cache_update = None
        self.updated_at = datetime.utcnow()
    
    def _cache_is_valid(self, max_age_seconds: int = 300) -> bool:
        if not self._last_cache_update:
            return False
        
        age = (datetime.utcnow() - self._last_cache_update).total_seconds()
        return age < max_age_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_type": self.graph_type.value,
            "nodes": [
                {**data, "id": node_id} for node_id, data in self.nx_graph.nodes(data=True)
            ],
            "edges": [
                {**data, "source": source, "target": target} 
                for source, target, data in self.nx_graph.edges(data=True)
            ],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "statistics": self.get_statistics().to_dict()
        }
    
    def __len__(self) -> int:
        return self.nx_graph.number_of_nodes()
    
    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self.nx_graph.nodes 