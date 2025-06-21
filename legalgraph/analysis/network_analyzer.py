import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
import math

from ..graph.models import LegalGraph


class NetworkAnalyzer:
    
    def __init__(self):
        self.centrality_algorithms = {
            'degree': nx.degree_centrality,
            'betweenness': nx.betweenness_centrality,
            'closeness': nx.closeness_centrality,
            'eigenvector': nx.eigenvector_centrality,
            'pagerank': nx.pagerank
        }
    
    def analyze_network(self, knowledge_graph: LegalGraph) -> Dict[str, Any]:
        
        # Convert to NetworkX graph for analysis
        nx_graph = self._convert_to_networkx(knowledge_graph)
        
        if nx_graph.number_of_nodes() == 0:
            return self._empty_analysis()
        
        # Basic network statistics
        stats = {
            'node_count': nx_graph.number_of_nodes(),
            'edge_count': nx_graph.number_of_edges(),
            'density': nx.density(nx_graph),
            'is_connected': nx.is_connected(nx_graph) if nx_graph.number_of_nodes() > 0 else False,
            'connected_components': nx.number_connected_components(nx_graph),
            'average_clustering': nx.average_clustering(nx_graph),
        }
        
        # Calculate diameter and average path length for connected graphs
        if stats['is_connected']:
            stats['diameter'] = nx.diameter(nx_graph)
            stats['average_path_length'] = nx.average_shortest_path_length(nx_graph)
        else:
            # For disconnected graphs, calculate for largest component
            largest_cc = max(nx.connected_components(nx_graph), key=len, default=set())
            if len(largest_cc) > 1:
                largest_subgraph = nx_graph.subgraph(largest_cc)
                stats['largest_component_diameter'] = nx.diameter(largest_subgraph)
                stats['largest_component_avg_path_length'] = nx.average_shortest_path_length(largest_subgraph)
                stats['largest_component_size'] = len(largest_cc)
        
        # Degree distribution
        degrees = [d for n, d in nx_graph.degree()]
        if degrees:
            stats['degree_distribution'] = {
                'min': min(degrees),
                'max': max(degrees),
                'mean': sum(degrees) / len(degrees),
                'median': sorted(degrees)[len(degrees) // 2]
            }
        
        # Centrality measures
        stats['centrality'] = self._calculate_all_centralities(nx_graph)
        
        # Community detection
        stats['communities'] = self._detect_communities(nx_graph)
        
        # Network motifs and patterns
        stats['motifs'] = self._analyze_motifs(nx_graph)
        
        # Hub analysis
        stats['hubs'] = self._identify_hubs(nx_graph)
        
        return stats
    
    def calculate_centrality(self, knowledge_graph: LegalGraph, 
                           algorithm: str = 'pagerank') -> List[Tuple[str, float]]:
        
        nx_graph = self._convert_to_networkx(knowledge_graph)
        
        if nx_graph.number_of_nodes() == 0:
            return []
        
        if algorithm not in self.centrality_algorithms:
            raise ValueError(f"Unknown centrality algorithm: {algorithm}")
        
        try:
            centrality_func = self.centrality_algorithms[algorithm]
            centrality_scores = centrality_func(nx_graph)
            
            # Sort by score descending
            sorted_scores = sorted(centrality_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
            
            return sorted_scores
            
        except Exception as e:
            # Fallback to degree centrality if other algorithms fail
            if algorithm != 'degree':
                return self.calculate_centrality(knowledge_graph, 'degree')
            else:
                return []
    
    def find_shortest_paths(self, knowledge_graph: LegalGraph, 
                          source: str, target: str) -> List[List[str]]:
        
        nx_graph = self._convert_to_networkx(knowledge_graph)
        
        try:
            # Find all shortest paths
            paths = list(nx.all_shortest_paths(nx_graph, source, target))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def identify_bridges(self, knowledge_graph: LegalGraph) -> List[Tuple[str, str]]:
        
        nx_graph = self._convert_to_networkx(knowledge_graph)
        
        try:
            bridges = list(nx.bridges(nx_graph))
            return bridges
        except:
            return []
    
    def detect_anomalies(self, knowledge_graph: LegalGraph) -> Dict[str, List[str]]:
        
        nx_graph = self._convert_to_networkx(knowledge_graph)
        anomalies = {
            'isolated_nodes': [],
            'high_degree_nodes': [],
            'bridge_nodes': [],
            'outlier_nodes': []
        }
        
        if nx_graph.number_of_nodes() == 0:
            return anomalies
        
        # Isolated nodes (degree 0)
        anomalies['isolated_nodes'] = [n for n, d in nx_graph.degree() if d == 0]
        
        # High degree nodes (potential hubs)
        degrees = [d for n, d in nx_graph.degree()]
        if degrees:
            mean_degree = sum(degrees) / len(degrees)
            std_degree = (sum((d - mean_degree) ** 2 for d in degrees) / len(degrees)) ** 0.5
            threshold = mean_degree + 2 * std_degree  # 2 standard deviations
            
            anomalies['high_degree_nodes'] = [n for n, d in nx_graph.degree() 
                                            if d > threshold]
        
        # Bridge nodes (articulation points)
        try:
            anomalies['bridge_nodes'] = list(nx.articulation_points(nx_graph))
        except:
            pass
        
        # Outlier nodes based on centrality
        try:
            pagerank_scores = nx.pagerank(nx_graph)
            scores = list(pagerank_scores.values())
            if scores:
                mean_score = sum(scores) / len(scores)
                std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
                threshold = mean_score + 2 * std_score
                
                anomalies['outlier_nodes'] = [n for n, s in pagerank_scores.items() 
                                            if s > threshold]
        except:
            pass
        
        return anomalies
    
    def _convert_to_networkx(self, legal_graph: LegalGraph) -> nx.Graph:
        
        if hasattr(legal_graph, 'to_networkx'):
            nx_graph = legal_graph.to_networkx()
        else:
            nx_graph = legal_graph
        
        return nx_graph
    
    def _empty_analysis(self) -> Dict[str, Any]:
        
        return {
            'node_count': 0,
            'edge_count': 0,
            'density': 0.0,
            'is_connected': False,
            'connected_components': 0,
            'average_clustering': 0.0,
            'centrality': {},
            'communities': [],
            'motifs': {},
            'hubs': []
        }
    
    def _calculate_all_centralities(self, nx_graph: nx.Graph) -> Dict[str, Dict[str, float]]:
        
        centralities = {}
        
        for name, algorithm in self.centrality_algorithms.items():
            try:
                centralities[name] = algorithm(nx_graph)
            except Exception as e:
                # Some algorithms may fail on certain graph types
                centralities[name] = {}
        
        return centralities
    
    def _detect_communities(self, nx_graph: nx.Graph) -> List[Dict[str, Any]]:
        
        communities = []
        
        try:
            # Use Louvain community detection
            import networkx.algorithms.community as nx_comm
            
            # Greedy modularity communities
            community_sets = nx_comm.greedy_modularity_communities(nx_graph)
            
            for i, community in enumerate(community_sets):
                communities.append({
                    'id': i,
                    'size': len(community),
                    'nodes': list(community),
                    'modularity': None  # Would need to calculate separately
                })
                
        except Exception as e:
            # Fallback: use connected components as communities
            for i, component in enumerate(nx.connected_components(nx_graph)):
                communities.append({
                    'id': i,
                    'size': len(component),
                    'nodes': list(component),
                    'type': 'connected_component'
                })
        
        return communities
    
    def _analyze_motifs(self, nx_graph: nx.Graph) -> Dict[str, int]:
        
        motifs = {
            'triangles': 0,
            'stars': 0,
            'paths': 0,
            'cliques': 0
        }
        
        try:
            # Count triangles
            motifs['triangles'] = sum(nx.triangles(nx_graph).values()) // 3
            
            # Count star motifs (nodes with degree > 2)
            motifs['stars'] = len([n for n, d in nx_graph.degree() if d > 2])
            
            # Count path motifs (simplified)
            motifs['paths'] = nx_graph.number_of_edges()
            
            # Find cliques
            cliques = list(nx.find_cliques(nx_graph))
            motifs['cliques'] = len([c for c in cliques if len(c) >= 3])
            
        except Exception as e:
            pass
        
        return motifs
    
    def _identify_hubs(self, nx_graph: nx.Graph) -> List[Dict[str, Any]]:
        
        hubs = []
        
        if nx_graph.number_of_nodes() == 0:
            return hubs
        
        try:
            # Calculate multiple centrality measures
            degree_centrality = nx.degree_centrality(nx_graph)
            pagerank_centrality = nx.pagerank(nx_graph)
            
            # Combine scores to identify hubs
            combined_scores = {}
            for node in nx_graph.nodes():
                combined_scores[node] = (
                    degree_centrality.get(node, 0) * 0.5 +
                    pagerank_centrality.get(node, 0) * 0.5
                )
            
            # Get top 10% as hubs
            num_hubs = max(1, nx_graph.number_of_nodes() // 10)
            top_nodes = sorted(combined_scores.items(), 
                             key=lambda x: x[1], reverse=True)[:num_hubs]
            
            for node, score in top_nodes:
                node_data = nx_graph.nodes[node]
                hubs.append({
                    'node_id': node,
                    'label': node_data.get('label', node),
                    'type': node_data.get('type', 'unknown'),
                    'degree': nx_graph.degree(node),
                    'hub_score': score,
                    'degree_centrality': degree_centrality.get(node, 0),
                    'pagerank_centrality': pagerank_centrality.get(node, 0)
                })
                
        except Exception as e:
            pass
        
        return hubs
    
    def calculate_network_resilience(self, knowledge_graph: LegalGraph) -> Dict[str, float]:
        
        nx_graph = self._convert_to_networkx(knowledge_graph)
        
        resilience = {
            'connectivity': 0.0,
            'robustness': 0.0,
            'redundancy': 0.0
        }
        
        if nx_graph.number_of_nodes() <= 1:
            return resilience
        
        try:
            # Connectivity: based on number of connected components
            num_components = nx.number_connected_components(nx_graph)
            resilience['connectivity'] = 1.0 / num_components if num_components > 0 else 0.0
            
            # Robustness: simulate random node removal
            original_nodes = nx_graph.number_of_nodes()
            test_graph = nx_graph.copy()
            
            # Remove 10% of nodes randomly
            import random
            nodes_to_remove = random.sample(list(test_graph.nodes()), 
                                          min(original_nodes // 10, original_nodes - 1))
            test_graph.remove_nodes_from(nodes_to_remove)
            
            # Check if main component is still connected
            if test_graph.number_of_nodes() > 0:
                largest_cc_size = len(max(nx.connected_components(test_graph), key=len, default=set()))
                resilience['robustness'] = largest_cc_size / original_nodes
            
            # Redundancy: based on average degree
            avg_degree = sum(d for n, d in nx_graph.degree()) / nx_graph.number_of_nodes()
            resilience['redundancy'] = min(avg_degree / 4.0, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            pass
        
        return resilience 