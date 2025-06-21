import time
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict
from datetime import datetime, timedelta

try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from .models import LegalGraph, GraphStats
from ..core.config import settings
from ..core.logging import get_logger
from ..core.exceptions import GraphAnalysisError


class NetworkAnalyzer:
    
    def __init__(self):
        self.logger = get_logger("legalgraph.graph.analyzer")
        
        if not NETWORKX_AVAILABLE:
            raise GraphAnalysisError("NetworkX is required for graph analysis")
    
    def analyze_centrality(self, graph: LegalGraph) -> Dict[str, Dict[str, float]]:
        start_time = time.time()
        
        self.logger.info(f"Analyzing centrality for graph with {len(graph)} nodes")
        
        centrality_results = {}
        
        try:
            # Degree centrality (always calculate)
            degree_centrality = nx.degree_centrality(graph.nx_graph)
            centrality_results['degree'] = degree_centrality
            
            # Betweenness centrality (for smaller graphs)
            if len(graph) <= 1000:
                betweenness_centrality = nx.betweenness_centrality(graph.nx_graph)
                centrality_results['betweenness'] = betweenness_centrality
            
            # Closeness centrality (for connected components)
            if len(graph) <= 500:
                try:
                    if graph.nx_graph.is_directed():
                        # For directed graphs, use weakly connected components
                        components = list(nx.weakly_connected_components(graph.nx_graph))
                    else:
                        components = list(nx.connected_components(graph.nx_graph))
                    
                    closeness_centrality = {}
                    for component in components:
                        if len(component) > 1:
                            subgraph = graph.nx_graph.subgraph(component)
                            component_closeness = nx.closeness_centrality(subgraph)
                            closeness_centrality.update(component_closeness)
                    
                    centrality_results['closeness'] = closeness_centrality
                except Exception as e:
                    self.logger.warning(f"Closeness centrality calculation failed: {e}")
            
            # Eigenvector centrality (for undirected graphs)
            if not graph.nx_graph.is_directed() and len(graph) <= 500:
                try:
                    eigenvector_centrality = nx.eigenvector_centrality(
                        graph.nx_graph, max_iter=1000, tol=1e-06
                    )
                    centrality_results['eigenvector'] = eigenvector_centrality
                except Exception as e:
                    self.logger.warning(f"Eigenvector centrality calculation failed: {e}")
            
            # PageRank (for directed graphs)
            if graph.nx_graph.is_directed() and len(graph) <= 1000:
                try:
                    pagerank = nx.pagerank(graph.nx_graph, max_iter=1000, tol=1e-06)
                    centrality_results['pagerank'] = pagerank
                except Exception as e:
                    self.logger.warning(f"PageRank calculation failed: {e}")
            
            # Update graph nodes with centrality scores
            self._update_node_centralities(graph, centrality_results)
            
            analysis_time = time.time() - start_time
            self.logger.info(f"Centrality analysis completed in {analysis_time:.2f}s")
            
            return centrality_results
            
        except Exception as e:
            self.logger.error(f"Centrality analysis failed: {e}")
            raise GraphAnalysisError(f"Failed to analyze centrality: {e}") from e
    
    def detect_communities(self, graph: LegalGraph) -> Dict[str, Any]:
        start_time = time.time()
        
        self.logger.info(f"Detecting communities in graph with {len(graph)} nodes")
        
        try:
            # Convert to undirected for community detection if needed
            if graph.nx_graph.is_directed():
                undirected_graph = graph.nx_graph.to_undirected()
            else:
                undirected_graph = graph.nx_graph
            
            communities_result = {
                'communities': [],
                'modularity': 0.0,
                'algorithm': 'louvain',
                'num_communities': 0
            }
            
            if len(graph) < 3:
                return communities_result
            
            # Use Louvain algorithm for community detection
            try:
                communities = community.louvain_communities(undirected_graph)
                communities_result['communities'] = [list(comm) for comm in communities]
                communities_result['num_communities'] = len(communities)
                
                # Calculate modularity
                communities_result['modularity'] = community.modularity(
                    undirected_graph, communities
                )
                
            except Exception as e:
                self.logger.warning(f"Louvain community detection failed: {e}")
                
                # Fallback to simple connected components
                if graph.nx_graph.is_directed():
                    components = list(nx.weakly_connected_components(graph.nx_graph))
                else:
                    components = list(nx.connected_components(graph.nx_graph))
                
                communities_result['communities'] = [list(comp) for comp in components]
                communities_result['num_communities'] = len(components)
                communities_result['algorithm'] = 'connected_components'
            
            # Analyze community characteristics
            community_analysis = self._analyze_communities(graph, communities_result['communities'])
            communities_result.update(community_analysis)
            
            analysis_time = time.time() - start_time
            self.logger.info(
                f"Community detection completed in {analysis_time:.2f}s: "
                f"{communities_result['num_communities']} communities found"
            )
            
            return communities_result
            
        except Exception as e:
            self.logger.error(f"Community detection failed: {e}")
            raise GraphAnalysisError(f"Failed to detect communities: {e}") from e
    
    def find_critical_paths(self, graph: LegalGraph, 
                           source_types: List[str] = None,
                           target_types: List[str] = None,
                           max_paths: int = 10) -> List[Dict[str, Any]]:
        start_time = time.time()
        
        if source_types is None:
            source_types = ['government_agency']
        if target_types is None:
            target_types = ['company', 'person']
        
        self.logger.info(
            f"Finding critical paths from {source_types} to {target_types}"
        )
        
        try:
            # Find nodes of specified types
            source_nodes = []
            target_nodes = []
            
            for node_id, node_data in graph.nx_graph.nodes(data=True):
                entity_type = node_data.get('entity_type', '')
                if entity_type in source_types:
                    source_nodes.append(node_id)
                elif entity_type in target_types:
                    target_nodes.append(node_id)
            
            critical_paths = []
            
            # Find shortest paths between source and target nodes
            for source in source_nodes[:20]:  # Limit to avoid excessive computation
                for target in target_nodes[:20]:
                    try:
                        if nx.has_path(graph.nx_graph, source, target):
                            path = nx.shortest_path(graph.nx_graph, source, target)
                            
                            if len(path) > 1:  # Valid path
                                path_info = self._analyze_path(graph, path)
                                critical_paths.append(path_info)
                                
                                if len(critical_paths) >= max_paths:
                                    break
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                
                if len(critical_paths) >= max_paths:
                    break
            
            # Sort by path strength/importance
            critical_paths.sort(key=lambda p: p['total_weight'], reverse=True)
            
            analysis_time = time.time() - start_time
            self.logger.info(
                f"Critical path analysis completed in {analysis_time:.2f}s: "
                f"{len(critical_paths)} paths found"
            )
            
            return critical_paths[:max_paths]
            
        except Exception as e:
            self.logger.error(f"Critical path analysis failed: {e}")
            raise GraphAnalysisError(f"Failed to find critical paths: {e}") from e
    
    def detect_anomalies(self, graph: LegalGraph) -> Dict[str, Any]:
        start_time = time.time()
        
        self.logger.info(f"Detecting anomalies in graph with {len(graph)} nodes")
        
        try:
            anomalies = {
                'high_degree_nodes': [],
                'isolated_high_value_nodes': [],
                'unusual_connection_patterns': [],
                'temporal_anomalies': [],
                'financial_anomalies': []
            }
            
            # Calculate statistics for anomaly detection
            degrees = dict(graph.nx_graph.degree())
            degree_values = list(degrees.values())
            
            if not degree_values:
                return anomalies
            
            avg_degree = sum(degree_values) / len(degree_values)
            max_degree = max(degree_values)
            
            # Detect high-degree nodes (potential hubs)
            degree_threshold = max(avg_degree * 3, 10)  # 3x average or at least 10
            
            for node_id, degree in degrees.items():
                if degree >= degree_threshold:
                    node_data = graph.nx_graph.nodes[node_id]
                    anomalies['high_degree_nodes'].append({
                        'node_id': node_id,
                        'canonical_name': node_data.get('canonical_name', ''),
                        'entity_type': node_data.get('entity_type', ''),
                        'degree': degree,
                        'degree_ratio': degree / avg_degree if avg_degree > 0 else 0
                    })
            
            # Detect isolated high-value nodes
            self._detect_isolated_high_value_nodes(graph, anomalies)
            
            # Detect unusual connection patterns
            self._detect_unusual_connections(graph, anomalies)
            
            # Detect temporal anomalies
            self._detect_temporal_anomalies(graph, anomalies)
            
            # Detect financial anomalies
            self._detect_financial_anomalies(graph, anomalies)
            
            analysis_time = time.time() - start_time
            total_anomalies = sum(len(v) for v in anomalies.values() if isinstance(v, list))
            
            self.logger.info(
                f"Anomaly detection completed in {analysis_time:.2f}s: "
                f"{total_anomalies} anomalies detected"
            )
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            raise GraphAnalysisError(f"Failed to detect anomalies: {e}") from e
    
    def analyze_temporal_patterns(self, graph: LegalGraph) -> Dict[str, Any]:
        start_time = time.time()
        
        self.logger.info("Analyzing temporal patterns")
        
        try:
            temporal_analysis = {
                'edge_timeline': [],
                'activity_periods': [],
                'temporal_clusters': [],
                'trend_analysis': {}
            }
            
            # Collect temporal data from edges
            edge_dates = []
            monthly_counts = defaultdict(int)
            yearly_counts = defaultdict(int)
            
            for source, target, edge_data in graph.nx_graph.edges(data=True):
                start_date = edge_data.get('start_date')
                if start_date:
                    if isinstance(start_date, str):
                        try:
                            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        except:
                            continue
                    
                    edge_dates.append({
                        'date': start_date,
                        'source': source,
                        'target': target,
                        'relationship_type': edge_data.get('relationship_type', ''),
                        'amount': edge_data.get('amount', 0)
                    })
                    
                    # Count by month and year
                    month_key = f"{start_date.year}-{start_date.month:02d}"
                    year_key = str(start_date.year)
                    monthly_counts[month_key] += 1
                    yearly_counts[year_key] += 1
            
            # Sort by date
            edge_dates.sort(key=lambda x: x['date'])
            temporal_analysis['edge_timeline'] = edge_dates
            
            # Identify activity periods
            if edge_dates:
                activity_periods = self._identify_activity_periods(edge_dates)
                temporal_analysis['activity_periods'] = activity_periods
            
            # Trend analysis
            temporal_analysis['trend_analysis'] = {
                'monthly_counts': dict(monthly_counts),
                'yearly_counts': dict(yearly_counts),
                'total_temporal_edges': len(edge_dates),
                'date_range': {
                    'earliest': edge_dates[0]['date'].isoformat() if edge_dates else None,
                    'latest': edge_dates[-1]['date'].isoformat() if edge_dates else None
                }
            }
            
            analysis_time = time.time() - start_time
            self.logger.info(
                f"Temporal analysis completed in {analysis_time:.2f}s: "
                f"{len(edge_dates)} temporal relationships analyzed"
            )
            
            return temporal_analysis
            
        except Exception as e:
            self.logger.error(f"Temporal analysis failed: {e}")
            raise GraphAnalysisError(f"Failed to analyze temporal patterns: {e}") from e
    
    def _update_node_centralities(self, graph: LegalGraph, centrality_results: Dict[str, Dict[str, float]]):
        for node_id in graph.nx_graph.nodes():
            centrality_scores = {}
            
            for centrality_type, scores in centrality_results.items():
                if node_id in scores:
                    centrality_scores[centrality_type] = scores[node_id]
            
            graph.nx_graph.nodes[node_id]['centrality_scores'] = centrality_scores
    
    def _analyze_communities(self, graph: LegalGraph, communities: List[List[str]]) -> Dict[str, Any]:
        community_analysis = {
            'community_sizes': [],
            'community_types': [],
            'inter_community_edges': 0,
            'intra_community_edges': 0
        }
        
        # Analyze community sizes and types
        for i, community in enumerate(communities):
            community_size = len(community)
            community_analysis['community_sizes'].append(community_size)
            
            # Analyze entity types in community
            type_counts = defaultdict(int)
            for node_id in community:
                node_data = graph.nx_graph.nodes.get(node_id, {})
                entity_type = node_data.get('entity_type', 'unknown')
                type_counts[entity_type] += 1
            
            community_analysis['community_types'].append({
                'community_id': i,
                'size': community_size,
                'entity_types': dict(type_counts),
                'dominant_type': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'unknown'
            })
        
        # Count inter vs intra community edges
        node_to_community = {}
        for i, community in enumerate(communities):
            for node_id in community:
                node_to_community[node_id] = i
        
        for source, target, edge_data in graph.nx_graph.edges(data=True):
            source_comm = node_to_community.get(source, -1)
            target_comm = node_to_community.get(target, -1)
            
            if source_comm == target_comm and source_comm != -1:
                community_analysis['intra_community_edges'] += 1
            else:
                community_analysis['inter_community_edges'] += 1
        
        return community_analysis
    
    def _analyze_path(self, graph: LegalGraph, path: List[str]) -> Dict[str, Any]:
        path_info = {
            'path': path,
            'length': len(path),
            'total_weight': 0.0,
            'entity_types': [],
            'relationship_types': [],
            'financial_amounts': []
        }
        
        # Analyze nodes in path
        for node_id in path:
            node_data = graph.nx_graph.nodes.get(node_id, {})
            path_info['entity_types'].append(node_data.get('entity_type', 'unknown'))
        
        # Analyze edges in path
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            
            if graph.nx_graph.has_edge(source, target):
                edge_data = graph.nx_graph[source][target]
                
                weight = edge_data.get('weight', 1.0)
                path_info['total_weight'] += weight
                
                rel_type = edge_data.get('relationship_type', 'unknown')
                path_info['relationship_types'].append(rel_type)
                
                amount = edge_data.get('amount')
                if amount:
                    path_info['financial_amounts'].append(amount)
        
        return path_info
    
    def _detect_isolated_high_value_nodes(self, graph: LegalGraph, anomalies: Dict[str, Any]):
        for node_id, node_data in graph.nx_graph.nodes(data=True):
            degree = graph.nx_graph.degree(node_id)
            
            # Look for nodes with low degree but high financial involvement
            if degree <= 2:  # Low connectivity
                total_amount = 0.0
                
                # Sum financial amounts from connected edges
                for neighbor in graph.nx_graph.neighbors(node_id):
                    if graph.nx_graph.has_edge(node_id, neighbor):
                        edge_data = graph.nx_graph[node_id][neighbor]
                        amount = edge_data.get('amount', 0)
                        if amount:
                            total_amount += amount
                    
                    if graph.nx_graph.has_edge(neighbor, node_id):
                        edge_data = graph.nx_graph[neighbor][node_id]
                        amount = edge_data.get('amount', 0)
                        if amount:
                            total_amount += amount
                
                if total_amount > 1_000_000:  # High financial value threshold
                    anomalies['isolated_high_value_nodes'].append({
                        'node_id': node_id,
                        'canonical_name': node_data.get('canonical_name', ''),
                        'entity_type': node_data.get('entity_type', ''),
                        'degree': degree,
                        'total_financial_amount': total_amount
                    })
    
    def _detect_unusual_connections(self, graph: LegalGraph, anomalies: Dict[str, Any]):
        # Look for nodes that connect different entity type clusters
        entity_type_connections = defaultdict(set)
        
        for source, target, edge_data in graph.nx_graph.edges(data=True):
            source_type = graph.nx_graph.nodes[source].get('entity_type', 'unknown')
            target_type = graph.nx_graph.nodes[target].get('entity_type', 'unknown')
            
            if source_type != target_type:
                entity_type_connections[source].add(target_type)
                entity_type_connections[target].add(source_type)
        
        # Find nodes connecting to many different entity types
        for node_id, connected_types in entity_type_connections.items():
            if len(connected_types) >= 4:  # Connected to 4+ different entity types
                node_data = graph.nx_graph.nodes[node_id]
                anomalies['unusual_connection_patterns'].append({
                    'node_id': node_id,
                    'canonical_name': node_data.get('canonical_name', ''),
                    'entity_type': node_data.get('entity_type', ''),
                    'connected_entity_types': list(connected_types),
                    'type_diversity': len(connected_types)
                })
    
    def _detect_temporal_anomalies(self, graph: LegalGraph, anomalies: Dict[str, Any]):
        current_time = datetime.utcnow()
        recent_threshold = current_time - timedelta(days=30)
        
        recent_activity_count = 0
        old_activity_count = 0
        
        for source, target, edge_data in graph.nx_graph.edges(data=True):
            start_date = edge_data.get('start_date')
            if start_date:
                if isinstance(start_date, str):
                    try:
                        start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    except:
                        continue
                
                if start_date > recent_threshold:
                    recent_activity_count += 1
                else:
                    old_activity_count += 1
        
        # Detect unusual recent activity spikes
        total_activity = recent_activity_count + old_activity_count
        if total_activity > 0:
            recent_ratio = recent_activity_count / total_activity
            
            if recent_ratio > 0.5:  # More than 50% of activity is recent
                anomalies['temporal_anomalies'].append({
                    'type': 'recent_activity_spike',
                    'recent_activity_count': recent_activity_count,
                    'total_activity_count': total_activity,
                    'recent_activity_ratio': recent_ratio
                })
    
    def _detect_financial_anomalies(self, graph: LegalGraph, anomalies: Dict[str, Any]):
        financial_amounts = []
        
        for source, target, edge_data in graph.nx_graph.edges(data=True):
            amount = edge_data.get('amount')
            if amount and amount > 0:
                financial_amounts.append({
                    'amount': amount,
                    'source': source,
                    'target': target,
                    'relationship_type': edge_data.get('relationship_type', '')
                })
        
        if not financial_amounts:
            return
        
        # Sort by amount
        financial_amounts.sort(key=lambda x: x['amount'], reverse=True)
        
        # Calculate statistics
        amounts = [fa['amount'] for fa in financial_amounts]
        avg_amount = sum(amounts) / len(amounts)
        max_amount = max(amounts)
        
        # Detect unusually large amounts
        large_amount_threshold = max(avg_amount * 10, 10_000_000)  # 10x average or $10M
        
        for fa in financial_amounts:
            if fa['amount'] >= large_amount_threshold:
                source_data = graph.nx_graph.nodes[fa['source']]
                target_data = graph.nx_graph.nodes[fa['target']]
                
                anomalies['financial_anomalies'].append({
                    'type': 'unusually_large_amount',
                    'amount': fa['amount'],
                    'amount_ratio': fa['amount'] / avg_amount,
                    'source_entity': source_data.get('canonical_name', ''),
                    'target_entity': target_data.get('canonical_name', ''),
                    'relationship_type': fa['relationship_type']
                })
    
    def _identify_activity_periods(self, edge_dates: List[Dict]) -> List[Dict[str, Any]]:
        if not edge_dates:
            return []
        
        # Group by month
        monthly_activity = defaultdict(list)
        for edge in edge_dates:
            month_key = f"{edge['date'].year}-{edge['date'].month:02d}"
            monthly_activity[month_key].append(edge)
        
        # Find months with high activity
        avg_monthly_activity = sum(len(edges) for edges in monthly_activity.values()) / len(monthly_activity)
        high_activity_threshold = max(avg_monthly_activity * 2, 5)  # 2x average or at least 5
        
        activity_periods = []
        for month, edges in monthly_activity.items():
            if len(edges) >= high_activity_threshold:
                total_amount = sum(edge.get('amount', 0) for edge in edges if edge.get('amount'))
                
                activity_periods.append({
                    'period': month,
                    'activity_count': len(edges),
                    'total_financial_amount': total_amount,
                    'activity_ratio': len(edges) / avg_monthly_activity if avg_monthly_activity > 0 else 0,
                    'relationship_types': list(set(edge['relationship_type'] for edge in edges if edge['relationship_type']))
                })
        
        # Sort by activity count
        activity_periods.sort(key=lambda x: x['activity_count'], reverse=True)
        
        return activity_periods 