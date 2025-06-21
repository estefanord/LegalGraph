
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import math

from .models import LegalGraph, GraphNode
from ..extraction.models import EntityType, RelationType
from ..core.config import settings
from ..core.logging import get_logger
from ..core.exceptions import RiskScoringError


@dataclass
class RiskFactor:
    name: str
    score: float  # 0.0 to 1.0
    weight: float  # Importance weight
    description: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class RiskExplanation:
    entity_id: str
    entity_name: str
    total_risk_score: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    
    # Individual risk factors
    risk_factors: List[RiskFactor] = field(default_factory=list)
    
    # Summary statistics
    num_enforcement_actions: int = 0
    total_settlement_amount: float = 0.0
    num_violations: int = 0
    centrality_percentile: float = 0.0
    
    # Temporal factors
    recent_activity_score: float = 0.0
    activity_trend: str = "STABLE"  # "INCREASING", "DECREASING", "STABLE"
    
    # Network factors
    high_risk_connections: int = 0
    network_influence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "total_risk_score": self.total_risk_score,
            "risk_level": self.risk_level,
            "risk_factors": [
                {
                    "name": rf.name,
                    "score": rf.score,
                    "weight": rf.weight,
                    "description": rf.description,
                    "evidence": rf.evidence
                } for rf in self.risk_factors
            ],
            "summary_statistics": {
                "num_enforcement_actions": self.num_enforcement_actions,
                "total_settlement_amount": self.total_settlement_amount,
                "num_violations": self.num_violations,
                "centrality_percentile": self.centrality_percentile
            },
            "temporal_factors": {
                "recent_activity_score": self.recent_activity_score,
                "activity_trend": self.activity_trend
            },
            "network_factors": {
                "high_risk_connections": self.high_risk_connections,
                "network_influence_score": self.network_influence_score
            }
        }


class RiskScorer:
    
    def __init__(self):
        self.logger = get_logger("legalgraph.graph.scoring")
        
        # Risk factor weights (sum should be 1.0)
        self.risk_weights = {
            'enforcement_history': 0.25,      # Past enforcement actions
            'financial_penalties': 0.20,      # Settlement amounts
            'violation_severity': 0.15,       # Types of violations
            'network_centrality': 0.15,       # Position in network
            'recent_activity': 0.10,          # Recent legal activity
            'connection_risk': 0.10,          # Risk from connections
            'temporal_patterns': 0.05         # Temporal risk patterns
        }
        
        # Risk level thresholds
        self.risk_thresholds = {
            'LOW': 0.3,
            'MEDIUM': 0.6,
            'HIGH': 0.8,
            'CRITICAL': 1.0
        }
        
        # Violation severity mapping
        self.violation_severity = {
            'securities fraud': 0.9,
            'insider trading': 0.8,
            'market manipulation': 0.8,
            'money laundering': 0.95,
            'wire fraud': 0.7,
            'mail fraud': 0.6,
            'bank fraud': 0.8,
            'bribery': 0.85,
            'racketeering': 0.95,
            'embezzlement': 0.7,
            'kickbacks': 0.75
        }
    
    def calculate_risk_scores(self, graph: LegalGraph) -> Dict[str, RiskExplanation]:
        start_time = time.time()
        
        self.logger.info(f"Calculating risk scores for {len(graph)} entities")
        
        try:
            risk_scores = {}
            
            # Pre-calculate network statistics for efficiency
            network_stats = self._calculate_network_statistics(graph)
            
            # Calculate risk for each entity
            for node_id, node_data in graph.nx_graph.nodes(data=True):
                risk_explanation = self._calculate_entity_risk(
                    graph, node_id, node_data, network_stats
                )
                risk_scores[node_id] = risk_explanation
                
                # Update node with risk score
                graph.nx_graph.nodes[node_id]['risk_score'] = risk_explanation.total_risk_score
                graph.nx_graph.nodes[node_id]['risk_level'] = risk_explanation.risk_level
            
            # Post-process to identify high-risk connections
            self._identify_high_risk_connections(graph, risk_scores)
            
            calculation_time = time.time() - start_time
            
            # Log summary statistics
            risk_levels = [rs.risk_level for rs in risk_scores.values()]
            risk_level_counts = {level: risk_levels.count(level) for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']}
            
            self.logger.info(
                f"Risk scoring completed in {calculation_time:.2f}s: "
                f"LOW: {risk_level_counts['LOW']}, MEDIUM: {risk_level_counts['MEDIUM']}, "
                f"HIGH: {risk_level_counts['HIGH']}, CRITICAL: {risk_level_counts['CRITICAL']}"
            )
            
            return risk_scores
            
        except Exception as e:
            self.logger.error(f"Risk scoring failed: {e}")
            raise RiskScoringError(f"Failed to calculate risk scores: {e}") from e
    
    def calculate_entity_risk(self, graph: LegalGraph, entity_id: str) -> Optional[RiskExplanation]:
        if entity_id not in graph.nx_graph.nodes:
            return None
        
        node_data = graph.nx_graph.nodes[entity_id]
        network_stats = self._calculate_network_statistics(graph)
        
        return self._calculate_entity_risk(graph, entity_id, node_data, network_stats)
    
    def _calculate_entity_risk(self, graph: LegalGraph, entity_id: str, 
                              node_data: Dict[str, Any], 
                              network_stats: Dict[str, Any]) -> RiskExplanation:
        
        risk_explanation = RiskExplanation(
            entity_id=entity_id,
            entity_name=node_data.get('canonical_name', 'Unknown'),
            total_risk_score=0.0,
            risk_level='LOW'
        )
        
        # Calculate individual risk factors
        risk_factors = []
        
        # 1. Enforcement History Risk
        enforcement_risk = self._calculate_enforcement_risk(graph, entity_id, risk_explanation)
        risk_factors.append(enforcement_risk)
        
        # 2. Financial Penalties Risk
        financial_risk = self._calculate_financial_risk(graph, entity_id, risk_explanation)
        risk_factors.append(financial_risk)
        
        # 3. Violation Severity Risk
        violation_risk = self._calculate_violation_risk(graph, entity_id, risk_explanation)
        risk_factors.append(violation_risk)
        
        # 4. Network Centrality Risk
        centrality_risk = self._calculate_centrality_risk(graph, entity_id, node_data, network_stats, risk_explanation)
        risk_factors.append(centrality_risk)
        
        # 5. Recent Activity Risk
        activity_risk = self._calculate_recent_activity_risk(graph, entity_id, risk_explanation)
        risk_factors.append(activity_risk)
        
        # 6. Connection Risk (placeholder - will be calculated later)
        connection_risk = RiskFactor(
            name="connection_risk",
            score=0.0,
            weight=self.risk_weights['connection_risk'],
            description="Risk from connections to high-risk entities"
        )
        risk_factors.append(connection_risk)
        
        # 7. Temporal Patterns Risk
        temporal_risk = self._calculate_temporal_risk(graph, entity_id, risk_explanation)
        risk_factors.append(temporal_risk)
        
        # Calculate weighted total risk score
        total_risk = 0.0
        for factor in risk_factors:
            total_risk += factor.score * factor.weight
        
        risk_explanation.risk_factors = risk_factors
        risk_explanation.total_risk_score = min(total_risk, 1.0)  # Cap at 1.0
        risk_explanation.risk_level = self._determine_risk_level(risk_explanation.total_risk_score)
        
        return risk_explanation
    
    def _calculate_enforcement_risk(self, graph: LegalGraph, entity_id: str, 
                                  risk_explanation: RiskExplanation) -> RiskFactor:
        enforcement_count = 0
        evidence = []
        
        # Count enforcement-related edges
        for neighbor in graph.nx_graph.neighbors(entity_id):
            if graph.nx_graph.has_edge(neighbor, entity_id):
                edge_data = graph.nx_graph[neighbor][entity_id]
                rel_type = edge_data.get('relationship_type', '')
                
                if rel_type in ['enforcement_action', 'charged_with', 'prosecuted_by']:
                    enforcement_count += 1
                    neighbor_name = graph.nx_graph.nodes[neighbor].get('canonical_name', neighbor)
                    evidence.append(f"{rel_type} by {neighbor_name}")
        
        risk_explanation.num_enforcement_actions = enforcement_count
        
        # Calculate risk score based on enforcement count
        if enforcement_count == 0:
            score = 0.0
        elif enforcement_count <= 2:
            score = 0.3 + (enforcement_count * 0.2)
        elif enforcement_count <= 5:
            score = 0.7 + ((enforcement_count - 2) * 0.1)
        else:
            score = 1.0
        
        return RiskFactor(
            name="enforcement_history",
            score=score,
            weight=self.risk_weights['enforcement_history'],
            description=f"Risk from {enforcement_count} enforcement actions",
            evidence=evidence
        )
    
    def _calculate_financial_risk(self, graph: LegalGraph, entity_id: str,
                                risk_explanation: RiskExplanation) -> RiskFactor:
        total_amount = 0.0
        settlement_count = 0
        evidence = []
        
        # Sum settlement amounts from edges
        for neighbor in graph.nx_graph.neighbors(entity_id):
            # Check outgoing edges (entity paying)
            if graph.nx_graph.has_edge(entity_id, neighbor):
                edge_data = graph.nx_graph[entity_id][neighbor]
                if edge_data.get('relationship_type') in ['settlement', 'paid_to']:
                    amount = edge_data.get('amount', 0)
                    if amount:
                        total_amount += amount
                        settlement_count += 1
                        evidence.append(f"Settlement of ${amount:,.0f}")
            
            # Check incoming edges (entity receiving)
            if graph.nx_graph.has_edge(neighbor, entity_id):
                edge_data = graph.nx_graph[neighbor][entity_id]
                if edge_data.get('relationship_type') in ['settlement', 'paid_to']:
                    amount = edge_data.get('amount', 0)
                    if amount:
                        total_amount += amount
                        settlement_count += 1
                        evidence.append(f"Received ${amount:,.0f}")
        
        risk_explanation.total_settlement_amount = total_amount
        
        # Calculate risk score based on total amount
        if total_amount == 0:
            score = 0.0
        elif total_amount < 1_000_000:  # < $1M
            score = 0.2
        elif total_amount < 10_000_000:  # < $10M
            score = 0.4
        elif total_amount < 100_000_000:  # < $100M
            score = 0.6
        elif total_amount < 1_000_000_000:  # < $1B
            score = 0.8
        else:  # >= $1B
            score = 1.0
        
        return RiskFactor(
            name="financial_penalties",
            score=score,
            weight=self.risk_weights['financial_penalties'],
            description=f"Risk from ${total_amount:,.0f} in settlements",
            evidence=evidence
        )
    
    def _calculate_violation_risk(self, graph: LegalGraph, entity_id: str,
                                risk_explanation: RiskExplanation) -> RiskFactor:
        violations = []
        max_severity = 0.0
        evidence = []
        
        # Find violation relationships
        for neighbor in graph.nx_graph.neighbors(entity_id):
            neighbor_data = graph.nx_graph.nodes[neighbor]
            neighbor_type = neighbor_data.get('entity_type', '')
            
            if neighbor_type == 'violation_type':
                violation_name = neighbor_data.get('canonical_name', '').lower()
                severity = self.violation_severity.get(violation_name, 0.5)  # Default severity
                
                violations.append({
                    'name': violation_name,
                    'severity': severity
                })
                
                max_severity = max(max_severity, severity)
                evidence.append(f"Violation: {violation_name} (severity: {severity:.1f})")
        
        risk_explanation.num_violations = len(violations)
        
        # Calculate risk score based on worst violation
        score = max_severity
        
        # Adjust for multiple violations
        if len(violations) > 1:
            score = min(score + (len(violations) - 1) * 0.1, 1.0)
        
        return RiskFactor(
            name="violation_severity",
            score=score,
            weight=self.risk_weights['violation_severity'],
            description=f"Risk from {len(violations)} violations (max severity: {max_severity:.1f})",
            evidence=evidence
        )
    
    def _calculate_centrality_risk(self, graph: LegalGraph, entity_id: str, node_data: Dict[str, Any],
                                 network_stats: Dict[str, Any], risk_explanation: RiskExplanation) -> RiskFactor:
        centrality_scores = node_data.get('centrality_scores', {})
        
        # Get centrality percentiles
        degree_percentile = self._get_percentile(
            centrality_scores.get('degree', 0.0),
            network_stats['degree_centrality_values']
        )
        
        betweenness_percentile = self._get_percentile(
            centrality_scores.get('betweenness', 0.0),
            network_stats['betweenness_centrality_values']
        )
        
        # Use the maximum percentile as the centrality risk
        max_percentile = max(degree_percentile, betweenness_percentile)
        risk_explanation.centrality_percentile = max_percentile
        
        # Convert percentile to risk score
        score = max_percentile / 100.0
        
        evidence = [
            f"Degree centrality percentile: {degree_percentile:.1f}%",
            f"Betweenness centrality percentile: {betweenness_percentile:.1f}%"
        ]
        
        return RiskFactor(
            name="network_centrality",
            score=score,
            weight=self.risk_weights['network_centrality'],
            description=f"Risk from network position (percentile: {max_percentile:.1f}%)",
            evidence=evidence
        )
    
    def _calculate_recent_activity_risk(self, graph: LegalGraph, entity_id: str,
                                      risk_explanation: RiskExplanation) -> RiskFactor:
        current_time = datetime.utcnow()
        recent_threshold = current_time - timedelta(days=365)  # Last year
        very_recent_threshold = current_time - timedelta(days=90)  # Last 3 months
        
        recent_edges = []
        very_recent_edges = []
        
        # Check all edges connected to this entity
        all_edges = []
        
        # Outgoing edges
        for target in graph.nx_graph.neighbors(entity_id):
            if graph.nx_graph.has_edge(entity_id, target):
                edge_data = graph.nx_graph[entity_id][target]
                all_edges.append(edge_data)
        
        # Incoming edges
        for source in graph.nx_graph.predecessors(entity_id) if graph.nx_graph.is_directed() else []:
            if graph.nx_graph.has_edge(source, entity_id):
                edge_data = graph.nx_graph[source][entity_id]
                all_edges.append(edge_data)
        
        # Analyze temporal patterns
        for edge_data in all_edges:
            start_date = edge_data.get('start_date')
            if start_date:
                if isinstance(start_date, str):
                    try:
                        start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    except:
                        continue
                
                if start_date > recent_threshold:
                    recent_edges.append(edge_data)
                    
                    if start_date > very_recent_threshold:
                        very_recent_edges.append(edge_data)
        
        # Calculate activity score
        recent_count = len(recent_edges)
        very_recent_count = len(very_recent_edges)
        
        if recent_count == 0:
            score = 0.0
            activity_trend = "STABLE"
        elif very_recent_count > recent_count * 0.5:  # More than 50% activity is very recent
            score = 0.8
            activity_trend = "INCREASING"
        elif recent_count > 3:
            score = 0.6
            activity_trend = "INCREASING"
        elif recent_count > 1:
            score = 0.4
            activity_trend = "STABLE"
        else:
            score = 0.2
            activity_trend = "STABLE"
        
        risk_explanation.recent_activity_score = score
        risk_explanation.activity_trend = activity_trend
        
        evidence = [
            f"Recent activity (last year): {recent_count} events",
            f"Very recent activity (last 3 months): {very_recent_count} events",
            f"Activity trend: {activity_trend}"
        ]
        
        return RiskFactor(
            name="recent_activity",
            score=score,
            weight=self.risk_weights['recent_activity'],
            description=f"Risk from recent activity ({recent_count} recent events)",
            evidence=evidence
        )
    
    def _calculate_temporal_risk(self, graph: LegalGraph, entity_id: str,
                               risk_explanation: RiskExplanation) -> RiskFactor:
        # Look for suspicious temporal patterns
        edge_dates = []
        
        # Collect all edge dates for this entity
        for neighbor in graph.nx_graph.neighbors(entity_id):
            if graph.nx_graph.has_edge(entity_id, neighbor):
                edge_data = graph.nx_graph[entity_id][neighbor]
                start_date = edge_data.get('start_date')
                if start_date:
                    if isinstance(start_date, str):
                        try:
                            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                            edge_dates.append(start_date)
                        except:
                            continue
            
            if graph.nx_graph.has_edge(neighbor, entity_id):
                edge_data = graph.nx_graph[neighbor][entity_id]
                start_date = edge_data.get('start_date')
                if start_date:
                    if isinstance(start_date, str):
                        try:
                            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                            edge_dates.append(start_date)
                        except:
                            continue
        
        if len(edge_dates) < 2:
            score = 0.0
            evidence = ["Insufficient temporal data"]
        else:
            # Sort dates
            edge_dates.sort()
            
            # Look for clustering of events (potential suspicious activity)
            clustered_events = 0
            for i in range(1, len(edge_dates)):
                time_diff = (edge_dates[i] - edge_dates[i-1]).days
                if time_diff <= 30:  # Events within 30 days
                    clustered_events += 1
            
            # Calculate risk based on event clustering
            clustering_ratio = clustered_events / len(edge_dates) if edge_dates else 0
            score = min(clustering_ratio * 2, 1.0)  # Scale to 0-1
            
            evidence = [
                f"Total events: {len(edge_dates)}",
                f"Clustered events: {clustered_events}",
                f"Clustering ratio: {clustering_ratio:.2f}"
            ]
        
        return RiskFactor(
            name="temporal_patterns",
            score=score,
            weight=self.risk_weights['temporal_patterns'],
            description=f"Risk from temporal patterns (clustering score: {score:.2f})",
            evidence=evidence
        )
    
    def _calculate_network_statistics(self, graph: LegalGraph) -> Dict[str, Any]:
        stats = {
            'degree_centrality_values': [],
            'betweenness_centrality_values': []
        }
        
        # Collect all centrality values for percentile calculations
        for node_id, node_data in graph.nx_graph.nodes(data=True):
            centrality_scores = node_data.get('centrality_scores', {})
            
            degree_centrality = centrality_scores.get('degree', 0.0)
            stats['degree_centrality_values'].append(degree_centrality)
            
            betweenness_centrality = centrality_scores.get('betweenness', 0.0)
            stats['betweenness_centrality_values'].append(betweenness_centrality)
        
        return stats
    
    def _identify_high_risk_connections(self, graph: LegalGraph, risk_scores: Dict[str, RiskExplanation]):
        high_risk_threshold = 0.7
        
        for entity_id, risk_explanation in risk_scores.items():
            high_risk_connections = 0
            connection_risk_score = 0.0
            
            # Check connections to other entities
            for neighbor in graph.nx_graph.neighbors(entity_id):
                if neighbor in risk_scores:
                    neighbor_risk = risk_scores[neighbor].total_risk_score
                    
                    if neighbor_risk >= high_risk_threshold:
                        high_risk_connections += 1
                        connection_risk_score += neighbor_risk * 0.1  # Each connection adds risk
            
            # Update connection risk factor
            connection_risk_score = min(connection_risk_score, 1.0)
            
            # Find and update the connection risk factor
            for factor in risk_explanation.risk_factors:
                if factor.name == "connection_risk":
                    factor.score = connection_risk_score
                    factor.description = f"Risk from {high_risk_connections} high-risk connections"
                    factor.evidence = [f"High-risk connections: {high_risk_connections}"]
                    break
            
            # Update risk explanation
            risk_explanation.high_risk_connections = high_risk_connections
            risk_explanation.network_influence_score = connection_risk_score
            
            # Recalculate total risk score
            total_risk = sum(factor.score * factor.weight for factor in risk_explanation.risk_factors)
            risk_explanation.total_risk_score = min(total_risk, 1.0)
            risk_explanation.risk_level = self._determine_risk_level(risk_explanation.total_risk_score)
            
            # Update graph node
            graph.nx_graph.nodes[entity_id]['risk_score'] = risk_explanation.total_risk_score
            graph.nx_graph.nodes[entity_id]['risk_level'] = risk_explanation.risk_level
    
    def _get_percentile(self, value: float, all_values: List[float]) -> float:
        if not all_values:
            return 0.0
        
        sorted_values = sorted(all_values)
        rank = sum(1 for v in sorted_values if v <= value)
        percentile = (rank / len(sorted_values)) * 100
        
        return percentile
    
    def _determine_risk_level(self, risk_score: float) -> str:
        if risk_score >= self.risk_thresholds['CRITICAL']:
            return 'CRITICAL'
        elif risk_score >= self.risk_thresholds['HIGH']:
            return 'HIGH'
        elif risk_score >= self.risk_thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_high_risk_entities(self, risk_scores: Dict[str, RiskExplanation], 
                              min_risk_level: str = 'HIGH') -> List[RiskExplanation]:
        risk_level_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        min_level_index = risk_level_order.index(min_risk_level)
        
        high_risk_entities = []
        for risk_explanation in risk_scores.values():
            entity_level_index = risk_level_order.index(risk_explanation.risk_level)
            if entity_level_index >= min_level_index:
                high_risk_entities.append(risk_explanation)
        
        # Sort by risk score (highest first)
        high_risk_entities.sort(key=lambda x: x.total_risk_score, reverse=True)
        
        return high_risk_entities 