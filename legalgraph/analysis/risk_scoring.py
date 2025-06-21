import re
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, date
from collections import Counter

from ..extraction.models import Entity
from ..ingestion.models import Document


@dataclass
class RiskScore:
    entity_id: str
    entity_text: str
    total_score: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_factors: List[str]
    factor_scores: Dict[str, float]
    document_count: int
    latest_mention: Optional[date] = None


class RiskScorer:
    
    def __init__(self):
        # Risk factor weights (total = 1.0)
        self.factor_weights = {
            'enforcement_frequency': 0.25,    # How often entity appears in enforcement
            'violation_severity': 0.20,       # Severity of violations
            'financial_penalties': 0.15,      # Size of financial penalties
            'recency': 0.15,                  # How recent the violations are
            'agency_diversity': 0.10,         # Multiple agencies involved
            'relationship_complexity': 0.10,  # Network connections
            'document_sentiment': 0.05        # Negative language intensity
        }
        
        # Violation severity patterns (higher score = more severe)
        self.severity_patterns = {
            'fraud': 0.9,
            'criminal': 0.95,
            'felony': 1.0,
            'conspiracy': 0.8,
            'money laundering': 0.9,
            'insider trading': 0.8,
            'market manipulation': 0.7,
            'securities violations': 0.6,
            'accounting fraud': 0.8,
            'tax evasion': 0.85,
            'bribery': 0.9,
            'corruption': 0.9,
            'embezzlement': 0.8,
            'wire fraud': 0.7,
            'mail fraud': 0.7,
            'racketeering': 0.95,
            'obstruction': 0.6,
            'perjury': 0.5,
            'disclosure violations': 0.4,
            'reporting violations': 0.3,
            'procedural violations': 0.2
        }
        
        # Financial penalty thresholds
        self.penalty_thresholds = {
            'critical': 100_000_000,  # $100M+
            'high': 10_000_000,       # $10M+
            'medium': 1_000_000,      # $1M+
            'low': 100_000            # $100K+
        }
        
        # Negative sentiment indicators
        self.negative_indicators = [
            'willful', 'intentional', 'deliberate', 'fraudulent',
            'deceptive', 'misleading', 'false', 'concealed',
            'manipulated', 'scheme', 'conspiracy', 'criminal',
            'illegal', 'unlawful', 'violation', 'breach',
            'misconduct', 'improper', 'unauthorized', 'prohibited'
        ]
    
    def calculate_risk_score(self, entity: Entity, documents: List[Document]) -> RiskScore:
        
        # Filter documents that mention this entity
        relevant_docs = self._filter_relevant_documents(entity, documents)
        
        if not relevant_docs:
            return RiskScore(
                entity_id=entity.id,
                entity_text=entity.text,
                total_score=0.0,
                risk_level='LOW',
                risk_factors=[],
                factor_scores={},
                document_count=0
            )
        
        # Calculate individual factor scores
        factor_scores = {}
        
        # 1. Enforcement Frequency (how often mentioned)
        factor_scores['enforcement_frequency'] = self._calculate_frequency_score(
            entity, relevant_docs, documents
        )
        
        # 2. Violation Severity (types of violations)
        factor_scores['violation_severity'] = self._calculate_severity_score(
            entity, relevant_docs
        )
        
        # 3. Financial Penalties (monetary amounts)
        factor_scores['financial_penalties'] = self._calculate_penalty_score(
            entity, relevant_docs
        )
        
        # 4. Recency (how recent are the violations)
        factor_scores['recency'] = self._calculate_recency_score(
            entity, relevant_docs
        )
        
        # 5. Agency Diversity (multiple agencies involved)
        factor_scores['agency_diversity'] = self._calculate_agency_diversity_score(
            entity, relevant_docs
        )
        
        # 6. Relationship Complexity (network connections)
        factor_scores['relationship_complexity'] = self._calculate_relationship_score(
            entity, relevant_docs
        )
        
        # 7. Document Sentiment (negative language)
        factor_scores['document_sentiment'] = self._calculate_sentiment_score(
            entity, relevant_docs
        )
        
        # Calculate weighted total score
        total_score = sum(
            factor_scores[factor] * self.factor_weights[factor]
            for factor in factor_scores
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(total_score)
        
        # Identify key risk factors
        risk_factors = self._identify_risk_factors(factor_scores, relevant_docs)
        
        # Find latest mention
        latest_mention = max(
            (doc.publish_date for doc in relevant_docs if doc.publish_date),
            default=None
        )
        
        return RiskScore(
            entity_id=entity.id,
            entity_text=entity.text,
            total_score=total_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            factor_scores=factor_scores,
            document_count=len(relevant_docs),
            latest_mention=latest_mention
        )
    
    def _filter_relevant_documents(self, entity: Entity, documents: List[Document]) -> List[Document]:
        relevant_docs = []
        entity_text_lower = entity.text.lower()
        
        for doc in documents:
            # Check title and content
            if (entity_text_lower in doc.title.lower() or 
                entity_text_lower in doc.content.lower()):
                relevant_docs.append(doc)
        
        return relevant_docs
    
    def _calculate_frequency_score(self, entity: Entity, relevant_docs: List[Document], 
                                 all_docs: List[Document]) -> float:
        if not all_docs:
            return 0.0
        
        # Frequency ratio
        frequency_ratio = len(relevant_docs) / len(all_docs)
        
        # Normalize to 0-1 scale (cap at 50% frequency = max score)
        frequency_score = min(frequency_ratio * 2, 1.0)
        
        # Boost score for high absolute counts
        if len(relevant_docs) >= 10:
            frequency_score = min(frequency_score * 1.2, 1.0)
        elif len(relevant_docs) >= 5:
            frequency_score = min(frequency_score * 1.1, 1.0)
        
        return frequency_score
    
    def _calculate_severity_score(self, entity: Entity, relevant_docs: List[Document]) -> float:
        if not relevant_docs:
            return 0.0
        
        max_severity = 0.0
        total_severity = 0.0
        violation_count = 0
        
        for doc in relevant_docs:
            content_lower = (doc.title + " " + doc.content).lower()
            
            for violation, severity in self.severity_patterns.items():
                if violation in content_lower:
                    max_severity = max(max_severity, severity)
                    total_severity += severity
                    violation_count += 1
        
        if violation_count == 0:
            return 0.1  # Minimal score for being in enforcement documents
        
        # Combine max severity with average severity
        avg_severity = total_severity / violation_count
        combined_severity = (max_severity * 0.7) + (avg_severity * 0.3)
        
        return combined_severity
    
    def _calculate_penalty_score(self, entity: Entity, relevant_docs: List[Document]) -> float:
        if not relevant_docs:
            return 0.0
        
        max_penalty = 0.0
        
        # Extract monetary amounts from documents
        money_pattern = re.compile(r'\$([0-9,]+(?:\.[0-9]{2})?)\s*(?:million|billion|thousand)?', re.IGNORECASE)
        
        for doc in relevant_docs:
            content = doc.title + " " + doc.content
            matches = money_pattern.findall(content)
            
            for match in matches:
                try:
                    amount = float(match.replace(',', ''))
                    
                    # Check for scale indicators
                    context = content[content.find(match):content.find(match) + 100].lower()
                    if 'billion' in context:
                        amount *= 1_000_000_000
                    elif 'million' in context:
                        amount *= 1_000_000
                    elif 'thousand' in context:
                        amount *= 1_000
                    
                    max_penalty = max(max_penalty, amount)
                except ValueError:
                    continue
        
        # Score based on penalty thresholds
        if max_penalty >= self.penalty_thresholds['critical']:
            return 1.0
        elif max_penalty >= self.penalty_thresholds['high']:
            return 0.8
        elif max_penalty >= self.penalty_thresholds['medium']:
            return 0.6
        elif max_penalty >= self.penalty_thresholds['low']:
            return 0.4
        elif max_penalty > 0:
            return 0.2
        else:
            return 0.0
    
    def _calculate_recency_score(self, entity: Entity, relevant_docs: List[Document]) -> float:
        if not relevant_docs:
            return 0.0
        
        current_date = datetime.now().date()
        scores = []
        
        for doc in relevant_docs:
            if not doc.publish_date:
                continue
            
            # Calculate days since publication
            days_since = (current_date - doc.publish_date).days
            
            # Exponential decay: more recent = higher score
            if days_since <= 30:
                score = 1.0
            elif days_since <= 90:
                score = 0.9
            elif days_since <= 180:
                score = 0.8
            elif days_since <= 365:
                score = 0.6
            elif days_since <= 730:  # 2 years
                score = 0.4
            elif days_since <= 1825:  # 5 years
                score = 0.2
            else:
                score = 0.1
            
            scores.append(score)
        
        # Return maximum recency score (most recent violation)
        return max(scores) if scores else 0.0
    
    def _calculate_agency_diversity_score(self, entity: Entity, relevant_docs: List[Document]) -> float:
        if not relevant_docs:
            return 0.0
        
        agencies = set()
        
        for doc in relevant_docs:
            if hasattr(doc, 'agencies') and doc.agencies:
                agencies.update(doc.agencies)
            
            # Also check document source
            if hasattr(doc, 'source_url'):
                if 'sec.gov' in doc.source_url:
                    agencies.add('SEC')
                elif 'justice.gov' in doc.source_url:
                    agencies.add('DOJ')
                elif 'consumerfinance.gov' in doc.source_url:
                    agencies.add('CFPB')
        
        # Score based on number of agencies
        agency_count = len(agencies)
        if agency_count >= 3:
            return 1.0
        elif agency_count == 2:
            return 0.7
        elif agency_count == 1:
            return 0.3
        else:
            return 0.0
    
    def _calculate_relationship_score(self, entity: Entity, relevant_docs: List[Document]) -> float:
        # This would integrate with relationship extraction results
        # For now, use a simple heuristic based on document complexity
        
        if not relevant_docs:
            return 0.0
        
        # Count mentions of other entities/companies
        entity_mention_pattern = re.compile(r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Co|Bank|Group|Holdings)\b')
        
        total_entities = 0
        for doc in relevant_docs:
            entities_in_doc = len(set(entity_mention_pattern.findall(doc.content)))
            total_entities += entities_in_doc
        
        # Normalize based on document count
        avg_entities_per_doc = total_entities / len(relevant_docs) if relevant_docs else 0
        
        # Score based on entity density
        if avg_entities_per_doc >= 10:
            return 1.0
        elif avg_entities_per_doc >= 5:
            return 0.7
        elif avg_entities_per_doc >= 2:
            return 0.4
        else:
            return 0.1
    
    def _calculate_sentiment_score(self, entity: Entity, relevant_docs: List[Document]) -> float:
        if not relevant_docs:
            return 0.0
        
        total_indicators = 0
        total_words = 0
        
        for doc in relevant_docs:
            content_words = (doc.title + " " + doc.content).lower().split()
            total_words += len(content_words)
            
            for indicator in self.negative_indicators:
                total_indicators += content_words.count(indicator)
        
        if total_words == 0:
            return 0.0
        
        # Calculate negative sentiment ratio
        sentiment_ratio = total_indicators / total_words
        
        # Normalize to 0-1 scale (cap at 5% negative words = max score)
        sentiment_score = min(sentiment_ratio * 20, 1.0)
        
        return sentiment_score
    
    def _determine_risk_level(self, total_score: float) -> str:
        if total_score >= 0.8:
            return 'CRITICAL'
        elif total_score >= 0.6:
            return 'HIGH'
        elif total_score >= 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _identify_risk_factors(self, factor_scores: Dict[str, float], 
                             relevant_docs: List[Document]) -> List[str]:
        risk_factors = []
        
        # Add factors with high scores
        for factor, score in factor_scores.items():
            if score >= 0.6:
                risk_factors.append(factor.replace('_', ' ').title())
        
        # Add specific violation types found
        violation_types = set()
        for doc in relevant_docs:
            content_lower = (doc.title + " " + doc.content).lower()
            for violation in self.severity_patterns:
                if violation in content_lower:
                    violation_types.add(violation.title())
        
        risk_factors.extend(list(violation_types)[:3])  # Top 3 violations
        
        return risk_factors[:5]  # Limit to top 5 factors 