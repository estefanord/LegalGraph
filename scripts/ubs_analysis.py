#!/usr/bin/env python3

import sys
import os
import pickle
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legalgraph.core.config import settings
from legalgraph.extraction.entity_extractor import HybridEntityExtractor
from legalgraph.extraction.relationships import RelationshipExtractor
from legalgraph.analysis.risk_scoring import RiskScorer
from legalgraph.analysis.network_analyzer import NetworkAnalyzer
from legalgraph.graph.constructor import GraphConstructor
from legalgraph.graph.knowledge_graph import KnowledgeGraphBuilder

def main():
    print('UBS Analysis')
    print('=' * 30)
    
    if not settings.openai_api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    print("Loading dataset...")
    try:
        with open('legal_dataset.pkl', 'rb') as f:
            all_documents = pickle.load(f)
        print(f"Loaded {len(all_documents)} documents")
    except FileNotFoundError:
        print("Dataset not found. Run data_acquisition.py first.")
        return
    
    print("Filtering UBS documents...")
    ubs_documents = []
    for doc in all_documents:
        content_lower = doc.content.lower()
        title_lower = doc.title.lower()
        if ('ubs' in content_lower or 'ubs' in title_lower):
            ubs_documents.append(doc)
    
    print(f"Found {len(ubs_documents)} UBS documents")
    
    if not ubs_documents:
        print("No UBS documents found")
        return
    
    print("Extracting entities...")
    
    try:
        with open('full_ubs_extraction_results.pkl', 'rb') as f:
            extraction_results = pickle.load(f)
        all_entities = extraction_results['entities']
        all_relationships = extraction_results['relationships']
        print(f"Loaded {len(all_entities)} entities, {len(all_relationships)} relationships")
    except FileNotFoundError:
        print("Running entity extraction...")
        
        entity_extractor = HybridEntityExtractor()
        relationship_extractor = RelationshipExtractor()
        
        all_entities = []
        all_relationships = []
        
        with tqdm(total=len(ubs_documents), desc="Processing", unit="docs") as pbar:
            for i, doc in enumerate(ubs_documents):
                try:
                    entities = entity_extractor.extract_entities(doc)
                    all_entities.extend(entities)
                    
                    relationships = relationship_extractor.extract_relationships(doc)
                    all_relationships.extend(relationships)
                    
                    if i % 5 == 0:
                        pbar.set_postfix({
                            "Entities": len(all_entities),
                            "Relationships": len(all_relationships)
                        })
                    pbar.update(1)
                    
                except Exception as e:
                    pbar.set_postfix({"Error": str(e)[:20]})
                    pbar.update(1)
                    continue
        
        print(f"Extracted {len(all_entities)} entities, {len(all_relationships)} relationships")
    
    print("Analyzing UBS entities...")
    
    ubs_entities = []
    for entity in all_entities:
        if ('ubs' in entity.text.lower() or 
            'ubs' in entity.canonical_name.lower() or
            any('ubs' in alias.lower() for alias in entity.aliases)):
            ubs_entities.append(entity)
    
    print(f"Found {len(ubs_entities)} UBS entities")
    
    entity_type_counts = Counter()
    for entity in ubs_entities:
        entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
        entity_type_counts[entity_type] += 1
    
    print("Entity types:")
    for entity_type, count in entity_type_counts.most_common():
        print(f"  {entity_type}: {count}")
    
    entities_by_type = defaultdict(list)
    for entity in ubs_entities:
        entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
        entities_by_type[entity_type].append(entity)
    
    for entity_type, entities in entities_by_type.items():
        print(f"\n{entity_type} ({len(entities)}):")
        unique_entities = {}
        for entity in entities:
            key = entity.canonical_name.lower()
            if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                unique_entities[key] = entity
        
        top_entities = sorted(unique_entities.values(), key=lambda x: x.confidence, reverse=True)[:5]
        for entity in top_entities:
            print(f"  {entity.text} ({entity.confidence:.2f})")
    
    print("\nCalculating risk scores...")
    
    risk_scorer = RiskScorer()
    ubs_risk_scores = []
    
    with tqdm(total=len(ubs_entities), desc="Risk Assessment", unit="entities") as pbar:
        for entity in ubs_entities:
            try:
                risk_score = risk_scorer.calculate_risk_score(entity, ubs_documents)
                ubs_risk_scores.append({
                    'entity': entity.text,
                    'canonical_name': entity.canonical_name,
                    'type': entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
                    'risk_score': risk_score.total_score,
                    'risk_level': risk_score.risk_level,
                    'factors': risk_score.risk_factors,
                    'document_count': risk_score.document_count,
                    'factor_scores': risk_score.factor_scores
                })
                pbar.update(1)
            except Exception as e:
                pbar.update(1)
                continue
    
    ubs_risk_scores.sort(key=lambda x: x['risk_score'], reverse=True)
    
    print("\nTop 10 highest risk entities:")
    for i, risk_entity in enumerate(ubs_risk_scores[:10], 1):
        print(f"  {i}. {risk_entity['entity']} ({risk_entity['type']})")
        print(f"     Risk: {risk_entity['risk_score']:.3f} ({risk_entity['risk_level']})")
        print(f"     Documents: {risk_entity['document_count']}")
        print(f"     Factors: {', '.join(risk_entity['factors'][:3])}")
        print()
    
    risk_levels = Counter([r['risk_level'] for r in ubs_risk_scores])
    print("Risk distribution:")
    for level, count in risk_levels.most_common():
        print(f"  {level}: {count}")
    
    print("\nBuilding network...")
    
    graph_constructor = GraphConstructor()
    
    try:
        knowledge_graph = graph_constructor.construct_graph(all_entities, all_relationships)
        
        network_analyzer = NetworkAnalyzer()
        network_stats = network_analyzer.analyze_network(knowledge_graph)
        
        print("Network stats:")
        print(f"  Nodes: {network_stats['node_count']}")
        print(f"  Edges: {network_stats['edge_count']}")
        print(f"  Density: {network_stats['density']:.4f}")
        print(f"  Components: {network_stats['connected_components']}")
        print(f"  Clustering: {network_stats['average_clustering']:.4f}")
        
        print("\nMost central entities:")
        try:
            centrality_scores = network_analyzer.calculate_centrality(knowledge_graph, 'pagerank')
            for i, (entity_id, score) in enumerate(centrality_scores[:10], 1):
                entity_text = entity_id
                for entity in all_entities:
                    if entity.id == entity_id:
                        entity_text = entity.text
                        break
                print(f"  {i}. {entity_text}: {score:.4f}")
        except Exception as e:
            print(f"  Centrality calculation failed: {e}")
        
        print("\nNetwork anomalies:")
        try:
            anomalies = network_analyzer.detect_anomalies(knowledge_graph)
            for anomaly_type, entities in anomalies.items():
                if entities:
                    print(f"  {anomaly_type.replace('_', ' ').title()}: {len(entities)}")
        except Exception as e:
            print(f"  Anomaly detection failed: {e}")
        
    except Exception as e:
        print(f"Network analysis failed: {e}")
        network_stats = {'node_count': 0, 'edge_count': 0}
    
    print("\nAnalyzing timeline...")
    
    doc_timeline = defaultdict(list)
    enforcement_by_year = defaultdict(lambda: defaultdict(int))
    
    for doc in ubs_documents:
        if hasattr(doc, 'publish_date') and doc.publish_date:
            year = doc.publish_date.year
            doc_timeline[year].append(doc)
            
            if hasattr(doc, 'source_url'):
                if 'sec.gov' in doc.source_url:
                    enforcement_by_year[year]['SEC'] += 1
                elif 'justice.gov' in doc.source_url:
                    enforcement_by_year[year]['DOJ'] += 1
                elif 'consumerfinance.gov' in doc.source_url:
                    enforcement_by_year[year]['CFPB'] += 1
    
    print("Timeline:")
    for year in sorted(doc_timeline.keys(), reverse=True)[:10]:
        count = len(doc_timeline[year])
        print(f"  {year}: {count} documents")
        
        year_breakdown = enforcement_by_year[year]
        if year_breakdown:
            breakdown_str = ", ".join([f"{agency}: {count}" for agency, count in year_breakdown.items()])
            print(f"    ({breakdown_str})")
    
    print("\nAnalyzing financial impact...")
    
    financial_entities = [e for e in all_entities if e.entity_type.value == 'money']
    financial_amounts = []
    
    print(f"Found {len(financial_entities)} financial mentions")
    
    for entity in financial_entities:
        try:
            text = entity.text.lower()
            if 'million' in text:
                multiplier = 1_000_000
            elif 'billion' in text:
                multiplier = 1_000_000_000
            else:
                multiplier = 1
            
            import re
            numbers = re.findall(r'[\d,]+\.?\d*', entity.text)
            if numbers:
                amount = float(numbers[0].replace(',', '')) * multiplier
                financial_amounts.append(amount)
        except:
            continue
    
    if financial_amounts:
        total_amount = sum(financial_amounts)
        max_amount = max(financial_amounts)
        avg_amount = total_amount / len(financial_amounts)
        
        print("Financial impact:")
        print(f"  Total: ${total_amount:,.0f}")
        print(f"  Largest: ${max_amount:,.0f}")
        print(f"  Average: ${avg_amount:,.0f}")
        print(f"  Mentions: {len(financial_amounts)}")
    
    print("\nGenerating report...")
    
    report = {
        'analysis_metadata': {
            'analysis_date': datetime.now().isoformat(),
            'total_documents_analyzed': len(ubs_documents),
            'total_entities_extracted': len(all_entities),
            'ubs_specific_entities': len(ubs_entities),
            'total_relationships': len(all_relationships),
            'analysis_scope': 'UBS Group AG'
        },
        
        'document_sources': {
            'SEC': len([d for d in ubs_documents if hasattr(d, 'source_url') and 'sec.gov' in d.source_url]),
            'DOJ': len([d for d in ubs_documents if hasattr(d, 'source_url') and 'justice.gov' in d.source_url]),
            'CFPB': len([d for d in ubs_documents if hasattr(d, 'source_url') and 'consumerfinance.gov' in d.source_url])
        },
        
        'entity_analysis': {
            'total_ubs_entities': len(ubs_entities),
            'entity_type_breakdown': dict(entity_type_counts),
            'extraction_methods': {
                'regex': len([e for e in ubs_entities if e.extraction_method == 'regex']),
                'spacy': len([e for e in ubs_entities if e.extraction_method == 'spacy']),
                'openai': len([e for e in ubs_entities if e.extraction_method == 'openai'])
            }
        },
        
        'risk_assessment': {
            'risk_level_distribution': dict(risk_levels),
            'highest_risk_entities': ubs_risk_scores[:10],
            'average_risk_score': sum(r['risk_score'] for r in ubs_risk_scores) / len(ubs_risk_scores) if ubs_risk_scores else 0
        },
        
        'network_analysis': network_stats,
        
        'temporal_analysis': {
            'enforcement_timeline': {str(k): len(v) for k, v in doc_timeline.items()},
            'enforcement_by_agency_year': {str(k): dict(v) for k, v in enforcement_by_year.items()},
            'peak_enforcement_year': max(doc_timeline.keys(), key=lambda y: len(doc_timeline[y])) if doc_timeline else None
        },
        
        'financial_impact': {
            'total_financial_exposure': sum(financial_amounts) if financial_amounts else 0,
            'largest_amount': max(financial_amounts) if financial_amounts else 0,
            'average_amount': sum(financial_amounts) / len(financial_amounts) if financial_amounts else 0,
            'financial_mentions_count': len(financial_amounts)
        }
    }
    
    with open('ubs_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("Report saved to 'ubs_analysis_report.json'")
    
    print(f"\nAnalysis complete")
    print(f"Documents: {len(ubs_documents)}")
    print(f"Entities: {len(all_entities)}")
    print(f"UBS entities: {len(ubs_entities)}")
    print(f"Relationships: {len(all_relationships)}")
    print(f"Risk assessments: {len(ubs_risk_scores)}")
    print(f"Financial exposure: ${sum(financial_amounts):,.0f}" if financial_amounts else "Financial exposure: pending")

if __name__ == "__main__":
    main() 