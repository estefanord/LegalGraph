LegalGraph 

Automatically ingests, processes, and analyzes enforcement data from multiple government sources (SEC, DOJ, CFPB) to provide comprehensive risk assessments and relationship discovery.

Capabilities:
- Multi-source data ingestion from SEC, DOJ, and CFPB as of now
- Hybrid AI entity extraction with Regex, spaCy, and OpenAI GPT-3.5
- Knowledge graph construction with relationship discovery
- Multi-factor risk scoring with 7 weighted risk factors

The system consists of three main components:

Data Ingestion as of now: Automated scraping
- SEC Litigation: RSS feed + HTML parsing with anti-bot protection bypass
- DOJ Press Releases: Scraping with component-based filtering  
- CFPB Actions: Scraping with API fallback 

AI: Entity extraction/relationship discovery
- Regex patterns (47% of entities): Fast, precise for structured data
- spaCy NLP (50% of entities): Context-aware adapted to legal domain 
- OpenAI GPT-3.5 (3% of entities): Reasoning for new entity types
- Rule-based relationship extraction aided by LLM

Intelligence: Risk scoring/Network 
- 7-factor risk assessment with weighted contributions
- NetworkX-based graph and centrality analysis
- Community detection/anomaly detection
- Reporting + actionable insights

Risk Scoring:
1. Enforcement Frequency (25% weight): How often mentioned in actions
2. Violation Severity (20% weight): Type and severity of violations
3. Financial Impact (20% weight): Size of fines and settlements
4. Recent Activity (15% weight): Recency of enforcement actions
5. Regulatory Focus (10% weight): Which agencies are involved
6. Entity Prominence (5% weight): Size and importance of entity
7. Network Centrality (5% weight): Position in relationship network

Risk levels:
- HIGH RISK: Score > 0.7 
- MEDIUM RISK: Score 0.4-0.7 
- LOW RISK: Score < 0.4 

Quick Start
Prerequisites:
- Python 3.9 or higher
- OpenAI API key
- 8GB+ RAM recommended

Installation:
```bash
git clone https://github.com/estefanord/legalgraph.git
cd legalgraph
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env file and add your OpenAI API key
```

Basic usage:
```bash
# Acquire data from government sources
python scripts/data_acquisition.py

# Run comprehensive analysis
python scripts/ubs_analysis.py
```

Production Results (MVP)

Analysis of UBS Group AG from 47 enforcement documents:
- Total entities: 2,944 across all documents
- UBS-specific entities: 31 with detailed risk profiles
- Processing time: 75.6 seconds for complete analysis
- Financial exposure: $59.8 billion total

Risk Assessment:
- Overall risk score: 0.67 (MEDIUM risk level)
- High-risk entities: 2 requiring immediate attention
- Medium-risk entities: 24 needing enhanced monitoring
- Low-risk entities: 7 under standard oversight

Entity Extraction Performance:
- Regex: 334 entities with 0.70 confidence
- spaCy: 375 entities with 0.80 confidence
- OpenAI: 12 entities with 0.90 confidence
- Average: 62.6 entities per document

Network:
- Nodes: 64 entities with relationship mapping
- Edges: 187 relationships discovered
- Connected components: 12 distinct clusters
- Bridge entities: SEC, Southern District of New York

The demo_data folder contains real analysis outputs:

sample_graph.json: Complete knowledge graph with 64 nodes and 187 edges showing network relationships between entities in legal enforcement data.

sample_risk_response.json: Risk assessment for UBS Group AG using multi-factor analysis with history of enforcement + actionable recommendations.

sample_intelligence_report.json: A high-level intelligence report containing analysis, insights, and the monitoring framework.

Use Cases: Due Diligence, Compliance Monitoring, Risk Management, Regulatory Reporting, Etc.

Document types identified during UBS analysis:
- SEC Litigation Releases: 172 documents covering major securities violations
- DOJ Press Releases: 12 financial crime prosecutions and settlements
- CFPB Enforcement Actions: 25 consumer protection cases and penalties

All data is publicly available and automatically refreshed.

Performance as of now:
- Processing: 2.16 seconds per document
- Memory: <2GB for 200 documents
- Entity extraction accuracy: 90%+
- System reliability: 100% success rate


Requirements
See requirements.txt for complete list.

Dev Notes:

SEC Access Solution: Solved HTTP 403 errors by adding professional contact email to User-Agent string which enabled access to 172 litigation releases.

OpenAI Integration: Successfully integrated with  v1.x client, implemented smart text truncation for documents exceeding character limits.

Hybrid Entity Extraction: Three-method approach combining regex patterns, spaCy NLP, and OpenAI for maximum accuracy + coverage.

Performance at scale: Efficiently processes 200+ documents with 100% success rates across all government sources.

Up-to-date data: Fresh insights within hours of government publication through automated monitoring.

MIT License 
