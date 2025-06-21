#!/usr/bin/env python3

import sys
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legalgraph.core.config import settings
from legalgraph.ingestion.sources.sec_litigation import SECLitigationSource
from legalgraph.ingestion.sources.doj_press import DOJPressSource
from legalgraph.ingestion.sources.cfpb_actions import CFPBEnforcementSource

import asyncio
import logging
from legalgraph.ingestion.sec_scraper import SECLitigationScraper
from legalgraph.ingestion.doj_scraper import DOJPressScraper  
from legalgraph.ingestion.cfpb_scraper import CFPBEnforcementScraper

def main():
    print('Data Acquisition')
    print('=' * 30)
    
    if not settings.openai_api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    all_documents = []
    
    print("\nSEC data...")
    start_time = time.time()
    with SECLitigationSource() as sec_scraper:
        sec_docs = sec_scraper.fetch_recent(days=10000)
        all_documents.extend(sec_docs)
        elapsed = time.time() - start_time
        print(f"SEC: {len(sec_docs)} documents ({elapsed:.1f}s)")
    
    print("DOJ data...")
    start_time = time.time()
    try:
        with DOJPressSource() as doj_scraper:
            doj_docs = doj_scraper.fetch_recent(days=2000)
            all_documents.extend(doj_docs)
            
            elapsed = time.time() - start_time
            print(f"DOJ: {len(doj_docs)} documents ({elapsed:.1f}s)")
            
    except Exception as e:
        print(f"DOJ issues: {e}")
    
    print("CFPB data...")
    start_time = time.time()
    try:
        with CFPBEnforcementSource() as cfpb_scraper:
            cfpb_docs = cfpb_scraper.fetch_recent(days=5000)
            all_documents.extend(cfpb_docs)
            
            elapsed = time.time() - start_time
            print(f"CFPB: {len(cfpb_docs)} documents ({elapsed:.1f}s)")
            
    except Exception as e:
        print(f"CFPB issues: {e}")
    
    print(f"\nTotal: {len(all_documents)} documents")
    
    print("Saving...")
    
    with open('legal_dataset.pkl', 'wb') as f:
        pickle.dump(all_documents, f)
    
    with open('dataset_summary.txt', 'w') as f:
        f.write(f"LEGAL DATASET SUMMARY\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"Total documents: {len(all_documents)}\n")
        f.write(f"Acquisition date: {datetime.now().isoformat()}\n\n")
        
        sec_count = len([d for d in all_documents if hasattr(d, 'source_url') and 'sec.gov' in d.source_url])
        doj_count = len([d for d in all_documents if hasattr(d, 'source_url') and 'justice.gov' in d.source_url])
        cfpb_count = len([d for d in all_documents if hasattr(d, 'source_url') and 'consumerfinance.gov' in d.source_url])
        
        f.write(f"DOCUMENTS BY SOURCE:\n")
        f.write(f"SEC Litigation: {sec_count}\n")
        f.write(f"DOJ Press Releases: {doj_count}\n")
        f.write(f"CFPB Enforcement: {cfpb_count}\n\n")
        
        major_banks = [
            "JPMorgan", "Wells Fargo", "Goldman Sachs", "Bank of America", 
            "Deutsche Bank", "Citigroup", "Morgan Stanley", "UBS", "Credit Suisse",
            "Barclays", "HSBC", "BNP Paribas", "Royal Bank", "TD Bank",
            "PNC Bank", "U.S. Bank", "Truist", "Capital One", "American Express"
        ]
        
        f.write(f"MAJOR FINANCIAL INSTITUTIONS:\n")
        bank_mentions = {}
        for bank in major_banks:
            count = sum(1 for doc in all_documents 
                       if bank.lower() in doc.content.lower() or bank.lower() in doc.title.lower())
            if count > 0:
                bank_mentions[bank] = count
        
        for bank, count in sorted(bank_mentions.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{bank}: {count} documents\n")
        
        f.write(f"\nTIMELINE:\n")
        years = {}
        for doc in all_documents:
            if hasattr(doc, 'publish_date') and doc.publish_date:
                year = doc.publish_date.year
                years[year] = years.get(year, 0) + 1
        
        for year in sorted(years.keys(), reverse=True):
            f.write(f"{year}: {years[year]} documents\n")
    
    print("\nAnalyzing coverage...")
    major_banks = [
        "JPMorgan", "Wells Fargo", "Goldman Sachs", "Bank of America", 
        "Deutsche Bank", "Citigroup", "Morgan Stanley", "UBS", "Credit Suisse",
        "Barclays", "HSBC", "BNP Paribas"
    ]
    
    bank_mentions = {}
    with tqdm(total=len(major_banks), desc="Banks", unit="banks") as pbar:
        for bank in major_banks:
            count = sum(1 for doc in all_documents 
                       if bank.lower() in doc.content.lower() or bank.lower() in doc.title.lower())
            if count > 0:
                bank_mentions[bank] = count
            pbar.set_description(f"Analyzing: {bank}")
            pbar.update(1)
    
    print("Bank coverage:")
    for bank, count in sorted(bank_mentions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {bank}: {count}")
    
    print(f"\nComplete")
    print(f"Documents: {len(all_documents)}")
    print(f"Banks covered: {len(bank_mentions)}")
    print(f"Saved as 'legal_dataset.pkl'")

if __name__ == "__main__":
    main() 