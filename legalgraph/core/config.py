
import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Settings:
    
    # Data source URLs
    sec_rss_url: str = "https://www.sec.gov/litigation/litreleases.xml"
    doj_press_url: str = "https://www.justice.gov/news"
    cfpb_enforcement_url: str = "https://www.consumerfinance.gov/enforcement/actions/"
    
    # Rate limiting
    request_delay: float = 1.0  # Seconds between requests
    max_retries: int = 3
    timeout: int = 30
    
    # Processing thresholds
    max_documents_per_source: int = 100
    entity_extraction_confidence_threshold: float = 0.5
    relationship_extraction_confidence_threshold: float = 0.6
    
    # OpenAI API configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 1000
    openai_temperature: float = 0.1
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File paths
    data_dir: str = "data"
    cache_dir: str = "cache"
    output_dir: str = "output"


# Global settings instance
settings = Settings() 