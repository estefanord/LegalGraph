
class SimpleSettings:
    
    def __init__(self):
        # API Configuration
        self.api_host = "0.0.0.0"
        self.api_port = 8000
        self.api_version = "v1"
        
        # Data Sources Configuration
        self.sec_rss_url = "https://www.sec.gov/litigation/litreleases.rss"
        self.sec_base_url = "https://www.sec.gov"
        self.doj_press_base_url = "https://www.justice.gov/news"
        self.cfpb_actions_url = "https://www.consumerfinance.gov/policy-compliance/enforcement/actions"
        
        # Scraping Configuration
        self.request_delay = 1.5  # Seconds between requests
        self.max_retries = 3
        self.timeout = 30
        self.user_agent = "LegalGraph/1.0 (Research Tool)"
        
        # Document Processing
        self.max_document_size = 10 * 1024 * 1024  # 10MB
        
        # Logging Configuration
        self.log_level = "INFO"
        self.log_file = "legalgraph.log"
        self.data_directory = "data"

# Global settings instance
settings = SimpleSettings() 