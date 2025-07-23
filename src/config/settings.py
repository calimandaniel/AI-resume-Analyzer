import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

class Settings:
    """Centralized configuration management"""
    
    def __init__(self):
        self._load_environment()
        self._load_config_file()
    
    def _load_environment(self):
        """Load environment variables"""
        # Look for .env file in src directory
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
    
    def _load_config_file(self):
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
    
    @property
    def google_api_key(self) -> str:
        return os.getenv("GOOGLE_API_KEY", "")
    
    @property
    def database_url(self) -> str:
        return os.getenv("DATABASE_URL", "")
    
    @property
    def chunk_size(self) -> int:
        return self.config.get('processing', {}).get('chunk_size', 400)
    
    @property
    def chunk_overlap(self) -> int:
        return self.config.get('processing', {}).get('chunk_overlap', 50)
    
    @property
    def embedding_model(self) -> str:
        return self.config.get('model', {}).get('embedding_model', 'models/embedding-001')
    
    @property
    def llm_model(self) -> str:
        return self.config.get('model', {}).get('llm_model', 'gemini-1.5-flash')

# Global settings instance
settings = Settings()