from typing import Any, Dict, List, Optional, Union
import json

from pydantic import AnyHttpUrl, Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Provides default values for development environments.
    """
    
    # API Configuration
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")
    DEBUG: bool = Field(False, env="DEBUG")
    ENVIRONMENT: str = Field("production", env="ENVIRONMENT")
    PROJECT_NAME: str = "Multiagent LLM System"
    PROJECT_DESCRIPTION: str = "A system that combines multiple agents for research, analysis, and summarization."
    PROJECT_VERSION: str = "0.1.0"
    
    # Security
    SECRET_KEY: Optional[str] = Field(None, env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(60, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"], env="CORS_ORIGINS")

    @validator("CORS_ORIGINS", pre=True)
    def validate_cors_origins(cls, v: Any) -> List[str]:
        """
        Validates and converts CORS_ORIGINS from various formats to a list of strings.
        """
        # If already a list, return as-is
        if isinstance(v, list):
            return v
        
        # If it's a string
        if isinstance(v, str):
            # Handle JSON string
            if v.startswith('[') and v.endswith(']'):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            
            # Handle comma-separated string
            if ',' in v:
                return [origin.strip() for origin in v.split(',')]
            
            # Single origin
            return [v]
        
        # Default to wildcard if can't parse
        return ["*"]
    
    # Database Configuration
    DATABASE_URI: Optional[str] = Field(None, env="DATABASE_URI")
    DATABASE_ECHO: bool = Field(False, env="DATABASE_ECHO")
    
    # Redis Configuration
    REDIS_URI: Optional[str] = Field(None, env="REDIS_URI")
    
    # RabbitMQ Configuration
    RABBITMQ_URI: Optional[str] = Field(None, env="RABBITMQ_URI")
    
    # API Keys (made optional to prevent startup errors)
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field("gpt-4", env="OPENAI_MODEL")
    
    AWS_ACCESS_KEY_ID: Optional[str] = Field(None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field("us-east-1", env="AWS_REGION")
    BEDROCK_MODEL: str = Field("anthropic.claude-3-sonnet-20240229-v1:0", env="BEDROCK_MODEL")
    
    # Optional additional configurations
    ANTHROPIC_API_KEY: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    VECTOR_DB_PROVIDER: Optional[str] = Field(None, env="VECTOR_DB_PROVIDER")
    VECTOR_DB_URI: Optional[str] = Field(None, env="VECTOR_DB_URI")
    
    # Monitoring and Infrastructure
    PROMETHEUS_ENABLED: bool = Field(False, env="PROMETHEUS_ENABLED")
    GRAFANA_ENABLED: bool = Field(False, env="GRAFANA_ENABLED")
    INSTANCE_TYPE: Optional[str] = Field(None, env="INSTANCE_TYPE")
    REGION: Optional[str] = Field(None, env="REGION")
    
    # Monitoring Ports
    PROMETHEUS_PORT: int = Field(9090, env="PROMETHEUS_PORT")
    GRAFANA_PORT: int = Field(3001, env="GRAFANA_PORT")
    
    # Vector Database and Additional Services
    PINECONE_HOST: Optional[str] = Field(None, env="PINECONE_HOST")
    PINECONE_REGION: Optional[str] = Field(None, env="PINECONE_REGION")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = 'ignore'  # This allows extra environment variables
    
    def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent.
        """
        agent_configs = {
            "researcher": {
                "temperature": self.RESEARCHER_TEMPERATURE,
                "model": self.RESEARCHER_MODEL,
                "prompt_template": self.RESEARCHER_PROMPT_TEMPLATE,
                "openai_api_key": self.OPENAI_API_KEY
            },
            "analyzer": {
                "temperature": self.ANALYZER_TEMPERATURE,
                "model": self.ANALYZER_MODEL,
                "prompt_template": self.ANALYZER_PROMPT_TEMPLATE,
                "openai_api_key": self.OPENAI_API_KEY
            },
            "summarizer": {
                "temperature": self.SUMMARIZER_TEMPERATURE,
                "model": self.SUMMARIZER_MODEL,
                "prompt_template": self.SUMMARIZER_PROMPT_TEMPLATE,
                "openai_api_key": self.OPENAI_API_KEY
            }
        }
        return agent_configs.get(agent_id, {})

    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration for an LLM provider.
        """
        llm_configs = {
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "model": self.OPENAI_MODEL
            },
            "bedrock": {
                "aws_access_key_id": self.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": self.AWS_SECRET_ACCESS_KEY,
                "aws_region": self.AWS_REGION,
                "model": self.BEDROCK_MODEL
            }
        }
        return llm_configs.get(provider, {})


# Create settings instance
settings = Settings()