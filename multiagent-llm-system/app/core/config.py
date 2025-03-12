
from typing import Any, Dict, List, Optional

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
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(60, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS
    CORS_ORIGINS: List[AnyHttpUrl] = Field([], env="CORS_ORIGINS")
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        """
        Validates and converts CORS_ORIGINS from string to list.
        """
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database Configuration
    DATABASE_URI: str = Field(..., env="DATABASE_URI")
    DATABASE_ECHO: bool = Field(False, env="DATABASE_ECHO")
    
    # Redis Configuration
    REDIS_URI: str = Field(..., env="REDIS_URI")
    
    # RabbitMQ Configuration
    RABBITMQ_URI: str = Field(..., env="RABBITMQ_URI")
    
    # API Keys
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field("gpt-4", env="OPENAI_MODEL")
    
    AWS_ACCESS_KEY_ID: str = Field(..., env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field("us-east-1", env="AWS_REGION")
    BEDROCK_MODEL: str = Field("anthropic.claude-3-sonnet-20240229-v1:0", env="BEDROCK_MODEL")
    
    JINA_API_KEY: str = Field(..., env="JINA_API_KEY")
    SERPER_API_KEY: str = Field(..., env="SERPER_API_KEY")
    
    LANGFUSE_PUBLIC_KEY: str = Field(..., env="LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY: str = Field(..., env="LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST: str = Field("https://cloud.langfuse.com", env="LANGFUSE_HOST")
    
    # Vector Database Configuration
    PINECONE_API_KEY: str = Field(..., env="PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = Field(..., env="PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = Field("multiagent-index", env="PINECONE_INDEX_NAME")
    VECTOR_DIMENSION: int = Field(1536, env="VECTOR_DIMENSION")
    VECTOR_METRIC: str = Field("cosine", env="VECTOR_METRIC")
    
    # Agent Configuration
    RESEARCHER_TEMPERATURE: float = Field(0.7, env="RESEARCHER_TEMPERATURE")
    RESEARCHER_MODEL: str = Field("gpt-4", env="RESEARCHER_MODEL")
    RESEARCHER_PROMPT_TEMPLATE: str = Field(
        "You are a research agent tasked with finding information about {query}. "
        "Search the web and extract relevant information.",
        env="RESEARCHER_PROMPT_TEMPLATE"
    )
    
    ANALYZER_TEMPERATURE: float = Field(0.2, env="ANALYZER_TEMPERATURE")
    ANALYZER_MODEL: str = Field("gpt-4", env="ANALYZER_MODEL")
    ANALYZER_PROMPT_TEMPLATE: str = Field(
        "You are an analysis agent tasked with processing information about {query}. "
        "Extract key insights and analyze the data.",
        env="ANALYZER_PROMPT_TEMPLATE"
    )
    
    SUMMARIZER_TEMPERATURE: float = Field(0.3, env="SUMMARIZER_TEMPERATURE")
    SUMMARIZER_MODEL: str = Field("gpt-4", env="SUMMARIZER_MODEL")
    SUMMARIZER_PROMPT_TEMPLATE: str = Field(
        "You are a summarization agent tasked with creating a concise summary about {query}. "
        "Create a clear, informative summary of the analysis.",
        env="SUMMARIZER_PROMPT_TEMPLATE"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary containing agent configuration
        """
        if agent_id == "researcher":
            return {
                "temperature": self.RESEARCHER_TEMPERATURE,
                "model": self.RESEARCHER_MODEL,
                "prompt_template": self.RESEARCHER_PROMPT_TEMPLATE,
                "openai_api_key": self.OPENAI_API_KEY
            }
        elif agent_id == "analyzer":
            return {
                "temperature": self.ANALYZER_TEMPERATURE,
                "model": self.ANALYZER_MODEL,
                "prompt_template": self.ANALYZER_PROMPT_TEMPLATE,
                "openai_api_key": self.OPENAI_API_KEY
            }
        elif agent_id == "summarizer":
            return {
                "temperature": self.SUMMARIZER_TEMPERATURE,
                "model": self.SUMMARIZER_MODEL,
                "prompt_template": self.SUMMARIZER_PROMPT_TEMPLATE,
                "openai_api_key": self.OPENAI_API_KEY
            }
        else:
            return {}
    
    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration for an LLM provider.
        
        Args:
            provider: Name of the provider (openai, bedrock)
            
        Returns:
            Dictionary containing provider configuration
        """
        if provider == "openai":
            return {
                "api_key": self.OPENAI_API_KEY,
                "model": self.OPENAI_MODEL
            }
        elif provider == "bedrock":
            return {
                "aws_access_key_id": self.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": self.AWS_SECRET_ACCESS_KEY,
                "aws_region": self.AWS_REGION,
                "model": self.BEDROCK_MODEL
            }
        else:
            return {}


settings = Settings()
