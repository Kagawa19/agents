#!/usr/bin/env python3
"""
Script to generate the multiagent LLM system project directory and files.
"""

import os
import sys
from pathlib import Path
import json

# File content dictionary (truncated for readability)
# For actual script, all file contents would be included here
FILE_CONTENTS = {
    # Root directory files
    ".env.example": """# Environment variables for the multiagent-llm-system

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
ENVIRONMENT=development
SECRET_KEY=your-secret-key-for-jwt
ACCESS_TOKEN_EXPIRE_MINUTES=60
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Database Configuration
DATABASE_URI=postgresql://postgres:postgres@db:5432/multiagent
DATABASE_ECHO=False

# Redis Configuration
REDIS_URI=redis://redis:6379/0

# RabbitMQ Configuration
RABBITMQ_URI=amqp://guest:guest@rabbitmq:5672//

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4

# Amazon Bedrock Configuration
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0

# Jina AI Configuration
JINA_API_KEY=your-jina-api-key

# Serper API Configuration
SERPER_API_KEY=your-serper-api-key

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com

# Vector Database Configuration
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX_NAME=multiagent-index
VECTOR_DIMENSION=1536
VECTOR_METRIC=cosine

# Prometheus and Grafana
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
""",

    ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs
logs/
*.log

# Database
*.db
*.sqlite3

# Docker
docker/prometheus/data/
docker/grafana/data/

# Miscellaneous
.DS_Store
.coverage
htmlcov/
.pytest_cache/
""",

    "README.md": """# Multiagent LLM System

A comprehensive multiagent system built with FastAPI, LangChain, and various LLM providers including OpenAI and Amazon Bedrock.

## Features

- 🤖 Multiple specialized agents (Researcher, Analyzer, Summarizer)
- 🔍 Web search capability via Serper API
- 📊 Document processing and embeddings with Jina AI
- 📝 Summarization with multiple LLM providers
- 📈 Comprehensive observability with Langfuse and Prometheus
- ⚡ Asynchronous processing with Celery and RabbitMQ
- 🚀 Real-time updates via WebSockets

## Architecture

The system follows a modular architecture with:
- FastAPI backend
- Celery task queue for background processing
- PostgreSQL database for structured data
- Vector databases (Pinecone/Weaviate/ChromaDB) for embeddings
- Redis for caching and Celery results
- Comprehensive monitoring via Langfuse and Prometheus/Grafana

## Installation and Setup

### Prerequisites
- Docker and Docker Compose
- API keys for:
  - OpenAI
  - Amazon Bedrock (AWS credentials)
  - Jina AI
  - Serper
  - Langfuse
  - Pinecone

### Setup
1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your API keys
3. Run `docker-compose up -d`
4. Access the API at http://localhost:8000/docs

## Usage

Send a query to the system:
```bash
curl -X POST "http://localhost:8000/api/query" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {your_token}" \\
  -d '{"query": "What are the latest developments in AI?", "workflow_id": "research"}'
```

## Development

### Running Tests
```bash
docker-compose exec api pytest
```

### Monitoring
- Langfuse: View traces of agent operations
- Prometheus/Grafana: Monitor system metrics
  - Grafana: http://localhost:3001
  - Prometheus: http://localhost:9090

## License

MIT
""",

    "pyproject.toml": """[build-system]
requires = ["setuptools>=42.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "multiagent-llm-system"
version = "0.1.0"
description = "Multiagent LLM system using FastAPI, LangChain, and various LLM providers"
readme = "README.md"
requires-python = ">=3.10"
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
license = {text = "MIT"}
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "fastapi>=0.103.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.3.0",
    "pydantic-settings>=2.0.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.12.0",
    "psycopg2-binary>=2.9.0",
    "python-jose>=3.3.0",
    "passlib>=1.7.4",
    "python-multipart>=0.0.6",
    "httpx>=0.24.0",
    "celery>=5.3.0",
    "redis>=4.6.0",
    "prometheus-client>=0.17.0",
    "langchain>=0.0.300",
    "langchain_openai>=0.0.2",
    "langchain_community>=0.0.9",
    "langfuse>=2.0.0",
    "boto3>=1.28.0",
    "jina>=3.15.0",
    "pinecone-client>=2.2.0",
    "weaviate-client>=3.24.0",
    "chromadb>=0.4.0",
    "llama-index>=0.8.0",
    "openai>=1.3.0",
    "websockets>=11.0.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.1.0",
    "black>=23.7.0",
    "isort>=5.12.0",
    "flake8>=6.1.0",
    "mypy>=1.5.0"
]

[tool.black]
line-length = 88
target-version = ["py310"]
include = '\\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
""",

    "setup.py": """from setuptools import setup, find_packages

setup(
    name="multiagent-llm-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.103.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.3.0",
        "sqlalchemy>=2.0.0",
        "celery>=5.3.0",
        "langchain>=0.0.300",
        "langfuse>=2.0.0",
        "openai>=1.3.0",
        # Other dependencies are listed in pyproject.toml
    ],
    python_requires=">=3.10",
)
""",

    "requirements.txt": """fastapi>=0.103.0
uvicorn>=0.23.0
pydantic>=2.3.0
pydantic-settings>=2.0.0
sqlalchemy>=2.0.0
alembic>=1.12.0
psycopg2-binary>=2.9.0
python-jose>=3.3.0
passlib>=1.7.4
python-multipart>=0.0.6
httpx>=0.24.0
celery>=5.3.0
redis>=4.6.0
prometheus-client>=0.17.0
langchain>=0.0.300
langchain_openai>=0.0.2
langchain_community>=0.0.9
langfuse>=2.0.0
boto3>=1.28.0
jina>=3.15.0
pinecone-client>=2.2.0
weaviate-client>=3.24.0
chromadb>=0.4.0
llama-index>=0.8.0
openai>=1.3.0
websockets>=11.0.0
pytest>=7.0.0
pytest-cov>=4.1.0
black>=23.7.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.5.0
"""
}

# App package files
APP_FILES = {
    "app/__init__.py": """
\"\"\"
Multiagent LLM system.
A system that combines multiple agents for research, analysis, and summarization.
\"\"\"

__version__ = "0.1.0"
""",

    "app/main.py": """
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.endpoints import agents, auth, health, query
from app.api.error_handlers import add_exception_handlers
from app.core.config import settings
from app.core.events import create_start_app_handler, create_stop_app_handler
from app.monitoring.logging import setup_logging
from app.monitoring.metrics import PrometheusMiddleware, setup_metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    \"\"\"
    Lifecycle manager for the FastAPI application.
    Handles application startup and shutdown events.
    \"\"\"
    # Startup
    startup_handler = create_start_app_handler(app)
    await startup_handler()
    
    # Yield control to the application
    yield
    
    # Shutdown
    shutdown_handler = create_stop_app_handler(app)
    await shutdown_handler()


def create_application() -> FastAPI:
    \"\"\"
    Creates and configures the FastAPI application.
    Sets up middleware, routes, and exception handlers.
    
    Returns:
        FastAPI: The configured application instance
    \"\"\"
    # Initialize logging
    setup_logging()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.PROJECT_VERSION,
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup monitoring
    app.add_middleware(PrometheusMiddleware)
    setup_metrics(app)
    
    # Add routes
    app.include_router(auth.router, prefix="/api", tags=["auth"])
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(query.router, prefix="/api", tags=["query"])
    app.include_router(agents.router, prefix="/api", tags=["agents"])
    
    # Add exception handlers
    add_exception_handlers(app)
    
    return app


def get_application() -> FastAPI:
    \"\"\"
    Returns the configured FastAPI application instance.
    
    Returns:
        FastAPI: The application instance
    \"\"\"
    return create_application()


app = get_application()
"""
}

# Core module files
CORE_FILES = {
    "app/core/__init__.py": """
\"\"\"
Core functionality for the multiagent LLM system.
\"\"\"
""",

    "app/core/config.py": """
from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl, Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    \"\"\"
    Application settings loaded from environment variables.
    Provides default values for development environments.
    \"\"\"
    
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
        \"\"\"
        Validates and converts CORS_ORIGINS from string to list.
        \"\"\"
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
        \"\"\"
        Get configuration for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary containing agent configuration
        \"\"\"
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
        \"\"\"
        Get configuration for an LLM provider.
        
        Args:
            provider: Name of the provider (openai, bedrock)
            
        Returns:
            Dictionary containing provider configuration
        \"\"\"
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
""",

    "app/core/security.py": """
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.schemas.auth import TokenData

# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 configuration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    \"\"\"
    Creates a JWT token from the given data.
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time delta
        
    Returns:
        JWT token as a string
    \"\"\"
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    \"\"\"
    Verifies a JWT token and returns its payload.
    
    Args:
        token: JWT token to verify
        
    Returns:
        Token payload
        
    Raises:
        HTTPException: If token is invalid
    \"\"\"
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_password_hash(password: str) -> str:
    \"\"\"
    Hashes a password.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    \"\"\"
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    \"\"\"
    Verifies a password against a hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches hash, False otherwise
    \"\"\"
    return pwd_context.verify(plain_password, hashed_password)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    \"\"\"
    Gets the current user from a token.
    
    Args:
        token: JWT token
        
    Returns:
        User data
        
    Raises:
        HTTPException: If token is invalid or expired
    \"\"\"
    payload = verify_token(token)
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In a real application, you would query the user from the database
    # This is a simplified example
    return {"sub": username, "is_active": True}
""",

    "app/core/events.py": """
import logging
from typing import Callable

from fastapi import FastAPI

from app.db.session import engine
from app.db.models import Base
from app.monitoring.tracer import setup_tracer


logger = logging.getLogger(__name__)

def create_start_app_handler(app: FastAPI) -> Callable:
    \"\"\"
    Creates a function to run at application startup.
    
    Args:
        app: FastAPI application
        
    Returns:
        Function to run at startup
    \"\"\"
    async def start_app() -> None:
        \"\"\"
        Runs at application startup.
        Initializes database, connections, and services.
        \"\"\"
        # Create database tables
        logger.info("Creating database tables")
        Base.metadata.create_all(bind=engine)
        
        # Set up tracer
        logger.info("Setting up tracer")
        setup_tracer()
        
        logger.info("Application startup complete")
    
    return start_app

def create_stop_app_handler(app: FastAPI) -> Callable:
    \"\"\"
    Creates a function to run at application shutdown.
    
    Args:
        app: FastAPI application
        
    Returns:
        Function to run at shutdown
    \"\"\"
    async def stop_app() -> None:
        \"\"\"
        Runs at application shutdown.
        Closes connections and performs cleanup.
        \"\"\"
        logger.info("Application shutdown complete")
    
    return stop_app
"""
}

# More modules would be defined here in similar fashion

# Directory structure
def create_directory_structure(base_dir):
    """Create the directory structure for the project."""
    directories = [
        "app",
        "app/api",
        "app/api/endpoints",
        "app/core",
        "app/agents",
        "app/orchestrator",
        "app/tools",
        "app/monitoring",
        "app/schemas",
        "app/db",
        "app/db/crud",
        "app/vector_db",
        "app/worker",
        "tests",
        "scripts",
        "alembic",
        "alembic/versions",
        "docker",
        "docker/nginx",
        "infra",
        "infra/terraform",
        "infra/kubernetes"
    ]
    
    for directory in directories:
        path = os.path.join(base_dir, directory)
        os.makedirs(path, exist_ok=True)
        # Create __init__.py files in Python package directories
        if directory.startswith("app/") or directory == "app":
            init_file = os.path.join(base_dir, directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write('"""\n{} module for the multiagent LLM system.\n"""\n'.format(
                        directory.split('/')[-1].capitalize()
                    ))

def write_file_contents(base_dir):
    """Write file contents to their respective paths."""
    # Write root directory files
    for filename, content in FILE_CONTENTS.items():
        file_path = os.path.join(base_dir, filename)
        with open(file_path, "w") as f:
            f.write(content)
    
    # Write app module files
    for filename, content in APP_FILES.items():
        file_path = os.path.join(base_dir, filename)
        with open(file_path, "w") as f:
            f.write(content)
    
    # Write core module files
    for filename, content in CORE_FILES.items():
        file_path = os.path.join(base_dir, filename)
        with open(file_path, "w") as f:
            f.write(content)
    
    # Additional modules would be written here in similar fashion

def main():
    """Main function to generate the project."""
    if len(sys.argv) < 2:
        print("Usage: python generate_project.py <output_directory>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create directory structure
    create_directory_structure(base_dir)
    
    # Write file contents
    write_file_contents(base_dir)
    
    print(f"Project generated at {os.path.abspath(base_dir)}")

if __name__ == "__main__":
    main()