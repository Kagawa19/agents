# Multiagent LLM System

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
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {your_token}" \
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
