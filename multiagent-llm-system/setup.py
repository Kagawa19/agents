from setuptools import setup, find_packages

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
