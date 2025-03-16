"""
Model selection logic for Amazon Bedrock.
Handles model selection, versioning, and capability matching.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class BedrockModels:
    """
    Model selection utilities for Amazon Bedrock.
    Handles model selection, versioning, and capability matching.
    """
    
    def __init__(
        self,
        tracer: Optional[LangfuseTracer] = None
    ):
        """
        Initialize Bedrock model selection utilities.
        
        Args:
            tracer: Optional LangfuseTracer instance for monitoring
        """
        self.tracer = tracer
        
        # Define available models
        self.models = {
            # Anthropic Claude models
            "claude": {
                "id": "anthropic.claude-v2",
                "provider": "anthropic",
                "family": "claude",
                "version": "2.0",
                "capabilities": ["text-generation", "qa", "summarization", "reasoning"],
                "max_tokens": 100000,
                "input_token_limit": 100000
            },
            "claude-instant": {
                "id": "anthropic.claude-instant-v1",
                "provider": "anthropic",
                "family": "claude",
                "version": "instant-1.0",
                "capabilities": ["text-generation", "qa", "summarization"],
                "max_tokens": 100000,
                "input_token_limit": 100000
            },
            "claude-2": {
                "id": "anthropic.claude-v2",
                "provider": "anthropic",
                "family": "claude",
                "version": "2.0",
                "capabilities": ["text-generation", "qa", "summarization", "reasoning"],
                "max_tokens": 100000,
                "input_token_limit": 100000
            },
            "claude-3-sonnet": {
                "id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "provider": "anthropic",
                "family": "claude",
                "version": "3-sonnet",
                "capabilities": ["text-generation", "qa", "summarization", "reasoning", "coding"],
                "max_tokens": 200000,
                "input_token_limit": 200000
            },
            
            # Amazon Titan models
            "titan-text": {
                "id": "amazon.titan-text-express-v1",
                "provider": "amazon",
                "family": "titan",
                "version": "express-v1",
                "capabilities": ["text-generation", "qa"],
                "max_tokens": 8000,
                "input_token_limit": 8000
            },
            "titan-embed": {
                "id": "amazon.titan-embed-text-v1",
                "provider": "amazon",
                "family": "titan",
                "version": "embed-text-v1",
                "capabilities": ["embeddings"],
                "max_tokens": 8000,
                "input_token_limit": 8000
            },
            
            # Meta Llama models
            "llama2-13b": {
                "id": "meta.llama2-13b-chat-v1",
                "provider": "meta",
                "family": "llama",
                "version": "2-13b",
                "capabilities": ["text-generation", "qa"],
                "max_tokens": 4096,
                "input_token_limit": 4096
            },
            "llama2-70b": {
                "id": "meta.llama2-70b-chat-v1",
                "provider": "meta",
                "family": "llama",
                "version": "2-70b",
                "capabilities": ["text-generation", "qa", "reasoning"],
                "max_tokens": 4096,
                "input_token_limit": 4096
            },
            
            # AI21 Jurassic models
            "jurassic-2": {
                "id": "ai21.j2-mid-v1",
                "provider": "ai21",
                "family": "jurassic",
                "version": "2-mid",
                "capabilities": ["text-generation", "qa"],
                "max_tokens": 8192,
                "input_token_limit": 8192
            },
            "jurassic-2-ultra": {
                "id": "ai21.j2-ultra-v1",
                "provider": "ai21",
                "family": "jurassic",
                "version": "2-ultra",
                "capabilities": ["text-generation", "qa", "summarization"],
                "max_tokens": 8192,
                "input_token_limit": 8192
            },
            
            # Cohere models
            "cohere-command": {
                "id": "cohere.command-text-v14",
                "provider": "cohere",
                "family": "command",
                "version": "text-v14",
                "capabilities": ["text-generation", "qa"],
                "max_tokens": 4096,
                "input_token_limit": 4096
            },
            "cohere-embed": {
                "id": "cohere.embed-english-v3",
                "provider": "cohere",
                "family": "embed",
                "version": "english-v3",
                "capabilities": ["embeddings"],
                "max_tokens": 4096,
                "input_token_limit": 4096
            }
        }
        
        # Define task-specific model recommendations
        self.task_recommendations = {
            "text-generation": ["claude-3-sonnet", "claude-2", "llama2-70b", "titan-text"],
            "qa": ["claude-3-sonnet", "claude-2", "llama2-70b", "jurassic-2-ultra"],
            "summarization": ["claude-3-sonnet", "claude-2", "jurassic-2-ultra"],
            "reasoning": ["claude-3-sonnet", "claude-2", "llama2-70b"],
            "coding": ["claude-3-sonnet", "claude-2"],
            "embeddings": ["titan-embed", "cohere-embed"]
        }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary
        """
        # Handle variations in model names
        model_name = model_name.lower()
        
        # Direct match
        if model_name in self.models:
            return self.models[model_name]
        
        # Try to match by prefix
        for name, info in self.models.items():
            if model_name.startswith(name):
                return info
            # Match by ID
            if model_name == info["id"]:
                return info
        
        # Default to Claude if not found
        logger.warning(f"Model '{model_name}' not recognized, defaulting to Claude")
        return self.models["claude"]
    
    def get_model_id(self, model_name: str) -> str:
        """
        Get the AWS Bedrock model ID for a model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            AWS Bedrock model ID
        """
        model_info = self.get_model_info(model_name)
        return model_info["id"]
    
    def recommend_model(
        self,
        task: str,
        provider: Optional[str] = None,
        min_tokens: int = 0
    ) -> str:
        """
        Recommend a model for a specific task.
        
        Args:
            task: Type of task (text-generation, qa, summarization, etc.)
            provider: Optional provider constraint (anthropic, amazon, meta, etc.)
            min_tokens: Minimum token capacity required
            
        Returns:
            Recommended model name
        """
        # Get recommended models for task
        if task not in self.task_recommendations:
            logger.warning(f"Task '{task}' not recognized, defaulting to text-generation")
            task = "text-generation"
        
        recommendations = self.task_recommendations[task]
        
        # Filter by provider if specified
        if provider:
            provider = provider.lower()
            filtered_models = []
            
            for model_name in recommendations:
                model_info = self.get_model_info(model_name)
                if model_info["provider"] == provider:
                    filtered_models.append(model_name)
            
            if filtered_models:
                recommendations = filtered_models
        
        # Filter by token capacity
        if min_tokens > 0:
            valid_models = []
            
            for model_name in recommendations:
                model_info = self.get_model_info(model_name)
                if model_info["input_token_limit"] >= min_tokens:
                    valid_models.append(model_name)
            
            if valid_models:
                return valid_models[0]
        
        # Return top recommendation if available, otherwise default to Claude
        return recommendations[0] if recommendations else "claude"
    
    def check_model_compatibility(
        self,
        model_name: str,
        task: str
    ) -> bool:
        """
        Check if a model is compatible with a given task.
        
        Args:
            model_name: Name of the model
            task: Type of task
            
        Returns:
            True if compatible, False otherwise
        """
        model_info = self.get_model_info(model_name)
        return task in model_info["capabilities"]
    
    def get_available_models(
        self,
        provider: Optional[str] = None,
        task: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get a list of available models with optional filtering.
        
        Args:
            provider: Optional provider filter
            task: Optional task capability filter
            
        Returns:
            List of model information dictionaries
        """
        # Start with all models
        available_models = list(self.models.values())
        
        # Filter by provider if specified
        if provider:
            provider = provider.lower()
            available_models = [
                model for model in available_models
                if model["provider"] == provider
            ]
        
        # Filter by task capability if specified
        if task:
            available_models = [
                model for model in available_models
                if task in model["capabilities"]
            ]
        
        return available_models
    
    def get_model_families(self) -> Dict[str, List[str]]:
        """
        Get a mapping of model families to model names.
        
        Returns:
            Dictionary mapping family names to lists of model names
        """
        families = {}
        
        for name, info in self.models.items():
            family = info["family"]
            if family not in families:
                families[family] = []
            families[family].append(name)
        
        return families
    
    def get_provider_models(self, provider: str) -> List[str]:
        """
        Get all models from a specific provider.
        
        Args:
            provider: Provider name (anthropic, amazon, meta, etc.)
            
        Returns:
            List of model names
        """
        provider = provider.lower()
        return [
            name for name, info in self.models.items()
            if info["provider"] == provider
        ]
    
    def get_model_by_capability(self, capability: str) -> List[str]:
        """
        Get all models with a specific capability.
        
        Args:
            capability: Capability name
            
        Returns:
            List of model names
        """
        return [
            name for name, info in self.models.items()
            if capability in info["capabilities"]
        ]
    
    def format_prompt(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format a prompt according to the model's expected input format.
        
        Args:
            model_name: Name of the model
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt in model-specific format
        """
        model_info = self.get_model_info(model_name)
        provider = model_info["provider"]
        
        # Format based on provider
        if provider == "anthropic":
            # Claude format
            formatted_prompt = {"prompt": ""}
            
            if system_prompt:
                formatted_prompt["prompt"] += f"\n\nHuman: <system>\n{system_prompt}\n</system>\n\n{prompt}\n\nAssistant:"
            else:
                formatted_prompt["prompt"] += f"\n\nHuman: {prompt}\n\nAssistant:"
                
            return formatted_prompt
            
        elif provider == "amazon":
            # Titan format
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 4096,
                    "stopSequences": [],
                    "temperature": 0.7,
                    "topP": 0.9
                }
            }
            
        elif provider == "meta":
            # Llama format
            if system_prompt:
                return {
                    "prompt": f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
                }
            else:
                return {
                    "prompt": f"<s>[INST] {prompt} [/INST]"
                }
                
        elif provider == "ai21":
            # Jurassic format
            return {
                "prompt": prompt,
                "maxTokens": 4096,
                "temperature": 0.7,
                "topP": 0.9,
                "stopSequences": [],
                "countPenalty": {"scale": 0},
                "presencePenalty": {"scale": 0},
                "frequencyPenalty": {"scale": 0}
            }
            
        elif provider == "cohere":
            # Cohere format
            return {
                "prompt": prompt,
                "max_tokens": 4096,
                "temperature": 0.7,
                "p": 0.9,
                "k": 0,
                "stop_sequences": [],
                "return_likelihoods": "NONE"
            }
            
        # Default format for unknown providers
        return {"prompt": prompt}