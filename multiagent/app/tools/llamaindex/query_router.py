"""
Query router for LlamaIndex.
Routes queries to appropriate indexes and retrieval methods.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class QueryRouter:
    """
    Routes queries to appropriate indexes and retrieval methods.
    Analyzes query intent and selects optimal retrieval strategy.
    """
    
    def __init__(
        self,
        tracer: Optional[LangfuseTracer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the query router.
        
        Args:
            tracer: Optional LangfuseTracer instance for monitoring
            config: Configuration parameters
        """
        self.tracer = tracer
        self.config = config or {}
        
        # Initialize index mappings
        self.index_mappings = self.config.get("index_mappings", {})
        
        # Initialize query classifiers
        self.classifiers = {
            "factual": self._is_factual_query,
            "comparison": self._is_comparison_query,
            "temporal": self._is_temporal_query,
            "conceptual": self._is_conceptual_query,
            "procedural": self._is_procedural_query
        }
    
    async def route_query(
        self,
        query: str,
        available_indexes: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route a query to appropriate indexes and retrieval methods.
        
        Args:
            query: Query string
            available_indexes: List of available index names
            context: Optional additional context
            
        Returns:
            Routing decision
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="route_query",
                input={"query": query, "available_indexes": available_indexes}
            )
        
        try:
            # Classify query intent
            query_type = self._classify_query(query)
            
            # Determine query complexity
            complexity = self._determine_complexity(query)
            
            # Select index based on query type and complexity
            selected_index = self._select_index(query, query_type, complexity, available_indexes)
            
            # Determine retrieval strategy
            retrieval_strategy = self._determine_retrieval_strategy(query_type, complexity)
            
            # Determine number of results to fetch
            top_k = self._determine_top_k(query_type, complexity)
            
            # Create routing decision
            result = {
                "query": query,
                "query_type": query_type,
                "complexity": complexity,
                "selected_index": selected_index,
                "retrieval_strategy": retrieval_strategy,
                "top_k": top_k
            }
            
            if span:
                span.update(output=result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error routing query: {str(e)}")
            
            # Default routing
            default_result = {
                "query": query,
                "query_type": "unknown",
                "complexity": "medium",
                "selected_index": available_indexes[0] if available_indexes else None,
                "retrieval_strategy": "semantic",
                "top_k": 5
            }
            
            if span:
                span.update(output={"error": str(e), "default_result": default_result})
            
            return default_result
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the query type based on intent.
        
        Args:
            query: Query string
            
        Returns:
            Query type
        """
        # Check each query type
        for query_type, classifier in self.classifiers.items():
            if classifier(query):
                return query_type
        
        # Default to factual
        return "factual"
    
    def _is_factual_query(self, query: str) -> bool:
        """Check if query is seeking facts or information."""
        factual_patterns = [
            r"what is",
            r"who is",
            r"where is",
            r"when did",
            r"how many",
            r"define",
            r"explain",
            r"tell me about"
        ]
        
        return any(re.search(pattern, query.lower()) for pattern in factual_patterns)
    
    def _is_comparison_query(self, query: str) -> bool:
        """Check if query is comparing multiple items."""
        comparison_patterns = [
            r"compare",
            r"difference between",
            r"similarities between",
            r"versus",
            r"vs",
            r"better than",
            r"worse than",
            r"pros and cons"
        ]
        
        return any(re.search(pattern, query.lower()) for pattern in comparison_patterns)
    
    def _is_temporal_query(self, query: str) -> bool:
        """Check if query has a temporal component."""
        temporal_patterns = [
            r"when",
            r"history of",
            r"timeline",
            r"evolution of",
            r"development of",
            r"in what year",
            r"how long",
            r"time period"
        ]
        
        return any(re.search(pattern, query.lower()) for pattern in temporal_patterns)
    
    def _is_conceptual_query(self, query: str) -> bool:
        """Check if query is about abstract concepts."""
        conceptual_patterns = [
            r"concept of",
            r"theory of",
            r"principles of",
            r"philosophy of",
            r"idea of",
            r"meaning of",
            r"implications of",
            r"why is"
        ]
        
        return any(re.search(pattern, query.lower()) for pattern in conceptual_patterns)
    
    def _is_procedural_query(self, query: str) -> bool:
        """Check if query is asking for steps or procedures."""
        procedural_patterns = [
            r"how to",
            r"steps to",
            r"process of",
            r"procedure for",
            r"instructions for",
            r"guide to",
            r"method for",
            r"approach to"
        ]
        
        return any(re.search(pattern, query.lower()) for pattern in procedural_patterns)
    
    def _determine_complexity(self, query: str) -> str:
        """
        Determine the complexity of the query.
        
        Args:
            query: Query string
            
        Returns:
            Complexity level (low, medium, high)
        """
        # Count words
        word_count = len(query.split())
        
        # Count entities (simplified)
        entity_patterns = [
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",  # Proper nouns
            r"\d+",  # Numbers
            r"[A-Z]{2,}"  # Acronyms
        ]
        
        entity_count = sum(len(re.findall(pattern, query)) for pattern in entity_patterns)
        
        # Check for complex structures
        complex_markers = [
            r"and.*(?:also|additionally)",
            r"not only.*but also",
            r"if.*then",
            r"because.*therefore",
            r"although.*nevertheless"
        ]
        
        complexity_score = sum(1 for pattern in complex_markers if re.search(pattern, query.lower()))
        
        # Determine complexity
        if word_count <= 5 and entity_count <= 1 and complexity_score == 0:
            return "low"
        elif word_count >= 15 or entity_count >= 3 or complexity_score >= 2:
            return "high"
        else:
            return "medium"
    
    def _select_index(
        self,
        query: str,
        query_type: str,
        complexity: str,
        available_indexes: List[str]
    ) -> str:
        """
        Select the best index for the query.
        
        Args:
            query: Query string
            query_type: Type of query
            complexity: Complexity level
            available_indexes: List of available indexes
            
        Returns:
            Selected index name
        """
        # Check if we have a mapping for this query type
        if query_type in self.index_mappings:
            type_mapping = self.index_mappings[query_type]
            if complexity in type_mapping:
                index_name = type_mapping[complexity]
                if index_name in available_indexes:
                    return index_name
        
        # Default to first available index
        if "general" in available_indexes:
            return "general"
        elif available_indexes:
            return available_indexes[0]
        else:
            return "default"
    
    def _determine_retrieval_strategy(
        self,
        query_type: str,
        complexity: str
    ) -> str:
        """
        Determine the best retrieval strategy for the query.
        
        Args:
            query_type: Type of query
            complexity: Complexity level
            
        Returns:
            Retrieval strategy
        """
        # Strategy mapping
        strategies = {
            "factual": {
                "low": "semantic",
                "medium": "hybrid",
                "high": "hybrid"
            },
            "comparison": {
                "low": "hybrid",
                "medium": "hybrid",
                "high": "mmr"
            },
            "temporal": {
                "low": "semantic",
                "medium": "hybrid",
                "high": "hybrid"
            },
            "conceptual": {
                "low": "semantic",
                "medium": "semantic",
                "high": "mmr"
            },
            "procedural": {
                "low": "hybrid",
                "medium": "hybrid",
                "high": "mmr"
            }
        }
        
        # Get strategy or default to semantic
        return strategies.get(query_type, {}).get(complexity, "semantic")
    
    def _determine_top_k(
        self,
        query_type: str,
        complexity: str
    ) -> int:
        """
        Determine the number of results to retrieve.
        
        Args:
            query_type: Type of query
            complexity: Complexity level
            
        Returns:
            Number of results to retrieve
        """
        # Top-k mapping
        top_k_values = {
            "factual": {
                "low": 3,
                "medium": 5,
                "high": 7
            },
            "comparison": {
                "low": 5,
                "medium": 7,
                "high": 10
            },
            "temporal": {
                "low": 3,
                "medium": 5,
                "high": 7
            },
            "conceptual": {
                "low": 5,
                "medium": 7,
                "high": 10
            },
            "procedural": {
                "low": 3,
                "medium": 5,
                "high": 7
            }
        }
        
        # Get top-k or default to 5
        return top_k_values.get(query_type, {}).get(complexity, 5)
    
    async def rewrite_query(
        self,
        query: str,
        llm_client: Any
    ) -> str:
        """
        Rewrite the query to improve retrieval performance.
        
        Args:
            query: Original query
            llm_client: LLM client for query rewriting
            
        Returns:
            Rewritten query
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="rewrite_query",
                input={"query": query}
            )
        
        try:
            # Prepare prompt for query rewriting
            prompt = f"""
            Please rewrite the following query to make it more effective for information retrieval.
            Expand any acronyms, add synonyms for important terms, and make it more specific.
            Original query: {query}
            Rewritten query:
            """
            
            # Generate rewritten query
            if hasattr(llm_client, "generate_text"):
                response = await llm_client.generate_text(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.3
                )
                rewritten_query = response.get("text", query).strip()
            else:
                logger.warning("LLM client does not have generate_text method")
                rewritten_query = query
            
            # Fallback to original query if rewrite is too short
            if len(rewritten_query) < len(query) / 2:
                rewritten_query = query
            
            if span:
                span.update(output={"rewritten_query": rewritten_query})
            
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return query
    
    async def generate_query_variations(
        self,
        query: str,
        llm_client: Any,
        num_variations: int = 3
    ) -> List[str]:
        """
        Generate variations of the query for better recall.
        
        Args:
            query: Original query
            llm_client: LLM client for generating variations
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="generate_query_variations",
                input={"query": query, "num_variations": num_variations}
            )
        
        try:
            # Prepare prompt for query variations
            prompt = f"""
            Please generate {num_variations} different variations of the following query.
            Each variation should express the same information need but using different words, synonyms, or phrasings.
            
            Original query: {query}
            
            Variations:
            1.
            """
            
            # Generate variations
            if hasattr(llm_client, "generate_text"):
                response = await llm_client.generate_text(
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.7
                )
                variations_text = response.get("text", "").strip()
                
                # Parse variations
                variations = []
                for line in variations_text.split("\n"):
                    # Remove leading numbers and punctuation
                    clean_line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                    if clean_line and clean_line != query:
                        variations.append(clean_line)
                
                # Ensure we have at least the original query
                if not variations:
                    variations = [query]
                
                # Limit to requested number
                variations = variations[:num_variations]
                
            else:
                logger.warning("LLM client does not have generate_text method")
                variations = [query]
            
            if span:
                span.update(output={"variations": variations})
            
            return variations
            
        except Exception as e:
            logger.error(f"Error generating query variations: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return [query]