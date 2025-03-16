"""
Analyzer agent implementation.
Responsible for analyzing and extracting insights from gathered information.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import time
import json
import asyncio
from datetime import datetime

from multiagent.app.agents.base import BaseAgent
from multiagent.app.tools.jina.jina_index import JinaIndex
from multiagent.app.tools.jina.jina_search import JinaSearch
from multiagent.app.tools.jina.jina_extract import JinaExtract
from multiagent.app.processors.jina_processor import JinaTaskProcessor
from multiagent.app.monitoring.tracer import LangfuseTracer
from multiagent.app.tools.serper import SerperTool
from multiagent.app.tools.scraper import WebScraper


logger = logging.getLogger(__name__)


class AnalyzerAgent(BaseAgent):
    """
    Agent for analyzing information and extracting insights.
    Uses Jina AI for semantic analysis and pattern recognition.
    Can use either OpenAI or Bedrock (Claude) for generative tasks.
    """
    
    def __init__(
        self,
        agent_id: str,
        tracer: LangfuseTracer,
        jina_tool: Any,
        openai_tool: Any,
        bedrock_tool: Optional[Any] = None,
        serper_tool: Optional[SerperTool] = None,
        scraper_tool: Optional[WebScraper] = None
    ):
        """
        Initialize the analyzer agent.
        
        Args:
            agent_id: Unique identifier for the agent
            tracer: LangfuseTracer instance for monitoring
            jina_tool: Jina AI tool for vector operations
            openai_tool: OpenAI tool for analysis capabilities
            bedrock_tool: Bedrock tool for alternative AI provider (Claude)
            serper_tool: Optional SerperTool for web search
            scraper_tool: Optional WebScraper for content extraction
        """
        super().__init__(agent_id=agent_id, tracer=tracer)
        self.jina_tool = jina_tool
        self.openai_tool = openai_tool
        self.bedrock_tool = bedrock_tool
        self.serper_tool = serper_tool
        self.scraper_tool = scraper_tool
        self.processor = None
        self.config = {}
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the analyzer with configuration parameters.
        
        Args:
            config: Configuration parameters for the analyzer
        """
        self.config = config
        
        # Initialize the Jina processor
        self.processor = JinaTaskProcessor(
            api_key=config.get("jina_api_key", ""),
            index_name=config.get("jina_index", "analyzer_index"),
            tracer=self.tracer
        )
        
        # Set up analysis parameters
        self.min_confidence = config.get("min_confidence", 0.65)
        self.max_insights = config.get("max_insights", 10)
        self.semantic_similarity_threshold = config.get("semantic_similarity", 0.75)
        
        # AI provider preferences
        self.preferred_ai_provider = config.get("preferred_ai_provider", "auto")  # "auto", "openai", or "bedrock"
        
        logger.info(f"Initialized {self.agent_id} with config: {json.dumps(config, default=str)}")
    
    async def get_ai_tool(self, task_type: str = "general") -> Tuple[Any, str]:
        """
        Get the appropriate AI tool based on configuration and availability.
        
        Args:
            task_type: Type of task ("summary", "pattern", "insight", "general")
            
        Returns:
            Tuple of (ai_tool, provider_name)
        """
        # Check for specific task preferences in config
        task_preference = self.config.get(f"preferred_provider_{task_type}", self.preferred_ai_provider)
        
        # Logic for provider selection
        if task_preference == "bedrock" and self.bedrock_tool:
            return self.bedrock_tool, "bedrock"
        elif task_preference == "openai" and self.openai_tool:
            return self.openai_tool, "openai"
        elif task_preference == "auto":
            # Auto-select based on task type and availability
            if task_type in ["summary", "insight"] and self.bedrock_tool:
                # Claude is better for these tasks
                return self.bedrock_tool, "bedrock"
            elif self.openai_tool:
                return self.openai_tool, "openai"
            elif self.bedrock_tool:
                return self.bedrock_tool, "bedrock"
        
        # Fall back to whatever is available
        if self.openai_tool:
            return self.openai_tool, "openai"
        elif self.bedrock_tool:
            return self.bedrock_tool, "bedrock"
        
        # No tool available
        raise ValueError(f"No AI tool available for task type: {task_type}")
    
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 500, 
        temperature: float = 0.4,
        task_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate text using the appropriate AI provider.
        
        Args:
            prompt: The text prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            task_type: Type of task for provider selection
            
        Returns:
            Response containing the generated text
        """
        ai_tool, provider = await self.get_ai_tool(task_type)
        
        try:
            if provider == "bedrock":
                # Use Claude via Bedrock
                response = await ai_tool.generate_text(
                    model="claude",  # Specify Claude model
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                # Use OpenAI
                response = await ai_tool.generate_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            return response
        except Exception as e:
            logger.error(f"Error generating text with {provider}: {str(e)}")
            
            # Try fallback if available
            if provider == "bedrock" and self.openai_tool:
                logger.info("Falling back to OpenAI")
                return await self.openai_tool.generate_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            elif provider == "openai" and self.bedrock_tool:
                logger.info("Falling back to Bedrock")
                return await self.bedrock_tool.generate_text(
                    model="claude",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                # Re-raise if no fallback available
                raise
    
    async def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and preprocess the input data.
        
        Args:
            input_data: Input data for analysis
            
        Returns:
            Validated and preprocessed input data
            
        Raises:
            ValueError: If required parameters are missing
        """
        # Check for required parameters
        if "query" not in input_data:
            raise ValueError("Missing required parameter 'query'")
        
        # Handle search_web flag
        if input_data.get("search_web", False) and not input_data.get("information"):
            # We'll search the web later, so no need to validate information yet
            return input_data
            
        if "information" not in input_data:
            raise ValueError("Missing required parameter 'information'")
            
        # Ensure information is in the expected format
        if not isinstance(input_data["information"], list):
            if isinstance(input_data["information"], dict):
                # Convert to list if it's a dictionary
                input_data["information"] = [input_data["information"]]
            else:
                # Try to parse as JSON if it's a string
                try:
                    parsed_info = json.loads(input_data["information"])
                    if isinstance(parsed_info, list):
                        input_data["information"] = parsed_info
                    else:
                        input_data["information"] = [parsed_info]
                except Exception:
                    # If parsing fails, wrap in a list
                    input_data["information"] = [{"content": str(input_data["information"])}]
        
        # Ensure each information item has required fields
        for i, item in enumerate(input_data["information"]):
            if "content" not in item and "text" not in item:
                # Try to use the whole item as content if no content field
                input_data["information"][i] = {"content": str(item)}
        
        return input_data
    
    async def search_and_scrape(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web and scrape content for the given query.
        
        Args:
            query: Search query
            num_results: Number of results to fetch
            
        Returns:
            List of scraped content items
        """
        with self.tracer.span(name="search_and_scrape") as span:
            logger.info(f"Searching and scraping for query: {query}")
            
            # Search using SerperTool
            if self.serper_tool is None:
                raise ValueError("SerperTool is required for search_and_scrape")
                
            search_results = await self.serper_tool.search(query, num_results=num_results)
            
            # Scrape content from search results
            if self.scraper_tool is None:
                raise ValueError("WebScraper is required for search_and_scrape")
                
            scraped_content = []
            for result in search_results:
                url = result.get("link")
                if url:
                    try:
                        content = await self.scraper_tool.scrape(url)
                        if content:
                            scraped_content.append({
                                "content": content,
                                "metadata": {
                                    "url": url,
                                    "title": result.get("title", ""),
                                    "snippet": result.get("snippet", "")
                                }
                            })
                    except Exception as e:
                        logger.warning(f"Error scraping {url}: {str(e)}")
                        
            logger.info(f"Scraped {len(scraped_content)} documents from search results")
            return scraped_content
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the analysis on the provided information.
        
        Args:
            input_data: Dictionary containing:
                - query: The original query
                - information: List of information items to analyze (optional if search_web=True)
                - search_web: Boolean indicating whether to search the web (default: False)
                - num_results: Number of search results to fetch (default: 5)
                
        Returns:
            Dictionary containing:
                - analysis_results: Dictionary of insights and patterns
                - confidence_scores: Confidence levels for different findings
                - methodology: Description of analysis approach
                - processing_time: Time taken for analysis
        """
        start_time = time.time()
        logger.info(f"Starting analysis for query: {input_data.get('query', '')[:50]}...")
        
        # Track execution with a trace span
        with self.tracer.span(name="analyzer_execute") as span:
            span.update(input=input_data)
            
            try:
                # Validate and preprocess the input
                validated_input = await self.validate_input(input_data)
                query = validated_input["query"]
                
                # Check if we should search the web
                if validated_input.get("search_web", False) and not validated_input.get("information"):
                    # Search and scrape the web
                    information = await self.search_and_scrape(
                        query, 
                        num_results=validated_input.get("num_results", 5)
                    )
                    validated_input["information"] = information
                
                information = validated_input["information"]
                
                # Process the information using Jina
                processed_data = await self.process_data(information)
                
                # Perform statistical analysis
                statistical_analysis = await self.perform_statistical_analysis(processed_data)
                
                # Perform semantic analysis
                semantic_analysis = await self.perform_semantic_analysis(processed_data, query)
                
                # Extract patterns and insights
                patterns, insights = await self.extract_patterns_and_insights(processed_data, query)
                
                # Calculate confidence scores
                confidence_scores = self.calculate_confidence_scores(
                    statistical_analysis,
                    semantic_analysis,
                    patterns,
                    insights
                )
                
                # Create structured output
                analysis_results = {
                    "summary": await self.generate_analysis_summary(query, insights, patterns),
                    "statistical_analysis": statistical_analysis,
                    "semantic_analysis": semantic_analysis,
                    "patterns": patterns,
                    "insights": insights,
                    "source_count": len(information),
                    "analyzed_at": datetime.utcnow().isoformat()
                }
                
                # Create methodology description
                methodology = self.describe_methodology(
                    statistical_analysis,
                    semantic_analysis,
                    patterns,
                    insights
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Build final result
                result = {
                    "analysis_results": analysis_results,
                    "confidence_scores": confidence_scores,
                    "methodology": methodology,
                    "processing_time": processing_time,
                    "status": "completed"
                }
                
                logger.info(f"Analysis completed in {processing_time:.2f} seconds")
                span.update(output=result)
                return result
                
            except Exception as e:
                error_msg = f"Error in analyzer execution: {str(e)}"
                logger.error(error_msg)
                span.update(output={"status": "error", "error": error_msg})
                
                # Return error result
                return {
                    "status": "error",
                    "error": error_msg,
                    "processing_time": time.time() - start_time
                }
    
    async def process_data(self, information: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process the information using Jina tools.
        
        Args:
            information: List of information items to process
            
        Returns:
            Processed data with vectors and metadata
        """
        with self.tracer.span(name="jina_data_processing") as span:
            logger.info(f"Processing {len(information)} information items with Jina")
            
            # Prepare documents for processing
            documents = []
            for item in information:
                content = item.get("content") or item.get("text", "")
                metadata = {k: v for k, v in item.items() if k not in ["content", "text"]}
                documents.append({
                    "content": content,
                    "metadata": metadata
                })
            
            # Process through JinaTaskProcessor
            processed_documents = await self.processor.process(
                documents=documents,
                operation="analyze",
                index=True
            )
            
            # Add vector embeddings using JinaExtract if not already present
            if not any("embedding" in doc for doc in processed_documents):
                try:
                    # Extract embeddings for each document
                    contents = [doc.get("content", "") for doc in processed_documents]
                    embeddings = await self.jina_tool.extract.get_embeddings(contents)
                    
                    # Add embeddings to the processed documents
                    for i, embedding in enumerate(embeddings):
                        if i < len(processed_documents):
                            processed_documents[i]["embedding"] = embedding
                except Exception as e:
                    logger.warning(f"Failed to extract embeddings: {str(e)}")
            
            logger.info(f"Processed {len(processed_documents)} documents with Jina")
            return processed_documents
    
    async def perform_statistical_analysis(self, processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform statistical analysis on the processed data.
        
        Args:
            processed_data: List of processed data items
            
        Returns:
            Statistical analysis results
        """
        with self.tracer.span(name="statistical_analysis"):
            logger.info("Performing statistical analysis")
            
            # Extract text for analysis
            texts = [item.get("content", "") for item in processed_data]
            
            # Calculate basic text statistics
            word_counts = [len(text.split()) for text in texts]
            char_counts = [len(text) for text in texts]
            
            # Calculate frequency distributions if there are enough documents
            term_frequency = {}
            if texts:
                # Combine all texts and tokenize
                all_text = " ".join(texts).lower()
                words = all_text.split()
                
                # Count term frequencies
                for word in words:
                    if len(word) > 3:  # Skip very short words
                        term_frequency[word] = term_frequency.get(word, 0) + 1
                
                # Keep only the top terms
                top_terms = sorted(term_frequency.items(), key=lambda x: x[1], reverse=True)[:50]
                term_frequency = {term: count for term, count in top_terms}
            
            # Extract metadata statistics if available
            metadata_stats = {}
            for item in processed_data:
                metadata = item.get("metadata", {})
                for key, value in metadata.items():
                    if key not in metadata_stats:
                        metadata_stats[key] = []
                    metadata_stats[key].append(value)
            
            # Analyze metadata distributions
            metadata_analysis = {}
            for key, values in metadata_stats.items():
                if all(isinstance(v, (int, float)) for v in values if v is not None):
                    # Numeric values - calculate statistics
                    numeric_values = [v for v in values if v is not None and isinstance(v, (int, float))]
                    if numeric_values:
                        metadata_analysis[key] = {
                            "type": "numeric",
                            "min": min(numeric_values),
                            "max": max(numeric_values),
                            "avg": sum(numeric_values) / len(numeric_values),
                            "count": len(numeric_values)
                        }
                else:
                    # Categorical values - count frequencies
                    frequencies = {}
                    for v in values:
                        if v is not None:
                            v_str = str(v)
                            frequencies[v_str] = frequencies.get(v_str, 0) + 1
                    
                    metadata_analysis[key] = {
                        "type": "categorical",
                        "frequencies": frequencies,
                        "most_common": max(frequencies.items(), key=lambda x: x[1])[0] if frequencies else None,
                        "count": len(values)
                    }
            
            # Return the statistical analysis results
            return {
                "document_count": len(processed_data),
                "avg_word_count": sum(word_counts) / len(word_counts) if word_counts else 0,
                "total_words": sum(word_counts),
                "avg_char_count": sum(char_counts) / len(char_counts) if char_counts else 0,
                "term_frequency": term_frequency,
                "metadata_analysis": metadata_analysis
            }
    
    async def perform_semantic_analysis(
        self,
        processed_data: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        Perform semantic analysis on the processed data.
        
        Args:
            processed_data: List of processed data items
            query: The original query for context
            
        Returns:
            Semantic analysis results
        """
        with self.tracer.span(name="semantic_analysis"):
            logger.info("Performing semantic analysis")
            
            # Extract embeddings if available
            embeddings = [item.get("embedding") for item in processed_data if "embedding" in item]
            
            results = {
                "semantic_clusters": [],
                "query_relevance": [],
                "key_concepts": []
            }
            
            # Skip detailed analysis if not enough embeddings
            if len(embeddings) < 2:
                logger.warning("Not enough embeddings for semantic analysis")
                return results
            
            try:
                # Get embedding for the query
                query_embedding = await self.jina_tool.extract.get_embeddings([query])
                
                # Calculate semantic similarity to query
                if query_embedding and embeddings:
                    similarities = await self.jina_tool.search.calculate_similarities(
                        query_embedding[0],
                        embeddings
                    )
                    
                    # Create query relevance scores
                    for i, similarity in enumerate(similarities):
                        if i < len(processed_data):
                            content_preview = processed_data[i].get("content", "")[:100]
                            results["query_relevance"].append({
                                "index": i,
                                "content_preview": content_preview,
                                "similarity": float(similarity),
                                "is_relevant": similarity >= self.semantic_similarity_threshold
                            })
                
                # Perform clustering on embeddings
                clusters = await self.jina_tool.search.cluster_embeddings(
                    embeddings,
                    n_clusters=min(5, len(embeddings) // 2) if len(embeddings) > 4 else 2
                )
                
                # Create semantic clusters
                for cluster_id, indices in clusters.items():
                    cluster_docs = [processed_data[i] for i in indices if i < len(processed_data)]
                    if cluster_docs:
                        # Extract representative text for this cluster
                        texts = [doc.get("content", "")[:200] for doc in cluster_docs]
                        
                        # Generate cluster label using AI
                        cluster_label = await self.generate_cluster_label(texts)
                        
                        results["semantic_clusters"].append({
                            "cluster_id": cluster_id,
                            "document_indices": indices,
                            "document_count": len(indices),
                            "label": cluster_label,
                            "sample_text": texts[0] if texts else ""
                        })
                
                # Extract key concepts using AI
                all_text = "\n".join([item.get("content", "")[:500] for item in processed_data[:10]])
                key_concepts = await self.extract_key_concepts(all_text, query)
                results["key_concepts"] = key_concepts
                
                return results
                
            except Exception as e:
                logger.error(f"Error in semantic analysis: {str(e)}")
                return results
    
    async def extract_patterns_and_insights(
        self,
        processed_data: List[Dict[str, Any]],
        query: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract patterns and insights from the processed data.
        
        Args:
            processed_data: List of processed data items
            query: The original query for context
            
        Returns:
            Tuple of (patterns, insights)
        """
        with self.tracer.span(name="extract_patterns_insights"):
            logger.info("Extracting patterns and insights")
            
            # Prepare text chunks for analysis
            text_chunks = []
            for item in processed_data:
                content = item.get("content", "")
                if content:
                    # Split content into manageable chunks
                    text_chunks.extend(self._split_text(content, max_chunk_size=1000))
            
            # Extract patterns
            patterns = await self.identify_patterns(text_chunks, query)
            
            # Generate insights based on patterns and data
            insights = await self.generate_insights(text_chunks, patterns, query)
            
            return patterns, insights
    
    async def identify_patterns(self, text_chunks: List[str], query: str) -> List[Dict[str, Any]]:
        """
        Identify patterns in the text chunks.
        
        Args:
            text_chunks: List of text chunks to analyze
            query: Original query for context
            
        Returns:
            List of identified patterns
        """
        # Combine a sample of text chunks for pattern analysis
        sample_text = "\n\n".join(text_chunks[:10]) if len(text_chunks) > 10 else "\n\n".join(text_chunks)
        
        # Generate patterns using AI
        prompt = f"""
        Analyze the following text and identify 3-5 key patterns related to the query: "{query}"
        
        For each pattern:
        1. Provide a clear pattern name
        2. Describe the pattern in detail
        3. Give specific evidence from the text
        4. Assign a confidence score (0.0-1.0) based on the strength of evidence
        
        Text to analyze:
        {sample_text[:4000]}
        """
        
        try:
            # Generate patterns using preferred AI provider
            response = await self.generate_text(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3,
                task_type="pattern"
            )
            
            # Parse the response into structured patterns
            raw_patterns = response.get("text", "")
            patterns = self._parse_patterns_from_text(raw_patterns)
            
            # Ensure proper structure for each pattern
            formatted_patterns = []
            for i, pattern in enumerate(patterns):
                formatted_pattern = {
                    "id": f"pattern_{i+1}",
                    "name": pattern.get("name", f"Unnamed Pattern {i+1}"),
                    "description": pattern.get("description", ""),
                    "evidence": pattern.get("evidence", ""),
                    "confidence": pattern.get("confidence", 0.7)
                }
                formatted_patterns.append(formatted_pattern)
            
            return formatted_patterns
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {str(e)}")
            return []
    
    async def generate_insights(
        self,
        text_chunks: List[str],
        patterns: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Generate insights based on text chunks and identified patterns.
        
        Args:
            text_chunks: List of text chunks
            patterns: Previously identified patterns
            query: Original query for context
            
        Returns:
            List of generated insights
        """
        # Combine patterns into a single string for the prompt
        patterns_text = ""
        for i, pattern in enumerate(patterns):
            patterns_text += f"Pattern {i+1}: {pattern.get('name', '')}\n"
            patterns_text += f"Description: {pattern.get('description', '')}\n\n"
        
        # Sample text for analysis
        sample_text = "\n\n".join(text_chunks[:5]) if text_chunks else ""
        
        # Generate insights using AI
        prompt = f"""
        Based on the following patterns and text samples related to the query: "{query}"
        
        Identified Patterns:
        {patterns_text}
        
        Text samples:
        {sample_text[:2000]}
        
        Generate 3-7 key insights that:
        1. Provide non-obvious conclusions
        2. Connect multiple pieces of information
        3. Answer the original query directly or indirectly
        4. Include specific supporting evidence
        5. Assign a confidence score (0.0-1.0) to each insight
        
        Format each insight with:
        - A clear insight title
        - A detailed explanation
        - Supporting evidence
        - Confidence score
        """
        
        try:
            # Generate insights using preferred AI provider (Claude via Bedrock works well for this)
            response = await self.generate_text(
                prompt=prompt,
                max_tokens=1200,
                temperature=0.4,
                task_type="insight"
            )
            
            # Parse the response into structured insights
            raw_insights = response.get("text", "")
            insights = self._parse_insights_from_text(raw_insights)
            
            # Ensure proper structure for each insight
            formatted_insights = []
            for i, insight in enumerate(insights):
                formatted_insight = {
                    "id": f"insight_{i+1}",
                    "title": insight.get("title", f"Unnamed Insight {i+1}"),
                    "explanation": insight.get("explanation", ""),
                    "evidence": insight.get("evidence", ""),
                    "confidence": insight.get("confidence", 0.7)
                }
                formatted_insights.append(formatted_insight)
            
            return formatted_insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []
    
    def calculate_confidence_scores(
        self,
        statistical_analysis: Dict[str, Any],
        semantic_analysis: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        insights: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate overall confidence scores for the analysis.
        
        Args:
            statistical_analysis: Statistical analysis results
            semantic_analysis: Semantic analysis results
            patterns: Identified patterns
            insights: Generated insights
            
        Returns:
            Dictionary of confidence scores
        """
        # Calculate average pattern confidence
        pattern_confidences = [p.get("confidence", 0) for p in patterns]
        avg_pattern_confidence = sum(pattern_confidences) / len(pattern_confidences) if pattern_confidences else 0
        
        # Calculate average insight confidence
        insight_confidences = [i.get("confidence", 0) for i in insights]
        avg_insight_confidence = sum(insight_confidences) / len(insight_confidences) if insight_confidences else 0
        
        # Calculate query relevance from semantic analysis
        query_relevances = [item.get("similarity", 0) for item in semantic_analysis.get("query_relevance", [])]
        avg_query_relevance = sum(query_relevances) / len(query_relevances) if query_relevances else 0
        
        # Calculate data quality score based on statistical analysis
        data_quality = min(1.0, (statistical_analysis.get("document_count", 0) / 10)) * 0.7 + 0.3
        
        # Calculate overall confidence
        overall_confidence = (
            avg_pattern_confidence * 0.25 +
            avg_insight_confidence * 0.35 +
            avg_query_relevance * 0.25 +
            data_quality * 0.15
        )
        
        return {
            "overall": overall_confidence,
            "patterns": avg_pattern_confidence,
            "insights": avg_insight_confidence,
            "query_relevance": avg_query_relevance,
            "data_quality": data_quality
        }
    
    def describe_methodology(
        self,
        statistical_analysis: Dict[str, Any],
        semantic_analysis: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        insights: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a description of the methodology used for analysis.
        
        Args:
            statistical_analysis: Statistical analysis results
            semantic_analysis: Semantic analysis results
            patterns: Identified patterns
            insights: Generated insights
            
        Returns:
            Description of the methodology
        """
        # Count documents analyzed
        doc_count = statistical_analysis.get("document_count", 0)
        
        # Count clusters identified
        cluster_count = len(semantic_analysis.get("semantic_clusters", []))
        
        # Calculate average confidence
        confidences = (
            [p.get("confidence", 0) for p in patterns] +
            [i.get("confidence", 0) for i in insights]
        )
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Determine which AI model was used
        ai_provider = "Claude (via Bedrock)" if self.preferred_ai_provider == "bedrock" else "OpenAI"
        if self.preferred_ai_provider == "auto":
            ai_provider = "Claude (via Bedrock) and OpenAI"
        
        return {
            "approach": "Multi-modal analysis combining statistical, semantic and pattern-based techniques",
            "tools_used": [
                "Jina AI for vector search and semantic analysis",
                f"{ai_provider} for pattern recognition and insight generation",
                "Statistical analysis for term frequency and metadata analysis",
                "Vectorization and clustering for topic identification"
            ],
            "data_processing": {
                "documents_analyzed": doc_count,
                "semantic_clusters_identified": cluster_count,
                "patterns_extracted": len(patterns),
                "insights_generated": len(insights)
            },
            "confidence_calculation": "Weighted average of pattern confidence, insight confidence, query relevance, and data quality",
            "average_confidence": avg_confidence,
            "limitations": [
                "Analysis limited to provided text content",
                "Semantic relationships may not capture industry-specific terminology",
                "Confidence scores are relative to the available data"
            ]
        }
    
    async def generate_analysis_summary(
        self,
        query: str,
        insights: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a summary of the analysis results.
        
        Args:
            query: Original query
            insights: Generated insights
            patterns: Identified patterns
            
        Returns:
            Summary text
        """
        # Extract top insights and patterns for the summary
        top_insights = sorted(insights, key=lambda x: x.get("confidence", 0), reverse=True)[:3]
        top_patterns = sorted(patterns, key=lambda x: x.get("confidence", 0), reverse=True)[:2]
        
        # Create summary points
        insight_points = "\n".join([
            f"- {insight.get('title', '')}: {insight.get('explanation', '')[:150]}..."
            for insight in top_insights
        ])
        
        pattern_points = "\n".join([
            f"- {pattern.get('name', '')}: {pattern.get('description', '')[:150]}..."
            for pattern in top_patterns
        ])
        
        # Create prompt for generating the summary
        prompt = f"""
        Write a concise analysis summary (250-300 words) for the query: "{query}"
        
        Top insights:
        {insight_points}
        
        Top patterns:
        {pattern_points}
        
        The summary should:
        1. Directly address the original query
        2. Synthesize the most important findings
        3. Highlight key relationships between insights
        4. Be written in a professional, analytical tone
        5. Avoid redundancy with the individual insights
        """
        
        try:
            # Generate summary using preferred AI provider (Claude via Bedrock works well for this)
            response = await self.generate_text(
                prompt=prompt,
                max_tokens=500,
                temperature=0.4,
                task_type="summary"
            )
            
            return response.get("text", "Analysis summary unavailable")
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {str(e)}")
            return "Analysis summary unavailable due to an error"
    
    async def generate_cluster_label(self, texts: List[str]) -> str:
        """
        Generate a label for a semantic cluster.
        
        Args:
            texts: Representative texts from the cluster
            
        Returns:
            Cluster label
        """
        # Combine a sample of texts
        combined_text = "\n\n".join(texts[:3]) if len(texts) > 3 else "\n\n".join(texts)
        
        # Create prompt for generating the label
        prompt = f"""
        Create a short (3-6 words), descriptive label for a group of related text passages.
        The label should capture the common theme or topic.
        
        Text passages:
        {combined_text[:1000]}
        
        Label:
        """
        
        try:
            # Generate label using preferred AI provider
            response = await self.generate_text(
                prompt=prompt,
                max_tokens=20,
                temperature=0.3,
                task_type="general"
            )
            
            # Clean and return the label
            label = response.get("text", "").strip()
            return label if label else "Unlabeled Cluster"
            
        except Exception as e:
            logger.error(f"Error generating cluster label: {str(e)}")
            return "Unlabeled Cluster"
    
    async def extract_key_concepts(self, text: str, query: str) -> List[Dict[str, Any]]:
        """
        Extract key concepts from text related to the query.
        
        Args:
            text: Text to analyze
            query: Original query for context
            
        Returns:
            List of key concepts
        """
        # Create prompt for extracting key concepts
        prompt = f"""
        Identify and explain 5-7 key concepts in the following text related to the query: "{query}"
        
        For each concept:
        1. Provide the concept name
        2. Give a short explanation (1-2 sentences)
        3. Rate the relevance to the query (0.0-1.0)
        
        Text:
        {text[:3500]}
        """
        
        try:
            # Generate key concepts using preferred AI provider
            response = await self.generate_text(
                prompt=prompt,
                max_tokens=800,
                temperature=0.3,
                task_type="general"
            )
            
            # Parse the response
            raw_concepts = response.get("text", "")
            concepts = self._parse_concepts_from_text(raw_concepts)
            
            # Ensure proper structure for each concept
            formatted_concepts = []
            for i, concept in enumerate(concepts):
                formatted_concept = {
                    "id": f"concept_{i+1}",
                    "name": concept.get("name", f"Concept {i+1}"),
                    "explanation": concept.get("explanation", ""),
                    "relevance": concept.get("relevance", 0.7)
                }
                formatted_concepts.append(formatted_concept)
            
            return formatted_concepts
            
        except Exception as e:
            logger.error(f"Error extracting key concepts: {str(e)}")
            return []
    
    def _split_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Split text into chunks of maximum size.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum chunk size
            
        Returns:
            List of text chunks
        """
        # Check if text is shorter than max size
        if len(text) <= max_chunk_size:
            return [text]
        
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph is too long, split by sentences
            if len(paragraph) > max_chunk_size:
                sentences = paragraph.replace(". ", ".\n").split("\n")
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
            else:
                # Add paragraph if it fits
                if len(current_chunk) + len(paragraph) <= max_chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _parse_patterns_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse patterns from AI-generated text.
        
        Args:
            text: Text containing pattern descriptions
            
        Returns:
            List of structured pattern dictionaries
        """
        patterns = []
        current_pattern = {}
        
        # Pattern for confidence extraction
        import re
        confidence_pattern = r"confidence(?:\s+score)?(?:\s*:\s*|\s+is\s+|\s+of\s+)(0\.\d+|[01])"
        
        # Split by pattern markers
        markers = ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4", "Pattern 5"]
        
        # Initial split
        sections = []
        for marker in markers:
            # Find each pattern section
            if marker in text:
                pattern_start = text.find(marker)
                text_after = text[pattern_start:]
                next_marker_position = float('inf')
                
                # Find the next marker position, if any
                for next_marker in markers:
                    if next_marker != marker and next_marker in text_after:
                        pos = text_after.find(next_marker)
                        if pos > 0 and pos < next_marker_position:
                            next_marker_position = pos
                
                # Extract the section
                if next_marker_position != float('inf'):
                    section = text_after[:next_marker_position].strip()
                else:
                    section = text_after.strip()
                
                sections.append(section)
        
        # Process each section
        for section in sections:
            lines = section.split("\n")
            current_pattern = {}
            current_field = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Extract pattern name
                if line.startswith("Pattern"):
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        current_pattern["name"] = parts[1].strip()
                    else:
                        # Try to find name on next line
                        continue
                
                # Extract description
                elif "description" in line.lower() or ":" in line and not current_field:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        field_name = parts[0].strip().lower()
                        if "name" in field_name:
                            current_pattern["name"] = parts[1].strip()
                        elif "desc" in field_name:
                            current_pattern["description"] = parts[1].strip()
                            current_field = "description"
                        elif "evidence" in field_name:
                            current_pattern["evidence"] = parts[1].strip()
                            current_field = "evidence"
                        elif "confidence" in field_name:
                            confidence_match = re.search(confidence_pattern, line.lower())
                            if confidence_match:
                                try:
                                    current_pattern["confidence"] = float(confidence_match.group(1))
                                except ValueError:
                                    current_pattern["confidence"] = 0.7
                
                # Continue field from previous line
                elif current_field:
                    if ":" in line and not line.startswith("  "):
                        # New field
                        parts = line.split(":", 1)
                        field_name = parts[0].strip().lower()
                        if "desc" in field_name:
                            current_pattern["description"] = parts[1].strip()
                            current_field = "description"
                        elif "evidence" in field_name:
                            current_pattern["evidence"] = parts[1].strip()
                            current_field = "evidence"
                        elif "confidence" in field_name:
                            confidence_match = re.search(confidence_pattern, line.lower())
                            if confidence_match:
                                try:
                                    current_pattern["confidence"] = float(confidence_match.group(1))
                                except ValueError:
                                    current_pattern["confidence"] = 0.7
                            current_field = None
                        else:
                            # Continue with current field
                            current_pattern[current_field] += " " + line
                    else:
                        # Continue with current field
                        current_pattern[current_field] += " " + line
                
                # Look for confidence score anywhere in the text
                if "confidence" in line.lower() and "confidence" not in current_pattern:
                    confidence_match = re.search(confidence_pattern, line.lower())
                    if confidence_match:
                        try:
                            current_pattern["confidence"] = float(confidence_match.group(1))
                        except ValueError:
                            current_pattern["confidence"] = 0.7
            
            # Ensure required fields
            if "name" in current_pattern:
                if "description" not in current_pattern:
                    current_pattern["description"] = ""
                if "evidence" not in current_pattern:
                    current_pattern["evidence"] = ""
                if "confidence" not in current_pattern:
                    current_pattern["confidence"] = 0.7
                
                patterns.append(current_pattern)
        
        # If no patterns were parsed, try a simpler approach
        if not patterns:
            current_pattern = {}
            pattern_number = 0
            
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                # Check for pattern identifier
                if line.startswith("Pattern"):
                    if current_pattern and "name" in current_pattern:
                        patterns.append(current_pattern)
                    current_pattern = {"name": "", "description": "", "evidence": "", "confidence": 0.7}
                    pattern_number += 1
                    
                    # Extract name if on same line
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        current_pattern["name"] = parts[1].strip()
                
                # Look for confidence
                elif "confidence" in line.lower():
                    confidence_match = re.search(confidence_pattern, line.lower())
                    if confidence_match:
                        try:
                            current_pattern["confidence"] = float(confidence_match.group(1))
                        except ValueError:
                            pass
                
                # Add content to description or evidence
                elif current_pattern:
                    if "evidence" in line.lower() and ":" in line:
                        current_pattern["evidence"] = line.split(":", 1)[1].strip()
                    elif "desc" in line.lower() and ":" in line:
                        current_pattern["description"] = line.split(":", 1)[1].strip()
                    elif "description" not in current_pattern or not current_pattern["description"]:
                        current_pattern["description"] = line
                    elif "evidence" not in current_pattern or not current_pattern["evidence"]:
                        current_pattern["evidence"] = line
            
            # Add the last pattern
            if current_pattern and "name" in current_pattern:
                patterns.append(current_pattern)
        
        # If still no patterns, create some basic ones from the text
        if not patterns:
            lines = text.split("\n")
            pattern_count = 0
            
            for i, line in enumerate(lines):
                if line.strip() and len(line.strip()) > 10:
                    pattern_count += 1
                    patterns.append({
                        "name": f"Pattern {pattern_count}",
                        "description": line.strip(),
                        "evidence": "",
                        "confidence": 0.7
                    })
                    
                    # Limit to 5 patterns
                    if pattern_count >= 5:
                        break
        
        return patterns
    
    def _parse_insights_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse insights from AI-generated text.
        
        Args:
            text: Text containing insight descriptions
            
        Returns:
            List of structured insight dictionaries
        """
        insights = []
        current_insight = {}
        
        # Pattern for confidence extraction
        import re
        confidence_pattern = r"confidence(?:\s+score)?(?:\s*:\s*|\s+is\s+|\s+of\s+)(0\.\d+|[01])"
        
        # Split by insight markers or numbered items
        sections = []
        
        # Try structured markers first (Insight 1, etc.)
        markers = ["Insight 1", "Insight 2", "Insight 3", "Insight 4", "Insight 5", "Insight 6", "Insight 7"]
        found_structured = False
        
        for marker in markers:
            if marker in text:
                found_structured = True
                break
        
        if found_structured:
            # Process structured format
            for marker in markers:
                # Find each insight section
                if marker in text:
                    insight_start = text.find(marker)
                    text_after = text[insight_start:]
                    next_marker_position = float('inf')
                    
                    # Find the next marker position, if any
                    for next_marker in markers:
                        if next_marker != marker and next_marker in text_after:
                            pos = text_after.find(next_marker)
                            if pos > 0 and pos < next_marker_position:
                                next_marker_position = pos
                    
                    # Extract the section
                    if next_marker_position != float('inf'):
                        section = text_after[:next_marker_position].strip()
                    else:
                        section = text_after.strip()
                    
                    sections.append(section)
        else:
            # Try numeric markers (1., 2., etc.)
            numeric_pattern = r'^\d+\.\s'
            lines = text.split('\n')
            section_start_indices = []
            
            for i, line in enumerate(lines):
                if re.match(numeric_pattern, line.strip()):
                    section_start_indices.append(i)
            
            if section_start_indices:
                for i in range(len(section_start_indices)):
                    start_idx = section_start_indices[i]
                    end_idx = section_start_indices[i+1] if i < len(section_start_indices) - 1 else len(lines)
                    section = '\n'.join(lines[start_idx:end_idx])
                    sections.append(section)
            else:
                # Fall back to paragraphs
                sections = text.split('\n\n')
        
        # Process each section
        for section in sections:
            lines = section.split('\n')
            current_insight = {}
            current_field = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Extract insight title
                if line.startswith("Insight") or re.match(r'^\d+\.', line):
                    title_parts = line.split(":", 1)
                    if len(title_parts) > 1:
                        current_insight["title"] = title_parts[1].strip()
                    else:
                        # Look for title on next line or use this as the title
                        if len(line.split()) > 1:  # More than just "Insight X"
                            marker_end = line.find(".") + 1 if "." in line else len(line.split()[0]) + 1
                            current_insight["title"] = line[marker_end:].strip()
                
                # Extract fields with explicit labels
                elif ":" in line:
                    field_parts = line.split(":", 1)
                    field_name = field_parts[0].strip().lower()
                    field_value = field_parts[1].strip()
                    
                    if "title" in field_name or "insight" in field_name:
                        current_insight["title"] = field_value
                    elif "explanation" in field_name or "detail" in field_name:
                        current_insight["explanation"] = field_value
                        current_field = "explanation"
                    elif "evidence" in field_name or "support" in field_name:
                        current_insight["evidence"] = field_value
                        current_field = "evidence"
                    elif "confidence" in field_name:
                        confidence_match = re.search(confidence_pattern, line.lower())
                        if confidence_match:
                            try:
                                current_insight["confidence"] = float(confidence_match.group(1))
                            except ValueError:
                                current_insight["confidence"] = 0.7
                        current_field = None
                
                # Continue field from previous line
                elif current_field:
                    if "confidence" in line.lower():
                        confidence_match = re.search(confidence_pattern, line.lower())
                        if confidence_match:
                            try:
                                current_insight["confidence"] = float(confidence_match.group(1))
                            except ValueError:
                                current_insight["confidence"] = 0.7
                            current_field = None
                    elif ":" in line and not line.startswith("  "):
                        # New field
                        field_parts = line.split(":", 1)
                        field_name = field_parts[0].strip().lower()
                        field_value = field_parts[1].strip()
                        
                        if "explanation" in field_name or "detail" in field_name:
                            current_insight["explanation"] = field_value
                            current_field = "explanation"
                        elif "evidence" in field_name or "support" in field_name:
                            current_insight["evidence"] = field_value
                            current_field = "evidence"
                        else:
                            # Continue with current field
                            current_insight[current_field] += " " + line
                    else:
                        # Continue with current field
                        current_insight[current_field] += " " + line
                
                # If no field is set but we have a title, assume explanation
                elif "title" in current_insight and not current_field:
                    if "explanation" not in current_insight:
                        current_insight["explanation"] = line
                        current_field = "explanation"
                    else:
                        current_insight["explanation"] += " " + line
            
            # Extract confidence if not already found
            if "confidence" not in current_insight:
                for line in lines:
                    if "confidence" in line.lower():
                        confidence_match = re.search(confidence_pattern, line.lower())
                        if confidence_match:
                            try:
                                current_insight["confidence"] = float(confidence_match.group(1))
                            except ValueError:
                                current_insight["confidence"] = 0.7
                            break
            
            # Ensure required fields
            if "title" in current_insight:
                if "explanation" not in current_insight:
                    current_insight["explanation"] = ""
                if "evidence" not in current_insight:
                    current_insight["evidence"] = ""
                if "confidence" not in current_insight:
                    current_insight["confidence"] = 0.7
                
                insights.append(current_insight)
        
        # If no insights were parsed, try to create some from the text
        if not insights:
            # Split into paragraphs and create insights
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                if i < 7:  # Limit to 7 insights
                    lines = paragraph.split('\n')
                    title = lines[0] if lines else f"Insight {i+1}"
                    explanation = ' '.join(lines[1:]) if len(lines) > 1 else paragraph
                    
                    insights.append({
                        "title": title,
                        "explanation": explanation,
                        "evidence": "",
                        "confidence": 0.7
                    })
        
        return insights
    
    def _parse_concepts_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse concepts from AI-generated text.
        
        Args:
            text: Text containing concept descriptions
            
        Returns:
            List of structured concept dictionaries
        """
        concepts = []
        
        # Pattern for relevance extraction
        import re
        relevance_pattern = r"relevance(?:\s+score)?(?:\s*:\s*|\s+is\s+|\s+of\s+)(0\.\d+|[01])"
        
        # Try both numbered list format and "Concept X" format
        section_patterns = [
            r'^\d+\.\s',  # Numbered list: "1. "
            r'^Concept \d+:',  # Concept format: "Concept 1:"
        ]
        
        lines = text.split('\n')
        sections = []
        current_section = []
        in_section = False
        
        # Split into sections
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            is_section_start = False
            for pattern in section_patterns:
                if re.match(pattern, line):
                    is_section_start = True
                    break
                    
            if is_section_start:
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
                in_section = True
            elif in_section:
                current_section.append(line)
                
        # Add the last section
        if current_section:
            sections.append('\n'.join(current_section))
            
        # Process each section
        for section in sections:
            lines = section.split('\n')
            
            # Initialize concept
            concept = {
                "name": "",
                "explanation": "",
                "relevance": 0.7
            }
            
            # First line should contain the concept name
            first_line = lines[0] if lines else ""
            
            # Extract name from first line
            if ":" in first_line:
                name_parts = first_line.split(":", 1)
                # Remove number/prefix from name
                prefix_end = re.search(r'\d+\.|\d+\)', name_parts[0])
                if prefix_end:
                    prefix_len = prefix_end.end()
                    if len(name_parts) > 1:  # "Concept 1: Name"
                        concept["name"] = name_parts[1].strip()
                    else:  # "1. Name:"
                        concept["name"] = name_parts[0][prefix_len:].strip()
                else:
                    concept["name"] = name_parts[1].strip() if len(name_parts) > 1 else name_parts[0].strip()
            else:
                # Try to extract from the first line without colon
                parts = first_line.split(None, 1)
                if len(parts) > 1 and (parts[0].isdigit() or parts[0].endswith('.')):
                    concept["name"] = parts[1]
                else:
                    concept["name"] = first_line
            
            # Extract explanation and relevance
            in_explanation = True
            for i in range(1, len(lines)):
                line = lines[i]
                
                # Check for relevance
                if "relevance" in line.lower():
                    relevance_match = re.search(relevance_pattern, line.lower())
                    if relevance_match:
                        try:
                            concept["relevance"] = float(relevance_match.group(1))
                        except ValueError:
                            pass
                    in_explanation = False
                elif "explanation" in line.lower() and ":" in line:
                    concept["explanation"] = line.split(":", 1)[1].strip()
                    in_explanation = True
                elif in_explanation:
                    if concept["explanation"]:
                        concept["explanation"] += " " + line
                    else:
                        concept["explanation"] = line
            
            # Add concept if it has a name
            if concept["name"]:
                concepts.append(concept)
        
        # If no concepts were found, try a simpler parsing approach
        if not concepts:
            # Look for lines that might contain concept names
            potential_concepts = []
            current_text = ""
            
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Check if line looks like a concept name
                if len(line) < 50 and line.endswith(':'):
                    if current_text:
                        potential_concepts.append((potential_concepts[-1][0] if potential_concepts else "Concept", current_text))
                    potential_concepts.append((line[:-1], ""))
                    current_text = ""
                else:
                    current_text += line + " "
            
            # Add the last concept
            if current_text and potential_concepts:
                potential_concepts.append((potential_concepts[-1][0], current_text))
            
            # Convert to the required format
            for i, (name, explanation) in enumerate(potential_concepts):
                if name and explanation:
                    concepts.append({
                        "name": name,
                        "explanation": explanation,
                        "relevance": 0.7
                    })
                elif name:
                    concepts.append({
                        "name": name,
                        "explanation": "",
                        "relevance": 0.7
                    })
        
        # Ensure we have at least some concepts
        if not concepts:
            # Create basic concepts from paragraphs
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            for i, paragraph in enumerate(paragraphs[:7]):  # Limit to 7 concepts
                first_sentence = paragraph.split('.')[0] if '.' in paragraph else paragraph
                remaining = paragraph[len(first_sentence)+1:] if '.' in paragraph else ""
                
                concepts.append({
                    "name": first_sentence[:50],
                    "explanation": remaining or first_sentence,
                    "relevance": 0.7
                })
        
        return concepts