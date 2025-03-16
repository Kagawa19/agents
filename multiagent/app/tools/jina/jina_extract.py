"""
Feature extraction for Jina.
Provides methods for creating embeddings and extracting features.
"""

import logging
import aiohttp
import json
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

class JinaExtract:
    """
    Feature extraction utilities for Jina.
    Provides methods for creating embeddings and extracting features.
    """
    
    def __init__(
        self,
        api_key: str,
        tracer: Optional[LangfuseTracer] = None
    ):
        """
        Initialize the Jina Extract utilities.
        
        Args:
            api_key: Jina AI API key
            tracer: Optional LangfuseTracer instance for monitoring
        """
        self.api_key = api_key
        self.tracer = tracer
        self.base_url = "https://api.jina.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def get_embeddings(
        self,
        texts: List[str],
        model: str = "jina-embeddings-v2-base-en"
    ) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to get embeddings for
            model: Embedding model to use
            
        Returns:
            List of embeddings
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_get_embeddings",
                input={"text_count": len(texts), "model": model}
            )
        
        try:
            # Check for empty inputs
            if not texts:
                return []
            
            # Remove None or empty values
            filtered_texts = [text for text in texts if text]
            if not filtered_texts:
                return [[0.0]] * len(texts)  # Return zero vectors for all empty inputs
            
            # Prepare request
            url = f"{self.base_url}/embeddings"
            payload = {
                "model": model,
                "input": filtered_texts
            }
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Jina API error ({response.status}): {error_text}")
                        if span:
                            span.update(output={"error": error_text})
                        return []
                    
                    # Parse response
                    result = await response.json()
            
            # Extract embeddings and reconstruct original order
            embeddings = []
            extracted = [item["embedding"] for item in result.get("data", [])]
            
            # Ensure original order and handle empty inputs
            i = 0
            for text in texts:
                if text:
                    if i < len(extracted):
                        embeddings.append(extracted[i])
                        i += 1
                    else:
                        # Fallback: empty vector if we've run out of results
                        dim = len(extracted[0]) if extracted else 768
                        embeddings.append([0.0] * dim)
                else:
                    # Empty vector for empty input
                    dim = len(extracted[0]) if extracted else 768
                    embeddings.append([0.0] * dim)
            
            if span:
                span.update(output={"embedding_count": len(embeddings)})
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []
    
    async def get_batch_embeddings(
        self,
        texts: List[str],
        model: str = "jina-embeddings-v2-base-en",
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Get embeddings for a large batch of texts by processing in smaller batches.
        
        Args:
            texts: List of texts to get embeddings for
            model: Embedding model to use
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embeddings
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_batch_embeddings",
                input={"text_count": len(texts), "batch_size": batch_size}
            )
        
        try:
            # Check for empty inputs
            if not texts:
                return []
            
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = await self.get_embeddings(batch, model)
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Processed embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            if span:
                span.update(output={"embedding_count": len(all_embeddings)})
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embeddings: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []
    
    async def extract_keywords(
        self,
        text: str,
        max_keywords: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords with scores
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_extract_keywords",
                input={"text_length": len(text)}
            )
        
        try:
            # Try extracting keywords using embeddings and clustering
            from multiagent.app.tools.jina.jina_search import JinaSearch
            
            # Extract sentences
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) < 3:
                # Split into phrases if not enough sentences
                sentences = re.split(r'[,;:]', text)
            
            # Get embeddings for sentences
            embeddings = await self.get_embeddings(sentences)
            
            # Create JinaSearch instance
            search_util = JinaSearch(api_key=self.api_key, tracer=self.tracer)
            
            # Cluster sentences
            clusters = await search_util.cluster_embeddings(
                embeddings=embeddings,
                n_clusters=min(5, len(sentences) // 2) if len(sentences) > 4 else 2
            )
            
            # Extract representative sentences from each cluster
            keywords = []
            for cluster_id, indices in clusters.items():
                if indices:
                    # Find sentence closest to cluster center (most representative)
                    representative_idx = indices[0]
                    representative_text = sentences[representative_idx]
                    
                    # Extract noun phrases as keywords (simplified)
                    words = representative_text.split()
                    if len(words) > 3:
                        phrase = " ".join(words[:3])
                    else:
                        phrase = representative_text
                    
                    keywords.append({
                        "keyword": phrase.strip().rstrip(".!?,;:"),
                        "score": 1.0 - (0.1 * cluster_id),  # Simple scoring
                        "cluster": cluster_id
                    })
            
            # Limit to max_keywords
            keywords = keywords[:max_keywords]
            
            if span:
                span.update(output={"keyword_count": len(keywords)})
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            # Fallback to simple frequency-based extraction
            try:
                import re
                from collections import Counter
                
                # Tokenize text and remove stop words
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                stop_words = {"the", "and", "is", "in", "to", "a", "of", "that", "for", "it", "with", "as", "be"}
                filtered_words = [word for word in words if word not in stop_words]
                
                # Count word frequencies
                word_counts = Counter(filtered_words)
                
                # Extract top keywords
                top_keywords = word_counts.most_common(max_keywords)
                
                return [
                    {"keyword": word, "score": count / max(word_counts.values())}
                    for word, count in top_keywords
                ]
            except Exception as e2:
                logger.error(f"Error in fallback keyword extraction: {str(e2)}")
                return []
    
    async def extract_entities(
        self,
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entities with types
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_extract_entities",
                input={"text_length": len(text)}
            )
        
        try:
            # Try using spaCy for entity extraction
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
            except ImportError:
                logger.warning("spaCy not installed or model not available")
                return []
            
            # Process text with spaCy
            doc = nlp(text[:100000])  # Limit text length for performance
            
            # Extract entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            if span:
                span.update(output={"entity_count": len(entities)})
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return []
    
    async def extract_sentences(
        self,
        text: str,
        min_length: int = 10,
        max_length: int = 200
    ) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Text to extract sentences from
            min_length: Minimum sentence length
            max_length: Maximum sentence length
            
        Returns:
            List of sentences
        """
        try:
            import re
            
            # Split text into sentences
            sentence_pattern = r'(?<=[.!?])\s+'
            sentences = re.split(sentence_pattern, text)
            
            # Filter sentences by length
            filtered_sentences = [
                s.strip() for s in sentences
                if s.strip() and min_length <= len(s.strip()) <= max_length
            ]
            
            return filtered_sentences
            
        except Exception as e:
            logger.error(f"Error extracting sentences: {str(e)}")
            return []
    
    async def compare_texts(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compare two texts for semantic similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_compare_texts",
                input={"text1_length": len(text1), "text2_length": len(text2)}
            )
        
        try:
            # Get embeddings for both texts
            embeddings = await self.get_embeddings([text1, text2])
            if len(embeddings) != 2:
                return 0.0
            
            # Calculate cosine similarity
            from multiagent.app.tools.jina.jina_search import JinaSearch
            search_util = JinaSearch(api_key=self.api_key, tracer=self.tracer)
            
            similarities = await search_util.calculate_similarities(
                query_embedding=embeddings[0],
                document_embeddings=[embeddings[1]]
            )
            
            similarity = similarities[0] if similarities else 0.0
            
            if span:
                span.update(output={"similarity": similarity})
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error comparing texts: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return 0.0
    
    async def summarize_text(
        self,
        text: str,
        max_length: int = 200
    ) -> str:
        """
        Generate a summary of text using extractive summarization.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            
        Returns:
            Summarized text
        """
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="jina_summarize_text",
                input={"text_length": len(text)}
            )
        
        try:
            # Extract sentences
            sentences = await self.extract_sentences(text)
            if not sentences:
                return ""
            
            # Get embeddings for sentences
            embeddings = await self.get_embeddings(sentences)
            if not embeddings:
                return ""
            
            # Average embedding to represent entire text
            import numpy as np
            avg_embedding = np.mean(embeddings, axis=0).tolist()
            
            # Calculate similarity of each sentence to the average
            from multiagent.app.tools.jina.jina_search import JinaSearch
            search_util = JinaSearch(api_key=self.api_key, tracer=self.tracer)
            
            similarities = await search_util.calculate_similarities(
                query_embedding=avg_embedding,
                document_embeddings=embeddings
            )
            
            # Rank sentences by similarity
            ranked_sentences = sorted(
                zip(sentences, similarities, range(len(sentences))),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Take top sentences (up to word limit)
            summary_sentences = []
            word_count = 0
            
            for sentence, _, idx in ranked_sentences:
                sentence_word_count = len(sentence.split())
                if word_count + sentence_word_count <= max_length:
                    summary_sentences.append((idx, sentence))
                    word_count += sentence_word_count
                else:
                    break
            
            # Sort sentences by original order
            summary_sentences.sort()
            summary = " ".join(sentence for _, sentence in summary_sentences)
            
            if span:
                span.update(output={"summary_length": len(summary)})
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            
            if span:
                span.update(output={"error": str(e)})
            
            return ""