"""
Content normalizer for standardizing text.
Provides functions for cleaning and standardizing content from various sources.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union
import html
import unicodedata
import json

logger = logging.getLogger(__name__)

class ContentNormalizer:
    """
    Tool for normalizing and standardizing content.
    Provides methods for cleaning text from different sources.
    """
    
    def __init__(self):
        """Initialize the content normalizer."""
        # Common web entities to clean
        self.html_entities = {
            "&nbsp;": " ",
            "&lt;": "<",
            "&gt;": ">",
            "&amp;": "&",
            "&quot;": "\"",
            "&apos;": "'",
            "&cent;": "¢",
            "&pound;": "£",
            "&yen;": "¥",
            "&euro;": "€",
            "&copy;": "©",
            "&reg;": "®"
        }
        
        # Common patterns to clean
        self.patterns = {
            "multiple_spaces": re.compile(r"\s+"),
            "html_tags": re.compile(r"<[^>]+>"),
            "urls": re.compile(r"https?://\S+"),
            "emails": re.compile(r"\S+@\S+\.\S+"),
            "line_breaks": re.compile(r"[\r\n]+"),
            "special_chars": re.compile(r"[^\w\s.,;:!?'\"\-–—()[\]{}]")
        }
    
    def normalize_text(
        self,
        text: str,
        remove_html: bool = True,
        remove_urls: bool = False,
        remove_emails: bool = False,
        normalize_whitespace: bool = True,
        fix_unicode: bool = True,
        max_length: Optional[int] = None
    ) -> str:
        """
        Normalize text content.
        
        Args:
            text: Text to normalize
            remove_html: Whether to remove HTML tags
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            normalize_whitespace: Whether to normalize whitespace
            fix_unicode: Whether to fix Unicode issues
            max_length: Maximum length of the normalized text
            
        Returns:
            Normalized text
        """
        try:
            if not text:
                return ""
            
            # Fix Unicode if requested
            if fix_unicode:
                text = self._fix_unicode(text)
            
            # Decode HTML entities
            text = html.unescape(text)
            
            # Custom entity replacement
            for entity, replacement in self.html_entities.items():
                text = text.replace(entity, replacement)
            
            # Remove HTML tags if requested
            if remove_html:
                text = self.patterns["html_tags"].sub(" ", text)
            
            # Remove URLs if requested
            if remove_urls:
                text = self.patterns["urls"].sub(" ", text)
            
            # Remove email addresses if requested
            if remove_emails:
                text = self.patterns["emails"].sub(" ", text)
            
            # Normalize whitespace if requested
            if normalize_whitespace:
                # Convert line breaks to spaces
                text = self.patterns["line_breaks"].sub(" ", text)
                # Normalize multiple spaces
                text = self.patterns["multiple_spaces"].sub(" ", text)
                # Trim whitespace
                text = text.strip()
            
            # Truncate if max_length is specified
            if max_length and len(text) > max_length:
                text = text[:max_length].rstrip() + "..."
            
            return text
            
        except Exception as e:
            logger.error(f"Error normalizing text: {str(e)}")
            return text
    
    def _fix_unicode(self, text: str) -> str:
        """
        Fix common Unicode issues.
        
        Args:
            text: Text to fix
            
        Returns:
            Fixed text
        """
        try:
            # Normalize Unicode characters
            text = unicodedata.normalize("NFKC", text)
            
            # Replace non-breaking spaces with regular spaces
            text = text.replace("\u00A0", " ")
            
            # Replace various dash characters with standard dashes
            text = text.replace("\u2013", "-").replace("\u2014", "-")
            
            return text
            
        except Exception as e:
            logger.error(f"Error fixing Unicode: {str(e)}")
            return text
    
    def normalize_json(
        self,
        json_content: Union[str, Dict, List],
        max_length: Optional[int] = None
    ) -> Union[Dict, List]:
        """
        Normalize JSON content.
        
        Args:
            json_content: JSON content as string or parsed object
            max_length: Maximum length for string values
            
        Returns:
            Normalized JSON object
        """
        try:
            # Parse JSON if it's a string
            if isinstance(json_content, str):
                json_obj = json.loads(json_content)
            else:
                json_obj = json_content
            
            # Process JSON object
            if isinstance(json_obj, dict):
                return self._normalize_json_dict(json_obj, max_length)
            elif isinstance(json_obj, list):
                return self._normalize_json_list(json_obj, max_length)
            else:
                return json_obj
            
        except Exception as e:
            logger.error(f"Error normalizing JSON: {str(e)}")
            if isinstance(json_content, (dict, list)):
                return json_content
            return {}
    
    def _normalize_json_dict(
        self,
        json_dict: Dict,
        max_length: Optional[int] = None
    ) -> Dict:
        """
        Normalize a JSON dictionary.
        
        Args:
            json_dict: Dictionary to normalize
            max_length: Maximum length for string values
            
        Returns:
            Normalized dictionary
        """
        result = {}
        for key, value in json_dict.items():
            if isinstance(value, str):
                result[key] = self.normalize_text(value, max_length=max_length)
            elif isinstance(value, dict):
                result[key] = self._normalize_json_dict(value, max_length)
            elif isinstance(value, list):
                result[key] = self._normalize_json_list(value, max_length)
            else:
                result[key] = value
        return result
    
    def _normalize_json_list(
        self,
        json_list: List,
        max_length: Optional[int] = None
    ) -> List:
        """
        Normalize a JSON list.
        
        Args:
            json_list: List to normalize
            max_length: Maximum length for string values
            
        Returns:
            Normalized list
        """
        result = []
        for item in json_list:
            if isinstance(item, str):
                result.append(self.normalize_text(item, max_length=max_length))
            elif isinstance(item, dict):
                result.append(self._normalize_json_dict(item, max_length))
            elif isinstance(item, list):
                result.append(self._normalize_json_list(item, max_length))
            else:
                result.append(item)
        return result
    
    def normalize_html(
        self,
        html_content: str,
        extract_text: bool = True,
        preserve_structure: bool = False
    ) -> str:
        """
        Normalize HTML content.
        
        Args:
            html_content: HTML content
            extract_text: Whether to extract text only
            preserve_structure: Whether to preserve paragraph structure
            
        Returns:
            Normalized text
        """
        try:
            from bs4 import BeautifulSoup
            
            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()
            
            if extract_text:
                if preserve_structure:
                    # Extract text while preserving paragraph structure
                    paragraphs = []
                    for p in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6"]):
                        text = p.get_text().strip()
                        if text:
                            paragraphs.append(text)
                    
                    text = "\n\n".join(paragraphs)
                else:
                    # Extract all text
                    text = soup.get_text()
                
                # Normalize the extracted text
                return self.normalize_text(text)
            else:
                # Return the prettified HTML
                return soup.prettify()
            
        except ImportError:
            logger.warning("BeautifulSoup not available for HTML normalization")
            # Fall back to basic HTML tag removal
            return self.normalize_text(html_content, remove_html=True)
            
        except Exception as e:
            logger.error(f"Error normalizing HTML: {str(e)}")
            return self.normalize_text(html_content, remove_html=True)
    
    def extract_main_content(self, html_content: str) -> str:
        """
        Extract main content from HTML using readability.
        
        Args:
            html_content: HTML content
            
        Returns:
            Extracted main content
        """
        try:
            from readability import Document
            
            # Parse with readability
            doc = Document(html_content)
            title = doc.title()
            content = doc.summary()
            
            # Extract text from the summary HTML
            text = self.normalize_html(
                content,
                extract_text=True,
                preserve_structure=True
            )
            
            # Combine title and content
            if title:
                return f"{title}\n\n{text}"
            return text
            
        except ImportError:
            logger.warning("Readability not available for content extraction")
            return self.normalize_html(html_content)
            
        except Exception as e:
            logger.error(f"Error extracting main content: {str(e)}")
            return self.normalize_html(html_content)
    
    def standardize_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize a document to a common format.
        
        Args:
            document: Document to standardize
            
        Returns:
            Standardized document
        """
        # Standard document format
        standard_doc = {
            "content": "",
            "metadata": {}
        }
        
        try:
            # Extract content field
            content = ""
            if "content" in document:
                content = document["content"]
            elif "text" in document:
                content = document["text"]
            elif "body" in document:
                content = document["body"]
            
            # Normalize content
            standard_doc["content"] = self.normalize_text(content)
            
            # Extract metadata
            metadata = {}
            meta_fields = ["title", "url", "source", "author", "date", "category"]
            
            for field in meta_fields:
                if field in document:
                    metadata[field] = document[field]
            
            # Handle nested metadata
            if "metadata" in document and isinstance(document["metadata"], dict):
                metadata.update(document["metadata"])
            
            # Add basic metadata if empty
            if not metadata and "id" in document:
                metadata["id"] = document["id"]
            
            standard_doc["metadata"] = metadata
            
            return standard_doc
            
        except Exception as e:
            logger.error(f"Error standardizing document: {str(e)}")
            
            # Return original document with minimal standardization
            if "content" not in document and "text" in document:
                document["content"] = document["text"]
            
            return document

# Singleton instance for easy import
normalizer = ContentNormalizer()