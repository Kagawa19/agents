"""
WebScraper for content extraction.
Extracts content from websites for analysis.
"""

import asyncio
import logging
import re
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
import ssl
import certifi

from bs4 import BeautifulSoup
from readability import Document

from multiagent.app.monitoring.tracer import LangfuseTracer


logger = logging.getLogger(__name__)


class WebScraper:
    """
    Tool for scraping and extracting content from websites.
    Provides methods for fetching pages and cleaning their content.
    """
    
    def __init__(
        self,
        user_agent: Optional[str] = None,
        timeout: int = 15,
        max_content_length: int = 100000,
        tracer: Optional[LangfuseTracer] = None
    ):
        """
        Initialize the WebScraper tool.
        
        Args:
            user_agent: Custom User-Agent string for HTTP requests
            timeout: Timeout for HTTP requests in seconds
            max_content_length: Maximum content length to process (in characters)
            tracer: Optional LangfuseTracer instance for monitoring
        """
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.tracer = tracer
        
        # Domains to be treated with extra parsing rules
        self.special_domains = {
            "twitter.com": self._parse_twitter,
            "youtube.com": self._parse_youtube,
            "docs.google.com": self._parse_google_docs,
            "github.com": self._parse_github
        }
    
    async def scrape(self, url: str) -> Optional[str]:
        """
        Scrape and extract the main content from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Extracted content or None if failed
        """
        # Create span for tracing if tracer is provided
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="web_scraper",
                input={"url": url}
            )
        
        try:
            logger.info(f"Scraping URL: {url}")
            
            # Check if URL is valid
            if not self._is_valid_url(url):
                logger.warning(f"Invalid URL: {url}")
                if span:
                    span.update(output={"error": "Invalid URL"})
                return None
            
            # Fetch HTML content
            html = await self._fetch_url(url)
            if not html:
                logger.warning(f"Failed to fetch HTML from URL: {url}")
                if span:
                    span.update(output={"error": "Failed to fetch HTML"})
                return None
            
            # Extract content
            domain = self._get_domain(url)
            
            # Check if domain requires special parsing
            if domain in self.special_domains:
                content = await self.special_domains[domain](url, html)
            else:
                # Use readability to extract main content
                content = self._extract_content(html)
                
                # If readability fails, try BeautifulSoup parsing
                if not content or len(content) < 200:
                    content = self._extract_with_beautifulsoup(html)
            
            # Clean content
            if content:
                content = self._clean_content(content)
                
                # Truncate if too long
                if len(content) > self.max_content_length:
                    content = content[:self.max_content_length] + "... (content truncated)"
                
                # Update span with success output
                if span:
                    span.update(output={"content_length": len(content)})
                
                logger.info(f"Successfully scraped {len(content)} characters from {url}")
                return content
            else:
                logger.warning(f"No content extracted from URL: {url}")
                if span:
                    span.update(output={"error": "No content extracted"})
                return None
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            # Update span with error
            if span:
                span.update(output={"error": str(e)})
            return None
    
    async def _fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch HTML content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
        
        try:
            # Create SSL context with certifi certs
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    ssl=ssl_context
                ) as response:
                    if response.status != 200:
                        logger.warning(f"HTTP error {response.status} for URL: {url}")
                        return None
                    
                    # Check content type
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                        logger.warning(f"Skipping non-HTML content type: {content_type}")
                        return None
                    
                    # Get content
                    html = await response.text()
                    return html
                    
        except aiohttp.ClientError as e:
            logger.error(f"Client error fetching URL {url}: {str(e)}")
            return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching URL {url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            return None
    
    def _extract_content(self, html: str) -> str:
        """
        Extract the main content using readability.
        
        Args:
            html: HTML content
            
        Returns:
            Extracted main content
        """
        try:
            # Parse with readability
            doc = Document(html)
            title = doc.title()
            content = doc.summary()
            
            # Extract text from HTML
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(" ", strip=True)
            
            if title:
                return f"{title}\n\n{text}"
            return text
            
        except Exception as e:
            logger.error(f"Error extracting content with readability: {str(e)}")
            return ""
    
    def _extract_with_beautifulsoup(self, html: str) -> str:
        """
        Extract content using BeautifulSoup when readability fails.
        
        Args:
            html: HTML content
            
        Returns:
            Extracted content
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove unwanted elements
            for tag in soup.find_all(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
            
            # Get title
            title = ""
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
            
            # Get main content
            main_content = ""
            
            # Try to find main content containers
            main_elements = soup.find_all(["article", "main", "div", "section"], 
                class_=lambda c: c and any(x in str(c).lower() for x in ["content", "article", "main", "body"]))
            
            if main_elements:
                # Find the main element with the most text
                main_element = max(main_elements, key=lambda e: len(e.get_text()))
                main_content = main_element.get_text(" ", strip=True)
            else:
                # If no main content container found, use the body
                body = soup.find("body")
                if body:
                    main_content = body.get_text(" ", strip=True)
            
            # Combine title and content
            if title and main_content:
                return f"{title}\n\n{main_content}"
            return main_content or title
            
        except Exception as e:
            logger.error(f"Error extracting content with BeautifulSoup: {str(e)}")
            return ""
    
    def _clean_content(self, content: str) -> str:
        """
        Clean and normalize the extracted content.
        
        Args:
            content: Extracted content
            
        Returns:
            Cleaned content
        """
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove redundant newlines but preserve paragraph breaks
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Fix spacing after punctuation
        content = re.sub(r'([.!?])\s*(?=\w)', r'\1 ', content)
        
        # Remove common boilerplate phrases
        boilerplate = [
            r'accept cookies',
            r'cookie policy',
            r'privacy policy',
            r'terms of service',
            r'all rights reserved',
            r'copyright \d{4}',
            r'subscribe to our newsletter',
            r'sign up for our newsletter',
            r'share this article',
            r'comments'
        ]
        
        for phrase in boilerplate:
            content = re.sub(r'(?i)' + phrase + r'.*?[\.\n]', '', content)
        
        # Rebuild paragraphs more cleanly
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        content = '\n\n'.join(paragraphs)
        
        return content.strip()
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid.
        
        Args:
            url: URL to check
            
        Returns:
            True if valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    def _get_domain(self, url: str) -> str:
        """
        Extract the domain from a URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain
        except Exception:
            return ""
    
    async def _parse_twitter(self, url: str, html: str) -> str:
        """
        Special parser for Twitter pages.
        
        Args:
            url: Twitter URL
            html: HTML content
            
        Returns:
            Extracted content
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract tweet text
            tweet_text_elements = soup.find_all("div", attrs={"data-testid": "tweetText"})
            tweet_text = ""
            
            if tweet_text_elements:
                for element in tweet_text_elements:
                    tweet_text += element.get_text() + "\n\n"
            
            # If tweet text not found, try standard extraction
            if not tweet_text:
                return self._extract_content(html)
                
            # Get author info
            author = ""
            author_element = soup.find("a", attrs={"data-testid": "User-Name"})
            if author_element:
                author = author_element.get_text()
            
            # Combine information
            if author:
                return f"Tweet by {author}:\n\n{tweet_text}"
            return tweet_text
            
        except Exception as e:
            logger.error(f"Error parsing Twitter page: {str(e)}")
            return self._extract_content(html)
    
    async def _parse_youtube(self, url: str, html: str) -> str:
        """
        Special parser for YouTube pages.
        
        Args:
            url: YouTube URL
            html: HTML content
            
        Returns:
            Extracted content
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract video title
            title = ""
            title_element = soup.find("meta", property="og:title")
            if title_element:
                title = title_element.get("content", "")
            
            # Extract video description
            description = ""
            description_element = soup.find("meta", property="og:description")
            if description_element:
                description = description_element.get("content", "")
            
            # Extract channel name
            channel = ""
            channel_element = soup.find("link", itemprop="name")
            if channel_element:
                channel = channel_element.get("content", "")
            
            # Combine information
            video_info = f"YouTube Video: {title}\n\n"
            if channel:
                video_info += f"Channel: {channel}\n\n"
            if description:
                video_info += f"Description:\n{description}"
            
            return video_info
            
        except Exception as e:
            logger.error(f"Error parsing YouTube page: {str(e)}")
            return self._extract_content(html)
    
    async def _parse_google_docs(self, url: str, html: str) -> str:
        """
        Special parser for Google Docs pages.
        
        Args:
            url: Google Docs URL
            html: HTML content
            
        Returns:
            Extracted content
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract document title
            title = ""
            title_element = soup.find("meta", property="og:title")
            if title_element:
                title = title_element.get("content", "")
            
            # Since Google Docs content is loaded dynamically, we can't easily extract it
            # Just provide a placeholder
            return f"Google Document: {title}\n\n[This is a Google Doc. Content cannot be directly extracted due to dynamic loading.]"
            
        except Exception as e:
            logger.error(f"Error parsing Google Docs page: {str(e)}")
            return self._extract_content(html)
    
    async def _parse_github(self, url: str, html: str) -> str:
        """
        Special parser for GitHub pages.
        
        Args:
            url: GitHub URL
            html: HTML content
            
        Returns:
            Extracted content
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Check if it's a repository page
            repo_about = soup.find("div", class_="BorderGrid-cell", id="about")
            if repo_about:
                # Extract repository information
                
                # Get repo name
                repo_name = ""
                repo_name_element = soup.find("strong", itemprop="name")
                if repo_name_element:
                    repo_name = repo_name_element.get_text(strip=True)
                
                # Get description
                description = ""
                description_element = soup.find("p", class_="f4 my-3")
                if description_element:
                    description = description_element.get_text(strip=True)
                
                # Get README content
                readme_content = ""
                readme_element = soup.find("div", id="readme")
                if readme_element:
                    article = readme_element.find("article")
                    if article:
                        readme_content = article.get_text(" ", strip=True)
                
                # Combine information
                content = f"GitHub Repository: {repo_name}\n\n"
                if description:
                    content += f"Description: {description}\n\n"
                if readme_content:
                    content += f"README:\n{readme_content}"
                
                return content
            
            # Check if it's a code file
            code_element = soup.find("div", class_="Box-body p-0 blob-wrapper data type-python")
            if code_element:
                # Extract file content
                lines = code_element.find_all("tr")
                code_content = ""
                for line in lines:
                    code_text = line.find("td", class_="blob-code")
                    if code_text:
                        code_content += code_text.get_text() + "\n"
                
                # Get file path
                file_path = ""
                path_element = soup.find("strong", class_="final-path")
                if path_element:
                    file_path = path_element.get_text(strip=True)
                
                return f"GitHub Code File: {file_path}\n\n{code_content}"
            
            # Default to standard extraction
            return self._extract_content(html)
            
        except Exception as e:
            logger.error(f"Error parsing GitHub page: {str(e)}")
            return self._extract_content(html)
    
    async def scrape_multiple(self, urls: List[str]) -> Dict[str, str]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            Dictionary mapping URLs to their extracted content
        """
        results = {}
        tasks = []
        
        # Create span for tracing if tracer is provided
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="web_scraper_multiple",
                input={"urls": urls, "count": len(urls)}
            )
        
        try:
            # Create tasks for each URL
            for url in urls:
                task = self.scrape(url)
                tasks.append((url, task))
            
            # Wait for all tasks to complete
            for url, task in tasks:
                try:
                    content = await task
                    if content:
                        results[url] = content
                except Exception as e:
                    logger.error(f"Error scraping URL {url}: {str(e)}")
            
            # Update span with success output
            if span:
                span.update(output={"urls_scraped": len(results)})
            
            logger.info(f"Successfully scraped {len(results)} out of {len(urls)} URLs")
            return results
            
        except Exception as e:
            logger.error(f"Error in scrape_multiple: {str(e)}")
            # Update span with error
            if span:
                span.update(output={"error": str(e)})
            return results
    
    async def get_metadata(self, url: str) -> Dict[str, Any]:
        """
        Get metadata from a URL (title, description, etc.).
        
        Args:
            url: URL to get metadata from
            
        Returns:
            Dictionary of metadata
        """
        # Create span for tracing if tracer is provided
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="web_scraper_metadata",
                input={"url": url}
            )
        
        try:
            logger.info(f"Getting metadata from URL: {url}")
            
            # Check if URL is valid
            if not self._is_valid_url(url):
                logger.warning(f"Invalid URL: {url}")
                if span:
                    span.update(output={"error": "Invalid URL"})
                return {"url": url, "error": "Invalid URL"}
            
            # Fetch HTML content
            html = await self._fetch_url(url)
            if not html:
                logger.warning(f"Failed to fetch HTML from URL: {url}")
                if span:
                    span.update(output={"error": "Failed to fetch HTML"})
                return {"url": url, "error": "Failed to fetch HTML"}
            
            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Initialize metadata
            metadata = {
                "url": url,
                "title": "",
                "description": "",
                "image": "",
                "site_name": "",
                "author": "",
                "published_date": "",
                "domain": self._get_domain(url)
            }
            
            # Get title
            title_tag = soup.find("title")
            if title_tag:
                metadata["title"] = title_tag.get_text(strip=True)
            
            # Get Open Graph metadata
            og_tags = {
                "og:title": "title",
                "og:description": "description",
                "og:image": "image",
                "og:site_name": "site_name"
            }
            
            for og_tag, meta_key in og_tags.items():
                meta_element = soup.find("meta", property=og_tag)
                if meta_element and meta_element.get("content"):
                    metadata[meta_key] = meta_element.get("content")
            
            # Get Twitter Card metadata
            twitter_tags = {
                "twitter:title": "title",
                "twitter:description": "description",
                "twitter:image": "image"
            }
            
            for twitter_tag, meta_key in twitter_tags.items():
                meta_element = soup.find("meta", {"name": twitter_tag})
                if meta_element and meta_element.get("content") and not metadata[meta_key]:
                    metadata[meta_key] = meta_element.get("content")
            
            # Get description from meta description if not found via OG/Twitter
            if not metadata["description"]:
                meta_desc = soup.find("meta", {"name": "description"})
                if meta_desc and meta_desc.get("content"):
                    metadata["description"] = meta_desc.get("content")
            
            # Get author
            author_element = soup.find("meta", {"name": "author"})
            if author_element and author_element.get("content"):
                metadata["author"] = author_element.get("content")
            
            # Get publication date
            date_elements = [
                soup.find("meta", {"name": "article:published_time"}),
                soup.find("meta", {"property": "article:published_time"}),
                soup.find("time"),
                soup.find("meta", {"name": "date"})
            ]
            
            for element in date_elements:
                if element:
                    if element.get("content"):
                        metadata["published_date"] = element.get("content")
                        break
                    elif element.get("datetime"):
                        metadata["published_date"] = element.get("datetime")
                        break
            
            # Update span with success output
            if span:
                span.update(output={"metadata": metadata})
            
            logger.info(f"Successfully extracted metadata from {url}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata from URL {url}: {str(e)}")
            # Update span with error
            if span:
                span.update(output={"error": str(e)})
            return {"url": url, "error": str(e)}

    async def extract_links(self, url: str) -> List[Dict[str, str]]:
        """
        Extract all links from a webpage.
        
        Args:
            url: URL to extract links from
            
        Returns:
            List of dictionaries with link information
        """
        # Create span for tracing if tracer is provided
        span = None
        if self.tracer:
            span = self.tracer.span(
                name="web_scraper_extract_links",
                input={"url": url}
            )
        
        try:
            logger.info(f"Extracting links from URL: {url}")
            
            # Fetch HTML content
            html = await self._fetch_url(url)
            if not html:
                logger.warning(f"Failed to fetch HTML from URL: {url}")
                if span:
                    span.update(output={"error": "Failed to fetch HTML"})
                return []
            
            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract all links
            links = []
            base_url = urlparse(url)
            base_domain = base_url.netloc
            
            for a_tag in soup.find_all("a", href=True):
                href = a_tag.get("href", "").strip()
                
                # Skip empty links and javascript links
                if not href or href.startswith("javascript:") or href == "#":
                    continue
                
                # Build absolute URL
                if href.startswith("/"):
                    # Relative URL
                    href = f"{base_url.scheme}://{base_domain}{href}"
                elif not href.startswith(("http://", "https://")):
                    # Partial URL
                    if href.startswith("./"):
                        href = href[2:]
                    href = f"{base_url.scheme}://{base_domain}/{href}"
                
                # Get link text
                link_text = a_tag.get_text(strip=True)
                
                # Create link info
                link_info = {
                    "url": href,
                    "text": link_text,
                    "is_internal": base_domain in href
                }
                
                links.append(link_info)
            
            # Remove duplicates based on URL
            unique_links = []
            seen_urls = set()
            
            for link in links:
                if link["url"] not in seen_urls:
                    seen_urls.add(link["url"])
                    unique_links.append(link)
            
            # Update span with success output
            if span:
                span.update(output={"link_count": len(unique_links)})
            
            logger.info(f"Extracted {len(unique_links)} unique links from {url}")
            return unique_links
            
        except Exception as e:
            logger.error(f"Error extracting links from URL {url}: {str(e)}")
            # Update span with error
            if span:
                span.update(output={"error": str(e)})
            return []