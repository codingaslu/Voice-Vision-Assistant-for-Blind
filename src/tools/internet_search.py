"""
Internet search module using DuckDuckGo
"""

import logging
from typing import List, Dict, Any, Optional

# Simple logger without custom handler, will use root logger's config
logger = logging.getLogger("internet-search")

class InternetSearch:
    """
    A class that handles internet searches using DuckDuckGo.
    This allows the agent to search for information on the web.
    """
    
    def __init__(self, max_results: int = 5):
        """Initialize the internet search tool."""
        self.max_results = max_results
        self._search = None
        self._search_detailed = None
        self._news_search = None
        self._last_query = ""
        self._last_results = ""
        
        # Initialize search tools
        try:
            self._initialize_search_tools()
            logger.info("Initialized internet search tools")
        except Exception as e:
            logger.error(f"Failed to initialize internet search tools: {e}")
    
    def _initialize_search_tools(self):
        """Initialize the search tools."""
        from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        
        # Basic search that returns a text summary
        self._search = DuckDuckGoSearchRun()
        
        # Detailed search that returns structured results with links
        self._search_detailed = DuckDuckGoSearchResults(output_format="list")
        
        # News search
        news_wrapper = DuckDuckGoSearchAPIWrapper(time="m", max_results=self.max_results)
        self._news_search = DuckDuckGoSearchResults(api_wrapper=news_wrapper, backend="news")
    
    async def search(self, query: str, include_news: bool = True) -> Dict[str, Any]:
        """
        Comprehensive search combining basic information, detailed results, and news.
        
        Args:
            query: The search query
            include_news: Whether to include news results
            
        Returns:
            Dictionary containing general results, detailed results with links, and news
        """
        self._last_query = query
        
        if self._search is None:
            self._initialize_search_tools()
        
        results = {
            "general_info": "",
            "detailed_results": [],
            "news_articles": []
        }
        
        # Perform general search
        try:
            general_results = self._search.invoke(query)
            results["general_info"] = general_results
        except Exception as e:
            logger.error(f"Error in general search: {e}")
            results["general_info"] = f"Error searching for general information: {str(e)}"
        
        # Perform detailed search
        try:
            detailed_results = self._search_detailed.invoke(query)
            results["detailed_results"] = detailed_results[:5]  # Limit to top 5 results
        except Exception as e:
            logger.error(f"Error in detailed search: {e}")
            results["detailed_results"] = [{"title": "Error", "snippet": f"Error performing detailed search: {str(e)}", "link": ""}]
        
        # Perform news search if requested
        if include_news:
            try:
                news_results = self._news_search.invoke(query)
                parsed_news = self._parse_news_results(news_results)
                results["news_articles"] = parsed_news[:3]  # Limit to top 3 news articles
            except Exception as e:
                logger.error(f"Error in news search: {e}")
                results["news_articles"] = [{"title": "Error", "snippet": f"Error searching for news: {str(e)}", "link": ""}]
        
        # Store results for future reference
        self._last_results = str(results)
        return results
    
    def _parse_news_results(self, results: str) -> List[Dict[str, Any]]:
        """Parse news results from string format to structured data."""
        parsed_results = []
        if isinstance(results, str):
            # Split by commas that separate entries
            items = results.split('snippet:')
            for item in items[1:]:  # Skip the first empty item
                try:
                    parts = item.split('link:')
                    snippet = parts[0].strip().rstrip(',')
                    
                    # Extract title
                    title_parts = snippet.split('title:')
                    if len(title_parts) > 1:
                        snippet = title_parts[0].strip().rstrip(',')
                        title = title_parts[1].split(',')[0].strip()
                    else:
                        title = "No title"
                    
                    # Extract link
                    link = parts[1].split('date:')[0].strip().rstrip(',')
                    
                    # Extract date if available
                    if 'date:' in parts[1]:
                        date = parts[1].split('date:')[1].split('source:')[0].strip().rstrip(',')
                        source = parts[1].split('source:')[1].strip().split(',')[0]
                    else:
                        date = ""
                        source = ""
                    
                    parsed_results.append({
                        "title": title,
                        "snippet": snippet,
                        "link": link,
                        "date": date,
                        "source": source
                    })
                except Exception as parse_error:
                    logger.error(f"Error parsing news result: {parse_error}")
        
        return parsed_results
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """
        Format comprehensive search results into a human-readable string.
        
        Args:
            results: Dictionary containing search results
            
        Returns:
            Formatted search results as a string
        """
        output = []
        
        # Add general information
        if results.get("general_info"):
            output.append("## General Information\n")
            output.append(results["general_info"])
            output.append("\n")
        
        # Add detailed results
        if results.get("detailed_results"):
            output.append("## Detailed Results\n")
            for i, result in enumerate(results["detailed_results"][:3], 1):
                output.append(f"{i}. **{result.get('title', 'No title')}**")
                output.append(f"   {result.get('snippet', 'No description')}")
                if result.get('link'):
                    output.append(f"   Link: {result['link']}")
                output.append("")
            output.append("\n")
        
        # Add news articles
        if results.get("news_articles"):
            output.append("## Recent News\n")
            for i, article in enumerate(results["news_articles"][:3], 1):
                output.append(f"{i}. **{article.get('title', 'No title')}**")
                output.append(f"   {article.get('snippet', 'No description')}")
                if article.get('date'):
                    output.append(f"   Published: {article['date']}")
                if article.get('source'):
                    output.append(f"   Source: {article['source']}")
                output.append("")
        
        return "\n".join(output)
    
    def get_last_results(self) -> str:
        """Get the results of the last search."""
        return self._last_results 