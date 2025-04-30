"""
Google Places Tool for Ally Vision Assistant

This tool integrates with Google Places API to find information about
locations, restaurants, businesses, and other points of interest.
"""

import os
import logging
from langchain_community.tools import GooglePlacesTool
from src.config import get_config

# Simple logger without custom handler
logger = logging.getLogger("google-places")

class PlacesSearch:
    """Handler for Google Places API integration."""
    
    def __init__(self):
        """Initialize the Google Places API handler."""
        # Get API key from config
        config = get_config()
        api_key = config["GPLACES_API_KEY"]
        
        # Set API key in environment
        if api_key:
            os.environ["GPLACES_API_KEY"] = api_key
            logger.info("Using API key from config")
        else:
            # Fallback to default key if not in config
            os.environ["GPLACES_API_KEY"] = "AIzaSyDsGkgEUESzXYUNuFZOm7_5AvLXQpCoIZw"
            logger.info("Using default API key")
            
        # Create the tool directly
        self.places_tool = GooglePlacesTool()
        logger.info("Google Places tool initialized successfully")
    
    async def search_places(self, query: str) -> str:
        """
        Search for places using Google Places API.
        
        Args:
            query: Search query for places
            
        Returns:
            Results as string
        """
        try:
            logger.info(f"Searching places: {query}")
            return self.places_tool.run(query)
        except Exception as e:
            logger.error(f"Error searching Google Places: {e}")
            return f"I encountered an error while searching for places: {str(e)}" 