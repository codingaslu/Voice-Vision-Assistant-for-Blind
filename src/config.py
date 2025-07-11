"""
This file allows you to configure which vision model provider to use.
"""

import os
import logging
from enum import Enum

# Get a logger for this module
logger = logging.getLogger("config")

# Vision provider options
class VisionProvider(Enum):
    GROQ = "groq"  # Use Groq API

# Helper function to determine vision provider from env
def _get_vision_provider():
    provider_str = os.environ.get("VISION_PROVIDER", "groq").lower()
    return VisionProvider.GROQ

# Default configuration
CONFIG = {
    # Select which vision provider to use
    "VISION_PROVIDER": _get_vision_provider(),
    
    # Groq configuration
    "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
    "GROQ_MODEL_ID": os.environ.get("GROQ_MODEL_ID", "meta-llama/llama-4-scout-17b-16e-instruct"),  # Can also use: meta-llama/llama-4-scout-17b-16e-instruct or llama-3.2-90b-vision-preview or llama-3.2-11b-vision-preview

    # Google Places API configuration
    "GPLACES_API_KEY": os.environ.get("GPLACES_API_KEY", ""),
    
    # Tavus virtual avatar configuration
    "ENABLE_AVATAR": os.environ.get("ENABLE_AVATAR", "false") == "true",
    "TAVUS_API_KEY": os.environ.get("TAVUS_API_KEY", ""),
    "TAVUS_REPLICA_ID": os.environ.get("TAVUS_REPLICA_ID", ""),
    "TAVUS_PERSONA_ID": os.environ.get("TAVUS_PERSONA_ID", ""),
    "TAVUS_AVATAR_NAME": os.environ.get("TAVUS_AVATAR_NAME", "ally-vision-avatar"),

    # Common configuration
    "MAX_TOKENS": 500,
    "TEMPERATURE": 0.7,
}

def get_config():
    """Get the current configuration."""
    return CONFIG

def use_groq():
    """Check if Groq is the current vision provider."""
    return CONFIG["VISION_PROVIDER"] == VisionProvider.GROQ 