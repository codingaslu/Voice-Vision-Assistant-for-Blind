"""
Ally Vision Assistant package.

Initializes environment variables from .env file at import time.
"""

from dotenv import load_dotenv

# Load environment variables at package import time
load_dotenv() 