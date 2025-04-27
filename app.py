#!/usr/bin/env python
"""
Ally Vision Assistant - Entry Point
A voice and vision assistant for blind and visually impaired users.
"""

import logging
import sys
# Import src package first to load environment variables
import src

# Configure optimized logging after environment variables are loaded
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
for log_module in ["livekit", "livekit.agents"]: logging.getLogger(log_module).setLevel(logging.ERROR)
for log_module in ["livekit.plugins", "livekit.rtc", "primp", "httpx"]: logging.getLogger(log_module).setLevel(logging.WARNING)

# Import after environment variables are loaded
from livekit.agents import WorkerOptions, cli
from src.main import entrypoint

# Main application logger
logger = logging.getLogger("ally-vision-app")

if __name__ == "__main__":
    logger.info("Starting Ally Vision Assistant")
    
    # Run the application using the entrypoint from main.py
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint)) 