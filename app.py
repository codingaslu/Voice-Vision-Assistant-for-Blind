#!/usr/bin/env python
"""
Ally Vision Assistant - Entry Point
A voice and vision assistant for blind and visually impaired users.
"""

import logging
import sys
from dotenv import load_dotenv
from livekit.agents import WorkerOptions, cli

# Import main entrypoint function
from src.main import entrypoint

# Configure optimized logging (all in one block)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
for log_module in ["livekit", "livekit.agents"]: logging.getLogger(log_module).setLevel(logging.ERROR)
for log_module in ["livekit.plugins", "livekit.rtc", "primp", "httpx"]: logging.getLogger(log_module).setLevel(logging.WARNING)

# Main application logger
logger = logging.getLogger("ally-vision-app")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    logger.info("Starting Ally Vision Assistant")
    
    # Run the application using the entrypoint from main.py
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint)) 