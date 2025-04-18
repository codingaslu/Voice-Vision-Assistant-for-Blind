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

# Reset any existing handlers
for handler in logging.root.handlers:
    logging.root.removeHandler(handler)

# Create a single console handler for all logs
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.root.addHandler(console)
logging.root.setLevel(logging.INFO)

# Silence specific noisy loggers
logging.getLogger("livekit").setLevel(logging.ERROR)
logging.getLogger("livekit.agents").setLevel(logging.ERROR)
logging.getLogger("livekit.plugins").setLevel(logging.WARNING)
logging.getLogger("livekit.rtc").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Setup main application logger
logger = logging.getLogger("ally-vision-app")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    logger.info("Starting Ally Vision Assistant")
    
    # Run the application using the entrypoint from main.py
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint)) 