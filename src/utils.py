"""
Utility functions for Ally Vision Assistant

This module provides common utility functions used across tools,
especially for authentication and date/time operations.
"""

import os 
import logging
from datetime import datetime
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# Simple logger without custom handler
logger = logging.getLogger("utils")

# Define scopes for various Google APIs
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/contacts.readonly",
    'https://www.googleapis.com/auth/gmail.readonly'
]

def get_current_date_time():
    """
    Get current date and time in a human-readable format.
    
    Returns:
        String formatted as YYYY-MM-DD HH:MM
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M")
        
def get_credentials():
    """
    Get and refresh Google API credentials.
    
    This function handles OAuth2 authentication for Google's APIs,
    including refreshing expired tokens and creating new ones when needed.
    
    Returns:
        Google OAuth2 credentials object
    
    Raises:
        FileNotFoundError: If credentials.json is missing
        Exception: Other authentication errors
    """
    creds = None
    
    try:
        # Check for existing token
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            
        # Refresh or create new token if needed
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired credentials")
                creds.refresh(Request())
            else:
                logger.info("Creating new credentials")
                if not os.path.exists('credentials.json'):
                    raise FileNotFoundError(
                        "credentials.json file not found. Please obtain OAuth credentials " +
                        "from the Google Developer Console."
                    )
                    
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
                
            # Save the credentials for future use
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
                logger.info("Credentials saved to token.json")
                
        return creds
        
    except FileNotFoundError as e:
        logger.error(f"Credentials file error: {e}")
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise 