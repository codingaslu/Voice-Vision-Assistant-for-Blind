"""
Calendar integration tool for Ally Vision Assistant

This tool integrates with Google Calendar API to manage calendar events,
allowing users to create events and view upcoming appointments.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from src.config import get_config
from src.utils import get_credentials, get_current_date_time

# Simple logger without custom handler
logger = logging.getLogger("calendar-tool")

class CalendarTool:
    """Handler for Google Calendar API integration."""
    
    def __init__(self):
        """Initialize the Google Calendar API handler."""
        # Get config from settings
        config = get_config()
        
        # Store initialization status
        self.is_ready = False
        
        try:
            # Get reference to credentials utility 
            self.is_ready = True
            logger.info("Calendar tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize calendar tool: {e}")
    
    async def manage_calendar(self, action: str, **kwargs) -> str:
        """
        Unified method to manage calendar operations.
        
        Args:
            action: The action to perform ("add_event" or "get_events")
            **kwargs: Arguments specific to the action
                For add_event: title, description, start_time
                For get_events: start_date, end_date
            
        Returns:
            Result message from the performed action
        """
        if not self.is_ready:
            return "Calendar tool is not properly initialized."
        
        try:
            if action == "add_event":
                # Check if start time is missing and use current time
                if "start_time" not in kwargs or not kwargs["start_time"]:
                    current_time = get_current_date_time()
                    kwargs["start_time"] = current_time
                    logger.info(f"Using current time for new event: {current_time}")
                
                return await self._add_event(**kwargs)
            elif action == "get_events":
                return await self._get_events(**kwargs)
            else:
                return f"Unsupported calendar action: {action}"
        except Exception as e:
            error_msg = f"Unexpected error in calendar action {action}: {e}"
            logger.error(error_msg)
            return f"An unexpected error occurred: {str(e)}"
    
    async def _add_event(self, title: str, description: str, start_time: str) -> str:
        """
        Add an event to the user's primary calendar.
        
        Args:
            title: Title/summary of the event
            description: Description of the event
            start_time: Start time in ISO format
            
        Returns:
            Confirmation message with event ID or error message
        """
        try:
            # Get credentials and build service
            creds = get_credentials()
            service = build("calendar", "v3", credentials=creds)

            # Convert the string to a datetime object
            event_datetime = datetime.fromisoformat(start_time)

            # Create event with 1-hour duration by default
            event = {
                'summary': title,
                'description': description,
                'start': {
                    'dateTime': event_datetime.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': (event_datetime + timedelta(hours=1)).isoformat(),
                    'timeZone': 'UTC',
                },
            }

            # Insert the event
            event = service.events().insert(calendarId='primary', body=event).execute()
            logger.info(f"Event created: {title} at {start_time}")
            return f"Event created successfully. Event ID: {event.get('id')}"

        except HttpError as error:
            error_msg = f"Error creating calendar event: {error}"
            logger.error(error_msg)
            return f"An error occurred: {error}"
    
    async def _get_events(self, start_date: str, end_date: str) -> str:
        """
        Get calendar events between specified dates.
        
        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format
            
        Returns:
            Formatted list of events or error message
        """
        try:
            # Get credentials and build service
            creds = get_credentials()
            service = build("calendar", "v3", credentials=creds)

            # Convert string times to datetime objects and ensure they're in UTC
            start_datetime = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
            end_datetime = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

            # Format date-times in RFC3339 format
            start_rfc3339 = start_datetime.isoformat().replace('+00:00', 'Z')
            end_rfc3339 = end_datetime.isoformat().replace('+00:00', 'Z')

            # Get events
            events_result = service.events().list(
                calendarId='primary',
                timeMin=start_rfc3339,
                timeMax=end_rfc3339,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])

            if not events:
                return "No events found in the specified time range."

            # Format events for output
            event_list = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                description = event.get('description', 'No description')
                event_list.append(f"Event: {event['summary']}, Description: {description}, Start: {start}")

            logger.info(f"Retrieved {len(event_list)} events")
            if event_list:
                return "\n".join(event_list)
            return "No events found for these dates"

        except HttpError as error:
            error_msg = f"Error retrieving calendar events: {error}"
            logger.error(error_msg)
            return f"An error occurred: {error}" 