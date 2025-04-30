"""
Communication tools for Ally Vision Assistant

This tool integrates with Google People and Gmail APIs to manage contacts
and emails, allowing users to find contacts, read emails, and send messages.
"""

import os
import re
import logging
import smtplib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parsedate_to_datetime
from src.config import get_config
from src.utils import get_credentials, get_current_date_time

# Simple logger without custom handler
logger = logging.getLogger("communication-tool")

class CommunicationTool:
    """Handler for Contacts and Email API integration."""
    
    def __init__(self):
        """Initialize the Communication API handler."""
        # Get config from settings
        config = get_config()
        
        # Store initialization status
        self.is_ready = False
        
        try:
            # Store email credentials
            self.sender_email = config.get("GMAIL_MAIL") or os.getenv("GMAIL_MAIL")
            self.app_password = config.get("GMAIL_APP_PASSWORD") or os.getenv("GMAIL_APP_PASSWORD")
            
            if not self.sender_email or not self.app_password:
                logger.warning("Gmail credentials not found. Send email functionality will be limited.")
            
            self.is_ready = True
            logger.info("Communication tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize communication tool: {e}")
    
    async def manage_communication(self, action: str, **kwargs) -> str:
        """
        Unified method to manage communication operations.
        
        Args:
            action: The action to perform ("find_contact", "read_emails", or "send_email")
            **kwargs: Arguments specific to the action
                For find_contact: name
                For read_emails: from_date, to_date, email (optional)
                For send_email: to, subject, body
            
        Returns:
            Result message from the performed action
        """
        if not self.is_ready:
            return "Communication tool is not properly initialized."
        
        try:
            # If reading emails without dates, default to today
            if action == "read_emails":
                # Use current date/time if not provided
                if "from_date" not in kwargs or not kwargs["from_date"]:
                    now = datetime.now()
                    # Default to start of today
                    kwargs["from_date"] = datetime(now.year, now.month, now.day).isoformat()
                    logger.info(f"Using default from_date: {kwargs['from_date']}")
                
                if "to_date" not in kwargs or not kwargs["to_date"]:
                    # Default to now
                    kwargs["to_date"] = datetime.now().isoformat()
                    logger.info(f"Using default to_date: {kwargs['to_date']}")
            
            if action == "find_contact":
                return await self._find_contact(**kwargs)
            elif action == "read_emails":
                return await self._read_emails(**kwargs)
            elif action == "send_email":
                return await self._send_email(**kwargs)
            else:
                return f"Unsupported communication action: {action}"
        except Exception as e:
            error_msg = f"Unexpected error in communication action {action}: {e}"
            logger.error(error_msg)
            return f"An unexpected error occurred: {str(e)}"
    
    async def _find_contact(self, name: str) -> str:
        """
        Find contact information by name.
        
        Args:
            name: Name of the contact to find
            
        Returns:
            Contact information or error message
        """
        try:
            # Get credentials and build service
            creds = get_credentials()
            service = build('people', 'v1', credentials=creds)

            # Search for the contact
            results = service.people().searchContacts(
                query=name,
                readMask='names,phoneNumbers,emailAddresses'
            ).execute()

            connections = results.get('results', [])

            if not connections:
                return f"No contact found with the name: {name}"

            matching_contacts = []

            for connection in connections:
                contact = connection['person']
                names = contact.get('names', [])
                if names:
                    unstructured_name = names[0].get('unstructuredName', '').lower()
                    # Prepare regex to identify first and last names
                    first_name_pattern = r'^(\w+)'  # Match first word
                    last_name_pattern = r'(\w+)$'   # Match last word
                    first_match = re.search(first_name_pattern, unstructured_name)
                    last_match = re.search(last_name_pattern, unstructured_name)

                    if (first_match and name.lower() == first_match.group(1)) or \
                        (last_match and name.lower() == last_match.group(1)) or \
                        (name.lower() == unstructured_name):
                        full_name = names[0].get('displayName', 'N/A')
                        phone_numbers = [phone.get('value', 'N/A') for phone in contact.get('phoneNumbers', [])]
                        emails = [email.get('value', 'N/A') for email in contact.get('emailAddresses', [])]

                        matching_contacts.append({
                            'name': full_name,
                            'phone_numbers': phone_numbers,
                            'emails': emails
                        })

            if not matching_contacts:
                return f"No contact found with the matching criteria: {name}"

            # Format the contacts for better display
            formatted_contacts = []
            for i, contact in enumerate(matching_contacts, 1):
                contact_info = [f"Contact {i}: {contact['name']}"]
                
                if contact['emails']:
                    contact_info.append(f"Emails: {', '.join(contact['emails'])}")
                
                if contact['phone_numbers']:
                    contact_info.append(f"Phones: {', '.join(contact['phone_numbers'])}")
                    
                formatted_contacts.append("\n".join(contact_info))
                
            return "\n\n".join(formatted_contacts)

        except HttpError as error:
            error_msg = f"Error finding contact: {error}"
            logger.error(error_msg)
            return f"An error occurred: {error}"

    async def _read_emails(self, from_date: str, to_date: str, email: Optional[str] = None) -> str:
        """
        Read emails from inbox within specified date range.
        
        Args:
            from_date: Start date in ISO format
            to_date: End date in ISO format
            email: Optional sender email to filter by
            
        Returns:
            Formatted list of emails or error message
        """
        try:
            # Get credentials and build service
            creds = get_credentials()
            service = build('gmail', 'v1', credentials=creds)

            # Convert datetime objects to timestamps
            from_date_ts = int(datetime.fromisoformat(from_date).timestamp())
            to_date_ts = int(datetime.fromisoformat(to_date).timestamp())

            query = f'after:{from_date_ts} before:{to_date_ts}'
            if email:
                query += f' from:{email}'

            results = service.users().messages().list(userId='me', q=query).execute()
            messages = results.get('messages', [])

            if not messages:
                return "No emails found in the specified time range."

            email_list = []
            # Get only first 10 emails to avoid long responses
            for message in messages[:10]:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()

                subject = next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject'), 'No Subject')
                from_email = next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'From'), 'Unknown Sender')
                date = next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'Date'), '')
                
                # Parse and format date
                try:
                    date_obj = parsedate_to_datetime(date)
                    if date_obj.tzinfo is None:
                        date_obj = date_obj.replace(tzinfo=timezone.utc)
                    formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S %Z")
                except:
                    formatted_date = date

                snippet = msg['snippet']
                email_list.append(f"From: {from_email}\nSubject: {subject}\nDate: {formatted_date}\nSnippet: {snippet}\n")

            logger.info(f"Retrieved {len(email_list)} emails")
            
            if len(messages) > 10:
                email_list.append(f"\n(Showing 10 of {len(messages)} emails. To see more, please narrow your search criteria.)")
                
            return "\n".join(email_list)

        except HttpError as error:
            error_msg = f"Error reading emails: {error}"
            logger.error(error_msg)
            return f"An error occurred: {error}"

    async def _send_email(self, to: str, subject: str, body: str) -> str:
        """
        Send an email to a recipient.
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body content
            
        Returns:
            Confirmation message or error message
        """
        try:
            if not self.sender_email or not self.app_password:
                return "Cannot send email: Gmail credentials not configured. Please set GMAIL_MAIL and GMAIL_APP_PASSWORD."

            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = to
            msg['Subject'] = subject
            
            # Add current timestamp in the body
            body_with_timestamp = f"{body}\n\nSent: {get_current_date_time()}"
            msg.attach(MIMEText(body_with_timestamp, 'plain'))

            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(self.sender_email, self.app_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, to, text)
            server.quit()
            
            logger.info(f"Email sent to {to} with subject: {subject}")
            return f"Email sent successfully to {to}."
            
        except Exception as e:
            error_msg = f"Error sending email: {e}"
            logger.error(error_msg)
            return f"Email was not sent successfully, error: {str(e)}" 