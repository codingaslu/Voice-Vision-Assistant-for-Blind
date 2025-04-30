import logging
import asyncio
from dataclasses import dataclass, field
from typing import Annotated, Optional, List
import time
from pydantic import Field
from livekit.agents import JobContext, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import deepgram, openai, elevenlabs, silero
from livekit.agents.llm.chat_context import ChatContext, ImageContent
from livekit.agents.llm.llm import ChatChunk, ChoiceDelta

# Import the tools
from .tools.visual import VisualProcessor
from .tools.internet_search import InternetSearch
from .tools.groq_handler import GroqHandler
from .tools.google_places import PlacesSearch
from .tools.calendar import CalendarTool
from .tools.communication import CommunicationTool

# Logger
logger = logging.getLogger("ally-vision-agent")

# Constants
VISION_SYSTEM_PROMPT = "You are Ally, a vision assistant for blind users. Provide extremely concise and clear descriptions. Focus only on the most important elements needed to answer the user's specific question. Ignore changes in color, brightness, or shading caused by sunglasses. Be direct and to the point. Answer as if the scene is fully visible without distortion. Avoid phrases like \"as seen\" or \"looks like,\" and do not describe things based only on visual appearance. When helpful, explain how a screen reader might announce elements. Tailor your response to what a blind user would truly want to know."

@dataclass
class UserData:
    # Core settings
    current_tool: str = "general"
    last_query: str = ""
    last_response: str = ""
    room_ctx: Optional[JobContext] = None
    
    # Tool instances
    visual_processor: VisualProcessor = None
    internet_search: InternetSearch = None
    groq_handler: Optional[GroqHandler] = None
    places_search: Optional[PlacesSearch] = None
    calendar_tool: Optional[CalendarTool] = None
    communication_tool: Optional[CommunicationTool] = None
    
    # Vision processing state
    _model_choice: Optional[str] = None
    _groq_analysis: Optional[str] = None
    _gpt_chunks: List[str] = field(default_factory=list)
    _analysis_complete: bool = False
    _add_chunk_callback = None

RunContext_T = RunContext[UserData]

class AllyVisionAgent(Agent):
    """
    A single agent that modifies LLM output before sending to TTS.
    Handles two different tools: internet and visual.
    """
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are Ally, a vision assistant for blind and visually impaired users.
            
            VISUAL QUERIES:
            - For questions about visual content, use analyze_vision tool
            - For ANY request about what you see, use analyze_vision tool immediately
            
            INTERNET SEARCHES:
            - For facts, data, news: use search_internet tool
            - Include sources when providing information from web
            - Use this to check latest information
            
            PLACES SEARCHES:
            - For restaurants, businesses, points of interest: use search_places tool
            - Help find locations, addresses, and business information
            
            CALENDAR MANAGEMENT:
            - For adding events to calendar: use manage_calendar with action="add_event"
            - For checking schedule: use manage_calendar with action="get_events"
            - Help manage appointments and meetings
            
            COMMUNICATION:
            - For finding contact information: use manage_communication with action="find_contact"
            - For reading emails: use manage_communication with action="read_emails"
            - For sending emails: use manage_communication with action="send_email"
            - Help stay connected with contacts

            GENERAL QUESTIONS:
            - Use your knowledge for general questions not requiring vision or search
            - Keep responses concise, clear, and helpful
            
            CONVERSATION:
            - Avoid mentioning tools or technical details
            - Focus on helpful, concise information
            - Be clear and jargon-free
            
            IMPORTANT NOTE:
            - Don't tell like this: "I don't actually "see" or have sight like a human. However, I can analyze visual content when provided through a tool. If there's a specific image or scene you want analyzed, feel free to let me know!" instead use tools to answer the question.
            """,
            stt=deepgram.STT(
                model="nova-2-general",
                smart_format=True,
                punctuate=True,
                language="en-US"
            ),
            llm=openai.LLM(model="gpt-4o", parallel_tool_calls=False),
            tts=elevenlabs.TTS(model="eleven_multilingual_v2")
        )
    
    async def on_enter(self) -> None:
        """Called when the agent is first started"""
        logger.info("Entering AllyVisionAgent")
        
        # Use session.say() to ensure the greeting is sent directly
        greeting = "Hi there! How can I help?"
        await self.session.say(greeting)
    
    async def on_message(self, text: str) -> None:
        """Override on_message to log user queries"""
        logger.info(f"USER QUERY: {text}")
        userdata: UserData = self.session.userdata
        userdata.last_query = text
        
        # Reset to general mode for each new query unless overridden by a tool
        userdata.current_tool = "general"
        
        await super().on_message(text)

    @function_tool()
    async def search_places(
        self,
        context: RunContext_T,
        query: Annotated[str, Field(description="Search query for places, businesses, restaurants, or points of interest")]
    ) -> str:
        """
        Search for places, businesses, and points of interest.
        Provides details like address, ratings, and opening hours.
        """
        userdata = context.userdata
        
        # Switch to places mode
        userdata.current_tool = "places"
        
        # Ensure we have the places search tool
        if userdata.places_search is None:
            userdata.places_search = PlacesSearch()
            logger.info("Created places search tool on demand")
        
        # Log the search query
        logger.info(f"Searching places: {query[:30]}...")
        
        try:
            # Perform places search
            results = await userdata.places_search.search_places(query)
            
            # Store the response for future reference
            userdata.last_response = results
            
            # Switch back to general mode after completing places search
            userdata.current_tool = "general"
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching for places: {e}")
            return f"I encountered an error while searching for places related to '{query}': {str(e)}"

    @function_tool()
    async def search_internet(
        self,
        context: RunContext_T,
        query: Annotated[str, Field(description="Search query for the web")]
    ) -> str:
        """
        Search for up-to-date information on the web.
        Provides results with source links.
        """
        userdata = context.userdata
        
        # Switch to internet mode
        userdata.current_tool = "internet"
        
        # Log the search query
        logger.info(f"Searching: {query[:30]}...")
        
        try:
            # Perform comprehensive search
            search_results = await userdata.internet_search.search(query)
            
            # Format the results for readability
            formatted_results = userdata.internet_search.format_results(search_results)
            
            # Add introduction
            response = f"Here's what I found about '{query}':\n\n{formatted_results}"
            
            # Store the response for future reference
            userdata.last_response = response
            
            # Switch back to general mode after completing internet search
            userdata.current_tool = "general"
            
            return response
            
        except Exception as e:
            logger.error(f"Error searching the internet: {e}")
            return f"I encountered an error while searching for information about '{query}': {str(e)}"
    
    async def _run_gpt_analysis(self, userdata, analysis_llm, visual_ctx):
        """Run GPT analysis and stream results through callback"""
        try:
            async with analysis_llm.chat(chat_ctx=visual_ctx) as stream:
                async for chunk in stream:
                    if chunk and hasattr(chunk.delta, 'content') and chunk.delta.content:
                        content = chunk.delta.content
                        userdata._gpt_chunks.append(content)
                        if userdata._add_chunk_callback:
                            userdata._add_chunk_callback(content)
            
            userdata._analysis_complete = True
        except Exception as e:
            # Handle error and add to stream
            error_msg = f"Error processing image: {str(e)}"
            userdata._gpt_chunks.append(error_msg)
            if userdata._add_chunk_callback:
                userdata._add_chunk_callback(error_msg)
            userdata._analysis_complete = True
    
    @function_tool()
    async def analyze_vision(
        self,
        context: RunContext_T,
        query: Annotated[str, Field(description="Query about the visual scene to analyze")]
    ) -> str:
        """Capture and analyze the current scene."""
        userdata = context.userdata
        userdata.current_tool = "visual"
        
        try:
            # Capture image - the error handling is already managed in visual_processor.capture_frame
            image = await userdata.visual_processor.capture_frame(userdata.room_ctx.room)
            if image is None:
                return "I couldn't capture a clear image from the camera."
            
            # Reset state
            userdata._gpt_chunks.clear()
            userdata._add_chunk_callback = None
            userdata._analysis_complete = False
            
            # Set up visual context
            visual_ctx = ChatContext()
            visual_ctx.add_message(role="system", content=VISION_SYSTEM_PROMPT)
            visual_ctx.add_message(
                role="user",
                content=[
                    f"Answer this query about the seeing the visual in front of the user.please dont mention the image or user word in your answer: {query}",
                    ImageContent(image=image)
                ]
            )
            
            # Initialize LLM and start analysis
            analysis_llm = openai.LLM(model="gpt-4o")
            asyncio.create_task(self._run_gpt_analysis(userdata, analysis_llm, visual_ctx))
            
            # Try Groq if available - simplified check
            groq_handler = userdata.groq_handler
            if groq_handler and getattr(groq_handler, 'is_ready', False):
                try:
                    model_choice, groq_analysis, _ = await groq_handler.model_choice_with_analysis(image, query)
                    userdata._model_choice = model_choice
                    userdata._groq_analysis = groq_analysis
                except Exception:
                    userdata._model_choice = "GPT" 
            else:
                userdata._model_choice = "GPT"
            
            return "Processing visual analysis..."
            
        except Exception as e:
            logger.error(f"Error in analyze_vision: {e}")
            return "I encountered an error while processing the visual information."
    
    async def _process_stream(self, chat_ctx, tools, userdata):
        """Process and stream responses based on current context"""
        full_response = ""
        
        # Handle vision analysis
        if userdata.current_tool == "visual" and userdata._model_choice:
            # GPT streaming case
            if userdata._model_choice == "GPT":
                # Setup streaming
                chunk_queue = asyncio.Queue()
                done_event = asyncio.Event()
                userdata._add_chunk_callback = lambda content: chunk_queue.put_nowait(content)
                
                # Add existing chunks
                for chunk in userdata._gpt_chunks:
                    chunk_queue.put_nowait(chunk)
                
                # Process chunks
                try:
                    while not (done_event.is_set() and chunk_queue.empty()):
                        try:
                            # Get next chunk with timeout
                            chunk = await asyncio.wait_for(chunk_queue.get(), timeout=0.1)
                            full_response += chunk
                            yield ChatChunk(
                                id=f"gptcmpl-{time.time()}",
                                delta=ChoiceDelta(role="assistant", content=chunk),
                                usage=None
                            )
                        except asyncio.TimeoutError:
                            # Set done if analysis is complete and queue is empty
                            if userdata._analysis_complete and chunk_queue.empty():
                                done_event.set()
                finally:
                    # Cleanup
                    userdata._add_chunk_callback = None
                    userdata._gpt_chunks.clear()
                    done_event.set()
                    logger.info(f"GPT vision analysis completed")
            
            # LLAMA/Groq single response
            elif userdata._groq_analysis:
                full_response = userdata._groq_analysis
                yield ChatChunk(
                    id=f"groqcmpl-{time.time()}",
                    delta=ChoiceDelta(role="assistant", content=full_response),
                    usage=None
                )
                userdata._groq_analysis = None
                logger.info(f"LLAMA vision analysis completed")
            
            # Fallback case
            else:
                full_response = "I couldn't analyze the image properly."
                yield ChatChunk(
                    id=f"groqcmpl-{time.time()}",
                    delta=ChoiceDelta(role="assistant", content=full_response),
                    usage=None
                )
            
            userdata._model_choice = None
        
        # Standard LLM processing
        else:
            async with self.llm.chat(chat_ctx=chat_ctx, tools=tools) as stream:
                async for chunk in stream:
                    if chunk and hasattr(chunk.delta, 'content') and chunk.delta.content:
                        full_response += chunk.delta.content
                    yield chunk
        
        userdata.last_response = full_response
    
    async def llm_node(self, chat_ctx, tools, model_settings=None):
        """Override llm_node to modify the output before sending to TTS"""
        userdata = self.session.userdata
        return self._process_stream(chat_ctx, tools, userdata)

    @function_tool()
    async def manage_calendar(
        self,
        context: RunContext_T,
        action: Annotated[str, Field(description="Action to perform: 'add_event' or 'get_events'")],
        title: Annotated[Optional[str], Field(description="Title of the event (for add_event only)")] = None,
        description: Annotated[Optional[str], Field(description="Description of the event (for add_event only)")] = None,
        start_time: Annotated[Optional[str], Field(description="Start time of the event in ISO format (for add_event only)")] = None,
        start_date: Annotated[Optional[str], Field(description="Start date in ISO format (for get_events only)")] = None,
        end_date: Annotated[Optional[str], Field(description="End date in ISO format (for get_events only)")] = None,
    ) -> str:
        """
        Manage calendar events - add new events or view scheduled events.
        
        For adding events, specify action='add_event', title, description, and start_time.
        For viewing events, specify action='get_events', start_date, and end_date.
        """
        userdata = context.userdata
        
        # Switch to calendar mode
        userdata.current_tool = "calendar"
        
        # Ensure we have the calendar tool
        if userdata.calendar_tool is None:
            userdata.calendar_tool = CalendarTool()
            logger.info("Created calendar tool on demand")
        
        # Prepare kwargs based on action
        kwargs = {}
        if action == "add_event":
            if not all([title, start_time]):
                return "Title and start time are required for adding events."
            kwargs = {
                "title": title,
                "description": description or "",
                "start_time": start_time
            }
            logger.info(f"Adding calendar event: {title} at {start_time}")
        elif action == "get_events":
            if not all([start_date, end_date]):
                return "Start date and end date are required for viewing events."
            kwargs = {
                "start_date": start_date,
                "end_date": end_date
            }
            logger.info(f"Getting calendar events from {start_date} to {end_date}")
        else:
            return f"Unsupported calendar action: {action}"
        
        try:
            # Call the unified calendar management method
            result = await userdata.calendar_tool.manage_calendar(action, **kwargs)
            
            # Store the response for future reference
            userdata.last_response = result
            
            # Switch back to general mode after completing calendar action
            userdata.current_tool = "general"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in calendar action {action}: {e}")
            return f"I encountered an error while performing the calendar operation: {str(e)}"

    @function_tool()
    async def manage_communication(
        self,
        context: RunContext_T,
        action: Annotated[str, Field(description="Action to perform: 'find_contact', 'read_emails', or 'send_email'")],
        name: Annotated[Optional[str], Field(description="Name of the contact to find (for find_contact only)")] = None,
        from_date: Annotated[Optional[str], Field(description="From date in ISO format (for read_emails only)")] = None,
        to_date: Annotated[Optional[str], Field(description="To date in ISO format (for read_emails only)")] = None,
        email: Annotated[Optional[str], Field(description="Email to filter by (optional for read_emails) or recipient (for send_email)")] = None,
        subject: Annotated[Optional[str], Field(description="Email subject (for send_email only)")] = None,
        body: Annotated[Optional[str], Field(description="Email body content (for send_email only)")] = None,
    ) -> str:
        """
        Manage contacts and emails - find contacts, read emails, or send messages.
        
        For finding contacts, specify action='find_contact' and name.
        For reading emails, specify action='read_emails', from_date, to_date, and optionally email.
        For sending emails, specify action='send_email', email (recipient), subject, and body.
        """
        userdata = context.userdata
        
        # Switch to communication mode
        userdata.current_tool = "communication"
        
        # Ensure we have the communication tool
        if userdata.communication_tool is None:
            userdata.communication_tool = CommunicationTool()
            logger.info("Created communication tool on demand")
        
        # Prepare kwargs based on action
        kwargs = {}
        if action == "find_contact":
            if not name:
                return "Contact name is required for finding contacts."
            kwargs = {"name": name}
            logger.info(f"Finding contact information for: {name}")
        elif action == "read_emails":
            if not all([from_date, to_date]):
                return "From date and to date are required for reading emails."
            kwargs = {
                "from_date": from_date,
                "to_date": to_date
            }
            if email:
                kwargs["email"] = email
            logger.info(f"Reading emails from {from_date} to {to_date}" + (f" from {email}" if email else ""))
        elif action == "send_email":
            if not all([email, subject, body]):
                return "Recipient email, subject, and body are required for sending emails."
            kwargs = {
                "to": email,
                "subject": subject,
                "body": body
            }
            logger.info(f"Sending email to: {email} with subject: {subject}")
        else:
            return f"Unsupported communication action: {action}"
        
        try:
            # Call the unified communication management method
            result = await userdata.communication_tool.manage_communication(action, **kwargs)
            
            # Store the response for future reference
            userdata.last_response = result
            
            # Switch back to general mode after completing communication action
            userdata.current_tool = "general"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in communication action {action}: {e}")
            return f"I encountered an error while performing the communication operation: {str(e)}"

async def entrypoint(ctx: JobContext):
    """Set up and start the voice agent with all required tools"""
    try:
        # Connect and initialize
        await ctx.connect()
        
        # Create user data with tools
        userdata = UserData()
        userdata.room_ctx = ctx
        userdata.visual_processor = VisualProcessor()
        userdata.internet_search = InternetSearch()
        userdata.places_search = PlacesSearch()
        userdata.calendar_tool = CalendarTool()
        userdata.communication_tool = CommunicationTool()

        # Initialize optional components with graceful fallbacks
        try:
            userdata.groq_handler = GroqHandler()
        except Exception as e:
            logger.warning(f"Vision will use GPT only: {e}")
            
        try:
            await userdata.visual_processor.enable_camera(ctx.room)
        except Exception as e:
            logger.warning(f"Camera setup failed: {e}")
        
        # Create and start agent
        agent = AllyVisionAgent()
        agent_session = AgentSession[UserData](
            userdata=userdata,
            stt=deepgram.STT(model="nova-2-general", smart_format=True, punctuate=True, language="en-US"),
            llm=openai.LLM(model="gpt-4o", parallel_tool_calls=False),
            tts=elevenlabs.TTS(model="eleven_multilingual_v2"),
            vad=silero.VAD.load(),
            max_tool_steps=3,
        )
        
        await agent_session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(),
        )
        
    except Exception as e:
        logger.error(f"Agent startup failed: {e}")