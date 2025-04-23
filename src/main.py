import logging
import asyncio
from dataclasses import dataclass, field
from typing import Annotated, Optional, List, Dict, Any
import time

from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import deepgram, openai, elevenlabs, silero
from livekit.agents.llm.chat_context import ChatContext, ImageContent
from livekit.agents.llm.llm import ChatChunk, ChoiceDelta

# Import the tools
from .tools.visual import VisualProcessor
from .tools.internet_search import InternetSearch

# Simple logger without custom handler, will use root logger's config
logger = logging.getLogger("ally-vision-agent")

# Load environment variables
load_dotenv()

@dataclass
class UserData:
    current_tool: str = "general"  #  tool: "visual" or "internet"
    last_query: str = ""
    last_response: str = ""
    visual_processor: Optional[VisualProcessor] = None
    internet_search: Optional[InternetSearch] = None
    room_ctx: Optional[JobContext] = None  # Store JobContext for room access

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
            - For ANY request about what you see, use analyze_vision tool immediately
            - Handling: GPT-4o for scenes/objects, Groq for people/sensitive content
            
            INTERNET SEARCHES:
            - For facts, data, news: use search_internet tool
            - Include sources when providing information from web
            - Use this to check latest information
            
            GENERAL QUESTIONS:
            - Use your knowledge for general questions not requiring vision or search
            - Keep responses concise, clear, and helpful
            
            CONVERSATION:
            - Avoid mentioning tools or technical details
            - Focus on helpful, concise information
            - Be clear and jargon-free
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
        
        # Get current user data
        userdata: UserData = self.session.userdata
        
        # Initialize the tools if they don't exist
        if userdata.visual_processor is None:
            userdata.visual_processor = VisualProcessor()
            logger.info("Initialized visual processor")
            
        if userdata.internet_search is None:
            userdata.internet_search = InternetSearch()
            logger.info("Initialized internet search tool")
        
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
        
        # Ensure we have the internet search tool
        if userdata.internet_search is None:
            userdata.internet_search = InternetSearch()
            logger.info("Created internet search tool on demand")
        
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
    
    @function_tool()
    async def analyze_vision(
        self,
        context: RunContext_T,
        query: Annotated[str, Field(description="Query about the visual scene to analyze")]
    ) -> str:
        """
        Capture and analyze the current scene.
        
        Uses a single API call to:
        1. Classify if humans are present (routes to LLAMA) or not (routes to GPT-4o)
        2. Get analysis for humans directly from Groq
        3. Preload GPT context for non-human scenes
        """
        userdata = context.userdata
        
        logger.info(f"Visual analysis: {query[:30]}...")
        
        try:
            # Ensure we have access to the room
            if userdata.room_ctx is None:
                logger.error("No room context available")
                return "I couldn't access the camera because the room connection is not available."
                
            room = userdata.room_ctx.room
            if not room:
                logger.error("No room available in stored context")
                return "I couldn't access the camera because the room connection is not available."
            
            # Ensure we have a visual processor
            if userdata.visual_processor is None:
                userdata.visual_processor = VisualProcessor()
                logger.info("Created visual processor on demand")
                
            # Switch to visual mode
            userdata.current_tool = "visual"
            
            # Capture frame first
            image = await userdata.visual_processor.capture_frame(room)
            if image is None:
                return "Failed to capture an image from the camera."
            
            # Get Groq handler
            groq_handler = userdata.visual_processor._groq_handler
            
            # Check if Groq handler is None and initialize if needed
            if groq_handler is None:
                try:
                    # Initialize Groq handler on demand
                    from src.tools.groq_handler import GroqHandler
                    userdata.visual_processor._groq_handler = GroqHandler()
                    groq_handler = userdata.visual_processor._groq_handler
                    logger.info("Initialized Groq handler on demand")
                except Exception as e:
                    logger.error(f"Failed to initialize Groq handler: {e}")
                    return "I'm having trouble with the vision system. Please try again."
            
            # Get model choice and analysis in a single call
            try:
                model_choice, groq_analysis, error = await groq_handler.model_choice_with_analysis(image, query)
                
                if error:
                    logger.error(f"Error from model_choice_with_analysis: {error}")
                    model_choice = "GPT"  # Default to GPT on error
            except Exception as e:
                logger.error(f"Unexpected error from Groq handler: {e}")
                model_choice = "GPT"  # Default to GPT on unexpected errors
                groq_analysis = ""
            
            logger.info(f"Model choice: {model_choice}")
            
            # Store the decision and analysis in userdata for llm_node to use
            userdata.visual_processor._model_choice = model_choice
            userdata.visual_processor._groq_analysis = groq_analysis
            userdata.visual_processor._last_image = image
            userdata.visual_processor._last_query = query
            
            # Initialize chunks array and callback for streaming
            userdata.visual_processor._gpt_chunks = []
            userdata.visual_processor._add_chunk_callback = None
            
            # If model choice is GPT, start preparing the GPT analysis context in parallel
            if model_choice == "GPT":
                # Log the query explicitly
                logger.info(f"Preparing GPT-4o analysis with query: '{query}'")
                
                # Create chat context for GPT
                visual_ctx = ChatContext()
                visual_ctx.add_message(
                    role="system",
                    content="You are Ally, a vision assistant for blind users. Give extremely concise descriptions. Focus directly on answering the user's specific question. Prioritize the most important visual elements relevant to their query. Avoid lengthy descriptions of background or irrelevant details. Be direct and to the point."
                )
                
                # Add the image and query
                visual_ctx.add_message(
                    role="user",
                    content=[
                        f"Answer briefly: {query}",
                        ImageContent(image=image)
                    ]
                )
                
                # Initialize LLM for analysis
                analysis_llm = openai.LLM(model="gpt-4o")
                
                # Start the GPT-4o analysis in parallel with streaming support
                async def run_gpt_analysis():
                    try:
                        logger.info(f"Running async GPT analysis with query: {query[:30]}...")
                        
                        # Initialize the analysis_complete flag to False
                        userdata.visual_processor._analysis_complete = False
                        
                        async with analysis_llm.chat(chat_ctx=visual_ctx) as stream:
                            async for chunk in stream:
                                if chunk and hasattr(chunk.delta, 'content') and chunk.delta.content:
                                    content = chunk.delta.content
                                    
                                    # Store the chunk for immediate access
                                    userdata.visual_processor._gpt_chunks.append(content)
                                    
                                    # If there's an active callback, send the chunk immediately
                                    if hasattr(userdata.visual_processor, '_add_chunk_callback') and userdata.visual_processor._add_chunk_callback:
                                        await userdata.visual_processor._add_chunk_callback(content)
                        
                        # Set the analysis_complete flag to True
                        userdata.visual_processor._analysis_complete = True
                        logger.info("Asynchronous GPT analysis completed")
                    except Exception as e:
                        logger.error(f"Error in async GPT analysis: {e}")
                        # Add error message as a chunk if callback exists
                        error_msg = f"Error processing image: {str(e)}"
                        userdata.visual_processor._gpt_chunks.append(error_msg)
                        if hasattr(userdata.visual_processor, '_add_chunk_callback') and userdata.visual_processor._add_chunk_callback:
                            await userdata.visual_processor._add_chunk_callback(error_msg)
                        
                        # Set the analysis_complete flag to True even on error
                        userdata.visual_processor._analysis_complete = True
                
                # Start analysis without awaiting
                asyncio.create_task(run_gpt_analysis())
            
            # Return a placeholder - actual analysis will happen in llm_node
            return "Processing visual analysis..."
            
        except Exception as e:
            logger.error(f"Error in analyze_vision: {e}")
            return f"I encountered an error while trying to analyze what's in front of me: {str(e)}"
    
    async def llm_node(self, chat_ctx, tools, model_settings=None):
        """Override llm_node to modify the output before sending to TTS"""
        # Access the LLM directly from self instance
        userdata: UserData = self.session.userdata
        current_tool = userdata.current_tool
        
        # Keep track of the full response
        full_response = ""
        
        async def process_stream():
            nonlocal full_response
            
            # Check if we need to handle visual analysis
            if (current_tool == "visual" and 
                hasattr(userdata.visual_processor, '_model_choice') and 
                userdata.visual_processor._model_choice):
                
                # Choose model based on the decision
                if userdata.visual_processor._model_choice == "GPT":
                    # Initialize a queue to store the chunks as they arrive
                    chunk_queue = asyncio.Queue()
                    processing_done = asyncio.Event()
                    
                    # Set up a callback to add chunks to the queue as they arrive
                    async def add_chunk_to_queue(content):
                        await chunk_queue.put(content)
                    
                    # Check if we have partial results available
                    if hasattr(userdata.visual_processor, '_gpt_chunks') and userdata.visual_processor._gpt_chunks:
                        # Add existing chunks to the queue
                        for chunk in userdata.visual_processor._gpt_chunks:
                            await chunk_queue.put(chunk)
                    
                    # Set up a task to update the _gpt_chunks callback
                    userdata.visual_processor._add_chunk_callback = add_chunk_to_queue
                    
                    # Process chunks as they arrive
                    try:
                        while not processing_done.is_set() or not chunk_queue.empty():
                            try:
                                # Get a chunk with a timeout to avoid blocking forever
                                chunk = await asyncio.wait_for(chunk_queue.get(), timeout=0.1)
                                full_response += chunk
                                
                                # Create a proper ChatChunk with the chunk content
                                yield ChatChunk(
                                    id=f"gptcmpl-{time.time()}",
                                    delta=ChoiceDelta(
                                        role="assistant",
                                        content=chunk
                                    ),
                                    usage=None
                                )
                            except asyncio.TimeoutError:
                                # Check if analysis is complete
                                if hasattr(userdata.visual_processor, '_analysis_complete') and userdata.visual_processor._analysis_complete:
                                    if chunk_queue.empty():
                                        processing_done.set()
                                # No chunk available yet, continue waiting
                                continue
                    except Exception as e:
                        logger.error(f"Error processing GPT chunks: {e}")
                    finally:
                        # Clean up
                        processing_done.set()
                        userdata.visual_processor._add_chunk_callback = None
                        if hasattr(userdata.visual_processor, '_gpt_chunks'):
                            userdata.visual_processor._gpt_chunks = []
                    
                else:  # LLAMA via Groq
                    # Use the pre-generated Groq analysis
                    if (hasattr(userdata.visual_processor, '_groq_analysis') and 
                        userdata.visual_processor._groq_analysis):
                        logger.info("Using pre-generated Groq analysis")
                        analysis_text = userdata.visual_processor._groq_analysis
                        # Clear the stored analysis
                        userdata.visual_processor._groq_analysis = None
                        
                        # Create a proper ChatChunk with the analysis
                        chunk = ChatChunk(
                            id=f"groqcmpl-{time.time()}",
                            delta=ChoiceDelta(
                                role="assistant",
                                content=analysis_text
                            ),
                            usage=None
                        )
                        yield chunk
                        
                        # Store the full response
                        full_response = analysis_text
                    else:
                        error_msg = "No pre-generated Groq analysis available"
                        logger.error(error_msg)
                        # Create error chunk in proper format
                        chunk = ChatChunk(
                            id=f"groqcmpl-{time.time()}",
                            delta=ChoiceDelta(
                                role="assistant",
                                content=error_msg
                            ),
                            usage=None
                        )
                        yield chunk
                        full_response = error_msg
                
                # Store the response and reset model choice
                userdata.last_response = full_response
                userdata.visual_processor._model_choice = None
                return
            
            # For non-visual queries, use the standard LLM processing
            async with self.llm.chat(chat_ctx=chat_ctx, tools=tools, tool_choice=None) as stream:
                async for chunk in stream:
                    if chunk is None:
                        continue
                    
                    # Extract content from chunk
                    content = getattr(chunk.delta, 'content', None) if hasattr(chunk, 'delta') else str(chunk)
                    
                    if content is None:
                        yield chunk
                        continue
                    
                    # Append to full response
                    full_response += content
                    
                    # Update the chunk with processed content
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                        chunk.delta.content = content
                    else:
                        chunk = content
                    
                    yield chunk
            
            # Store the response
            userdata.last_response = full_response
        
        return process_stream()

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    # Initialize user data with tools and room context
    userdata = UserData()
    userdata.visual_processor = VisualProcessor()
    userdata.internet_search = InternetSearch()
    userdata.current_tool = "general"  # Explicitly set initial tool mode
    userdata.room_ctx = ctx  # Store the JobContext for room access
    
    # Enable the camera once at startup
    try:
        await userdata.visual_processor.enable_camera(ctx.room)
    except Exception as e:
        logger.error(f"Failed to enable camera: {e}")
    
    # Create agent session with VAD to avoid warning
    agent_session = AgentSession[UserData](
        userdata=userdata,
        stt=deepgram.STT(
            model="nova-2-general",
            smart_format=True,
            punctuate=True,
            language="en-US"
        ),
        llm=openai.LLM(model="gpt-4o", parallel_tool_calls=False),
        tts=elevenlabs.TTS(model="eleven_multilingual_v2"),
        vad=silero.VAD.load(),  # Add VAD to avoid warning
        max_tool_steps=3,
    )
    
    # Create processor agent
    processor_agent = AllyVisionAgent()
    
    # Start the agent session
    await agent_session.start(
        agent=processor_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )