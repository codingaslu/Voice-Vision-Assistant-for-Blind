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
    groq_fallback_used: bool = False  # Flag to track if Groq fallback was already used
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
        
        # Reset fallback flag for each new query
        userdata.groq_fallback_used = False
        
        await super().on_message(text)
    
    
    @function_tool()
    async def search_internet(
        self,
        context: RunContext_T,
        query: Annotated[str, Field(description="The search query to look up information on the web")]
    ) -> str:
        """
        Search the internet for information. Use this when the user asks for
        facts, information, or data that might require up-to-date information.
        This tool provides comprehensive results including general information,
        detailed results with links, and recent news articles on the topic.
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
        query: Annotated[str, Field(description="What aspects of the image to analyze - will be routed to GPT-4o for general scenes or Groq for human/sensitive content")]
    ) -> str:
        """
        Capture and analyze visual information from the camera.
        
        This tool:
        1. Captures a current frame from the camera
        2. Intelligently routes analysis to the appropriate model:
           - GPT-4o for general scenes, objects, text reading
           - Groq (Llama model) for human descriptions and sensitive content
        3. Provides a detailed description of what's visible
        
        Use this for any request where visual information is needed.
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
            
            # Start base64 conversion early
            base64_task = None
            if userdata.visual_processor._groq_handler:
                base64_task = asyncio.create_task(
                    userdata.visual_processor._groq_handler._convert_and_optimize_image(image)
                )
            
            # Get Groq handler for model choice
            groq_handler = userdata.visual_processor._groq_handler
            
            # Verify Groq is available
            if not groq_handler or not groq_handler.is_ready:
                logger.error("Groq not available for model choice, falling back to GPT-4o")
                decision_llm = openai.LLM(model="gpt-4o")
                
                # Make the prompt extremely direct
                chat_ctx = ChatContext()
                chat_ctx.add_message(
                    role="system",
                    content="""You are making a simple binary classification:

- If the image contains ANY humans, people, faces, or sensitive content → respond with EXACTLY 'LLAMA'
- If the image ONLY contains objects, landscapes, animals, plants, or text → respond with EXACTLY 'GPT'

DO NOT ADD ANY EXPLANATION. ONLY RESPOND WITH THE SINGLE WORD 'LLAMA' OR 'GPT'."""
                )
                chat_ctx.add_message(
                    role="user",
                    content=[
                        "Does this image contain any humans or sensitive content? If yes, reply only with 'LLAMA'. If no, reply only with 'GPT'.",
                        ImageContent(image=image)
                    ]
                )
                
                # Make a single call to get the model choice
                model_choice = ""
                async with decision_llm.chat(chat_ctx=chat_ctx) as response:
                    async for chunk in response:
                        if chunk and hasattr(chunk.delta, 'content') and chunk.delta.content:
                            model_choice += chunk.delta.content
            else:
                # Use Groq for model choice
                logger.info("Using Groq for model choice decision")
                
                # Convert image to base64 if not already done
                if base64_task:
                    base64_image = await base64_task
                    # Clear the task since we're using it now
                    base64_task = None
                else:
                    base64_image = await groq_handler._convert_and_optimize_image(image)
                
                if not base64_image:
                    logger.error("Failed to convert image to base64 for model choice")
                    model_choice = "GPT"  # Default to GPT if conversion fails
                else:
                    # Make the call to Groq
                    try:
                        completion = await groq_handler.client.chat.completions.create(
                            model=groq_handler.model_id,
                            messages=[
                                {"role": "system", "content": """You are making a simple binary classification:

- If the image contains ANY humans, people, faces, or sensitive content → respond with EXACTLY 'LLAMA'
- If the image ONLY contains objects, landscapes, animals, plants, or text → respond with EXACTLY 'GPT'

DO NOT ADD ANY EXPLANATION. ONLY RESPOND WITH THE SINGLE WORD 'LLAMA' OR 'GPT'."""},
                                {"role": "user", "content": [
                                    {"type": "text", "text": "Does this image contain any humans or sensitive content? If yes, reply only with 'LLAMA'. If no, reply only with 'GPT'."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]}
                            ],
                            max_tokens=10,
                            temperature=0.0,
                            stream=False
                        )
                        
                        # Extract model choice from response
                        model_choice = completion.choices[0].message.content.strip().upper()
                        logger.info(f"Groq model choice: {model_choice}")
                    except Exception as e:
                        logger.error(f"Error getting model choice from Groq: {e}")
                        model_choice = "GPT"  # Default to GPT if Groq fails
            
            # Clean and validate the response
            if model_choice not in ["LLAMA", "GPT"]:
                # Default to GPT for invalid responses
                logger.warning(f"Invalid model choice: {model_choice}, defaulting to GPT")
                model_choice = "GPT"
            
            logger.info(f"Model choice: {model_choice}")
            
            # Store the decision in userdata for llm_node to use
            userdata.visual_processor._model_choice = model_choice
            userdata.visual_processor._last_image = image
            userdata.visual_processor._last_query = query
            
            # If we started base64 conversion but didn't use it, store it for later use
            if base64_task:
                userdata.visual_processor._base64_image = base64_task
            
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
                
                # Create a new chat context for visual analysis
                visual_ctx = ChatContext()
                visual_ctx.add_message(
                    role="system",
                    content="You are Ally, a vision assistant for blind users. Give extremely concise descriptions. Focus directly on answering the user's specific question. Prioritize the most important visual elements relevant to their query. Avoid lengthy descriptions of background or irrelevant details. Be direct and to the point."
                )
                
                # Add the image and query
                visual_ctx.add_message(
                    role="user",
                    content=[
                        f"Answer briefly: {userdata.visual_processor._last_query}",
                        ImageContent(image=userdata.visual_processor._last_image)
                    ]
                )
                
                # Choose model based on the decision
                if userdata.visual_processor._model_choice == "GPT":
                    analysis_llm = openai.LLM(model="gpt-4o")  # Use GPT-4o for general analysis
                    
                    # Stream the analysis
                    async with analysis_llm.chat(chat_ctx=visual_ctx) as stream:
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
                            
                            # Process content based on the current tool
                            processed_content = content
                            
                            # Update the chunk with processed content
                            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                                chunk.delta.content = processed_content
                            else:
                                chunk = processed_content
                            
                            yield chunk
                else:  # LLAMA via Groq
                    # Use Groq for LLAMA analysis
                    try:
                        # Get Groq client from the handler
                        groq_handler = userdata.visual_processor._groq_handler
                        
                        # Verify Groq connection
                        if not groq_handler.is_ready:
                            if not await groq_handler.verify_connection():
                                error_msg = "Groq connection not available. Please check your API key and connection."
                                logger.error(error_msg)
                                yield {"delta": {"content": error_msg}}
                                return
                        
                        # Convert image to base64
                        logger.info("Processing Groq vision analysis...")
                        # Use pre-converted base64 image if available
                        if hasattr(userdata.visual_processor, '_base64_image'):
                            base64_image = await userdata.visual_processor._base64_image
                            # Clear the stored task
                            userdata.visual_processor._base64_image = None
                        else:
                            # Convert image to base64 if not already done
                            base64_image = await groq_handler._convert_and_optimize_image(userdata.visual_processor._last_image)
                        
                        if not base64_image:
                            raise ValueError("Failed to convert image to base64")
                        
                        # Stream the response from Groq
                        completion = await groq_handler.client.chat.completions.create(
                            model=groq_handler.model_id,
                            messages=[
                                {"role": "system", "content": "You are Ally, a vision assistant for blind users. Give extremely concise descriptions. Focus directly on answering the user's specific question. Prioritize the most important visual elements relevant to their query. Avoid lengthy descriptions of background or irrelevant details. Be direct and to the point."},
                                {"role": "user", "content": [
                                    {"type": "text", "text": f"Answer briefly: {userdata.visual_processor._last_query}"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]}
                            ],
                            temperature=groq_handler.temperature,
                            max_tokens=groq_handler.max_tokens,
                            top_p=1,
                            stream=True
                        )
                        
                        # Process the streaming response using the handler
                        async for chunk in groq_handler.stream_response(completion):
                            yield chunk
                        
                        logger.info("Groq vision analysis completed")
                                    
                    except Exception as e:
                        error_msg = f"Error using Groq for analysis: {str(e)}"
                        logger.error(f"Groq error details: {str(e)}")
                        full_response = error_msg
                        yield {"delta": {"content": error_msg}}
                
                # Store the response
                userdata.last_response = full_response
                userdata.visual_processor._model_choice = None  # Reset the choice
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
                    
                    # Process content based on the current tool
                    processed_content = content
                    
                    # Update the chunk with processed content
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                        chunk.delta.content = processed_content
                    else:
                        chunk = processed_content
                    
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