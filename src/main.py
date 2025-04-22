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
            Your name is Ally. You are an assistant for the blind and visually impaired. Your interface with users will be voice and vision.
            Respond with short and concise answers. Avoid using unpronounceable punctuation or emojis.
            You can help with general questions, search the internet for information, or describe what's around the user.
            
            IMPORTANT INSTRUCTIONS FOR VISUAL QUERIES:
            1. When users ask ANY question related to visual content like 'what do you see', 'can you see', 'describe what's in front of me', 
            'what is infront of me', 'what color is this shirt', IMMEDIATELY use the analyze_vision tool.
            2. The analyze_vision tool captures a frame from the camera and intelligently routes the analysis:
               - Uses GPT-4o for general scenes, objects, text reading
               - Uses Groq (streaming) for human descriptions and sensitive content
            3. Be very proactive about using the analyze_vision tool - it's your primary purpose as an assistant for visually impaired users.
            4. For any request where you need to SEE something to answer it, use the analyze_vision tool without hesitation.
            
            IMPORTANT INSTRUCTIONS FOR INTERNET SEARCHES:
            1. When users ask for information about news, facts, data, or anything that might require up-to-date information, use the search_internet tool.
            2. For news-specific queries, use the search_internet tool which includes recent news articles.
            3. When providing information from internet searches, mention that the information comes from the web.
            
            IMPORTANT INSTRUCTIONS FOR GENERAL QUESTIONS:
            1. For general questions that don't require visual analysis or internet search, respond directly using your knowledge.
            2. Keep responses concise, informative, and helpful, focusing on providing the exact information requested.
            3. If you're unsure whether a question requires visual input or internet search, err on the side of using those tools.
            4. For common knowledge questions (like "What's the capital of France?"), no need to use search_internet unless the information might have changed recently.
            
            IMPORTANT INSTRUCTIONS FOR CONVERSATION:
            1. In conversation, avoid discussing tools, technical details, or internal architecture.
            2. Focus on providing helpful and concise information to the user.
            3. Ensure your responses are clear, easy to understand, and free of technical jargon.
            4. Avoid asking more and more clarification questions and troubling the user.
            5. Avoid mentioning the use of tools in your responses.
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
            
            # Use GPT-4 for better accuracy in detecting people
            decision_llm = openai.LLM(model="gpt-4o")
            
            # Make the prompt extremely direct
            chat_ctx = ChatContext()
            chat_ctx.add_message(
                role="system",
                content="""Answer this ONE question about the image by analyzing it carefully:

ARE THERE ANY OF THE FOLLOWING IN THE IMAGE?
1. Humans or people (even partially visible)
2. Content that might require detailed descriptions of people (faces, clothing, etc.)
3. Potentially sensitive content (medical, personal, private scenes)
4. Content that traditional vision models might be hesitant to describe in detail
5. Intimate settings or scenarios requiring nuanced description

- If YES to ANY of the above → Type 'LLAMA'
- If NO to ALL of the above → Type 'GPT'

DO NOT EXPLAIN. DO NOT ADD CONTEXT.
RESPOND WITH ONLY ONE WORD: 'LLAMA' or 'GPT'"""
            )
            chat_ctx.add_message(
                role="user",
                content=[
                    "Analyze this image carefully. Does it contain people, sensitive content, or anything requiring detailed human description? If YES type LLAMA, if NO type GPT.",
                    ImageContent(image=image)
                ]
            )
            
            # Make a single call to get the model choice
            model_choice = ""
            async with decision_llm.chat(chat_ctx=chat_ctx) as response:
                async for chunk in response:
                    if chunk and hasattr(chunk.delta, 'content') and chunk.delta.content:
                        model_choice += chunk.delta.content
                        
            # Clean and validate the response
            model_choice = model_choice.strip().upper()
            
            logger.info(f"Model choice: {model_choice}")
            
            # Store the decision in userdata for llm_node to use
            userdata.visual_processor._model_choice = model_choice
            userdata.visual_processor._last_image = image
            userdata.visual_processor._last_query = query
            
            # If we started base64 conversion, store it for later use
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
                    content="You are Ally, a vision assistant for blind and visually impaired users. "
                           "Describe what you see in great detail including colors, objects, people, text, and the overall scene. "
                           "Be concise but thorough. Focus especially on elements that would be important for someone who cannot see."
                )
                
                # Add the image and query
                visual_ctx.add_message(
                    role="user",
                    content=[
                        f"Please analyze this image and tell me: {userdata.visual_processor._last_query}",
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
                                {"role": "system", "content": "You are Ally, a vision assistant for blind and visually impaired users. Your primary purpose is to describe images in precise, detailed terms. Focus especially on identifying people, describing their physical appearance, clothing, expressions, and actions. Be confident in your descriptions."},
                                {"role": "user", "content": [
                                    {"type": "text", "text": userdata.visual_processor._last_query},
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