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

# Logger
logger = logging.getLogger("ally-vision-agent")

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
            # Validate room connection
            if not userdata.room_ctx or not userdata.room_ctx.room:
                return "I couldn't access the camera because the room connection is not available."
            
            # Capture image from camera
            image = await userdata.visual_processor.capture_frame(userdata.room_ctx.room)
            if image is None:
                return "I couldn't capture a clear image from the camera."
            
            # Reset state
            userdata._gpt_chunks.clear()
            userdata._add_chunk_callback = None
            userdata._analysis_complete = False
            
            # Set up visual context
            visual_ctx = ChatContext()
            visual_ctx.add_message(
                role="system",
                content="You are Ally, a vision assistant for blind users. Provide extremely concise and clear descriptions. Focus only on the most important elements needed to answer the user's specific question. Ignore changes in color, brightness, or shading caused by sunglasses. Be direct and to the point. Answer as if the scene is fully visible without distortion. Avoid phrases like \"as seen\" or \"looks like,\" and do not describe things based only on visual appearance. When helpful, explain how a screen reader might announce elements. Tailor your response to what a blind user would truly want to know."
            )
            
            visual_ctx.add_message(
                role="user",
                content=[
                    f"Answer this query about the seeing the visual in front of the user.please dont mention the image or user word in your answer: {query}",
                    ImageContent(image=image)
                ]
            )
            
            # Initialize LLM
            analysis_llm = openai.LLM(model="gpt-4o")
            
            # Always start GPT analysis
            asyncio.create_task(self._run_gpt_analysis(userdata, analysis_llm, visual_ctx))
            
            # Try Groq if available
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
                chunk_queue = asyncio.Queue()
                done_event = asyncio.Event()
                
                # Create callback for receiving chunks
                userdata._add_chunk_callback = lambda content: chunk_queue.put_nowait(content)
                
                # Add existing chunks
                for chunk in userdata._gpt_chunks:
                    chunk_queue.put_nowait(chunk)
                
                # Process chunks
                try:
                    while not (done_event.is_set() and chunk_queue.empty()):
                        try:
                            chunk = await asyncio.wait_for(chunk_queue.get(), timeout=0.1)
                            full_response += chunk
                            yield ChatChunk(
                                id=f"gptcmpl-{time.time()}",
                                delta=ChoiceDelta(role="assistant", content=chunk),
                                usage=None
                            )
                        except asyncio.TimeoutError:
                            if userdata._analysis_complete and chunk_queue.empty():
                                done_event.set()
                finally:
                    userdata._add_chunk_callback = None
                    userdata._gpt_chunks.clear()
                    done_event.set()
            
            # LLAMA/Groq single response
            elif userdata._groq_analysis:
                full_response = userdata._groq_analysis
                yield ChatChunk(
                    id=f"groqcmpl-{time.time()}",
                    delta=ChoiceDelta(role="assistant", content=full_response),
                    usage=None
                )
                userdata._groq_analysis = None
            
            # No analysis case
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
                    if chunk and hasattr(chunk.delta, 'content'):
                        content = chunk.delta.content
                        if content:
                            full_response += content
                    yield chunk
        
        userdata.last_response = full_response
    
    async def llm_node(self, chat_ctx, tools, model_settings=None):
        """Override llm_node to modify the output before sending to TTS"""
        userdata = self.session.userdata
        return self._process_stream(chat_ctx, tools, userdata)

async def entrypoint(ctx: JobContext):
    """Set up and start the voice agent with all required tools"""
    try:
        # Connect to room
        await ctx.connect()
        
        # Initialize user data and tools
        userdata = UserData()
        userdata.room_ctx = ctx
        userdata.visual_processor = VisualProcessor()
        userdata.internet_search = InternetSearch()
        
        # Try to initialize Groq (optional)
        try:
            userdata.groq_handler = GroqHandler()
        except Exception as e:
            logger.warning(f"Vision will use GPT only: {e}")
        
        # Initialize camera
        try:
            await userdata.visual_processor.enable_camera(ctx.room)
        except Exception as e:
            logger.warning(f"Camera setup failed: {e}")
        
        # Start agent session
        agent_session = AgentSession[UserData](
            userdata=userdata,
            stt=deepgram.STT(model="nova-2-general", smart_format=True, punctuate=True, language="en-US"),
            llm=openai.LLM(model="gpt-4o", parallel_tool_calls=False),
            tts=elevenlabs.TTS(model="eleven_multilingual_v2"),
            vad=silero.VAD.load(),
            max_tool_steps=3,
        )
        
        await agent_session.start(
            agent=AllyVisionAgent(),
            room=ctx.room,
            room_input_options=RoomInputOptions(),
        )
        
    except Exception as e:
        logger.error(f"Agent startup failed: {e}")