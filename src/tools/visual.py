import asyncio
import logging
import os
import time
from typing import Optional, List, Tuple, Dict, Any

from livekit import rtc
from livekit.agents.llm.chat_context import ChatContext, ImageContent
from livekit.plugins import openai
from PIL import Image

# Simple logger without custom handler, will use root logger's config
logger = logging.getLogger("visual-processor")

class VisualProcessor:
    """
    A class that handles capturing and processing frames from a video track.
    This is used to provide vision capabilities to the agent.
    """
    
    def __init__(self):
        self.latest_frame: Optional[Image.Image] = None
        self._frames_buffer: List[Image.Image] = []
        self._buffer_size = 5
        self._last_query: str = ""
        self._last_response: str = ""
        self._cached_video_track: Optional[rtc.RemoteVideoTrack] = None
        
        # Initialize Groq handler at startup
        try:
            from src.tools.groq_handler import GroqHandler
            self._groq_handler = GroqHandler()
            logger.info("Initialized Groq handler at startup")
        except Exception as e:
            logger.error(f"Failed to initialize Groq handler: {e}")
            self._groq_handler = None
    
    async def enable_camera(self, room: rtc.Room) -> None:
        """Send a signal to enable the camera for the remote participant."""
        logger.info("Enabling camera...")
        try:
            await room.local_participant.publish_data(
                "camera_enable", reliable=True, topic="camera"
            )
            logger.info("Camera enable signal sent")
        except Exception as e:
            logger.error(f"Error enabling camera: {e}")
            raise
    
    async def get_video_track(self, room: rtc.Room, timeout: float = 10.0) -> rtc.RemoteVideoTrack:
        """
        Sets up video track handling using LiveKit's subscription model.
        Returns the first available video track or raises TimeoutError.
        """
        # Return cached track if available
        if self._cached_video_track is not None:
            return self._cached_video_track
            
        logger.info("Waiting for video track...")
        video_track_future = asyncio.Future[rtc.RemoteVideoTrack]()
        
        # First check existing tracks in case we missed the subscription event
        for participant in room.remote_participants.values():
            logger.info(f"Checking participant: {participant.identity}")
            for pub in participant.track_publications.values():
                if (pub.track and 
                    pub.track.kind == rtc.TrackKind.KIND_VIDEO and 
                    isinstance(pub.track, rtc.RemoteVideoTrack)):
                    
                    logger.info(f"Found existing video track: {pub.track.sid}")
                    self._cached_video_track = pub.track
                    video_track_future.set_result(pub.track)
                    return self._cached_video_track

        # Set up listener for future video tracks
        @room.on("track_subscribed") 
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if (not video_track_future.done() and 
                track.kind == rtc.TrackKind.KIND_VIDEO and 
                isinstance(track, rtc.RemoteVideoTrack)):
                
                logger.info(f"Subscribed to video track: {track.sid}")
                self._cached_video_track = track
                video_track_future.set_result(track)

        # Add timeout in case no video track arrives
        try:
            track = await asyncio.wait_for(video_track_future, timeout=timeout)
            self._cached_video_track = track
            return track
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for video track after {timeout} seconds")
            raise TimeoutError(f"No video track received within {timeout} seconds")
    
    async def capture_frame(self, room: rtc.Room) -> Optional[Image.Image]:
        """Capture a single best frame from the video track."""
        logger.info("Capturing frame...")
        try:
            # Get the video track
            video_track = await self.get_video_track(room)
            
            # Clear the buffer
            self._frames_buffer = []
            
            # Capture frames until buffer is filled
            async for event in rtc.VideoStream(video_track):
                frame = event.frame
                self._frames_buffer.append(frame)
                self.latest_frame = frame  # Update the latest frame
                
                # Once we have enough frames, select the best one
                if len(self._frames_buffer) >= self._buffer_size:
                    best_frame = await self._select_best_frame()
                    return best_frame
            
            # If we exit the loop without enough frames but have at least one
            if self._frames_buffer:
                return self._frames_buffer[-1]
            
            return None
        
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    async def _select_best_frame(self) -> Image.Image:
        """Select the best frame from the buffer."""
        return self._frames_buffer[-1]
    
    def get_latest_frame(self) -> Optional[Image.Image]:
        """Get the most recently captured frame."""
        return self.latest_frame
        
    async def analyze_image(self, image: Image.Image, query: str) -> str:
        """
        Analyze an image using OpenAI's GPT-4o model.
        
        Args:
            image: The image to analyze
            query: The user's query about the image
            
        Returns:
            The analysis result as a string
        """
        # Store the query for potential fallback
        self._last_query = query
        
        # Clean query if needed
        if not query or query.strip() == "":
            query = "What can you see in this image? Describe everything visible in detail."
        
        logger.info(f"Analyzing image with query: {query}")
        
        try:
            # Create a new chat context for the LLM query
            chat_ctx = ChatContext()
            
            # Add a system message to provide context for the analysis
            chat_ctx.add_message(
                role="system",
                content="You are Ally, a vision assistant for blind and visually impaired users. "
                        "Describe what you see in great detail including colors, objects, people, text, and the overall scene. "
                        "Be concise but thorough. Focus especially on elements that would be important for someone who cannot see."
            )
            
            # Add the image as a user message
            chat_ctx.add_message(
                role="user",
                content=[
                    f"Please analyze this image and tell me: {query}",
                    ImageContent(image=image)
                ]
            )
            
            # Create a separate LLM instance
            direct_llm = openai.LLM(model="gpt-4o")
            
            # Get the response
            start_time = time.time()
            response_text = ""
            async with direct_llm.chat(chat_ctx=chat_ctx) as stream:
                async for chunk in stream:
                    if chunk and hasattr(chunk.delta, 'content') and chunk.delta.content:
                        response_text += chunk.delta.content
            
            elapsed = time.time() - start_time
            logger.info(f"Analysis complete in {elapsed:.1f}s")
            
            # Store the response for potential fallback
            self._last_response = response_text
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return f"Error analyzing image: {str(e)}"
    
    async def groq_fallback(self, original_response: str = "") -> str:
        """
        Use Groq as a fallback to better analyze the latest image.
        
        Args:
            original_response: The original analysis from OpenAI
            
        Returns:
            Enhanced analysis from Groq
        """
        logger.info("Using Groq fallback")
        
        # Check if we have an image to process
        if self.latest_frame is None:
            logger.error("No image available for Groq fallback")
            return "No image available to analyze with Groq."
        
        # Get the query to use
        query = self._last_query
        if not query or query.lower().strip() in ["see", "look", "view", "camera"]:
            query = "What can you see in this image? Describe everything visible in detail, including any people, their clothing, and the surrounding environment."
        
        try:
            # Use already initialized Groq handler
            if self._groq_handler is None:
                # Try to initialize if not already done
                from src.tools.groq_handler import GroqHandler
                self._groq_handler = GroqHandler()
                logger.info("Initialized Groq handler on demand")
            
            # Process image with Groq
            groq_response = await self._groq_handler.process_image(self.latest_frame, query)
            
            return groq_response
            
        except Exception as e:
            logger.error(f"Error with Groq API fallback: {e}")
            if original_response:
                return f"Couldn't get enhanced analysis from Groq. Original analysis: {original_response}"
            else:
                return f"Error using Groq for analysis: {str(e)}"
    
    async def capture_and_analyze(self, room: rtc.Room, query: str) -> Tuple[str, bool]:
        """
        Capture a frame and analyze it in one step.
        
        Args:
            room: The LiveKit room
            query: The user's query
            
        Returns:
            Tuple of (analysis_text, used_fallback)
        """
        logger.info(f"Processing visual query: {query[:30]}...")
        
        try:
            # Capture frame
            image = await self.capture_frame(room)
            if image is None:
                return "Failed to capture an image from the camera.", False
            
            # Analyze with OpenAI
            openai_response = await self.analyze_image(image, query)
            
            # Check for refusal indicators that might require Groq
            refusal_indicators = [
                "cannot identify", "can't identify", "cannot recognize", 
                "can't recognize", "cannot describe people", "don't identify",
                "unable to identify", "cannot provide details about people",
                "privacy reasons", "cannot determine who", "unable to tell who",
                "can't specifically identify", "cannot specifically identify", 
                "unable to specifically identify", "apologize", "oversight",
                "unable to identify", "unable to tell", "unable to describe",
                "unable to recognize", "unable to provide details about",
                "unable to determine who", "unable to tell who", "unable to describe people",
                "unable to recognize people", "unable to provide details about people"
            ]
            
            # Determine if Groq fallback is needed
            need_fallback = any(indicator in openai_response.lower() for indicator in refusal_indicators)
            
            # Use Groq fallback if necessary
            if need_fallback:
                logger.info("Using Groq fallback due to limitation detection")
                groq_response = await self.groq_fallback(openai_response)
                return groq_response, True
            else:
                return openai_response, False
            
        except Exception as e:
            logger.error(f"Error in capture_and_analyze: {e}")
            return f"Error capturing and analyzing image: {str(e)}", False 