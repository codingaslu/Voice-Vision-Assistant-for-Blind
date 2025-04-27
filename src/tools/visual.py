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
        
        # Initialize Groq handler placeholder - will be set by main.py
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
        
