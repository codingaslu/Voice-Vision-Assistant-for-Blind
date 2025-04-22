"""
Groq Handler Module - Optimized for the visually impaired assistance
"""

import os
import base64
import io
import logging
import time
import asyncio
from PIL import Image
from groq import AsyncGroq
from src.config import get_config
from livekit.agents.llm.llm import ChatChunk, ChoiceDelta, CompletionUsage

# Simple logger without custom handler, will use root logger's config
logger = logging.getLogger("groq-handler")

class GroqHandler:
    """Streamlined handler for Groq API integration."""
    
    def __init__(self):
        """Initialize the Groq API handler with minimal setup."""
        config = get_config()
        self.api_key = config["GROQ_API_KEY"]
        self.model_id = "meta-llama/llama-4-scout-17b-16e-instruct"  # Set default model
        self.max_tokens = config.get("MAX_TOKENS", 1024)
        self.temperature = config.get("TEMPERATURE", 0.7)
        
        # Set up Groq client
        os.environ["GROQ_API_KEY"] = self.api_key
        self.client = AsyncGroq(api_key=self.api_key)
        self.is_ready = True  # Will be verified on first use
        self._verified = False
    
    async def verify_connection(self):
        """Verify the connection is working."""
        if self._verified:
            return True
            
        if not self.api_key:
            logger.error("No API key provided for Groq")
            self.is_ready = False
            return False
            
        try:
            # Test with a simple completion
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
                stream=False
            )
            self.is_ready = True
            self._verified = True
            logger.info("Groq handler verified successfully")
            return True
        except Exception as e:
            logger.error(f"Groq connection verification failed: {e}")
            self.is_ready = False
            return False
    
    async def process_image(self, image, query: str) -> str:
        """Process an image with Groq API."""
        # Verify connection on first use
        if not self._verified:
            if not await self.verify_connection():
                return "Vision API not configured or connection failed. Please check your API key."
        
        try:
            # Convert image to base64
            base64_image = await self._convert_and_optimize_image(image)
            if not base64_image:
                return "Couldn't process this image format."
            
            # Call API with image and query
            completion = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are Ally, a vision assistant for blind and visually impaired users. Your primary purpose is to describe images in precise, detailed terms. Focus especially on identifying people, describing their physical appearance (age, gender, ethnicity, hair color/style, facial features), clothing, expressions, and actions. Provide detailed descriptions of the environment, objects, text, and spatial relationships. Be confident in your descriptions - visually impaired users rely on your detailed observations. Be concise but thorough, focusing on what's most important for someone who cannot see."},
                    {"role": "user", "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            return completion
            
        except Exception as e:
            logger.error(f"Error in process_image: {e}")
            return None

    async def stream_response(self, completion):
        """Process streaming response from Groq and yield properly formatted ChatChunks."""
        if completion is None:
            yield ChatChunk(
                id=f"groqcmpl-{time.time()}",
                delta=ChoiceDelta(
                    role="assistant",
                    content="Error: No response from Groq API",
                    tool_calls=[]
                ),
                usage=None
            )
            return

        got_any_content = False
        try:
            async for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    got_any_content = True
                    
                    # Create a ChatChunk that matches GPT's format
                    yield ChatChunk(
                        id=f"groqcmpl-{time.time()}",
                        delta=ChoiceDelta(
                            role="assistant",
                            content=content,
                            tool_calls=[]
                        ),
                        usage=None
                    )
            
            if not got_any_content:
                error_msg = "No content received from Groq stream"
                logger.error(error_msg)
                yield ChatChunk(
                    id=f"groqcmpl-{time.time()}",
                    delta=ChoiceDelta(
                        role="assistant",
                        content=error_msg,
                        tool_calls=[]
                    ),
                    usage=None
                )
            
            # Final chunk with usage stats
            yield ChatChunk(
                id=f"groqcmpl-{time.time()}",
                delta=None,
                usage=CompletionUsage(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                    cache_creation_tokens=0,
                    cache_read_tokens=0
                )
            )
            
        except Exception as stream_error:
            logger.error(f"Error during stream processing: {stream_error}")
            yield ChatChunk(
                id=f"groqcmpl-{time.time()}",
                delta=ChoiceDelta(
                    role="assistant",
                    content=f"Error during stream processing: {str(stream_error)}",
                    tool_calls=[]
                ),
                usage=None
            )
            
    async def _convert_and_optimize_image(self, image, target_mb=3.5):
        """Convert image to base64 string with size optimization."""
        try:
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                if hasattr(image, 'to_pil'):
                    image = image.to_pil()
                elif hasattr(image, 'to_ndarray'):
                    import numpy as np
                    image = Image.fromarray(np.uint8(image.to_ndarray()))
                elif hasattr(image, 'data') and hasattr(image, 'width') and hasattr(image, 'height'):
                    # Handle VideoFrame from LiveKit
                    try:
                        import numpy as np
                        import cv2
                        
                        data_len = len(image.data)
                        bytes_per_pixel = data_len / (image.width * image.height)
                        
                        if 1.4 < bytes_per_pixel < 1.6:  # YUV format
                            yuv = np.frombuffer(image.data, dtype=np.uint8)
                            yuv = yuv.reshape((image.height * 3 // 2, image.width))
                            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        elif 2.9 < bytes_per_pixel < 4.1:  # RGB/RGBA format
                            channels = round(bytes_per_pixel)
                            img_array = np.frombuffer(image.data, dtype=np.uint8)
                            img_array = img_array.reshape((image.height, image.width, channels))
                            if channels == 4:
                                img_array = img_array[:, :, :3]
                            image = Image.fromarray(img_array)
                        else:
                            return None
                    except Exception as e:
                        logger.error(f"Error converting VideoFrame: {e}")
                        return None
                else:
                    return None
            
            # Optimize image size if needed
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            size_mb = len(buffer.getvalue()) * 1.4 / (1024 * 1024)
            
            if size_mb > target_mb:
                # Try reducing quality first
                if size_mb <= target_mb * 1.2:  # If close to target, just reduce quality
                    for quality in [75, 65, 55, 45]:
                        buffer = io.BytesIO()
                        image.save(buffer, format="JPEG", quality=quality)
                        if len(buffer.getvalue()) * 1.4 / (1024 * 1024) <= target_mb:
                            buffer.seek(0)
                            image = Image.open(buffer)
                            break
                else:  # Need to resize
                    scale = (target_mb / size_mb) ** 0.5
                    new_size = tuple(int(dim * scale) for dim in image.size)
                    image = image.resize(new_size, Image.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error in _convert_and_optimize_image: {e}")
            return None 