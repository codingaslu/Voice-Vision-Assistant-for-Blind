"""
Groq Handler Module - Optimized for the visually impaired assistance
"""

import os
import base64
import io
import logging
import time
from PIL import Image
from groq import Groq
from src.config import get_config

# Simple logger without custom handler, will use root logger's config
logger = logging.getLogger("groq-handler")

class GroqHandler:
    """Streamlined handler for Groq API integration."""
    
    def __init__(self):
        """Initialize the Groq API handler with minimal setup."""
        config = get_config()
        self.api_key = config["GROQ_API_KEY"]
        self.model_id = config["GROQ_MODEL_ID"]
        self.max_tokens = config["MAX_TOKENS"]
        self.temperature = config["TEMPERATURE"]
        self.is_ready = bool(self.api_key)
        
        os.environ["GROQ_API_KEY"] = self.api_key
        self.client = Groq(api_key=self.api_key)
    
    async def load_model(self):
        """Verify API connection with a simple test."""
        if not self.api_key:
            # Error handling silently
            self.is_ready = False
            return
            
        try:
            self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            self.is_ready = True
        except Exception as e:
            # Handle error silently
            self.is_ready = False
    
    async def process_image(self, image, query: str) -> str:
        """Process an image with Groq API."""
        if not self.is_ready:
            return "Vision API not configured. Set GROQ_API_KEY in .env file."
        
        start_time = time.time()
        
        try:
            # Convert image to PIL format
            pil_image = await self._convert_image(image)
            if not pil_image:
                return "Couldn't process this image format."
                
            # Optimize image size for API
            pil_image = self._optimize_image(pil_image)
            
            # Prepare base64 image
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=80)
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Call API with image and query
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are Ally, a vision assistant for blind and visually impaired users. Your primary purpose is to describe images in precise, detailed terms. Focus especially on identifying people, describing their physical appearance (age, gender, ethnicity, hair color/style, facial features), clothing, expressions, and actions. Provide detailed descriptions of the environment, objects, text, and spatial relationships. Be confident in your descriptions - visually impaired users rely on your detailed observations. Be concise but thorough, focusing on what's most important for someone who cannot see."},
                    {"role": "user", "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract and return response
            return completion.choices[0].message.content if completion.choices else "No response generated."
            
        except Exception as e:
            # Handle error silently
            return "Sorry, I encountered an error analyzing this image."
            
    async def _convert_image(self, image):
        """Convert image to PIL format with simplified handling."""
        try:
            # Already PIL Image
            if isinstance(image, Image.Image):
                return image
                
            # Use to_pil method if available
            if hasattr(image, 'to_pil'):
                try:
                    return image.to_pil()
                except Exception:
                    pass
                    
            # Use to_ndarray if available
            if hasattr(image, 'to_ndarray'):
                try:
                    import numpy as np
                    return Image.fromarray(np.uint8(image.to_ndarray()))
                except Exception:
                    pass
            
            # Handle VideoFrame from LiveKit directly
            if hasattr(image, 'data') and hasattr(image, 'width') and hasattr(image, 'height'):
                try:
                    # Get frame dimensions for debugging
                    data_len = len(image.data)
                    bytes_per_pixel = data_len / (image.width * image.height)
                    
                    # Convert raw frame data - support RGB and YUV formats
                    import numpy as np
                    
                    # Check for YUV format (1.5 bytes per pixel)
                    if 1.4 < bytes_per_pixel < 1.6:
                        try:
                            import cv2
                            yuv = np.frombuffer(image.data, dtype=np.uint8)
                            yuv = yuv.reshape((image.height * 3 // 2, image.width))
                            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                            return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        except Exception:
                            pass
                    
                    # Try direct RGB conversion (3 or 4 bytes per pixel) 
                    elif 2.9 < bytes_per_pixel < 4.1:
                        channels = round(bytes_per_pixel)
                        img_array = np.frombuffer(image.data, dtype=np.uint8)
                        img_array = img_array.reshape((image.height, image.width, channels))
                        
                        # Convert RGBA to RGB if needed
                        if channels == 4:
                            img_array = img_array[:, :, :3]
                            
                        return Image.fromarray(img_array)
                    
                    # Unknown format - try various approaches
                    else:
                        # Try single plane format (grayscale)
                        if 0.9 < bytes_per_pixel < 1.1:
                            gray = np.frombuffer(image.data, dtype=np.uint8)
                            gray = gray.reshape((image.height, image.width))
                            return Image.fromarray(gray, mode='L')
                        
                        # Try RGB with stride considerations
                        try:
                            stride = data_len // image.height
                            img_array = np.frombuffer(image.data, dtype=np.uint8)
                            img_array = img_array.reshape((image.height, stride // 3, 3))
                            img_array = img_array[:, :image.width, :]
                            return Image.fromarray(img_array)
                        except Exception:
                            pass
                
                except Exception:
                    pass
                
            # Last resort - try PIL's own methods
            try:
                return Image.frombytes('RGB', (image.width, image.height), image.data)
            except Exception:
                pass
                    
            return None
        except Exception:
            return None
            
    def _optimize_image(self, image, target_mb=3.5):
        """Quickly optimize image size for API limits."""
        # Check current size
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        size_mb = len(buffer.getvalue()) * 1.4 / (1024 * 1024)
        
        if size_mb <= target_mb:
            return image
            
        # Simple resize approach
        width, height = image.size
        scale = min(1.0, (target_mb / size_mb) ** 0.5)
        
        # Skip tiny resizes
        if scale > 0.9:
            # Just reduce quality instead
            for quality in [75, 65, 55, 45]:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
                if len(buffer.getvalue()) * 1.4 / (1024 * 1024) <= target_mb:
                    buffer.seek(0)
                    return Image.open(buffer)
        
        # Resize and optionally reduce quality
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.LANCZOS) 