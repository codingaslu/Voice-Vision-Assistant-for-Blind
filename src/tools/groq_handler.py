"""
Groq Handler Module - Optimized for the visually impaired assistance
"""

import os
import base64
import io
import logging
import time
import json
from PIL import Image
from groq import AsyncGroq
from ..config import get_config

# Simple logger without custom handler, will use root logger's config
logger = logging.getLogger("groq-handler")

# Constants
SYSTEM_PROMPT = """Binary classifier for images. Respond in JSON format:

IF image contains humans/person/faces → "LLAMA" + answer query like you see the visual.
IF NO humans/faces → "GPT" + empty string

JSON format:
{
  "model_choice": "GPT" or "LLAMA",
  "analysis": "" if GPT, answer if LLAMA
}"""

class GroqHandler:
    """Streamlined handler for Groq API integration."""
    
    def __init__(self):
        """Initialize the Groq API handler with minimal setup."""
        try:
            config = get_config()
            self.api_key = config["GROQ_API_KEY"]
            self.model_id = config["GROQ_MODEL_ID"]
            self.max_tokens = config["MAX_TOKENS"]
            self.temperature = config["TEMPERATURE"]
            
            logger.info(f"Initializing Groq handler with model: {self.model_id}")
            
            # Quick validation and setup
            if not self.api_key:
                logger.error("No GROQ_API_KEY provided in configuration")
                self.is_ready = False
                self._verified = False
                return
                
            # Set up Groq client
            os.environ["GROQ_API_KEY"] = self.api_key
            self.client = AsyncGroq(api_key=self.api_key)
            self.is_ready = True
            self._verified = False
            self._image_cache = {}
            logger.info("Groq client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error during Groq handler initialization: {e}")
            self.is_ready = False
            self._verified = False
    
    async def verify_connection(self):
        """Verify the connection is working."""
        if self._verified:
            return True
            
        if not self.api_key:
            logger.error("No API key provided for Groq")
            self.is_ready = False
            return False
            
        # Just check API key presence, no need for a test call
        self.is_ready = True
        self._verified = True
        logger.info(f"Groq handler verified with model {self.model_id}")
        return True
    
    async def model_choice_with_analysis(self, image, query: str):
        """
        Make a model choice (LLAMA vs GPT) and get analysis in a single call.
        
        Args:
            image: Image to analyze
            query: User query about the image
            
        Returns:
            Tuple of (model_choice, analysis, error)
        """
        # Quick validation
        if not self.is_ready:
            return "GPT", "", "Vision API not configured"
            
        # Verify connection on first use
        if not self._verified and not await self.verify_connection():
            return "GPT", "", "Vision API connection failed"
        
        try:
            # Convert image to base64
            base64_image = await self._convert_and_optimize_image(image)
            if not base64_image:
                return "GPT", "", "Failed to process the image"
            
            # Make the call to Groq
            completion = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Answer this query about the seeing the visual in front of the user.please dont mention the image or user word in your answer: {query}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=self.max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
                stream=False
            )
            
            # Parse response
            response_json = completion.choices[0].message.content
            response_data = json.loads(response_json)
            
            # Extract data
            model_choice = response_data.get("model_choice", "GPT").upper()
            groq_analysis = response_data.get("analysis", "") if model_choice == "LLAMA" else ""
            
            # Validate
            if model_choice not in ["LLAMA", "GPT"]:
                logger.warning(f"Invalid model choice: {model_choice}, defaulting to GPT")
                model_choice = "GPT"
            
            logger.info(f"Model choice: {model_choice}, analysis available: {bool(groq_analysis)}")
            return model_choice, groq_analysis, None
            
        except Exception as e:
            logger.error(f"Error in model_choice_with_analysis: {e}")
            return "GPT", "", str(e)
            
    async def _convert_and_optimize_image(self, image, target_mb=3.5):
        """Convert image to base64 string with size optimization."""
        try:
            # Try to use cached image
            if hasattr(image, 'tobytes'):
                try:
                    image_hash = hash(image.tobytes())
                    if image_hash in self._image_cache:
                        return self._image_cache[image_hash]
                except Exception:
                    pass  # Continue if hashing fails

            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                if hasattr(image, 'to_pil'):
                    image = image.to_pil()
                elif hasattr(image, 'to_ndarray'):
                    import numpy as np
                    arr = image.to_ndarray()
                    if arr.dtype != np.uint8:
                        arr = np.uint8(arr)
                    image = Image.fromarray(arr)
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
                            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
                            image = Image.fromarray(rgb)
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
            
            # Check current size and optimize if needed
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            size_mb = len(buffer.getvalue()) * 1.4 / (1024 * 1024)
            
            # If small enough, use as is
            if size_mb <= target_mb:
                encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Cache result
                if hasattr(image, 'tobytes'):
                    try:
                        self._image_cache[hash(image.tobytes())] = encoded
                    except Exception:
                        pass
                
                return encoded
            
            # Need optimization
            if size_mb <= target_mb * 1.2:
                # Just reduce quality
                min_q, max_q = 45, 85
                while min_q < max_q - 5:
                    mid_q = (min_q + max_q) // 2
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=mid_q)
                    if len(buffer.getvalue()) * 1.4 / (1024 * 1024) <= target_mb:
                        min_q = mid_q
                    else:
                        max_q = mid_q
                
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=min_q)
            else:
                # Resize the image
                scale = (target_mb / size_mb) ** 0.5
                new_size = tuple(int(dim * scale) for dim in image.size)
                image = image.resize(new_size, Image.BICUBIC)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=85)
            
            # Encode and cache
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            if hasattr(image, 'tobytes'):
                try:
                    self._image_cache[hash(image.tobytes())] = encoded
                except Exception:
                    pass
                
            return encoded
            
        except Exception as e:
            logger.error(f"Error in _convert_and_optimize_image: {e}")
            return None 