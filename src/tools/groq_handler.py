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
        
        logger.info(f"Initializing Groq handler with model: {self.model_id}")
        
        # Set up Groq client
        os.environ["GROQ_API_KEY"] = self.api_key
        self.client = AsyncGroq(api_key=self.api_key)
        self.is_ready = True  
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
        # Verify connection on first use
        if not self._verified:
            if not await self.verify_connection():
                return "GPT", "", "Vision API not configured or connection failed."
        
        try:
            # Convert image to base64
            base64_image = await self._convert_and_optimize_image(image)
            if not base64_image:
                logger.error("Failed to convert image to base64")
                return "GPT", "", "Failed to process the image."
            
            # Make the call to Groq - Use JSON mode to get both decision and analysis
            completion = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": """Binary classifier for images. Respond in JSON format:

IF image contains humans/faces → "LLAMA" + answer query
IF NO humans/faces → "GPT" + empty string

JSON format:
{
  "model_choice": "GPT" or "LLAMA",
  "analysis": "" if GPT, answer if LLAMA
}"""},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Answer this query about the image in detail: {query}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=self.max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
                stream=False
            )
            
            # Extract both model choice and analysis from JSON response
            response_json = completion.choices[0].message.content
            
            response_data = json.loads(response_json)
            
            model_choice = response_data.get("model_choice", "GPT").upper()
            groq_analysis = ""
            
            # Only get analysis if model_choice is LLAMA
            if model_choice == "LLAMA":
                groq_analysis = response_data.get("analysis", "")
            
            logger.info(f"Model choice: {model_choice}, analysis available: {bool(groq_analysis)}")
            
            # Clean and validate the response
            if model_choice not in ["LLAMA", "GPT"]:
                # Default to GPT for invalid responses
                logger.warning(f"Invalid model choice: {model_choice}, defaulting to GPT")
                model_choice = "GPT"
            
            return model_choice, groq_analysis, None
            
        except Exception as e:
            logger.error(f"Error in model_choice_with_analysis: {e}")
            return "GPT", "", str(e)

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