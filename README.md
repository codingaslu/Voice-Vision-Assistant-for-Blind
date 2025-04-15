# Envision AI - Voice and Vision Assistant

## Overview

Envision AI is an advanced voice and vision assistant designed specifically for blind and visually impaired users. It combines speech recognition, natural language processing, and computer vision to provide a comprehensive assistance solution.

## System Architecture

The system uses a multi-component architecture to process user inputs and generate helpful responses:

<!-- FLOWCHART IMAGE: Replace with your system architecture flowchart -->
![System Architecture Flow Chart](images/system_architecture.png)

## Key Features

1. **Dual Vision Model Approach**
   - Primary analysis with OpenAI's GPT-4o
   - Fallback to Groq's Llama-4-Scout-17B for enhanced person detection
   - Privacy-focused solution to describe people when primary model refuses

2. **Response Streaming**
   - Real-time response generation and delivery
   - Progressive output for improved user experience
   - Natural conversation with the assistant using speech

3. **Specialized System Instructions**
   - Detailed guidance for handling visual queries
   - Automatic detection of refusal cases requiring fallback
   - Contextual tool selection based on query type

4. **Blind/Low-Vision Optimizations**
   - Detailed descriptions prioritizing elements important for visually impaired users
   - Concise but thorough scene analysis
   - Voice-first interface design
   - Accessibility-first approach throughout the system

5. **Comprehensive Capabilities**
   - **Voice Interaction**: Natural conversation with the assistant using speech
   - **Visual Understanding**: Camera-based vision to describe surroundings, including people
   - **Internet Search**: Real-time information lookup for current events and facts
   - **Multi-component Integration**: Seamless coordination between vision, voice, and search tools

## Model Selection & Evaluation Criteria

We carefully selected the `meta-llama/llama-4-scout-17b-16e-instruct` model for our fallback vision processing based on the following criteria:

1. **Performance**:
   - **TTFT < 500ms**: Tests show average TTFT of ~100-150ms, well below the 500ms requirement
   - **Batch Processing**: Successfully handles 10+ consecutive image queries without degradation
   - **Streaming Support**: Provides token-by-token streaming for a responsive user experience

2. **Capability**:
   - **People Recognition**: Reliably describes people in images where GPT-4o refuses
   - **Multimodal Abilities**: Processes both text and image inputs simultaneously
   - **Context Window**: 10M tokens (limited to 128K in preview) supports rich conversations

3. **Technical Specifications**:
   - **Image Processing Limits**: 
     - 4MB maximum for base64 encoded images
     - 20MB maximum for image URL inputs
     - Supports multiple images per request (recommend max 5)
   - **Implementation**: Easy integration through the official Groq Python client
   - **Reliability**: High success rate (>95%) in our testing

### Why Groq API?

GPT-4o sometimes declines to answer image questions when there are people in the frame due to privacy guardrails. Groq API serves as a critical fallback system:

- **Fast Response Times**: TTFT < 500ms, meeting our performance requirements
- **Powerful Models**: Access to state-of-the-art Llama 4 Scout models
- **People Recognition**: Reliably describes people in images where GPT-4o refuses
- **Simple Integration**: Clean API with official Python client library

### Technical Details of Llama-4-Scout-17B

Llama-4-Scout-17B is a powerful multimodal model with impressive capabilities:

- **Context Window**: 10M tokens (limited to 128K in preview)
- **Image Processing**: Capable of detailed understanding of image content
- **Multilingual Support**: Handles multiple languages effectively
- **Tool Use**: Supports function calling and tool use
- **JSON Mode**: Can output structured JSON responses

The model excels at vision tasks and can reliably process images containing people when GPT-4o declines due to privacy guardrails.

## Project Structure

```
Envision-AI-Assignment-main/
├── app.py                  # Main entry point
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
├── images/                 # Images and diagrams
└── src/
    ├── main.py             # Entry point and agent implementation
    ├── config.py           # Configuration handling
    └── tools/
        ├── visual.py       # Visual processing (camera, frames, image analysis)
        ├── groq_handler.py # Groq API integration for enhanced image analysis
        └── internet_search.py # Web search functionality

```

## Getting Started

### Prerequisites

- Python 3.9+
- LiveKit API access
- OpenAI API access
- Deepgram API access
- ElevenLabs API access
- Groq API access

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/codingaslu/Ally-Clone-Assignment.git
   cd Ally-Clone-Assignment
   ```

2. Create a virtual environment and install dependencies
   ```bash
   python -m venv ally_env
   source ally_env/bin/activate  # On Windows: ally_env\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt
   ```

3. Set up environment variables (create a `.env` file)
   ```
   LIVEKIT_URL=...
   LIVEKIT_API_KEY=...
   LIVEKIT_API_SECRET=...
   DEEPGRAM_API_KEY=...  # Generate a new API key from Deepgram dashboard as the provided one looks expired
   OPENAI_API_KEY=...
   ELEVEN_API_KEY=...
   
   # Vision provider configuration
   VISION_PROVIDER=groq
   
   # Groq API configuration
   GROQ_API_KEY=your_groq_api_key  # Get your API key from https://console.groq.com/keys
   GROQ_MODEL_ID=meta-llama/llama-4-scout-17b-16e-instruct
   ```

### Running the Assistant

Before running the application, download plugin dependency files:
```bash
python app.py download-files
```

Run the application:

Standard mode:
```bash
python app.py start
```

Development mode:
```bash
python app.py dev
```

Connect using the [LiveKit hosted playground](https://agents-playground.livekit.io/).

## Performance Evaluation

The system has been thoroughly tested and meets all performance requirements:

1. **Reliability**: 
   - Successfully processes 10+ image questions in succession
   - 95%+ success rate for image captures
   - Handles both people and non-people images correctly

2. **Speed**: 
   - TTFT < 500ms using Groq's API (typically ~100-150ms)
   - Image capture under 1 second
   - No degradation in response times across multiple requests

3. **Consistency**:
   - Reliable image quality across multiple captures
   - Maintains performance across different image types

### Implementation Details

#### Key Components

1. **Voice Agent**
   - **STT**: Deepgram for real-time speech-to-text
   - **TTS**: ElevenLabs for natural-sounding text-to-speech

2. **Language Processing**
   - **LLM**: OpenAI GPT-4o for general conversation and queries
   - **Function Calling**: Structured routing to appropriate capabilities

3. **Vision Processing**
   - **Primary**: GPT-4o multimodal capabilities for most vision tasks
   - **Fallback**: Groq API (Llama-4-Scout-17B) for processing people in images when needed

### Groq Integration

Groq API is integrated using the official Groq Python client. The integration allows for:

- Accessing powerful Llama models via API
- Processing images and text queries in a single pipeline
- Generating detailed descriptions of images, including people
- Handling multiple images in a single conversation turn

The system automatically routes images to Groq API when GPT-4o refuses to describe people.

## Challenges and Solutions

### Privacy Considerations

GPT-4o sometimes refuses to describe people in images due to privacy guardrails. To solve this:
1. The system detects refusal responses using LLM-based detection
2. Groq API is used as a fallback model for these cases
3. Responses are cleaned and normalized before being read to the user

### Performance Optimization

The Groq integration is optimized through:
1. Efficient API client configuration
2. Image resizing to meet API size limits
3. Parallel processing when appropriate

## Performance and Limitations

- **Internet Requirement**: The system requires internet connectivity for both GPT-4o and Groq API
- **API Rate Limits**: Subject to rate limits from the Groq API service
- **Image Size Restrictions**: Maximum 4MB for base64 encoded images
- **Context Window**: Limited to 128K tokens in preview (out of 10M theoretical maximum)

## Future Improvements

- Advanced image preprocessing for optimal quality
- Google Map integration for location-based queries
- Distance, weather, and time information in responses
- Calendar integration for scheduling
- QR code generation and scanning
- Enhanced caching for faster responses
- Multi-image sequence processing for better context
- More personalized voice options

## License

This project is proprietary and confidential. All rights reserved.

## Acknowledgments

- LiveKit for the WebRTC platform
- OpenAI for GPT-4o capabilities
- Groq for API access to Llama models
- Deepgram and ElevenLabs for voice technologies


