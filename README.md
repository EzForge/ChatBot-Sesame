# Building a Voice-Based AI Agent: A Comprehensive Guide

I'll walk you through creating a completely local, Docker-based voice AI agent that can process speech input, generate responses using a lightweight LLM, and produce natural-sounding voice output.

## Overview

We'll build a system with these components:
- Speech-to-Text: Whisper (open-source)
- Language Model: Llama-3.2-1B
- Text-to-Speech: CSM (Controllable Speech Model)
- Orchestration: Python script

Let's break this down into manageable steps.

## Step 1: Environment Setup

First, let's ensure you have the necessary prerequisites installed on your Windows 11 system.

### Installing Docker

1. Install WSL 2 (Windows Subsystem for Linux):
   ```
   wsl --install
   ```

2. Download and install Docker Desktop for Windows from the [Docker website](https://www.docker.com/products/docker-desktop/)

3. Verify installation:
   ```
   docker --version
   docker-compose --version
   ```

### Project Structure

Create a directory structure for your project:

```
voice-agent/
├── docker-compose.yml
├── stt/
│   └── Dockerfile
├── llm/
│   └── Dockerfile
├── tts/
│   └── Dockerfile
└── app/
    ├── Dockerfile
    └── app.py
```

## Step 2: Setting Up Speech-to-Text (Whisper)

We'll use OpenAI's Whisper, which is open-source and can run locally.

### Create the STT Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg
RUN pip install torch torchaudio openai-whisper
RUN pip install fastapi uvicorn python-multipart

COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Create a server.py file in the stt directory:

```python
from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile
import os

app = FastAPI()

# Load Whisper model
model = whisper.load_model("base")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Create a temporary file to store the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name
    
    try:
        # Transcribe the audio file
        result = model.transcribe(temp_audio_path)
        transcription = result["text"]
        
        # Clean up the temporary file
        os.unlink(temp_audio_path)
        
        return {"text": transcription}
    except Exception as e:
        # Clean up the temporary file in case of error
        os.unlink(temp_audio_path)
        return {"error": str(e)}

@app.get("/health/")
def health_check():
    return {"status": "healthy"}

```

## Step 3: Setting Up the Language Model (Llama-3.2-1B)

We'll use llama.cpp, which allows efficient running of LLMs on consumer hardware.

### Create the LLM Dockerfile:

```dockerfile
FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    wget

# Clone and build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    make && \
    pip3 install -e .

# Install server dependencies
RUN pip3 install fastapi uvicorn

# Copy the server script
COPY server.py /app/
COPY download-model.sh /app/

# Create model directory
RUN mkdir -p /app/models

# Make the download script executable
RUN chmod +x /app/download-model.sh

# Download model on first run
RUN /app/download-model.sh

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Create a download-model.sh script in the llm directory:

```bash
#!/bin/bash

MODEL_DIR="/app/models"
MODEL_URL="https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/model-q4_0.gguf"
MODEL_FILENAME="llama-3.2-1b-q4_0.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILENAME}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading Llama-3.2-1B model..."
    wget -O "$MODEL_PATH" "$MODEL_URL"
    echo "Model downloaded successfully!"
else
    echo "Model already exists, skipping download."
fi

```

### Create a server.py file in the llm directory:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import json
import os

app = FastAPI()

MODEL_PATH = "/app/models/llama-3.2-1b-q4_0.gguf"

class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

@app.post("/generate/")
async def generate_text(request: QueryRequest):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Model file not found")
    
    # Prepare prompt with proper formatting for Llama
    formatted_prompt = f"[INST] {request.prompt} [/INST]"
    
    # Run llama.cpp
    try:
        cmd = [
            "/app/llama.cpp/main",
            "-m", MODEL_PATH,
            "-n", str(request.max_tokens),
            "--temp", str(request.temperature),
            "--prompt", formatted_prompt,
            "-f", "/dev/null",  # Avoid writing to stdout directly
            "--json"  # Output in JSON format
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {result.stderr}")
        
        # Parse the JSON output from llama.cpp
        try:
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if line.strip().startswith('{') and '"content":' in line:
                    response_json = json.loads(line)
                    if 'content' in response_json:
                        # Extract just the model's response
                        return {"text": response_json['content']}
            
            # If we didn't find proper JSON, return the raw output
            return {"text": result.stdout.strip()}
            
        except json.JSONDecodeError:
            # Fallback: return the raw output
            return {"text": result.stdout.strip()}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
def health_check():
    return {"status": "healthy"}

```

## Step 4: Setting Up Text-to-Speech (CSM)

We'll use the Controllable Speech Model (CSM) from Sesame AI Labs.

### Create the TTS Dockerfile:

```dockerfile
FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg

# Clone the CSM repository
RUN git clone https://github.com/SesameAILabs/csm.git

# Install Python dependencies
WORKDIR /app/csm
RUN pip install -e .
RUN pip install fastapi uvicorn python-multipart

# Create a directory for models
RUN mkdir -p /app/models

# Copy the download script and server
COPY download-model.sh /app/
COPY server.py /app/

# Make the download script executable
RUN chmod +x /app/download-model.sh

# Download model on first run
RUN /app/download-model.sh

WORKDIR /app

EXPOSE 8002

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
```

### Create a download-model.sh script in the tts directory:

```bash
#!/bin/bash

MODEL_DIR="/app/models"
mkdir -p "$MODEL_DIR"

# Download CSM model
if [ ! -d "$MODEL_DIR/csm-1b" ]; then
    echo "Downloading CSM-1B model..."
    git clone https://huggingface.co/SesameAILabs/csm-1b "$MODEL_DIR/csm-1b"
    echo "CSM-1B model downloaded successfully!"
else
    echo "CSM-1B model already exists, skipping download."
fi

```

### Create a server.py file in the tts directory:

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torchaudio
import os
import time
import tempfile
from csm.models.csm import CSM
from csm.inference.generate import load_model, generate_audio

app = FastAPI()

# Load CSM model
MODEL_PATH = "/app/models/csm-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CSM model on {device}...")
model = load_model(MODEL_PATH, device=device)
print("Model loaded successfully!")

class TTSRequest(BaseModel):
    text: str
    speaker_id: str = "default"  # Default speaker if none specified
    speed: float = 1.0           # Speed factor (1.0 is normal)

@app.post("/synthesize/")
async def synthesize_speech(request: TTSRequest):
    try:
        # Create output directory if it doesn't exist
        os.makedirs("/tmp/csm_output", exist_ok=True)
        
        # Generate a unique filename
        timestamp = int(time.time())
        output_path = f"/tmp/csm_output/speech_{timestamp}.wav"
        
        # Generate audio using CSM
        generate_audio(
            model=model,
            text=request.text,
            output_path=output_path,
            speaker_id=request.speaker_id,
            device=device
        )
        
        # Return the audio file
        return FileResponse(
            path=output_path, 
            media_type="audio/wav", 
            filename="response.wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
def health_check():
    return {"status": "healthy"}

```

## Step 5: Creating the Orchestration Layer

Now, let's create the main application that ties all components together.

### Create the App Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    portaudio19-dev \
    ffmpeg \
    python3-pyaudio \
    build-essential

# Install Python dependencies
RUN pip install fastapi uvicorn python-multipart requests pyaudio sounddevice soundfile

# Copy the application code
COPY app.py .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Create an app.py file in the app directory:

```python
import os
import io
import time
import tempfile
import uvicorn
import requests
import sounddevice as sd
import soundfile as sf
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

# Service URLs
STT_SERVICE_URL = "http://stt:8000/transcribe"
LLM_SERVICE_URL = "http://llm:8001/generate"
TTS_SERVICE_URL = "http://tts:8002/synthesize"

class ConversationInput(BaseModel):
    audio_file: UploadFile = File(...)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice AI Agent</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .container {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            button {
                padding: 10px 20px;
                margin: 10px;
                font-size: 16px;
                cursor: pointer;
            }
            #status, #transcription, #response {
                margin: 10px 0;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                width: 100%;
                min-height: 40px;
            }
            audio {
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Voice AI Agent</h1>
            <p>Click and hold the Record button to start recording, release to stop.</p>
            
            <button id="recordButton">Record</button>
            
            <div>
                <h3>Status:</h3>
                <div id="status">Ready</div>
            </div>
            
            <div>
                <h3>Your speech:</h3>
                <div id="transcription"></div>
            </div>
            
            <div>
                <h3>AI Response:</h3>
                <div id="response"></div>
            </div>
            
            <audio id="audioPlayer" controls></audio>
        </div>

        <script>
            const recordButton = document.getElementById('recordButton');
            const statusDiv = document.getElementById('status');
            const transcriptionDiv = document.getElementById('transcription');
            const responseDiv = document.getElementById('response');
            const audioPlayer = document.getElementById('audioPlayer');
            
            let mediaRecorder;
            let audioChunks = [];
            
            recordButton.addEventListener('mousedown', startRecording);
            recordButton.addEventListener('mouseup', stopRecording);
            recordButton.addEventListener('mouseleave', stopRecording);
            
            async function startRecording() {
                audioChunks = [];
                statusDiv.textContent = 'Recording...';
                
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    
                    mediaRecorder.ondataavailable = event => {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.start();
                } catch (err) {
                    statusDiv.textContent = 'Error: ' + err.message;
                }
            }
            
            function stopRecording() {
                if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
                
                mediaRecorder.stop();
                statusDiv.textContent = 'Processing...';
                
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const formData = new FormData();
                    formData.append('audio', audioBlob);
                    
                    try {
                        const response = await fetch('/process', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            transcriptionDiv.textContent = result.transcription;
                            responseDiv.textContent = result.response;
                            
                            // Update the audio player
                            audioPlayer.src = result.audio_url;
                            
                            statusDiv.textContent = 'Ready';
                        } else {
                            const error = await response.text();
                            statusDiv.textContent = 'Error: ' + error;
                        }
                    } catch (err) {
                        statusDiv.textContent = 'Error: ' + err.message;
                    }
                };
            }
        </script>
    </body>
    </html>
    """

@app.post("/process")
async def process_conversation(audio: UploadFile = File(...)):
    # Create a temporary file to store the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        content = await audio.read()
        temp_audio.write(content)
        audio_path = temp_audio.name
    
    try:
        # Step 1: Speech-to-Text
        with open(audio_path, "rb") as audio_file:
            files = {"file": (os.path.basename(audio_path), audio_file, "audio/wav")}
            stt_response = requests.post(STT_SERVICE_URL, files=files)
            
        if stt_response.status_code != 200:
            os.unlink(audio_path)
            raise HTTPException(status_code=500, detail=f"STT service error: {stt_response.text}")
            
        transcription = stt_response.json().get("text", "")
        
        # Step 2: Language Model Processing
        llm_payload = {
            "prompt": transcription,
            "max_tokens": 256,
            "temperature": 0.7
        }
        
        llm_response = requests.post(LLM_SERVICE_URL, json=llm_payload)
        
        if llm_response.status_code != 200:
            os.unlink(audio_path)
            raise HTTPException(status_code=500, detail=f"LLM service error: {llm_response.text}")
            
        response_text = llm_response.json().get("text", "")
        
        # Step 3: Text-to-Speech
        tts_payload = {
            "text": response_text,
            "speaker_id": "default",
            "speed": 1.0
        }
        
        tts_response = requests.post(TTS_SERVICE_URL, json=tts_payload)
        
        if tts_response.status_code != 200:
            os.unlink(audio_path)
            raise HTTPException(status_code=500, detail=f"TTS service error: {tts_response.text}")
            
        # Save the audio response
        timestamp = int(time.time())
        output_path = f"/tmp/response_{timestamp}.wav"
        
        with open(output_path, "wb") as f:
            f.write(tts_response.content)
        
        # Clean up input audio file
        os.unlink(audio_path)
        
        # Return the response
        return {
            "transcription": transcription,
            "response": response_text,
            "audio_url": f"/audio/{timestamp}"
        }
    
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{timestamp}")
async def get_audio(timestamp: int):
    audio_path = f"/tmp/response_{timestamp}.wav"
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(audio_path, media_type="audio/wav")

if __name__ == "__main__":
    # Create temporary directory for audio files
    os.makedirs("/tmp", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8080)

```

## Step 6: Creating the Docker Compose File

Now, let's create a docker-compose.yml file in the root directory to orchestrate all the services:

```yaml
version: '3'

services:
  stt:
    build: ./stt
    ports:
      - "8000:8000"
    volumes:
      - stt_data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/"]
      interval: 30s
      timeout: 10s
      retries: 3

  llm:
    build: ./llm
    ports:
      - "8001:8001"
    volumes:
      - llm_data:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health/"]
      interval: 30s
      timeout: 10s
      retries: 3

  tts:
    build: ./tts
    ports:
      - "8002:8002"
    volumes:
      - tts_data:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health/"]
      interval: 30s
      timeout: 10s
      retries: 3

  app:
    build: ./app
    ports:
      - "8080:8080"
    depends_on:
      - stt
      - llm
      - tts
    volumes:
      - app_data:/tmp

volumes:
  stt_data:
  llm_data:
  tts_data:
  app_data:

```

## Step 7: Building and Running the System

With all components in place, it's time to build and run the system:

1. Navigate to your project directory:
   ```
   cd voice-agent
   ```

2. Build and start the containers:
   ```
   docker-compose up --build
   ```

   Note: The first build may take some time as it downloads all dependencies and models.

3. Access the voice agent interface by opening a browser and navigating to:
   ```
   http://localhost:8080
   ```

## Step 8: Using the Voice Agent

1. Open the web interface at http://localhost:8080
2. Click and hold the "Record" button to speak
3. Release the button to stop recording
4. The system will:
   - Transcribe your speech
   - Process it with the LLM
   - Generate a voice response
   - Play the audio

## Troubleshooting

### Common Issues:

1. **Resource Limitations**: LLMs and TTS models can be resource-intensive. If you face performance issues:
   - Edit the docker-compose.yml to limit CPU/memory allocation
   - Consider using smaller models
   - Check Docker Desktop settings to increase allocated resources

2. **Microphone Access**: Ensure your browser has permission to access your microphone.

3. **Model Downloads Failing**: If model downloads fail during container build:
   - Download the models manually and place them in the appropriate directories
   - Update the Docker files to use local models

4. **Container Communication**: If services can't communicate:
   - Check that the service names in the app.py match those in docker-compose.yml
   - Verify that all containers are running with `docker-compose ps`

## Further Customization

### Alternative Models:

- **STT**: You could replace Whisper with Wav2Vec2 or Vosk
- **LLM**: Llama-3.2-1B can be swapped with any local model compatible with llama.cpp
- **TTS**: CSM can be replaced with other TTS solutions like Piper or Coqui TTS

### Performance Optimization:

- Add caching for frequently used responses
- Implement streaming responses for a more interactive experience
- Use quantized models for faster inference on CPU

## Conclusion

You now have a fully functional voice-based AI agent running entirely locally on your Windows 11 machine using Docker containers. This setup provides privacy, control, and the ability to operate offline, while still delivering high-quality voice interactions.

The modular architecture allows for easy upgrades or replacements of individual components as better open-source alternatives become available.
