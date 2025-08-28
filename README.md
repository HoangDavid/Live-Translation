# Live Translation Pipeline

### Currently a work in progress

A low-latency live translation pipeline designed to provide real-time captions, translation, and dubbing for video conferencing and streaming applications. The system prioritizes speed and lightweight deployment by leveraging Faster Whisper Tiny for speech-to-text, OPUS-MT models for machine translation, and Piper for text-to-speech synthesis.

### How to use
1. Install dependencies
```bash
pip install -r requirements.txt
```

2) Install Machine Translation and Voice models

- **Piper TTS voices** (pick these from the repo):
  - https://huggingface.co/rhasspy/piper-voices  
    - **vi:** `vi/vi_VN/vais1000/medium`  
    - **en:** `en/en_US/amy/low`  
    - **fr:** `fr/fr_FR/gilles/low`

- **Machine Translation (OPUS-MT / Marian)** (pick these model pages):
  - https://huggingface.co/Helsinki-NLP/  
    - `opus-mt-vi-en`, `opus-mt-en-vi`, `opus-mt-fr-en`, `opus-mt-en-fr`, `opus-mt-fr-vi`, `opus-mt-vi-fr`

Then update your model url in the .env file

3. Run the service:
you can run test indepedently for now by running each layers in /pipeline, run:
python app/pipeline/stt.py
python app/pipeline/mt.py
python app/pipeline/tts.py

Note: running thet test required you to open your mic, please have that enabled before runnnig the tests.

