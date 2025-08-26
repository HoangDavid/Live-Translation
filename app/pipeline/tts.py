import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice, SynthesisConfig

MODEL  = "app/pipeline/voices/vi/vi_VN-vivos-x_low.onnx"
CONFIG = "app/pipeline/voices/vi/vi_VN-vivos-x_low.onnx.json"

voice = PiperVoice.load(MODEL, config_path=CONFIG)
text = "Xin chào. Mình tên là Việt. Bạn tên là gì?"

# Get first chunk to learn sample rate / channels
gen = voice.synthesize(text, SynthesisConfig())
first = next(gen)

sr = first.sample_rate or getattr(voice, "sample_rate", 22050)
ch = first.sample_channels or 1

def write_chunk(stream, chunk):
    # Prefer float32 from chunk (your print shows audio_float_array populated)
    if getattr(chunk, "audio_float_array", None) is not None:
        buf = chunk.audio_float_array
    elif getattr(chunk, "audio", None):  # int16 bytes fallback
        buf = np.frombuffer(chunk.audio, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        return
    if ch > 1 and buf.ndim == 1:
        buf = buf.reshape(-1, ch)
    stream.write(buf)

with sd.OutputStream(samplerate=sr, channels=ch, dtype="float32") as stream:
    write_chunk(stream, first)
    for chunk in gen:
        write_chunk(stream, chunk)

print("Done.")

class TTS:
    def __init__():
        pass

    def new_tts_task():
        pass