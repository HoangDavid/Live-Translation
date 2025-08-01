import sounddevice as sd
import queue
import numpy as np
import sys
import time
from faster_whisper import WhisperModel, BatchedInferencePipeline

MODEL_SIZE = "tiny"
LANGUAGE = "en"
SAMPLE_RATE = 16_000
BLOCK_MS = 1
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000


TRANSCRIBE_RATE = 0.3

# TODO: sentence pause

q = queue.Queue()
model = WhisperModel(MODEL_SIZE, compute_type="int8")
pipe = BatchedInferencePipeline(model=model)

def flush_and_corrects(prev, new):
    ops = {"flush_words": [], "replace": []}

    overlap = 0
    if len(new) > len(prev):
        ops["flush_words"] = new[len(prev):]
    
    overlap = min(len(new), len(prev))

    
    for i in range(overlap):
        if prev[i] != new[i]:
            ops["replace"].append((prev[i], new[i]))
        else:
            continue

    return ops



def cb(indata, frames, t, status):
    q.put(indata[:, 0].copy())

def loop():
    buf = np.zeros(0, dtype=np.float32)
    last_trans = time.monotonic()
    silence = 0.0
    prev = []

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SAMPLES, callback=cb):
        while True:
            try:
                data = q.get()
                buf = np.concatenate((buf, data))
                now = time.monotonic()

                if now - last_trans >= TRANSCRIBE_RATE:
                    segments, _ = pipe.transcribe(buf, language=LANGUAGE, beam_size=10, batch_size=10, vad_filter=True)

                    
                    new = []
                    for segment in segments:
                        new += segment.text.strip().split()
                    
                    ops = flush_and_corrects(prev, new)
                    flush_words = ops["flush_words"]
                    replace = ops["replace"]

                    if new != []:
                        text = ' '.join(new)
                        print(text, flush=True)

                    for r in replace:
                        old_word, new_word = r
                        print(f"replace: {old_word} -> {new_word}")
                    

                    print()
                    if len(new) <= len(prev):
                        silence += TRANSCRIBE_RATE
                    else:
                        prev = new
                    
                    
                        
                    if silence > 0.5:
                        print("reset window")
                        # every window reset -> send to mt -> tts
                        buf = np.zeros(0, dtype=np.float32)
                        silence = 0.0
                        prev = []
                        

                    last_trans = time.monotonic()

                    
                     
            except KeyboardInterrupt:
                print("\nexiting...")
                sys.exit()

if __name__ == "__main__":
    loop()