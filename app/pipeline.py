import sounddevice as sd
import queue
import numpy as np
import sys
import time
from faster_whisper import WhisperModel, BatchedInferencePipeline
from difflib import SequenceMatcher

MODEL_SIZE = "tiny"
LANGUAGE = "en"
SAMPLE_RATE = 16_000
BLOCK_MS = 1
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000

TRANSCRIBE_RATE = 0.3
SILENCE_TOL = 1.2
WINDOW_LIMIT = 20

q = queue.Queue()
model = WhisperModel(MODEL_SIZE, compute_type="int8")
pipe = BatchedInferencePipeline(model=model)

# words to be flushed and corrections
def flush_and_corrects(prev, new):
    ops = {"flush_words": [], "replace": []}

    overlap = 0
    if len(new) > len(prev):
        ops["flush_words"] = new[len(prev):]
    
    sm = SequenceMatcher(a=prev, b=new)
    matched = sm.find_longest_match(0, len(prev), 0, len(new))
    
    prev_midex = matched.a + matched.size
    new_mindex = matched.b + matched.size

    old_rem = prev[prev_midex:]
    new_rem = new[new_mindex:]

    overlap = min(len(old_rem), len(new_rem))
    
    for i, k in zip(old_rem[:overlap], new_rem[:overlap]):
        if i != k:
            ops["replace"].append((i, k))
        else:
            continue

    return ops


# adding audio to queue
def cb(indata, frames, t, status):
    q.put(indata[:, 0].copy())


# transcription loop
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

                    if flush_words != []:
                        text = ' '.join(new)
                        print(text, flush=True)
                        silence = 0.0
                        

                    for r in ops["replace"]:
                        old_word, new_word = r
                        # print(f"replace: {old_word} -> {new_word}")
                    
                    # print()

                    silence += now - last_trans
                    prev = new                    
                        
                    if silence >= SILENCE_TOL or len(new) >= WINDOW_LIMIT:
                        print("reset window")
                        buf = np.zeros(0, dtype=np.float32)
                        silence = 0.0
                        prev = []
                        

                    last_trans = time.monotonic()
                     
            except KeyboardInterrupt:
                print("\nexiting...")
                sys.exit()

if __name__ == "__main__":
    loop()