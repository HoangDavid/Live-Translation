import asyncio, os, concurrent, functools
import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from piper.voice import PiperVoice, SynthesisConfig, Iterable, AudioChunk
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env") 

@dataclass
class TTSTask:
    task: asyncio.Task
    inboundQ: asyncio.Queue
    outboundQ: asyncio.Queue
    stop_evt:  asyncio.Event
    sample_rate: int
    channel: int
    

class TTS:
    def __init__(self, pool_size=4):
        # thread pool size
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)

        # Load up piper voices
        self.models = {"vi": os.getenv("VI_MODEL"), "fr": os.getenv("FR_MODEL"), "en": os.getenv("EN_MODEL")}
        self.configs = {"vi": os.getenv("VI_MODEL_CONFIG"), "fr": os.getenv("FR_MODEL_CONFIG"), "en": os.getenv("EN_MODEL_CONFIG")}
        print(self.models)
        print(self.configs)

    # Load a new tts task
    def new_tts_task(self, inboundQ: asyncio.Queue, tgt_lang: str) -> TTSTask:
        loop = asyncio.get_running_loop()
        stop_evt = asyncio.Event()
        outboundQ = asyncio.Queue()
        voice = PiperVoice.load(model_path=self.models[tgt_lang], config_path=self.configs[tgt_lang])

        # sample first to get voice model sample rate and channels
        trial = voice.synthesize("abcd")
        first = next(trial)
        sr = first.sample_rate or getattr(voice, "sample_rate", 16000)
        ch = first.sample_channels or 1

        task = loop.create_task(self._worker(inboundQ, outboundQ, voice, stop_evt))

        return TTSTask(task=task, inboundQ=inboundQ, outboundQ=outboundQ, stop_evt=stop_evt, sample_rate=sr, channel=ch)


    # Load a worker task
    async def _worker(self, inboundQ: asyncio.Queue, outboundQ: asyncio.Queue, voice: PiperVoice, stop_evt: asyncio.Event):

        synth_config = SynthesisConfig()
        try:
            while not stop_evt.is_set():
                try:
                    translation = await asyncio.wait_for(inboundQ.get(), timeout=0.01)
                except TimeoutError:
                    await asyncio.sleep(0)
                    continue

                
                audio_chunks = await self._synthesize(translation, voice, synth_config)
                for chunk in audio_chunks:
                    outboundQ.put_nowait(chunk)

                    
        except Exception as e:
            print("Failed: ", e)

    async def _synthesize(self, text: str, voice: PiperVoice, config: SynthesisConfig) -> Iterable[AudioChunk]:

        audio_chuks = await asyncio.get_running_loop().run_in_executor(
            self._pool,
            functools.partial(
                voice.synthesize,
                text,
                config
            )
        )

        return audio_chuks
    


def write_chunk(stream, chunk, ch):
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
    

async def test():
    tts = TTS()
    inboundQ = asyncio.Queue()
    inboundQ.put_nowait("Xin chào. Mình tên là Việt. Bạn tên là gì?")
    inboundQ.put_nowait( "Chào bạn. Mình tên là Hoàng.  Bạn khoẻ không nhỉ?")
    worker = tts.new_tts_task(inboundQ, "vi")

    with sd.OutputStream(samplerate=worker.sample_rate, channels=worker.channel, dtype="float32") as stream:
        while True:
            try:
                chunk = await asyncio.wait_for(worker.outboundQ.get(), timeout=0.01)
            except TimeoutError:
                await asyncio.sleep(0)
                continue
            

            write_chunk(stream, chunk, worker.channel)


if __name__ == "__main__":
    asyncio.run(test())


