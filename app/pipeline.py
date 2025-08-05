import sounddevice as sd
import numpy as np
import os
import functools
import asyncio, concurrent.futures
from faster_whisper import WhisperModel, BatchedInferencePipeline
from dataclasses import dataclass
from difflib import SequenceMatcher

@dataclass
class Worker:
    task: asyncio.Task
    inbound: asyncio.Queue
    outbound: asyncio.Queue
    languge: str
    stop: asyncio.Event


# Speech to text module
class STT:
    def __init__(self, model_size="tiny", sample_rate=16_000,
                transcribe_rate=0.5, context_limit=5, pool_size=os.cpu_count()):
        
        # whisper config
        self.transcribe_rate = transcribe_rate # how often the model get called
        self.context_limit = context_limit # default by 10 seconds of audio

        # audio sampling config
        self.sample_rate = sample_rate
        self.block_samples = self.sample_rate // 1000 # process 16 chunk of 1 ms audios

        # stt model thread pooling for heavier tasks
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)

        # load model once 
        self._model = WhisperModel(model_size, compute_type="int8")
        self._pipe = BatchedInferencePipeline(model=self._model)


    async def new_stt_task(self, language) -> Worker:
        # spawn a background stt worker
        loop = asyncio.get_running_loop()
        stop_evt = asyncio.Event()
        inboundQ = asyncio.Queue()
        outboundQ = asyncio.Queue()
        
        task = loop.create_task(self._worker(language, inboundQ, outboundQ, stop_evt))

        return Worker(task=task, languge=language, inbound=inboundQ, outbound=outboundQ, stop=stop_evt)


    async def _worker(self, language: str, inbound: asyncio.Queue, outbound: asyncio.Queue, stop: asyncio.Event) -> ValueError:
        buf = np.zeros(0, dtype=np.float32)
        last_ts = asyncio.get_event_loop().time()
        prev_words = []

        
        try:
            while not stop.is_set():
                try:
                    # get and add audio chunk to buffering
                    chunk = await asyncio.wait_for(inbound.get(), timeout=self.transcribe_rate)
                    
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.0)
                    continue


                buf = np.concatenate((buf, chunk))
                if (buf.shape[0] / self.sample_rate) >= self.context_limit:
                    # reduce the window by 2 second
                    buf = buf[int(self.sample_rate * 1):]


                now = asyncio.get_event_loop().time()
                if now - last_ts >= self.transcribe_rate:
                    cur_words = await self._transcribe(prev_words, language, buf)
                    if cur_words:
                        print(cur_words)
                
                    # yeilding to other coroutines
                    last_ts = now
                
                await asyncio.sleep(0)

        except Exception as e:
            print("Failed: ", e)


    async def _transcribe(self, prev, language, buf) -> list[str]:
        try:
            func = functools.partial(
                self._pipe.transcribe, buf,
                language=language, beam_size=10,
                batch_size=8, vad_filter=True,
            )

            segs, _ = await asyncio.get_running_loop().run_in_executor(self._pool, func)

            raw = ' '.join(seg.text for seg in segs).strip()
            tokens = raw.split()
                
            return tokens
        except Exception as e:
            print("Transcription faield: ", e)
        

async def test():
    stt = STT()
    worker = await stt.new_stt_task("en")
    loop = asyncio.get_running_loop()

    def cb(indata, frames, t, status):
        loop.call_soon_threadsafe(
            worker.inbound.put_nowait,
            indata[:, 0].copy()
        )
    
    with sd.InputStream(samplerate=stt.sample_rate, channels=1, dtype="float32", blocksize=stt.block_samples, callback=cb):
        while True:
            try:
                await asyncio.sleep(0)
            except KeyboardInterrupt:
                break
    
    worker.stop.set()
    await worker.task
    print("\nexiting...")


if __name__ == "__main__":
    asyncio.run(test())
