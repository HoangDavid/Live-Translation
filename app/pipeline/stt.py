import sounddevice as sd
import numpy as np
import functools, sys, asyncio, concurrent.futures
from nanoid import generate
from faster_whisper import WhisperModel, BatchedInferencePipeline
from dataclasses import dataclass
from difflib import SequenceMatcher

@dataclass
class Worker:
    task: asyncio.Task
    inbound: asyncio.Queue
    outbound: asyncio.Queue
    stop: asyncio.Event


# Speech to text module
class STT:
    def __init__(self, model_size="tiny", sample_rate=16_000, chunk_size=10,
                transcribe_rate=0.5, context_limit=4, pool_size=4):
        
        # whisper config
        self.transcribe_rate = transcribe_rate # how often the model get called
        self.context_limit = context_limit # default by 10 seconds of audio

        # audio sampling config
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size # audio chunk in ms
        self.block_samples = self.sample_rate * chunk_size  // 1000 # process 16 chunk of 10 ms audios

        # stt model thread pooling for heavier tasks
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)

        # load model once 
        self._model = WhisperModel(model_size, compute_type="int8")
        self._pipe = BatchedInferencePipeline(model=self._model)



    async def new_stt_task(self, language) -> Worker:
        # spawn a background stt worker task
        loop = asyncio.get_running_loop()
        stop_evt = asyncio.Event()
        inboundQ = asyncio.Queue()
        outboundQ = asyncio.Queue()
        
        task = loop.create_task(self._worker(language, inboundQ, outboundQ, stop_evt))

        return Worker(task=task, inbound=inboundQ, outbound=outboundQ, stop=stop_evt)


    async def _worker(self, language: str, inbound: asyncio.Queue, outbound: asyncio.Queue, stop: asyncio.Event):
        buf = np.zeros(0, dtype=np.float32)
        last_ts = asyncio.get_event_loop().time()
        prev_tokens = []
        
        try:
            while not stop.is_set():
                try:
                    # get and add audio chunk to buffering
                    chunk = await asyncio.wait_for(inbound.get(), timeout=0.01)

                except asyncio.TimeoutError:
                    # yeild
                    await asyncio.sleep(0)
                    continue

                # trim window when reach limit
                if (buf.shape[0] / self.sample_rate) >= self.context_limit:
                    buf = buf[int(self.sample_rate * 1): ]

                    # trim off old tokens
                    drop_frac  = 1.0 / self.context_limit
                    drop_count = int(len(prev_tokens) * drop_frac)
                    prev_tokens = prev_tokens[drop_count:]
                    

                buf = np.concatenate((buf, chunk))
                
                # call transcribe at fixed rate
                now = asyncio.get_event_loop().time()
                if now - last_ts >= self.transcribe_rate:
                    prev_tokens, ops = await self._transcribe(prev_tokens, language, buf)
                    outbound.put_nowait(ops)
                    last_ts = now
                
                # yeild
                await asyncio.sleep(0)

        except Exception as e:
            print("Failed: ", e)
            sys.exit()


    async def _transcribe(self, prev_tokens, language, buf) -> list[str]:
        try:
            # run through stt model
            segments, _ = await asyncio.get_running_loop().run_in_executor(
                self._pool,
                functools.partial(
                self._pipe.transcribe, buf,
                language=language, beam_size=8,
                batch_size=16, vad_filter=True,
            ))
            
            new_tokens = []
            for segment in segments:
                if segment.text:
                    for w in segment.text.strip().split():
                        new_id = generate("0123456789abcdef", 8)
                        new_tokens.append((new_id, w))
            
            # run diffing algo
            updated_prev, ops = await asyncio.get_running_loop().run_in_executor(
                self._pool, functools.partial(
                self._get_ops, prev_tokens, new_tokens
            ))
                
            return updated_prev, ops
        
        except Exception as e:
            print("Transcription faield: ", e)
            sys.exit()


    # corrections and flush new tokens
    def _get_ops(self, prev_tokens, new_tokens):
        try:
            ops = {"flush": [], "replace": []}
            updated_prev = []

            sm = SequenceMatcher(a=[n for (_, n) in prev_tokens], 
                                 b=[n for (_, n) in new_tokens])

            # empty sequence
            if not new_tokens:
                return updated_prev, ops
            
            # finding corrections based on previous inference
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    updated_prev.extend(new_tokens[j1:j2])

                elif tag == "replace":
                    prev_seq = prev_tokens[i1:i2]
                    new_seq = new_tokens[j1:j2]

                    overlap = min(len(prev_seq), len(new_seq))

                    for k in range(overlap):
                        id, _ = prev_seq[k]
                        _, fix = new_seq[k]
                        ops["replace"].append((id, fix))

                    if len(new_seq) > len(prev_seq):
                        ops["flush"].extend([new_seq[k] for k in range(overlap, len(new_seq))])

                    updated_prev.extend(new_tokens[j1:j2])

                elif tag == "insert":
                    ops["flush"].extend((new_tokens[j1:j2]))
                    updated_prev.extend(new_tokens[j1:j2])


            return updated_prev, ops

        except Exception as e:
            print("Failed to get ops: ", e)
            sys.exit()

        

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
                await asyncio.sleep(2)
                worker.languge = "vi"
            except KeyboardInterrupt:
                break
    
    worker.stop.set()
    await worker.task
    print("\nexiting...")


if __name__ == "__main__":
    asyncio.run(test())

        
    
        
    
        