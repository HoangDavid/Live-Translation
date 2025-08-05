import sounddevice as sd
import numpy as np
import os,functools, sys, asyncio, concurrent.futures
from nanoid import generate
from faster_whisper import WhisperModel, BatchedInferencePipeline
from dataclasses import dataclass
from difflib import SequenceMatcher
import webrtcvad

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
        self.block_samples = self.sample_rate * 10 // 1000 # process 16 chunk of 1 ms audios

        # silence detection
        self.vad = webrtcvad.Vad(3)
        self.silence_tol = 0.8

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

        return Worker(task=task, languge=language, inbound=inboundQ, outbound=outboundQ, stop=stop_evt)


    async def _worker(self, language: str, inbound: asyncio.Queue, outbound: asyncio.Queue, stop: asyncio.Event) -> ValueError:
        buf = np.zeros(0, dtype=np.float32)
        last_ts = asyncio.get_event_loop().time()
        prev_tokens = []
        silence = 0
        
        try:
            while not stop.is_set():
                try:
                    # get and add audio chunk to buffering
                    chunk = await asyncio.wait_for(inbound.get(), timeout=self.transcribe_rate)
                    if await self._is_silence(chunk):
                        # TODO: add silence to process less chunks
                        await asyncio.sleep(0)
                        continue

                except asyncio.TimeoutError:
                    # yeild
                    await asyncio.sleep(0)
                    continue
                
                # trim window when reach limit
                if (buf.shape[0] / self.sample_rate) >= self.context_limit or silence >= self.silence_tol:
                    buf = buf[int(self.sample_rate * 1): ]
                    # TODO: reset prev_tokens correctly

                buf = np.concatenate((buf, chunk))
                
                # call transcribe at fixed rate
                now = asyncio.get_event_loop().time()
                if now - last_ts >= self.transcribe_rate:
                    prev_tokens, ops = await self._transcribe(prev_tokens, language, buf)
                    # TODO: add ops to outbound buffer
                
                    last_ts = now
                
                # yeild
                await asyncio.sleep(0)

        except Exception as e:
            print("Failed: ", e)
            sys.exit()

    async def _is_silence(self, chunk) -> bool:
        pcm16 = (chunk * 32767).astype(np.int16).tobytes()
        is_speech = self.vad.is_speech(pcm16, sample_rate=self.sample_rate)

        return is_speech

    async def _transcribe(self, prev_tokens, language, buf) -> list[str]:
        try:
            # run through stt model
            segments, _ = await asyncio.get_running_loop().run_in_executor(
                self._pool,
                functools.partial(
                self._pipe.transcribe, buf,
                language=language, beam_size=8,
                batch_size=16, vad_filter=True,
            )
)
            
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
            ops = {"flush": [], "delete": [], "insert": [], "replace": []}
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
                    ops["replace"].extend([(prev_seq[k], new_seq[k]) for k in range(overlap)])

                    if len(new_seq) > len(prev_seq):
                        ops["flush"].extend([new_seq[k] for k in range(overlap, len(new_seq))])

                    elif len(new_seq) < len(prev_seq):
                        ops["delete"].extend([prev_seq[k] for k in range(overlap, len(prev_seq))])

                    updated_prev.extend(new_seq[j1:j2])

                elif tag == "insert":
                    if prev_tokens != []:
                        a_idx = None
                        b_idx = None

                        if i1 - 1 >= 0:
                            a_idx, _ = prev_tokens[i1 - 1]
                        if i1 + 1 < len(prev_tokens):
                            b_idx, _ = prev_tokens[i1 + 1]
                                                
                        ops["insert"].append((new_tokens[j1:j2][0], (a_idx, b_idx)))

                    else:
                        ops["flush"].extend((new_tokens[j1:j2]))

                    updated_prev.extend(new_tokens[j1:j2])

                elif tag == "delete":
                    ops["delete"].extend(prev_tokens[i1:i2])


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
                await asyncio.sleep(0)
            except KeyboardInterrupt:
                break
    
    worker.stop.set()
    await worker.task
    print("\nexiting...")


if __name__ == "__main__":
    asyncio.run(test())

        
    
        
    
        