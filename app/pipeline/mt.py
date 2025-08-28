import sentencepiece as spm
import sys, concurrent, asyncio, functools, os
from transformers import MarianMTModel, MarianTokenizer
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

@dataclass
class MTTask:
    task: asyncio.Task
    inboundQ: asyncio.Queue
    outboundQ: asyncio.Queue
    stop_evt: asyncio.Event

class MT:
    def __init__(self, pool_size=4):
        
        # models
        self._models = {
            "mt-en-vi": os.getenv("MT_EN_VI"),
            "mt-vi-en": os.getenv("MT_VI_EN"),
            "mt-fr-en": os.getenv("MT_FR_EN"),
            "mt-en-fr": os.getenv("MT_EN_FR"),
            "mt-vi-fr": os.getenv("MT_VI_FR"),
            "mt-fr-vi": os.getenv("MT_FR_VI"),
        }

        # mt model thread pooiling for heavier tasks
        self._pool =  concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)


    def new_mt_task(self, inboundQ: asyncio.Queue, src_lang: str, tgt_lang: str) -> MTTask:
        loop = asyncio.get_running_loop()
        stop_evt = asyncio.Event()
        outboundQ = asyncio.Queue()

        # load model and tokenizer
        model_name =  self._models["mt"+"-"+src_lang+"-"+tgt_lang]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        # queue task
        task = loop.create_task(self._worker(model, tokenizer, inboundQ, outboundQ, stop_evt))

        return MTTask(task=task, inboundQ=inboundQ, outboundQ=outboundQ, stop_evt=stop_evt)
    


    async def _worker(self, model: MarianMTModel, tokenizer: MarianTokenizer, inboundQ: asyncio.Queue, outboundQ: asyncio.Queue, stop_evt: asyncio.Event):

        try:
            while not stop_evt.is_set():
                try:
                    # wait for translation from stt layer
                    transcription = await asyncio.wait_for(inboundQ.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.0)
                    continue
                
                translation = await asyncio.get_running_loop().run_in_executor(
                    self._pool, functools.partial(
                        self._translate,
                        model, tokenizer,
                        transcription
                    ))
                    
                outboundQ.put_nowait(translation)


        except Exception as e:
            print("Failed: ", e) 
            sys.exit()

    def _translate(self, model: MarianMTModel, tokenizer: MarianTokenizer, transcription: str) -> str:
        src_text = [transcription]
        batch = tokenizer(src_text, return_tensors="pt", padding=True)

        # generate translation
        gen = model.generate(**batch)

        # decode
        tgt_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        return tgt_texts[0]



async def test():
    mt = MT()
    inboundQ = asyncio.Queue()
    task = mt.new_mt_task(inboundQ=inboundQ, src_lang="en", tgt_lang="fr")

    inboundQ.put_nowait("hi")
    inboundQ.put_nowait("how are you?")
    inboundQ.put_nowait("im fine. thank you! How is your wife?")
    while True:
        try:
            try:
                translation = await asyncio.wait_for(task.outboundQ.get(), timeout=0.01)
                print("translation: ", translation)
            except:
                await asyncio.sleep(0)
                continue
        except KeyboardInterrupt:
            print("/nexiting...")
            task.stop_evt.set()
            break
        


if __name__ == "__main__":
    asyncio.run(test())
    

        



    
