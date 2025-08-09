import ctranslate2, sys, concurrent, asyncio, functools
from pathlib import Path
from models.small100_int8.tokenization_small100 import SMALL100Tokenizer
from dataclasses import dataclass

MODEL_DIR = Path(__file__).resolve().parent / "models" / "small100_int8"

if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))


@dataclass
class Worker:
    task: asyncio.Task
    inbound: asyncio.Queue
    outbound: asyncio.Queue
    stop: asyncio.Event


class MT:
    def __init__(self, inter_thread=1, intra_thread=4, pool_size=4, beam_size=4, lag=1.3):
        
        self.lag = lag # translation lag
        self.context_limit = 20
        
        # model config
        self._translator = ctranslate2.Translator(
            str(MODEL_DIR),
            device="cpu",
            inter_threads=inter_thread,
            intra_thread=intra_thread,
        )

        # mt model thread pooiling for heavier tasks
        self._pool =  concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)


    async def new_mt_task(self, src_lang, tgt_lang):
        loop = asyncio.get_running_loop()
        stop_evt = asyncio.Event()
        inboundQ = asyncio.Queue()
        outboundQ = asyncio.Queue()

        task = loop.create_task(self._worker(src_lang, tgt_lang, inboundQ, outboundQ, stop_evt))

        return Worker(task=task, inbound=inboundQ, outbound=outboundQ, stop=stop_evt)

    async def _worker(self, src_lang: str, tgt_lang: str, inbound: asyncio.Queue, outbound: asyncio.Queue, stop: asyncio.Event):
        tokenizer = SMALL100Tokenizer.from_pretrained(MODEL_DIR)
        
        word2id = {}
        buffer = []

        deadline = asyncio.get_event_loop().time() + self.lag

        try:
            while not stop.is_set():
                try:
                    ops = await asyncio.wait_for(inbound.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    await asyncio.sleep(0)
                    continue

                for (id, w) in ops["flush"]:
                    word2id[id] = w
                    buffer.append(id)

                for (id, w) in ops["replace"]:
                    if id not in word2id:
                        continue
                    word2id[id] = w

                now = asyncio.get_event_loop().time()
                if now >= deadline or len(buffer) >= self.context_limit:
                    # TODO: corect and translate
                    word2id, buffer, translation = await asyncio.get_running_loop().run_in_executor(
                        self._pool,
                        functools.partial(
                            self._translate, tokenizer,
                            word2id, buffer,
                            src_lang, tgt_lang
                        )
                    )

                    outbound.put_nowait(translation)



        except Exception as e:
            print("Failed: ", e) 
            sys.exit()

    def _translate(self, tokenizer, word2id, buffer, src_lang, tgt_lang):

        try:

            # translation
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = tgt_lang
            
            text = ' '.join(word2id[id] for id in buffer)

            src_ids = tokenizer.encode(text, add_special_tokens=True)
            src_tokens = tokenizer.convert_ids_to_tokens(src_ids)
            tgt_prefix = [tokenizer.lang_code_to_token[tgt_lang]]

            out = self._translate.translate_batch(
            [src_tokens],
            target_prefix=[tgt_prefix],
            beam_size=4
            )

            tgt_tokens  = out[0].hypotheses[0][1:]
            translation = tokenizer.decode(tokenizer.convert_tokens_to_ids(tgt_tokens),
                        skip_special_tokens=True)
            
            # update 2word and reset buffer
            for id in buffer:
                if id in word2id:
                    del word2id[id]

            return word2id, buffer, translation
            
        except Exception as e:
            print("Failed: ", e)

            sys.exit()
        
    

        



    
