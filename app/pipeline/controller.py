import asyncio
from tts import TTS, TTSTask
from stt import STT, STTTask
from mt import MT, MTTask
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class Worker:
    caption: Optional[STTTask] = None
    translation: Optional[MTTask] = None
    dubbing: Optional[TTSTask] = None

class Controller:
    def __init__(self, lag=2):
        
        # Initialize layers
        self._stt: STT = STT()
        self._mt: MT = MT()
        self._tts: TTS = TTS()

        # Lag to build context for translation
        self.lag: float = lag

        # Track tasks per peer in the meeting
        self._tasks: Dict[str, Worker] = {}
    
    def _start(self, peerID: str) -> Worker:
        if peerID not in self.tasks:
            self._tasks[peerID] = Worker()

        return self._tasks[peerID]
    
    def start_caption(self, peerID: str, language: str):
        worker = self._start(peerID)
        
        if worker.caption is None:
            worker.caption = self._stt.new_stt_task(language=language)
    
    def start_translation(self, peerID: str, src_lang: str, tgt_lang: str):
        worker = self._start(peerID)

        if worker.caption is None:
            worker.caption = self._stt.new_stt_task(language=src_lang)

        if worker.translation is None:
            captionQ = worker.caption.outboundQ
            worker.translation = self._mt.new_mt_task(inboundQ=captionQ, src_lang=src_lang, tgt_lang=tgt_lang)

        
    def start_dubbing(self, peerID: str, src_lang: str, tgt_lang: str):
        worker = self._start(peerID)
        
        if worker.caption is None:
            worker.caption = self._stt.new_stt_task(language=src_lang)

        if worker.translation is None:
            captionQ = worker.caption.outboundQ
            worker.translation = self._mt.new_mt_task(inboundQ=captionQ, src_lang=src_lang, tgt_lang=tgt_lang)

        if worker.dubbing is None:
            translationQ = worker.translation.outboundQ
            worker.dubbing = self._tts.new_tts_task(inboundQ=translationQ, tgt_lang=tgt_lang)

    
    def stop_caption(self, peerID: str):
        if peerID not in self._tasks:
            return

        worker = self._tasks[peerID]
        if worker.caption:
            worker.caption.stop_evt.set()
        
    def stop_translation(self, peerID: str):
        if peerID not in self._tasks:
            return

        worker = self._tasks[peerID]
        if worker.translation:
            worker.translation.stop_evt.set()

    def stop_dubbing(self, peerID: str):
        if peerID not in self._tasks:
            return

        worker = self._tasks[peerID]
        if worker.dubbing:
            worker.dubbing.stop_evt.set()

    def stop_all_tasks(self, peerID: str):
        self.stop_dubbing(peerID)
        self.stop_translation(peerID)
        self.stop_caption(peerID)



#  open mic and test
async def test():
    pass

if __name__ == "__main__":
    asyncio.run(test)