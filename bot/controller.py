from service.stt import STT, STTTask
from service.mt import MT, MTTask
from service.tts import TTS, TTSTask
import asyncio


class Controller:
     
    def __init__(self):
        self.inboundQ = asyncio.Queue()
        self.outboundQ = asyncio.Queue()

        self.stt = STT()
        self.stttask: STTTask = None

    # start live captions based off users' speaking language
    async def start_live_captions(self, language: str) -> bool:
        if self.stttask != None:
            self.stttask = self.stt.new_stt_task(language=language)

        return True
    
    async def stop_live_captions(self) -> bool:
        if self.stttask != None:
            self.stttask.stop_evt.set()
            self.stttask = None

        return True

    

    