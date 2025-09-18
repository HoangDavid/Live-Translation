import redis.asyncio as redis
import os, asyncio
from controller import Controller
from typing import Literal

# Listen and Push to Redis
REDIS_URL = os.getenv()
BOT_CHAN_IN = os.getenv()
BOT_CHAN_OUT = os.getenv()


async def listen(c: Controller):
    r = redis.from_url(REDIS_URL, decode_responses=True)
    ps = r.pubsub()

    # Subcribe to data channel from user
    await ps.subscribe(BOT_CHAN_IN)

    try:
        async for msg in ps.listen():
            pass

    finally:
        await ps.close()
        await r.close()


async def publish(c: Controller):
    r = redis.from_url(REDIS_URL, decode_responses=True)
    try:
        pass
        # while True:
        #     asyncio.sleep(0)
    finally:
        await r.close()


async def main():
    controller = Controller()
    
    


