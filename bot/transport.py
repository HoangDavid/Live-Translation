import redis.asyncio as redis
import os, asyncio
from controller import Controller
from typing import Literal

# Listen and Push to Redis
REDIS_URL = os.getenv("REDIS_URL")
BOT_CHAN_IN = os.getenv("BOT_CHAN_IN")
BOT_CHAN_OUT = os.getenv("BOT_CHAN_OUT")


async def listener(c: Controller):
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


async def sender(c: Controller):
    r = redis.from_url(REDIS_URL, decode_responses=True)
    try:
        while True:
            asyncio.sleep(0)
    finally:
        await r.close()


async def main():
    controller = Controller()
    await asyncio.gather(listener(c=controller), sender(c=controller))


if __name__ == "__main__":
    main()
    
    


