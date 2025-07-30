import asyncio
import uvicorn
from typing import Dict
from fastapi import FastAPI

# pip freeze > requirements.txt every install

bots: Dict[str, asyncio.Event] = {}
lock = asyncio.Lock()


app = FastAPI()

@app.post("/spawn")
async def spawn():
    pass

@app.post("/kill")
async def kill():
    pass

def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=7001, reload=True)

if __name__ == "__main__":
    main()

