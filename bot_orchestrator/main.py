import uvicorn
from fastapi import FastAPI

# pip freeze > requirements.txt every install
# python -m bot_orchestrator.main to run orchestrator

app = FastAPI()

@app.post("/spawn")
async def spawn():

    '''
    TODO:
    - Spawn a bot process + a bot designated channel
    - Send audio from client via  and send live captions from redis subcription using WS
    '''
    
    
    return {"status": "start running", "bot_id": 1}

@app.post("/kill")
async def kill():
    pass

def main() -> None:
    uvicorn.run("bot_orchestrator.main:app", host="0.0.0.0", port=7001, reload=True)

if __name__ == "__main__":
    main()

