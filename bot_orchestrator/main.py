import uvicorn
from fastapi import FastAPI

# pip freeze > requirements.txt every install
# python -m bot_orchestrator.main to run orchestrator

app = FastAPI()

@app.post("/spawn")
async def spawn():
    return {"status": "start running", "bot_id": 1}

@app.post("/kill")
async def kill():
    pass

def main() -> None:
    uvicorn.run("bot_orchestrator.main:app", host="0.0.0.0", port=7001, reload=True)

if __name__ == "__main__":
    main()

