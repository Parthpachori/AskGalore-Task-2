import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from graph import create_assistant_graph
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Healthcare Physiotherapy Assistant API")

# Initialize the LangGraph
assistant = create_assistant_graph()

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_user"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

static_path = os.path.join(os.path.dirname(__file__), "..", "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def root():
    static_html = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    return FileResponse(static_html)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Configuration for LangGraph persistence
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Prepare the state
        inputs = {
            "messages": [HumanMessage(content=request.message)]
        }
        
        # Invoke the graph
        result = assistant.invoke(inputs, config=config)
        
        return ChatResponse(
            response=result["response"],
            thread_id=request.thread_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
