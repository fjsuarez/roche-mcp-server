from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import MCP_ChatBot
from contextlib import asynccontextmanager

# Global chatbot instance
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global chatbot
    chatbot = MCP_ChatBot()
    await chatbot.initialize()
    yield
    # Shutdown
    if chatbot and hasattr(chatbot, 'session') and chatbot.session:
        await chatbot.cleanup()

app = FastAPI(lifespan=lifespan)

# Configure CORS for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    auth_token: str = None

class ChatResponse(BaseModel):
    response: str

class ForecastRequest(BaseModel):
    data: dict

class ForecastResponse(BaseModel):
    forecast: str
    insights: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests from React frontend"""
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        print(f"Received message: {request.message}")
        print(f"Auth token received: {request.auth_token}")
        
        response = await chatbot.process_query(request.message, request.auth_token)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/clear")
async def clear_chat_history():
    """Clear the conversation history"""
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        chatbot.clear_history()
        return {"message": "Chat history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_endpoint(request: ForecastRequest):
    """Handle forecast requests with JSON data"""
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        print(f"Received forecast data: {request.data}")
        
        # Use the specialized forecast method
        result = await chatbot.process_forecast(request.data)
        return ForecastResponse(
            forecast=result["forecast"],
            insights=result["insights"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)