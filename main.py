from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from pymongo import MongoClient
from datetime import datetime
import os

# 🔹 Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")

# 🔹 FastAPI instance
app = FastAPI(title="StudyBot API 🚀", description="AI-powered Study Assistant with memory")

# 🔹 LLM Setup (Updated model)
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b"  # Powerful 120B model
)

# 🔹 MongoDB Setup
client = MongoClient(MONGO_URI)
db = client["studybot"]
collection = db["chat_history"]

# 🔹 Request Model
class ChatRequest(BaseModel):
    message: str

# 🔹 Chat Endpoint
@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message

    # 🔹 Fetch previous chat history (for memory/context)
    past_chats = collection.find().sort("timestamp", 1)
    messages = [{"role": "system", "content": "You are a helpful study assistant."}]

    for chat in past_chats:
        messages.append({"role": "user", "content": chat["student_question"]})
        messages.append({"role": "assistant", "content": chat["bot_answer"]})

    # 🔹 Add current message
    messages.append({"role": "user", "content": user_message})

    # 🔹 Get response from LLM
    response = llm.invoke(messages)
    answer = response.content

    # 🔹 Save chat to MongoDB
    collection.insert_one({
        "student_question": user_message,
        "bot_answer": answer,
        "timestamp": datetime.now()
    })

    # 🔹 Return response
    return {"response": answer}

# 🔹 Optional: Health check endpoint
@app.get("/")
def root():
    return {"message": "StudyBot API is running 🚀"}