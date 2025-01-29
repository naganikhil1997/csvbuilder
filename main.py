import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import pandas as pd
import numpy as np
from fastapi.responses import StreamingResponse
import asyncio
import ast
import logging
from dotenv import load_dotenv
# hello
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='chatbot.log',
                    filemode='a')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except Exception as e:
        logger.error(f"Error parsing embedding: {e}")
        return []

try:
    df = pd.read_csv('csvconverter.csv')
    df['ada_embedding'] = df.ada_embedding.apply(safe_eval).apply(np.array)
    logger.info("Loaded existing embedded documents")
except FileNotFoundError:
    df = pd.DataFrame(columns=['content', 'ada_embedding'])
    logger.info("No existing embedded documents found, starting with an empty dataframe")
except Exception as e:
    logger.error(f"Error loading embedded documents: {e}")
    df = pd.DataFrame(columns=['content', 'ada_embedding'])
    logger.info("Starting with an empty dataframe due to loading error")

chat_history = []

# Function to get embeddings
# async def get_embedding(content):
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")  
    embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    return np.array(embedding)

def cosine_similarity(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0  
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class TextUpload(BaseModel):
    content: str

@app.post("/upload")
async def upload_text(text_upload: TextUpload):
    content = text_upload.content
    embedding = get_embedding(content)
    
    new_row = pd.DataFrame({'content': [content], 'ada_embedding': [embedding]})
    global df
    df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_csv('csvconverter.csv', index=False)
    
    return {"message": "Text uploaded and embedded successfully"}

async def generate_response(message: str):
    global chat_history
    chat_history.append(message)
    logger.info(f"Human Question: {message}")
    print(f"Human Question: {message}")
    query_embedding = get_embedding(message)

    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, query_embedding))

    similar_docs = df.sort_values('similarities', ascending=False).head(3)
    context = " ".join(similar_docs['content'].tolist())
    print(f"Retrieved Context: {context}")
    
    system_message = f"""
    You are an AI assistant for a food application that provides detailed recipe information based on ingredients available in a warehouse. Your role is to assist users by suggesting recipes using the ingredients stored in a CSV file.
 Here are some key points about the food application:
- The CSV file contains details of various ingredients available in the warehouse, including their names, quantities, and categories.
- Users can inquire about recipes that can be made using available ingredients, required quantities, cooking methods, and estimated nutritional values.
- Each recipe suggestion includes the dish name, ingredients list, preparation instructions, estimated cost, calorie count, and serving size.
- If an ingredient is missing, you can suggest alternative ingredients or provide recommendations for purchasing.

    Retrieved Context: {context}
    Human's Chat History (previous questions):
    {chat_history}

    Please answer in a professional, concise, and informative tone using the provided context.
    """

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ],
        stream=True
    )


    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield f"data: {chunk.choices[0].delta.content}\n\n"
        await asyncio.sleep(0.1)

    yield "data: [DONE]\n\n"

@app.get("/chat")
async def chat(request: Request):
    message = request.query_params.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="Message parameter is required")
    
    return StreamingResponse(generate_response(message), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)