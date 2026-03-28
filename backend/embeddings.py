import os
from google import genai
from dotenv import load_dotenv


load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_GENERATIVE_AI_API_KEY"))

async def embed_text(text):
    result = await client.aio.models.embed_content(
        model="gemini-embedding-001", 
        contents=text)
    return [e.values for e in result.embeddings]