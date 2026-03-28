import os
import sys
import asyncio
sys.path.append(os.getcwd())
from pypdf import PdfReader
from db import engine, Document
from embeddings import embed_text
from langchain_text_splitters import RecursiveCharacterTextSplitter

async def main():
    chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for f in [f for f in os.listdir("data") if f.endswith(".pdf")]:
        text = "".join([p.extract_text() for p in PdfReader(f"data/{f}").pages if p.extract_text()])
        chunks = chunker.split_text(text)
        vectors = await embed_text(chunks)
        async with engine.begin() as conn:
            for c, v in zip(chunks, vectors):
                await conn.execute(Document.__table__.insert().values(content=c, embedding=v))

if __name__ == "__main__":
    asyncio.run(main())
