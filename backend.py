from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader


from dotenv import load_dotenv
import os

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pdf_work import chaining
from url_screenshot import screenshot

app = FastAPI(title="backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; adjust this as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs", status_code=308)

# Load environment variables
load_dotenv()

# Models for API input
class URLInput(BaseModel):
    url: str

class PDFInput(BaseModel):
    pdf: dict #since we will recieve image in base 64

class QuestionInput(BaseModel):
    docs: List[Document]
    question: str


# FastAPI Endpoints
@app.post("/extract_url/")
def extract_url(input: URLInput):
    docs = screenshot(input.url)
    return {
        # "url scrapped info":docs,
        "extracted_information": docs}

@app.post("/extract_pdf/")
def extract_pdf(input: PDFInput):    
    docs = chaining(input.pdf)
    return {
        # "url scrapped info":docs,
        "extracted_information": docs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
