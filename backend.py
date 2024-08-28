from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain_pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import LLMChain

from dotenv import load_dotenv
import os

from pinecone import Pinecone as pc
from pinecone import PodSpec

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(title="backend API")

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs", status_code=308)

# Load environment variables
load_dotenv()

# Models for API input
# Models for API input
class URLInput(BaseModel):
    url: str

class QuestionInput(BaseModel):
    docs: List[Document]
    question: str

def url_loader(user_input):
    loader = WebBaseLoader(user_input)
    docs = loader.load()
    return docs

def extractor(docs):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, top_p=0.85)
    
    llm_prompt_template = """You are an assistant designed to summarize articles into their key components. Your output should be well-organized in HTML format, with each key component sorted in order. If the article contains diagrams or visual elements, ensure to include them as descriptions within the HTML structure.

Here’s what you need to do:
1. Summarize the article into its main sections: Introduction, Main Content, Diagrams (if any), and Conclusion.
2. For each section, create a corresponding HTML tag.
3. If the article includes diagrams or visuals, describe them in a `<figure>` tag with a `<figcaption>` for the description.
4. Ensure the HTML structure is properly ordered and formatted.

Input Article: {context}

Output the summarized article in the following HTML format:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Summarized Article</title>
</head>
<body>
    <section>
        <h1>Introduction</h1>
        <p>{Introduction}</p>
    </section>
    <section>
        <h2>Main Content</h2>
        <p>{MainContent}</p>
    </section>
    {if_diagrams_present}
    <section>
        <h3>Diagrams</h3>
        <figure>
            <img src="{diagram_source}" alt="Diagram description">
            <figcaption>{diagram_caption}</figcaption>
        </figure>
    </section>
    {/if_diagrams_present}
    <section>
        <h4>Conclusion</h4>
        <p>{Conclusion}</p>
    </section>
</body>
</html>
"""

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)
    chain = LLMChain(llm=llm, prompt=llm_prompt)

    response = chain.run({"context": docs})
    return response

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def pine_index(docs, gemini_embeddings):
    pine_client = pc()
    index_name = "langchain-demo"
    if index_name not in pine_client.list_indexes().names():
        print("Creating index")
        pine_client.create_index(name=index_name,
                                 metric="cosine",
                                 dimension=768,
                                 spec=PodSpec(
                                     environment="gcp-starter",
                                     pod_type="starter",
                                     pods=1)
        )
        print(pine_client.describe_index(index_name))

    vectorstore = Pinecone.from_documents(docs, gemini_embeddings, index_name=index_name)
    retriever = vectorstore.as_retriever()
    return retriever

def gemini(retriever, question):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, top_p=0.85)
    llm_prompt_template = """You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:"""

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)

# FastAPI Endpoints
@app.post("/extract/")
def extract_information(input: URLInput):
    docs = url_loader(input.url)
    result = extractor(docs)
    return {"extracted_information": result}

# @app.post("/ask-question/")
# def ask_question(input: QuestionInput):
#     try:
        # gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # retriever = pine_index(input.docs, gemini_embeddings)
        # answer = gemini(retriever, input.question)
        # return {"answer": answer}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
