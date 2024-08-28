from langchain import hub
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone
from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv
import os

from pinecone import Pinecone as pc
from pinecone import PodSpec


def url(user_input):
    # when URL
    loader = WebBaseLoader(user_input)
    docs = loader.load()
    return docs
def extractor(docs):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, top_p=0.85)
    
    llm_prompt_template = """You are an assistant designed to extract specific information from text.
    The information to be extracted includes:
    - Name
    - Description
    - Subject
    - Department
    - University Name

    Extract the relevant information from the provided context. If any information is not available, mention "Not found."

    Context: {context}

    Extracted Information:
    Name: 
    Description: 
    Subject: 
    Department: 
    University Name:"""

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)
    chain = LLMChain(llm=llm, prompt=llm_prompt)

    # Execute the chain with the provided documents
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
    # test vector store
    retriever = vectorstore.as_retriever()
    print(len(retriever.invoke("MMLU")))
    
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
    print(rag_chain.invoke(question))


if __name__ == "__main__":
    user_input = input("Enter query:  ")
    
    docs = url(user_input)
    
    load_dotenv()

    os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
    os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

    # gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # question = input("What is your query:  ")
    # retriever = pine_index(docs, gemini_embeddings)
    # gemini(retriever, question)

    print(extractor(docs))