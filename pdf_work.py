from dotenv import load_dotenv
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import base64, httpx


from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

def chaining(pages):
    cleaned = [item['cleanImage'] for item in pages['pages_api']]
    os.environ['GOOGLE_API_KEY'] = 'AIzaSyCxBuFSEBJKhPZ6P4JzM46IhfVGzUfnbzU'
    
    class Paper(BaseModel):
        page: Optional[str] = Field(description="page heading")
        summary: Optional[str] = Field(description="summary of page")
        formula: Optional[str] = Field(description="formulas in  LaTeX-style mathematics ONLY")
        diagrams: Optional[str] = Field(description="charts and diagram data as json clearly described")
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    # # message = HumanMessage(
    # #     content=[
    # #         {"type":"text","text":"describe the research in this page and extract the graph values and labels as a json such that it work"},
    # #         {
    # #             "type":"image_data",
    # #             "image_data":{"data":f"data:image/jpeg;base64,{image_data}"}
    # #         },
    # #     ],
    # # )
    # # response = model.invoke([message])

    parser = PydanticOutputParser(pydantic_object=Paper)


    prompt = ChatPromptTemplate.from_messages([
        (
            "system","Return the requested response object in {language}.\n'{format_instructions}'\n"
        ),
        (
            "human", [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                },
            ],
        )
    ])


    all_images = [{"language":"English", 
    "format_instructions": parser.get_format_instructions(),
    "image_data":  data} for data in cleaned]
    chain = prompt | model | parser

    # results = chain.batch(all_images, config={"max_concurrency": 2})
    results = chain.batch(all_images)
    return results