from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

llm = ChatOpenAI(temperature=0.6)
prompt = PromptTemplate(
    input_variables=['phrase'],
    template="Translate the following sentence to Portuguese: {phrase}"
)


@app.post('/translate')
async def translate(request: Request):
    data = await request.json()
    phrase = data.get('phrase')
    if not phrase:
        raise HTTPException(422, 'Phrase to be translated was not provided')
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain({ 'phrase': phrase })
    return JSONResponse({ "translation": response['text'] })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
