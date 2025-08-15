from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

class QueryRequest(BaseModel):
    query: str

class QueryAnswer(BaseModel):
    answer: str
    sources: list[str]


app = FastAPI(title='Policy document - QueryBot')

embedder = OllamaEmbeddings(model='nomic-embed-text')

vector_db = FAISS.load_local('policy-index', embedder, allow_dangerous_deserialization=True)

retriever = vector_db.as_retriever(
    search_kwargs={'k': 3}
)

llm = Ollama(
    model = 'deepseek-r1:1.5b'
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Simple concatenation of chunks
    retriever=retriever,
    return_source_documents=True  # We want the source texts
)

@app.post('/query', response_model= QueryAnswer)
async def query_policy(request: QueryRequest):
    try:
        # 1. Execute RAG Chain
        result = rag_chain.invoke({"query": request.query})
        
        # 2. Format Response
        return QueryAnswer(
            answer=result["result"],
            sources=[doc.page_content[:200] + "..." for doc in result["source_documents"]]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "llm": "llama3", "embedder": "nomic-embed-text"}

