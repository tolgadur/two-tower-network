from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from inference import Inference

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 1


class QueryResponse(BaseModel):
    documents: list[str]
    similarities: list[float]


def create_endpoints(inference: Inference):
    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest):
        docs, vals = inference.find_nearest_neighbors(query=request.query, k=request.k)
        return QueryResponse(documents=docs, similarities=vals)

    return app
