from typing import TypedDict,Literal,Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class Graphstate(TypedDict):
    tema: str
    pesquisa: Optional[str]
    post_final: Optional[str]
    feedback: Optional[str]
    proxima_acao: Optional[Literal["pesquisar", "redigir", "finalizar"]]

class SupervisorDecision(BaseModel):
    
    proxima_acao: Literal["pesquisar","redigir", "finalizar"] = Field(
        description="Escolha 'pesquisar' se precisar de dados do tema, 'redigir' para criar ou corrigir o post, ou 'finalizar' se o post estiver perfeito."
    )

    justificativa: str = Field(description="Breve razão de ter escolhido essa ação.")