from typing import TypedDict,Literal,Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class GraphState(TypedDict):
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

# NODE 1
def researcher_node(state: GraphState) -> dict:
    print("\n🔍 [AGENTE PESQUISADOR]: Buscando dados e tendências sobre o tema...")
   
    tema = state["tema"]

    prompt = f"""
    Você é um Analista de Business Intelligence (BI) e Pesquisador de Mercado Sênior.
    Sua tarefa é coletar insights, dados relevantes, dores do público-alvo e gatilhos mentais sobre o tema: "{tema}".
    
    Traga informações brutas e insights profundos estruturados em tópicos:
    1. Estatísticas ou fatos relevantes (pode estimar baseado no conhecimento do modelo).
    2. Principais dores ou desejos do público sobre isso.
    3. Três palavras-chave ou conceitos que não podem ficar de fora.
    
    Seja puramente analítico e informativo. Não escreva o post final.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"pesquisa": response.text}

