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



def writer_node(state:GraphState) -> dict:
    print("\n✍️ [AGENTE REDATOR]: Transformando dados em um post de alto impacto...")
    tema = state["tema"]
    pesquisa = state["pesquisa"]
    feedback_supervisor = state.get("feedback")

    if feedback_supervisor:
        prompt = f"""
        Você é um Copywriter Sênior especialista em LinkedIn.
        Você criou um post sobre o tema "{tema}", mas o Supervisor rejeitou com o seguinte feedback:
        
        "{feedback_supervisor}"
        
        Aqui está a versão que você escreveu anteriormente:
        ---
        {state["post_final"]}
        ---
        
        Reescreva o post aplicando as correções cirurgicamente, mantendo o tom profissional, magnético e focado em engajamento.
        """

    else:
        prompt = f"""
        Você é um Copywriter Sênior especialista em posts de alta performance para o LinkedIn.
        Sua tarefa é transformar o relatório de pesquisa bruta abaixo em um post magnético.
        
        Tema: "{tema}"
        
        Dados da Pesquisa de Mercado:
        ---
        {pesquisa}
        ---
        
        Diretrizes do Post:
        1. **Gancho (Hook) Forte:** Comece com uma frase que faça o advogado parar o feed.
        2. **Conteúdo:** Use os dados de impacto e conceitos (como Augmentação Jurídica) que a pesquisa trouxe.
        3. **Formatação:** Use frases curtas, espaçadas (fácil leitura no celular) e bullet points bem organizados.
        4. **CTA (Chamada para Ação):** Termine com uma pergunta instigante para gerar comentários.
        5. Use no máximo 3 a 5 hashtags relevantes ao final.
        """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {"post_final": response.text, "feedback": None}

# TESTE
if __name__ == "__main__":
    # 1. Criamos um estado inicial simulando a entrada do usuário
    estado_teste: GraphState = {
        "tema": "O impacto da Inteligência Artificial no mercado de Advocacia em 2026",
        "pesquisa": None,
        "post_final": None,
        "feedback": None,
        "proxima_acao": None
    }

   # 1. Roda o Pesquisador
    resultado_pesquisa = researcher_node(estado_teste)
    estado_teste["pesquisa"] = resultado_pesquisa["pesquisa"]
    
    print("\n--- ✅ Pesquisa Concluída! Passando dados para o Redator... ---")

    # 2. Roda o Redator passando o estado atualizado com a pesquisa
    resultado_redator = writer_node(estado_teste)
    
    print("\n🚀 POST GERADO PELO REDATOR PARA O LINKEDIN:")
    print("-" * 60)
    print(resultado_redator["post_final"])
    print("-" * 60)