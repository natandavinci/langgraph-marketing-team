from typing import TypedDict,Literal,Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END
import os
import json
from dotenv import load_dotenv
from langchain_core.runnables.graph import MermaidDrawMethod
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



def supervisor_node(state:GraphState) -> dict:
    print("\n👑 [AGENTE SUPERVISOR]: Avaliando o trabalho do time...")
    tema = state["tema"]
    pesquisa = state.get("pesquisa","Nenhuma pesquisa feita ainda.")
    post_final = state.get("post_final", "Nenhum post escrito ainda.")

    prompt = f"""
    Você é o Diretor de Criação e Supervisor de uma agência de Marketing Digital de Elite.
    Sua função é avaliar o progresso do time e decidir estritamente qual deve ser o próximo passo do fluxo de trabalho.
    
    Tema do Projeto: "{tema}"
    
    Status Atual do Trabalho:
    - Pesquisa Coletada: {pesquisa}
    - Última Versão do Post Criada: {post_final}
    
    Suas Opções de Próxima Ação:
    - "pesquisar": Escolha isso se a pesquisa atual estiver vazia, incompleta ou se você achar que faltam dados cruciais para o tema.
    - "redigir": Escolha isso se o post ainda não foi escrito OU se você leu o post atual e achou que ele pode ser melhorado (nesse caso, você DEVE preencher a justificativa com o que o redator precisa corrigir).
    - "finalizar": Escolha isso APENAS se o post estiver perfeito, magnético, sem erros e pronto para publicação no LinkedIn.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SupervisorDecision
        )
    )

    decisao = json.loads(response.text)

    print(f"➡️ [DECISÃO DO SUPERVISOR]: Próxima Ação -> {decisao['proxima_acao'].upper()}")
    print(f"💬 [JUSTIFICATIVA]: {decisao['justificativa']}")

    feedback_to_team = decisao["justificativa"] if decisao["proxima_acao"] == "redigir" else None

    return {
        "proxima_acao": decisao["proxima_acao"],
        "feedback": feedback_to_team
    }

def router_supervisor(state: GraphState):
    return state["proxima_acao"]

# Creating graph
graph = StateGraph(GraphState)

graph.add_node("researcher_node",
               researcher_node)

graph.add_node("writer_node",
               writer_node)

graph.add_node("supervisor_node",
               supervisor_node)

# EDGES

graph.set_entry_point("researcher_node")

graph.add_edge("researcher_node",
               "writer_node")

graph.add_edge("writer_node",
               "supervisor_node")

graph.add_conditional_edges("supervisor_node",
                            router_supervisor,
                            {
                                "pesquisar": "researcher_node",
                                "redigir": "writer_node",
                                "finalizar": END
                            }
                            )


app = graph.compile()

# TESTE DO GRAFO COMPLETO
if __name__ == "__main__":
    png_bytes = app.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API
    )

    with open("grafo_exemplo1.png", "wb") as f:
        f.write(png_bytes)

    estado_inicial: GraphState = {
        "tema": "O impacto da Inteligência Artificial no mercado de Advocacia em 2026",
        "pesquisa": None,
        "post_final": None,
        "feedback": None,
        "proxima_acao": None
    }

    print("🔥 Iniciando a Agência Multi-Agente de Marketing...")
    
    # Executamos o grafo e acompanhamos as entregas
    for evento in app.stream(estado_inicial, stream_mode="values"):
        if evento.get("post_final") and evento.get("proxima_acao") == "finalizar":
            print("\n🏆 [SISTEMA]: Post Aprovado pelo Supervisor e Pronto para Publicação!")
            print("-" * 60)
            print(evento["post_final"])
            print("-" * 60)
            break