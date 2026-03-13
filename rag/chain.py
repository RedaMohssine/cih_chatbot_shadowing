"""RAG Chain : assemble le retriever + LLM Groq (Llama 3.3 70B) pour répondre aux questions."""

import logging

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retriever import retrieve, format_context
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def get_llm() -> ChatGroq:
    """Initialise le LLM Groq."""
    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt template
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """Contexte :
{context}

---

Question : {question}"""),
])


def ask(question: str, chat_history: list = None) -> dict:
    """Pose une question au RAG et retourne la réponse + sources, en tenant compte de l'historique."""
    if chat_history is None:
        chat_history = []

    # 1. Récupérer les documents pertinents
    docs = retrieve(question)

    if not docs:
        return {
            "answer": "Je n'ai trouvé aucune information pertinente dans ma base de données. "
                      "Je vous invite à contacter votre conseiller CIH Bank.",
            "sources": [],
        }

    # 2. Formater le contexte
    context = format_context(docs)

    # 3. Générer la réponse
    llm = get_llm()
    chain = PROMPT | llm | StrOutputParser()
    try:
        answer = chain.invoke({
            "context": context, 
            "question": question,
            "chat_history": chat_history
        })
    except Exception as e:
        error_msg = str(e)
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            answer = ("⚠️ **Quota API épuisé.** "
                      "Veuillez réessayer plus tard.\n\n"
                      "En attendant, voici les sources trouvées dans la base CIH Bank (voir ci-dessous).")
        else:
            logger.error("Erreur LLM: %s", error_msg)
            answer = f"❌ Erreur lors de la génération de la réponse : {error_msg[:200]}"

    # 4. Extraire les sources
    sources = [
        {
            "title": doc.metadata.get("title", "Sans titre"),
            "category": doc.metadata.get("category", ""),
            "url": doc.metadata.get("source_url", ""),
        }
        for doc in docs
    ]

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    # Test rapide
    questions = [
        "Quels sont les avantages de la carte Visa Gold ?",
        "Quelle est la différence entre CODE30 et CODE60 ?",
        "Je suis MRE en France, quelle offre me convient ?",
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = ask(q)
        print(f"R: {result['answer'][:500]}...")
        print(f"Sources: {[s['title'] for s in result['sources']]}")
