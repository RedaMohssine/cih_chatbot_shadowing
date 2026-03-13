"""Interface Streamlit pour le chatbot CIH Bank."""

import sys
from pathlib import Path

# Ajouter le dossier rag/ au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from chain import ask
from indexer import load_vectorstore, run_indexing
from config import CHROMA_DIR, GROQ_API_KEY

# --- Page config ---
st.set_page_config(
    page_title="CIH Bank Assistant",
    page_icon="🏦",
    layout="centered",
)

# --- CSS personnalisé ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean layout */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 850px;
    }

    /* Minimalist typography */
    .title-text {
        text-align: center;
        color: #1e293b;
        font-weight: 600;
        font-size: 2.2rem;
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }
    
    .made-by {
        text-align: center;
        font-size: 0.95rem;
        color: #64748b;
        margin-top: 5px;
        margin-bottom: 25px;
        font-weight: 500;
    }
    
    .subtitle-text {
        text-align: center;
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 35px;
        line-height: 1.5;
    }

    /* Source tags - clean style */
    .source-tag {
        display: inline-block;
        background: #f1f5f9;
        color: #334155;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8em;
        font-weight: 500;
        margin: 2px;
        border: 1px solid #e2e8f0;
    }

    /* Chat Messages styling - refined and minimal */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 15px;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    /* Avatar tweaks if needed */
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #f58220;
    }
    
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #283a97;
    }

    /* Hide standard Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image(r"c:\Users\Reda\Documents\CIH Shadowing\logo_cih.jpg", use_container_width=True)

st.markdown('<h1 class="title-text">Assistant CIH Bank</h1>', unsafe_allow_html=True)
st.markdown('<p class="made-by">Made by Mohamed Mohssine</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">L\'intelligence artificielle dédiée à vos besoins bancaires, crédits et accompagnement quotidien.</p>', unsafe_allow_html=True)

# --- Vérification API key ---
api_key = GROQ_API_KEY
if not api_key:
    api_key = st.text_input("🔑 Entrez votre clé API Groq :", type="password")
    if api_key:
        import os
        os.environ["GROQ_API_KEY"] = api_key
        st.rerun()
    else:
        st.warning("Veuillez configurer votre clé API Groq (variable d'environnement `GROQ_API_KEY` ou saisie ci-dessus).")
        st.stop()

# --- Vérification base vectorielle ---
if not CHROMA_DIR.exists():
    st.warning("⚠️ La base vectorielle n'existe pas encore.")
    if st.button("🔄 Lancer l'indexation"):
        with st.spinner("Indexation en cours... Cela peut prendre quelques minutes."):
            run_indexing()
        st.success("✅ Indexation terminée !")
        st.rerun()
    st.stop()

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    category = src.get("category", "")
                    title = src.get("title", "Sans titre")
                    url = src.get("url", "")
                    st.markdown(f"- **{title}** ({category})" + (f" — [lien]({url})" if url else ""))

# --- Input utilisateur ---
if question := st.chat_input("Posez votre question..."):
    # Afficher la question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            # Format history for LangChain: list of (role, content) tuples
            chat_history = []
            for m in st.session_state.messages[:-1]: # Exclude the current question we just appended
                role = "human" if m["role"] == "user" else "ai"
                chat_history.append((role, m["content"]))
                
            result = ask(question, chat_history=chat_history)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📚 Sources"):
                for src in result["sources"]:
                    category = src.get("category", "")
                    title = src.get("title", "Sans titre")
                    url = src.get("url", "")
                    st.markdown(f"- **{title}** ({category})" + (f" — [lien]({url})" if url else ""))

    # Sauvegarder dans l'historique
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ À propos")
    st.markdown("""
    Cet assistant utilise les données du site **cihbank.ma** pour répondre à vos questions.

    **Catégories couvertes :**
    - 🧑 Particuliers
    - 💼 Professionnels
    - 🌍 Marocains Du Monde
    - 🏢 Entreprises
    - 🏛️ Institutionnel
    """)

    st.divider()

    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Réindexer les données"):
        with st.spinner("Réindexation en cours..."):
            run_indexing()
        st.success("✅ Réindexation terminée !")
