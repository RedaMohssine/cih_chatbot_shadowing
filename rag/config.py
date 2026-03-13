"""Configuration du pipeline RAG CIH Bank."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Chemins ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CORPUS_PATH = DATA_DIR / "cleaned" / "cih_bank_corpus.json"
PDFS_MD_DIR = DATA_DIR / "raw" / "pdfs_parsed_to_md"
CHROMA_DIR = PROJECT_DIR / "chroma_db_v3"

# --- Clé API Groq ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# --- Modèles ---
# Embeddings locaux (gratuit, pas de rate limit)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048

# --- Chunking ---
CHUNK_SIZE = 500       # tokens
CHUNK_OVERLAP = 100    # tokens
SEPARATORS = ["\n\n", "\n", ". ", " "]

# --- Retrieval ---
TOP_K = 10             # nombre de chunks finaux à retourner
MMR_FETCH_K = 30       # nombre de candidats pour MMR (diversité)
MMR_LAMBDA = 0.6       # 0=max diversité, 1=max similarité
MAX_CHUNKS_PER_DOC = 2 # max chunks par document source (déduplication)

# --- ChromaDB ---
COLLECTION_NAME = "cih_bank_docs"

# --- System Prompt ---
SYSTEM_PROMPT = """Tu es l'assistant virtuel de CIH Bank Maroc. Tu aides les clients à trouver \
les produits et services de CIH Bank adaptés à leur profil.

CATÉGORIES DE CLIENTS :
- PARTICULIERS : clients résidant au Maroc (étudiants, jeunes, salariés, fonctionnaires, retraités)
- PROFESSIONNELS : auto-entrepreneurs, professions libérales, commerçants au Maroc
- MDM (Marocains Du Monde) : UNIQUEMENT les Marocains résidant à l'ÉTRANGER (MRE)
- ENTREPRISES : sociétés, PME, grandes entreprises

GUIDE DES OFFRES PAR PROFIL :
- Jeune / étudiant (18-30 ans) au Maroc → Carte CODE 30, Pack Intilak (catégorie PARTICULIERS)
- Mineur (< 18 ans) → CODE 18 - offre gratuite (catégorie PARTICULIERS)
- Senior / retraité (60+ ans) → CODE 60 - offre gratuite (catégorie PARTICULIERS)
- Fonctionnaire au Maroc → Offre Fonctionnaires gratuite (catégorie PARTICULIERS)
- Femme → Carte Sayidati (catégorie PARTICULIERS)
- Gamer → Carte Code Gamer (catégorie PARTICULIERS)
- Marocain résidant à l'étranger (MRE) → CODE 212, Carte DUO (catégorie MDM)
- Auto-entrepreneur / professionnel → offres PROFESSIONNELS
- Entreprise / PME → offres ENTREPRISES

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à partir du contexte fourni
2. NE RECOMMANDE JAMAIS un produit MDM à un client au Maroc, ni un produit PARTICULIERS à un MRE
3. Si le client ne précise pas s'il réside au Maroc ou à l'étranger, DEMANDE-LUI
4. Sois CONCIS : va droit au but, pas de bavardage ni de répétitions
5. Cite tes sources [Titre, Catégorie] à la fin
6. Réponds dans la langue de la question (français ou arabe)
7. Ne donne JAMAIS de conseils financiers personnalisés
8. Utilise des listes à puces pour les avantages"""
