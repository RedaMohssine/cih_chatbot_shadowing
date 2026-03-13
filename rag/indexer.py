"""Indexation : charge le corpus, nettoie, découpe en chunks, embed localement, stocke dans ChromaDB."""

import json
import re
import shutil
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    CORPUS_PATH,
    PDFS_MD_DIR,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    COLLECTION_NAME,
)


def clean_content(text: str) -> str:
    """Nettoie le contenu brut en supprimant la navigation et le boilerplate du site CIH."""
    # Supprimer le header de navigation principal (du début jusqu'à "Comment ca marche ?")
    text = re.sub(
        r'^.*?Comment ca marche \?',
        '',
        text,
        count=1,
        flags=re.DOTALL,
    )
    # Supprimer les menus secondaires résiduels
    text = re.sub(
        r'(Ouvrir un compte\s*)?CIH Online\s*(Corporate\s*)?L\'agence la plus proche\s*Simuler un crédit\s*Nos partenaires\s*Comment ca marche \?',
        '',
        text,
    )
    text = re.sub(
        r'CIH Online\s*Centres d\'Affaires\s*Notice Crédit\s*Nos Innovations\s*Comment ca marche \?',
        '',
        text,
    )
    # Supprimer le footer récurrent
    text = re.sub(
        r'Téléchargez l\'application CIH BANK.*$',
        '',
        text,
        flags=re.DOTALL,
    )
    # Nettoyer les espaces multiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def load_corpus(corpus_path: Path = CORPUS_PATH) -> list[Document]:
    """Charge le corpus JSON nettoyé et les PDFs parsés en .md."""
    documents = []

    # --- Corpus JSON (pages web) ---
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for doc in data["documents"]:
        metadata = {
            "id": doc["id"],
            "title": doc["title"],
            "category": doc["category"],
            "source_url": doc.get("source_url", ""),
            "source_type": doc.get("type", "webpage"),
        }
        documents.append(Document(page_content=clean_content(doc["content"]), metadata=metadata))

    # --- PDFs parsés (.md) ---
    if PDFS_MD_DIR.exists():
        for md_file in PDFS_MD_DIR.glob("*.md"):
            text = md_file.read_text(encoding="utf-8")
            if len(text.strip()) < 50:
                continue
            metadata = {
                "id": f"pdf_{md_file.stem}",
                "title": md_file.stem.replace("-", " ").title(),
                "category": "institutionnel",
                "source_url": "",
                "source_type": "pdf",
            }
            documents.append(Document(page_content=text, metadata=metadata))

    print(f"✅ {len(documents)} documents chargés ({len(data['documents'])} web + {len(documents) - len(data['documents'])} PDFs)")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Découpe les documents en chunks avec overlap. Préfixe chaque chunk avec ses métadonnées."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Préfixer chaque chunk avec catégorie + titre pour améliorer la recherche vectorielle
    for chunk in chunks:
        meta = chunk.metadata
        prefix = f"[{meta.get('category', '').upper()}] {meta.get('title', '')}\n"
        chunk.page_content = prefix + chunk.page_content

    print(f"✅ {len(chunks)} chunks créés (taille={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """Crée le modèle d'embeddings local."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vectorstore(chunks: list[Document]) -> Chroma:
    """Embed les chunks localement et stocke dans ChromaDB (pas de rate limit)."""
    embeddings = get_embeddings()

    # Nettoyer une éventuelle ancienne base
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    print(f"✅ Base vectorielle créée dans {CHROMA_DIR} ({len(chunks)} vecteurs)")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Charge une base vectorielle existante."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def run_indexing():
    """Pipeline complet d'indexation."""
    print("🔄 Démarrage de l'indexation...")
    documents = load_corpus()
    chunks = chunk_documents(documents)
    vectorstore = create_vectorstore(chunks)
    print("🎉 Indexation terminée !")
    return vectorstore


if __name__ == "__main__":
    run_indexing()
