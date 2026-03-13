"""Nettoyage des données brutes et construction du corpus RAG."""

import json
import os
import re
import glob

from config import RAW_DATA_DIR, CLEANED_DATA_DIR

# Résidus de navigation à supprimer
NAV_PATTERNS = [
    r'^Accueil$', r'^Home$', r'^Menu$', r'^Toggle navigation$',
    r'^Rechercher$', r'^Search$', r'^Fermer$', r'^Close$',
    r'^En savoir \+$', r'^En savoir plus$', r'^Lire la suite$',
    r'^[×›‹»«]$', r'^Voir (tout|plus)$', r'^Afficher plus$',
]
NAV_RE = [re.compile(p, re.IGNORECASE) for p in NAV_PATTERNS]


def load_raw_data():
    """Charge tous les .json depuis data/raw/ (récursif)."""
    pattern = os.path.join(RAW_DATA_DIR, "**", "*.json")
    files = sorted(glob.glob(pattern, recursive=True))
    print(f"📂 {len(files)} fichiers trouvés")

    docs = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                docs.append(json.load(fp))
        except Exception as e:
            print(f"   ⚠️ {f}: {e}")
    return docs


def clean_text(text):
    """Nettoie un texte : supprime nav résiduelle, normalise espaces/sauts de ligne."""
    if not text:
        return ""

    # Supprimer les lignes de navigation
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            lines.append("")
        elif not any(p.match(line) for p in NAV_RE):
            lines.append(line)

    text = '\n'.join(lines)
    text = text.replace('\t', ' ')
    text = re.sub(r' {2,}', ' ', text)       # espaces multiples → un seul
    text = re.sub(r'\n{3,}', '\n\n', text)    # 3+ sauts → 2
    return text.strip()


def deduplicate(docs):
    """Supprime les documents dont les 500 premiers chars sont identiques."""
    seen = set()
    unique = []
    for doc in docs:
        fingerprint = doc.get('content', '')[:500]
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(doc)

    removed = len(docs) - len(unique)
    if removed:
        print(f"   🔄 {removed} doublons supprimés")
    return unique


def build_corpus(docs):
    """Construit le corpus JSON final prêt pour le RAG."""
    corpus = {
        "metadata": {
            "source": "cihbank.ma",
            "total_documents": len(docs),
            "total_characters": sum(len(d.get('content', '')) for d in docs),
            "categories": sorted(set(d.get('category', 'unknown') for d in docs)),
        },
        "documents": [
            {
                "id": f"cih_{doc.get('category', 'unknown')}_{i:03d}",
                "title": doc.get('title', ''),
                "category": doc.get('category', 'unknown'),
                "source_url": doc.get('url', ''),
                "content": doc.get('content', ''),
                "content_length": len(doc.get('content', '')),
                "name": doc.get('name', ''),
                "type": "webpage",
            }
            for i, doc in enumerate(docs)
        ],
    }
    return corpus


def clean_all():
    """Pipeline complet : charger → nettoyer → dédupliquer → filtrer → sauvegarder."""
    print("🧹 Nettoyage CIH Bank\n")

    # 1. Charger
    docs = load_raw_data()
    if not docs:
        print("❌ Aucun document. Lance d'abord cih_scraper.py")
        return None

    # 2. Nettoyer
    for doc in docs:
        doc['content'] = clean_text(doc.get('content', ''))
        doc['content_length'] = len(doc['content'])
    print(f"   ✅ {len(docs)} nettoyés")

    # 3. Dédupliquer
    docs = deduplicate(docs)

    # 4. Filtrer les contenus trop courts
    before = len(docs)
    docs = [d for d in docs if d.get('content_length', 0) >= 50]
    removed = before - len(docs)
    if removed:
        print(f"   🗑️ {removed} trop courts supprimés")

    # 5. Construire et sauvegarder
    corpus = build_corpus(docs)
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    path = os.path.join(CLEANED_DATA_DIR, "cih_bank_corpus.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    m = corpus['metadata']
    print(f"\n📊 {m['total_documents']} docs  |  {m['total_characters']:,} chars")
    print(f"📂 {', '.join(m['categories'])}")
    print(f"💾 {path}")

    return corpus


if __name__ == "__main__":
    clean_all()
