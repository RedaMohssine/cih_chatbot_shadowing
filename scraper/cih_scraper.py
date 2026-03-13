"""Scraper principal CIH Bank — télécharge les pages HTML et PDFs."""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re

from config import (
    HEADERS, REQUEST_DELAY, REQUEST_TIMEOUT,
    URLS, PDF_URLS, RAW_DATA_DIR,
)

# Éléments HTML et classes CSS à supprimer avant extraction
REMOVE_TAGS = ['script', 'style', 'noscript', 'nav', 'footer']
REMOVE_CLASSES = [
    'menu', 'header', 'sidebar', 'breadcrumb',
    'cookie', 'popup', 'modal', 'social-share', 'share-buttons',
]
SKIP_TEXTS = {'Accueil', 'Home', 'Menu', 'Fermer', 'Close', 'EN', 'FR', 'AR'}
CONTENT_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'span', 'div', 'td', 'th']


def fetch_page(url):
    """Télécharge une page et retourne le HTML, ou None si erreur."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Erreur: {e}")
        return None


def extract_text(html, url):
    """Extrait le texte structuré d'une page HTML (sans nav/footer/scripts)."""
    soup = BeautifulSoup(html, 'lxml')

    # Titre
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip().replace(" | CIH BANK", "").strip()

    # Supprimer les éléments non désirés
    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()
    for cls in REMOVE_CLASSES:
        for el in soup.find_all(class_=re.compile(cls, re.IGNORECASE)):
            el.decompose()

    main = soup.find('main') or soup.find('article') or soup.find('body')
    if not main:
        return None

    # Extraire le texte en gardant la structure markdown
    for i in range(1, 7):
        for tag in main.find_all(f'h{i}'):
            tag.insert_before(f"\n\n{'#' * i} ")
            tag.insert_after("\n\n")

    for tag in main.find_all('li'):
        tag.insert_before("\n- ")
        tag.insert_after("\n")

    for tag in main.find_all(['p', 'div', 'section', 'article', 'br', 'td', 'th']):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = main.get_text(separator=" ")

    # Nettoyage basique des espaces et sauts de ligne
    text = text.replace('\r', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' \n', '\n', text)
    text = re.sub(r'\n ', '\n', text)
    
    sections = []
    seen = set()
    for line in text.split('\n'):
        line = line.strip()
        if len(line) < 3 and not line.startswith('-'):
            continue
        # Ignorer le texte exact de la navigation s'il est isolé
        clean_text_no_markdown = re.sub(r'^#+\s*|-\s*', '', line).strip()
        if clean_text_no_markdown in SKIP_TEXTS or clean_text_no_markdown in seen:
            continue
        seen.add(clean_text_no_markdown)
        sections.append(line)

    content = "\n".join(sections)
    content = re.sub(r'\n{3,}', '\n\n', content).strip()

    return {"title": title, "url": url, "content": content, "content_length": len(content)}


def download_pdf(url, name):
    """Télécharge un PDF brut dans data/raw/pdfs/. Retourne (ok, size_kb)."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        pdf_dir = os.path.join(RAW_DATA_DIR, 'pdfs')
        os.makedirs(pdf_dir, exist_ok=True)

        with open(os.path.join(pdf_dir, f"{name}.pdf"), 'wb') as f:
            f.write(response.content)

        return True, len(response.content) / 1024
    except Exception as e:
        print(f"   ❌ Erreur PDF {name}: {e}")
        return False, 0


def save_json(data, category, filename):
    """Sauvegarde un document en JSON dans data/raw/<category>/."""
    category_dir = os.path.join(RAW_DATA_DIR, category)
    os.makedirs(category_dir, exist_ok=True)
    with open(os.path.join(category_dir, f"{filename}.json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scrape_all():
    """Orchestre le scraping complet : pages HTML + PDFs."""
    total = sum(len(urls) for urls in URLS.values()) + len(PDF_URLS)
    success, errors = 0, 0
    all_data = []

    print(f"🏦 CIH Bank Scraper — {total} URLs (~{total * REQUEST_DELAY / 60:.0f} min)\n")

    # Pages HTML
    for category, urls in URLS.items():
        print(f"📂 {category.upper()} ({len(urls)} pages)")
        for i, (url, name) in enumerate(urls, 1):
            print(f"   [{i}/{len(urls)}] {name}...", end=" ")

            html = fetch_page(url)
            if not html:
                errors += 1
                continue

            data = extract_text(html, url)
            if not data or data['content_length'] < 10:
                print(f"⚠️ trop court")
                errors += 1
                continue

            data['category'] = category
            data['name'] = name
            save_json(data, category, name)
            all_data.append(data)
            success += 1
            print(f"✅ ({data['content_length']} chars)")
            time.sleep(REQUEST_DELAY)

    # PDFs
    print(f"\n📄 PDFs ({len(PDF_URLS)} documents)")
    for url, name in PDF_URLS:
        print(f"   📄 {name}...", end=" ")
        ok, size_kb = download_pdf(url, name)
        if ok:
            success += 1
            print(f"✅ ({size_kb:.0f} KB)")
        else:
            errors += 1
        time.sleep(REQUEST_DELAY)

    # Rapport
    print(f"\n{'=' * 50}")
    print(f"✅ {success}/{total}  |  ❌ {errors}/{total}")
    print(f"📝 {sum(d['content_length'] for d in all_data):,} caractères")
    print(f"📁 {RAW_DATA_DIR}")

    return all_data


if __name__ == "__main__":
    scrape_all()
