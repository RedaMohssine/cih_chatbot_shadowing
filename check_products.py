import json
data = json.load(open("data/cleaned/cih_bank_corpus.json", "r", encoding="utf-8"))
for doc in data["documents"]:
    title = doc["title"].lower()
    if any(k in title for k in ["code 30", "code30", "code 18", "code18", "code 60", "code60", "code 212", "pack intilak", "offres packag"]):
        print(f"\n=== {doc['title']} | {doc['category']} ===")
        # Find the actual product description (skip first 300 chars of nav)
        content = doc["content"]
        # Find where the product name appears
        for keyword in [doc["title"], "CODE"]:
            idx = content.find(keyword, 100)  # skip header
            if idx > 0:
                print(content[idx:idx+500])
                break
        else:
            print(content[200:700])
