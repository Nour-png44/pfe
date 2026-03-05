import json
import os
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

# ─── Configuration ────────────────────────────────────────────────────────────
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB", "easybulk")
COLLECTION = "user_contact"
JSON_FILE  = "data/user_contact.json"  # chemin vers votre fichier

# ─── Connexion MongoDB ─────────────────────────────────────────────────────────
client = MongoClient(MONGO_URI)
db     = client[MONGO_DB]
col    = db[COLLECTION]


# ─── 1. Chargement du fichier JSON (format NDJSON : 1 doc par ligne) ──────────
def load_json_file(filepath: str) -> list[dict]:
    docs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                # Convertir l'ObjectId MongoDB exporté {"$oid": "..."} en string
                if isinstance(doc.get("_id"), dict) and "$oid" in doc["_id"]:
                    doc["_id"] = doc["_id"]["$oid"]
                if isinstance(doc.get("campagne_ref"), dict) and "$oid" in doc.get("campagne_ref", {}):
                    doc["campagne_ref"] = doc["campagne_ref"]["$oid"]
                docs.append(doc)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Ligne {i} ignorée : {e}")
    return docs


# ─── 2. Import dans MongoDB ────────────────────────────────────────────────────
def import_to_mongo(docs: list[dict]):
    if not docs:
        print("Aucun document à importer.")
        return

    # Vider la collection avant import (optionnel)
    col.drop()

    result = col.insert_many(docs, ordered=False)
    print(f"  ✅ {len(result.inserted_ids)} documents importés dans '{COLLECTION}'")

    # Créer des index utiles
    col.create_index([("msisdn", ASCENDING)])
    col.create_index([("campagne_id", ASCENDING)])
    col.create_index([("status", ASCENDING)])
    col.create_index([("status_last_updated_at", ASCENDING)])
    print("  ✅ Index créés : msisdn, campagne_id, status, status_last_updated_at")


# ─── 3. Statistiques de base ───────────────────────────────────────────────────
def print_stats():
    total = col.count_documents({})
    print(f"\n📊 Statistiques collection '{COLLECTION}':")
    print(f"   Total documents  : {total}")

    # Répartition par statut
    pipeline_status = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    print("   Statuts SMS :")
    status_labels = {1: "Envoyé", 2: "Échoué", 3: "En attente", 8: "En cours"}
    for r in col.aggregate(pipeline_status):
        label = status_labels.get(r["_id"], "Inconnu")
        print(f"     - Statut {r['_id']} ({label}) : {r['count']}")

    # Répartition par encoding
    pipeline_enc = [
        {"$group": {"_id": "$encoding", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    print("   Encodings :")
    for r in col.aggregate(pipeline_enc):
        print(f"     - {r['_id']} : {r['count']}")

    # Coût total
    pipeline_cost = [{"$group": {"_id": None, "total_cost": {"$sum": "$cost"}}}]
    cost_result = list(col.aggregate(pipeline_cost))
    if cost_result:
        print(f"   Coût total SMS   : {cost_result[0]['total_cost']}")

    # Campagnes
    campagnes = col.distinct("campagne_id")
    campagnes = [c for c in campagnes if c is not None]
    print(f"   Campagnes        : {campagnes}")


# ─── 4. Exemples de requêtes utiles ───────────────────────────────────────────
def exemple_requetes():
    print("\n🔍 Exemples de requêtes :")

    # Récupérer tous les SMS d'une campagne
    campagne_id = 1
    sms_campagne = list(col.find({"campagne_id": campagne_id}, {"msisdn": 1, "status": 1, "message": 1, "_id": 0}))
    print(f"   SMS campagne {campagne_id} : {len(sms_campagne)} contacts")

    # SMS échoués
    echecs = col.count_documents({"status": 2})
    print(f"   SMS échoués (status=2) : {echecs}")

    # SMS UTF16 (arabe, etc.)
    utf16 = col.count_documents({"encoding": "UTF16"})
    print(f"   SMS encodés UTF16 : {utf16}")

    # Numéros uniques
    msisdns = col.distinct("msisdn")
    print(f"   Numéros uniques (msisdn) : {len(msisdns)}")


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"📂 Chargement de {JSON_FILE}...")
    docs = load_json_file(JSON_FILE)
    print(f"   {len(docs)} documents chargés.")

    print("\n📥 Import dans MongoDB...")
    import_to_mongo(docs)

    print_stats()
    exemple_requetes()

    client.close()
    print("\n✅ Terminé.")