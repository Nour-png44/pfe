import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

# ══════════════════════════════════════════════════════════════
# 1. CHARGEMENT DU FICHIER user_contact.json
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
json_path = os.path.join(BASE_DIR, "data", "user_contact.json")

records = []
with open(json_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

df_raw = pd.DataFrame(records)
print("✅ Données chargées :", df_raw.shape)

# ══════════════════════════════════════════════════════════════
# 2. EXTRACTION DE L'HISTORIQUE DES STATUTS
# 8 = envoyé | 4 = en attente | 1 = livré (succès)
# ══════════════════════════════════════════════════════════════
rows = []

for _, row in df_raw.iterrows():
    msisdn         = row.get("msisdn", "")
    campagne_id    = row.get("campagne_id", None)
    encoding       = row.get("encoding", "GSM_7BIT")
    cost           = row.get("cost", 1)
    status_history = row.get("status_history", [])

    if not isinstance(status_history, list) or len(status_history) < 2:
        continue

    history_df = pd.DataFrame(status_history)
    history_df["status_last_updated_at"] = pd.to_datetime(
        history_df["status_last_updated_at"], errors="coerce"
    )
    history_df = history_df.sort_values("status_last_updated_at")

    t_start = history_df["status_last_updated_at"].iloc[0]
    t_end   = history_df["status_last_updated_at"].iloc[-1]

    # Durée en MINUTES
    duration_minutes = (t_end - t_start).total_seconds() / 60.0
    if duration_minutes <= 0:
        duration_minutes = 0.1

    statuses      = history_df["status"].tolist()
    event         = 1 if 1 in statuses else 0
    heure_envoi   = t_start.hour
    jour_semaine  = t_start.dayofweek
    nb_tentatives = len(status_history)
    is_utf16      = 1 if encoding == "UTF16" else 0

    rows.append({
        "msisdn"        : msisdn,
        "campagne_id"   : campagne_id,
        "duration"      : round(duration_minutes, 4),
        "event"         : event,
        "heure_envoi"   : heure_envoi,
        "jour_semaine"  : jour_semaine,
        "nb_tentatives" : nb_tentatives,
        "is_utf16"      : is_utf16,
        "cost"          : cost,
    })

df = pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING PAR CONTACT
# ══════════════════════════════════════════════════════════════
taux = df.groupby("msisdn")["event"].mean().reset_index()
taux.columns = ["msisdn", "taux_delivrabilite"]

def nb_echecs_consecutifs(events):
    count = 0
    for e in reversed(events):
        if e == 0:
            count += 1
        else:
            break
    return count

echecs = df.groupby("msisdn")["event"].apply(
    lambda x: nb_echecs_consecutifs(x.tolist())
).reset_index()
echecs.columns = ["msisdn", "nb_echecs_consecutifs"]

def time_since_last_success(group):
    successes = group[group["event"] == 1]["duration"]
    if len(successes) == 0:
        return group["duration"].max() * 2
    return group["duration"].max() - successes.max()

time_success = df.groupby("msisdn").apply(
    time_since_last_success, include_groups=False
).reset_index()
time_success.columns = ["msisdn", "time_since_last_success"]

df = df.merge(taux, on="msisdn", how="left")
df = df.merge(echecs, on="msisdn", how="left")
df = df.merge(time_success, on="msisdn", how="left")

# ══════════════════════════════════════════════════════════════
# 4. MODÈLE DE COX
# ══════════════════════════════════════════════════════════════
cox_features = [
    "duration", "event", "heure_envoi", "jour_semaine",
    "nb_tentatives", "is_utf16", "cost",
    "taux_delivrabilite", "nb_echecs_consecutifs", "time_since_last_success",
]

# Garder msisdn séparément pour le rapport final
df_model = df[cox_features + ["msisdn"]].dropna()
df_model = df_model[df_model["duration"] > 0]
df_cox   = df_model[cox_features]

cph = CoxPHFitter(penalizer=0.1)
cph.fit(df_cox, duration_col="duration", event_col="event")

# ══════════════════════════════════════════════════════════════
# 5. ÉVALUATION
# ══════════════════════════════════════════════════════════════
c_index = concordance_index(
    df_cox["duration"],
    -cph.predict_partial_hazard(df_cox),
    df_cox["event"]
)

# ══════════════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ══════════════════════════════════════════════════════════════
output_dir = os.path.join(BASE_DIR, "models", "survival-cox")
os.makedirs(output_dir, exist_ok=True)

# Graphique 1 : Coefficients Cox
plt.figure(figsize=(10, 6))
cph.plot()
plt.title("Cox Model — Impact des features sur le risque de non-livraison")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cox_coefficients.png"))
plt.show()

# Graphique 2 : Kaplan-Meier
kmf = KaplanMeierFitter()
kmf.fit(df_cox["duration"], event_observed=df_cox["event"])
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title("Kaplan-Meier — Probabilité de livraison dans le temps")
plt.xlabel("Durée (minutes)")
plt.ylabel("Probabilité de livraison")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kaplan_meier.png"))
plt.show()

# Graphique 3 : Courbe de survie d'un contact exemple
sample = df_cox.iloc[[0]]
survival_func = cph.predict_survival_function(sample)
plt.figure(figsize=(10, 6))
survival_func.plot()
plt.title("Cox — Disponibilité d'un contact dans le temps")
plt.xlabel("Durée (minutes)")
plt.ylabel("P(disponible)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cox_survival_curve.png"))
plt.show()

# ══════════════════════════════════════════════════════════════
# 7. CONTACT AVAILABILITY SCORE (0-100)
# Score calculé à 30 minutes
# ══════════════════════════════════════════════════════════════
survival_at_30min = cph.predict_survival_function(df_cox, times=[30]).T
df_model = df_model.copy()
df_model["availability_score"] = (survival_at_30min.values.flatten() * 100).round(1)

def classify(score):
    if score >= 70:
        return "✅ VALIDE"
    elif score >= 40:
        return "⚠️  SUSPECT"
    else:
        return "❌ INVALIDE"

df_model["classification"] = df_model["availability_score"].apply(classify)

# ══════════════════════════════════════════════════════════════
# 8. RAPPORT FINAL — UN RÉSULTAT PAR CONTACT
# ══════════════════════════════════════════════════════════════
total    = len(df_model)
valide   = (df_model["classification"] == "✅ VALIDE").sum()
suspect  = (df_model["classification"] == "⚠️  SUSPECT").sum()
invalide = (df_model["classification"] == "❌ INVALIDE").sum()
livres   = int(df_model["event"].sum())

print("\n" + "="*60)
print("📋  RAPPORT — DISPONIBILITÉ DES CONTACTS SMS")
print("="*60)
print(f"📱 Total analysés  : {total}")
print(f"✅ Valides         : {valide}  ({valide/total*100:.1f}%)")
print(f"⚠️  Suspects        : {suspect}  ({suspect/total*100:.1f}%)")
print(f"❌ Invalides       : {invalide}  ({invalide/total*100:.1f}%)")

print("\n" + "─"*60)
print("📞 DÉTAIL PAR CONTACT")
print("─"*60)
print(f"{'Numéro':<20} {'Score':>10}   {'Statut'}")
print("─"*60)

# Trier par score décroissant
df_sorted = df_model[["msisdn", "availability_score", "classification"]].sort_values(
    "availability_score", ascending=False
).drop_duplicates(subset="msisdn")

for _, row in df_sorted.iterrows():
    print(f"{row['msisdn']:<20} {row['availability_score']:>6.1f}/100   {row['classification']}")

print("─"*60)
print(f"\n🎯 C-index = {c_index:.2f} → {'✅ Bon modèle' if c_index >= 0.7 else '⚠️ Acceptable'}")

# Sauvegarde CSV
scores_path = os.path.join(output_dir, "availability_scores.csv")
df_sorted.to_csv(scores_path, index=False)

print(f"\n💾 Résultats sauvegardés : {scores_path}")
print("="*60)
print("✅ Pipeline terminé avec succès !")
print("="*60)