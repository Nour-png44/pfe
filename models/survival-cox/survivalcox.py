import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from datetime import datetime

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
print("✅ Données brutes chargées :", df_raw.shape)
print(f"   Contacts uniques (avant regroupement) : {df_raw['msisdn'].nunique()}")
print(f"   Contacts totaux  (avec doublons)      : {len(df_raw)}")

# ══════════════════════════════════════════════════════════════
# 2. REGROUPEMENT DES CONTACTS REDONDANTS PAR MSISDN
# Un même numéro peut apparaître plusieurs fois dans la base
# → On fusionne tout l'historique DLR de chaque MSISDN
# ══════════════════════════════════════════════════════════════
print("\n⏳ Regroupement des contacts redondants par MSISDN...")

grouped = {}

for _, row in df_raw.iterrows():
    msisdn         = row.get("msisdn", "")
    encoding       = row.get("encoding", "GSM_7BIT")
    cost           = row.get("cost", 1)
    status_history = row.get("status_history", [])

    if not msisdn or not isinstance(status_history, list):
        continue

    if msisdn not in grouped:
        grouped[msisdn] = {
            "msisdn"         : msisdn,
            "encoding"       : encoding,
            "cost_list"      : [],
            "status_history" : [],
        }

    grouped[msisdn]["cost_list"].append(cost)
    grouped[msisdn]["status_history"].extend(status_history)

print(f"✅ {len(grouped)} contacts uniques après regroupement")
print(f"   (doublons fusionnés : {len(df_raw) - len(grouped)} entrées regroupées)")

# ══════════════════════════════════════════════════════════════
# 3. EXTRACTION HISTORIQUE DLR + NETTOYAGE PAR MSISDN
# DLR = Delivery Report
# Statuts : 8 = SENT | 4 = PENDING | 1 = DELIVERED
# ══════════════════════════════════════════════════════════════
print("\n⏳ Extraction et nettoyage historique DLR...")

NOW = pd.Timestamp(datetime.now())
rows = []

for msisdn, data in grouped.items():
    status_history = data["status_history"]

    if len(status_history) < 2:
        continue

    history_df = pd.DataFrame(status_history)

    # Nettoyage : convertir les dates
    history_df["status_last_updated_at"] = pd.to_datetime(
        history_df["status_last_updated_at"], errors="coerce"
    )
    history_df = history_df.dropna(subset=["status_last_updated_at"])

    # Normalisation : garder uniquement statuts connus
    history_df = history_df[history_df["status"].isin([1, 4, 8])]

    if len(history_df) < 2:
        continue

    # Trier chronologiquement
    history_df = history_df.sort_values("status_last_updated_at").reset_index(drop=True)

    t_start = history_df["status_last_updated_at"].iloc[0]
    t_end   = history_df["status_last_updated_at"].iloc[-1]

    # Duration en minutes
    duration_minutes = (t_end - t_start).total_seconds() / 60.0
    if duration_minutes <= 0:
        duration_minutes = 0.1

    statuses = history_df["status"].tolist()
    event    = 1 if 1 in statuses else 0

    rows.append({
        "msisdn"       : msisdn,
        "encoding"     : data["encoding"],
        "cost"         : np.mean(data["cost_list"]),
        "duration"     : round(duration_minutes, 4),
        "event"        : event,
        "t_start"      : t_start,
        "t_end"        : t_end,
        "statuses"     : statuses,
        "heure_envoi"  : t_start.hour,
        "jour_semaine" : t_start.dayofweek,
        "nb_envois"    : len(history_df),
    })

df = pd.DataFrame(rows)
print(f"✅ {len(df)} contacts prêts pour l'analyse")
print(f"   Livrés     (event=1) : {df['event'].sum()}")
print(f"   Non livrés (event=0) : {(df['event']==0).sum()}")

# ══════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING TEMPOREL PAR MSISDN
# ══════════════════════════════════════════════════════════════
print("\n⏳ Feature Engineering temporel...")

features_list = []

for msisdn, group in df.groupby("msisdn"):
    group = group.sort_values("t_start")

    # Total envois
    total_envois = int(group["nb_envois"].sum())

    # Taux de livraison
    taux_livraison = round(group["event"].mean(), 4)

    # Taux d'échec
    taux_echec = round(1 - taux_livraison, 4)

    # Échecs consécutifs
    echecs_consecutifs = 0
    for e in reversed(group["event"].tolist()):
        if e == 0:
            echecs_consecutifs += 1
        else:
            break

    # Jours depuis dernier succès
    succes = group[group["event"] == 1]
    if len(succes) > 0:
        dernier_succes      = succes["t_end"].max()
        jours_depuis_succes = round((NOW - dernier_succes).total_seconds() / 86400.0, 2)
    else:
        jours_depuis_succes = 999.0

    # Time Decay Score : w = exp(-λ * jours_depuis_envoi)
    LAMBDA = 0.1
    poids_list = []
    for _, r in group.iterrows():
        jours_ago = (NOW - r["t_start"]).total_seconds() / 86400.0
        poids     = np.exp(-LAMBDA * jours_ago)
        poids_list.append(poids * r["event"])
    time_decay_score = round(np.sum(poids_list) / (len(poids_list) + 1e-9), 6)

    last = group.iloc[-1]

    features_list.append({
        "msisdn"              : msisdn,
        "total_envois"        : total_envois,
        "taux_livraison"      : taux_livraison,
        "taux_echec"          : taux_echec,
        "echecs_consecutifs"  : echecs_consecutifs,
        "jours_depuis_succes" : jours_depuis_succes,
        "time_decay_score"    : time_decay_score,
        "heure_envoi"         : round(group["heure_envoi"].mean(), 2),
        "jour_semaine"        : round(group["jour_semaine"].mean(), 2),
        "nb_tentatives"       : round(group["nb_envois"].mean(), 2),
        "is_utf16"            : int(last["encoding"] == "UTF16"),
        "cost"                : round(group["cost"].mean(), 2),
        "duration"            : last["duration"],
        "event"               : last["event"],
    })

df_features = pd.DataFrame(features_list)

print("✅ Features calculées :")
print(df_features[[
    "msisdn", "total_envois", "taux_livraison", "taux_echec",
    "echecs_consecutifs", "jours_depuis_succes", "time_decay_score"
]].to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 5. MODÈLE DE COX
# ══════════════════════════════════════════════════════════════
print("\n⏳ Entraînement du modèle Cox...")

cox_features = [
    "duration", "event",
    "total_envois", "taux_livraison", "taux_echec",
    "echecs_consecutifs", "jours_depuis_succes", "time_decay_score",
    "heure_envoi", "jour_semaine", "nb_tentatives", "is_utf16", "cost",
]

df_model = df_features[cox_features + ["msisdn"]].dropna()
df_model = df_model[df_model["duration"] > 0].copy()
df_cox   = df_model[cox_features]

cph = CoxPHFitter(penalizer=0.1)
cph.fit(df_cox, duration_col="duration", event_col="event")
print("✅ Modèle Cox entraîné")

# ══════════════════════════════════════════════════════════════
# 6. ÉVALUATION
# ══════════════════════════════════════════════════════════════
c_index = concordance_index(
    df_cox["duration"],
    -cph.predict_partial_hazard(df_cox),
    df_cox["event"]
)

# ══════════════════════════════════════════════════════════════
# 7. VISUALISATIONS
# ══════════════════════════════════════════════════════════════
output_dir = os.path.join(BASE_DIR, "models", "survival-cox")
os.makedirs(output_dir, exist_ok=True)

# Graphique 1 : Coefficients Cox
plt.figure(figsize=(12, 7))
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

# Graphique 3 : Courbe de survie d'un contact
sample = df_cox.iloc[[0]]
survival_func = cph.predict_survival_function(sample)
plt.figure(figsize=(10, 6))
survival_func.plot()
plt.title("Cox — Courbe de disponibilité d'un contact")
plt.xlabel("Durée (minutes)")
plt.ylabel("P(disponible)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cox_survival_curve.png"))
plt.show()

# ══════════════════════════════════════════════════════════════
# 8. PRÉDICTIONS TEMPORELLES COX
# • P(NA à 7 jours)
# • P(NA à 30 jours)
# • Temps estimé avant NA
# ══════════════════════════════════════════════════════════════
print("\n⏳ Calcul des prédictions temporelles...")

T_7J  = 7  * 24 * 60
T_30J = 30 * 24 * 60

surv_7j  = cph.predict_survival_function(df_cox, times=[T_7J]).T
surv_30j = cph.predict_survival_function(df_cox, times=[T_30J]).T

df_model["p_na_7j"]  = ((1 - surv_7j.values.flatten())  * 100).round(1)
df_model["p_na_30j"] = ((1 - surv_30j.values.flatten()) * 100).round(1)

median_survival = cph.predict_median(df_cox)
df_model["temps_avant_na_jours"] = (median_survival / (24 * 60)).round(2)
df_model["temps_avant_na_jours"] = df_model["temps_avant_na_jours"].replace(
    [np.inf, -np.inf], 999
).fillna(999)

print("✅ Prédictions temporelles calculées")

# ══════════════════════════════════════════════════════════════
# 9. RAPPORT FINAL
# ══════════════════════════════════════════════════════════════
total  = len(df_model)
livres = int(df_model["event"].sum())

df_sorted = df_model[[
    "msisdn", "p_na_7j", "p_na_30j", "temps_avant_na_jours"
]].sort_values("p_na_7j", ascending=True).drop_duplicates(subset="msisdn")

print("\n" + "="*70)
print("📋  RAPPORT COX — PRÉDICTIONS TEMPORELLES PAR CONTACT")
print("="*70)
print(f"📱 Contacts uniques analysés : {total}")
print(f"📨 Contacts livrés           : {livres} / {total} ({livres/total*100:.1f}%)")
print(f"🎯 C-index                   : {c_index:.2f} → {'✅ Bon modèle' if c_index >= 0.7 else '⚠️ Acceptable'}")

print("\n" + "─"*70)
print(f"{'Numéro':<18} {'P(NA 7j)':>10} {'P(NA 30j)':>10} {'Temps avant NA':>16}")
print("─"*70)

for _, row in df_sorted.iterrows():
    tps = f"{row['temps_avant_na_jours']}j" if row['temps_avant_na_jours'] < 999 else ">999j"
    print(
        f"{row['msisdn']:<18} "
        f"{row['p_na_7j']:>8.1f}%  "
        f"{row['p_na_30j']:>8.1f}%  "
        f"{tps:>14}"
    )

print("─"*70)
print("\n📖 Légende :")
print("   P(NA 7j)       → Probabilité NON disponible dans 7 jours")
print("   P(NA 30j)      → Probabilité NON disponible dans 30 jours")
print("   Temps avant NA → Temps estimé avant indisponibilité")
print("   ⚠️  Classification finale → rôle de XGBoost (collègue)")

# ══════════════════════════════════════════════════════════════
# 10. SAUVEGARDE — PRÊT POUR FUSION AVEC XGBOOST
# ══════════════════════════════════════════════════════════════
scores_path = os.path.join(output_dir, "cox_predictions.csv")
df_sorted.to_csv(scores_path, index=False)

print(f"\n💾 Prédictions sauvegardées : {scores_path}")
print("   → Fichier prêt pour la fusion avec XGBoost")
print("="*70)
print("✅ Pipeline Survival Cox terminé avec succès !")
print("="*70)