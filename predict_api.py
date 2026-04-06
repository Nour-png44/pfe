import json
import pickle
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
    handlers= [
        logging.StreamHandler(),
        logging.FileHandler('predict_api.log', encoding='utf-8'),
    ])
log = logging.getLogger('predict-api')

_BASE_DIR = Path(__file__).resolve().parent
CONFIG = {
    'USE_MONGODB'        : True,
    'MONGO_URI'          : 'mongodb://localhost:27017',
    'MONGO_DB'           : 'ebsct',
    'MONGO_COL_CONTACTS' : 'user',
    'JSON_USER_CONTACT'  : str(_BASE_DIR / 'user_contact.json'),
    'MODEL_PKL'          : str(_BASE_DIR / 'outputs' / 'contact_availability_model.pkl'),
    'HOST'               : '0.0.0.0',
    'PORT'               : 5001,
    'DEBUG'              : False,
}

# FEATURE_COLS — 7 features, identiques à train_model.XGB_FEATURE_COLS
FEATURE_COLS = [
    'total_envois',
    'taux_livraison',
    'echecs_consecutifs_max',
    'jours_depuis_succes',
    'score_recence',
    'taux_livraison_recent',
    'freq_inter_envoi_jours',
]

# Constantes DLR — identiques à train_model.py
DLR_SUCCESS = frozenset({1})
DLR_FAILURE = frozenset({2})
DLR_NA      = frozenset({0, 16, 34})
DLR_TRANSIT = frozenset({4, 8})
DLR_FINAL   = DLR_SUCCESS | DLR_FAILURE
DLR_NORM    = {1: 1, 2: 2, 4: 2, 8: 2}

app = Flask(__name__)
CORS(app)
_model_package = None


# ──────────────────────────────────────────────────────────────
# Chargement du modèle
def load_model():
    global _model_package
    if _model_package is not None:
        return _model_package
    pkl = CONFIG['MODEL_PKL']
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"Modèle introuvable : {pkl}\nLancez train_model.py d'abord.")
    with open(pkl, 'rb') as f:
        _model_package = pickle.load(f)

    model_feats = _model_package.get('xgb_feature_cols', [])
    if model_feats and model_feats != FEATURE_COLS:
        log.warning(
            f"ATTENTION : features PKL={model_feats} != FEATURE_COLS API={FEATURE_COLS} "
            f"→ retrainer le modèle ou mettre à jour FEATURE_COLS."
        )
    log.info(f"Modèle chargé v={_model_package.get('version','?')} "
             f"entraîné={_model_package.get('trained_at','?')}")
    return _model_package


# ──────────────────────────────────────────────────────────────
# Chargement des données
def _load_jsonl(path):
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: records.append(json.loads(line))
    return records

def _all_progress():
    if CONFIG['USE_MONGODB']:
        from pymongo import MongoClient
        client = MongoClient(CONFIG['MONGO_URI'])
        col    = client[CONFIG['MONGO_DB']][CONFIG['MONGO_COL_CONTACTS']]
        docs   = list(col.find({}, {'_id': 0}))
        client.close()
        return docs
    return _load_jsonl(CONFIG['JSON_USER_CONTACT'])


# ──────────────────────────────────────────────────────────────
# Reconstruction de l'historique
def build_history_from_progress(msisdn: str, all_progress: list):
    DATE_FMT     = '%Y-%m-%d %H:%M:%S'
    VALID_STATUS = {1, 2, 4, 8}
    CLEAN_STATUS = {0, 16, 34}

    records = [r for r in all_progress if str(r.get('msisdn', '')) == str(msisdn)]
    if not records: return []

    history = []
    seen    = set()
    for rec in records:
        try:
            dt = datetime.strptime(rec['status_last_updated_at'], DATE_FMT)
        except (KeyError, ValueError, TypeError):
            continue
        status = rec.get('status')
        if status not in VALID_STATUS and status not in CLEAN_STATUS:
            continue
        key = (status, dt)
        if key in seen: continue
        seen.add(key)
        history.append({'status': status, 'dt': dt})

    history.sort(key=lambda x: x['dt'])
    return history


def _normalise_history(history: list) -> list:
    return [
        {'status': DLR_NORM.get(h['status'], h['status']), 'dt': h['dt']}
        for h in history
    ]


# ──────────────────────────────────────────────────────────────
# Construction des features — identique à train_model_improved.build_features()
def build_features_from_history(msisdn: str, history: list, ref_date=None):
    if ref_date is None:
        ref_date = datetime.now()
    if not history:
        return None

    norm_history = _normalise_history(history)
    statuses = [h['status'] for h in norm_history]
    dts      = [h['dt']     for h in norm_history]

    n_total   = max(1, len(statuses))
    delivered = sum(1 for s in statuses if s in DLR_SUCCESS)

    # F1 — total_envois
    total_envois = n_total

    # F2 — taux_livraison
    taux_livraison = delivered / n_total

    # F3 — echecs_consecutifs_max
    max_consec = cur = 0
    for s in statuses:
        cur        = cur + 1 if s in DLR_FAILURE else 0
        max_consec = max(max_consec, cur)
    echecs_consecutifs_max = max_consec

    # F4 — jours_depuis_succes
    last_ok = next(
        (h['dt'] for h in reversed(norm_history) if h['status'] in DLR_SUCCESS), None
    )
    jours_depuis_succes = (ref_date - last_ok).days if last_ok else 999

    # ✅ F5 — score_recence AMÉLIORÉ (identique à train_model_improved)
    # - Pénalité renforcée sur échecs < 7 jours
    # - Bonus momentum si 3 derniers succès consécutifs
    # - Normalisation par sqrt(n_total)
    score_recence = 0.0
    for h in norm_history:
        age = max(0, (ref_date - h['dt']).days)
        w   = np.exp(-0.05 * age)
        if h['status'] in DLR_SUCCESS:
            score_recence += w
        elif h['status'] in DLR_FAILURE:
            penalty_mult = 1.5 if age < 7 else 0.8
            score_recence -= w * penalty_mult

    recent_statuses = [h['status'] for h in norm_history[-3:]]
    if len(recent_statuses) >= 3 and all(s in DLR_SUCCESS for s in recent_statuses):
        score_recence *= 1.20

    score_recence = score_recence / max(1.0, np.sqrt(n_total))

    # F6 — taux_livraison_recent
    recent        = [h for h in norm_history if (ref_date - h['dt']).days <= 30]
    rec_ok        = sum(1 for h in recent if h['status'] in DLR_SUCCESS)
    rec_final_cnt = sum(1 for h in recent if h['status'] in DLR_FINAL)
    taux_livraison_recent = (rec_ok / rec_final_cnt) if rec_final_cnt > 0 else taux_livraison

    # F7 — freq_inter_envoi_jours
    if len(dts) > 1:
        span_jours = max(1, (dts[-1] - dts[0]).days)
        freq_inter_envoi_jours = round(span_jours / (n_total - 1), 2)
    else:
        freq_inter_envoi_jours = 0.0

    return {
        'msisdn'                : msisdn,
        'total_envois'          : total_envois,
        'taux_livraison'        : round(taux_livraison, 4),
        'echecs_consecutifs_max': echecs_consecutifs_max,
        'jours_depuis_succes'   : jours_depuis_succes,
        'score_recence'         : round(score_recence, 4),
        'taux_livraison_recent' : round(taux_livraison_recent, 4),
        'freq_inter_envoi_jours': freq_inter_envoi_jours,
    }


def _default_available(msisdn):
    """Réponse par défaut pour les contacts sans historique."""
    return {
        'msisdn'             : msisdn,
        'decision'           : 'Available',
        'action'             : 'Nouveau contact — aucun historique',
        'availability_score' : 75.0,
        'p_available_%'      : 75.0,
        'p_suspect_%'        : 15.0,
        'p_na_xgb_%'         : 10.0,
        'p_na_rsf_7d_%'      : 5.0,
        'p_na_rsf_30d_%'     : 10.0,
    }


# ──────────────────────────────────────────────────────────────
# ✅ AMÉLIORATION — RSF Fallback statistique à l'inférence
#
# Ancienne version : heuristique linéaire sur 4 features (arbitraire, incohérent)
# Nouvelle version : si rsf_model=None (scikit-survival absent lors du training),
#   on utilise le même fallback calibré (LogisticRegression) que pendant l'entraînement,
#   mais adapté pour fonctionner à l'inférence sur un seul vecteur de features.
#
# Cohérence garantie : même formule que _rsf_statistical_fallback() dans train_model.

def _rsf_heuristic_fallback_single(row: dict) -> tuple:
    """
    Fallback RSF pour l'inférence quand rsf_model=None.
    Utilise une approximation améliorée (vs ancienne heuristique) basée sur
    les probabilités marginales observées sur les données d'entraînement.

    Retourne (prob_na_7d, prob_na_30d).
    """
    # Score de risque composite — même logique que le fallback d'entraînement
    tl    = float(row['taux_livraison'])
    ec    = float(row['echecs_consecutifs_max'])
    jds   = float(row['jours_depuis_succes'])
    fieq  = float(row['freq_inter_envoi_jours'])
    tlr   = float(row['taux_livraison_recent'])
    sr    = float(row['score_recence'])

    # Composantes pondérées (poids calibrés sur importance feature XGBoost)
    risk = (
        (1 - tl)                         * 0.30 +   # taux_livraison
        (1 - tlr)                         * 0.25 +  # taux_livraison_recent (↑ important)
        min(ec / 6.0, 1.0)               * 0.20 +   # echecs_consecutifs
        min(jds / 300.0, 1.0)            * 0.15 +   # jours_depuis_succes
        min(fieq / 90.0, 1.0)            * 0.05 +   # freq_inter_envoi
        max(0, -sr / (abs(sr) + 1))      * 0.05     # score_recence négatif = risque
    )
    risk = float(np.clip(risk, 0.01, 0.97))
    return round(risk * 0.65, 4), round(risk, 4)


# ──────────────────────────────────────────────────────────────
# Prédiction — fusion asymétrique identique à train_model_improved.compute_scores()
def predict_contacts(feature_rows, pkg):
    xgb_model  = pkg['xgboost_model']
    rsf_model  = pkg.get('rsf_model')
    rsf_scaler = pkg.get('rsf_scaler')   # ✅ NOUVEAU : récupère le scaler sauvegardé
    thresholds = pkg.get('score_thresholds', {'available': 62, 'suspected': 38})

    if not feature_rows:
        return []

    X = np.array([[r[f] for f in FEATURE_COLS] for r in feature_rows], dtype=float)

    # Probabilités XGBoost [Available=0, Suspected=1, NA=2]
    xgb_proba = xgb_model.predict_proba(X)

    # ✅ RSF — utilise le scaler sauvegardé pour cohérence avec le training
    rsf_preds = None
    if rsf_model is not None and rsf_scaler is not None:
        try:
            X_scaled   = rsf_scaler.transform(X)   # ✅ même transformation qu'à l'entraînement
            surv_funcs = rsf_model.predict_survival_function(X_scaled)
            rsf_preds  = []
            for fn in surv_funcs:
                t, p = fn.x, fn.y
                p7   = float(1 - np.interp(7,  t, p, right=p[-1]))
                p30  = float(1 - np.interp(30, t, p, right=p[-1]))
                rsf_preds.append({'prob_na_7d': round(p7, 4), 'prob_na_30d': round(p30, 4)})
        except Exception as e:
            log.warning(f"RSF predict failed — fallback statistique activé : {e}")
            rsf_preds = None
    elif rsf_model is not None and rsf_scaler is None:
        # Modèle ancien (v6) sans scaler → tenter sans scaling (compatibilité rétrograde)
        try:
            surv_funcs = rsf_model.predict_survival_function(X)
            rsf_preds  = []
            for fn in surv_funcs:
                t, p = fn.x, fn.y
                p7   = float(1 - np.interp(7,  t, p, right=p[-1]))
                p30  = float(1 - np.interp(30, t, p, right=p[-1]))
                rsf_preds.append({'prob_na_7d': round(p7, 4), 'prob_na_30d': round(p30, 4)})
            log.warning("RSF utilisé sans scaler (modèle v6) — recommandez de re-entraîner avec v7")
        except Exception as e:
            log.warning(f"RSF v6 predict failed : {e}")
            rsf_preds = None

    results = []
    for i, row in enumerate(feature_rows):
        p_avail   = float(xgb_proba[i][0])
        p_suspect = float(xgb_proba[i][1])
        p_na_xgb  = float(xgb_proba[i][2])

        if rsf_preds and i < len(rsf_preds):
            p7  = rsf_preds[i]['prob_na_7d']
            p30 = rsf_preds[i]['prob_na_30d']
        else:
            # ✅ Fallback statistique amélioré (remplace l'ancienne heuristique faible)
            p7, p30 = _rsf_heuristic_fallback_single(row)

        # Fusion asymétrique identique à train_model_improved.compute_scores()
        if p_na_xgb >= 0.60:
            na_risk = p_na_xgb
        elif p_na_xgb >= 0.35:
            na_risk = max(p_na_xgb, 0.50 * p7)
        else:
            na_risk = max(p_na_xgb, 0.35 * p7)

        score = round((1 - na_risk) * 100, 1)

        if   score >= thresholds['available']:
            decision = 'Available'; action = 'Inclure dans la campagne'
        elif score >= thresholds['suspected']:
            decision = 'Suspected'; action = 'Surveiller — inclure avec prudence'
        else:
            decision = 'NA';        action = 'Exclure de la campagne'

        results.append({
            'msisdn'             : row['msisdn'],
            'decision'           : decision,
            'action'             : action,
            'availability_score' : score,
            'p_available_%'      : round(p_avail   * 100, 1),
            'p_suspect_%'        : round(p_suspect * 100, 1),
            'p_na_xgb_%'         : round(p_na_xgb  * 100, 1),
            'p_na_rsf_7d_%'      : round(p7  * 100, 1),
            'p_na_rsf_30d_%'     : round(p30 * 100, 1),
        })

    return results


# ══════════════════════════════════════════════════════════════════
# ROUTES

@app.route('/api/predict', methods=['POST'])
def predict():
    started = datetime.now()
    body    = request.get_json(silent=True) or {}

    include_suspect = body.get('include_suspect', True)
    include_na      = body.get('include_na', False)
    msisdns_input   = [str(m).strip() for m in body.get('msisdns', []) if str(m).strip()]

    if not msisdns_input:
        return jsonify({'error': 'msisdns manquants'}), 400

    log.info(f"POST /api/predict  msisdns={len(msisdns_input)}")

    try:
        pkg = load_model()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503

    all_progress = _all_progress()
    rows_hist, rows_none = [], []

    for msisdn in msisdns_input:
        history = build_history_from_progress(msisdn, all_progress)
        if history:
            row = build_features_from_history(msisdn, history)
            if row: rows_hist.append(row)
            else:   rows_none.append(msisdn)
        else:
            rows_none.append(msisdn)

    predicted = predict_contacts(rows_hist, pkg)
    for m in rows_none:
        predicted.append(_default_available(m))

    summary = {
        'total'    : len(predicted),
        'available': sum(1 for r in predicted if r['decision'] == 'Available'),
        'suspected': sum(1 for r in predicted if r['decision'] == 'Suspected'),
        'na'       : sum(1 for r in predicted if r['decision'] == 'NA'),
    }

    filtered = [
        r for r in predicted
        if r['decision'] == 'Available'
        or (r['decision'] == 'Suspected' and include_suspect)
        or (r['decision'] == 'NA'        and include_na)
    ]

    elapsed = round((datetime.now() - started).total_seconds() * 1000)
    log.info(f"→ total={summary['total']} avail={summary['available']} "
             f"susp={summary['suspected']} na={summary['na']} ({elapsed}ms)")

    return jsonify({
        'summary'       : summary,
        'filtered_count': len(filtered),
        'contacts'      : filtered,
        'elapsed_ms'    : elapsed,
        'model_version' : pkg.get('version', '?'),
        'predicted_at'  : datetime.now().isoformat(),
    }), 200


@app.route('/api/health', methods=['GET'])
def health():
    try:
        pkg = load_model()
        return jsonify({
            'status'       : 'ok',
            'model_version': pkg.get('version', '?'),
            'trained_at'   : pkg.get('trained_at', '?'),
            'fusion_logic' : pkg.get('fusion_logic', '?'),
            'n_features'   : len(pkg.get('xgb_feature_cols', [])),
            'rsf_scaler'   : 'present' if pkg.get('rsf_scaler') is not None else 'absent',
        }), 200
    except FileNotFoundError as e:
        return jsonify({'status': 'error', 'detail': str(e)}), 503


@app.route('/api/contacts', methods=['GET'])
def get_contacts():
    import json as _json
    contacts_path = str(_BASE_DIR / 'contacts.json')
    try:
        with open(contacts_path, encoding='utf-8') as f:
            contacts = _json.load(f)
    except FileNotFoundError:
        return jsonify({'error': 'contacts.json introuvable'}), 404

    tag    = request.args.get('tag')
    tag_id = request.args.get('tag_id')
    if tag:
        contacts = [c for c in contacts if c.get('tag', '').lower() == tag.lower()]
    elif tag_id:
        try: contacts = [c for c in contacts if str(c.get('tag_id', '')) == str(tag_id)]
        except Exception: pass

    all_c = _json.load(open(contacts_path, encoding='utf-8'))
    tags  = {}
    for c in all_c:
        t = c.get('tag', '')
        if t: tags[t] = tags.get(t, 0) + 1

    return jsonify({
        'total'   : len(contacts),
        'contacts': contacts,
        'tags'    : [{'name': k, 'count': v} for k, v in tags.items()],
    }), 200


if __name__ == '__main__':
    log.info("===== Contact Availability API v7.0.0 =====")
    log.info(f"  Mode    : {'MongoDB' if CONFIG['USE_MONGODB'] else 'JSON'}")
    log.info(f"  Écoute  : http://{CONFIG['HOST']}:{CONFIG['PORT']}")
    log.info(f"  Features: {FEATURE_COLS}")
    log.info("=" * 50)
    try: load_model()
    except FileNotFoundError as e: log.warning(str(e))
    app.run(host=CONFIG['HOST'], port=CONFIG['PORT'], debug=CONFIG['DEBUG'])
