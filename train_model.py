# ==========================Importation des bibliothèques standards==========================================
import json
import warnings
import os
import logging
import time
from threading import Thread
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib  # pour remplacer pickle
warnings.filterwarnings('ignore')

# ================================CONFIGURATION GÉNÉRALE =============================================================
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler('training.log', encoding='utf-8'),
    ],)
log = logging.getLogger('training')
_BASE_DIR = Path(__file__).resolve().parent
CONFIG = {
    'USE_MONGODB'          : False,
    'MONGO_URI'            : 'mongodb://localhost:27017',
    'MONGO_DB'             : 'test',
    'MONGO_COL'            : 'contacts',
    'JSON_PATH'            : str(_BASE_DIR / 'user_contact.json'),
    'OUTPUT_DIR'           : str(_BASE_DIR / 'outputs'),
    'MODEL_PKL'            : str(_BASE_DIR / 'outputs' / 'model_assets.joblib'),  # joblib
    'MODEL_XGB'            : str(_BASE_DIR / 'outputs' / 'xgboost_model.json'),
    'PREDICTIONS_CSV'      : str(_BASE_DIR / 'outputs' / 'contact_predictions.csv'),
    'METRICS_JSON'         : str(_BASE_DIR / 'outputs' / 'eval_metrics.json'),
    'SCHEDULE_TIME'        : '02:00',
    'RETRAIN_INTERVAL_DAYS': 14,
    'RUN_NOW'              : True,
    'ENABLE_HYPERPARAMETER_TUNING': True,
    'TUNING_CV_FOLDS'             : 5,
}
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

VALID_STATUSES    = {1, 2, 4, 8}
STATUSES_TO_CLEAN = {0, 16, 34}
DLR_NORM    = {1: 1, 2: 2, 4: 2, 8: 2}
DLR_SUCCESS = frozenset({1})
DLR_FAILURE = frozenset({2})
DLR_NA = frozenset({0, 16, 34})
DLR_TRANSIT = frozenset({4, 8})
DLR_FINAL   = DLR_SUCCESS | DLR_FAILURE

XGB_FEATURE_COLS = [
    'total_envois',
    'taux_livraison',
    'echecs_consecutifs_max',
    'jours_depuis_succes',
    'score_recence',
    'taux_livraison_recent',
    'freq_inter_envoi_jours',]
RSF_EXTRA_COLS = [
    'evenement_na',
    'duree_jours',]
SCORE_THRESHOLDS = {
    'available' : 62,
    'suspected' : 38,}

# =========================BLOC 1A : Extraction des données (avec curseur MongoDB) ==================================
def load_from_mongodb():
    """Générateur : parcourt la collection sans tout charger en RAM."""
    from pymongo import MongoClient
    client = MongoClient(CONFIG['MONGO_URI'])
    col = client[CONFIG['MONGO_DB']][CONFIG['MONGO_COL']]
    cursor = col.find({}, {'_id': 0}).batch_size(10000)
    for doc in cursor:
        yield doc
    client.close()

def load_from_json():
    with open(CONFIG['JSON_PATH'], encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# ============================== Filtrage et validation (itérative) =================================================
def filter_and_validate(generator):
    """Parcourt le générateur, valide et déduplique sans accumuler tous les documents."""
    DATE_FMT = '%Y-%m-%d %H:%M:%S'
    valid = []
    stats = {
        'total_raw': 0, 'invalid_msisdn': 0,
        'invalid_date': 0, 'unknown_status': 0,
        'duplicate_sms': 0, 'valid': 0,
    }
    seen_sms = set()

    for r in generator:
        stats['total_raw'] += 1
        msisdn = str(r.get('msisdn', '')).strip()
        if not msisdn or len(msisdn) < 8 or not msisdn.lstrip('+').isdigit():
            stats['invalid_msisdn'] += 1
            continue

        try:
            dt = datetime.strptime(r['status_last_updated_at'], DATE_FMT)
        except (KeyError, ValueError, TypeError):
            stats['invalid_date'] += 1
            continue

        status = r.get('status')
        if status not in VALID_STATUSES and status not in STATUSES_TO_CLEAN:
            stats['unknown_status'] += 1
            continue

        key = (msisdn, status, dt)
        if key in seen_sms:
            stats['duplicate_sms'] += 1
            continue
        seen_sms.add(key)

        valid.append({'msisdn': msisdn, 'status': status, 'dt': dt})
        stats['valid'] += 1

    log.info(f" {stats['valid']} valides | {stats['total_raw'] - stats['valid']} rejetés")
    log.info(f"  MSISDN invalide:{stats['invalid_msisdn']} | Date invalide:{stats['invalid_date']}")
    log.info(f"  Statuts inconnus:{stats['unknown_status']} | SMS dupliqués:{stats['duplicate_sms']}")
    return valid, stats

# ==========================Normalisation des statuts DLR=============================================================
def normalise_statuses(valid_records):
    normalised_count = 0
    for rec in valid_records:
        orig = rec['status']
        norm = DLR_NORM.get(orig, orig)
        if norm != orig:
            normalised_count += 1
        rec['status'] = norm

    log.info(f" {normalised_count} statuts normalisés (4->2, 8->2)")

    groups = defaultdict(list)
    for r in valid_records:
        groups[r['msisdn']].append(r)

    contacts = []
    for msisdn, recs in groups.items():
        recs.sort(key=lambda x: x['dt'])
        history = [{'status': r['status'], 'dt': r['dt']} for r in recs]
        contacts.append({
            'msisdn'      : msisdn,
            'history'     : history,
            'final_status': recs[-1]['status'],
            'n_sends'     : len(recs),
        })

    log.info(f"BLOC 1C — {len(contacts)} MSISDNs uniques")
    return contacts

def extract_and_clean(raw_generator):
    valid, _ = filter_and_validate(raw_generator)
    return normalise_statuses(valid)

# =========================== Feature Engineering (ref_date explicite) ===============================================
def _label(statuses, final_status):
    if final_status in DLR_NA:
        return 'NA'
    fails   = sum(1 for s in statuses if s in DLR_FAILURE)
    success = sum(1 for s in statuses if s in DLR_SUCCESS)
    consec = cur = 0
    for s in statuses:
        cur = cur + 1 if s in DLR_FAILURE else 0
        consec = max(consec, cur)
    if consec >= 3:
        return 'Suspected'
    if fails > 0 and success == 0:
        return 'Suspected'
    if fails >= 3 and fails > success:
        return 'Suspected'
    return 'Available'

def build_features(contacts, ref_date):
    """
    Construit les features avec une date de référence fixe (évite le data leakage temporel).
    """
    rows = []
    for c in contacts:
        hist = c['history']
        statuses = [h['status'] for h in hist]
        dts = [h['dt'] for h in hist]

        n_total   = max(1, len(statuses))
        delivered = sum(1 for s in statuses if s in DLR_SUCCESS)
        failed = sum(1 for s in statuses if s in DLR_FAILURE)

        total_envois = n_total
        taux_livraison = delivered / n_total

        max_consec = cur = 0
        for s in statuses:
            cur = cur + 1 if s in DLR_FAILURE else 0
            max_consec = max(max_consec, cur)
        echecs_consecutifs_max = max_consec

        last_ok = next((h['dt'] for h in reversed(hist) if h['status'] in DLR_SUCCESS), None)
        jours_depuis_succes = (ref_date - last_ok).days if last_ok else 999

        score_recence = 0.0
        for h in hist:
            age = max(0, (ref_date - h['dt']).days)
            w = np.exp(-0.05 * age)
            if h['status'] in DLR_SUCCESS:
                score_recence += w
            elif h['status'] in DLR_FAILURE:
                penalty_mult = 1.5 if age < 7 else 0.8
                score_recence -= w * penalty_mult

        recent_statuses = [h['status'] for h in hist[-3:]]
        if len(recent_statuses) >= 3 and all(s in DLR_SUCCESS for s in recent_statuses):
            score_recence *= 1.20

        score_recence = score_recence / max(1.0, np.sqrt(n_total))

        recent = [h for h in hist if (ref_date - h['dt']).days <= 30]
        rec_ok = sum(1 for h in recent if h['status'] in DLR_SUCCESS)
        rec_final_cnt = sum(1 for h in recent if h['status'] in DLR_FINAL)
        taux_livraison_recent = (rec_ok / rec_final_cnt) if rec_final_cnt > 0 else taux_livraison

        if len(dts) > 1:
            span_jours = max(1, (dts[-1] - dts[0]).days)
            freq_inter_envoi_jours = round(span_jours / (n_total - 1), 2)
        else:
            freq_inter_envoi_jours = 0.0

        duree_jours = max(1, (dts[-1] - dts[0]).days) if len(dts) > 1 else 1
        evenement_na = bool(c['final_status'] in DLR_NA)

        rows.append({
            'msisdn'                : c['msisdn'],
            'total_envois'          : total_envois,
            'taux_livraison'        : round(taux_livraison, 4),
            'echecs_consecutifs_max': echecs_consecutifs_max,
            'jours_depuis_succes'   : jours_depuis_succes,
            'score_recence'         : round(score_recence, 4),
            'taux_livraison_recent' : round(taux_livraison_recent, 4),
            'freq_inter_envoi_jours': freq_inter_envoi_jours,
            'label'                 : _label(statuses, c['final_status']),
            'duree_jours'           : duree_jours,
            'evenement_na'          : evenement_na,
        })

    df = pd.DataFrame(rows)
    log.info(f"Features → {len(df)} contacts | labels : {df['label'].value_counts().to_dict()}")
    return df

# ======================== Encodage des labels ======================================================================
def encode_labels(df_real):
    df = df_real.copy()
    df['source'] = 'real'
    label_map = {'Available': 0, 'Suspected': 1, 'NA': 2}
    df['label_enc'] = df['label'].map(label_map)
    log.info(f"Dataset → {len(df)} contacts | labels={df['label'].value_counts().to_dict()}")
    return df, label_map

# ======================= XGBoost Hyperparameter ===================================================================
def _default_xgb_params():
    return dict(
        n_estimators = 400,
        max_depth = 4,
        learning_rate = 0.03,
        subsample = 0.80,
        colsample_bytree = 0.75,
        min_child_weight = 5,
        gamma = 0.15,
        reg_alpha = 0.15,
        reg_lambda = 1.2,
    )

def _tune_xgboost_optuna(X_tr, y_tr, X_val, y_val, n_trials=50):
    import optuna
    from sklearn.metrics import f1_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            n_estimators = trial.suggest_int('n_estimators', 200, 600, step=50),
            max_depth = trial.suggest_int('max_depth', 3, 7),
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.10, log=True),
            subsample = trial.suggest_float('subsample', 0.60, 0.95),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.50, 0.95),
            min_child_weight = trial.suggest_int('min_child_weight', 2, 10),
            gamma = trial.suggest_float('gamma', 0.0, 0.5),
            reg_alpha = trial.suggest_float('reg_alpha', 0.0, 0.5),
            reg_lambda = trial.suggest_float('reg_lambda', 0.5, 2.0),
        )
        m = xgb.XGBClassifier(
            **params,
            objective='multi:softprob', num_class=3,
            eval_metric='mlogloss', early_stopping_rounds=20,
            use_label_encoder=False, random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = m.predict(X_val)
        return f1_score(y_val, preds, average='macro', zero_division=0)

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    log.info(f"Optuna tuning terminé — {n_trials} trials | meilleur F1-macro val={study.best_value:.4f}")
    log.info(f"  Meilleurs hyperparamètres : {best}")
    return best

def _tune_xgboost_grid(X_tr, y_tr, X_val, y_val):
    from sklearn.metrics import f1_score
    from itertools import product

    grid = {
        'max_depth'       : [3, 4, 5],
        'learning_rate'   : [0.03, 0.05, 0.08],
        'min_child_weight': [3, 5, 8],
    }

    best_f1, best_params = -1, {}
    base = dict(n_estimators=400, subsample=0.80, colsample_bytree=0.75,
                gamma=0.15, reg_alpha=0.15, reg_lambda=1.2)

    keys = list(grid.keys())
    values = list(grid.values())

    for combo in product(*values):
        params = {**base, **dict(zip(keys, combo))}
        m = xgb.XGBClassifier(
            **params,
            objective='multi:softprob', num_class=3,
            eval_metric='mlogloss', early_stopping_rounds=20,
            use_label_encoder=False, random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        f1 = f1_score(m.predict(X_val), y_val, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_params = f1, params

    log.info(f"GridSearch fallback terminé | meilleur F1-macro val={best_f1:.4f}")
    log.info(f"  Meilleurs hyperparamètres : {best_params}")
    return best_params

# ──────────────────Training XGBoost (sans fuite de scaling car pas de scaling ici) ────────────────────────────────
def train_xgboost(df):
    from sklearn.model_selection import train_test_split

    X = df[XGB_FEATURE_COLS].values
    y = df['label_enc'].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval,
    )

    log.info(f"Splits XGBoost : train={len(X_tr)} | val_ES={len(X_val)} | test={len(X_test)}")

    best_params = None
    if CONFIG.get('ENABLE_HYPERPARAMETER_TUNING', True):
        try:
            import optuna
            log.info("Optuna disponible → lancement du tuning bayésien (50 trials)...")
            best_params = _tune_xgboost_optuna(X_tr, y_tr, X_val, y_val, n_trials=50)
        except ImportError:
            log.warning("Optuna absent → fallback GridSearch partiel")
            try:
                best_params = _tune_xgboost_grid(X_tr, y_tr, X_val, y_val)
            except Exception as e:
                log.warning(f"GridSearch échoué ({e}) → hyperparamètres par défaut améliorés")
        except Exception as e:
            log.warning(f"Optuna échoué ({e}) → hyperparamètres par défaut améliorés")

    if best_params is None:
        best_params = _default_xgb_params()
        log.info("Utilisation des hyperparamètres par défaut améliorés")

    model = xgb.XGBClassifier(
        **best_params,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        early_stopping_rounds=30,
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    log.info(f"XGBoost entraîné — {model.best_iteration+1} arbres — feature importances :")
    imp = sorted(zip(XGB_FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
    for feat, score in imp:
        log.info(f"    {feat:28s}  {score:.4f}")

    return model, X_test, y_test

# ────────────────────Évaluation XGBOOST ────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, label_map):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, classification_report,)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except ValueError as e:
        log.warning(f"AUC non calculable : {e}")
        auc = float('nan')

    inv_map = {v: k for k, v in label_map.items()}
    target_names = [inv_map[i] for i in sorted(inv_map)]

    log.info("BLOC 5B — Evaluation XGBoost (test 20%, seed=42, jamais vu) :")
    log.info(f"    N={len(y_test)}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    if not np.isnan(auc):
        log.info(f"    AUC={auc:.4f}")
    for line in classification_report(y_test, y_pred, target_names=target_names, zero_division=0).splitlines():
        if line.strip():
            log.info(f"      {line}")

    metrics = {
        'accuracy': round(float(acc), 4), 'precision_macro': round(float(prec), 4),
        'recall_macro': round(float(rec), 4), 'f1_macro': round(float(f1), 4),
        'auc_roc_macro': round(float(auc), 4) if not np.isnan(auc) else None,
        'test_split_size': 0.20, 'test_split_seed': 42,
        'n_test_samples': len(y_test), 'evaluated_at': datetime.now().isoformat(),
    }
    with open(CONFIG['METRICS_JSON'], 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info(f"Métriques → {CONFIG['METRICS_JSON']}")
    return metrics

# ==================================== RSF avec Feature Scaling (sans fuite) ========================================
def train_rsf(df):
    """
    Entraîne un Random Survival Forest avec scaling ajusté uniquement sur le train.
    Retourne le modèle, le DataFrame des probabilités NA et le scaler.
    """
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X = df[XGB_FEATURE_COLS].values
        event = df['evenement_na'].astype(bool).values
        time = df['duree_jours'].clip(lower=1).values
        y_rsf = Surv.from_arrays(event=event, time=time)

        # Split pour éviter toute fuite
        X_train, X_eval, y_train, y_eval, idx_train, idx_eval = train_test_split(
            X, y_rsf, np.arange(len(X)), test_size=0.30, random_state=42, stratify=event
        )

        # Scaling : fit uniquement sur X_train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_eval_scaled = scaler.transform(X_eval)

        rsf = RandomSurvivalForest(
            n_estimators=200,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
        )
        rsf.fit(X_train_scaled, y_train)

        c_tr = rsf.score(X_train_scaled, y_train)
        c_ev = rsf.score(X_eval_scaled, y_eval)
        log.info(f"RSF — C-index train={c_tr:.4f} | eval(honnête)={c_ev:.4f}")
        if c_tr - c_ev > 0.10:
            log.warning("RSF : écart C-index > 0.10 → surapprentissage possible")

        # Prédictions sur l'ensemble des données (transform global)
        X_all_scaled = scaler.transform(X)
        surv_funcs = rsf.predict_survival_function(X_all_scaled)
        rows = []
        for fn in surv_funcs:
            t, p = fn.x, fn.y
            p7 = float(1 - np.interp(7, t, p, right=p[-1]))
            p30 = float(1 - np.interp(30, t, p, right=p[-1]))
            rows.append({'prob_na_7d': round(p7, 4), 'prob_na_30d': round(p30, 4)})

        return rsf, pd.DataFrame(rows), scaler

    except ImportError:
        log.warning("scikit-survival absent → RSF fallback statistique activé")
        rows = _rsf_statistical_fallback(df)
        return None, pd.DataFrame(rows), None

def _rsf_statistical_fallback(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    import warnings

    log.info("RSF fallback → LogisticRegression calibrée (Platt scaling)")

    X = df[XGB_FEATURE_COLS].values.astype(float)
    y = df['evenement_na'].astype(int).values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    method = 'isotonic' if len(X) >= 200 else 'sigmoid'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        base_lr = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42,
            class_weight='balanced', solver='lbfgs', multi_class='ovr',
        )
        cal_lr = CalibratedClassifierCV(base_lr, cv=3, method=method)
        cal_lr.fit(X_sc, y)

    proba_na = cal_lr.predict_proba(X_sc)[:, 1]

    rows = []
    for p in proba_na:
        p = float(np.clip(p, 0.01, 0.98))
        rows.append({
            'prob_na_7d': round(p * 0.65, 4),
            'prob_na_30d': round(p, 4),
        })

    log.info(f"RSF fallback — prob_na_30d : mean={np.mean([r['prob_na_30d'] for r in rows]):.3f} | "
             f"std={np.std([r['prob_na_30d'] for r in rows]):.3f}")
    return rows

# ========================== Contact Availability Score ===============================================================
def compute_scores(xgb_model, rsf_preds, X):
    xgb_proba = xgb_model.predict_proba(X)
    rows = []
    thr = SCORE_THRESHOLDS

    for i in range(len(X)):
        p_avail = float(xgb_proba[i][0])
        p_suspect = float(xgb_proba[i][1])
        p_na_xgb = float(xgb_proba[i][2])

        if i < len(rsf_preds):
            p7 = float(rsf_preds.iloc[i]['prob_na_7d'])
            p30 = float(rsf_preds.iloc[i]['prob_na_30d'])
        else:
            p7 = p30 = p_na_xgb

        if p_na_xgb >= 0.60:
            na_risk = p_na_xgb
        elif p_na_xgb >= 0.35:
            na_risk = max(p_na_xgb, 0.50 * p7)
        else:
            na_risk = max(p_na_xgb, 0.35 * p7)

        score = round((1 - na_risk) * 100, 1)

        if score >= thr['available']:
            decision, action = 'Available', 'Inclure dans la campagne'
        elif score >= thr['suspected']:
            decision, action = 'Suspected', 'Surveiller — inclure avec prudence'
        else:
            decision, action = 'NA', 'Exclure de la campagne'

        rows.append({
            'availability_score': score,
            'p_available_%': round(p_avail * 100, 1),
            'p_suspect_%': round(p_suspect * 100, 1),
            'p_na_xgb_%': round(p_na_xgb * 100, 1),
            'p_na_rsf_7d_%': round(p7 * 100, 1),
            'p_na_rsf_30d_%': round(p30 * 100, 1),
            'decision': decision,
            'action': action,
        })
    return pd.DataFrame(rows)

# ========================== Sauvegarde du modèle (joblib + JSON) ===================================================
def save_model(xgb_model, rsf_model, rsf_scaler, label_map, eval_metrics=None):
    # Sauvegarde du modèle XGBoost au format JSON (standard, interopérable)
    xgb_model.save_model(CONFIG['MODEL_XGB'])

    # Sauvegarde des artefacts avec joblib (sécurisé et robuste)
    artifacts = {
        'rsf_model': rsf_model,
        'rsf_scaler': rsf_scaler,
        'label_map': label_map,
        'xgb_feature_cols': XGB_FEATURE_COLS,
        'rsf_extra_cols': RSF_EXTRA_COLS,
        'status_codes': {1: 'LIVRE', 2: 'NON_LIVRE'},
        'dlr_norm_mapping': {1: 1, 2: 2, 4: 2, 8: 2},
        'statuses_cleaned': sorted(STATUSES_TO_CLEAN),
        'fusion_logic': 'asymmetric_xgb_primary_v7',
        'score_thresholds': SCORE_THRESHOLDS,
        'eval_metrics': eval_metrics or {},
        'trained_at': datetime.now().isoformat(),
        'version': '7.1.0',  # version avec corrections
    }
    joblib.dump(artifacts, CONFIG['MODEL_PKL'])
    size = os.path.getsize(CONFIG['MODEL_PKL']) // 1024
    log.info(f"Modèle v7.1.0 → {CONFIG['MODEL_PKL']}  ({size} KB)")

def load_model_assets():
    """Charge les artefacts (hors modèle XGBoost) depuis le fichier joblib."""
    return joblib.load(CONFIG['MODEL_PKL'])

# ========================== PIPELINE PRINCIPAL =====================================================================
def run_training():
    start = datetime.now()
    log.info("===== DEBUT TRAINING v7.1.0 (sans fuite, mémoire optimisée) =====")

    try:
        # Chargement des données (générateur)
        if CONFIG['USE_MONGODB']:
            raw_gen = load_from_mongodb()
        else:
            raw_gen = load_from_json()

        valid_records, _ = filter_and_validate(raw_gen)
        contacts = normalise_statuses(valid_records)

        # Date de référence = dernier événement des données (évite fuite temporelle)
        all_dates = [evt['dt'] for contact in contacts for evt in contact['history']]
        ref_date = max(all_dates) if all_dates else datetime.now()
        log.info(f"Date de référence pour l'entraînement : {ref_date}")

        df_real = build_features(contacts, ref_date=ref_date)
        df_all, label_map = encode_labels(df_real)

        # XGBoost
        xgb_model, X_test, y_test = train_xgboost(df_all)
        eval_metrics = evaluate_model(xgb_model, X_test, y_test, label_map)

        # RSF (avec scaling sans fuite)
        rsf_model, rsf_preds, rsf_scaler = train_rsf(df_real)  # on utilise df_real, pas df_all

        # Scores finaux
        X_real = df_real[XGB_FEATURE_COLS].values
        scores_df = compute_scores(xgb_model, rsf_preds.iloc[:len(df_real)], X_real)

        result_df = pd.concat([
            df_real[['msisdn', 'label']].reset_index(drop=True),
            scores_df.reset_index(drop=True),
        ], axis=1)

        result_df.to_csv(CONFIG['PREDICTIONS_CSV'], index=False)

        dist = result_df['decision'].value_counts().to_dict()
        total = len(result_df)
        log.info(f"Distribution finale ({total} contacts) :")
        for lbl, cnt in dist.items():
            log.info(f"  {lbl:12s} → {cnt:4d}  ({cnt/total*100:.1f}%)")

        save_model(xgb_model, rsf_model, rsf_scaler, label_map, eval_metrics)
        log.info(f"===== TRAINING TERMINE ({(datetime.now()-start).total_seconds():.1f}s) =====")

    except Exception as e:
        log.error(f"ERREUR TRAINING : {e}", exc_info=True)

# =============================== CRON JOB : Inférence quotidienne ==================================================
def run_daily_inference():
    log.info("=== CronJob INFERENCE QUOTIDIENNE ===")
    if not os.path.exists(CONFIG['MODEL_PKL']) or not os.path.exists(CONFIG['MODEL_XGB']):
        log.warning("Modèle introuvable — lancez d'abord le training.")
        return

    start = datetime.now()
    try:
        # Chargement des artefacts
        artifacts = load_model_assets()
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(CONFIG['MODEL_XGB'])
        rsf_model = artifacts['rsf_model']
        rsf_scaler = artifacts['rsf_scaler']
        # label_map, thresholds etc. peuvent être récupérés si besoin

        # Données fraîches
        if CONFIG['USE_MONGODB']:
            raw_gen = load_from_mongodb()
        else:
            raw_gen = load_from_json()

        valid_records, _ = filter_and_validate(raw_gen)
        contacts = normalise_statuses(valid_records)

        # Pour l'inférence, la date de référence est maintenant
        ref_date = datetime.now()
        df_real = build_features(contacts, ref_date=ref_date)

        if df_real.empty:
            log.warning("Aucune donnée pour l'inférence")
            return

        X_real = df_real[XGB_FEATURE_COLS].values

        # RSF : prédictions avec le scaler sauvegardé
        if rsf_model is not None and rsf_scaler is not None:
            X_scaled = rsf_scaler.transform(X_real)
            surv_funcs = rsf_model.predict_survival_function(X_scaled)
            rsf_rows = []
            for fn in surv_funcs:
                t, p = fn.x, fn.y
                p7 = float(1 - np.interp(7, t, p, right=p[-1]))
                p30 = float(1 - np.interp(30, t, p, right=p[-1]))
                rsf_rows.append({'prob_na_7d': round(p7, 4), 'prob_na_30d': round(p30, 4)})
            rsf_df = pd.DataFrame(rsf_rows)
        else:
            # Fallback
            _, rsf_df, _ = train_rsf(df_real)

        scores_df = compute_scores(xgb_model, rsf_df.iloc[:len(df_real)], X_real)

        result_df = pd.concat([
            df_real[['msisdn', 'label']].reset_index(drop=True),
            scores_df.reset_index(drop=True),
        ], axis=1)
        result_df.to_csv(CONFIG['PREDICTIONS_CSV'], index=False)

        dist = result_df['decision'].value_counts().to_dict()
        total = len(result_df)
        log.info(f"Inférence terminée — {total} contacts")
        for lbl, cnt in dist.items():
            log.info(f"  {lbl:12s} -> {cnt:4d}  ({cnt/total*100:.1f}%)")
        log.info(f"Durée : {(datetime.now()-start).total_seconds():.1f}s")

    except Exception as e:
        log.error(f"ERREUR inférence : {e}", exc_info=True)

# ========================= PLANIFICATION NOCTURNE ==================================================================
def _seconds_until(hhmm):
    now = datetime.now()
    h, m = map(int, hhmm.split(':'))
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()

def _scheduler_loop():
    schedule_time = CONFIG['SCHEDULE_TIME']
    retrain_interval = CONFIG.get('RETRAIN_INTERVAL_DAYS', 14)
    last_retrain = None

    while True:
        wait = _seconds_until(schedule_time)
        h, m = divmod(int(wait // 60), 60)
        now = datetime.now()
        need_retrain = last_retrain is None or (now - last_retrain).days >= retrain_interval

        if need_retrain:
            log.info(f"Prochain RE-TRAINING dans {h}h {m}min")
        else:
            log.info(f"Prochain cycle dans {h}h {m}min (inférence — re-training dans ~{retrain_interval-(now-last_retrain).days}j)")

        time.sleep(wait)
        if need_retrain:
            log.info("=== CronJob RE-TRAINING ===")
            run_training()
            last_retrain = datetime.now()
        else:
            run_daily_inference()

def start_scheduler():
    if CONFIG['RUN_NOW']:
        log.info("Exécution immédiate au démarrage...")
        run_training()
    t = Thread(target=_scheduler_loop, daemon=True, name='nightly-training')
    t.start()
    log.info(f"Scheduler actif — training planifié à {CONFIG['SCHEDULE_TIME']}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        log.info("Scheduler arrêté.")

# ===================================== point d'entrée ==============================================================
if __name__ == '__main__':
    start_scheduler()