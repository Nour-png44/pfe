-- Table recommandee pour le moteur "Data Sanitize Engine"
-- Concue a partir :
-- 1. du guide EasyBulk 2026 (carnets d'adresses, campagnes, tags, groupes)
-- 2. des donnees reelles du projet (contacts.json, user_contact.json, campagne.json)
-- 3. des sorties IA existantes (predict_api.py, train_model.py)

CREATE TABLE contact_datasanitize (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Contexte metier EasyBulk
    organisation VARCHAR(150) NOT NULL,
    groupe_id BIGINT,
    groupe_name VARCHAR(150),
    campagne_id BIGINT,
    contact_id BIGINT,
    user_contact_id BIGINT,
    source_type VARCHAR(30) NOT NULL DEFAULT 'ADDRESS_BOOK',

    -- Donnees source
    first_name VARCHAR(120),
    last_name VARCHAR(120),
    email_raw VARCHAR(255),
    email_normalized VARCHAR(255),
    msisdn_raw VARCHAR(32),
    msisdn_normalized VARCHAR(32) NOT NULL,
    country_code VARCHAR(8),
    message_encoding VARCHAR(20),

    -- Tag metier avant/apres sanitation
    tag_before VARCHAR(50),
    tag_before_id INT,
    tag_after VARCHAR(50),
    tag_after_id INT,

    -- Controles de qualite
    is_msisdn_valid BOOLEAN NOT NULL DEFAULT FALSE,
    is_email_valid BOOLEAN NOT NULL DEFAULT FALSE,
    is_duplicate BOOLEAN NOT NULL DEFAULT FALSE,
    duplicate_key VARCHAR(255),
    has_delivery_history BOOLEAN NOT NULL DEFAULT FALSE,
    history_events_count INT NOT NULL DEFAULT 0,
    last_delivery_status INT,
    last_delivery_at TIMESTAMP,

    -- Features IA deja utilisees par le projet
    total_envois INT NOT NULL DEFAULT 0,
    taux_livraison DECIMAL(6,4),
    echecs_consecutifs_max INT,
    jours_depuis_succes INT,
    score_recence DECIMAL(10,4),
    taux_livraison_recent DECIMAL(6,4),
    freq_inter_envoi_jours DECIMAL(10,2),

    -- Resultat IA
    availability_score DECIMAL(6,2),
    decision VARCHAR(20),
    action_recommandee VARCHAR(120),
    p_available_pct DECIMAL(6,2),
    p_suspect_pct DECIMAL(6,2),
    p_na_xgb_pct DECIMAL(6,2),
    p_na_rsf_7d_pct DECIMAL(6,2),
    p_na_rsf_30d_pct DECIMAL(6,2),

    -- Resultat metier Data Sanitize
    sanitize_status VARCHAR(20) NOT NULL,
    sanitize_reason TEXT,
    anomaly_flags_json TEXT,
    model_version VARCHAR(20),

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    sanitized_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_contact_datasanitize_decision
        CHECK (decision IN ('Available', 'Suspected', 'NA') OR decision IS NULL),

    CONSTRAINT chk_contact_datasanitize_status
        CHECK (sanitize_status IN ('CLEAN', 'REVIEW', 'EXCLUDE'))
);

CREATE INDEX idx_contact_datasanitize_msisdn
    ON contact_datasanitize (msisdn_normalized);

CREATE INDEX idx_contact_datasanitize_org_group
    ON contact_datasanitize (organisation, groupe_id);

CREATE INDEX idx_contact_datasanitize_campaign
    ON contact_datasanitize (campagne_id);

CREATE INDEX idx_contact_datasanitize_status
    ON contact_datasanitize (sanitize_status, decision);

CREATE INDEX idx_contact_datasanitize_tag_after
    ON contact_datasanitize (tag_after_id, tag_after);
