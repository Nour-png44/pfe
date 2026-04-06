-- Deuxieme table recommandee pour le moteur "Data Sanitize Engine"
-- Usage : stocker une ligne par anomalie detectee sur un contact.
-- Cette table est complementaire a une table principale de contacts nettoyes.

CREATE TABLE contact_datasanitize_issue (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Contexte EasyBulk
    organisation VARCHAR(150) NOT NULL,
    groupe_id BIGINT,
    groupe_name VARCHAR(150),
    campagne_id BIGINT,
    contact_id BIGINT,
    user_contact_id BIGINT,

    -- Identifiant fonctionnel du contact
    msisdn_normalized VARCHAR(32) NOT NULL,
    email_normalized VARCHAR(255),

    -- Details de l'anomalie
    issue_type VARCHAR(50) NOT NULL,
    issue_category VARCHAR(40) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    issue_source VARCHAR(20) NOT NULL,
    issue_score DECIMAL(6,2),

    -- Valeur analysee et correction proposee
    field_name VARCHAR(50) NOT NULL,
    raw_value TEXT,
    normalized_value TEXT,
    proposed_value TEXT,

    -- Decision operationnelle
    auto_fixable BOOLEAN NOT NULL DEFAULT FALSE,
    fix_status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    resolution_action VARCHAR(120),
    resolution_note TEXT,

    -- Trace IA / regles
    rule_code VARCHAR(50),
    model_version VARCHAR(20),
    explanation TEXT,

    -- Workflow de revue
    reviewed_by VARCHAR(150),
    reviewed_at TIMESTAMP,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_contact_datasanitize_issue_type
        CHECK (
            issue_type IN (
                'INVALID_MSISDN',
                'INVALID_EMAIL',
                'DUPLICATE_MSISDN',
                'DUPLICATE_EMAIL',
                'MISSING_NAME',
                'MISSING_GROUP',
                'NO_HISTORY',
                'LOW_DELIVERY_RATE',
                'HIGH_FAILURE_RATE',
                'BLACKLISTED_PATTERN',
                'INVALID_COUNTRY_CODE',
                'INCONSISTENT_TAG'
            )
        ),

    CONSTRAINT chk_contact_datasanitize_issue_category
        CHECK (
            issue_category IN (
                'FORMAT',
                'DUPLICATE',
                'COMPLETENESS',
                'DELIVERABILITY',
                'BUSINESS_RULE',
                'AI_DECISION'
            )
        ),

    CONSTRAINT chk_contact_datasanitize_issue_severity
        CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),

    CONSTRAINT chk_contact_datasanitize_issue_source
        CHECK (issue_source IN ('RULE', 'ML', 'USER', 'IMPORT')),

    CONSTRAINT chk_contact_datasanitize_issue_fix_status
        CHECK (fix_status IN ('PENDING', 'AUTO_FIXED', 'REVIEWED', 'REJECTED'))
);

CREATE INDEX idx_contact_datasanitize_issue_msisdn
    ON contact_datasanitize_issue (msisdn_normalized);

CREATE INDEX idx_contact_datasanitize_issue_campaign
    ON contact_datasanitize_issue (campagne_id);

CREATE INDEX idx_contact_datasanitize_issue_type
    ON contact_datasanitize_issue (issue_type, severity);

CREATE INDEX idx_contact_datasanitize_issue_status
    ON contact_datasanitize_issue (fix_status, reviewed_at);
