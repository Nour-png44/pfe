-- Table non liee aux contacts unitaires.
-- Elle trace chaque execution du moteur Data Sanitize
-- sur une campagne, un groupe, un import CSV ou un appel API.

CREATE TABLE datasanitize_batch (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Contexte metier EasyBulk
    batch_code VARCHAR(60) NOT NULL UNIQUE,
    organisation VARCHAR(150) NOT NULL,
    groupe_id BIGINT,
    groupe_name VARCHAR(150),
    campagne_id BIGINT,
    campagne_libelle VARCHAR(180),

    -- Origine de l'execution
    source_type VARCHAR(30) NOT NULL,
    trigger_mode VARCHAR(30) NOT NULL,
    requested_by VARCHAR(150),
    requested_role VARCHAR(30),

    -- Versioning du moteur
    model_version VARCHAR(20),
    rules_version VARCHAR(20),
    pipeline_version VARCHAR(20),

    -- Volume traite
    total_records INT NOT NULL DEFAULT 0,
    total_unique_records INT NOT NULL DEFAULT 0,
    clean_count INT NOT NULL DEFAULT 0,
    review_count INT NOT NULL DEFAULT 0,
    exclude_count INT NOT NULL DEFAULT 0,

    -- Metriques de qualite
    invalid_msisdn_count INT NOT NULL DEFAULT 0,
    invalid_email_count INT NOT NULL DEFAULT 0,
    duplicate_count INT NOT NULL DEFAULT 0,
    missing_required_fields_count INT NOT NULL DEFAULT 0,
    no_history_count INT NOT NULL DEFAULT 0,

    -- Metriques IA agregées
    avg_availability_score DECIMAL(6,2),
    min_availability_score DECIMAL(6,2),
    max_availability_score DECIMAL(6,2),
    suspected_count INT NOT NULL DEFAULT 0,
    na_count INT NOT NULL DEFAULT 0,

    -- Sortie et audit
    output_table_name VARCHAR(120),
    output_report_path VARCHAR(255),
    summary_json TEXT,
    error_message TEXT,

    -- Suivi d'execution
    status VARCHAR(20) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    duration_ms BIGINT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_datasanitize_batch_source_type
        CHECK (
            source_type IN (
                'ADDRESS_BOOK',
                'CAMPAIGN',
                'CSV_IMPORT',
                'API',
                'SCHEDULED_JOB'
            )
        ),

    CONSTRAINT chk_datasanitize_batch_trigger_mode
        CHECK (
            trigger_mode IN (
                'MANUAL',
                'PRE_SEND',
                'POST_IMPORT',
                'API_REQUEST',
                'NIGHTLY',
                'RETRAIN_PIPELINE'
            )
        ),

    CONSTRAINT chk_datasanitize_batch_role
        CHECK (
            requested_role IN (
                'SUPERADMIN',
                'ADMIN',
                'SUPERVISEUR',
                'AGENT',
                'VISIONNEUR'
            ) OR requested_role IS NULL
        ),

    CONSTRAINT chk_datasanitize_batch_status
        CHECK (
            status IN (
                'PENDING',
                'RUNNING',
                'COMPLETED',
                'PARTIAL_SUCCESS',
                'FAILED',
                'CANCELLED'
            )
        )
);

CREATE INDEX idx_datasanitize_batch_org_group
    ON datasanitize_batch (organisation, groupe_id);

CREATE INDEX idx_datasanitize_batch_campaign
    ON datasanitize_batch (campagne_id);

CREATE INDEX idx_datasanitize_batch_status
    ON datasanitize_batch (status, started_at);

CREATE INDEX idx_datasanitize_batch_source
    ON datasanitize_batch (source_type, trigger_mode);
