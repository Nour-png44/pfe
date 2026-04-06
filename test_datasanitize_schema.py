import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def read_sql(filename):
    return (BASE_DIR / filename).read_text(encoding="utf-8")


class ContactDatasanitizeSchemaTests(unittest.TestCase):
    def test_contact_table_keeps_core_constraints_and_indexes(self):
        sql = read_sql("datasanitize_table.sql")

        self.assertIn("CREATE TABLE contact_datasanitize", sql)
        self.assertIn("msisdn_normalized VARCHAR(32) NOT NULL", sql)
        self.assertIn(
            "CHECK (decision IN ('Available', 'Suspected', 'NA') OR decision IS NULL)",
            sql,
        )
        self.assertIn(
            "CHECK (sanitize_status IN ('CLEAN', 'REVIEW', 'EXCLUDE'))",
            sql,
        )
        self.assertIn(
            "CREATE INDEX idx_contact_datasanitize_status",
            sql,
        )
        self.assertIn(
            "ON contact_datasanitize (sanitize_status, decision)",
            sql,
        )

    def test_issue_table_keeps_expected_issue_enums_and_review_status_index(self):
        sql = read_sql("datasanitize_issue_table.sql")

        self.assertIn("CREATE TABLE contact_datasanitize_issue", sql)
        self.assertIn("issue_type IN (", sql)
        self.assertIn("'INVALID_MSISDN'", sql)
        self.assertIn("'LOW_DELIVERY_RATE'", sql)
        self.assertIn("'AI_DECISION'", sql)
        self.assertIn(
            "CHECK (fix_status IN ('PENDING', 'AUTO_FIXED', 'REVIEWED', 'REJECTED'))",
            sql,
        )
        self.assertIn(
            "CREATE INDEX idx_contact_datasanitize_issue_status",
            sql,
        )
        self.assertIn(
            "ON contact_datasanitize_issue (fix_status, reviewed_at)",
            sql,
        )

    def test_batch_table_keeps_unique_batch_code_and_execution_status_values(self):
        sql = read_sql("datasanitize_batch_table.sql")

        self.assertIn("CREATE TABLE datasanitize_batch", sql)
        self.assertIn("batch_code VARCHAR(60) NOT NULL UNIQUE", sql)
        self.assertIn("source_type IN (", sql)
        self.assertIn("'CSV_IMPORT'", sql)
        self.assertIn("'RETRAIN_PIPELINE'", sql)
        self.assertIn("'PARTIAL_SUCCESS'", sql)
        self.assertIn("'CANCELLED'", sql)
        self.assertIn(
            "CREATE INDEX idx_datasanitize_batch_source",
            sql,
        )
        self.assertIn(
            "ON datasanitize_batch (source_type, trigger_mode)",
            sql,
        )


if __name__ == "__main__":
    unittest.main()
