import unittest
from datetime import datetime
from unittest.mock import patch

import train_model


class TrainModelTests(unittest.TestCase):
    def test_load_records_falls_back_to_json_when_mongodb_is_unavailable(self):
        with patch.dict(train_model.CONFIG, {"USE_MONGODB": True, "JSON_PATH": "fallback.json"}, clear=False), \
             patch.object(train_model, "load_from_mongodb", side_effect=RuntimeError("mongo down")), \
             patch.object(train_model.Path, "exists", return_value=True), \
             patch.object(train_model, "load_from_json", return_value=[{"msisdn": "21600000000"}]) as load_json:
            records = train_model.load_records()

        self.assertEqual(records, [{"msisdn": "21600000000"}])
        load_json.assert_called_once_with()

    def test_filter_and_validate_uses_final_sms_status_without_history(self):
        valid, stats = train_model.filter_and_validate(
            [
                {
                    "msisdn": "21600000000",
                    "status": 1,
                    "status_last_updated_at": "2026-03-01 10:00:00",
                }
            ]
        )

        self.assertEqual(stats["valid"], 1)
        self.assertEqual(
            valid,
            [
                {
                    "msisdn": "21600000000",
                    "status": 1,
                    "dt": datetime(2026, 3, 1, 10, 0, 0),
                }
            ],
        )

    def test_extract_and_clean_groups_contacts_by_final_status_history(self):
        contacts = train_model.extract_and_clean(
            [
                {
                    "msisdn": "21600000000",
                    "status": 1,
                    "status_last_updated_at": "2026-03-01 10:00:00",
                    "status_history": [
                        {"status": 8, "status_last_updated_at": "2026-03-01 09:59:00"},
                        {"status": 1, "status_last_updated_at": "2026-03-01 10:00:00"},
                    ],
                },
                {
                    "msisdn": "21600000000",
                    "status": 8,
                    "status_last_updated_at": "2026-03-02 10:00:00",
                },
                {
                    "msisdn": "21600000000",
                    "status": 2,
                    "status_last_updated_at": "2026-03-03 10:00:00",
                },
                {
                    "msisdn": "21600000000",
                    "status": 2,
                    "status_last_updated_at": "2026-03-03 10:00:00",
                },
            ]
        )

        self.assertEqual(len(contacts), 1)
        self.assertEqual(contacts[0]["msisdn"], "21600000000")
        self.assertEqual(contacts[0]["n_sends"], 3)
        self.assertEqual(contacts[0]["final_status"], 2)
        self.assertEqual(
            contacts[0]["history"],
            [
                {"status": 1, "dt": datetime(2026, 3, 1, 10, 0, 0)},
                {"status": 2, "dt": datetime(2026, 3, 2, 10, 0, 0)},
                {"status": 2, "dt": datetime(2026, 3, 3, 10, 0, 0)},
            ],
        )

    def test_build_features_applies_recent_failure_penalty_and_success_momentum(self):
        ref_date = datetime(2026, 4, 1, 12, 0, 0)
        features = train_model.build_features(
            [
                {
                    "msisdn": "21600000000",
                    "history": [
                        {"status": 1, "dt": datetime(2026, 3, 12, 12, 0, 0)},
                        {"status": 1, "dt": datetime(2026, 3, 22, 12, 0, 0)},
                        {"status": 1, "dt": datetime(2026, 3, 30, 12, 0, 0)},
                    ],
                    "final_status": 1,
                    "n_sends": 3,
                },
                {
                    "msisdn": "21600000001",
                    "history": [
                        {"status": 1, "dt": datetime(2026, 3, 12, 12, 0, 0)},
                        {"status": 2, "dt": datetime(2026, 3, 22, 12, 0, 0)},
                        {"status": 2, "dt": datetime(2026, 3, 30, 12, 0, 0)},
                    ],
                    "final_status": 2,
                    "n_sends": 3,
                },
            ],
            ref_date=ref_date,
        )

        available = features.loc[features["msisdn"] == "21600000000"].iloc[0]
        suspected = features.loc[features["msisdn"] == "21600000001"].iloc[0]

        self.assertEqual(available["label"], "Available")
        self.assertEqual(suspected["label"], "Suspected")
        self.assertGreater(available["score_recence"], suspected["score_recence"])
        self.assertEqual(available["taux_livraison_recent"], 1.0)
        self.assertEqual(suspected["taux_livraison_recent"], 0.3333)
        self.assertEqual(available["jours_depuis_succes"], 2)
        self.assertEqual(suspected["jours_depuis_succes"], 20)


if __name__ == "__main__":
    unittest.main()
