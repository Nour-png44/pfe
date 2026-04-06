import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import predict_api


class PredictApiTests(unittest.TestCase):
    def test_build_history_from_progress_uses_final_sms_statuses(self):
        history = predict_api.build_history_from_progress(
            "21600000000",
            [
                {
                    "msisdn": "21600000000",
                    "status": 4,
                    "status_last_updated_at": "2026-03-03 10:00:00",
                },
                {
                    "msisdn": "21600000000",
                    "status": 1,
                    "status_last_updated_at": "2026-03-02 10:00:00",
                },
                {
                    "msisdn": "21600000000",
                    "status": 34,
                    "status_last_updated_at": "2026-03-05 10:00:00",
                },
                {
                    "msisdn": "21600000000",
                    "status": 1,
                    "status_last_updated_at": "not-a-date",
                },
                {
                    "msisdn": "21600000000",
                    "status": 4,
                    "status_last_updated_at": "2026-03-03 10:00:00",
                },
                {
                    "msisdn": "999",
                    "status": 1,
                    "status_last_updated_at": "2026-03-01 10:00:00",
                },
            ],
        )

        self.assertEqual(
            history,
            [
                {"status": 1, "dt": datetime(2026, 3, 2, 10, 0, 0)},
                {"status": 4, "dt": datetime(2026, 3, 3, 10, 0, 0)},
                {"status": 34, "dt": datetime(2026, 3, 5, 10, 0, 0)},
            ],
        )

    def test_build_features_from_history_normalises_transit_statuses(self):
        ref_date = datetime(2026, 4, 1, 12, 0, 0)
        features = predict_api.build_features_from_history(
            "21600000000",
            [
                {"status": 1, "dt": ref_date - timedelta(days=40)},
                {"status": 1, "dt": ref_date - timedelta(days=20)},
                {"status": 4, "dt": ref_date - timedelta(days=5)},
            ],
            ref_date=ref_date,
        )

        self.assertEqual(features["msisdn"], "21600000000")
        self.assertEqual(features["total_envois"], 3)
        self.assertEqual(features["taux_livraison"], 0.6667)
        self.assertEqual(features["echecs_consecutifs_max"], 1)
        self.assertEqual(features["jours_depuis_succes"], 20)
        self.assertEqual(features["taux_livraison_recent"], 0.5)
        self.assertEqual(features["freq_inter_envoi_jours"], 17.5)

    def test_predict_route_filters_results_and_uses_default_for_missing_history(self):
        def fake_build_history(msisdn, _all_progress):
            if msisdn == "21600000002":
                return []
            return [{"status": 1, "dt": datetime(2026, 3, 1, 10, 0, 0)}]

        def fake_build_features(msisdn, history, ref_date=None):
            return {"msisdn": msisdn, "total_envois": len(history)}

        def fake_predict_contacts(rows, _pkg):
            decisions = {
                "21600000001": "Suspected",
                "21600000003": "NA",
            }
            actions = {
                "21600000001": "Surveiller",
                "21600000003": "Exclure",
            }
            return [
                {
                    "msisdn": row["msisdn"],
                    "decision": decisions[row["msisdn"]],
                    "action": actions[row["msisdn"]],
                    "availability_score": 42.0,
                    "p_available_%": 20.0,
                    "p_suspect_%": 30.0,
                    "p_na_xgb_%": 50.0,
                    "p_na_rsf_7d_%": 25.0,
                    "p_na_rsf_30d_%": 50.0,
                }
                for row in rows
            ]

        with patch.object(predict_api, "load_model", return_value={"version": "test-model"}), \
             patch.object(predict_api, "_all_progress", return_value=[{"msisdn": "ignored"}]), \
             patch.object(predict_api, "build_history_from_progress", side_effect=fake_build_history), \
             patch.object(predict_api, "build_features_from_history", side_effect=fake_build_features), \
             patch.object(predict_api, "predict_contacts", side_effect=fake_predict_contacts):
            client = predict_api.app.test_client()
            response = client.post(
                "/api/predict",
                json={
                    "msisdns": ["21600000001", "21600000002", "21600000003"],
                    "include_suspect": False,
                    "include_na": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()

        self.assertEqual(
            payload["summary"],
            {
                "total": 3,
                "available": 1,
                "suspected": 1,
                "na": 1,
            },
        )
        self.assertEqual(payload["filtered_count"], 2)
        self.assertEqual(
            [contact["msisdn"] for contact in payload["contacts"]],
            ["21600000003", "21600000002"],
        )
        self.assertEqual(payload["contacts"][1]["decision"], "Available")
        self.assertEqual(payload["contacts"][1]["action"], "Nouveau contact — aucun historique")
        self.assertEqual(payload["model_version"], "test-model")

    def test_predict_route_rejects_missing_msisdns(self):
        client = predict_api.app.test_client()

        response = client.post("/api/predict", json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "msisdns manquants"})


if __name__ == "__main__":
    unittest.main()
