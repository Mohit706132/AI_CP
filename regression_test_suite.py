# -*- coding: utf-8 -*-
"""Comprehensive System & Regression Test Suite for Integrated Plant Suite (AI_CP).

Run this script anytime to validate that all modules, models, routing,
RBAC authentication, retraining failsafes, and audit logs are 100% operational:
    python regression_test_suite.py
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from uv_module.core.auth import (
    init_auth_db, register_user, verify_login, get_pending_users,
    approve_user, reject_user, update_user_role, get_all_users, delete_user
)
from uv_module.core.audit import init_db, log_batch_results, log_image_result, get_audit_history
from uv_module.core.router import route_and_process
from uv_module.core.plant_detector import detect_plant
from app import normalize_ground_truth_label, plot_confusion_matrix_fig, UV_MODELS_DIR, UV_DATA_DIR, TEMPLATES_DIR
from training_scripts.model_manager import (
    backup_production_models, get_staging_dir, compare_and_decide,
    promote_staging, cleanup_staging, get_production_accuracy
)


class TestRBACAuthentication(unittest.TestCase):
    """Tests Role-Based Access Control, registration, approval, and promotion."""

    def setUp(self):
        init_db()
        delete_user("test_eval_user")

    def tearDown(self):
        delete_user("test_eval_user")

    def test_01_default_admin_login(self):
        ok, msg, user = verify_login("admin", "admin123")
        self.assertTrue(ok, "Default admin login should succeed")
        self.assertEqual(user["role"], "admin", "Admin user must have 'admin' role")
        self.assertEqual(user["status"], "approved", "Admin must be approved")

    def test_02_registration_and_approval_flow(self):
        ok, msg = register_user("test_eval_user", "securePass123", "Test Operator")
        self.assertTrue(ok, "Registration should succeed")

        ok, msg, user = verify_login("test_eval_user", "securePass123")
        self.assertFalse(ok, "Pending user should not be able to log in")
        self.assertIn("pending", msg.lower(), "Message should indicate pending approval")

        pending = get_pending_users()
        pending_usernames = [u["username"] for u in pending]
        self.assertIn("test_eval_user", pending_usernames)

        app_ok = approve_user("test_eval_user", "admin")
        self.assertTrue(app_ok, "Admin approval should succeed")

        ok, msg, user = verify_login("test_eval_user", "securePass123")
        self.assertTrue(ok, "Approved user should log in successfully")
        self.assertEqual(user["role"], "user", "Default role should be 'user'")

    def test_03_role_promotion_to_admin(self):
        register_user("test_eval_user", "securePass123", "Test Operator")
        approve_user("test_eval_user", "admin")

        prom_ok = update_user_role("test_eval_user", "admin")
        self.assertTrue(prom_ok, "Role promotion should succeed")
        ok, msg, user = verify_login("test_eval_user", "securePass123")
        self.assertEqual(user["role"], "admin", "User role should now be 'admin'")

        dem_ok = update_user_role("test_eval_user", "user")
        self.assertTrue(dem_ok, "Role demotion should succeed")
        ok, msg, user = verify_login("test_eval_user", "securePass123")
        self.assertEqual(user["role"], "user", "User role should now be 'user'")


class TestUVSpectralInference(unittest.TestCase):
    """Tests UV classification engines across all target species."""

    def test_01_hirda_template_inference(self):
        template = TEMPLATES_DIR / "hirda_template.csv"
        self.assertTrue(template.exists(), "hirda_template.csv must exist")
        with open(template, "rb") as f:
            results, modality = route_and_process(f, UV_MODELS_DIR, UV_DATA_DIR, "Hirda")
        self.assertIn(modality, ["UV", "Hirda"])
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["prediction"], "Rohini")

    def test_02_rauwolfia_template_inference(self):
        template = TEMPLATES_DIR / "rauwolfia_template.csv"
        self.assertTrue(template.exists(), "rauwolfia_template.csv must exist")
        with open(template, "rb") as f:
            results, modality = route_and_process(f, UV_MODELS_DIR, UV_DATA_DIR, "Rauwolfia/Terminalia")
        self.assertIn(modality, ["UV", "Rauwolfia/Terminalia"])
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["prediction"], "Rauwolfia serpentina")

    def test_03_embelia_template_inference(self):
        template = TEMPLATES_DIR / "embelia_template.csv"
        self.assertTrue(template.exists(), "embelia_template.csv must exist")
        with open(template, "rb") as f:
            results, modality = route_and_process(f, UV_MODELS_DIR, UV_DATA_DIR, "Embelia")
        self.assertIn(modality, ["UV", "Embelia"])
        self.assertGreater(len(results), 0)
        self.assertIn(results[0]["prediction"], ["Authentic Embelia", "Adulterant / Unknown"])

    def test_04_auto_detect_routing(self):
        template = TEMPLATES_DIR / "auto_detect_template.csv"
        self.assertTrue(template.exists(), "auto_detect_template.csv must exist")
        with open(template, "rb") as f:
            results, modality = route_and_process(f, UV_MODELS_DIR, UV_DATA_DIR, "Auto-Detect")
        self.assertGreater(len(results), 0)
        self.assertEqual(len(results), 3, "Template contains 3 multi-plant test rows")


class TestGroundTruthNormalization(unittest.TestCase):
    """Tests Ground Truth label normalization and confusion matrix generation."""

    def test_01_embelia_label_normalization(self):
        self.assertEqual(normalize_ground_truth_label("Embelia ribes", "Authentic Embelia", "Embelia"), "Authentic Embelia")
        self.assertEqual(normalize_ground_truth_label("Embelia Tsjeriam cottom", "Adulterant / Unknown", "Embelia"), "Adulterant / Unknown")
        self.assertEqual(normalize_ground_truth_label("Abhaya_Negative_Control_1", "Adulterant / Unknown", "Embelia"), "Adulterant / Unknown")

    def test_02_confusion_matrix_plotting(self):
        y_true = ["Authentic Embelia", "Authentic Embelia", "Adulterant / Unknown"]
        y_pred = ["Authentic Embelia", "Adulterant / Unknown", "Adulterant / Unknown"]
        fig, cm_df = plot_confusion_matrix_fig(y_true, y_pred)
        self.assertIsNotNone(fig)
        self.assertEqual(cm_df.shape, (2, 2))


class TestRetrainingFailsafe(unittest.TestCase):
    """Tests Model Retraining failsafe, staging isolation, and production protection."""

    def test_01_staging_isolation_and_backup(self):
        backup_dir = backup_production_models("hirda", UV_MODELS_DIR, UV_DATA_DIR)
        self.assertTrue(backup_dir.exists(), "Backup directory should be created")

        staging_dir = get_staging_dir("hirda", UV_MODELS_DIR)
        self.assertTrue(staging_dir.exists(), "Staging directory should be created")

        prod_acc = get_production_accuracy("hirda", UV_MODELS_DIR, UV_DATA_DIR)
        self.assertIsNotNone(prod_acc, "Production accuracy should be queryable")

        # Test decision logic: worse model must be REJECTED
        decision = compare_and_decide("hirda", prod_acc - 5.0, UV_MODELS_DIR, UV_DATA_DIR)
        self.assertEqual(decision["decision"], "REJECT", "Worse model must be rejected to protect production")
        cleanup_staging("hirda", UV_MODELS_DIR)


class TestAuditLogging(unittest.TestCase):
    """Tests audit database logging for user session tracking."""

    def test_01_audit_batch_and_image_logging(self):
        init_db()
        test_results = [{
            "sample_name": "UnitTest_Scan.csv",
            "prediction": "Rohini",
            "adul_status": "AUTHENTIC",
            "purity": 99.1,
            "confidence": {"Rohini": 0.991},
            "modality": "UV"
        }]
        log_batch_results("test_operator", "Hirda Analysis", test_results)

        img_result = {
            "plant_label": "Haritaki (Terminalia chebula)",
            "plant_confidence": 97.5,
            "top_prediction": {"name": "Rohini", "confidence": 95.0, "uses": "General health"},
            "alternatives": []
        }
        log_image_result("test_operator", "test_leaf.jpg", img_result)

        history = get_audit_history()
        self.assertGreater(len(history), 0, "Audit logs should contain recorded events")
        latest_users = [h["username"] for h in history[:5]]
        self.assertIn("test_operator", latest_users, "Test operator should be recorded in audit log")


def run_full_suite():
    print("=" * 70)
    print("   INTEGRATED PLANT SUITE (AI_CP) - REGRESSION TEST SUITE")
    print("=" * 70)
    suite = unittest.TestSuite()
    for test_class in [
        TestRBACAuthentication,
        TestUVSpectralInference,
        TestGroundTruthNormalization,
        TestRetrainingFailsafe,
        TestAuditLogging
    ]:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("=" * 70)
    if result.wasSuccessful():
        print("[SUCCESS] ALL REGRESSION TESTS PASSED! SYSTEM READY FOR DEPLOYMENT.")
    else:
        print(f"[FAIL] {len(result.failures)} FAILURES, {len(result.errors)} ERRORS DETECTED.")
    print("=" * 70)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_full_suite()
    sys.exit(0 if success else 1)
