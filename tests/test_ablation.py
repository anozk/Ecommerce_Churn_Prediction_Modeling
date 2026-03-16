"""
Unit tests for run_ablation()
Run with:  python -m unittest test_ablation -v
"""
import copy
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# ── XGBoost swap note ─────────────────────────────────────────────────────────
# In your project replace the two lines below with:
#   from xgboost import XGBClassifier
#   XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
XGBClassifier = GradientBoostingClassifier  # stand-in for sandbox (no xgboost wheel)

# ── function under test ───────────────────────────────────────────────────────

def run_ablation(pipeline, X_train, y_train, X_val, y_val, features_to_drop):
    X_train_ab = X_train.drop(columns=features_to_drop)
    X_val_ab   = X_val.drop(columns=features_to_drop)
    pipe = copy.deepcopy(pipeline)
    pipe.fit(X_train_ab, y_train)
    preds = pipe.predict(X_val_ab)
    return f1_score(y_val, preds, average="binary", pos_label=1)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    XGBClassifier(n_estimators=50, random_state=42)),
    ])


def _make_data():
    rng = np.random.default_rng(0)
    n   = 120
    X   = pd.DataFrame({
        "age":            rng.normal(35, 10, n),
        "CashbackAmount": rng.normal(50, 15, n),
        "Gender":         rng.choice([0, 1], n),
    })
    y     = pd.Series((X["age"] > 35).astype(int).values)
    split = 80
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


# ── 1. return type and range ──────────────────────────────────────────────────

class TestReturnValue(unittest.TestCase):

    def setUp(self):
        self.pipe = _make_pipeline()
        self.X_tr, self.y_tr, self.X_v, self.y_v = _make_data()

    def test_returns_float(self):
        result = run_ablation(self.pipe, self.X_tr, self.y_tr, self.X_v, self.y_v, [])
        self.assertIsInstance(result, float)

    def test_score_between_0_and_1(self):
        result = run_ablation(self.pipe, self.X_tr, self.y_tr, self.X_v, self.y_v, [])
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_empty_drop_list_is_reproducible(self):
        r1 = run_ablation(self.pipe, self.X_tr, self.y_tr, self.X_v, self.y_v, [])
        r2 = run_ablation(self.pipe, self.X_tr, self.y_tr, self.X_v, self.y_v, [])
        self.assertAlmostEqual(r1, r2, places=10)


# ── 2. feature dropping ───────────────────────────────────────────────────────

class TestFeatureDropping(unittest.TestCase):

    def setUp(self):
        self.pipe = _make_pipeline()
        self.X_tr, self.y_tr, self.X_v, self.y_v = _make_data()

    def test_dropped_column_absent_from_fit(self):
        seen = []
        original_fit = Pipeline.fit

        def recording_fit(self_pipe, X, y=None, **kw):
            seen.extend(X.columns.tolist())
            return original_fit(self_pipe, X, y, **kw)

        with patch.object(Pipeline, "fit", recording_fit):
            run_ablation(self.pipe, self.X_tr, self.y_tr,
                         self.X_v, self.y_v, ["Gender"])

        self.assertNotIn("Gender", seen)
        self.assertIn("age", seen)
        self.assertIn("CashbackAmount", seen)

    def test_single_feature_drop_returns_valid_score(self):
        score = run_ablation(self.pipe, self.X_tr, self.y_tr,
                             self.X_v, self.y_v, ["CashbackAmount"])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_multiple_feature_drop_returns_valid_score(self):
        score = run_ablation(self.pipe, self.X_tr, self.y_tr,
                             self.X_v, self.y_v, ["CashbackAmount", "Gender"])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_drop_nonexistent_column_raises(self):
        with self.assertRaises((KeyError, ValueError)):
            run_ablation(self.pipe, self.X_tr, self.y_tr,
                         self.X_v, self.y_v, ["nonexistent_col"])


# ── 3. deep-copy isolation ────────────────────────────────────────────────────

class TestDeepCopyIsolation(unittest.TestCase):

    def setUp(self):
        self.pipe = _make_pipeline()
        self.X_tr, self.y_tr, self.X_v, self.y_v = _make_data()

    def test_original_pipeline_not_fitted(self):
        run_ablation(self.pipe, self.X_tr, self.y_tr, self.X_v, self.y_v, [])
        with self.assertRaises(NotFittedError):
            check_is_fitted(self.pipe["clf"])

    def test_successive_calls_independent(self):
        score_a = run_ablation(self.pipe, self.X_tr, self.y_tr,
                               self.X_v, self.y_v, ["age"])
        score_b = run_ablation(self.pipe, self.X_tr, self.y_tr,
                               self.X_v, self.y_v, ["age"])
        self.assertAlmostEqual(score_a, score_b, places=10)


# ── 4. F1 metric specifics ────────────────────────────────────────────────────

class TestF1Metric(unittest.TestCase):

    def _mock_run(self, y_true, y_pred):
        X   = pd.DataFrame({"a": range(len(y_true))})
        y   = pd.Series(y_true)
        mock_pipe = MagicMock(spec=Pipeline)
        mock_pipe.predict.return_value = np.array(y_pred)
        with patch("copy.deepcopy", return_value=mock_pipe):
            return run_ablation(mock_pipe, X, y, X, y, [])

    def test_perfect_predictor_f1_one(self):
        self.assertAlmostEqual(self._mock_run([0,0,1,1], [0,0,1,1]), 1.0)

    def test_all_wrong_predictor_f1_zero(self):
        self.assertAlmostEqual(self._mock_run([1,1,1,1], [0,0,0,0]), 0.0)

    def test_targets_class_1_not_class_0(self):
        score = self._mock_run([0,0,0,1,1,1], [0,0,0,0,0,0])
        self.assertAlmostEqual(score, 0.0)


# ── 5. integration: ablation loop ────────────────────────────────────────────

class TestAblationLoop(unittest.TestCase):

    def setUp(self):
        self.pipe = _make_pipeline()
        self.X_tr, self.y_tr, self.X_v, self.y_v = _make_data()

    def test_loop_produces_correct_dataframe_shape(self):
        baseline = run_ablation(self.pipe, self.X_tr, self.y_tr,
                                self.X_v, self.y_v, [])
        results = []
        for feat in self.X_tr.columns:
            f1 = run_ablation(self.pipe, self.X_tr, self.y_tr,
                              self.X_v, self.y_v, [feat])
            results.append({
                "dropped_features":    feat,
                "f1_class1":           f1,
                "delta_from_baseline": f1 - baseline,
            })
        df = pd.DataFrame(results)
        self.assertEqual(set(df.columns),
                         {"dropped_features", "f1_class1", "delta_from_baseline"})
        self.assertEqual(len(df), len(self.X_tr.columns))
        self.assertTrue(df["f1_class1"].between(0.0, 1.0).all())

    def test_ablation_produces_nonzero_deltas(self):
        baseline = run_ablation(self.pipe, self.X_tr, self.y_tr,
                                self.X_v, self.y_v, [])
        deltas = [
            run_ablation(self.pipe, self.X_tr, self.y_tr,
                         self.X_v, self.y_v, [f]) - baseline
            for f in self.X_tr.columns
        ]
        self.assertTrue(any(abs(d) > 1e-9 for d in deltas),
            "All deltas are zero — model may not be learning from features")


if __name__ == "__main__":
    unittest.main(verbosity=2)
