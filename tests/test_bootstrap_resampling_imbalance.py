# test_churn.py
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import auc


# -----------------------------------------------
# 1. TEST: XG_Precision_Recall_MyFunction
# -----------------------------------------------
def test_precision_recall_basic():
    """Test that function returns correct keys"""
    df = pd.DataFrame({
        'True_Label': [1, 0, 1, 0, 1],
        'Predicted_Probability': [0.8, 0.3, 0.9, 0.2, 0.7]
    })
    thresholds = [0.5]
    result = XG_Precision_Recall_MyFunction(df, thresholds)
    assert 0.5 in result
    assert 'Precision' in result[0.5]
    assert 'Recall' in result[0.5]


def test_precision_recall_values():
    """Test that precision and recall are between 0 and 1"""
    df = pd.DataFrame({
        'True_Label': [1, 0, 1, 0, 1],
        'Predicted_Probability': [0.8, 0.3, 0.9, 0.2, 0.7]
    })
    thresholds = [0.3, 0.5, 0.8]
    result = XG_Precision_Recall_MyFunction(df, thresholds)
    for t in thresholds:
        assert 0 <= result[t]['Precision'] <= 1
        assert 0 <= result[t]['Recall'] <= 1


def test_precision_recall_no_positives():
    """Test edge case when threshold is very high (no positives predicted)"""
    df = pd.DataFrame({
        'True_Label': [1, 0, 1],
        'Predicted_Probability': [0.3, 0.2, 0.1]
    })
    result = XG_Precision_Recall_MyFunction(df, [0.99])
    assert result[0.99]['Precision'] == 1.0
    assert result[0.99]['Recall'] == 0.0


# -----------------------------------------------
# 2. TEST: sort_and_calculate_auc
# -----------------------------------------------
def test_auc_between_0_and_1():
    """AUC should always be between 0 and 1"""
    recall = np.array([0.1, 0.4, 0.7, 0.9])
    precision = np.array([0.9, 0.7, 0.5, 0.3])
    result = sort_and_calculate_auc(recall, precision)
    assert 0 <= result <= 1


def test_auc_unsorted_input():
    """Function should handle unsorted recall values"""
    recall = np.array([0.9, 0.1, 0.7, 0.4])
    precision = np.array([0.3, 0.9, 0.5, 0.7])
    result = sort_and_calculate_auc(recall, precision)
    assert 0 <= result <= 1


# -----------------------------------------------
# 3. TEST: Thresholds
# -----------------------------------------------
def test_combined_thresholds_sorted():
    """Thresholds should be sorted after combining"""
    assert XG_combined_thresholds == sorted(XG_combined_thresholds)


def test_thresholds_range():
    """All thresholds should be between 0 and 1"""
    assert all(0 <= t <= 1 for t in XG_combined_thresholds)


# -----------------------------------------------
# 4. TEST: Bootstrap Results
# -----------------------------------------------
def test_bootstrap_rounds_count():
    """Should have exactly 1000 bootstrap rounds"""
    assert len(true_labels_pred_list) == 1000


def test_bootstrap_dataframe_columns():
    """Each bootstrap df should have required columns"""
    for df in true_labels_pred_list[:5]:  # check first 5
        assert 'True_Label' in df.columns
        assert 'Predicted_Probability' in df.columns


def test_confidence_interval_order():
    """Lower CI should always be <= Median <= Upper CI"""
    assert all(lower_p <= median_p)
    assert all(median_p <= upper_p)
